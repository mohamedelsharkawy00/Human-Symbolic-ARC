#pragma once

#include "environment.hpp"
#include "global_app_state.hpp"
#include "cuda_dsl.cuh"   // GPU_KERNEL_MAX_OBJECTS (was pulled in via gui.hpp on Windows)
#ifdef _MSC_VER
#include "gui.hpp"        // full GUI/Vulkan rendering
#else
#include "gui_stub.hpp"   // headless: GPU-buffer helpers only, no rendering
#endif
#include "logger.hpp"
#ifdef _MSC_VER
#include "os_utilities.hpp"  // Win32 window helpers; GUI-only
#endif
#include "puzzle_loader.hpp"
#include "sound_manager.hpp"

#include <algorithm>
#include <cstdlib>   // std::getenv (ARC_CHALLENGES_JSON)
#include <cstring>
#include <filesystem>
// Always use fmtlib so both builds format identically. Call sites qualify
// fmt::format explicitly (std::format leaks in via <format> + `using namespace std`).
#include "fmt/format.h"
#include <fstream>
#include <map>
#include <mutex>
// placement new (reset_search reinstantiates search_state in place)
#include <new>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _MSC_VER
#include <windows.h>
#endif

namespace arc {
using namespace std;

// visited: per-slot/level dedup set (flat_hash_set, ~16 B/entry incl. load
// factor). Cleared when full and at each level boundary. With
// MAX_CANDIDATES_PER_SLOT = 200k a single slot cannot insert anywhere near
// 1M distinct hashes before merge, so 1M is already above the reachable
// per-slot max; the old 10M was pure dead ceiling (~160 MB) — 1M ≈ 16 MB.
static constexpr size_t MAX_VISITED_NODES = 1'000'000;
// global_cache: persists for the whole puzzle (unordered_map<u64,
// cache_entry>, cache_entry = 16 B padded, ~40 B/entry incl. node + bucket
// overhead). Value is in caching the set of scored programs that actually
// recur within a solve; that working set is far below 1M. 10M ≈ 400 MB of
// mostly-cold entries; 1M ≈ 40 MB still covers the realistic recurrence set.
static constexpr size_t MAX_CACHE_ENTRIES = 1'000'000;

// Fast open-addressing hash set for uint64_t (cache-friendly visited set).
// Uses power-of-2 table with linear probing. ~2x memory of std::unordered_set
// but dramatically faster lookups due to no pointer chasing.
struct flat_hash_set_u64 {
    static constexpr uint64_t EMPTY = 0;
    std::vector<uint64_t> slots;
    size_t mask = 0;
    size_t count = 0;

    // 4M slots × 8B = 32MB. Measured faster than 2M (16MB) on this workload
    // despite spilling L3 — the larger table keeps load factor low enough that
    // probe chains stay short, and the hot probe set is localized in practice.
    flat_hash_set_u64() { resize_table(1 << 22); }

    void resize_table(size_t new_cap) {
        std::vector<uint64_t> old = std::move(slots);
        slots.assign(new_cap, EMPTY);
        mask = new_cap - 1;
        count = 0;
        for (uint64_t v : old)
            if (v != EMPTY)
                insert(v);
    }

    // Boost-style hash combining leaves weak low bits, which clustered hard
    // under linear probing. splitmix64 finalizer mixes them in ~2 cycles.
    static uint64_t mix(uint64_t x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31; return x;
    }
    size_t probe(uint64_t h) const { return mix(h) & mask; }

    bool contains(uint64_t v) const {
        if (v == EMPTY) v = 2;
        size_t i = probe(v);
        while (true) {
            uint64_t s = slots[i];
            if (s == v) return true;
            if (s == EMPTY) return false;
            i = (i + 1) & mask;
        }
    }

    void insert(uint64_t v) {
        if (v == EMPTY) v = 2;
        if (count * 2 > mask) resize_table((mask + 1) * 2); // 50% load
        size_t i = probe(v);
        while (true) {
            uint64_t s = slots[i];
            if (s == v) return;
            if (s == EMPTY) {
                slots[i] = v;
                count++;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    // One-walk variant: returns true if the value was newly inserted, false
    // if already present. Replaces a contains+insert pair (which did two
    // probe walks) with a single walk.
    bool insert_if_new(uint64_t v) {
        if (v == EMPTY) v = 2;
        if (count * 2 > mask) resize_table((mask + 1) * 2);
        size_t i = probe(v);
        while (true) {
            uint64_t s = slots[i];
            if (s == v) return false;
            if (s == EMPTY) {
                slots[i] = v;
                count++;
                return true;
            }
            i = (i + 1) & mask;
        }
    }

    size_t size() const { return count; }

    void clear() {
        std::fill(slots.begin(), slots.end(), EMPTY);
        count = 0;
    }

    // Shrink back to initial capacity so the next puzzle starts with a
    // cache-friendly 32MB table instead of an oversized one from the previous run.
    void reset() {
        slots.assign(1 << 22, EMPTY);
        mask = (1 << 22) - 1;
        count = 0;
    }

};
static constexpr size_t MAX_HISTORY_POOL = 1000;
// Per-slot ceiling. current_slot_candidates holds every candidate of a slot
// live simultaneously (each is also borrowed by pending_evaluation) until
// merge_slot_candidates clears it — so this constant directly sets the
// dominant live-tree memory term. 1M trees/slot was the practical OOM driver;
// the slot is distilled into the <=1024 beam anyway, so a far lower ceiling
// costs almost nothing in search quality. Enforced via the existing back-
// pressure check in generate_branch (total_candidates_this_slot >= this).
static constexpr int MAX_CANDIDATES_PER_SLOT = 200000;

// Everything one puzzle's search owns, in one type. Reset is reinstantiation
// (destroy + reconstruct), so the member initializers below are the single
// list of opening values — a field added here is reset correctly by
// construction, with nothing to keep in sync.
struct search_state {
  struct champion {
    node *n;
    float score;
    uint64_t grid_hash = 0;
  };
  // A refinement slot on the unary chain is just how many steps down from the
  // root the replacement happens.
  struct slot_t {
    int depth = 0;
  };
  struct cache_entry { float score; uint64_t grid_hash; };

  // Total order on candidates: score first, then tree hash. Arrival order
  // carries no information, so no tie may be broken by it — untied ranking is
  // what makes a fresh run and a switched run the same search.
  // Proven 2026-07-14 on 007bbfb7: score-only ties let container iteration
  // order pick the beam, and fresh (50s, 11M attempts) vs switched (10s,
  // 2.8M) diverged repeatably from one 8-candidate ripple in level 2.
  static bool ranks_higher(const champion &a, const champion &b) {
    if (a.score != b.score) return a.score > b.score;
    uint64_t ha = a.n ? a.n->hash : 0, hb = b.n ? b.n->hash : 0;
    return ha > hb;
  }

  vector<node *> history_pool;
  // history_pool sorted ascending by max_depth, rebuilt when the pool changes,
  // so deep entries can be excluded by a prefix bound rather than an inner-loop check.
  vector<node *> history_by_depth;
  // For each allowed max_depth d in [0..MAX_SEARCH_DEPTH], bucket_cutoff[d] is
  // the number of entries in history_by_depth with max_depth <= d. The generator
  // reads this once per primitive to bound its iteration; depth-rejected entries
  // never enter the loop.
  int bucket_cutoff[MAX_SEARCH_DEPTH + 1] = {0};

  int m_gen_phase = 0;
  int m_gen_primitive_idx = 0;
  int m_gen_outer_idx = 0;
  int m_gen_inner_idx = 0;
  int m_dry_yields = 0;       // consecutive yields with zero new candidates
  int m_last_cand_at_yield = 0; // total_candidates_this_slot at last yield
  int m_consec_dry_beam_entries = 0; // consecutive fully-dry beam entries across generate_branch calls

  vector<champion> beam;
  vector<slot_t> slots_to_refine;
  int current_slot_idx = 0;
  int beam_idx = 0;

  bool initialized = false;

  flat_hash_set_u64 visited;
  flat_hash_set_u64 visited_grids;
  queue<node *> pending_evaluation;

  unordered_map<uint64_t, cache_entry> global_cache;
  float all_time_best_score = SCORE_MIN;

  int active_example_count = 1;

  vector<champion> current_slot_candidates;
  vector<champion> level_candidates;

  bool is_generating_layer = false;

  // Layer locking (toggle: app_state.layer_locking). The locked prefix is a
  // committed layer chain every candidate must build on. It is created ONLY by
  // a verified commit — the candidate painted zero wrong cells on every active
  // pair — never by filtering the running beam. Candidates lacking it are
  // rejected at admission; slots never point inside it.
  node *locked_prefix = nullptr;
  uint64_t locked_hash = 0;
  // cells the locked prefix already explains (its correct paint, all pairs)
  int locked_correct_cells = 0;
  // colors the locked prefix already explains COMPLETELY (every target cell of
  // that color correct, all pairs). "Finishing a color" is a commit trigger of
  // its own — layered solving is one color at a time — and no later lock may
  // un-finish one.
  bool locked_color_done[10] = {};
  // Focus color — the human way: work on ONE unfinished color at a time.
  // Picked at each lock commit: the unfinished color with the largest residual
  // (big areas are the substrate — walls/backgrounds first). Scoring gets a
  // strong bonus on the focus color's target cells. If the focus makes no
  // progress for FOCUS_ROTATE_LEVELS level boundaries, rotate to the next
  // unfinished color (a stalled color often depends on one not yet painted —
  // stalling discovers the right order). A full circle of rotations with no
  // progress anywhere drops the focus (-1 = free search, current behavior).
  int focus_color = -1;
  int focus_best_correct = 0;        // best correct count seen for the focus color
  bool focus_progress_this_level = false;
  int focus_levels_no_progress = 0;  // rotation counter
  bool focus_tried[10] = {};         // colors already focused since the last progress
  int locked_color_residual[10] = {}; // target cells of color v the lock leaves unexplained
  int locked_color_total[10] = {};    // target cells of color v, all pairs
  // set on commit; the beam restarts from the lock at the next level boundary
  bool lock_commit_pending = false;
  // Ladder exit — one strike: a correct lock can still be the wrong
  // decomposition (009d5c81: "paint every shape 7" froze at 0.4s, before the
  // real shape->color layer could exist in the beam). Two full focus circles
  // without a single lock commit = both the focused and free strategies
  // failed on this prefix — drop the ladder and turn locking OFF for the
  // rest of the puzzle. Layering gets one honest attempt; when it proves
  // confused, flat search (which solves depth-8 puzzles on its own) takes
  // over for good. Retry machinery (opener bans, failure memory, re-arming)
  // was tried 2026-07-11 and never once paid off — every rebuilt ladder was
  // another cheap trap.
  int focus_circles_no_lock = 0;     // completed focus circles since last commit
  bool locking_suspended = false;    // set by drop_ladder, permanent per puzzle
  // The human order: grasp the whole first, decompose when stuck. Flat search
  // runs alone from t=0; locking arms at the first level-stagnation event.
  // This deletes the opener race — by the time locks may commit, the beam is
  // deep enough that substantive layers exist to out-cover cheap ones.
  bool locking_armed = false;
  // Probe verdicts depend only on (tree hash, lock state): remember the
  // failures and skip them until the lock state changes (commit or drop) —
  // beam survivors otherwise re-pay the full-train eval at every boundary.
  set<uint64_t> probe_failed;
  // the beam member being refined (phase_1 wraps it directly in locked mode)
  node *m_current_base = nullptr;

  int total_candidates_this_slot = 0;

  // Sentence sweep: the sentences this level may use, and how far through them
  // the generator has got. sweep_beam_idx walks the beam, sweep_idx the sweep.
  vector<sentence> sweep;
  int sweep_beam_idx = 0;
  int sweep_idx = 0;
  bool sweep_ready = false;

  // Where a worker iteration's time actually goes, reported once per level.
  // Counters, not log lines — logging the hot path is what distorted the
  // first measurement.
  // ns_wait is not here: it lives in solver_state so the render thread can
  // read it, and this class reads it back for the line below.
  long long ns_generate = 0, ns_launch = 0, ns_harvest = 0, ns_idle = 0;
  long long ns_wait_at_last_level = 0;
  // Harvest, split three ways; harvest minus these is the unaccounted rest.
  long long ns_tally = 0, ns_winner = 0, ns_free = 0;
  // The winner block, split the same way.
  long long ns_fetch = 0, ns_analyze = 0, ns_best = 0;

  int stagnant_levels = 0;
  float prev_beam_top = SCORE_MIN;

  bool m_reducing_for_lock = false; // re-entry guard for the lean-lock reduction
  bool m_focus_initialized = false; // pre-lock focus picked from raw target counts

  // A promotion that fills the train set IS the solve, but marking the task
  // solved can't happen at that moment (it can happen mid-update_best_solution,
  // whose own solved/pause writes would collide). The verified tree waits here;
  // the worker loop marks the task solved at the top of its next pass.
  node *solve_found_by_promotion = nullptr;
  int solve_found_node_count = 0;

  static void release_nodes(vector<node*>& v) {
    for (auto* n : v) if (n) environment::remove(n);
    v.clear();
  }

  static void release_candidates(vector<champion>& v) {
    for (auto& c : v) if (c.n) environment::remove(c.n);
    v.clear();
  }

  search_state() = default;
  search_state(const search_state&) = delete;
  search_state& operator=(const search_state&) = delete;

  // Owned trees go back to the pool. pending_evaluation only borrows from
  // current_slot_candidates, so it is drained before those are released.
  ~search_state() {
    release_nodes(history_pool);
    release_candidates(beam);
    release_candidates(level_candidates);
    while (!pending_evaluation.empty())
      pending_evaluation.pop();
    release_candidates(current_slot_candidates);
    if (locked_prefix) environment::remove(locked_prefix);
    if (solve_found_by_promotion) environment::remove(solve_found_by_promotion);
  }
};

// Adds the enclosing scope's wall time to a counter. Two clock reads, no
// formatting and no file write, so it can sit on the hot path.
template <typename counter>
struct scope_ns {
  counter &total;
  chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
  ~scope_ns() {
    total += chrono::duration_cast<chrono::nanoseconds>(
                 chrono::steady_clock::now() - t0).count();
  }
};

// A GPU wait, published as it happens so a reader mid-wait sees it.
struct gpu_wait_scope {
  solver_state &state;
  chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
  explicit gpu_wait_scope(solver_state &s) : state(s) {
    state.gpu_wait_since.store(chrono::duration_cast<chrono::nanoseconds>(
        t0.time_since_epoch()).count());
  }
  ~gpu_wait_scope() {
    state.gpu_wait_ns += chrono::duration_cast<chrono::nanoseconds>(
        chrono::steady_clock::now() - t0).count();
    state.gpu_wait_count++;
    state.gpu_wait_since.store(0);
  }
};

class beam_solver : public environment, private search_state {
private:
  enum class phase_result {
    yield,
    next_phase,
    candidate_ready,
    retry,
    break_loop
  };

public:
  static constexpr int FOCUS_ROTATE_LEVELS = 8;
  void drop_ladder();

  // Is the locked chain on n's arg spine? (unary DSL: one spine, hash = identity)
  bool contains_locked(const node *n) const {
    for (const node *cur = n; cur; cur = cur->arg)
      if (cur->hash == locked_hash) return true;
    return false;
  }

  void log_debug(const string &msg) { m_log.debug(msg); }
  void log_timing(const string &msg) { m_log.timing(msg); }
  void clear_logs() { m_log.clear(); }
  void start_benchmark_report(const string &path) { m_log.start_benchmark_report(path); }
  void log_benchmark_result(const string &result) { m_log.benchmark_result(result); }
  void finish_benchmark_report() { m_log.finish_benchmark_report(); }
  void benchmark_begin() { m_log.benchmark_begin(); }
  void benchmark_record(const puzzle &p, long long attempts, float elapsed) { m_log.benchmark_record(p, attempts, elapsed); }
  void log_score_formula(const node *tree, const vector<score_breakdown> &breakdowns);

private:
  atomic<long long> *m_global_attempts = nullptr;

public:
  // ~search_state releases every tree the search still owns.
  ~beam_solver() {
    stop_thread();
  }

  // The one reset, shared by every transition (puzzle switch, dataset change,
  // R): reinstantiate the whole per-puzzle state, same values as opening.
  void reset_search() {
    search_state &s = *this;
    s.~search_state();
    new (&s) search_state();
  }


  void report_score(const node *n, float score, uint64_t g_hash);
  bool check_candidate_compatibility(node_type type, node* arg);

  // True if a color-literal op of `type` with literal `value` is allowed for the
  // current puzzle. When pinning is on, a literal is pinned to the side it
  // speaks about — read literals to the colors some train INPUT has, written
  // ones to the colors some train OUTPUT has — and BidiColorMap variants to the
  // swap pairs learned from the train diffs, collapsing the 10x/45x fan-outs.
  // Anything else (non-color ops, pinning off, no learned set, InteriorColor's
  // -1 "auto") is always allowed.
  inline bool color_literal_allowed(node_type type, int16_t value) const {
    if (!m_app_state.pin_color_literals) return true;
    if (type == node_type::BidiColorMap) {
      if (m_puzzle.color_swap_variants_count == 0) return true;  // not learned -> full fan-out
      if (value < 0 || value >= 45) return true;   // out-of-range guard
      return m_puzzle.color_swap_variants[value];
    }
    const bool reads_color = type == node_type::SelectObjectsContainingColor ||
                             type == node_type::SelectColor;
    const bool writes_color = type == node_type::AddColor ||
                              type == node_type::Recolor;
    if (!reads_color && !writes_color) return true;  // InteriorColor (-1 auto) left alone
    if (value < 0 || value >= 10) return true;   // out-of-range guard
    // NOTE: hard bg pruning was tried here (2026-07-20) and REJECTED —
    // completeness beats speed in ARC. The background steers by RANKING only
    // (ranked_variant_value); fan-out reduction beyond pinning comes later
    // via candidate-side checks (see Background Per-Side Split note).
    // A color a chain asks about has to be one the input can answer with, and a
    // color it lays down has to be one an output actually shows.
    const bool *side = reads_color ? m_puzzle.colors_in_any_input
                                   : m_puzzle.colors_in_any_output;
    const int side_count = reads_color ? m_puzzle.colors_in_any_input_count
                                       : m_puzzle.colors_in_any_output_count;
    if (side_count == 0) return true;            // no set -> don't pin
    return side[value];
  }
  // RANKING (Background Per-Side Split note, level 1): iterate color-literal
  // fans through a permutation that tries the elected canvas FIRST — reading
  // ops (RemoveColor/Contains, "strip/test the canvas") lead with the INPUT
  // canvas, writing ops (AddColor/Recolor, "paint/turn into") with the OUTPUT
  // canvas. Ordering only, never exclusion: a wrong or absent election costs
  // nothing (everything is still enumerated, just later) — the tolerance
  // contract. Implemented as a swap of index 0 with the canvas's index.
  inline int16_t ranked_variant_value(node_type type, int idx) const {
    int16_t v = get_variant_value(type, idx);
    const bool read_op  = type == node_type::SelectObjectsContainingColor;
    const bool write_op = type == node_type::AddColor  || type == node_type::Recolor;
    if (!read_op && !write_op) return v;
    int8_t bg = read_op ? m_puzzle.bg.input_elected : m_puzzle.bg.output_elected;
    if (write_op && bg < 0) bg = m_puzzle.bg.input_elected;
    if (bg < 0) return v;
    int variants = get_variant_count(type);
    int bg_idx = -1;
    for (int i = 0; i < variants; ++i)
      if (get_variant_value(type, i) == bg) { bg_idx = i; break; }
    if (bg_idx < 0) return v;
    if (idx == 0) return bg;
    if (idx == bg_idx) return get_variant_value(type, 0);
    return v;
  }

  phase_result handle_retry(int& batch_count);
  bool check_yield(int &batch_count);

private:
  node *get_node_at_path(node *root, int depth);
  node *copy_and_replace(const node *root, int depth, node *new_subtree);
  static uint64_t predict_replaced_hash(const node *root, int depth, uint64_t new_subtree_hash);
  node *create_unary_node(node_type type, node *arg0);

  void seed_starter_nodes();
  // Every (op, literal) this puzzle allows in one beat of the four-beat grammar.
  void beat_choices(op_role want, vector<flat_program_op> &out) const;
  // Every sentence the grammar allows here. The beat order satisfies legal_next
  // by construction — a GET starts, a SELECT marks, an ACT spends the mark, a
  // COMBINE lands it — so only the puzzle's own gates take anything out. One set,
  // not two: a sentence names its own source wherever it sits.
  void enumerate_sentences(vector<sentence> &out) const;
  void update_history_from_beam();
  void collect_subtrees(node *n, unordered_set<uint64_t> &existing, int &added);
  void rebuild_skeleton_beam();
  void prepare_next_level_slots();
  void find_slots_recursive(node *n, int depth, vector<slot_t> &out, set<int> &visited);
  void reset_d2_generator();

  static constexpr int MAX_STAGNANT_LEVELS = 5;
  static constexpr int MAX_TILE_OPS = 2;

  static int count_tile_ops(const node *n) {
    int c = 0;
    for (; n; n = n->arg)
      if (n->type == node_type::TileX || n->type == node_type::TileY) ++c;
    return c;
  }

  phase_result generate_phase_prune(const slot_t &slot, node* target, int &batch_count, node *&candidate_sub);
  phase_result generate_phase_0(const slot_t &slot, int &batch_count, node *&candidate_sub, bool &created_new);
  phase_result generate_phase_1(const slot_t &slot, int &batch_count, node *&candidate_sub, bool &created_new);
  phase_result execute_current_phase(const slot_t &slot, node* target, int &batch_count, node *&candidate_sub, bool &created_new);

  bool generate_branch();
  // One batch of whole sentences. True when the level has no more to give.
  bool generate_sentence_batch();
  void commit_sentence_level();
  void commit_and_advance_level();
  void advance_beam_idx();
  bool process_current_phase(const slot_t &slot, node* base, node* target, int& batch_count);
  bool add_candidate_if_new(node* base, int depth, node* candidate_sub, bool created_new);
  void reset_generator_counters();
  void merge_slot_candidates();
  void commit_level_champion();

private:
  // The task set, sliced in-memory from a challenges JSON (taken as-is) and
  // iterated by puzzle_idx. Replaces the old per-task file list.
  vector<challenge_task> m_puzzle_tasks;
  // Ground-truth test outputs, keyed by task id, loaded from the matching
  // solutions JSON (empty when none exists, e.g. the competition test set).
  // Attached to each task's test examples at load so the test row can be scored.
  map<string, vector<grid>> m_solutions;
  size_t m_base_gpu_buffer_size = 0;
  atomic<int> m_queued_puzzle_delta{0};
  // Which dataset(s) are loaded into m_puzzle_tasks, as a bitmask of dataset
  // flags (see below). Multiple sets can be checked at once; their tasks are
  // concatenated into one list (e.g. TRAIN|EVAL = 1000+120 = 1120 puzzles).
  atomic<int> m_dataset_flags{0};
  // A pending dataset-flag change requested from the GUI; -1 means none. The
  // worker picks it up at a safe point and rebuilds m_puzzle_tasks (like reset).
  atomic<int> m_queued_dataset_flags{-1};

public:
  // Dataset selection bitmask. Each set is one challenges/solutions JSON pair.
  enum dataset_flag { DS_TRAIN = 1, DS_EVAL = 2, DS_TEST = 4 };
  inline int get_dataset_flags() const { return m_dataset_flags.load(); }
  // Queue a new dataset selection (bitmask). Applied by the worker; ignored if
  // empty so the list is never blanked out from under the solver.
  inline void set_datasets(int flags) {
    if (flags == 0) return;
    m_queued_dataset_flags.store(flags);
  }

  // Build m_puzzle_tasks + m_solutions from every checked dataset, concatenated.
  // The three sets share the same {"id":{"train":[...],"test":[...]}} shape, so
  // we just slice each selected challenges JSON in-memory (taken as-is) and load
  // its sibling _solutions file for ground truth. Each set:
  //   - training:   1000 tasks (has answer keys) — dev / mining.
  //   - evaluation:  120 tasks (unseen, has answer keys) — honest local score.
  //   - test:        240 tasks (the competition set, no solutions) — submission.
  // Duplicate ids across sets are dropped (first checked set wins) so puzzle_idx
  // stays a clean 1:1 index. Records the applied selection in m_dataset_flags.
  inline void load_dataset(int flags) {
    const string kaggle_dir =
        "kaggle/input/competitions/arc-prize-2026-arc-agi-2/";
    struct set_info { int flag; const char *stem; };
    static const set_info sets[] = {
        {DS_TRAIN, "arc-agi_training_challenges.json"},
        {DS_EVAL,  "arc-agi_evaluation_challenges.json"},
        {DS_TEST,  "arc-agi_test_challenges.json"},
    };

    vector<challenge_task> tasks;
    map<string, vector<grid>> solutions;
    set<string> seen;
    for (const set_info &s : sets) {
      if (!(flags & s.flag)) continue;
      string challenges_json = kaggle_dir + s.stem;
      auto loaded = load_challenges_json(challenges_json);
      for (auto &t : loaded)
        if (seen.insert(t.id).second)
          tasks.push_back(std::move(t));

      // Ground-truth answers: sibling file with "_challenges" -> "_solutions".
      // Missing (e.g. the competition test set) just yields no answers.
      string solutions_json = challenges_json;
      if (size_t at = solutions_json.rfind("_challenges"); at != string::npos)
        solutions_json.replace(at, sizeof("_challenges") - 1, "_solutions");
      auto sol = load_solutions_json(solutions_json);
      for (auto &kv : sol)
        solutions.emplace(kv.first, std::move(kv.second));
    }

    m_puzzle_tasks = std::move(tasks);
    m_solutions = std::move(solutions);
    m_dataset_flags.store(flags);
    m_app_state.dataset_flags.store(flags);
    std::fprintf(stderr, "dataset flags %d: %zu tasks, %zu solutions\n",
                 flags, m_puzzle_tasks.size(), m_solutions.size());
  }

  struct candidate_result {
    grid state;
    float score = 0.0f;
    bool valid = false;
  };

  struct candidate_batch {
    vector<candidate_result> train;
    float total_train_score = 0;
    bool all_train_solved = true;
  };

  struct flat_candidate_chunk {
    vector<node *> trees;
    unique_ptr<flat_program_batch> programs = make_unique<flat_program_batch>();
    vector<flat_program> flat_programs;
    vector<int> node_counts;

    flat_candidate_chunk() = default;
    flat_candidate_chunk(flat_candidate_chunk&& other) noexcept
        : trees(std::move(other.trees)),
          programs(std::move(other.programs)),
          flat_programs(std::move(other.flat_programs)),
          node_counts(std::move(other.node_counts)) {
      if (!programs)
        programs = make_unique<flat_program_batch>();
      other.trees.clear();
      other.flat_programs.clear();
      other.node_counts.clear();
      if (!other.programs)
        other.programs = make_unique<flat_program_batch>();
    }
    flat_candidate_chunk& operator=(flat_candidate_chunk&& other) noexcept {
      if (this == &other)
        return *this;
      clear();
      trees = std::move(other.trees);
      programs = std::move(other.programs);
      flat_programs = std::move(other.flat_programs);
      node_counts = std::move(other.node_counts);
      if (!programs)
        programs = make_unique<flat_program_batch>();
      other.trees.clear();
      other.flat_programs.clear();
      other.node_counts.clear();
      if (!other.programs)
        other.programs = make_unique<flat_program_batch>();
      return *this;
    }
    flat_candidate_chunk(const flat_candidate_chunk&) = delete;
    flat_candidate_chunk& operator=(const flat_candidate_chunk&) = delete;

    void clear() {
      for (node *tree : trees)
        if (tree)
          environment::remove(tree);
      trees.clear();
      flat_programs.clear();
      node_counts.clear();
      if (!programs)
        programs = make_unique<flat_program_batch>();
      programs->clear();
    }

    void reserve(int capacity) {
      if (capacity <= 0)
        return;
      trees.reserve(capacity);
      flat_programs.reserve(capacity);
      node_counts.reserve(capacity);
    }

    int size() const { return (int)trees.size(); }

    ~flat_candidate_chunk() { clear(); }
  };

  inline beam_solver(global_app_state &app_state)
      : m_app_state(app_state), m_log(app_state, m_solver_state) {}

  inline void init(vector<int> &gpu_buffer, map<uint32_t, grid> &chars,
                   grid &bg_grid) {
    m_gpu_buffer = &gpu_buffer;

    init_gui_buffers(gpu_buffer, chars, bg_grid);
    // Pool of preview slots, one per possible example row. Measured against
    // the ARC training+evaluation sets: max train=10, max test=4, max total
    // per puzzle=12. 32 leaves generous headroom for unseen puzzle sets.
    m_solver_state.preview_grids.clear();
    m_solver_state.preview_grids.reserve(32);
    for (int i = 0; i < 32; ++i)
      m_solver_state.preview_grids.push_back(allocate_empty_grid(gpu_buffer));

    // Object-list preview slots: [example][object]. Per-example object cap
    // matches the kernel's GPU_KERNEL_MAX_OBJECTS (the largest list the GPU
    // can produce in a single op).
    m_solver_state.preview_object_grids.clear();
    m_solver_state.preview_object_grids.resize(32);
    for (int i = 0; i < 32; ++i) {
      m_solver_state.preview_object_grids[i].reserve(GPU_KERNEL_MAX_OBJECTS);
      for (int j = 0; j < GPU_KERNEL_MAX_OBJECTS; ++j)
        m_solver_state.preview_object_grids[i].push_back(allocate_empty_grid(gpu_buffer));
    }
    m_base_gpu_buffer_size = gpu_buffer.size();

    // Pick the initial dataset(s). ARC_CHALLENGES_JSON still wins (Kaggle points
    // it at the test/eval file it selected; headless has no GUI to toggle). With
    // no override the GUI default is TRAIN (dev/mining); the user can check more
    // sets at runtime and they get concatenated into one list.
    int initial = DS_TRAIN;
    if (const char *env = std::getenv("ARC_CHALLENGES_JSON")) {
      string e = env;
      if (e.find("test") != string::npos) initial = DS_TEST;
      else if (e.find("eval") != string::npos) initial = DS_EVAL;
      else initial = DS_TRAIN;
    }
#ifdef _MSC_VER
    // GUI resume: restore the saved dataset selection before the saved puzzle
    // index is read below, so the index lands in the same list it came from.
    else {
      int saved = settings_int(read_settings_file(), "datasets", 0) & (DS_TRAIN | DS_EVAL | DS_TEST);
      if (saved > 0) initial = saved;
    }
#endif
    load_dataset(initial);

    if (!m_puzzle_tasks.empty()) {
      int saved_idx = settings_int(read_settings_file(), "puzzle", -1);
      if (saved_idx >= 0 && saved_idx < (int)m_puzzle_tasks.size())
        m_solver_state.puzzle_idx.store(saved_idx);
      m_solver_state.puzzle_idx = max(0, m_solver_state.puzzle_idx.load());
      m_solver_state.puzzle_idx = min((int)m_puzzle_tasks.size() - 1, m_solver_state.puzzle_idx.load());

      const challenge_task &ct = m_puzzle_tasks[m_solver_state.puzzle_idx];
      m_puzzle_loaded = load_single_puzzle(ct, m_puzzle, gpu_buffer, m_base_gpu_buffer_size, solutions_for(ct.id),
                                           m_solver_state.puzzle_idx, (int)m_puzzle_tasks.size(), m_dataset_flags.load());
      if (m_puzzle_loaded) {
        m_log.set_puzzle_id(m_puzzle.id);
#ifdef _MSC_VER
        // Per-puzzle log file: dev/GUI telemetry. Skipped headless -- on Kaggle
        // it writes 240+ files into /kaggle/working and exhausts the disk quota.
        string path = "logs/" + m_puzzle.id + ".txt";
        filesystem::create_directory("logs");
        ofstream(path, ios::trunc);
#endif
        clear_logs();
      }
    }

    m_app_state.needs_upload = true;
    if (m_puzzle_loaded) {
      puzzle &p = m_puzzle;

      if (p.best_tree) {
        m_solver_state.current_tree = environment::copy(p.best_tree);
        if (p.is_solved) {
          active_example_count = (int)p.train.size();
        }
        for (auto &ex : p.train) {
          grid dummy;
          environment::score_tree_output(
              p.best_tree, ex, dummy, p, p.best_node_count);
          update_display_buffer(ex.current, dummy, *m_gpu_buffer);
        }
      }
    }
  }

  inline void start_thread() {
    log_debug("[*] Starting Beam Solver thread...");
    m_thread = jthread(&beam_solver::worker, this);
  }

  inline void stop_thread() {
    m_app_state.stop_threads = true;
    if (m_thread.joinable()) {
      m_thread.request_stop();
      m_thread.join();
    }
  }

  inline int current_puzzle_idx() const { return m_solver_state.puzzle_idx.load(); }

  // ID of the currently loaded puzzle (for copy-to-clipboard). Falls back to
  // the filename stem if no puzzle is loaded yet.
  inline string current_puzzle_id() const {
    scoped_lock lock(m_solver_state.mtx);
    if (m_puzzle_loaded)
      return m_puzzle.id;
    int i = m_solver_state.puzzle_idx.load();
    if (i >= 0 && i < (int)m_puzzle_tasks.size())
      return m_puzzle_tasks[i].id;
    return string();
  }

  inline void change_puzzle(int delta) {
    if (m_puzzle_tasks.empty())
      return;
    m_queued_puzzle_delta += delta;
  }

  // Jump to a puzzle by ID (the filename stem). Tries an exact match first,
  // then falls back to the first puzzle whose ID starts with `id`. Matching is
  // case-insensitive. Returns true and queues the jump if a match was found.
  inline bool jump_to_puzzle_id(const string &id) {
    if (m_puzzle_tasks.empty() || id.empty())
      return false;
    auto lower = [](string s) {
      for (char &c : s) c = (char)tolower((unsigned char)c);
      return s;
    };
    string needle = lower(id);
    int exact = -1, prefix = -1;
    for (int i = 0; i < (int)m_puzzle_tasks.size(); ++i) {
      string stem = lower(m_puzzle_tasks[i].id);
      if (stem == needle) { exact = i; break; }
      if (prefix < 0 && stem.compare(0, needle.size(), needle) == 0)
        prefix = i;
    }
    int target = (exact >= 0) ? exact : prefix;
    if (target < 0)
      return false;
    m_queued_puzzle_delta += target - m_solver_state.puzzle_idx.load();
    return true;
  }


  inline void reset_solver() {
    m_reset_solver_flag = true;
  }

  void reload_current_puzzle();

  inline void import_solution() {
    if (!m_puzzle_loaded) return;
    puzzle &p = m_puzzle;
    string sol_path = "solutions/" + p.id + ".txt";
    if (filesystem::exists(sol_path)) {
      ifstream sf(sol_path);
      string sol_str, line;
      while (getline(sf, line)) {
        // Strip CR (Windows line endings).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Trim leading/trailing whitespace.
        size_t a = line.find_first_not_of(" \t");
        if (a == string::npos) { if (!sol_str.empty()) break; else continue; }
        size_t b = line.find_last_not_of(" \t");
        line = line.substr(a, b - a + 1);
        // First line of the metadata block ("Puzzle:", "Train:", etc.) ends the
        // program. Metadata lines are "Key: value" pairs and carry a ':';
        // op tokens never do.
        if (line.find(':') != string::npos) break;
        if (!sol_str.empty()) sol_str += ' ';
        sol_str += line;
      }
      // Drop a dangling trailing separator from line continuations.
      while (!sol_str.empty() && sol_str.back() == ' ')
        sol_str.pop_back();
      if (!sol_str.empty()) {
        lock_guard<mutex> lock(m_solver_state.mtx);
        // Importing wants the facts, so it asks for them. Ops that learn a
        // table (ShapeColorMap and kin) have nothing to say about a puzzle
        // nobody has read, and would paint an empty grid for every example.
        // A reload with the inference pane shut skips inference, and while
        // paused nothing ever fills it in -- so ask here, never assume.
        if (!p.inferred)
          infer_into_puzzle(p, m_solver_state.puzzle_idx,
                            (int)m_puzzle_tasks.size(), m_dataset_flags.load());
        apply_solution(p, m_solver_state, *m_gpu_buffer, sol_str, m_log);
        if (p.is_solved) {
          active_example_count = (int)p.train.size();
        }
        m_app_state.needs_upload = true;
        log_debug("Manual solution imported.");
      }
    }
  }

  // Unified result formatter — used by both manual Export and benchmark logging.
  inline string format_solution_result(const puzzle &p,
                                       const solver_state &state,
                                       const global_app_state &app) {
    float elapsed = get_elapsed(app, state);
    string time_str = format_elapsed(elapsed);

    string train_status = p.is_solved ? "Solved" : "Timeout";
    string test_status = p.is_test_solved ? "Solved"
                       : (p.is_solved ? "Failed" : "N/A");

    long long attempts = state.attempts.load();
    int node_count = p.best_node_count;
    int max_depth = p.best_tree ? (int)p.best_tree->max_depth : 0;

    float best_score = 0.0f;
    for (const auto &ex : p.train)
      best_score += ex.best_score;

    string sol_str = p.best_tree ? environment::stringify_tree_via_flat(p.best_tree) : "";

    // Solution on line 1 (raw, no prefix) for import compatibility.
    string out;
    out += sol_str + "\n";
    out += "Puzzle:       " + p.id + "\n";
    out += "Train:        " + train_status + "\n";
    out += "Test:         " + test_status + "\n";
    out += "Elapsed:      " + time_str + "\n";
    out += fmt::format("Attempts:     {}\n", attempts);
    out += fmt::format("Nodes:        {}\n", node_count);
    out += fmt::format("Best Score:   {:.4f}\n", best_score);
    out += fmt::format("Max Depth:    {}\n", max_depth);
    return out;
  }

  inline void export_solution() {
    if (!m_puzzle_loaded) return;
    puzzle &p = m_puzzle;
    if (!p.best_tree) return;

    string result = format_solution_result(p, m_solver_state, m_app_state);

#ifdef _MSC_VER
    // Per-puzzle solution dump: dev/GUI feature. Skipped headless (disk quota).
    filesystem::create_directories("solutions");
    string sol_path = "solutions/" + p.id + ".txt";
    ofstream sf(sol_path);
    if (sf.is_open()) {
      sf << result;
      log_debug("Manual solution exported to " + sol_path);
    }
#endif
  }

  inline void toggle_pause() {
    bool new_paused = !m_app_state.is_paused;
    if (new_paused) {
      m_app_state.pause_start_time = chrono::steady_clock::now();
      m_app_state.is_paused = true;
    } else {
      auto now = chrono::steady_clock::now();
      auto dt = now - m_app_state.pause_start_time;
      scoped_lock lock(m_solver_state.mtx);
      m_solver_state.start_time += dt;
      m_app_state.is_paused = false;
    }
  }

  inline solver_state &get_solver_state() { return m_solver_state; }
  inline int get_puzzle_count() const { return (int)m_puzzle_tasks.size(); }
  // Ground-truth test outputs for a task id, or nullptr when none are loaded
  // (no solutions file, or this id isn't in it). Passed to load_single_puzzle.
  inline const vector<grid> *solutions_for(const string &id) const {
    auto it = m_solutions.find(id);
    return it != m_solutions.end() ? &it->second : nullptr;
  }
  inline const puzzle &get_puzzle(int idx) const {
    static puzzle dummy;
    return m_puzzle_loaded ? m_puzzle : dummy;
  }

private:
  atomic<bool> m_reset_solver_flag{false};

  inline bool should_pause_or_wait(int &target_idx) {
    target_idx = m_solver_state.puzzle_idx;
    if (target_idx < 0 || target_idx >= (int)m_puzzle_tasks.size() || !m_puzzle_loaded) {
      this_thread::sleep_for(chrono::milliseconds(10));
      return true;
    }

    if (m_app_state.is_paused) {
      this_thread::sleep_for(chrono::milliseconds(10));
      return true;
    }
    return false;
  }

  bool pump_generate(const puzzle &p, std::atomic<long long> *global_attempts);
  void log_tree_record(node *tree, float score, long long attempts);
  void update_best_solution(int target_idx, puzzle &p, node *tree, int node_count, candidate_batch &res);
  // Layer-lock commit check: CPU re-eval of a root-Union tree against the
  // FULL train set (a lock is permanent, the active set has no authority to
  // certify permanence); commits it as the locked prefix if it qualifies. Called from
  // update_best_solution AND on beam survivors at each level boundary — a
  // color-finishing layer must not need to beat the global best score to lock.
  // Commits ONLY when the tree finishes a new color — each layer must complete
  // before the next begins. ("+N total cells" locking was tried 2026-07-03:
  // greedy micro-locks one op deeper each burned the depth budget dry.)
  // Probe mode (commit=false): report qualification + coverage (cells
  // explained, newly finished colors) without freezing — the boundary sweep
  // ranks all qualifiers by coverage and commits only the best. First-
  // qualifier-wins froze cheap layers that strand the remainder (009d5c81:
  // "paint every shape 7" beat "color each shape by its shape").
  bool try_lock_layer(node *tree, int node_count, puzzle &p, int active,
                      int *out_correct = nullptr, int *out_new_colors = nullptr,
                      bool commit = true);
  // Shared evaluator: per-color totals/correct + schedule-purity (wrong paint
  // only under undone colors). Evaluates the first `pairs` train examples.
  bool layer_fit(node *tree, int node_count, puzzle &p, int pairs,
                 int color_total[10], int color_correct[10], int &correct);
  // Pick the focus color: the unfinished color with the largest residual.
  // exclude: when rotating, skip the stalled current focus.
  void pick_focus_color(int exclude = -1);
  void worker();
  void update_example_results(vector<example> &examples, const vector<candidate_result> &results, int limit = -1);
  void clear_test_current_views(puzzle &p);
  void update_test_current_views(puzzle &p, node *tree, int node_count, bool all_train_solved);
  bool check_old_tree_solves(puzzle &p, int active);
  // Greedy leave-one-out reduction of a candidate tree over the active train
  // examples. Repeatedly drops the single op whose removal most improves the
  // summed train score (node-count penalty included, so a true no-op strictly
  // wins by shedding its penalty), keeping the same solved set. Returns the
  // reduced tree (caller owns it) and its node count; may return `tree` itself
  // unchanged. Runs on each new best to keep the beam building from a clean,
  // short prefix instead of dragging dead ops forward.
  // deep = full-depth leave-k-out (lock commits only); default is the cheap k=1 pass
  node *reduce_best_tree(node *tree, puzzle &p, int active, int &out_node_count, bool deep = false);
  std::pair<float, bool> eval_best_tree_train(puzzle &p, int idx);
  static void invalidate_best_tree_cache(puzzle &p) { p.best_tree_eval.clear(); }
  int try_promote_examples(node *tree, puzzle &p, int &active, int node_count);
  void finalize_full_train_solve(puzzle &p);
  bool evaluate_test_after_solve(node *tree, puzzle &p, int node_count);
  void reset_test_display(puzzle &p);
  void refresh_best_display_scores(puzzle &p, int active, int node_count);
  void try_update_current_display(int target_idx, puzzle &p, node *tree, int node_count, const candidate_batch &res);

  global_app_state &m_app_state;
  solver_state m_solver_state;
  logger m_log;
  puzzle m_puzzle;
  bool m_puzzle_loaded = false;
  jthread m_thread;
  vector<int> *m_gpu_buffer = nullptr;
};

}

#include "fixed_dsl.hpp"
#include "dsl_executor.hpp"
#include "solver_executor.hpp"
