#pragma once
// THE SOLVER. Give it a puzzle, call step() until it is solved or you run out
// of time, read the best program off the puzzle. That is the whole contract.
// It owns no thread, knows no window, reads no settings file and plays no
// sound: a driver does those. The Kaggle main drives it in a plain loop; the
// GUI drives it from a thread of its own and watches it through
// solver_observer. Nothing in this folder may include anything from gui/.

#include <atomic>
#include "12_environment.hpp"
#include "11_dsl.cuh"   // GPU_KERNEL_MAX_OBJECTS, the cuda_dsl_* calls
#include "13_puzzle_loader.hpp"
#include "14_solver_log.hpp"

#include <algorithm>
#include <cstring>
#include "10_text_format.hpp"
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

    // Hashes arrive finalized (environment::mix64), but probing mixes again so
    // this table stays correct for any caller's numbering.
    size_t probe(uint64_t h) const { return mix64(h) & mask; }

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
    uint64_t state_hash = 0;
  };
  // A refinement slot on the unary chain is just how many steps down from the
  // root the replacement happens.
  struct slot_t {
    int depth = 0;
  };
  struct cache_entry { float score; uint64_t state_hash; };

  // Total order on candidates: score first, then the OP LIST'S OWN ORDER. Ties
  // are common — every additive selector is inert once the marks are already
  // there — and a tie broken by hash picks a spelling at random, so the beam
  // fills with whichever nonsense hashed highest. The ops are declared in an
  // order that means something, so the earlier word wins, and a shorter program
  // beats a longer one that starts the same.
  // Arrival order still breaks nothing, which is the point. Proven 2026-07-14 on
  // 007bbfb7: score-only ties let container iteration order pick the beam, and
  // fresh (50s, 11M attempts) vs switched (10s, 2.8M) diverged repeatably from
  // one 8-candidate ripple in level 2.
  static bool ranks_higher(const champion &a, const champion &b) {
    if (a.score != b.score) return a.score > b.score;
    const node *x = a.n, *y = b.n;
    for (; x && y; x = x->arg, y = y->arg) {
      if (x->type != y->type) return to_underlying(x->type) < to_underlying(y->type);
      if (x->value != y->value) return x->value < y->value;
    }
    return x == nullptr && y != nullptr;
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
  flat_hash_set_u64 visited_states;
  queue<node *> pending_evaluation;

  unordered_map<uint64_t, cache_entry> global_cache;
  float all_time_best_score = SCORE_MIN;

  int active_example_count = 1;
  // How many examples a fresh puzzle starts scoring against. One means the
  // usual promotion ladder; more means judge on that many from the first
  // candidate. Set once from --start-examples, clamped to the train set.
  int start_example_count = 1;
  // Stop after this many attempts, 0 meaning never. A count stops in the same
  // place every run; a clock stops wherever the machine happened to be.
  long long stop_after_attempts = 0;


  vector<champion> current_slot_candidates;
  vector<champion> level_candidates;

  bool is_generating_layer = false;

  // the beam member being refined (phase_1 wraps it directly in locked mode)
  node *m_current_base = nullptr;

  int total_candidates_this_slot = 0;

  // Sentence sweep: the sentences this level may use, and how far through them
  // the generator has got. sweep_beam_idx walks the beam, sweep_idx the sweep.
  vector<sentence> sweep;
  int sweep_beam_idx = 0;
  int sweep_idx = 0;
  bool sweep_ready = false;
  // What the generator actually did this level. Generate-and-reject has no place
  // here, so anything walked and not kept is a number that should be zero.
  long long pairs_walked = 0;
  long long pairs_kept = 0;
  long long pairs_already_seen = 0;
  long long members_skipped = 0;

  // Where a step's time actually goes, reported once per level.
  // Counters, not log lines — logging the hot path is what distorted the
  // first measurement.
  // ns_wait is not here: it lives in search_progress so the render thread can
  // read it, and this class reads it back for the line below.
  long long ns_generate = 0, ns_launch = 0, ns_harvest = 0, ns_idle = 0;
  long long ns_wait_at_last_level = 0;
  // Harvest, split three ways; harvest minus these is the unaccounted rest.
  long long ns_tally = 0, ns_winner = 0, ns_free = 0;
  // The winner block, split the same way.
  long long ns_fetch = 0, ns_analyze = 0, ns_best = 0;

  int stagnant_levels = 0;
  float prev_beam_top = SCORE_MIN;

  // A promotion that fills the train set IS the solve, but marking the task
  // solved can't happen at that moment (it can happen mid-update_best_solution,
  // whose own solved/pause writes would collide). The verified tree waits here;
  // step() marks the task solved at the top of the next one.
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
  search_progress &state;
  chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
  explicit gpu_wait_scope(search_progress &s) : state(s) {
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

// How the search is set up. Fixed for the solver's life.
struct solver_options {
  // Search a SENTENCE at a time instead of an op at a time: every candidate is
  // a whole (GET SELECT ACT COMBINE), so every candidate has painted and can be
  // scored honestly. Off = the op-at-a-time beam, for A/B.
  bool sentence_search = true;
  // Pin color-literal ops (AddColor/Recolor/RemoveColor/InteriorColor) to the
  // colors that side of the puzzle actually shows, during candidate
  // generation — collapses the 10x literal fan-out. Behavioral change, so it's
  // a switch: off = try all 10 colors (original behavior) for A/B + safe revert.
  bool pin_color_literals = true;
  // logs/<id>.txt and logs/inference/<id>.txt. For a person reading along;
  // off on Kaggle, where nobody reads them and the disk quota would not hold them.
  bool write_logs = false;
};

// Something the search did that a watcher may want to mark: a sound, a pause.
struct search_news {
  // Who found it. The beam, a promotion that filled the train set, or a program
  // inference spelled before the search began.
  enum class source { beam, promotion, inference };
  source from = source::beam;
  int promotions = 0;       // examples added to the active set by this tree
  bool improved = false;    // a new best that is neither a solve nor a promotion
  bool simplified = false;  // a solved puzzle got a shorter program
  bool train_solved = false;
  bool test_solved = false;
};

// WHAT THE SEARCH SHOWS SOMEBODY WATCHING. The solver never draws: it says what
// it did, and a watcher that wants pictures makes them. Every call arrives on
// the thread driving the solver, most with progress().mtx held, so a watcher
// drawing under that lock sees a whole update or none of it. With no watcher
// the solver skips the work that only makes pictures.
struct solver_observer {
  virtual ~solver_observer() = default;
  // A puzzle was just parsed, or failed to parse and is empty.
  virtual void loaded(const puzzle & /*p*/) {}
  // One example's picture in one column: what a tree painted there, or
  // nullptr for nothing.
  enum class column { current, best };
  virtual void painted(bool /*is_test*/, size_t /*example*/, column /*which*/,
                       const grid * /*g*/) {}
  // The score beside an example's CURRENT picture; below SCORE_MIN when
  // nothing has been tried.
  virtual void scored_current(bool /*is_test*/, size_t /*example*/, float /*score*/) {}
  // Something visible changed.
  virtual void changed() {}
  // A step that mattered. Arrives with the lock NOT held.
  virtual void news(const search_news & /*n*/) {}
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
  explicit beam_solver(solver_options options = {}, solver_observer *observer = nullptr)
      : m_options(options), m_observer(observer), m_log(m_progress, options.write_logs) {}

  // ~search_state releases every tree the search still owns; this is the one
  // it lent to whoever is watching.
  ~beam_solver() { environment::remove(m_progress.current_tree); }

  beam_solver(const beam_solver &) = delete;
  beam_solver &operator=(const beam_solver &) = delete;

  // The watcher, or none. Set before the first load.
  void set_observer(solver_observer *observer) { m_observer = observer; }

  // Both of these live on the search state, so a load puts them back to their
  // defaults: set them after load() for them to hold on that puzzle.
  // Stop filling batches after this many attempts, 0 meaning never. A count
  // stops in the same place every run; a clock stops wherever the machine
  // happened to be. What stopping means (a pause) is the driver's to decide.
  void set_stop_after_attempts(long long n) { stop_after_attempts = n > 0 ? n : 0; }
  long long get_stop_after_attempts() const { return stop_after_attempts; }
  // Seeds the live count too, so it holds from the very next step.
  void set_start_example_count(int n) {
    start_example_count = n < 1 ? 1 : n;
    active_example_count = start_example_count;
  }

  // Debug and tree lines in the log as well as timing, when it writes at all.
  void set_verbose(bool on) { m_log.verbose = on; }
  void log_debug(const string &msg) { m_log.debug(msg); }
  void log_timing(const string &msg) { m_log.timing(msg); }
  void log_score_formula(const node *tree, const vector<gpu_score_terms> &terms);
  logger &log() { return m_log; }

  search_progress &progress() { return m_progress; }
  const search_progress &progress() const { return m_progress; }
  bool loaded() const { return m_puzzle_loaded; }
  // Takes no index: one puzzle is loaded at a time, so this is that one.
  // Another thread reads it under progress().mtx.
  const puzzle &get_puzzle() const {
    static puzzle dummy;
    return m_puzzle_loaded ? m_puzzle : dummy;
  }

  // Parse `task` and start over on it: every trace of the last puzzle goes,
  // search, pipeline and all. Only parses: nothing is inferred until infer()
  // asks. `test_solutions` are its answers when known, so the test row can be
  // scored. `label` only heads the inference readout.
  bool load(const challenge_task &task, const vector<grid> *test_solutions,
            const puzzle_label &label = {});

  // Inference, once and whole, then each program it spelled, once, before the
  // search spends a single candidate. step() asks for both itself.
  void infer();
  // Only the first half: the facts, without trying what they spelled. What
  // running any program on the puzzle needs first.
  void read_facts();

  // ONE TURN OF THE SEARCH: generate a batch, launch it on the GPU, and score
  // the batch launched last turn while this one runs. Call it until the
  // puzzle is solved or time is up; it never blocks for long.
  void step();

  // A whole program in the solver's own spelling, made the best and scored.
  // The import path, and how what inference spelled is tried.
  void apply_program(const string &program);

  // The one reset, shared by every transition: reinstantiate the whole
  // per-puzzle state, same values as opening.
  void reset_search() {
    search_state &s = *this;
    s.~search_state();
    new (&s) search_state();
  }

  void report_score(const node *n, float score, uint64_t g_hash);

  // True if a color-literal op of `type` with literal `value` is allowed for the
  // current puzzle. When pinning is on, a literal is pinned to the side it
  // speaks about — read literals to the colors some train INPUT has, written
  // ones to the colors some train OUTPUT has — and BidiColorMap variants to the
  // swap pairs learned from the train diffs, collapsing the 10x/45x fan-outs.
  // Anything else (non-color ops, pinning off, no learned set, InteriorColor's
  // -1 "auto") is always allowed.
  inline bool color_literal_allowed(node_type type, int16_t value) const {
    if (!m_options.pin_color_literals) return true;
    if (type == node_type::BidiColorMap) {
      if (m_puzzle.inf.color_swaps.count == 0) return true;  // not learned -> full fan-out
      if (value < 0 || value >= COLOR_PAIR_COUNT) return true;   // out-of-range guard
      return m_puzzle.inf.color_swaps.variant[value];
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
    const bool *side = reads_color ? m_puzzle.inf.colors.in_any_input
                                   : m_puzzle.inf.colors.in_any_output;
    // NO SET, NO PINNING. Asked of the set itself, not of a count kept beside
    // it — a count is the same fact said twice, and it drifted the moment one
    // of the two was filled and the other was not.
    bool anything_known = false;
    for (int i = 0; i < COLOR_COUNT; ++i) anything_known |= side[i];
    if (!anything_known) return true;
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
    const auto& background = m_puzzle.inf.background;
    int8_t bg = read_op ? background.input_elected() : background.output_elected();
    if (write_op && bg < 0) bg = background.input_elected();
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
  void enumerate_sentences(bool sentence_opens_program, vector<sentence> &out) const;
  void update_history_from_beam();
  void collect_subtrees(node *n, unordered_set<uint64_t> &existing, int &added);
  void rebuild_skeleton_beam();
  void prepare_next_level_slots();
  void find_slots_recursive(node *n, int depth, vector<slot_t> &out, set<int> &visited_depths);
  void reset_d2_generator();

  static constexpr int MAX_STAGNANT_LEVELS = 5;
  static constexpr int MAX_TILE_OPS = 2;

  static int count_tile_ops(const node *n) {
    int c = 0;
    for (; n; n = n->arg)
      if (n->type == node_type::TileX || n->type == node_type::TileY) ++c;
    return c;
  }

  phase_result generate_phase_prune(node* target, int &batch_count, node *&candidate_sub);
  phase_result generate_phase_0(const slot_t &slot, int &batch_count, node *&candidate_sub, bool &created_new);
  phase_result generate_phase_1(const slot_t &slot, int &batch_count, node *&candidate_sub, bool &created_new);
  phase_result execute_current_phase(const slot_t &slot, node* target, int &batch_count, node *&candidate_sub, bool &created_new);

  bool generate_branch();
  // One batch of whole sentences. True when the level has no more to give.
  bool generate_sentence_batch();
  void commit_sentence_level();
  void log_level_timing();
  void commit_and_advance_level();
  void advance_beam_idx();
  bool process_current_phase(const slot_t &slot, node* base, node* target, int& batch_count);
  bool add_candidate_if_new(node* base, int depth, node* candidate_sub, bool created_new);
  void reset_generator_counters();
  void merge_slot_candidates();
  void commit_level_champion();

public:
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

private:
  // THE TWO-SLOT GPU PIPELINE: what is still in flight between one step and
  // the next. A batch launched this step is scored the next, while its
  // successor runs.
  // A winner's picture is wanted for the watcher and the grid dedup, not for
  // the decision, so it is asked for now and read one harvest later.
  // Everything the read still needs is kept here, because the batch's own
  // scores are overwritten by then.
  struct deferred_winner {
    node *tree = nullptr;
    int node_count = 0, active = 0;
    int load_id = -1;
    float score[16] = {};
    bool solved[16] = {};
    gpu_score_terms terms[16] = {};
    bool valid = false;
  };
  vector<node *> m_prev_nodes[2];
  bool m_slot_pending[2] = {false, false};
  deferred_winner m_deferred[2];
  // Drain what the GPU still holds for this solver, then return the node pool
  // to its opening state. Every tree anywhere is dead after this.
  void drain_pipeline();

  bool pump_generate(std::atomic<long long> *global_attempts);
  void log_tree_record(node *tree, float score, long long attempts);
  void update_best_solution(int load_id, puzzle &p, node *tree, int node_count, candidate_batch &res);
  void update_example_results(bool is_test, const vector<candidate_result> &results, int limit = -1);
  void update_test_current_views(puzzle &p, node *tree, int node_count, bool all_train_solved);
  bool check_old_tree_solves(puzzle &p, int active);
  // apply_program without the lock, for a caller already holding it.
  void score_program(const string &program);
  bool try_what_inference_said(puzzle &p);
  // Gated on the ATTEMPT, not on `inferred`: the pane can infer a puzzle long
  // before anything asks what its reading spelled.
  bool asked_inference_yet = false;
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
  void clear_views(puzzle &p, bool is_test);
  void refresh_best_display_scores(puzzle &p, int active, int node_count);
  void try_update_current_display(int load_id, puzzle &p, node *tree, int node_count, const candidate_batch &res);

  // What a watcher is told, each a no-op with nobody watching.
  void show(bool is_test, size_t i, solver_observer::column which, const grid *g) {
    if (m_observer) m_observer->painted(is_test, i, which, g);
  }
  // A tree the search now stands on: its picture is the best and the current.
  void show_both(bool is_test, size_t i, const grid *g) {
    show(is_test, i, solver_observer::column::best, g);
    show(is_test, i, solver_observer::column::current, g);
  }
  void show_current_score(bool is_test, size_t i, float score) {
    if (m_observer) m_observer->scored_current(is_test, i, score);
  }
  // Nothing tried on this example: no picture, and a score below any real one.
  void clear_current(bool is_test, size_t i) {
    show_current_score(is_test, i, SCORE_MIN - 1.0f);
    show(is_test, i, solver_observer::column::current, nullptr);
  }
  // The best score is the search's own, kept to compare trees by; the picture
  // is the watcher's.
  void clear_best(float &best, bool is_test, size_t i) {
    best = SCORE_MIN - 1.0f;
    show(is_test, i, solver_observer::column::best, nullptr);
  }
  // One number, two homes: the search keeps it to compare trees by, the
  // watcher keeps it to label the CURRENT column with.
  void set_example_scores(float &best, bool is_test, size_t i, float score) {
    best = score;
    show_current_score(is_test, i, score);
  }
  void changed() {
    if (m_observer) m_observer->changed();
  }
  void news(const search_news &n) {
    if (m_observer) m_observer->news(n);
  }

  atomic<long long> *m_global_attempts = nullptr;

  solver_options m_options;
  solver_observer *m_observer = nullptr;
  search_progress m_progress;
  logger m_log;
  puzzle m_puzzle;
  bool m_puzzle_loaded = false;
  puzzle_label m_label;
  // Counts loads, so a result the GPU hands back for a puzzle that has since
  // been replaced is recognised and dropped.
  int m_load_id = 0;
};

}

#include "16_fixed_dsl.hpp"
#include "17_dsl_executor.hpp"
#include "18_solver_executor.hpp"
