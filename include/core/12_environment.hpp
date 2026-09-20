#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Single home of the DSL op list (NODE_PRIMITIVES) and the node_type enum.
#include "11_dsl.cuh"
// The one door to everything below the engine: infer(), and through it the
// core types (arc_color, grid, example, output_dimension_rule) and the limits.
#include "7_infer.hpp"

namespace arc {
	using namespace std;

	// Official C++23 std::to_underlying implementation (available in C++20 via this helper)
	template <typename Enum>
	[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept {
		return static_cast<std::underlying_type_t<Enum>>(e);
	}

	// Squashing many numbers into one, so the search can ask "have I already tried
	// this?" by comparing a single number. Start at 0, fold once per value, and
	// finish with mix64 -- a raw fold spreads its bits badly, and two near-miss
	// programs landing on one number means the second is never tried.
	inline uint64_t hash_fold(uint64_t seed, uint64_t value) {
		return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
	}

	// splitmix64 finalizer: about two cycles, and every bit ends up doing work.
	inline uint64_t mix64(uint64_t x) {
		x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
		x ^= x >> 27; x *= 0x94d049bb133111ebULL;
		x ^= x >> 31; return x;
	}

	// A grid as one number, for the score log to tag a run with. Only ever
	// printed, never compared, so it does not finish with mix64.
	inline uint64_t grid_hash(const grid& g) {
		uint64_t h = 0;
		for (int r = 0; r < g.height; ++r)
			for (int c = 0; c < g.width; ++c) h = hash_fold(h, g.data[r][c]);
		h = hash_fold(h, g.width);
		h = hash_fold(h, g.height);
		return h;
	}

	// --- Search Sentinels ---
	constexpr int NODE_COUNT_MAX = 1000000;
	constexpr int DISTANCE_MAX = 100;
	// The worst a score can be, so anything untried sits below every real one.
	constexpr float SCORE_MIN = -1000000.0f;
	constexpr size_t BEAM_WIDTH = 128;
	const int MAX_BATCH_SIZE = 5000;
	constexpr int MAX_SEARCH_DEPTH = 32;

	// --- Program Tree Constraints ---

	// Scoring weights live in 11_dsl.cuh (GPU_SCORE_W_*, single home).
	// All scoring runs in the GPU kernel (dsl_exec_and_score_kernel_v2);
	// the CPU uses the weights only for the debug/analysis breakdown.

	// Max distinct (color,w,h,count) tuples we track per target example.
	constexpr int MAX_TARGET_OBJECTS = 16;

	// node_type and the NODE_PRIMITIVES op list live in 11_dsl.cuh (single home).

	inline constexpr const char* type_name[] = {
	#define X(id, name, low, high, role) name,
		NODE_PRIMITIVES
	#undef X
	};

	inline constexpr node_type primitive_types[] = {
	#define X(id, n, low, high, role) node_type::id,
		NODE_PRIMITIVES
	#undef X
	};
	constexpr int primitive_count = (int)size(primitive_types);

	// --- Shared node-type metadata helpers (used by serializer, GUI, solver) ---

	inline constexpr int type_var_low[] = {
	#define X(id, n, low, high, role) low,
		NODE_PRIMITIVES
	#undef X
	};

	inline constexpr int type_var_high[] = {
	#define X(id, n, low, high, role) high,
		NODE_PRIMITIVES
	#undef X
	};

	// An op takes an argument only when its range holds more than one value.
	inline bool is_annotated(node_type type) {
		return type_var_low[to_underlying(type)] != type_var_high[to_underlying(type)];
	}

	// The printed annotation IS the value, so reading one back is reading a
	// number: no op translates its own argument, so nothing has to be undone.
	inline void append_annotation(node_type type, int16_t value, string& out) {
		if (is_annotated(type)) {
			out += '[';
			out += to_string((int)value);
			out += ']';
		}
	}

	inline bool is_parameterized(node_type type) {
		return is_annotated(type);
	}

	inline int get_variant_count(node_type type) {
		return type_var_high[to_underlying(type)] - type_var_low[to_underlying(type)] + 1;
	}

	inline int16_t get_variant_value(node_type t, int idx) {
		return (int16_t)(type_var_low[to_underlying(t)] + idx);
	}

	// Not thread-safe: all node alloc/free must happen on the thread driving the
	// solver, or on another thread while holding search_progress::mtx (which the
	// solver also takes).
	// Cap chunk growth so the pool can't briefly request multi-GB blocks
	// from upstream and trip the 8 GB job-object limit set in enforce_memory_limit().
	// Without this cap the doubling growth strategy crashed the app with bad_alloc
	// after ~10 minutes once the pool tried to grow past the job limit.
	inline std::pmr::unsynchronized_pool_resource node_pool{
		std::pmr::pool_options{
			/*.max_blocks_per_chunk =*/ 4096,
			/*.largest_required_pool_block =*/ 256
		}
	};

	struct node {
		// Non-atomic: only the thread driving the solver mutates nodes. Any
		// other thread reads node fields under search_progress::mtx (which the
		// solver also takes), so refcount changes are already serialized.
		mutable int ref_count;
		// every op is unary, so a program is a chain; null = chain start
		node* arg = nullptr;
		uint64_t hash = 0;
		node_type type;
		int16_t value = 0;
		int8_t max_depth = 1;
		// Generation debt (host-side, computed once at birth from arg + type):
		// the state AFTER this op runs, so the generator can steer the next op.
		bool sel_waiting = false;    // a selection made, no act on it yet
		bool work_unlanded = false;  // touched the list since the last combine
		bool something_picked = false;  // marks exist on the list right now
		node_type last_combine = node_type::type_count;  // kind of the most recent combine (type_count = none yet)

		node() : ref_count(1) {}
		node(node_type t, int16_t v = 0) : ref_count(1), type(t), value(v) {}
		node(const node& other) : ref_count(1), arg(other.arg), hash(other.hash), type(other.type), value(other.value), max_depth(other.max_depth),
			sel_waiting(other.sel_waiting), work_unlanded(other.work_unlanded),
			something_picked(other.something_picked), last_combine(other.last_combine) {}
		node& operator=(const node& other) {
			if (this != &other) {
				arg = other.arg; hash = other.hash;
				type = other.type; value = other.value; max_depth = other.max_depth;
				sel_waiting = other.sel_waiting; work_unlanded = other.work_unlanded;
				something_picked = other.something_picked; last_combine = other.last_combine;
			}
			return *this;
		}

		void update_depth() {
			max_depth = arg ? static_cast<int8_t>(arg->max_depth + 1) : (int8_t)1;
		}

		static void* operator new(size_t size) {
			return node_pool.allocate(size);
		}
		static void operator delete(void* ptr) {
			node_pool.deallocate(ptr, sizeof(node));
		}
	};

	// --- Generation debt: set the three debt fields on `n` from its arg + type.
	// Called once at birth (create_unary_node and the starter builders), so the
	// debt never drifts. `n->arg` is the chain so far (null = n is a GET start).
	inline void set_generation_debt(node* n) {
		bool prev_sel = n->arg ? n->arg->sel_waiting : false;
		bool prev_picked = n->arg ? n->arg->something_picked : false;
		node_type prev_combine = n->arg ? n->arg->last_combine : node_type::type_count;
		node_type t = n->type;
		if (is_get_op(t)) {
			// A fresh source: nothing selected, but the new list owes a landing.
			n->sel_waiting = false; n->work_unlanded = true; n->last_combine = prev_combine;
			// A fetch hands back a brand new list, and new objects carry no marks --
			// except CurrentObjects, which hands back the SAME objects it was given,
			// marks and all. Saying otherwise made the grammar demand a selection
			// the kernel had already made, and every additive selector in that slot
			// was then a different spelling of one program.
			n->something_picked = (t == node_type::CurrentObjects) ? prev_picked : false;
		} else if (is_select_op(t)) {
			// A selection now waits to be acted on; touched the list.
			n->sel_waiting = true; n->work_unlanded = true; n->last_combine = prev_combine;
			n->something_picked = true;
		} else if (is_combine_op(t)) {
			// Lands the list: work discharged, selection must already be gone.
			n->sel_waiting = prev_sel; n->work_unlanded = false; n->last_combine = t;
			// A paint passes the list through untouched, marks included.
			n->something_picked = prev_picked;
		} else { // ACT
			// The first act clears any waiting selection; re-owes a landing.
			n->sel_waiting = false; n->work_unlanded = true; n->last_combine = prev_combine;
			// Marks ride along on the objects an act hands forward -- unless the
			// act hands back different objects, and then there is nothing for a
			// mark to be attached to.
			n->something_picked = hands_back_new_objects(t) ? false : prev_picked;
		}
	}

	// --- The legality rules (each op is legal when ...): true if wrapping the
	// chain `arg` with `type` is allowed. `arg == null` means `type` is the very
	// first op. One home for the five role rules the generator steers by.
	inline bool legal_next(node* arg, node_type type) {
		bool sel_waiting = arg ? arg->sel_waiting : false;
		// Rule 0: an op that does nothing without a pick is never worth offering
		// before one. It would only respell the chain it wraps, one op longer.
		// `SelectAll` is how a chain says "all of them", so nothing is lost.
		if (needs_a_pick(type) && !(arg && arg->something_picked)) return false;
		// Rule 1: NOTHING IS IMPLICIT. A program names its own source, so the
		// first op is a GET and only a GET; later a GET is a refetch, legal right
		// after a COMBINE. Every op past the first therefore HAS a source by
		// induction, which is why no rule below has to ask.
		if (arg == nullptr) return is_get_op(type) && type != node_type::CurrentObjects;
		if (is_get_op(type)) return is_combine_op(arg->type);
		if (is_select_op(type)) return true;              // Rule 2
		if (is_act_op(type))    return true;              // Rule 3
		if (is_combine_op(type)) {                        // Rule 4
			// same-type-repeat barred; cross-type combine chains stay legal.
			bool same_repeat = is_combine_op(arg->type) && arg->type == type;
			return !sel_waiting && !same_repeat;
		}
		return false;
	}

	// Rule 5: a program may STOP (be a complete solution) only when nothing is
	// waiting to be acted on and the last work has been landed.
	inline bool legal_stop(node* tail) {
		if (!tail) return false;                  // empty program is not a solution
		return !tail->sel_waiting && !tail->work_unlanded;
	}

	// Compact summary of one connected component in a grid: bbox size,
	// position, pixel count. This is all the CC-match score needs — no per-cell mask.
	// Multiset match: for each attempt CC, find an unused target CC that matches
	// by (w,h,count); count it as shape match.
	// Position score is the continuous bonus per shape-
	// matched pair, normalized to [0,1] by max_dim and summed across all matches.
	// Returns {shape_matches, position_score}. O(N*M) with
	// N,M ≤ MAX_TARGET_OBJECTS — fine.
	//
	// Known limitation: shape match is exact on (w,h,count). A near-miss CC
	// (e.g. a 2x3 object where the target is 3x3) earns 0 shape/position
	// reward and the score collapses back to the per-pixel terms. That's why
	// the pre-CC-term score function happily settled on outputs like puzzle
	// 09629e4f's wide-rectangle near-miss: right colors + right approximate
	// positions = high histogram + distance + rarity, while the structurally
	// correct shape was several pixel-edits away through a per-pixel-penalty
	// valley. Loosening this to a fuzzy shape match (e.g. IoU-style) would
	// extend the gradient to these near-misses but risks rewarding "right
	// colors in any object" — out of scope here; the score reflects the strict
	// "same shape exists" semantics by design.
	struct cc_match_result {
		int shape_matches = 0;
		float position_score = 0.0f;  // sum over matched pairs, each in [0,1]
	};
	static inline cc_match_result match_object_summaries(
		const object_summary* a, int an,
		const object_summary* t, int tn,
		int /*unused: grid max_dim, kept for ABI*/)
	{
		bool used[MAX_TARGET_OBJECTS]{};
		cc_match_result r;
		for (int i = 0; i < an; ++i) {
			int best_j = -1;
			for (int j = 0; j < tn; ++j) {
				if (used[j]) continue;
				if (a[i].w != t[j].w || a[i].h != t[j].h || a[i].count != t[j].count) continue;
				best_j = j;
				break;
			}
			if (best_j >= 0) {
				used[best_j] = true;
				r.shape_matches++;
				// Continuous position reward, normalized by the target CC's own
				// footprint (max(w,h)) — not by grid size. A 1px miss on a 3x3
				// block scores ~0.83 (mostly right); the same miss on a 1x1 dot
				// scores 0.5 (real penalty). Symmetric across CC sizes.
				int acx = a[i].x0 * 2 + (a[i].w - 1);
				int acy = a[i].y0 * 2 + (a[i].h - 1);
				int tcx = t[best_j].x0 * 2 + (t[best_j].w - 1);
				int tcy = t[best_j].y0 * 2 + (t[best_j].h - 1);
				int dx = acx > tcx ? acx - tcx : tcx - acx;
				int dy = acy > tcy ? acy - tcy : tcy - acy;
				int cheb = (dx > dy ? dx : dy);  // halved units
				int tsize = t[best_j].w > t[best_j].h ? t[best_j].w : t[best_j].h;
				if (tsize < 1) tsize = 1;
				float norm = (float)cheb / (float)(tsize * 2);
				float ps = 1.0f - norm;
				if (ps < 0.0f) ps = 0.0f;
				r.position_score += ps;
			}
		}
		return r;
	}


	struct puzzle {
		string id;
		vector<example> train, test;
		// EVERYTHING INFERENCE WORKED OUT, kept whole. It used to be computed
		// into a local and hand-copied out field by field before that local
		// died, which is why there were reduced copies of it here and a second
		// set of copies again on the way to the GPU. The object outlives the
		// function now, so nothing has to be copied out of it.
		inference inf;
		output_dimension_rule rule;
		bool invariant_colors[COLOR_COUNT] = {}; // one bit: the elected input background
		// Allowed colors for color-LITERAL ops, one set per SIDE: a literal the
		// chain READS by can only name a color some train input has, one it
		// WRITES can only name a color some train output has. Bridged from
		// inference (in_any_input / in_any_output). A count of 0 means "not set
		// / don't pin" (e.g. empty puzzle) so generation falls back to all 10.
		// Ops inference declared dead for this puzzle: their backing fact is
		// absent, so they can only no-op. Verdict filled once, right after
		// infer(); generation never offers a dead op.
		bool dead_ops[(int)node_type::type_count] = {};
		// PROGRAMS INFERENCE ALREADY SPELLED, most likely first. Bridged from
		// say_the_reading; empty when the reading had nothing sayable. Tried
		// before the search starts -- a sentence that was read off the pairs is
		// one the beam never has to score its way to.
		vector<string> suggested_programs;
		// Learned color swaps, by BidiColorMap's variant index k in [0,44].
		// Bridged from inference (inf.color_swaps). Pins the op's 45-way fan-out
		// in candidate generation; count==0 means "not learned / don't pin".
		// Learned shape->color lookup for the ShapeColorMap op: hash of a small
		// glyph's occupancy -> color the main object is repainted to. Bridged from
		// inference (inf.shape_color_map); count==0 means "not learned" -> op no-ops.
		// Per-train-pair map of cells the input already gets right (input ==
		// output). Bridged from inference (inf.unchanged.kept, Identity puzzles
		// only; empty otherwise). Layer-lock commit uses it: wrong paint on a
		// kept cell is vandalism (nothing repaints it), wrong paint on a
		// changed cell is fine (a later layer overwrites it anyway).
		node* best_tree = nullptr;
		int best_node_count = NODE_COUNT_MAX;
		// WHAT THE CHAMPION SCORES ON EACH PAIR, indexed like train and test.
		// It describes best_tree, not the pair, which is why it lives here
		// beside it: replace the tree and every one of these is stale together.
		// The search reads the sum back to decide whether a new tree beats the
		// old one, so it is not a picture and cannot live with the GUI.
		vector<float> best_train_scores, best_test_scores;
		// Grown on demand, so nothing has to keep these sized in step with the
		// pairs; an index never scored yet reads as the worst a score can be.
		float& best_score(bool is_test, size_t i) {
			vector<float>& s = is_test ? best_test_scores : best_train_scores;
			if (s.size() <= i) s.resize(i + 1, SCORE_MIN);
			return s[i];
		}
		float best_score_or_worst(bool is_test, size_t i) const {
			const vector<float>& s = is_test ? best_test_scores : best_train_scores;
			return i < s.size() ? s[i] : SCORE_MIN;
		}
		// Has inference run on this puzzle yet. It is the expensive half of a
		// load and nothing reads it until the pane opens or the search starts,
		// so it is done on demand and this says whether it has been.
		bool inferred = false;
		bool is_solved = false;
		bool is_test_solved = false;
		// Raw (un-penalized) score+solved for best_tree on each train example.
		// Filled lazily on first read after best_tree is swapped; cleared by
		// invalidate_best_tree_cache. The grid output is intentionally NOT cached
		// because callers either don't need it or write into a per-call temp.
		struct best_tree_slot { float score = 0; bool solved = false; bool valid = false; };
		vector<best_tree_slot> best_tree_eval;
	};

	struct flat_program_op {
		node_type type;
		int16_t value = 0;
	};

	struct flat_program {
		// Programs are padded to the GPU batch length, so this IS the GPU limit.
		static constexpr int MAX_OPS = GPU_MAX_OPS;
		array<flat_program_op, MAX_OPS> ops{};
		int count = 0;

		void clear() { count = 0; }

		bool push(node_type type, int16_t value) {
			if (count >= MAX_OPS)
				return false;
			ops[count++] = { type, value };
			return true;
		}
	};

	// A SENTENCE is one beat of the grammar: fetch a list, mark part of it, change
	// what is marked, land it on the canvas. The rules the four roles obey live
	// with legal_next above; this is the shape they make.
	struct sentence {
		flat_program_op get{}, select{}, act{}, combine{};
	};

	// A program read as sentences. A program IS sentences and nothing else —
	// (GET SELECT ACT COMBINE)* — so one that does not divide into whole beats is
	// not a program this reading can name. broke_at is the op that did not fit, so
	// a caller can say where rather than only that it failed.
	struct program_sentences {
		static constexpr int MAX_SENTENCES = flat_program::MAX_OPS / 4;
		array<sentence, MAX_SENTENCES> sentences{};
		int count = 0;
		bool ok = false;
		int broke_at = -1;
	};

	// Does each slot hold the role its beat calls for?
	inline bool well_formed(const sentence& one) {
		return is_get_op(one.get.type) && is_select_op(one.select.type) &&
		       is_act_op(one.act.type) && is_combine_op(one.combine.type);
	}

	// Read a flat program as four-beat sentences. Every op must carry the role its
	// beat calls for, and the last op must close a sentence — a beat left open is a
	// sentence stopped mid-word.
	inline program_sentences read_sentences(const flat_program& program) {
		program_sentences out;
		sentence building{};
		for (int i = 0; i < program.count; ++i) {
			const flat_program_op& op = program.ops[i];
			const int beat = i % 4;
			const bool fits = beat == 0 ? is_get_op(op.type)
			                : beat == 1 ? is_select_op(op.type)
			                : beat == 2 ? is_act_op(op.type)
			                            : is_combine_op(op.type);
			if (!fits) {
				out.broke_at = i;
				return out;
			}
			if (beat == 0) building.get = op;
			else if (beat == 1) building.select = op;
			else if (beat == 2) building.act = op;
			else {
				building.combine = op;
				if (out.count >= program_sentences::MAX_SENTENCES) {
					out.broke_at = i;
					return out;
				}
				out.sentences[out.count++] = building;
				building = sentence{};
			}
		}
		if (program.count == 0 || program.count % 4 != 0) {
			out.broke_at = program.count;
			return out;
		}
		out.ok = true;
		return out;
	}

	// Write a sentence onto the end of a program, in beat order. Refused when a
	// slot holds the wrong role or the program has no room for all four — half a
	// sentence is not a sentence, so nothing is written unless all of it can be.
	inline bool write_sentence(const sentence& one, flat_program& program) {
		if (!well_formed(one)) return false;
		if (program.count + 4 > flat_program::MAX_OPS) return false;
		program.push(one.get.type, one.get.value);
		program.push(one.select.type, one.select.value);
		program.push(one.act.type, one.act.value);
		program.push(one.combine.type, one.combine.value);
		return true;
	}

	struct flat_program_batch {
		// Caps the accumulated chunk (so it also caps job_count at active=1). Heap-
		// allocated only (make_unique), so growing this array is safe. Matches
		// TARGET_JOBS (2048) — the generator can't fill more than that per level anyway.
		static constexpr int MAX_PROGRAMS = 2048;
		static constexpr int MAX_OPS = flat_program::MAX_OPS;

		array<uint16_t, MAX_PROGRAMS> lengths{};
		array<node_type, MAX_PROGRAMS * MAX_OPS> op_types{};
		array<int16_t, MAX_PROGRAMS * MAX_OPS> op_values{};
		int count = 0;

		void clear() {
			count = 0;
			lengths.fill(0);
		}

		bool push(const flat_program& program) {
			if (count >= MAX_PROGRAMS || program.count < 0 || program.count > MAX_OPS)
				return false;

			lengths[count] = (uint16_t)program.count;
			const int base = count * MAX_OPS;
			for (int i = 0; i < program.count; ++i) {
				op_types[base + i] = program.ops[i].type;
				op_values[base + i] = program.ops[i].value;
			}
			for (int i = program.count; i < MAX_OPS; ++i) {
				op_types[base + i] = node_type::Input;
				op_values[base + i] = 0;
			}

			++count;
			return true;
		}

		const node_type* program_types(int idx) const {
			return op_types.data() + idx * MAX_OPS;
		}

		const int16_t* program_values(int idx) const {
			return op_values.data() + idx * MAX_OPS;
		}
	};


	class environment {
	public:
		virtual ~environment() = default;

		// THE SCORER'S TARGETS: one output segmented into per-color connected
		// components on the background inference elected, written the way the
		// kernel reads them. Used by the CC-match score term to reward partial
		// structural progress. Returns how many.
		//
		// WORKED OUT WHERE THEY ARE UPLOADED, for the main search and the
		// scratch path alike, and never kept on the puzzle: a kept copy was read
		// against whatever background stood when it was made, which before
		// inference is none, and the two paths then scored the same program
		// differently.
		//
		// STAYS IN THE ENGINE — deliberately NOT moved to inference/. Despite
		// the detect_* name and sitting in the same load block as the migrated
		// invariant extractors, this is NOT an invariant: it reads NO direction
		// (no across_inputs / across_outputs / in_to_out comparison). It segments
		// each output INDEPENDENTLY and keeps a SEPARATE result per train pair —
		// per-output scorer prep, not a cross-example fact. By the inference
		// backbone's own definition (an invariant is read along a direction and
		// holds across all pairs) it fails on both counts, so it belongs to the
		// scorer, not the inference layer. Do not "finish the migration" by
		// pulling this — the invariant migration is complete at four extractors.
		static inline int8_t fill_target_objects(const puzzle& p, const grid& output,
		                                         gpu_object_summary* dst) {
			object_summary found[MAX_TARGET_OBJECTS];
			int count = find_objects_by_color(output, p.inf.background.output_elected(),
			                                  found, MAX_TARGET_OBJECTS);
			if (count > GPU_MAX_TARGET_OBJECTS) count = GPU_MAX_TARGET_OBJECTS;
			for (int k = 0; k < count; ++k) {
				dst[k].w = found[k].w;
				dst[k].h = found[k].h;
				dst[k].x0 = found[k].x0;
				dst[k].y0 = found[k].y0;
				dst[k].count = found[k].count;
			}
			for (int k = count; k < GPU_MAX_TARGET_OBJECTS; ++k) dst[k] = {};
			return (int8_t)count;
		}

		// Detect invariant colors: present in ALL training inputs.
		// MOVED to 7_infer.hpp (arc::detect_color_presence).
		// Reading the always-present (across_inputs) colors is inference's job. The
		// engine only needs the single dominant background color for
		// segmentation (Colors skips it, Split erases it), and takes it from
		// inf.colors.background_color. p.invariant_colors is the one place that
		// still holds a copy, because fixed_init_context cannot name `inference`.

		// OutputColorsBySize: train output colors ranked by total pixel count (desc),
		// summed across all train outputs. Verified consistent: every train output must
		// produce the same color sequence under by-pixel-count desc. If inconsistent,
		// count = 0 (op no-ops). Includes 0 if it appears in outputs.
		// NOTE: this counts pixels per color (not per connected region). For 08ed6ac7
		// every color = exactly one connected region so the two coincide. For puzzles
		// where one color has multiple disjoint regions of different sizes, a connected-
		// components-aware variant is needed (see notes/2_dsl/learned_orderings.txt).
		// detect_output_colors_by_size MOVED to 7_infer.hpp
		// (arc::detect_output_colors_by_size). Ranking across-outputs colors by
		// pixel count is inference's job; fill_puzzle_level hands
		// inf.colors.output_colors_by_size / _count straight to the kernel for the
		// OutputColorsBySize op.

		// detect_output_colors MOVED to 7_infer.hpp (arc::detect_output_colors).
		// Reading the across_outputs palette is inference's job. Nothing in the
		// engine consumes the ordered palette since the InferredOutputColors op
		// was deleted; the by-size ranking is the surviving consumer.

		// detect_size_rule MOVED to 7_infer.hpp (arc::output_dimension_rule::infer_rule).
		// Reading a rule from the examples is inference's job; the output_dimension_rule TYPE
		// stays here because the DSL/GPU path sizes output grids from it. The
		// engine no longer computes it directly — puzzle_loader calls arc::infer(p),
		// which returns an `inference` object; its output_dimension_rule is copied to p.rule.
		// See 7_infer.hpp for the verified-foundation notes (1,147-puzzle corpus).

		// What one node's hash is, from its parts alone. Single home of the
		// recipe: update_hash finalizes real nodes with it, and the beam
		// predicts the same number without building a tree
		// (beam_solver::predict_replaced_hash), so the two cannot drift.
		// The search compares these for EQUALITY -- two programs landing on one
		// number are one program as far as it can tell, and the second is never
		// tried -- so the weak fold is finalized before it leaves.
		static inline uint64_t node_hash(int type, uint64_t value,
		                                 uint64_t child_hash, bool has_child) {
			uint64_t h = 0x9e3779b97f4a7c15ULL + (uint64_t)type;
			h = hash_fold(h, value);
			if (has_child) h = hash_fold(h, child_hash);
			return mix64(h);
		}

		static inline void update_hash(node* n) {
			if (!n)
				return;
			n->update_depth();
			set_generation_debt(n);  // debt is finalized here, the one node-finalize chokepoint
			n->hash = node_hash(to_underlying(n->type), (uint64_t)n->value,
			                    n->arg ? n->arg->hash : 0, n->arg != nullptr);
		}


		static inline int count_nodes(const node* n) {
			int c = 0;
			for (; n; n = n->arg) ++c;
			return c;
		}

		static inline void remove(node* n) {
			if (!n)
				return;
			if (--n->ref_count == 0) {
				remove(n->arg);
				delete n;
			}
		}

		static inline bool flatten_unary_program(const node* n, flat_program& out) {
			out.clear();
			if (!n)
				return true;

			array<const node*, flat_program::MAX_OPS> rev{};
			int rev_count = 0;
			for (const node* curr = n; curr; curr = curr->arg) {
				if (rev_count >= flat_program::MAX_OPS)
					return false;
				rev[rev_count++] = curr;
			}

			for (int i = rev_count - 1; i >= 0; --i) {
				if (!out.push(rev[i]->type, rev[i]->value))
					return false;
			}
			return true;
		}

		template <typename TFn>
		static inline bool with_flattened_unary_program(const node* tree, TFn&& fn) {
			flat_program program;
			if (flatten_unary_program(tree, program))
				return fn(program);
			return false;
		}

		static inline string stringify_flat_program(const flat_program& program) {
			string out;
			out.reserve(program.count * 16);
			for (int i = 0; i < program.count; ++i) {
				if (i > 0)
					out += ' ';
				out += type_name[to_underlying(program.ops[i].type)];
				append_annotation(program.ops[i].type, program.ops[i].value, out);
			}
			return out;
		}

		static inline string stringify_tree_via_flat(const node* tree) {
			string out;
			with_flattened_unary_program(tree, [&](const flat_program& flat) {
				out = stringify_flat_program(flat);
				return true;
			});
			return out;
		}

		// Single home of the puzzle-level upload: everything an op reads that
		// belongs to the whole puzzle rather than one example. Both upload paths
		// call this, so a new learned fact is added here once and cannot be
		// missed by the other path.
		// READ STRAIGHT OFF WHAT INFERENCE SAID. Every one of these used to be
		// copied onto the puzzle at load and read back from there, so a fact
		// reached the kernel only if somebody remembered to add its copy line —
		// and a forgotten one showed up as a zero, silently. There is nothing
		// left to forget: this is the only hop, and it is from the source.
		static inline void fill_puzzle_level(cuda_dsl_example_data& ex, const puzzle& p) {
			// Output colors ranked by total pixel count, descending.
			for (int i = 0; i < COLOR_COUNT; ++i) ex.output_colors_by_size[i] = p.inf.colors.output_colors_by_size[i];
			ex.output_colors_by_size_count = (int8_t)p.inf.colors.output_colors_by_size_count;

			// Learned shape -> color lookup for the ShapeColorMap op. ONLY WHEN THE
			// RULE STANDS: a detector that gave up part way through leaves the keys
			// it had already learned behind it, and `valid` is the whole of what
			// says whether they were ever meant to be used.
			const auto& learned = p.inf.shape_color_map;
			for (int i = 0; i < 16; ++i) {
				ex.shape_color_keys[i] = learned.holds() ? learned.keys[i] : 0;
				ex.shape_color_vals[i] = learned.holds() ? (int8_t)learned.colors[i] : (int8_t)0;
			}
			ex.shape_color_count = learned.holds() ? (int8_t)learned.count : (int8_t)0;

			// The canvas the INPUT figures sit on — what segmentation excludes.
			ex.background_color = p.inf.colors.background_color;

			// The canvas the output figures sit on. The kernel starts the canvas
			// in this color, so the skeleton's closing underlay costs no nodes.
			ex.output_background_color = p.inf.background.output_elected();

			ex.rule_type = (int8_t)p.inf.size.Chosen;
			ex.rule_params[0] = (int16_t)p.inf.size.width;
			ex.rule_params[1] = (int16_t)p.inf.size.height;
			ex.rule_params[2] = (int16_t)p.inf.size.width_numerator;
			ex.rule_params[3] = (int16_t)p.inf.size.width_denominator;
			ex.rule_params[4] = (int16_t)p.inf.size.height_numerator;
			ex.rule_params[5] = (int16_t)p.inf.size.height_denominator;
		}

		// Execute program on GPU (scratch path). Returns GPU score and solved flag.
		// Everything puzzle-level (size rule, palettes, learned tables) is read off
		// `p`, so a new learned fact needs no new parameter here.
		static bool execute_flat_program(
			const flat_program& program, const example& ex, grid& output,
			const puzzle& p, int node_count, float& out_score, bool& out_solved,
			gpu_score_terms* out_terms = nullptr);

		static inline bool execute_tree_via_flat(
			const node* tree, const example& ex, grid& output,
			const puzzle& p, int node_count, float& out_score, bool& out_solved,
			gpu_score_terms* out_terms = nullptr) {
			return with_flattened_unary_program(tree, [&](const flat_program& flat) {
				return execute_flat_program(flat, ex, output, p, node_count, out_score, out_solved, out_terms);
			});
		}

		// Execute + score in one GPU call. Returns {score, solved}.
		// GPU kernel already applies the node-count penalty.
		static inline pair<float, bool> score_tree_output(
			const node* tree, const example& ex, grid& output,
			const puzzle& p, int node_count, gpu_score_terms* out_terms = nullptr) {
			float s = 0; bool slv = false;
			execute_tree_via_flat(tree, ex, output, p, node_count, s, slv, out_terms);
			return { s, slv };
		}


		// Parse pipeline format: "A -> B[v] -> C" (deepest leaf first, root last).
		// All ops must be unary except the leaf (arity 0). Returns nullptr on failure.
		static inline node* parse_pipeline(const string& str) {
			vector<pair<node_type, int16_t>> tokens;
			size_t pos = 0;
			auto skip_ws = [&]() {
				while (pos < str.size() && isspace((unsigned char)str[pos]))
					pos++;
			};
			while (pos < str.size()) {
				skip_ws();
				if (pos >= str.size()) break;
				size_t name_start = pos;
				while (pos < str.size() && (isalnum((unsigned char)str[pos]) || str[pos] == '_'))
					pos++;
				if (pos == name_start) return nullptr;
				string name = str.substr(name_start, pos - name_start);
				int match = -1;
				for (int i = 0; i < (int)node_type::type_count; ++i) {
					string tn = type_name[i];
					if (tn.size() == name.size() &&
						equal(tn.begin(), tn.end(), name.begin(),
							[](char a, char b){ return toupper((unsigned char)a) == toupper((unsigned char)b); })) {
						match = i;
						break;
					}
				}
				if (match == -1) return nullptr;
				int16_t value = 0;
				if (pos < str.size() && str[pos] == '[') {
					pos++;
					size_t vs = pos;
					while (pos < str.size() && str[pos] != ']') pos++;
					int parsed = 0;
					try { parsed = stoi(str.substr(vs, pos - vs)); } catch (...) { parsed = 0; }
					value = (int16_t)parsed;
					if (pos < str.size() && str[pos] == ']') pos++;
				}
				tokens.push_back({(node_type)match, value});
			}
			if (tokens.empty()) return nullptr;
			node* curr = new node();
			curr->type = tokens[0].first;
			curr->value = tokens[0].second;
			update_hash(curr);
			for (size_t i = 1; i < tokens.size(); ++i) {
				node* parent = new node();
				parent->type = tokens[i].first;
				parent->value = tokens[i].second;
				parent->arg = curr;
				parent->update_depth();
				update_hash(parent);
				curr = parent;
			}
			return curr;
		}

		static inline node* parse(const string& str) {
			if (str.empty())
				return nullptr;
			// Programs are stored exclusively in the comma-pipeline format.
			return parse_pipeline(str);
		}

		[[nodiscard]] static inline node* copy(const node* n) {
			if (n)
				n->ref_count++;
			return const_cast<node*>(n);
		}

		// Build a fresh unary-chain tree from a flat op list. Mirrors the leaf->root
		// chaining in parse_pipeline: ops[0] is the leaf (deepest), ops[count-1] the
		// root. Used by the leave-one-out reducer to materialise a tree with one op
		// dropped, so it can be scored on the GPU like any other candidate. Returns
		// nullptr on empty input. Caller owns the returned tree (remove() to free).
		static inline node* build_chain_from_flat(const flat_program& flat) {
			if (flat.count <= 0)
				return nullptr;
			node* curr = new node();
			curr->type = flat.ops[0].type;
			curr->value = flat.ops[0].value;
			update_hash(curr);
			for (int i = 1; i < flat.count; ++i) {
				node* parent = new node();
				parent->type = flat.ops[i].type;
				parent->value = flat.ops[i].value;
				parent->arg = curr;
				parent->update_depth();
				update_hash(parent);
				curr = parent;
			}
			return curr;
		}
	};

}

// 17_dsl_executor.hpp is now included from 15_solver_beam.hpp after 16_fixed_dsl.hpp
