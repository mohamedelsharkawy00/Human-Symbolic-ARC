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
#include "cuda_dsl.cuh"
// The one door to everything below the engine: infer(), and through it the
// core types (arc_color, grid, example, output_dimension_rule) and the limits.
#include "inference/4_infer.hpp"

namespace arc {
	using namespace std;

	// Official C++23 std::to_underlying implementation (available in C++20 via this helper)
	template <typename Enum>
	[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept {
		return static_cast<std::underlying_type_t<Enum>>(e);
	}

	// --- Search Sentinels ---
	constexpr int NODE_COUNT_MAX = 1000000;
	constexpr int DISTANCE_MAX = 100;
	constexpr size_t BEAM_WIDTH = 1024;
	const int MAX_BATCH_SIZE = 5000;
	constexpr int MAX_SEARCH_DEPTH = 32;

	// --- Program Tree Constraints ---

	// Scoring weights live in cuda_dsl.cuh (GPU_SCORE_W_*, single home).
	// All scoring runs in the GPU kernel (dsl_exec_and_score_kernel_v2);
	// the CPU uses the weights only for the debug/analysis breakdown.

	// Max distinct (color,w,h,count) tuples we track per target example.
	constexpr int MAX_TARGET_OBJECTS = 16;

	// Values live in cuda_dsl.cuh (GPU_MOVE_DELTAS_INIT, single home).
	inline constexpr int move_deltas[] = GPU_MOVE_DELTAS_INIT;
	inline constexpr int move_variant_count = (int)size(move_deltas);

	// node_type and the NODE_PRIMITIVES op list live in cuda_dsl.cuh (single home).

	inline constexpr const char* type_name[] = {
	#define X(id, name, is_ann, var, role) name,
		NODE_PRIMITIVES
	#undef X
	};

	inline constexpr node_type primitive_types[] = {
	#define X(id, n, is_ann, var, role) node_type::id,
		NODE_PRIMITIVES
	#undef X
	};
	constexpr int primitive_count = (int)size(primitive_types);

	// --- Shared node-type metadata helpers (used by serializer, GUI, solver) ---

	inline constexpr bool type_is_annotated[] = {
	#define X(id, n, is_ann, var, role) (bool)is_ann,
		NODE_PRIMITIVES
	#undef X
	};

	inline constexpr int type_var_count[] = {
	#define X(id, n, is_ann, var, role) var,
		NODE_PRIMITIVES
	#undef X
	};

	inline bool is_annotated(node_type type) {
		return type_is_annotated[to_underlying(type)];
	}

	// Inverse of the printed annotation: maps the rendered annotation number
	// back to the raw node::value index. Slam/Move types print move_deltas[value]
	// instead of value directly, so parsing must invert that mapping.
	inline int16_t parse_annotation_value(node_type type, int parsed) {
		if (type == node_type::MoveRow || type == node_type::MoveColumn ||
			type == node_type::SlamRow || type == node_type::SlamColumn) {
			for (int i = 0; i < (int)size(move_deltas); ++i)
				if (move_deltas[i] == parsed) return (int16_t)i;
			return 0;
		}
		return (int16_t)parsed;
	}

	inline void append_annotation(node_type type, int16_t value, string& out) {
		if (is_annotated(type)) {
			out += '[';
			if (type == node_type::MoveRow || type == node_type::MoveColumn ||
				type == node_type::SlamRow || type == node_type::SlamColumn) {
				out += to_string((int)move_deltas[value]);
			} else {
				out += to_string((int)value);
			}
			out += ']';
		}
	}

	inline bool is_parameterized(node_type type) {
		return is_annotated(type);
	}

	inline int get_variant_count(node_type type) {
		return type_var_count[to_underlying(type)];
	}

	inline int16_t get_variant_value(node_type t, int idx) {
		if (t == node_type::ScaleUp || t == node_type::ScaleDown || t == node_type::Layout) return (int16_t)(idx + 2);
		if (t == node_type::ExtendOrthogonal || t == node_type::ExtendDiagonal) return (int16_t)(idx + 2);
		if (t == node_type::Skew) return (idx == 0) ? -1 : 1;
		if (t == node_type::DiagTile) return (int16_t)idx;
		if (t == node_type::TileX || t == node_type::TileY) return (int16_t)(idx + 1);
		if (t == node_type::InteriorColor) return (int16_t)(idx - 1); // -1..9
		return (int16_t)idx;
	}

	// Not thread-safe: all node alloc/free must happen on the solver worker thread,
	// or on the main thread while holding solver_state::mtx (which the worker also takes).
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
		// Non-atomic: only the worker thread mutates nodes. The main thread
		// reads node fields under solver_state::mtx (which the worker also
		// takes), so refcount changes are already serialized.
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
		bool prev_work = n->arg ? n->arg->work_unlanded : false;
		bool prev_picked = n->arg ? n->arg->something_picked : false;
		node_type prev_combine = n->arg ? n->arg->last_combine : node_type::type_count;
		node_type t = n->type;
		if (is_get_op(t)) {
			// A fresh source: nothing selected, but the new list owes a landing.
			n->sel_waiting = false; n->work_unlanded = true; n->last_combine = prev_combine;
			// A fetch hands back a brand new list, and new objects carry no marks.
			n->something_picked = false;
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
		// The chain starts from the input whether or not anyone says so, so a
		// null arg already HAS a source.
		bool has_get = true;
		// Rule 0: an op that does nothing without a pick is never worth offering
		// before one. It would only respell the chain it wraps, one op longer.
		// `SelectAll` is how a chain says "all of them", so nothing is lost.
		if (needs_a_pick(type) && !(arg && arg->something_picked)) return false;
		if (is_get_op(type)) {
			// Rule 1: first op, or right after a combine. A leading Input is
			// barred: it is a second spelling of the start every chain already
			// has, and two spellings of one program is fan for nothing.
			if (arg == nullptr) return type != node_type::Input;
			return is_combine_op(arg->type);
		}
		if (is_select_op(type)) return has_get;          // Rule 2
		if (is_act_op(type))    return has_get;          // Rule 3
		if (is_combine_op(type)) {                        // Rule 4
			// same-type-repeat barred; cross-type combine chains stay legal.
			bool same_repeat = arg && is_combine_op(arg->type) && arg->type == type;
			return has_get && !sel_waiting && !same_repeat;
		}
		return false;
	}

	// Rule 5: a program may STOP (be a complete solution) only when nothing is
	// waiting to be acted on and the last work has been landed.
	inline bool legal_stop(node* tail) {
		if (!tail) return false;                  // empty program is not a solution
		return !tail->sel_waiting && !tail->work_unlanded;
	}

	// Compact summary of one connected component in a grid: color, bbox size,
	// position, pixel count. This is all the CC-match score needs — no per-cell mask.
	struct object_summary {
		int8_t color = -1;
		int8_t w = 0;
		int8_t h = 0;
		int8_t x0 = 0;
		int8_t y0 = 0;
		int16_t count = 0;
	};

	// Segment a grid into per-color connected components (8-connectivity).
	// Each color is segmented independently so that touching different-color
	// regions don't merge into one component (which would happen with the
	// default "everything non-background" segmentation). Background = pixel
	// value < 0 (TRANSPARENT) or == 0 (puzzle convention for "empty").
	// Writes up to MAX_TARGET_OBJECTS summaries; extras are silently dropped.
	static inline int segment_objects_by_color_cpu(const grid& g, object_summary* out, int max_out) {
		if (g.width <= 0 || g.height <= 0) return 0;
		int n = 0;
		bool visited[MAX_GRID_DIM][MAX_GRID_DIM]{};
		// Static BFS queue sized to a full grid; small enough to stack-alloc.
		int16_t qr[MAX_GRID_DIM * MAX_GRID_DIM];
		int16_t qc[MAX_GRID_DIM * MAX_GRID_DIM];
		static constexpr int8_t dr[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
		static constexpr int8_t dc[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
		for (int r = 0; r < g.height; ++r) {
			for (int c = 0; c < g.width; ++c) {
				if (visited[r][c]) continue;
				int v = g.data[r][c];
				// Skip background: TRANSPARENT (-1) and 0 (puzzle "empty").
				if (v < 0 || v == 0) { visited[r][c] = true; continue; }
				int head = 0, tail = 0;
				qr[tail] = (int16_t)r; qc[tail] = (int16_t)c; tail++;
				visited[r][c] = true;
				int x0 = c, y0 = r, x1 = c, y1 = r;
				while (head < tail) {
					int cr = qr[head], cc = qc[head]; head++;
					if (cc < x0) x0 = cc;
					if (cr < y0) y0 = cr;
					if (cc > x1) x1 = cc;
					if (cr > y1) y1 = cr;
					for (int d = 0; d < 8; ++d) {
						int nr = cr + dr[d], nc = cc + dc[d];
						if (nr < 0 || nr >= g.height || nc < 0 || nc >= g.width) continue;
						if (visited[nr][nc]) continue;
						if (g.data[nr][nc] != v) continue;
						visited[nr][nc] = true;
						qr[tail] = (int16_t)nr; qc[tail] = (int16_t)nc; tail++;
					}
				}
				if (n < max_out) {
					out[n].color = (int8_t)v;
					out[n].w = (int8_t)(x1 - x0 + 1);
					out[n].h = (int8_t)(y1 - y0 + 1);
					out[n].x0 = (int8_t)x0;
					out[n].y0 = (int8_t)y0;
					out[n].count = (int16_t)tail;
					n++;
				}
			}
		}
		return n;
	}

	// Multiset match: for each attempt CC, find an unused target CC that matches
	// by (w,h,count); count it as shape match. If color also matches, count it
	// as color match too. Position score is the continuous bonus per shape-
	// matched pair, normalized to [0,1] by max_dim and summed across all matches.
	// Returns {shape_matches, color_matches, position_score}. O(N*M) with
	// N,M ≤ MAX_TARGET_OBJECTS — fine.
	//
	// Known limitation: shape match is exact on (w,h,count). A near-miss CC
	// (e.g. a 2x3 blob where the target is 3x3) earns 0 shape/color/position
	// reward and the score collapses back to the per-pixel terms. That's why
	// the pre-CC-term score function happily settled on outputs like puzzle
	// 09629e4f's wide-rectangle near-miss: right colors + right approximate
	// positions = high histogram + distance + rarity, while the structurally
	// correct shape was several pixel-edits away through a per-pixel-penalty
	// valley. Loosening this to a fuzzy shape match (e.g. IoU-style) would
	// extend the gradient to these near-misses but risks rewarding "right
	// colors in any blob" — out of scope here; the score reflects the strict
	// "same shape exists" semantics by design.
	struct cc_match_result {
		int shape_matches = 0;
		int color_matches = 0;
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
			bool best_color_match = false;
			for (int j = 0; j < tn; ++j) {
				if (used[j]) continue;
				if (a[i].w != t[j].w || a[i].h != t[j].h || a[i].count != t[j].count) continue;
				bool cm = (a[i].color == t[j].color);
				// Prefer a target CC that matches color too, so color credit is
				// granted whenever possible without depending on iteration order.
				if (best_j < 0 || (cm && !best_color_match)) {
					best_j = j;
					best_color_match = cm;
					if (cm) break;
				}
			}
			if (best_j >= 0) {
				used[best_j] = true;
				r.shape_matches++;
				if (best_color_match) r.color_matches++;
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

	struct score_breakdown {
		int mismatches = 0;
		float exact_term = 0.0f;
		float histogram_error = 0.0f;
		float histogram_term = 0.0f;
		float distance_error = 0.0f;
		float distance_term = 0.0f;
		float rarity_score = 0.0f;
		float rarity_term = 0.0f;
		float node_penalty = 0.0f;
		float dimension_penalty = 0.0f;
		// CC-match: independent shape, color, and position rewards.
		// shape_matches: attempt CCs whose (w,h,count) matches some unused target.
		// color_matches: same plus matching color.
		// position_score: sum of per-pair continuous (1 - chebyshev/max_dim) values
		// over the same matched pairs, in [0, shape_matches].
		int shape_matches = 0;
		int color_matches = 0;
		float position_score = 0.0f;
		float shape_term = 0.0f;
		float color_term = 0.0f;
		float position_term = 0.0f;
		float total = 0.0f;
		bool dims_match = false;
		bool solved = false;
	};

	struct puzzle {
		string id;
		vector<example> train, test;
		output_dimension_rule rule;
		bool invariant_colors[10] = {}; // one bit: the elected input background
		// Allowed colors for color-LITERAL ops (AddColor/Recolor/RemoveColor/
		// InteriorColor): the union of every color in any train input or output.
		// Bridged from inference (inf.colors.allowed_color). Used to pin the literal
		// fan-out in candidate generation. allowed_colors_count==0 means "not set
		// / don't pin" (e.g. empty puzzle) so generation falls back to all 10.
		bool allowed_colors[10] = {};
		int  allowed_colors_count = 0;
		// Ops inference declared dead for this puzzle: their backing fact is
		// absent, so they can only no-op. Verdict filled once, right after
		// infer(); generation never offers a dead op.
		bool dead_ops[(int)node_type::type_count] = {};
		// Learned color swaps, by BidiColorMap's variant index k in [0,44].
		// Bridged from inference (inf.color_swaps). Pins the op's 45-way fan-out
		// in candidate generation; count==0 means "not learned / don't pin".
		bool color_swap_variants[45] = {};
		int  color_swap_variants_count = 0;
		int8_t output_colors[10] = {};   // ordered: union of colors across train outputs, first-appearance row-major across train pairs
		int output_colors_count = 0;
		int8_t output_colors_by_size[10] = {};  // train-output regions sorted by pixel count desc, colors in that order; consistent across train (else count=0)
		int output_colors_by_size_count = 0;
		// Learned shape->color lookup for the ShapeColorMap op: hash of a small
		// glyph's occupancy -> color the main object is repainted to. Bridged from
		// inference (inf.shape_color_map); count==0 means "not learned" -> op no-ops.
		uint32_t shape_color_keys[16] = {};
		int8_t   shape_color_vals[16] = {};
		int      shape_color_count = 0;
		// Per-train-pair map of cells the input already gets right (input ==
		// output). Bridged from inference (inf.unchanged.kept, Identity puzzles
		// only; empty otherwise). Layer-lock commit uses it: wrong paint on a
		// kept cell is vandalism (nothing repaints it), wrong paint on a
		// changed cell is fine (a later layer overwrites it anyway).
		vector<array<bool, MAX_GRID_DIM * MAX_GRID_DIM>> kept_cells;
		// Per-train-example connected-component summaries of target output.
		// Computed once at puzzle-load by segment_objects_by_color_cpu. Used
		// by the CC-match score term to reward partial structural progress.
		// Sized to train.size(); inner array padded to MAX_TARGET_OBJECTS for
		// cheap upload to GPU template alongside count.
		struct target_obj_slot {
			object_summary objs[MAX_TARGET_OBJECTS];
			int count = 0;
		};
		vector<target_obj_slot> target_objects;
		// Background election readout, for GUI display / verification. Plain
		// data (no dependency on the inference family): puzzle_loader copies the four
		// per-direction elections off the inference object here. See
		// background_hypotheses (inference/2_background.hpp) for the source of truth.
		struct background_readout {
			struct cand { int8_t color = -1; int embeds = 0, dominance = 0; bool persists = false; int score = 0, span = 0, gaps = 0; bool underneath = false; };
			// one ranked list per direction; [0] is that direction's winner
			vector<cand> across_inputs, across_outputs, across_inputs_outputs;
			vector<int8_t> per_pair;     // one elected color per train pair
			int8_t input_elected = -1;   // the input-side canvas verdict
			int8_t output_elected = -1;  // the output-side canvas verdict
			bool per_pair_mode = false;  // global abstains but each pair has its own bg
			bool input_vetoed = false;   // a confident pair decisively contradicts the input winner
		};
		background_readout bg;
		// Object correspondence readout, for GUI display / verification. Same
		// deal as background_readout: plain data, filled by puzzle_loader off
		// correspondence_facts (inference/3_correspondence.hpp), one entry per edge.
		// Centers are in HALF CELLS (2*x0 + w) so an even-sized object centers
		// BETWEEN two cells instead of being nudged off by one.
		struct correspondence_readout {
			struct line {
				int8_t pair = 0;
				// Center of the piece, and of where it landed
				int16_t from_col = 0, from_row = 0, to_col = 0, to_row = 0;
				// Which reading of the split this line belongs to, as a place
				// in `readings`. 'C' walks these.
				int8_t rank = 0;
				// The two boxes the line joins, in cells: where the piece sits
				// and where it landed. Drawn one pixel thin.
				int8_t from_x = 0, from_y = 0, from_w = 0, from_h = 0;
				int8_t to_x = 0, to_y = 0, to_w = 0, to_h = 0;
				// What the line CLAIMS, which the drawing cannot say: found as
				// itself, found wearing another color, or not found at all and
				// only sitting on what it overlaps most.
				enum : int8_t { exact, recolored, sits_on };
				int8_t kind = exact;
				// Which of the eight ways round it sat, how many of its cells
				// another piece covers, its own color (-1 = several), and how
				// many cells run off the grid. ORDER IS LOAD-BEARING: these are
				// filled by a positional aggregate initialiser in puzzle_loader.
				// hidden/outside are wider than a byte because a piece can reach
				// 900 cells, and a byte would show a wrapped number.
				int8_t pose = 0;
				int16_t hidden = 0;
				int8_t color = -1;
				int16_t outside = 0;
				// WHICH GRID THIS LINE STARTS ON. A reading holds edges read
				// both ways round, so the side belongs to the line and not to
				// the reading it came from.
				bool from_input = true;
			};
			vector<line> lines;
			// One entry per reading of the split, IN THE ORDER THE TABLE IS
			// DRAWN: one row per cut, one column per freedom, so a reading's
			// place here IS its cell. What it cut on, what it cost and what it
			// threw away is why one reading leads and another trails.
			struct reading {
				// WHERE THIS READING SITS IN THE TABLE: which cut down the
					// side, which freedoms across the top. Two bytes, not two
					// copied sentences — the names have one home below.
					int8_t cut_kind = 0, match_kind = 0;
				int pieces = 0, left_over = 0, cut = 0, box_cells = 0;
				// EVERY piece the reading cuts out, both sides, all pairs. The
				// lines can only ever reach pieces that HAVE an edge, so the
				// ones that went nowhere or came from nothing are invisible
				// without this — and those are the interesting ones.
				struct piece_row {
					int8_t pair = 0;
					bool from_input = true;
					int8_t color = -1, x = 0, y = 0, w = 0, h = 0;
					int16_t cells = 0;
					// How many edges it has, read from its own side
					int16_t edges = 0;
					// The first line it got, an index into lines; -1 = nothing
					// answers it, which is the whole reason this list exists.
					int16_t first_line = -1;
					// SOMETHING IS ITS PARTNER — see object_info::has_partner. Not read
					// off first_line: that is credited by overlapping BOXES, and
					// a ring's box holds its own hole.
					bool has_partner = false;
					// ANOTHER OBJECT ON THIS GRID HAS THE SAME SHAPE. Copies of
					// one thing, not several things that resemble each other —
					// which is the difference between a motif that repeats and
					// a grid full of coincidences.
					// WHICH SET OF COPIES, -1 for none. A grid can hold several
					// motifs at once and one color cannot say which box belongs
					// with which, so the set is a number and the pane steps it.
					int16_t motif = -1;
					// THE MEDOID of that set — the member that disagrees least
					// with all the others, elected in find_copies where the cells
					// still exist. See object_info::medoid.
					bool medoid = false;
					// THE BLANK IT FILLED, or the blank it left — the patch of
					// background on the OTHER grid this piece covers, asked only
					// where nothing answers it. Closed in by colored cells it is a hole.
					// blank_w 0 = it covers no blank. See object_info's blanks.
					int8_t blank_x = 0, blank_y = 0, blank_w = 0, blank_h = 0;
					int16_t blank_cells = 0;
					bool blank_enclosed = false;
				};
				vector<piece_row> pieces_seen;
				// EVERY match found and thrown away, with where it would have
				// landed and why it went. A missing line is only readable next
				// to the reason it is missing.
				struct dropped_row {
					int8_t pair = 0;
					int16_t piece = 0;
					// line::kind, and lost to chance / explained nothing new /
					// not in the fitting
					int8_t kind = 0, why = 0;
					// Covered cells — wider than a byte, same reason as line
					int16_t hidden = 0;
					// Where the piece sits and where it WANTED to land, the same
					// two boxes a line joins, so a refusal draws like a landing.
					int8_t from_x = 0, from_y = 0, from_w = 0, from_h = 0;
					int8_t to_x = 0, to_y = 0, to_w = 0, to_h = 0;
						// Which grid the refused piece sits on — same reason as
						// line::from_input: both readings are kept, so the side
						// belongs to the row and not to the reading.
						bool from_input = true;
				};
				vector<dropped_row> dropped_seen;
				// EVERY CELL THE READING COULD NOT ACCOUNT FOR, named by the
				// grid it sits on. The pane's left-over number is this list's
				// length, so the number and what you can point at cannot drift.
				struct left_over_row {
					int8_t pair = 0;
					bool from_input = true;
					int8_t x = 0, y = 0;
				};
				vector<left_over_row> left_over_seen;
				struct pair_read {
					// How the forward pass decided (correspondence_facts::how)
					int8_t how = 0;
					// Matches found and thrown away, and why
					int dropped = 0, chance = 0, stale = 0, did_not_fit = 0;
					// Output cells no edge of any kind accounts for
					int left_over = 0;
				};
				vector<pair_read> pairs;
			};
			vector<reading> readings;
			// WHICH ONE INFERENCE CHOSE, a place in `readings`. The list is in
			// table order, so the winner can sit anywhere in it and being first
			// no longer says so.
			int8_t chosen = 0;
			// Non-background cells across both grids of every train pair. What a
			// table cell is a share OF — a bare count of what a reading could not
			// account for says nothing without the size of the thing it failed at.
			int cells = 0;
			// The cut names live once, beside the cuts themselves —
			// arc::split_names in inference/3_correspondence.hpp.
		};
		correspondence_readout correspondence;
		// Everything else infer() read, for the GUI pane. Same plain-data deal:
		// only the bits not already spelled out above as fields the engine uses.
		// One bit per color in the masks.
		struct facts_readout {
			uint16_t consumed = 0, introduced = 0, preserved = 0;
			uint16_t every_input = 0, every_output = 0, every_grid = 0, any_grid = 0;
			// Share of cells the input already gets right, worst pair
			bool unchanged_valid = false;
			float min_kept = 0.0f;
			// Cells each color won and lost across all pairs
			bool diff_valid = false;
			int gained[10] = {}, lost[10] = {};
			// What correspondence made of each pair
			struct pair_counts {
				int objects = 0, out_objects = 0;
				// Where each input piece went, where each output landing came
				// from: nowhere/nothing, one, several. Same names as
				// correspondence_facts::pair_info, straight across.
				int deleted = 0, persisted = 0, copied = 0;
				int invented = 0, from_one = 0, from_many = 0;
				// Most places one piece went, most pieces met at one landing
				int most_landings = 0, most_sources = 0;
				// Non-background output cells an exact edge does account for:
				// what the output CARRIED OVER rather than made.
				int carried_cells = 0;
				// Cells no EXACT edge accounts for, so cells the output MADE
				// rather than carried over. Not the same as a reading's left
				// over, which counts what no edge of ANY kind accounts for —
				// that one belongs to the reading and lives with the lines.
				int created_cells = 0;
			};
			vector<pair_counts> pairs;
		};
		facts_readout facts;
		node* best_tree = nullptr;
		int best_node_count = NODE_COUNT_MAX;
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

		// Segment each train-example's target output into per-color connected
		// components and cache the summaries on the puzzle. Used by the
		// CC-match score term.
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
		static inline void detect_target_objects(puzzle& p) {
			p.target_objects.assign(p.train.size(), puzzle::target_obj_slot{});
			for (size_t i = 0; i < p.train.size(); ++i) {
				auto& slot = p.target_objects[i];
				slot.count = segment_objects_by_color_cpu(
					p.train[i].output, slot.objs, MAX_TARGET_OBJECTS);
			}
		}

		// Detect invariant colors: present in ALL training inputs.
		// MOVED to inference/4_infer.hpp (arc::detect_color_presence).
		// Reading the always-present (across_inputs) colors is inference's job. The
		// engine only needs the single dominant background color for
		// segmentation (Colors skips it, Split erases it); puzzle_loader bridges
		// that from inf.colors.background_color
		// into p.invariant_colors. The inference version is CORRECTED to keep the
		// full invariant-color set (the engine version kept only the most-frequent).

		// OutputColorsBySize: train output colors ranked by total pixel count (desc),
		// summed across all train outputs. Verified consistent: every train output must
		// produce the same color sequence under by-pixel-count desc. If inconsistent,
		// count = 0 (op no-ops). Includes 0 if it appears in outputs.
		// NOTE: this counts pixels per color (not per connected region). For 08ed6ac7
		// every color = exactly one connected region so the two coincide. For puzzles
		// where one color has multiple disjoint regions of different sizes, a connected-
		// components-aware variant is needed (see notes/2_dsl/learned_orderings.txt).
		// detect_output_colors_by_size MOVED to inference/4_infer.hpp
		// (arc::detect_output_colors_by_size). Ranking across-outputs colors by
		// pixel count is inference's job; puzzle_loader bridges
		// inf.colors.output_colors_by_size / _count onto the puzzle for the
		// OutputColorsBySize op. (The inference version reads the background color
		// directly to skip the background, same value as the old p.invariant_colors.)

		// detect_output_colors MOVED to inference/4_infer.hpp (arc::detect_output_colors).
		// Reading the across_outputs palette is inference's job. Nothing in the
		// engine consumes the ordered palette since the InferredOutputColors op
		// was deleted; the by-size ranking is the surviving consumer.

		// detect_size_rule MOVED to inference/4_infer.hpp (arc::output_dimension_rule::infer_rule).
		// Reading a rule from the examples is inference's job; the output_dimension_rule TYPE
		// stays here because the DSL/GPU path sizes output grids from it. The
		// engine no longer computes it directly — puzzle_loader calls arc::infer(p),
		// which returns an `inference` object; its output_dimension_rule is copied to p.rule.
		// See inference/4_infer.hpp for the verified-foundation notes (1,147-puzzle corpus).

		static inline void update_hash(node* n) {
			if (!n)
				return;
			n->update_depth();
			set_generation_debt(n);  // debt is finalized here, the one node-finalize chokepoint
			uint64_t h = 0x9e3779b97f4a7c15ULL + to_underlying(n->type);
			h ^= (uint64_t)n->value + 0x9e3779b9 + (h << 6) + (h >> 2);
			if (n->arg)
				h ^= n->arg->hash + 0x9e3779b9 + (h << 6) + (h >> 2);
			n->hash = h;
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
		static inline void fill_puzzle_level(cuda_dsl_example_data& ex, const puzzle& p) {
			// Output colors ranked by total pixel count, descending.
			for (int i = 0; i < 10; ++i) ex.output_colors_by_size[i] = p.output_colors_by_size[i];
			ex.output_colors_by_size_count = (int8_t)p.output_colors_by_size_count;

			// Learned shape -> color lookup for the ShapeColorMap op.
			for (int i = 0; i < 16; ++i) {
				ex.shape_color_keys[i] = p.shape_color_keys[i];
				ex.shape_color_vals[i] = (int8_t)p.shape_color_vals[i];
			}
			ex.shape_color_count = (int8_t)p.shape_color_count;

			// invariant_colors holds exactly the dominant background color
			// (see puzzle_loader CAVEAT), so its one set bit IS the background.
			ex.background_color = -1;
			for (int i = 0; i < 10; ++i)
				if (p.invariant_colors[i]) { ex.background_color = (int8_t)i; break; }

			// The canvas the output figures sit on. The kernel starts the canvas
			// in this color, so the skeleton's closing underlay costs no nodes.
			ex.output_background_color = p.bg.output_elected;

			ex.rule_type = (int8_t)p.rule.Chosen;
			ex.rule_params[0] = (int16_t)p.rule.width;
			ex.rule_params[1] = (int16_t)p.rule.height;
			ex.rule_params[2] = (int16_t)p.rule.width_numerator;
			ex.rule_params[3] = (int16_t)p.rule.width_denominator;
			ex.rule_params[4] = (int16_t)p.rule.height_numerator;
			ex.rule_params[5] = (int16_t)p.rule.height_denominator;
		}

		// Execute program on GPU (scratch path). Returns GPU score and solved flag.
		// Everything puzzle-level (size rule, palettes, learned tables) is read off
		// `p`, so a new learned fact needs no new parameter here.
		static bool execute_flat_program(
			const flat_program& program, const example& ex, grid& output,
			const puzzle& p, int node_count, float& out_score, bool& out_solved);

		static inline bool execute_tree_via_flat(
			const node* tree, const example& ex, grid& output,
			const puzzle& p, int node_count, float& out_score, bool& out_solved) {
			return with_flattened_unary_program(tree, [&](const flat_program& flat) {
				return execute_flat_program(flat, ex, output, p, node_count, out_score, out_solved);
			});
		}

		// Execute + score in one GPU call. Returns {score, solved}.
		// GPU kernel already applies the node-count penalty.
		static inline pair<float, bool> score_tree_output(
			const node* tree, const example& ex, grid& output,
			const puzzle& p, int node_count) {
			float s = 0; bool slv = false;
			execute_tree_via_flat(tree, ex, output, p, node_count, s, slv);
			return { s, slv };
		}

		static inline score_breakdown analyze_output_score(
			const grid& output, const example& ex, int node_count,
			const object_summary* target_objs = nullptr, int target_obj_count = 0) {
			score_breakdown b;
			b.dims_match = (output.width == ex.output.width && output.height == ex.output.height);

			int target_hist[10] = {};
			int attempt_hist[10] = {};
			int match_by_color[10] = {};

			const int tw = ex.output.width;
			const int th = ex.output.height;
			const int total_target_pixels = tw * th;

			float exact_error = 0.0f;
			for (int r = 0; r < th; ++r) {
				for (int c = 0; c < tw; ++c) {
					int t = ex.output.data[r][c];
					int a = (r < output.height && c < output.width) ? output.data[r][c] : TRANSPARENT;
					if (t != a) {
						b.mismatches++;
						if (a != TRANSPARENT) exact_error += GPU_SCORE_W_EXACT_MATCH;
					} else if (t >= 0 && t < 10) {
						match_by_color[t]++;
					}
					if (t >= 0 && t < 10) target_hist[t]++;
					if (a >= 0 && a < 10) {
						attempt_hist[a]++;
						b.distance_error += (float)ex.distance_maps[a][r][c];
					}
				}
			}

			for (int i = 0; i < 10; ++i) {
				if (target_hist[i] > 0)
					b.rarity_score += match_by_color[i] * ((float)total_target_pixels / target_hist[i]);
				int d = target_hist[i] - attempt_hist[i];
				b.histogram_error += (float)abs(d);
			}

			b.exact_term = -exact_error;
			b.histogram_term = -b.histogram_error * GPU_SCORE_W_HISTOGRAM;
			b.distance_term = -b.distance_error * GPU_SCORE_W_DISTANCE_MAP;
			b.rarity_term = b.rarity_score * GPU_SCORE_W_RARITY_MATCH;
			b.node_penalty = node_count * GPU_SCORE_W_NODE_PENALTY;
			b.dimension_penalty = (float)(abs(ex.output.width - output.width) + abs(ex.output.height - output.height));

			// CC-match terms: segment the attempt grid the same way the target
			// was segmented at puzzle-load, then multiset-match. Skipped when
			// the caller didn't supply target summaries (legacy paths).
			if (target_objs && target_obj_count > 0) {
				object_summary attempt_objs[MAX_TARGET_OBJECTS];
				int an = segment_objects_by_color_cpu(output, attempt_objs, MAX_TARGET_OBJECTS);
				int max_dim = max(ex.output.width, ex.output.height);
				auto m = match_object_summaries(attempt_objs, an, target_objs, target_obj_count, max_dim);
				b.shape_matches = m.shape_matches;
				b.color_matches = m.color_matches;
				b.position_score = m.position_score;
				b.shape_term = m.shape_matches * GPU_SCORE_W_SHAPE_MATCH;
				b.color_term = m.color_matches * GPU_SCORE_W_COLOR_MATCH;
				b.position_term = m.position_score * GPU_SCORE_W_POSITION_MATCH;
			}

			b.total = b.exact_term
				+ b.histogram_term
				+ b.distance_term
				+ b.rarity_term
				+ b.shape_term
				+ b.color_term
				+ b.position_term
				- b.node_penalty
				- b.dimension_penalty;
			b.solved = b.dims_match && (b.mismatches == 0);
			return b;
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
					value = parse_annotation_value((node_type)match, parsed);
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

// dsl_executor.hpp is now included from solver_beam.hpp after fixed_dsl.hpp
