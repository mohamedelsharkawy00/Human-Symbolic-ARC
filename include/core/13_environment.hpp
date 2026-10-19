#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Single home of the DSL op list (NODE_PRIMITIVES) and the node_type enum.
#include "12_dsl.cuh"
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

	// Squashing many numbers into one, so a program can be named by a single
	// number. Start at 0, fold once per value, and finish with mix64 -- a raw
	// fold spreads its bits badly, and two programs landing on one number would
	// be taken for one.
	inline uint64_t hash_fold(uint64_t seed, uint64_t value) {
		return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
	}

	// splitmix64 finalizer: about two cycles, and every bit ends up doing work.
	inline uint64_t mix64(uint64_t x) {
		x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
		x ^= x >> 27; x *= 0x94d049bb133111ebULL;
		x ^= x >> 31; return x;
	}

	constexpr int DISTANCE_MAX = 100;
	// The worst a score can be, so anything untried sits below every real one.
	constexpr float SCORE_MIN = -1000000.0f;

	// Max distinct (color,w,h,count) tuples we track per target example.
	constexpr int MAX_TARGET_OBJECTS = 16;

	// node_type and the NODE_PRIMITIVES op list live in 8_vocabulary.hpp (single home).

	inline constexpr const char* type_name[] = {
	#define X(id, low, high, role, inferred, clears) #id,
		NODE_PRIMITIVES
	#undef X
	};

	// --- Shared node-type metadata helpers (used by serializer, GUI) ---

	// An op takes an argument only when its range holds more than one value.
	inline bool is_annotated(node_type type) {
		return row_of(type).low != row_of(type).high;
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

	struct node {
		// Non-atomic: a tree shared across threads is only touched under the
		// lock its owner holds (search_progress::mtx).
		int ref_count = 1;
		// every op is unary, so a program is a chain; null = chain start
		node* arg = nullptr;
		uint64_t hash = 0;
		node_type type{};
		int16_t value = 0;
	};

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
		// PROGRAMS INFERENCE ALREADY SPELLED, most likely first. Bridged from
		// say_the_reading; empty when the reading had nothing sayable.
		vector<string> suggested_programs;
		node* best_tree = nullptr;
		// WHAT best_tree SCORES ON EACH PAIR, indexed like train and test. It
		// describes best_tree, not the pair, which is why it lives here beside
		// it: replace the tree and every one of these is stale together.
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
		// load and nothing reads it until the pane opens or the reading is
		// asked for, so it is done on demand and this says whether it has been.
		bool inferred = false;
		bool is_solved = false;
		bool is_test_solved = false;
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

	// TOO LONG TO RUN, SAID OUT LOUD. flatten_unary_program refuses a chain that
	// outgrows one batch slot and returns false, and every caller reads a false as
	// "it drew nothing" -- which is also exactly what a program that paints nothing
	// looks like. So the two paths that hand a WRITTEN program to the engine ask
	// here first and name the refusal, rather than showing an empty grid and
	// letting a reader decide the program was wrong.
	inline bool too_long_to_run(int op_count) { return op_count > flat_program::MAX_OPS; }

	// The same-color groups of a grid as their summaries alone, in the order the
	// walk finds them. Writes at most max_out and returns how many; a grid with more
	// groups than that silently loses the rest, so callers size the array for the
	// worst case they care about.
	inline int find_objects_by_color(const grid& one_grid, int background,
	                                 object_summary* out, int max_out) {
		int found = 0;
		object_walk_scratch scratch;
		each_object(one_grid, background, grouping::touching_same_color, scratch, [&](const object_summary& one,
		                                   const int16_t*, int) {
			if (found < max_out) out[found++] = one;
		});
		return found;
	}

	class environment {
	public:
		// THE SCORER'S TARGETS: one output segmented into per-color connected
		// components on the background inference elected, written the way the
		// kernel reads them. Used by the CC-match score term to reward partial
		// structural progress. Returns how many.
		//
		// WORKED OUT WHERE THEY ARE UPLOADED, and never kept on the puzzle: a
		// kept copy was read against whatever background stood when it was made,
		// which before inference is none.
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
			int count = find_objects_by_color(output, p.inf.ground_election.output_elected(),
			                                  found, MAX_TARGET_OBJECTS);
			if (count > GPU_MAX_TARGET_OBJECTS) count = GPU_MAX_TARGET_OBJECTS;
			for (int k = 0; k < count; ++k) {
				dst[k].w = found[k].width;
				dst[k].h = found[k].height;
				dst[k].x0 = found[k].col;
				dst[k].y0 = found[k].row;
				dst[k].count = found[k].count;
			}
			for (int k = count; k < GPU_MAX_TARGET_OBJECTS; ++k) dst[k] = {};
			return (int8_t)count;
		}

		// What one node's hash is, from its parts alone: the program up to and
		// including it, as one number. The GUI names the op a person hovers by
		// it, and two programs landing on one number would be taken for one, so
		// the weak fold is finalized before it leaves.
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
		// belongs to the whole puzzle rather than one example. A new learned
		// fact is added here once.
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
			// it had already learned behind it, and holds() is the whole of what
			// says whether they were ever meant to be used.
			const auto& learned = p.inf.shape_color_map;
			ex.shape_colors = learned.holds() ? static_cast<const shape_table&>(learned) : shape_table{};


			ex.rule_type = (int8_t)p.inf.size.Chosen;
			ex.rule_params[0] = (int16_t)p.inf.size.width;
			ex.rule_params[1] = (int16_t)p.inf.size.height;
			ex.rule_params[2] = (int16_t)p.inf.size.width_numerator;
			ex.rule_params[3] = (int16_t)p.inf.size.width_denominator;
			ex.rule_params[4] = (int16_t)p.inf.size.height_numerator;
			ex.rule_params[5] = (int16_t)p.inf.size.height_denominator;
		}

		// Execute program on GPU. Returns GPU score and solved flag.
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
	};

}

// 18_dsl_executor.hpp is included from 16_solver.hpp after 17_fixed_dsl.hpp
