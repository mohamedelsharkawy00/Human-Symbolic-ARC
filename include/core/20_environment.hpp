#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// What a program's ops hand each other, and through it the op list.
#include "19_dsl.hpp"
// The one door to everything below the engine: infer(), and through it the
// core types (arc_color, grid, example, output_dimension_rule) and the limits.
#include "13_infer.hpp"

namespace arc {
	using namespace std;

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

	// node_type and the NODE_PRIMITIVES op list live in 14_vocabulary.hpp (single home).

	inline constexpr const char* type_name[] = {
	#define X(id, low, high, role, inferred, clears) #id,
		NODE_PRIMITIVES
	#undef X
	};

	// --- Shared node-type metadata helpers (used by serializer, GUI) ---

	// An op takes an argument when its range holds more than one value, or
	// when it says its choices in slots.
	inline bool is_annotated(node_type type) {
		return row_of(type).low != row_of(type).high || said_in_slots(type);
	}

	// The printed annotation IS the value -- a number, a partner op's choice by
	// name (value_written), or Tile's slots by name (tile_written) -- so reading
	// one back is reading what was written: no op translates its own argument,
	// so nothing has to be undone.
	inline void append_annotation(node_type type, int16_t value, const tile_value& tile, string& out) {
		if (is_annotated(type)) {
			out += " [";
			out += said_in_slots(type) ? tile_written(tile) : value_written(type, value);
			out += ']';
		}
	}

	struct node {
		// every op is unary, so a program is a chain; null = chain start
		node* arg = nullptr;
		uint64_t hash = 0;
		node_type type{};
		int16_t value = 0;
		// What a Tile says; every other op says its one number in `value`.
		tile_value tile;
	};

	struct puzzle {
		string id;
		vector<example> train, test;
		// EVERYTHING INFERENCE WORKED OUT, kept whole: the ops read their facts
		// straight off it, so no fact has a copy to go stale.
		inference inf;
		// PROGRAMS INFERENCE ALREADY SPELLED, most likely first. Bridged from
		// say_the_reading; empty when the reading had nothing sayable.
		vector<string> suggested_programs;
		node* best_tree = nullptr;
		// Has inference run on this puzzle yet. It is the expensive half of a
		// load and nothing reads it until the pane opens or the reading is
		// asked for, so it is done on demand and this says whether it has been.
		bool inferred = false;
		bool is_solved = false;
		bool is_test_solved = false;
	};

	// One op of a program, as it runs.
	struct flat_program_op {
		node_type type;
		int16_t value = 0;
		tile_value tile;
	};

	// A program in the order it runs, its first op first.
	using flat_program = vector<flat_program_op>;

	class environment {
	public:
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

		// A Tile's slots folded into one number for its node's hash.
		static inline uint64_t tile_hash(const tile_value& tile) {
			uint64_t h = 0;
			auto fold_number = [&](const tile_number& number) {
				h = hash_fold(h, (uint64_t)(uint8_t)number.count);
				for (const tile_number::term& term : number.terms) {
					h = hash_fold(h, (uint64_t)term.minus);
					h = hash_fold(h, (uint64_t)term.measure);
				}
			};
			for (const tile_value::step& step : tile.steps) {
				fold_number(step.down);
				fold_number(step.across);
			}
			for (const tile_bound& bound : { tile.start, tile.end }) {
				h = hash_fold(h, (uint64_t)bound.edge);
				h = hash_fold(h, (uint64_t)(uint8_t)bound.steps);
			}
			return hash_fold(h, (uint64_t)tile.spread);
		}

		static inline void update_hash(node* n) {
			if (!n)
				return;
			n->hash = node_hash((int)n->type, said_in_slots(n->type) ? tile_hash(n->tile) : (uint64_t)n->value,
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
			remove(n->arg);
			delete n;
		}

		// The chain from its leaf up, in the order it runs.
		static inline flat_program flatten_unary_program(const node* n) {
			flat_program out;
			for (; n; n = n->arg) out.push_back({ n->type, n->value, n->tile });
			reverse(out.begin(), out.end());
			return out;
		}

		static inline string stringify_flat_program(const flat_program& program) {
			string out;
			out.reserve(program.size() * 16);
			for (const flat_program_op& op : program) {
				if (!out.empty())
					out += ' ';
				out += type_name[(int)op.type];
				append_annotation(op.type, op.value, op.tile, out);
			}
			return out;
		}

		static inline string stringify_tree_via_flat(const node* tree) {
			return stringify_flat_program(flatten_unary_program(tree));
		}

		// WHAT THE PROGRAM `tree` DRAWS ON ONE EXAMPLE: its canvas once the last
		// op has painted, and into `held`, when asked, the list that op held.
		// Every fact an op reads is read off `p`, so a new learned fact needs no
		// new parameter here. 38_ops.hpp runs it.
		static grid draw(const node* tree, const example& ex, const puzzle& p, object_list* held = nullptr);

		// A program as stringify_flat_program writes it, "Input Recolor [3] Union",
		// its leaf first and its root last. Nothing when a word names no op or a
		// value cannot be read.
		static inline node* parse(const string& str) {
			vector<flat_program_op> tokens;
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
				flat_program_op token{ (node_type)match };
				const bool slots = said_in_slots(token.type);
				// One space parts the name from its brackets, as append_annotation
				// writes it: Recolor [3].
				if (str.compare(pos, 2, " [") == 0) {
					pos += 2;
					size_t vs = pos;
					while (pos < str.size() && str[pos] != ']') pos++;
					const string written = str.substr(vs, pos - vs);
					if (slots) {
						const optional<tile_value> tile = tile_read(written);
						if (!tile) return nullptr;
						token.tile = *tile;
					} else if (said_by_name(token.type)) {
						const optional<int> named = value_read(token.type, written);
						if (!named) return nullptr;
						token.value = (int16_t)*named;
					} else {
						int parsed = 0;
						try { parsed = stoi(written); } catch (...) { parsed = 0; }
						token.value = (int16_t)parsed;
					}
					if (pos < str.size() && str[pos] == ']') pos++;
				} else if (slots) {
					return nullptr;
				}
				tokens.push_back(token);
			}
			if (tokens.empty()) return nullptr;
			node* curr = new node();
			curr->type = tokens[0].type;
			curr->value = tokens[0].value;
			curr->tile = tokens[0].tile;
			update_hash(curr);
			for (size_t i = 1; i < tokens.size(); ++i) {
				node* parent = new node();
				parent->type = tokens[i].type;
				parent->value = tokens[i].value;
				parent->tile = tokens[i].tile;
				parent->arg = curr;
				update_hash(parent);
				curr = parent;
			}
			return curr;
		}
	};

}
