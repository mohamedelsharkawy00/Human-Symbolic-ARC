// This file reads the examples like a human.
//
// Three kinds of answer:
//   FACTS      observations computed from the examples.
//   RULES      claims checked against the examples.
//   ELECTIONS  ranked interpretations, supplied by upstream detectors.
//
// Each answer owns the detector that fills it (color_facts::detect,
// piece_slam_rule::detect). Shared drawing helpers support those detectors;
// infer() asks them all in order.
#pragma once
#include "1_readings.hpp"
#include "2_size.hpp"
#include "3_background.hpp"
#include "6_correspondence.hpp"
#include "7_vocabulary.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <numeric>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// Tracks which colors appear, disappear, or persist across training examples.
// A bool[COLOR_COUNT] is a set of colors: true means the color is present.
struct color_facts {
    // Color · across_input_output
    // in input, gone in output
    bool consumed[COLOR_COUNT]   = {};
    // Absent in input, present in output
    bool introduced[COLOR_COUNT] = {};
    // In both
    bool preserved[COLOR_COUNT]  = {};

    // Color · across_inputs
    bool in_every_input[COLOR_COUNT] = {};
    bool in_any_input[COLOR_COUNT]   = {};

    // Color · across_outputs
    bool in_every_output[COLOR_COUNT] = {};
    bool in_any_output[COLOR_COUNT]   = {};
    // First-appearance order, kept for the log readout; no op consumes it now
    int8_t output_colors[COLOR_COUNT] = {};
    int    output_colors_count = 0;
    // Set only if all outputs rank identically
    int8_t output_colors_by_size[COLOR_COUNT] = {};
    int    output_colors_by_size_count = 0;

    // Color · across_inputs_outputs
    // in every train grid, inputs and outputs alike
    bool in_every_grid[COLOR_COUNT] = {};
    // In at least one train grid
    bool in_any_grid[COLOR_COUNT]   = {};

    // in_any_input and in_any_output are safe for LITERAL pinning only, and one
    // set per side — never a global whitelist (lethal, measured).

    // Collects all basic color facts that can be learned without knowing the background yet.
    static color_facts detect(const vector<example>& train) {
        color_facts facts;
        facts.detect_presence(train);
        facts.detect_output_colors(train);
        return facts;
    }

    // Records colors from most-used to least-used only when every output agrees
    // on the ranking, excluding the supplied background.
    void detect_output_colors_by_size(const vector<example>& train, int8_t background) {
        output_colors_by_size_count = 0;
        if (train.empty()) return;

        // Biggest first; returns how many colors this output actually has.
        auto rank_for_output = [&](const grid& out, int8_t ranking[COLOR_COUNT]) -> int {
            int counts[COLOR_COUNT];
            count_colors(out, counts, background);
            // Only the first `present` entries are ever read, and each is written
            // before it is, so there is nothing to start them at.
            int order[COLOR_COUNT] = {}, present = 0;
            for (int color = 0; color < COLOR_COUNT; ++color) {
                if (counts[color] <= 0) continue;
                order[present++] = color;
            }
            for (int i = 1; i < present; ++i)
                for (int j = i; j > 0 && counts[order[j]] > counts[order[j-1]]; --j)
                    swap(order[j], order[j-1]);
            for (int i = 0; i < present; ++i) ranking[i] = (int8_t)order[i];
            return present;
        };

        int8_t first_ranking[COLOR_COUNT] = {};
        int first_count = rank_for_output(train[0].output, first_ranking);
        for (size_t i = 1; i < train.size(); ++i) {
            int8_t ranking[COLOR_COUNT] = {};
            int count = rank_for_output(train[i].output, ranking);
            if (count != first_count) return;
            for (int rank = 0; rank < count; ++rank)
                if (ranking[rank] != first_ranking[rank]) return;
        }
        for (int rank = 0; rank < first_count; ++rank)
            output_colors_by_size[rank] = first_ranking[rank];
        output_colors_by_size_count = first_count;
    }

    // Formats a color set as comma-separated digits for display and logging, or "-" if empty.
    static string as_text(const bool member[COLOR_COUNT]) {
        string listed;
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (member[color]) {
                if (!listed.empty()) listed += ',';
                listed += (char)('0' + color);
            }
        return listed.empty() ? string("-") : listed;
    }

private:
    // One pass over the train pairs fills every presence set.
    void detect_presence(const vector<example>& train) {
        if (train.empty()) return;

        for (int i = 0; i < COLOR_COUNT; ++i) {
            in_every_input[i]  = true;
            in_every_output[i] = true;
            in_every_grid[i]   = true;
            consumed[i] = true; introduced[i] = true; preserved[i] = true;
        }

        for (const auto& ex : train) {
            bool in[COLOR_COUNT], out[COLOR_COUNT];
            present_colors(ex.input, in);
            present_colors(ex.output, out);
            for (int i = 0; i < COLOR_COUNT; ++i) {
                in_every_input[i]  = in_every_input[i]  && in[i];
                in_any_input[i]    = in_any_input[i]    || in[i];
                in_every_output[i] = in_every_output[i] && out[i];
                in_any_output[i]   = in_any_output[i]   || out[i];
                in_every_grid[i]   = in_every_grid[i]   && in[i] && out[i];
                in_any_grid[i]     = in_any_grid[i]     || in[i] || out[i];
                consumed[i]   = consumed[i]   && (in[i] && !out[i]);
                introduced[i] = introduced[i] && (!in[i] && out[i]);
                preserved[i]  = preserved[i]  && (in[i] && out[i]);
            }
        }

    }

    // Records the output color palette in order of first appearance for logs and the GUI.
    void detect_output_colors(const vector<example>& train) {
        bool seen[COLOR_COUNT] = {};
        output_colors_count = 0;
        for (const auto& ex : train)
            walk_colored_cells(ex.output, TRANSPARENT, [&](int, int, int color) {
                if (seen[color]) return;
                seen[color] = true;
                output_colors[output_colors_count++] = (int8_t)color;
            });
    }
};

// Which cells kept their color from input to output, one map per example.
struct unchanged_cells_facts {
    // One map per train pair: true = the cell kept its color
    vector<array<bool, MAX_GRID_CELLS>> kept;
    // Per pair: share of cells kept
    vector<float> kept_fraction;
    // The worst pair — the safe number to gate on
    float min_kept_fraction = 0.0f;
    // False until filled in (infer() only runs this on same-size puzzles)
    bool computed = false;

    // Fills the maps above by comparing each input to its output, cell by cell.
    static unchanged_cells_facts detect(const vector<example>& train) {
        unchanged_cells_facts facts;
        if (!pairs_overlay(train)) return facts;

        facts.kept.resize(train.size());
        facts.kept_fraction.assign(train.size(), 0.0f);
        facts.min_kept_fraction = 1.0f;

        for (size_t i = 0; i < train.size(); ++i) {
            const grid& in = train[i].input;
            const grid& out = train[i].output;
            auto& kept = facts.kept[i];
            kept.fill(false);
            int same = 0;
            for (int row = 0; row < in.height && row < MAX_GRID_DIM; ++row)
                for (int col = 0; col < in.width && col < MAX_GRID_DIM; ++col)
                    if (in.data[row][col] == out.data[row][col]) {
                        kept[cell_index(row, col)] = true;
                        ++same;
                    }
            facts.kept_fraction[i] = (float)same / (float)(in.width * in.height);
            facts.min_kept_fraction = min(facts.min_kept_fraction, facts.kept_fraction[i]);
        }
        facts.computed = true;
        return facts;
    }
};

// Which cells each color gained and lost, per example.
struct color_diff_facts {
    // One map per train pair per color, indexed [pair][color]
    // gained: output cell is this color, input cell wasn't
    vector<array<vector<pair<int8_t, int8_t>>, COLOR_COUNT>> gained;
    // Lost: input cell was this color, output cell isn't
    vector<array<vector<pair<int8_t, int8_t>>, COLOR_COUNT>> lost;
    // Per color, across all pairs: total gained / lost cell counts
    int gained_count[COLOR_COUNT] = {};
    int lost_count[COLOR_COUNT] = {};
    bool computed = false;

    // Records every changed cell under the color it lost and the color it gained.
    static color_diff_facts detect(const vector<example>& train) {
        color_diff_facts facts;
        if (!pairs_overlay(train)) return facts;

        facts.gained.resize(train.size());
        facts.lost.resize(train.size());
        for (size_t i = 0; i < train.size(); ++i) {
            const grid& in = train[i].input;
            const grid& out = train[i].output;
            for (int row = 0; row < in.height; ++row)
                for (int col = 0; col < in.width; ++col) {
                    int was = in.data[row][col], now = out.data[row][col];
                    if (was == now) continue;
                    if (is_color(now)) {
                        facts.gained[i][now].push_back({(int8_t)row, (int8_t)col});
                        facts.gained_count[now]++;
                    }
                    if (is_color(was)) {
                        facts.lost[i][was].push_back({(int8_t)row, (int8_t)col});
                        facts.lost_count[was]++;
                    }
                }
        }
        facts.computed = true;
        return facts;
    }
};

// Candidate color pairs: changed A cells become B and changed B cells become A.
// Unchanged occurrences may remain; this alone does not establish a swap that
// can be said out loud.
struct color_swap_candidates {
    // Indexed by the op's variant k = row-major pair {a,b}, a<b — the same
    // encoding gpu_exec_bidi_color_map decodes
    bool variant[COLOR_PAIR_COUNT] = {};
    // Subset a sentence may say. A swap is spoken over CurrentObjects, never
    // over the grid, and the background is in no piece -- so a color that only
    // ever sits in the background has nothing to say about the swap, and the
    // half of the pair that never shows up in a piece is true by saying nothing.
    bool over_the_pieces[COLOR_PAIR_COUNT] = {};
    int  count = 0;

    // Keeps pairs whose changed cells are consistent with a swap in every example.
    static color_swap_candidates detect(const vector<example>& train,
                                  const color_diff_facts& diff, int8_t background) {
        color_swap_candidates candidates;
        if (!diff.computed || !pairs_overlay(train)) return candidates;

        for (int variant = 0; variant < COLOR_PAIR_COUNT; ++variant) {
            const auto [first, second] = color_pair::from_index(variant);
            // Neither color ever changes anywhere, so this pair says nothing
            if (diff.lost_count[first] == 0 && diff.lost_count[second] == 0) continue;
            bool swaps = true;
            for (size_t i = 0; swaps && i < train.size(); ++i) {
                const grid& out = train[i].output;
                auto became = [&](const vector<pair<int8_t, int8_t>>& cells, int want) {
                    for (auto [row, col] : cells)
                        if (out.data[row][col] != want) return false;
                    return true;
                };
                swaps = became(diff.lost[i][first], second) &&
                        became(diff.lost[i][second], first);
            }
            if (!swaps) continue;
            candidates.variant[variant] = true;
            candidates.count++;

            bool in_every_piece = true;
            for (const auto& pair : train) {
                const grid& in = pair.input;
                const grid& out = pair.output;
                for (int row = 0; in_every_piece && row < in.height; ++row)
                    for (int col = 0; col < in.width; ++col) {
                        const int was = in.data[row][col];
                        if (was == background) continue;   // in no piece to be swapped
                        if ((was == first && out.data[row][col] != second) ||
                            (was == second && out.data[row][col] != first)) {
                            in_every_piece = false;
                            break;
                        }
                    }
                if (!in_every_piece) break;
            }
            candidates.over_the_pieces[variant] = in_every_piece;
        }
        return candidates;
    }
};

// Maps shapes to colors (rejected if any shape needs two different colors).
struct shape_color_map_rule {
    static constexpr int MAX_KEYS = 16;
    uint32_t keys[MAX_KEYS]   = {};  // FNV-1a hash of the key glyph's occupancy
    int8_t   colors[MAX_KEYS] = {};  // color the target is repainted to
    int      count = 0;
    bool holds() const { return count > 0; }
    // WHICH DETECTOR FILLED IT. Two of them share this storage and mean
    // different things by it: body_recolor repaints an object, interior_fill
    // paints the hole inside one. A consumer that cannot tell them apart is
    // feeding one op the other's table, so the table says which it is.
    enum : int8_t { claimed_by_nobody, body_recolor, interior_fill } claimed_by
        = claimed_by_nobody;

    // Looks at small key shapes and learns what color they say to paint the target shapes.
    static shape_color_map_rule detect_body_recolor(const vector<example>& train,
                                                    int8_t background) {
        shape_color_map_rule rule;
        if (!pairs_overlay(train)) return {};

        for (const auto& ex : train) {
            const grid& in = ex.input;
            const grid& out = ex.output;

            // Per-color box over the input (background excluded). An absent color
            // leaves its box empty, so the boxes alone say who is here.
            cell_box box[COLOR_COUNT];
            color_boxes(in, box, background);

            // Key = the color with the smallest box, target = the largest.
            int key = -1, target = -1;
            int smallest_box = 1 << 30, largest_box = -1;
            for (int color = 0; color < COLOR_COUNT; ++color) {
                if (box[color].empty()) continue;
                int box_area = box[color].area();
                if (box_area < smallest_box) { smallest_box = box_area; key = color; }
                if (box_area > largest_box)  { largest_box = box_area; target = color; }
            }
            if (key < 0 || target < 0 || key == target) return {};  // need two distinct classes

            // The color the target wears in the output (read at its first input pixel).
            int8_t repainted_to = -1;
            for (int row = box[target].row0; row <= box[target].row1 && repainted_to < 0; ++row)
                for (int col = box[target].col0; col <= box[target].col1; ++col)
                    if (in.data[row][col] == target) {
                        repainted_to = (int8_t)out.data[row][col]; break;
                    }
            if (!is_color(repainted_to)) return {};

            // Hash the key glyph's occupancy.
            const int key_width = box[key].width();
            const int key_height = box[key].height();
            uint32_t key_shape = shape_occupancy_hash(key_width, key_height, [&](int i) {
                return in.data[box[key].row0 + i / key_width][box[key].col0 + i % key_width] == key;
            });

            // Insert with the consistency gate.
            if (!rule.learn(key_shape, repainted_to)) return {};
        }
        if (rule.holds()) rule.claimed_by = shape_color_map_rule::body_recolor;
        return rule;
    }

    // Learns which color fills the hollow inside of each shape. Shares storage with
    // detect_body_recolor, so infer() runs this only when that one finds nothing.
    static shape_color_map_rule detect_interior_fill(const vector<example>& train,
                                                     int8_t background) {
        shape_color_map_rule rule;
        if (!pairs_overlay(train)) return {};

        object_walk_scratch scratch;
        bool occupied[MAX_GRID_CELLS] = {};
        bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
        for (const auto& ex : train) {
            const grid& in = ex.input;
            const grid& out = ex.output;
            bool consistent = true;
            each_object(in, background, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                if (!consistent) return;
                const int left = box.x0, top = box.y0, width = box.w, height = box.h;
                fill_n(occupied, width * height, false);
                for (int i = 0; i < count; ++i)
                    occupied[(cell_row(cells[i]) - top) * width + cell_col(cells[i]) - left] = true;
                mark_outside(height, width,
                    [&](int row, int col) { return occupied[row * width + col]; }, outside);

                // Fill color = the output color on interior cells whose INPUT was
                // background — the cells the fill actually paints. Cells that are
                // non-background in the input (inside-noise like a marker pixel)
                // survive unchanged, so they are skipped. Must be one color across
                // the painted interior.
                int fill = -1; bool any_painted = false;
                for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    int id = row * width + col;
                    if (occupied[id] || outside[row][col]) continue;         // not interior
                    int in_row = top + row, in_col = left + col;
                    if (in.data[in_row][in_col] != background) continue;  // inside-noise
                    int out_color = out.data[in_row][in_col];
                    if (!is_color(out_color) || (any_painted && out_color != fill)) {
                        consistent = false;
                        return;
                    }
                    fill = out_color;
                    any_painted = true;
                }
                if (!any_painted) return;

                uint32_t object_shape =
                    shape_occupancy_hash(width, height, [&](int i) { return occupied[i]; });
                consistent = rule.learn(object_shape, (int8_t)fill);
            });
            if (!consistent) return {};
        }
        if (rule.holds()) rule.claimed_by = shape_color_map_rule::interior_fill;
        return rule;
    }

private:
    // Remembers which color goes with this shape. Returns false if the shape gets conflicting colors.
    bool learn(uint32_t shape, int8_t color) {
        for (int i = 0; i < count; ++i)
            if (keys[i] == shape) return colors[i] == color;
        if (count >= MAX_KEYS) return false;
        keys[count] = shape;
        colors[count] = color;
        count++;
        return true;
    }
};

// A repeating pattern carried across the whole output, with its colors remapped.
struct motif_continuation_rule {
    bool held = false;
    recoloring color_map;
    bool holds() const { return held; }

    // Checks that the input is one tiling pattern and reads the recolor off the output.
    static motif_continuation_rule detect(const vector<example>& train,
                                          const correspondence_election& correspondence) {
        motif_continuation_rule rule;
        if (train.empty() || correspondence.chosen_pairs().size() != train.size() ||
            correspondence.segmentation != segmentation_methods::minimal_repeating_set) return rule;
        for (size_t i = 0; i < train.size(); ++i) {
            const auto& pair = correspondence.chosen_pairs()[i];
            if (pair.edges.size() != 1 || !pair.left_over.empty()) return {};
            const auto& edge = pair.edges.front();
            if (!edge.from_input || !edge.same_shape || edge.pose || edge.hidden || edge.outside ||
                edge.from < 0 || edge.to < 0 || edge.from >= (int)pair.input_info.size() ||
                edge.to >= (int)pair.output_info.size()) return {};
            const auto& from = pair.input_info[edge.from].summary;
            const auto& into = pair.output_info[edge.to].summary;
            if (from.x0 || from.y0 || into.x0 || into.y0 ||
                from.w != into.w || from.h != into.h) return {};
            auto instances_only = [](const auto& objects, int representative) {
                for (int j = 0; j < (int)objects.size(); ++j)
                    if (j != representative && objects[j].instance_of != representative) return false;
                return true;
            };
            if (pair.output_info.size() < 2 ||
                !instances_only(pair.input_info, edge.from) ||
                !instances_only(pair.output_info, edge.to)) return {};
            const auto& input = train[i].input;
            const auto& output = train[i].output;
            const auto [width, height] = object_segmentation::pattern_block_of(input);
            // Pattern is input-only. A block discovered only on the output cannot
            // justify spelling Pattern with a different unit on the input.
            if (width != from.w || height != from.h) return {};
            for (int r = 0; r < input.height; ++r)
                for (int c = 0; c < input.width; ++c)
                    if (input.data[r][c] != input.data[r % height][c % width]) return {};
            for (int r = 0; r < output.height; ++r)
                for (int c = 0; c < output.width; ++c) {
                    const int was = input.data[r % height][c % width];
                    const int now = output.data[r][c];
                    if (!is_color(was) || !is_color(now)) return {};
                    if (rule.color_map[was] >= 0 && rule.color_map[was] != now) return {};
                    rule.color_map[was] = (int8_t)now;
                }
        }
        rule.held = true;
        return rule;
    }
};

// Seat on the colour cycle; lean 0 walks row + col, lean 1 row - col.
ARC_HOST_DEVICE inline int diagonal_seat(int row, int col, int lean, int colors) {
    const int diagonal = lean ? row - col : row + col;
    return ((diagonal % colors) + colors) % colors;
}

// THE SAME CONTINUATION ALONG THE DIAGONALS. The colors cycle as the diagonals
// are crossed, the input paints part of that cycle, and the output is the cycle
// carried over every cell -- the motif above with diagonals in place of a box.
//
// THE PERIOD IS HOW MANY COLORS THERE ARE, which is DiagTile's arithmetic and
// not a choice made here: the op counts the colors it was handed and steps the
// cycle by that count. Diagonals that repeat on any other period are a reading
// this has no way to say, so it does not claim them.
//
// ONE COLOR IS NO CYCLE. With a single color every diagonal is the same
// diagonal, the claim comes out true of any grid painted all one color, and it
// would say "diagonals" about a puzzle that never mentioned them.
//
// AND IT HAS TO CARRY SOMETHING. An output the input already was is the cycle
// redrawn where it already sat, which says nothing about diagonals either.
struct diagonal_continuation_rule {
    bool held = false;
    // DiagTile's own argument: 0 for the diagonals of constant row + column,
    // 1 for those of constant row - column.
    int8_t lean = 0;
    bool holds() const { return held; }

    // Reads the cycle off each input and keeps the lean that redraws every
    // output exactly. Nothing about the colors is kept: which color sits on
    // which diagonal is read off the grid in hand, differently in every pair,
    // and that is what makes this a rule rather than a table.
    static diagonal_continuation_rule detect(const vector<example>& train,
                                             const correspondence_election& correspondence,
                                             int8_t background) {
        // The cycle is read off the cells where they sit, so the pieces have to
        // be the input's own cells in the input's own frame.
        if (train.empty() || !boxed_by_its_cells(correspondence.segmentation) ||
            !is_color(background)) return {};
        auto redraws_every_output = [&](int8_t lean) {
            bool carried_something = false;
            for (const auto& pair : train) {
                int counts[COLOR_COUNT];
                count_colors(pair.input, counts, background);
                int colors = 0;
                for (int color = 0; color < COLOR_COUNT; ++color) colors += counts[color] > 0;
                if (colors < 2) return false;
                auto seat_of = [&](int row, int col) { return diagonal_seat(row, col, lean, colors); };
                int8_t cycle[COLOR_COUNT];
                for (int seat = 0; seat < colors; ++seat) cycle[seat] = -1;
                bool agrees = true;
                walk_colored_cells(pair.input, background, [&](int row, int col, int color) {
                    int8_t& seated = cycle[seat_of(row, col)];
                    agrees = agrees && (seated < 0 || seated == color);
                    seated = (int8_t)color;
                });
                if (!agrees) return false;
                // A seat no cell sat in is a color the cycle never named. The op
                // hands the pieces back untouched rather than guess it, so a
                // cycle with a hole in it is not a sentence.
                for (int seat = 0; seat < colors; ++seat) if (cycle[seat] < 0) return false;
                for (int row = 0; row < pair.output.height; ++row)
                    for (int col = 0; col < pair.output.width; ++col) {
                        if (pair.output.data[row][col] != cycle[seat_of(row, col)]) return false;
                        carried_something = carried_something ||
                            row >= pair.input.height || col >= pair.input.width ||
                            pair.input.data[row][col] != pair.output.data[row][col];
                    }
            }
            return carried_something;
        };
        for (const int8_t lean : {(int8_t)0, (int8_t)1})
            if (redraws_every_output(lean)) return {true, lean};
        return {};
    }
};

// HOW A RULE HERE IS CHECKED RATHER THAN GUESSED: replayed. `draw` draws one
// pair's output from its input, or says false when it cannot, and the claim
// holds only when every drawing comes out exactly.
template <class Draw>
bool redraws_every_output(const vector<example>& train, Draw draw) {
    for (const example& pair : train) {
        grid drawn;
        if (!draw(pair, drawn) || drawn != pair.output) return false;
    }
    return true;
}

// A REPEATING PICTURE WITH NOISE ON IT, AND THE OUTPUT IS THE PICTURE CLEAN.
// The input is a lattice of copies that noise has scribbled over, so no cut of
// it finds the copies -- 0607ce86 cuts into rubble every way it is read. What
// finds them is the vote: each cell takes the majority of the cells whole
// periods away. Asked by running that vote, VoteDenoise's own arithmetic, on
// every input under the elected ground and holding only where it draws every
// output exactly. The ground is kept in, because it is the evidence the period
// search is built on (notes/4 - To Do/VoteDenoise Needs The Background.md).
//
// AND IT HAS TO CLEAN SOMETHING. An output the input already was is a grid
// that happens to repeat, which says nothing about noise.
struct periodic_denoise_rule {
    bool held = false;
    bool holds() const { return held; }

    static periodic_denoise_rule detect(const vector<example>& train, int8_t background) {
        if (train.empty() || !is_color(background)) return {};
        bool cleaned_something = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const grid& input = pair.input;
            if (!input.same_size(pair.output)) return false;
            drawn = input;
            const bool voted = vote_denoise_cells(input.width, input.height, background,
                [&](int r, int c) -> int8_t {
                    const int8_t v = (int8_t)input.data[r][c];
                    return v == TRANSPARENT ? background : v;
                },
                [&](int r, int c, int8_t v) { drawn.data[r][c] = v; });
            cleaned_something = cleaned_something || drawn != input;
            return voted;
        });
        return { redraws && cleaned_something };
    }
};

// WHAT A CUT IS ALLOWED TO FOLLOW. Where the cut falls -- which axis, which
// color, how many panes -- is read off the grid; these are only the permissions,
// the way degrees_of_freedom holds them for matching.
struct split_freedom {
    // A full-span line of a color that is not the background: a wall.
    bool wall = true;
    // A full-span line of the background: the same shape of evidence, thinner.
    bool blank = true;
    // No line at all, k equal parts. It fits almost any grid, so on its own it
    // says nothing and something else has to license it.
    bool bare = true;
    // Cut both axes at once, leaving a lattice of panes instead of a row.
    bool lattice = true;
    // Panes that are not all one size: a picture beside a key, rather than a row
    // of things to compare.
    bool uneven = true;

    static constexpr int FREEDOMS = 5;
    static constexpr const char* freedom_names_lead = "The Cut May";
    static constexpr const char* const freedom_names[FREEDOMS] =
        { "Follow A Wall", "Follow A Blank", "Have No Line",
          "Go Both Ways", "Leave Panes Uneven" };
    constexpr bool may(int freedom) const {
        return freedom == 0 ? wall : freedom == 1 ? blank : freedom == 2 ? bare
             : freedom == 3 ? lattice : uneven;
    }
    string label() const { return freedoms_said(*this, "Not Cut At All"); }
};

// HOW A GRID WAS CUT INTO PANES, named by five numbers, so two pairs of
// different sizes can still be asked whether they were cut the same way.
struct pane_cut {
    // What the cut followed. The order is the precedence: plainest first.
    enum : int8_t { Wall, Blank, Bare };
    // Which way it ran. BothWays leaves a lattice rather than a row of panes.
    enum : int8_t { Across, Down, BothWays };

    int8_t cut = Wall;
    int8_t axis = Across;
    // The separator color, or -1 when the cut followed no line.
    int8_t color = -1;
    // How many panes the cut leaves. One of these is 1 unless it went both ways.
    int8_t down = 1, across = 1;

    bool operator==(const pane_cut& other) const = default;
};

// The input is not one grid but several laid side by side -- panes -- with a
// wall, a blank line, or nothing between them. input_blocks_rule below cuts the
// OUTPUT into input-sized blocks; this cuts the INPUT into panes.
//
// Nothing here is scored. A cut holds only when the SAME one holds on every
// training pair and no rival survives beside it: a split the pairs agree on
// without argument is a fact, and anything less is not a split at all. Ties
// break wall > blank > bare, fewest assumptions first.
struct pane_split_facts : pane_cut {
    // Where the panes sit in one training input, in reading order.
    struct pane {
        int8_t top = 0, left = 0, width = 0, height = 0;

        // Read local coordinates, optionally through a rotation or reflection.
        int color_at(const grid& source, int row, int col, int pose = 0) const {
            const auto [source_row, source_col] = pose_source(pose, width, height, row, col);
            return source.data[top + source_row][left + source_col];
        }
        // Whether the pane reads through this pose at its own size.
        bool keeps_size(int pose) const {
            return posed_size(pose, width, height) == pair<int, int>{ width, height };
        }
        // The pane cut out of its grid, as a grid of its own.
        grid drawn(const grid& source) const {
            grid out(width, height);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) out.data[row][col] = color_at(source, row, col);
            return out;
        }
    };
    vector<vector<pane>> pairs;

    bool held = false;
    bool holds() const { return held; }

    // A pane this small is a cell, not a grid: without the floor any 3x3 can
    // claim to be nine panes. It is what keeps the lattice honest.
    static constexpr int SMALLEST_PANE = 2;

    // Rows and columns painted one color from end to end.
    static void full_lines(const grid& one, int color,
                           vector<int>& rows, vector<int>& cols) {
        rows.clear();
        cols.clear();
        for (int r = 0; r < one.height; ++r) {
            bool whole = true;
            for (int c = 0; c < one.width && whole; ++c) whole = one.data[r][c] == color;
            if (whole) rows.push_back(r);
        }
        for (int c = 0; c < one.width; ++c) {
            bool whole = true;
            for (int r = 0; r < one.height && whole; ++r) whole = one.data[r][c] == color;
            if (whole) cols.push_back(c);
        }
    }

    // The stretches left between the cut lines. Neighbouring lines leave nothing
    // between them, so a wall two cells thick costs no extra freedom.
    static vector<pair<int, int>> spans_between(int length, const vector<int>& lines) {
        vector<pair<int, int>> spans;
        int start = 0;
        for (int line : lines) {
            if (line > start) spans.push_back({ start, line });
            start = line + 1;
        }
        if (length > start) spans.push_back({ start, length });
        return spans;
    }

    // The same stretches when no line made them: k equal parts.
    static vector<pair<int, int>> even_spans(int length, int parts) {
        vector<pair<int, int>> spans;
        for (int i = 0; i < parts; ++i)
            spans.push_back({ i * (length / parts), (i + 1) * (length / parts) });
        return spans;
    }

    // Lays the panes out in reading order, and says no when they fail a guard.
    static bool lay_out(const vector<pair<int, int>>& rows,
                        const vector<pair<int, int>>& cols, bool uneven,
                        vector<pane>& panes) {
        panes.clear();
        for (auto [top, bottom] : rows)
            for (auto [left, right] : cols) {
                const int height = bottom - top, width = right - left;
                if (height < SMALLEST_PANE || width < SMALLEST_PANE) return false;
                panes.push_back({ (int8_t)top, (int8_t)left, (int8_t)width, (int8_t)height });
            }
        if (panes.size() < 2) return false;
        if (!uneven)
            for (const auto& one_pane : panes)
                if (one_pane.width != panes.front().width ||
                    one_pane.height != panes.front().height) return false;
        return true;
    }

    // Every way this one grid could be cut, under these permissions.
    static void readings_of(const grid& one, int8_t background, split_freedom allowed,
                            vector<pair<pane_cut, vector<pane>>>& found) {
        found.clear();
        vector<int> row_lines, col_lines;
        vector<pane> panes;
        // A color that is absent offers no full line, so it costs nothing to ask.
        for (int color = 0; color < COLOR_COUNT; ++color) {
            const int8_t cut = color == background ? Blank : Wall;
            if (!(cut == Wall ? allowed.wall : allowed.blank)) continue;
            full_lines(one, color, row_lines, col_lines);
            const auto rows = spans_between(one.height, row_lines);
            const auto cols = spans_between(one.width, col_lines);
            if (allowed.lattice && rows.size() > 1 && cols.size() > 1) {
                if (lay_out(rows, cols, allowed.uneven, panes))
                    found.push_back({ { cut, BothWays, (int8_t)color,
                                        (int8_t)rows.size(), (int8_t)cols.size() }, panes });
                continue;
            }
            if (cols.size() > 1 && rows.size() == 1 && lay_out(rows, cols, allowed.uneven, panes))
                found.push_back({ { cut, Across, (int8_t)color, 1, (int8_t)cols.size() }, panes });
            if (rows.size() > 1 && cols.size() == 1 && lay_out(rows, cols, allowed.uneven, panes))
                found.push_back({ { cut, Down, (int8_t)color, (int8_t)rows.size(), 1 }, panes });
        }
        if (!allowed.bare) return;
        const vector<pair<int, int>> whole_height{ { 0, one.height } };
        const vector<pair<int, int>> whole_width{ { 0, one.width } };
        for (int parts = 2; parts <= 4; ++parts) {
            if (one.width % parts == 0 &&
                lay_out(whole_height, even_spans(one.width, parts), allowed.uneven, panes))
                found.push_back({ { Bare, Across, -1, 1, (int8_t)parts }, panes });
            if (one.height % parts == 0 &&
                lay_out(even_spans(one.height, parts), whole_width, allowed.uneven, panes))
                found.push_back({ { Bare, Down, -1, (int8_t)parts, 1 }, panes });
        }
        if (!allowed.lattice) return;
        for (int rows = 2; rows <= 3; ++rows)
            for (int cols = 2; cols <= 3; ++cols)
                if (one.height % rows == 0 && one.width % cols == 0 &&
                    lay_out(even_spans(one.height, rows), even_spans(one.width, cols),
                            allowed.uneven, panes))
                    found.push_back({ { Bare, BothWays, -1, (int8_t)rows, (int8_t)cols }, panes });
    }

    // Cuts a grid that was not among the pairs the same way the pairs were cut,
    // so whatever the panes come to can be asked of a test input too. The caller
    // checks the pane sizes it needs; this only reproduces the reading.
    bool panes_of(const grid& one, vector<pane>& found) const {
        if (!held) return false;
        vector<int> row_lines, col_lines;
        vector<pair<int, int>> rows, cols;
        if (cut == Bare) {
            if (one.height % down || one.width % across) return false;
            rows = even_spans(one.height, down);
            cols = even_spans(one.width, across);
        } else {
            if (!is_color(color)) return false;
            full_lines(one, color, row_lines, col_lines);
            rows = spans_between(one.height, row_lines);
            cols = spans_between(one.width, col_lines);
        }
        if ((int)rows.size() != down || (int)cols.size() != across) return false;
        return lay_out(rows, cols, true, found);
    }

    // Reads every training input, keeps the readings they all agree on, and holds
    // only when exactly one of those survives the precedence.
    static pane_split_facts detect(const vector<example>& train, int8_t background,
                                   split_freedom allowed = {}) {
        if (train.empty()) return {};
        vector<pair<pane_cut, vector<pane>>> found;
        vector<pane_cut> agreed;
        for (size_t i = 0; i < train.size(); ++i) {
            readings_of(train[i].input, background, allowed, found);
            // A wall is a wall whether or not the answer is a pane wide: the
            // input alone shows it, so it is admitted on its own evidence. A cut
            // that follows no line shows nothing, and fits almost any grid, so it
            // is admitted only when the output is the size of one pane. The
            // asymmetry is the point -- it is the line between a fact and a guess.
            vector<pane_cut> here;
            for (const auto& [named, panes] : found) {
                if (named.cut != Bare) { here.push_back(named); continue; }
                for (const auto& one_pane : panes)
                    if (one_pane.width == train[i].output.width &&
                        one_pane.height == train[i].output.height) {
                        here.push_back(named);
                        break;
                    }
            }
            if (i == 0) { agreed = here; continue; }
            erase_if(agreed, [&](const pane_cut& named) {
                return ranges::find(here, named) == here.end(); });
            if (agreed.empty()) return {};
        }
        if (agreed.empty()) return {};
        // Fewest assumptions first: a wall outranks a blank line, which outranks
        // no line at all. Whatever is still tied after that is an argument, not a fact.
        const int8_t plainest = ranges::min(agreed, {}, &pane_cut::cut).cut;
        erase_if(agreed, [&](const pane_cut& named) { return named.cut != plainest; });
        if (agreed.size() != 1) return {};

        const pane_cut won = agreed.front();
        pane_split_facts facts{ won };
        for (const auto& train_pair : train) {
            readings_of(train_pair.input, background, allowed, found);
            for (const auto& [named, panes] : found)
                if (named == won) { facts.pairs.push_back(panes); break; }
        }
        if (facts.pairs.size() != train.size()) return {};
        facts.held = true;
        return facts;
    }
};

// WHAT THE PANES SAY TOGETHER, once they have been cut apart.
//
// The plain way to read a pair of panes is to write down every combination of
// colors that turned up and what each produced. That is a table of 10^k rows
// which cannot answer a combination it never saw, and it carries no reason --
// sixteen rows where the puzzle has one idea.
//
// So the colors are dropped first. At each position only ONE thing is asked of
// each pane: is there something here, or is this ground? That leaves a mask of
// k bits, and for every mask the answer is either a color to paint or one named
// pane to copy. Both intersection and priority fall out of that shape without
// being named: an AND is the mask with every bit set painting one color, a
// priority order is each mask copying its leading pane. And a mask that never
// turned up in training is still answerable, because the colors that would have
// made it no longer matter.
struct pane_mask_rule {
    // What one mask says. A pane index copies that pane's own color at the cell;
    // otherwise paint is laid down. Exactly one of the two is set.
    struct answer {
        int8_t paint = -1;
        int8_t copy_pane = -1;
        bool said() const { return paint >= 0 || copy_pane >= 0; }
    };
    // Indexed by the mask, so 1 << panes entries. A mask nothing ever said is
    // left unsaid rather than guessed.
    vector<answer> by_mask;
    // The pose every pane after the first is read through. Panes facing each
    // other across a wall are mirrored, not shifted, and without this their
    // masks line up wrong and every reading conflicts.
    int8_t posed = 0;
    // How many panes each mask counts, so a reader knows how wide the bits are.
    int8_t panes = 0;
    bool held = false;
    bool holds() const { return held; }

    // What pane p shows at (r, c). The first pane is read as it lies; the rest
    // are turned into its frame, which is what makes them comparable.
    static int shown(const vector<pane_split_facts::pane>& boxes, int p, const grid& input,
                     int r, int c, int pose) {
        return boxes[p].color_at(input, r, c, p ? pose : 0);
    }
    // Which panes have something at (r, c), one bit each.
    static int mask_at(const vector<pane_split_facts::pane>& boxes, const grid& input,
                       int r, int c, int pose, int8_t background) {
        int mask = 0;
        for (int p = 0; p < (int)boxes.size(); ++p)
            if (shown(boxes, p, input, r, c, pose) != background) mask |= 1 << p;
        return mask;
    }

    // Reads every training pair under one pose and says what each mask means,
    // or says nothing when two cells with the same mask disagree beyond repair.
    static pane_mask_rule read_with(const vector<example>& train,
                                    const pane_split_facts& split,
                                    int8_t background, int pose) {
        pane_mask_rule rule;
        if (train.size() != split.pairs.size() || split.pairs.empty()) return {};
        const int panes = (int)split.pairs.front().size();
        if (panes < 2 || panes > 8) return {};
        rule.panes = (int8_t)panes;
        rule.posed = (int8_t)pose;
        rule.by_mask.assign((size_t)1 << panes, {});
        // A pane may only copy when every cell of that mask agreed it could, so
        // the candidates start open and are struck off as the pairs disagree.
        vector<uint8_t> may_copy((size_t)1 << panes, (uint8_t)((1u << panes) - 1));
        vector<uint8_t> painted((size_t)1 << panes, 0);
        for (size_t which = 0; which < train.size(); ++which) {
            const auto& boxes = split.pairs[which];
            const grid& input = train[which].input;
            const grid& output = train[which].output;
            if ((int)boxes.size() != panes) return {};
            for (const auto& box : boxes)
                if (box.width != output.width || box.height != output.height || !box.keeps_size(pose))
                    return {};
            for (int r = 0; r < output.height; ++r)
                for (int c = 0; c < output.width; ++c) {
                    const int mask = mask_at(boxes, input, r, c, pose, background);
                    const int now = output.data[r][c];
                    for (int p = 0; p < panes; ++p)
                        if (shown(boxes, p, input, r, c, pose) != now)
                            may_copy[mask] &= (uint8_t)~(1u << p);
                    if (!painted[mask]) {
                        painted[mask] = 1;
                        rule.by_mask[mask].paint = (int8_t)now;
                    } else if (rule.by_mask[mask].paint != now) {
                        rule.by_mask[mask].paint = -1;
                    }
                }
        }
        // A mask that always painted the same color says so; one that did not
        // must name a pane it always copied, or it says nothing and the rule falls.
        for (size_t mask = 0; mask < rule.by_mask.size(); ++mask) {
            if (!painted[mask]) continue;
            if (rule.by_mask[mask].paint >= 0) continue;
            if (!may_copy[mask]) return {};
            for (int p = 0; p < panes; ++p)
                if (may_copy[mask] >> p & 1) { rule.by_mask[mask].copy_pane = (int8_t)p; break; }
        }
        rule.held = true;
        return rule;
    }

    // Draws what the panes of one input come to. Says false when the split will
    // not cut this grid, or when a mask turns up that training never explained --
    // declining beats guessing, since a guess here is a wrong answer that looks
    // like a right one.
    bool draw(const grid& input, const pane_split_facts& split, int8_t background,
              grid& drawn) const {
        vector<pane_split_facts::pane> boxes;
        if (!holds() || !split.panes_of(input, boxes)) return false;
        if ((int)boxes.size() != panes) return false;
        const auto& first = boxes.front();
        for (const auto& box : boxes)
            if (box.width != first.width || box.height != first.height || !box.keeps_size(posed))
                return false;
        drawn = grid(first.width, first.height);
        for (int r = 0; r < drawn.height; ++r)
            for (int c = 0; c < drawn.width; ++c) {
                const answer& says = by_mask[mask_at(boxes, input, r, c, posed, background)];
                if (!says.said()) return false;
                drawn.data[r][c] = says.paint >= 0 ? says.paint
                                                   : shown(boxes, says.copy_pane, input, r, c, posed);
            }
        return true;
    }

    // Tries every pose and keeps the plainest one that redraws every output. As
    // is comes first, so a pose is only ever worn when it is doing work.
    static pane_mask_rule detect(const vector<example>& train,
                                 const pane_split_facts& split, int8_t background) {
        if (!split.holds() || !is_color(background)) return {};
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            const pane_mask_rule rule = read_with(train, split, background, pose);
            if (rule.holds() && redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    return rule.draw(pair.input, split, background, drawn);
                })) return rule;
        }
        return {};
    }
};

// WHICH PANE THE ANSWER IS, when the answer is one of the panes untouched.
//
// pane_mask_rule reads the panes against each other and makes a new drawing.
// This one makes no drawing at all: it hands back a pane, and the only question
// is which. So the whole rule is the reason for choosing, and the ways to choose
// are ordered plainest first -- a property the pane HAS beats a place it SITS,
// because a place is a magic number and a property is a reason. The first way
// that names the right pane in every training pair wins; nothing is scored.
struct pane_pick_rule {
    enum : int8_t {
        // It is the one unlike its fellows, by drawing or by palette.
        OddOneOut, OddColor,
        // It is the one its fellows repeat.
        TheyRepeat,
        // It is the one that breaks a symmetry the others keep, or keeps one
        // they break. `which` is the pose the symmetry folds along.
        NotSymmetric, OnlySymmetric,
        // It is the one carrying most or least. Measured and never needed so
        // far, carried because they cost one line each and the panes are cut.
        Fullest, Emptiest, MostColored, LeastColored,
        // It is simply the one in that place. Last, because it gives no reason.
        AtIndex,
        WAYS
    };
    int8_t way = AtIndex;
    // The pose a symmetry folds along, or the place AtIndex means.
    int8_t which = 0;
    bool held = false;
    bool holds() const { return held; }

    // The four folds a square drawing can sit on, as poses: along the diagonal,
    // top to bottom, left to right, and half way round.
    static constexpr int8_t FOLDS[] = { 1, 2, 4, 6 };

    static bool same_drawing(const grid& one, const pane_split_facts::pane& left,
                             const pane_split_facts::pane& right) {
        return left.drawn(one) == right.drawn(one);
    }
    static bool folds_onto_itself(const grid& one, const pane_split_facts::pane& box, int pose) {
        if (!box.keeps_size(pose)) return false;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c)
                if (box.color_at(one, r, c) != box.color_at(one, r, c, pose)) return false;
        return true;
    }
    static uint16_t palette_of(const grid& one, const pane_split_facts::pane& box) {
        uint16_t seen = 0;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c) {
                const int was = box.color_at(one, r, c);
                if (is_color(was)) seen |= (uint16_t)(1u << was);
            }
        return seen;
    }
    static int carried_by(const grid& one, const pane_split_facts::pane& box, int8_t background) {
        int count = 0;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c)
                count += box.color_at(one, r, c) != background;
        return count;
    }

    // The one place where a count stands alone, or -1 when none does or several do.
    static int only_extreme(const vector<int>& counts, bool most) {
        int best = counts.front();
        for (int one : counts) best = most ? max(best, one) : min(best, one);
        int found = -1;
        for (size_t i = 0; i < counts.size(); ++i)
            if (counts[i] == best) { if (found >= 0) return -1; found = (int)i; }
        return found;
    }

    // Which pane this way names on this grid, or -1 when it names none.
    static int pick(const grid& one, const vector<pane_split_facts::pane>& boxes,
                    int8_t background, int8_t way, int8_t which) {
        const int count = (int)boxes.size();
        if (count < 2) return -1;
        switch (way) {
        case AtIndex:
            return which < count ? which : -1;
        case OddOneOut: case TheyRepeat: {
            vector<int> alike(count, 0);
            for (int i = 0; i < count; ++i)
                for (int j = 0; j < count; ++j)
                    alike[i] += i != j && same_drawing(one, boxes[i], boxes[j]);
            // Alone among fellows that have each other, or one of the crowd.
            const int found = only_extreme(alike, way == TheyRepeat);
            if (found < 0 || alike[found] != (way == TheyRepeat ? count - 1 : 0)) return -1;
            return way == TheyRepeat || alike[found] == 0 ? found : -1;
        }
        case OddColor: {
            vector<int> alike(count, 0);
            for (int i = 0; i < count; ++i)
                for (int j = 0; j < count; ++j)
                    alike[i] += i != j && palette_of(one, boxes[i]) == palette_of(one, boxes[j]);
            const int found = only_extreme(alike, false);
            return found >= 0 && alike[found] == 0 ? found : -1;
        }
        case NotSymmetric: case OnlySymmetric: {
            int found = -1;
            for (int i = 0; i < count; ++i) {
                const bool folds = folds_onto_itself(one, boxes[i], which);
                if (folds != (way == OnlySymmetric)) continue;
                if (found >= 0) return -1;
                found = i;
            }
            return found;
        }
        case Fullest: case Emptiest: {
            vector<int> carried;
            for (const auto& box : boxes) carried.push_back(carried_by(one, box, background));
            return only_extreme(carried, way == Fullest);
        }
        case MostColored: case LeastColored: {
            vector<int> many;
            for (const auto& box : boxes) many.push_back(popcount(palette_of(one, box)));
            return only_extreme(many, way == MostColored);
        }
        default: return -1;
        }
    }

    // Hands back the pane this rule names, as a grid of its own.
    bool draw(const grid& input, const pane_split_facts& split, int8_t background,
              grid& drawn) const {
        vector<pane_split_facts::pane> boxes;
        if (!holds() || !split.panes_of(input, boxes)) return false;
        const int found = pick(input, boxes, background, way, which);
        if (found < 0) return false;
        drawn = boxes[(size_t)found].drawn(input);
        return true;
    }

    // Walks the ways in order and keeps the first that names the right pane in
    // every pair. Order is the whole tiebreak: a reason before a place.
    static pane_pick_rule detect(const vector<example>& train,
                                 const pane_split_facts& split, int8_t background) {
        if (!split.holds() || train.empty()) return {};
        auto names_it = [&](int8_t way, int8_t which) {
            const pane_pick_rule trying{ way, which, true };
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                return trying.draw(pair.input, split, background, drawn);
            });
        };
        for (int8_t way = 0; way < WAYS; ++way) {
            if (way == NotSymmetric || way == OnlySymmetric) {
                for (int8_t fold : FOLDS)
                    if (names_it(way, fold)) return { way, fold, true };
            } else if (way == AtIndex) {
                for (int8_t place = 0; place < (int8_t)split.pairs.front().size(); ++place)
                    if (names_it(way, place)) return { way, place, true };
            } else if (names_it(way, 0)) {
                return { way, 0, true };
            }
        }
        return {};
    }
};

// The output is a grid of input-sized blocks, each one a turned copy of the
// input or plain background. Every pose that draws a block is kept, not just one.
// Which blocks are copies is the correspondence's to say; this is what it adds.
struct input_blocks_rule {
    bool held = false;
    // One cell-level map shared by every copy; -1 means the color keeps itself.
    recoloring color_map;
    // OR EVERY COPY IS ITS INPUT'S NEGATIVE: figure and ground trade places.
    // The figure's color comes off each pair's own input and is kept nowhere,
    // so this is a derivation and not a table -- 0692e18c paints its copies 6,
    // 7 and 4, then asks about 3, which no table read off those pairs holds.
    bool negative = false;
    // The poses whose own figure is where the copies are: the filled blocks are
    // exactly the figure cells of the input posed that way, in every pair.
    uint8_t figure_poses = 0;
    struct layout {
        int8_t rows = 0, cols = 0;
        // Bit p is set when pose p draws the block; 0 is a block of background.
        uint8_t poses[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    };
    vector<layout> pairs;
    bool holds() const { return held; }

    // Reads the copies off the correspondence, where every block was already
    // asked whether it is the input, turned and recolored: the block cut's own
    // best reading, elected or not. What is asked here is only what places the
    // copies and what they wear -- never, a second time, whether they are copies.
    static input_blocks_rule detect(const vector<example>& train,
                                    const correspondence_election& correspondence,
                                    int input_background, int output_background) {
        if (train.empty()) return {};
        for (const auto& pair : train)
            if (pair.input.empty() ||
                pair.output.width % pair.input.width || pair.output.height % pair.input.height) return {};
        const auto* blocks = correspondence.best_reading_of(segmentation_methods::smallest_of_two_grids);
        if (!blocks || blocks->pairs.size() != train.size()) return {};
        // Background blocks need a ground both sides agree on.
        const int ground = is_color(input_background) && input_background == output_background
                           ? input_background : -1;

        input_blocks_rule rule;
        rule.figure_poses = ground < 0 ? 0 : 0xFF;
        // What each pair's copies wear, color by color of its input; -1 for a
        // color the input does not have, and everywhere in a pair with no copy.
        vector<recoloring> worn(train.size());
        bool copied_any = false;
        for (size_t which = 0; which < train.size(); ++which) {
            const example& pair = train[which];
            const auto& read = blocks->pairs[which];
            const int width = pair.input.width, height = pair.input.height;
            // The input is the one block the output was cut into copies of.
            if (read.input_info.size() != 1) return {};
            layout at;
            at.rows = (int8_t)(pair.output.height / height);
            at.cols = (int8_t)(pair.output.width / width);
            vector<bool> copied(read.output_info.size(), false);
            bool has_copy = false;
            for (const auto& edge : read.edges) {
                if (!edge.from_input || edge.to < 0 || edge.to >= (int)read.output_info.size()) continue;
                const auto& landing = read.output_info[edge.to].summary;
                at.poses[landing.y0 / height][landing.x0 / width] |= edge.ways_round;
                recoloring wears;
                walk_colored_cells(pair.input, TRANSPARENT, [&](int, int, int was) {
                    wears[was] = edge.color_map.apply(was);
                });
                // Copies of one rule wear one set of colors, within a pair at least.
                if (has_copy && wears != worn[which]) return {};
                worn[which] = wears;
                copied[edge.to] = true;
                has_copy = true;
            }
            copied_any = copied_any || has_copy;
            // A block with anything in it that is no copy is something else drawn
            // there; a block with nothing in it is ground, where there is one.
            if (!ranges::all_of(copied, [](bool one) { return one; })) return {};
            for (int r = 0; r < at.rows; ++r)
                for (int c = 0; c < at.cols; ++c)
                    if (!at.poses[r][c] && ground < 0) return {};
            for (int pose = 0; pose < POSE_COUNT; ++pose) {
                const auto [posed_width, posed_height] = posed_size(pose, width, height);
                bool figure = posed_width == at.cols && posed_height == at.rows;
                for (int r = 0; r < at.rows && figure; ++r)
                    for (int c = 0; c < at.cols && figure; ++c) {
                        const auto [row, col] = pose_source(pose, width, height, r, c);
                        figure = pair.input.data[row][col] != ground
                                 ? (at.poses[r][c] >> pose & 1) != 0 : at.poses[r][c] == 0;
                    }
                if (!figure) rule.figure_poses &= (uint8_t)~(1 << pose);
            }
            rule.pairs.push_back(at);
        }
        if (!copied_any) return {};

        // WHAT THE COPIES WEAR. One table every pair agrees on is what they
        // literally show, so it is asked first. Colors that change from pair to
        // pair are the evidence that each copy takes its ink from its own input,
        // and the one such reading claimed is the negative: figure and ground
        // trading places. A figure of several colors has no one negative.
        recoloring table;
        bool one_table = true;
        for (const recoloring& wears : worn)
            for (int was = 0; was < COLOR_COUNT; ++was) {
                if (wears[was] < 0) continue;
                one_table = one_table && (table[was] < 0 || table[was] == wears[was]);
                table[was] = wears[was];
            }
        if (one_table) {
            for (int was = 0; was < COLOR_COUNT; ++was)
                rule.color_map[was] = table.changes(was) ? table[was] : (int8_t)-1;
        } else {
            if (ground < 0) return {};
            for (size_t which = 0; which < train.size(); ++which) {
                const recoloring& wears = worn[which];
                if (wears == recoloring{}) continue;
                const int8_t figure = the_figure_color(train[which].input, ground);
                if (!is_color(figure) ||
                    (wears[ground] >= 0 && wears[ground] != figure) ||
                    wears[figure] != ground) return {};
            }
            rule.negative = true;
        }
        rule.held = true;
        return rule;
    }
};

// Read just the pieces, in the same size-and-position order used by matching.
inline vector<object> pieces_of(const grid& input, int8_t background,
                                segmentation_methods segmentation) {
    vector<object_segmentation::object_info> info;
    vector<object> pieces;
    object_segmentation::segment_for_correspondence(input, background, segmentation, info, &pieces);
    return pieces;
}

// Slide cells within each row or column, preserving their order and colors.
// Each cell moves by step, stopping early at the edge or the cell ahead.
// Read from the original cells and write into separate, already-cleared storage.
// Returns whether any cell moved. Selection and dimension limits belong to callers.
template <class At, class Put>
ARC_HOST_DEVICE inline bool slam_cells(int width, int height, bool rows, int step,
                                       At at, Put put) {
    const int lines = rows ? height : width;
    const int length = rows ? width : height;
    const int direction = step > 0 ? 1 : -1;
    bool moved = false;
    for (int line = 0; line < lines; ++line) {
        int wall = step > 0 ? length - 1 : 0;
        for (int i = 0; i < length; ++i) {
            const int source = step > 0 ? length - 1 - i : i;
            const int color = rows ? at(line, source) : at(source, line);
            if (color == TRANSPARENT) continue;
            int target = source + step;
            if (step > 0 ? target > wall : target < wall) target = wall;
            put(rows ? line : target, rows ? target : line, color);
            wall = target - direction;
            moved = moved || target != source;
        }
    }
    return moved;
}

// Every piece slammed its own cells against one side of its own box.
// Only held when sliding every piece that way redraws every output exactly.
struct piece_slam_rule {
    // True when the rows slide sideways, false when the columns slide up or down.
    bool rows = true;
    // +1 toward the right or the bottom, -1 toward the left or the top.
    int8_t step = 0;
    bool holds() const { return step != 0; }

    // Tries all four slide directions and keeps the one that redraws every output.
    static piece_slam_rule detect(const vector<example>& train,
                                  const correspondence_election& correspondence,
                                  int8_t input_background, int output_background) {
        // The slide happens in each piece's own box, and the cells it leaves are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !is_color(output_background) || !pairs_overlay(train)) return {};
        // Draws each output from its input's pieces slid that way, noting whether any
        // cell moved at all.
        auto draws_every_output = [&](bool rows, int step, bool& moved) {
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                const auto pieces = pieces_of(pair.input, input_background, correspondence.segmentation);
                drawn = grid(pair.output.width, pair.output.height);
                drawn.fill(output_background);
                bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                for (const object& piece : pieces) {
                    bool collision = false;
                    moved |= slam_cells(piece.summary.w, piece.summary.h, rows, step,
                        [&](int row, int col) { return (int)piece.cells[row][col]; },
                        [&](int row, int col, int color) {
                            row += piece.summary.y0;
                            col += piece.summary.x0;
                            // Two pieces landing on one cell would leave paint order to decide.
                            collision = collision || painted[row][col];
                            painted[row][col] = true;
                            drawn.data[row][col] = color;
                        });
                    if (collision) return false;
                }
                return true;
            });
        };
        for (const bool rows : {true, false})
            for (const int step : {1, -1}) {
                bool moved = false;
                if (draws_every_output(rows, step, moved) && moved)
                    return {rows, (int8_t)step};
            }
        return {};
    }
};

// Paint at an absolute grid position. Transparent cells leave the canvas alone.
// False means some paint fell outside; callers must discard that partial drawing.
inline bool paint_piece(grid& canvas, const object& piece, int row0, int col0) {
    for (int row = 0; row < piece.summary.h; ++row)
        for (int col = 0; col < piece.summary.w; ++col) {
            const int color = piece.cells[row][col];
            if (color == TRANSPARENT) continue;
            if (!canvas.contains(row0 + row, col0 + col)) return false;
            canvas.data[row0 + row][col0 + col] = color;
        }
    return true;
}

// THE PIECES LAID OUT AS ONE DIAGONAL CHAIN, each hung off the bottom-right
// corner of the one before it, overlapping it by the single cell they share.
// Only held when replaying that chain redraws every output exactly, which is
// what makes the claim checkable rather than a guess at what the layout means:
// a chain that does not come out cell for cell is not what happened.
struct overlap_corner_rule {
    bool held = false;
    bool holds() const { return held; }

    // Lays every input's pieces into the chain and keeps the claim only if that
    // drew each output on the nose.
    static overlap_corner_rule detect(const vector<example>& train,
                                      const correspondence_election& correspondence,
                                      int8_t input_background, int output_background) {
        // Each piece is hung off the last one's box, so the pieces have to be
        // measured in their own, and the cells no piece reaches are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !is_color(output_background) || train.empty()) return {};
        // OverlapCorner lays out as many objects as one list holds; a chain this
        // detector verified past that point is one the op would not draw.
        constexpr size_t most_the_op_lays_out = MAX_OBJECTS_IN_A_LIST;
        bool laid_any_piece_somewhere_new = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const auto pieces = pieces_of(pair.input, input_background, correspondence.segmentation);
            // One piece put in the corner says nothing about chaining.
            if (pieces.size() < 2 || pieces.size() > most_the_op_lays_out) return false;
            // The chain runs left to right, ties top to bottom -- the order the
            // op itself walks the objects in, so the reading and the word agree
            // about which piece is first.
            vector<const object*> chain;
            for (const object& piece : pieces) chain.push_back(&piece);
            stable_sort(chain.begin(), chain.end(), [](const object* one, const object* other) {
                return one->summary.x0 != other->summary.x0
                     ? one->summary.x0 < other->summary.x0
                     : one->summary.y0 < other->summary.y0;
            });
            // Tied corners leave the order to the op's list, which we can't see.
            for (size_t i = 1; i < chain.size(); ++i)
                if (same_corner(chain[i]->summary, chain[i - 1]->summary)) return false;
            drawn = grid(pair.output.width, pair.output.height);
            drawn.fill(output_background);
            // The head of the chain sits in the corner; each piece after it puts
            // its top-left on the last one's bottom-right and paints over it,
            // which is the paint order the closing Union gives them for free.
            int x = 0, y = 0;
            for (const object* piece : chain) {
                if (!paint_piece(drawn, *piece, y, x)) return false;
                laid_any_piece_somewhere_new = laid_any_piece_somewhere_new ||
                    x != piece->summary.x0 || y != piece->summary.y0;
                x += piece->summary.w - 1;
                y += piece->summary.h - 1;
            }
            return true;
        });
        // Pieces already sitting in the chain were never laid into one.
        return { redraws && laid_any_piece_somewhere_new };
    }
};

// Shift a non-empty box flush against another, without changing the other axis.
// The larger gap chooses the axis (ties go across); the nearer side chooses
// the shift (ties go left or up). Overlapping boxes have negative gaps.
// The result is {down, across}, in grid coordinates.
ARC_HOST_DEVICE inline std::pair<int, int> shift_to_touch(const cell_box& moving,
                                                         const cell_box& target) {
    const int left = target.col0 - moving.col1 - 1;
    const int right = target.col1 - moving.col0 + 1;
    const int above = target.row0 - moving.row1 - 1;
    const int below = target.row1 - moving.row0 + 1;
    int across = abs(left) <= abs(right) ? left : right;
    int down = abs(above) <= abs(below) ? above : below;
    const int column_gap = max(target.col0 - moving.col1, moving.col0 - target.col1);
    const int row_gap = max(target.row0 - moving.row1, moving.row0 - target.row1);
    if (column_gap >= row_gap) down = 0;
    else across = 0;
    return {down, across};
}

// PIECES SLID UP AGAINST THE ONES THAT STAY. One colour is already where it
// belongs and the other closes the gap until the two touch. That is what
// AlignToCanvas does, and this reads the same movement off the pairs: the still
// pieces are painted, the box their cells make is the whole of what the op gets
// to see of the canvas, and every other piece lands flush against that box,
// along whichever axis it stood further apart on. Only held when replaying it
// redraws every output exactly, which is what makes the claim checkable rather
// than a guess at where the pieces were headed.
//
// TWO COLOURS, BECAUSE A PICK NAMES A PIECE BY ITS COLOUR. The sentence has to
// say which pieces stay and which slide, and SelectColor is the only word that
// names a piece by what it is -- so a grid where three colours share the work
// is a movement with no sentence, and is not claimed here.
//
// shift_to_touch is shared with GPU execution, so both choose the same landing.
struct slide_to_touch_rule {
    int8_t stays  = -1;   // the colour that is already where it belongs
    int8_t slides = -1;   // the colour that closes the gap until they touch
    bool holds() const { return is_color(stays) && is_color(slides); }

    static slide_to_touch_rule detect(const vector<example>& train,
                                      const correspondence_election& correspondence,
                                      int8_t input_background, int output_background) {
        // The pieces slide in grid coordinates and the cells they leave behind
        // are ground, so the two grids have to lie over one another; and the
        // boxes measured here are the boxes Trim hands the op.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !is_color(output_background) || !pairs_overlay(train)) return {};

        // The colours the pieces wear: two of them, the same two in every pair,
        // or there is no pair of picks that can say this.
        int8_t colors[2] = { -1, -1 };
        for (const auto& pair : train) {
            vector<object_segmentation::object_info> info;
            object_segmentation::segment_for_correspondence(pair.input, input_background,
                                                            correspondence.segmentation, info, nullptr);
            // One list is all the op is ever handed, so a grid cut into more
            // pieces than that is one the word could not draw anyway.
            if (info.empty() || info.size() > MAX_OBJECTS_IN_A_LIST) return {};
            for (const auto& piece : info) {
                // A piece of several colours answers to both picks and to neither.
                if (!is_color(piece.color)) return {};
                if (piece.color == colors[0] || piece.color == colors[1]) continue;
                if (!is_color(colors[0]))      colors[0] = piece.color;
                else if (!is_color(colors[1])) colors[1] = piece.color;
                else return {};
            }
        }
        if (!is_color(colors[0]) || !is_color(colors[1])) return {};

        // Draws each output from its input's pieces, one colour standing still
        // and the other sliding up against it, noting whether anything moved at
        // all.
        auto draws_every_output = [&](int8_t stays, int8_t slides, bool& moved) {
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                vector<object_segmentation::object_info> info;
                vector<object> pieces;
                object_segmentation::segment_for_correspondence(pair.input, input_background,
                                                                correspondence.segmentation, info, &pieces);
                drawn = grid(pair.output.width, pair.output.height);
                drawn.fill(output_background);
                // WHAT THE OP SEES OF THE CANVAS is the box around everything
                // already painted, read once, before the first piece slides --
                // so what slides is measured against the pieces that stay, and
                // never against another piece that is still on its way.
                cell_box target;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != stays) continue;
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.y0, pieces[i].summary.x0)) return false;
                    const auto& box = pieces[i].summary;
                    target.include_cell(box.y0, box.x0);
                    target.include_cell(box.y0 + box.h - 1, box.x0 + box.w - 1);
                }
                // An empty canvas is nothing to touch, and the op answers that
                // by leaving every piece exactly where it stood.
                if (target.empty()) return false;
                bool anything_slid = false;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != slides) continue;
                    anything_slid = true;
                    const auto [down, across] = shift_to_touch(box_of(pieces[i].summary), target);
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.y0 + down,
                                     pieces[i].summary.x0 + across)) return false;
                    moved = moved || down != 0 || across != 0;
                }
                return anything_slid;
            });
        };

        // Which colour stays is the reading's to find, not to assume.
        for (const int which : { 0, 1 }) {
            bool moved = false;
            const int8_t stays = colors[which], slides = colors[1 - which];
            // Pieces already touching were never slid up against anything.
            if (draws_every_output(stays, slides, moved) && moved) return { stays, slides };
        }
        return {};
    }
};

// THE SEEDS THE OP WILL BE HANDED. The DSL fills its seed list off the input
// before a program runs (16_fixed_dsl.hpp), and this says the same list from
// this side of the fence, so a replay below can be checked against what the op
// will really do. The rule itself is not restated -- 4_objects.hpp's read_seeds
// is the one copy of it -- only the grid it is asked about.
struct marker_piece {
    object_summary summary;
    int8_t color = 0;
};

inline vector<marker_piece> seeds_the_op_will_see(const grid& input, int8_t never_changes) {
    vector<marker_piece> pieces;
    object_walk_scratch scratch;
    // -1 excludes nothing: the ground is a piece here like any other, and it is
    // dropped below for its colour, not for being the ground.
    each_object(input, -1, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int found) {
        if (found <= 0) return;
        const int8_t color = (int8_t)input.data[cell_row(cells[0])][cell_col(cells[0])];
        if (color == never_changes) return;
        pieces.push_back({ box, color });
    });
    vector<object_summary> boxes;
    for (const marker_piece& piece : pieces) boxes.push_back(piece.summary);
    vector<marker_piece> seeds;
    for (int which : read_seeds(boxes).seeds) seeds.push_back(pieces[which]);
    return seeds;
}

// Calls land(row, col) for each stamp of tpl along marker until it leaves the
// canvas or land returns false.
template <class Land>
ARC_HOST_DEVICE inline void project_along(const cell_box& tpl, const cell_box& marker,
                                          int width, int height, Land land) {
    const int across = (marker.col0 + marker.col1) / 2 - (tpl.col0 + tpl.col1) / 2;
    const int down   = (marker.row0 + marker.row1) / 2 - (tpl.row0 + tpl.row1) / 2;
    int col = tpl.col0, row = tpl.row0;
    if (across > 0)      col = marker.col0;
    else if (across < 0) col = marker.col1 - (tpl.width() - 1);
    if (down > 0)        row = marker.row0;
    else if (down < 0)   row = marker.row1 - (tpl.height() - 1);
    const int step_col = col - tpl.col0, step_row = row - tpl.row0;
    if (step_col == 0 && step_row == 0) return;
    for (int repeat = 1; repeat <= MAX_GRID_DIM; ++repeat) {
        const int at_col = tpl.col0 + repeat * step_col;
        const int at_row = tpl.row0 + repeat * step_row;
        if (at_col + tpl.width() <= 0 || at_col >= width ||
            at_row + tpl.height() <= 0 || at_row >= height) return;
        if (!land(at_row, at_col)) return;
    }
}

// THE TEMPLATE STAMPED ALONG EVERY MARKER. Each marker stands away from the
// template and so says a direction; the template lands flush against that
// marker, then repeats at the same step, wearing the marker's colour, until it
// has left the grid. The input stays underneath it all. Only held when
// replaying that redraws every output exactly, which is what makes the claim
// checkable rather than a guess at what the markers meant.
struct project_on_seeds_rule {
    bool held = false;
    bool holds() const { return held; }

    static project_on_seeds_rule detect(const vector<example>& train,
                                        const correspondence_election& correspondence,
                                        int8_t input_background) {
        // The template is handed over in the frame its cells make, which is
        // what the Trim in front of the word gives it.
        if (!boxed_by_its_cells(correspondence.segmentation) || train.empty()) return {};
        // ProjectOnSeeds hands back the copy of the input it draws on plus one
        // object per stamp, and they share one list, so the drawing runs out one
        // stamp before the list does.
        constexpr int most_the_op_stamps = MAX_OBJECTS_IN_A_LIST - 1;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            // Stamping paints over a copy of the input, so a canvas of another
            // size is a different drawing than the one replayed here.
            if (!pair.input.same_size(pair.output)) return false;
            const auto pieces = pieces_of(pair.input, input_background, correspondence.segmentation);
            if (pieces.empty()) return false;
            // The template is whichever piece SortBySize[0] puts first, and that
            // word ranks by cells -- not the box that picked the markers. The op
            // reads the two off different lists, so this has to as well.
            // Tied for most cells: the op's list order picks, which we can't see.
            const object* tpl = &pieces[0];
            if (pieces.size() > 1 && pieces[1].summary.count == tpl->summary.count) return false;

            const vector<marker_piece> markers = seeds_the_op_will_see(pair.input, input_background);
            if (markers.empty()) return false;

            drawn = pair.input;
            int stamped = 0;
            const cell_box tpl_box = box_of(tpl->summary);
            for (const marker_piece& marker : markers)
                project_along(tpl_box, box_of(marker.summary), drawn.width, drawn.height,
                              [&](int at_row, int at_col) {
                    if (stamped >= most_the_op_stamps) return false;
                    // A stamp hanging over the edge is drawn as far as it fits.
                    for (int r = 0; r < tpl->summary.h; ++r)
                        for (int c = 0; c < tpl->summary.w; ++c) {
                            if (tpl->cells[r][c] == TRANSPARENT) continue;
                            const int row = at_row + r, col = at_col + c;
                            if (!drawn.contains(row, col)) continue;
                            drawn.data[row][col] = marker.color;
                        }
                    ++stamped;
                    return true;
                });
            // Nothing stamped is the input handed straight back, which every
            // other reading already says without a word this long.
            return stamped > 0;
        });
        return { redraws };
    }
};

// WHAT BRIDGE DRAWS, said once for the op and for the reading that replays it.
// Each piece reaches for the nearest piece of its own colour whose centre shares
// its row or its column, and the box between the two is filled in that colour.
// `count` pieces in the op's own order: box_of_piece(i) is piece i's cell_box,
// color_of(i) the colour the op reads off its first cell (-1 for none).
// land(bar, colour) sees each bar in the order the op appends them, and says
// false once the list has no room for another. gpu_exec_bridge calls it, so a
// reading and the op cannot draw different bars.
template <class BoxOf, class ColorOf, class Land>
ARC_HOST_DEVICE inline void bridge_bars(int count, BoxOf box_of_piece, ColorOf color_of, Land land) {
    constexpr int most_pairs = 64;
    int16_t lows[most_pairs], highs[most_pairs];
    int pairs = 0;
    auto across = [&](int i) { const cell_box box = box_of_piece(i); return (box.col0 + box.col1) / 2; };
    auto down = [&](int i) { const cell_box box = box_of_piece(i); return (box.row0 + box.row1) / 2; };
    for (int i = 0; i < count; ++i) {
        const int8_t color = color_of(i);
        if (color < 0) continue;
        const int my_across = across(i), my_down = down(i);
        // Ties keep the first found, as the op's list has them.
        int partner = -1, nearest = 0x7FFF;
        for (int j = 0; j < count; ++j) {
            if (j == i || color_of(j) != color) continue;
            const int their_across = across(j), their_down = down(j);
            if (their_across != my_across && their_down != my_down) continue;
            const int distance = abs(their_across - my_across) + abs(their_down - my_down);
            if (distance < nearest) { nearest = distance; partner = j; }
        }
        if (partner < 0) continue;
        const int low = min(i, partner), high = max(i, partner);
        bool said = false;
        for (int p = 0; p < pairs && !said; ++p) said = lows[p] == low && highs[p] == high;
        if (said || pairs >= most_pairs) continue;
        lows[pairs] = (int16_t)low;
        highs[pairs] = (int16_t)high;
        ++pairs;
    }
    for (int p = 0; p < pairs; ++p) {
        const cell_box one = box_of_piece(lows[p]), other = box_of_piece(highs[p]);
        const int apart_across = abs(across(highs[p]) - across(lows[p]));
        const int apart_down = abs(down(highs[p]) - down(lows[p]));
        if (apart_across == 0 && apart_down == 0) continue;
        // The gap along the way they lie apart, as deep as the two together.
        cell_box bar;
        if (apart_across >= apart_down) {
            bar.col0 = min(one.col1, other.col1) + 1; bar.col1 = max(one.col0, other.col0) - 1;
            bar.row0 = min(one.row0, other.row0);     bar.row1 = max(one.row1, other.row1);
        } else {
            bar.row0 = min(one.row1, other.row1) + 1; bar.row1 = max(one.row0, other.row0) - 1;
            bar.col0 = min(one.col0, other.col0);     bar.col1 = max(one.col1, other.col1);
        }
        // Pieces that touch or overlap along that way leave no gap to fill.
        if (bar.empty()) continue;
        const int8_t color = color_of(lows[p]);
        if (!land(bar, color < 0 ? (int8_t)0 : color)) return;
    }
}

// THE GAP BETWEEN TWO PIECES OF ONE COLOUR, FILLED, and laid under what is
// already there, so a bar that crosses a wall goes behind it. That is what
// Bridge does, and this reads it off the pairs by doing it: the input's pieces,
// cut the way Split cuts them and in the order Split hands them over, each
// reaching for its nearest partner along a row or a column. Only held when
// replaying that redraws every output exactly.
//
// THE CUT IS THE ELECTED ONE WHEN IT CAN SAY THIS, and otherwise the first
// connected cut that can -- a rule about one way of cutting is read off that
// cut, elected or not. Colors is never asked: it hands one piece per colour,
// and a colour has no partner of its own to reach for. 070dd51e elects Colors
// and 22eb0ac0 the whole grid, and both are bridges under Split[1].
struct bridge_rule {
    bool held = false;
    segmentation_methods segmentation = segmentation_methods::unset;
    bool holds() const { return held; }

    static bridge_rule detect(const vector<example>& train,
                              const correspondence_election& correspondence,
                              int8_t input_background, int output_background) {
        // The bars fill cells that were ground, and the rest of the grid stays
        // where it was, so the two grids lie over one another.
        if (!is_color(output_background) || !pairs_overlay(train)) return {};
        vector<segmentation_methods> cuts{ correspondence.segmentation };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::connected_multicolor })
            if (cut != correspondence.segmentation) cuts.push_back(cut);
        for (const auto cut : cuts) {
            if (cut != segmentation_methods::connected_monochrome &&
                cut != segmentation_methods::connected_multicolor) continue;
            bool bridged = false;
            const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                // The pieces as Split hands them over: the one walk, in its order.
                vector<object> pieces;
                vector<int8_t> colors;
                object_walk_scratch scratch;
                each_object(pair.input, input_background,
                            cut == segmentation_methods::connected_multicolor
                                ? grouping::touching_any_color : grouping::touching_same_color,
                            scratch, [&](const object_summary& box, const int16_t* cells, int found) {
                    pieces.push_back(object::from_cells(pair.input, box, cells, found));
                    // The op reads a piece's colour off its first cell, which
                    // is the one the walk started from.
                    colors.push_back((int8_t)pair.input.data[cell_row(cells[0])][cell_col(cells[0])]);
                });
                // One list is all the op is handed.
                if (pieces.empty() || pieces.size() > MAX_OBJECTS_IN_A_LIST) return false;
                drawn = grid(pair.output.width, pair.output.height);
                drawn.fill(output_background);
                bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                for (const object& piece : pieces)
                    piece.each_cell([&](int row, int col, int8_t color) {
                        drawn.data[row][col] = color;
                        painted[row][col] = true;
                    });
                // The bars share the pieces' list, so they run out where it does,
                // and each is laid under whatever is already drawn.
                int room = MAX_OBJECTS_IN_A_LIST - (int)pieces.size();
                bridge_bars((int)pieces.size(),
                            [&](int which) { return box_of(pieces[which].summary); },
                            [&](int which) { return colors[which]; },
                            [&](const cell_box& bar, int8_t color) {
                    if (room-- <= 0) return false;
                    for (int row = bar.row0; row <= bar.row1; ++row)
                        for (int col = bar.col0; col <= bar.col1; ++col) {
                            if (!drawn.contains(row, col) || painted[row][col]) continue;
                            drawn.data[row][col] = color;
                            painted[row][col] = true;
                            bridged = true;
                        }
                    return true;
                });
                return true;
            });
            // No bar laid anywhere is the input handed straight back.
            if (redraws && bridged) return { true, cut };
        }
        return {};
    }
};

// Collects facts, rules, and elections about one puzzle.
struct inference {
    output_dimension_rule size;
    color_facts colors;
    shape_color_map_rule shape_color_map;
    unchanged_cells_facts unchanged;
    color_diff_facts color_diff;
    color_swap_candidates color_swaps;
    motif_continuation_rule motif_continuation;
    diagonal_continuation_rule diagonal_continuation;
    periodic_denoise_rule periodic_denoise;
    piece_slam_rule piece_slam;
    overlap_corner_rule overlap_corner;
    slide_to_touch_rule slide_to_touch;
    project_on_seeds_rule project_on_seeds;
    bridge_rule bridge;
    input_blocks_rule input_blocks;
    pane_split_facts pane_split;
    pane_mask_rule pane_mask;
    pane_pick_rule pane_pick;
    correspondence_election correspondence;
    background_election background;
};

// Asks every question about the training pairs, in order, and returns the answers.
inline inference infer(const vector<example>& train) {
    inference inf;
    inf.size = output_dimension_rule::infer_rule(train);
    inf.colors = color_facts::detect(train);  // rule-independent
    // Background: input-side and output-side elected independently by role,
    // ranked lists kept alive (see step 2 note on background_election).
    // The input side's verdict is what segmentation excludes.
    inf.background = elect_background(train);
    const int8_t ground_in = inf.background.input_elected();
    const int8_t ground_out = inf.background.output_elected();
    // by_size excludes the background, so it can only rank after the election.
    inf.colors.detect_output_colors_by_size(train, ground_in);
    // The frame before the figures: whether this input is really several grids
    // side by side. It needs the elected background to tell a wall from a blank
    // line, so it cannot run ahead of the election -- only ahead of everything
    // that would want to read the panes.
    inf.pane_split = pane_split_facts::detect(train, ground_in);
    // What the panes come to, once there are panes. It reads the masks, not the
    // colors, so a combination the pairs never showed is still one it can answer.
    inf.pane_mask = pane_mask_rule::detect(train, inf.pane_split, ground_in);
    // And when the answer is a pane rather than something made from the panes,
    // the only question left is which one. Asked after the mask, because making
    // a drawing is the stronger claim and should get first refusal.
    inf.pane_pick = pane_pick_rule::detect(train, inf.pane_split, ground_in);
    // Edge level of correspondence, under the elected background, with the way
    // of cutting the grid up chosen by what it manages to explain.
    inf.correspondence = select_segmentation(train, ground_in);
    inf.motif_continuation = motif_continuation_rule::detect(train, inf.correspondence);
    // The same continuation read along the diagonals. Asked after the motif
    // because a repeating box is the plainer claim and should get first refusal.
    inf.diagonal_continuation = diagonal_continuation_rule::detect(train, inf.correspondence, ground_in);
    // A noisy lattice cleaned by the vote. It reads the whole input, not the
    // pieces, so it needs nothing the election chose -- only the ground.
    inf.periodic_denoise = periodic_denoise_rule::detect(train, ground_in);
    inf.piece_slam = piece_slam_rule::detect(train, inf.correspondence, ground_in, ground_out);
    inf.overlap_corner = overlap_corner_rule::detect(train, inf.correspondence, ground_in, ground_out);
    // Pieces that slid up against the ones that stayed. Asked after the chain
    // because a layout that puts every piece somewhere new is the larger claim
    // and should get first refusal; this one leaves most of the grid alone.
    inf.slide_to_touch = slide_to_touch_rule::detect(train, inf.correspondence, ground_in, ground_out);
    inf.project_on_seeds = project_on_seeds_rule::detect(train, inf.correspondence, ground_in);
    // Gaps between pieces of one colour, filled. It reads its own cut when the
    // elected one cannot say it, so it does not wait on the election's verdict.
    inf.bridge = bridge_rule::detect(train, inf.correspondence, ground_in, ground_out);
    inf.input_blocks = input_blocks_rule::detect(train, inf.correspondence, ground_in, ground_out);

    // Dims is detector #0: its rule decides which questions even apply
    // (Backward Deduction step 1, applied to detectors instead of ops).
    // Identity = input and output overlay cell-for-cell, so the whole
    // pair-relation toolbox below is meaningful. Each detector re-asks the
    // same question itself (pairs_overlay), so this gate states the intent
    // and none of them depends on being called from here.
    if (inf.size.Chosen == output_dimension_rule::Identity) {
        inf.unchanged   = unchanged_cells_facts::detect(train);
        inf.color_diff  = color_diff_facts::detect(train);
        inf.color_swaps = color_swap_candidates::detect(train, inf.color_diff, ground_in);
        inf.shape_color_map = shape_color_map_rule::detect_body_recolor(train, ground_in);
        // Fallback: same shape->color table, but keyed on each object's own
        // shape and applied as an interior fill (ShapeToInteriorColorMap)
        // instead of a body recolor. Only run when the body-recolor
        // hypothesis didn't validate; the winner is recorded in claimed_by,
        // which is what tells the two ops apart downstream.
        if (!inf.shape_color_map.holds())
            inf.shape_color_map = shape_color_map_rule::detect_interior_fill(train, ground_in);
    }
    // Scale / Fixed / BoundingBox: no rule-specific detectors yet.
    // Scale could later reuse unchanged_cells through the ratio (cell<->block).
    return inf;
}

}  // namespace arc
