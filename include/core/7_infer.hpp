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
#include "8_vocabulary.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <numeric>
#include <optional>
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

    // Collects all basic color facts.
    static color_facts detect(const vector<example>& train) {
        color_facts facts;
        facts.detect_presence(train);
        facts.detect_output_colors(train);
        return facts;
    }

    // Records colors from most-used to least-used only when every output agrees
    // on the ranking.
    void detect_output_colors_by_size(const vector<example>& train) {
        output_colors_by_size_count = 0;
        if (train.empty()) return;

        // Biggest first; returns how many colors this output actually has.
        auto rank_for_output = [&](const grid& out, int8_t ranking[COLOR_COUNT]) -> int {
            int counts[COLOR_COUNT];
            count_colors(out, counts);
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
                                  const color_diff_facts& diff) {
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
                        if (was == TRANSPARENT) continue;   // in no piece to be swapped
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

// A stationary object's number of enclosed spaces determines its body color.
// Include unchanged objects and ground in the checks: a changed subset alone
// could endorse a key that also selects objects which must keep another color.
struct hole_color_rule {
    static constexpr int key_count = 5; // SelectObjectsWithHoles[0..4]
    array<int8_t, key_count> colors = { -1, -1, -1, -1, -1 };
    bool held = false;
    bool holds() const { return held; }

    static hole_color_rule detect(const vector<example>& train) {
        hole_color_rule rule;
        if (!pairs_overlay(train)) return {};
        object_walk_scratch scratch;
        bool changed = false;
        for (const auto& ex : train) {
            bool consistent = true;
            for (int r = 0; r < ex.input.height; ++r)
                for (int c = 0; c < ex.input.width; ++c)
                    if (ex.input.data[r][c] == TRANSPARENT && ex.output.data[r][c] != TRANSPARENT)
                        return {};
            each_object(ex.input, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    if (!consistent) return;
                    bool occupied[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                    int color = -1;
                    for (int i = 0; i < count; ++i) {
                        const int r = cell_row(cells[i]), c = cell_col(cells[i]);
                        occupied[r - box.y0][c - box.x0] = true;
                        const int target = ex.output.data[r][c];
                        if (!is_color(target) || (color >= 0 && color != target)) {
                            consistent = false;
                            return;
                        }
                        color = target;
                        changed = changed || target != ex.input.data[r][c];
                    }
                    const int key = object_hole_count(box.h, box.w,
                        [&](int r, int c) { return occupied[r][c]; });
                    if (key >= key_count || (rule.colors[key] >= 0 && rule.colors[key] != color)) {
                        consistent = false;
                        return;
                    }
                    rule.colors[key] = (int8_t)color;
                });
            if (!consistent) return {};
        }
        rule.held = changed;
        return rule;
    }
};

// Read just the pieces, in the same size-and-position order used by matching.
inline vector<object> pieces_of(const grid& input, segmentation_methods segmentation) {
    vector<object_segmentation::object_info> info;
    vector<object> pieces;
    object_segmentation::segment_for_correspondence(input, segmentation, info, &pieces);
    return pieces;
}

// THE PIECES A WORD IS HANDED, as the cut's word hands them over, cut once per
// pair and read by every replay below: where each lies, and whether the word's
// pick takes it -- the questions the partner helpers ask (each_partner and kin,
// 1_readings.hpp), asked here as the ops ask them of their objects, so a replay
// and the op it replays cannot find different partners or fits.
struct partner_pieces {
    const vector<object>& pieces;
    vector<cell_box> boxes;
    vector<bool> picked;
    // The pieces of a cut, the ones `marks` names picked.
    partner_pieces(const vector<object>& cut, vector<bool> marks)
        : pieces(cut), boxes(cut.size()), picked(std::move(marks)) {
        for (size_t i = 0; i < pieces.size(); ++i) boxes[i] = pieces[i].painted_box();
    }
    // The pieces of a cut, the ones `pick` takes picked (picked_pieces, below).
    partner_pieces(const vector<object>& cut, const piece_pick& pick);
    // What the partner helpers ask.
    int count() const { return (int)pieces.size(); }
    cell_box box(int i) const { return boxes[i]; }
    bool painted(int i, int row, int col) const { return pieces[i].holds(row, col); }
    bool is_picked(int i) const { return picked[i]; }
    int color(int i) const { return pieces[i].first_color(); }
    // Whether piece i holds a cell of `color`.
    bool holds(int i, int color) const { return (palette(pieces[i]) >> color & 1) != 0; }
    bool every_piece_picked() const { return ranges::all_of(picked, [](bool one) { return one; }); }
    // Piece i's one partner by `how` among the pieces not picked, or -1.
    int partner(partner_by how, int i) const { return partner_of(how, i, *this); }
    // Whether piece i has any partner by `how` among them, one or several.
    bool partnered(partner_by how, int i) const {
        bool any = false;
        each_partner(how, i, *this, [&](int) { any = true; });
        return any;
    }
    // Whether piece i would fit into a gap of its one partner by `how`.
    bool fits(partner_by how, int i) const {
        const int other = partner(how, i);
        return other >= 0 && would_fit(i, other, *this);
    }
    // Piece i's one partner of `kind` (partner_of_kind) among the pieces not picked, or -1.
    int partner_of_kind(int kind, int i) const { return arc::partner_of_kind(kind, i, *this); }
    // HOW FAR PIECE i GOES, as the partner moves take it (the kernel's AlignTo and
    // MoveToward, through the same helpers), or false when it has no one partner.
    // Level with its partner's rows -- or, with `columns`, columns:
    bool goes_level(int i, bool columns, partner_by how, heading_step& by) const {
        const int other = partner(how, i);
        if (other < 0) return false;
        by = columns ? heading_step{ 0, boxes[other].col0 - boxes[i].col0 }
                     : heading_step{ boxes[other].row0 - boxes[i].row0, 0 };
        return true;
    }
    // Toward its partner of the kind `value` % PARTNER_KINDS names, by what
    // `value` / PARTNER_KINDS names (move_toward), on a canvas `width` by `height`:
    bool goes_toward(int i, int value, int width, int height, heading_step& by) const {
        const int other = partner_of_kind(value % PARTNER_KINDS, i);
        if (other < 0) return false;
        by = move_toward((toward_by)(value / PARTNER_KINDS), i, other, *this, width, height);
        return true;
    }
    // The color piece i takes from its partners by `how`, or -1.
    int color_from(partner_by how, int i) const { return partner_color(how, i, *this); }
};

// Maps shapes to colors (rejected if any shape needs two different colors).
struct shape_color_map_rule : shape_table {
    // WHO FILLED IT. The readers share this storage and mean different things by
    // it: body_recolor repaints a piece in the color its shape names
    // (ShapeColorMap), interior_fill paints the hole inside one (ShapeTo-
    // InteriorColorMap). A consumer that cannot tell them apart is feeding one
    // op the other's table, so the table says which it is.
    enum : int8_t { claimed_by_nobody, body_recolor, interior_fill } claimed_by
        = claimed_by_nobody;
    // WHAT THE BODY RECOLOR WAS READ OFF, pair by pair: the elected reading's
    // input pieces it took as the key and the target, numbered as the readout
    // numbers them, and the color the reading carried the target into.
    struct read_in_pair {
        int16_t key = -1, target = -1;
        int8_t became = -1;
    };
    vector<read_in_pair> read;

    // A SMALL KEY'S SHAPE NAMES THE COLOR THE LARGEST PIECE IS REPAINTED, AND
    // THE KEY IS USED UP (009d5c81), said in three words (9_sentences.hpp): the
    // smallest piece takes the color its own shape names (ShapeColorMap), the
    // largest takes the color of its nearest piece, the key (ColorFrom), and the
    // smallest goes (DeleteSelection). The elected reading already says which
    // piece kept its shape and what color it came out in, so what is asked here
    // is only whether those words would repaint that piece, and what the key's
    // shape says -- never, a second time, what was recolored. The picks take the
    // smallest and the largest by painted cells, so each must be the only one
    // its size, or the order the pieces arrive in would choose, and the key
    // must be the largest one's one nearest piece. Reading the key consumes it,
    // so the reading must have it going nowhere in every pair. A target that
    // keeps its own color is a key naming it; one pair at least has to change
    // something.
    static shape_color_map_rule detect_body_recolor(const vector<example>& train,
                                                    const correspondence_election& correspondence) {
        const auto& pairs = correspondence.chosen_pairs();
        if (!pairs_overlay(train) || pairs.size() != train.size() ||
            !boxed_by_its_cells(correspondence.segmentation)) return {};
        shape_color_map_rule rule;
        bool changed = false;
        for (size_t which = 0; which < train.size(); ++which) {
            const auto& pieces = pairs[which].input_info;
            auto cells_of = [&](int piece) { return (int)pieces[piece].summary.count; };
            int key = -1, target = -1;
            for (int piece = 0; piece < (int)pieces.size(); ++piece) {
                if (key < 0 || cells_of(piece) < cells_of(key)) key = piece;
                if (target < 0 || cells_of(piece) > cells_of(target)) target = piece;
            }
            auto alone_in_its_size = [&](int pick) {
                for (int piece = 0; piece < (int)pieces.size(); ++piece)
                    if (piece != pick && cells_of(piece) == cells_of(pick)) return false;
                return true;
            };
            if (key < 0 || key == target || !alone_in_its_size(key) ||
                !alone_in_its_size(target) || !is_color(pieces[target].color) ||
                pieces[key].has_partner) return {};

            // What the reading carried the target into, one color for all of it.
            int8_t became = -1;
            for (const auto& edge : pairs[which].edges) {
                if (!edge.from_input || edge.from != target || !edge.same_shape) continue;
                const int8_t worn = edge.exact ? pieces[target].color : edge.recolored_to(pieces[target].color);
                if (!is_color(worn) || (became >= 0 && became != worn)) return {};
                became = worn;
            }
            if (became < 0) return {};
            changed = changed || became != pieces[target].color;

            const auto cut = pieces_of(train[which].input, correspondence.segmentation);
            if (cut.size() != pieces.size()) return {};
            // The largest takes its color from its one nearest piece, as ColorFrom
            // finds it among the pieces not picked: that must be the key.
            vector<bool> largest(cut.size(), false);
            largest[(size_t)target] = true;
            if (partner_pieces(cut, largest).partner(partner_by::nearest, target) != key) return {};
            if (!rule.learn(key_of(shape_key::shape, cut[key]), became)) return {};
            rule.read.push_back({ (int16_t)key, (int16_t)target, became });
        }
        if (!changed) return {};
        rule.claimed_by = shape_color_map_rule::body_recolor;
        return rule;
    }

    // Learns which color fills the hollow inside of each shape. Shares storage with
    // detect_body_recolor, so infer() runs this only when that one finds nothing.
    // A table that only ever fills with the ground paints nothing, and as the
    // word for a recolor it stood in for changes it never makes -- so one shape
    // at least has to be filled with a color the ground is not.
    static shape_color_map_rule detect_interior_fill(const vector<example>& train) {
        shape_color_map_rule rule;
        if (!pairs_overlay(train)) return {};
        bool fills_something = false;

        object_walk_scratch scratch;
        bool occupied[MAX_GRID_CELLS] = {};
        bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
        for (const auto& ex : train) {
            const grid& in = ex.input;
            const grid& out = ex.output;
            bool consistent = true;
            each_object(in, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                if (!consistent) return;
                const int left = box.x0, top = box.y0, width = box.w, height = box.h;
                fill_n(occupied, width * height, false);
                for (int i = 0; i < count; ++i)
                    occupied[(cell_row(cells[i]) - top) * width + cell_col(cells[i]) - left] = true;
                mark_outside(height, width,
                    [&](int row, int col) { return occupied[row * width + col]; }, outside);

                // Fill color = the output color on interior cells whose INPUT was
                // TRANSPARENT — the cells the fill actually paints. Cells that are
                // non-TRANSPARENT in the input (inside-noise like a marker pixel)
                // survive unchanged, so they are skipped. Must be one color across
                // the painted interior.
                int fill = -1; bool any_painted = false;
                for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    int id = row * width + col;
                    if (occupied[id] || outside[row][col]) continue;         // not interior
                    int in_row = top + row, in_col = left + col;
                    if (in.data[in_row][in_col] != TRANSPARENT) continue;  // inside-noise
                    int out_color = out.data[in_row][in_col];
                    if (!is_color(out_color) || (any_painted && out_color != fill)) {
                        consistent = false;
                        return;
                    }
                    fill = out_color;
                    any_painted = true;
                }
                if (!any_painted) return;

                const uint32_t object_shape = canonical_shape_hash(width, height,
                    [&](int row, int col) { return occupied[row * width + col]; });
                consistent = rule.learn(object_shape, (int8_t)fill);
                fills_something = fills_something || fill != TRANSPARENT;
            });
            if (!consistent) return {};
        }
        if (!fills_something) return {};
        if (rule.holds()) rule.claimed_by = shape_color_map_rule::interior_fill;
        return rule;
    }

    // THE TABLE'S KEY FOR A PIECE, of the kind ShapeColorMap's value names: its
    // shape, whichever way round it lies, or its symmetry (shape_key_of, the key
    // the ops look up).
    static uint32_t key_of(shape_key kind, const object& piece) {
        return shape_key_of((int)kind, piece.summary.w, piece.summary.h,
                            [&](int row, int col) { return piece.cells[row][col]; });
    }
    // The color the table names for a piece's key, as the ops look it up.
    using shape_table::color_of;
    int color_of(shape_key kind, const object& piece) const { return color_of(key_of(kind, piece)); }

    // A KEY THE PAIRS SHOW: a picked piece's key, the one color it came out in,
    // the pairs it is seen in, a bit each, and whether an input it is seen in
    // shows that color anywhere.
    struct seen_key { uint32_t key; int8_t color; uint32_t pairs; bool shown; };
    // The colors a grid shows, a bit each.
    static uint16_t colors_shown(const grid& one) {
        uint16_t shown = 0;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (is_color(one.data[row][col])) shown = (uint16_t)(shown | 1u << one.data[row][col]);
        return shown;
    }
    // Notes that `key` came out in `color` in pair `which`, whose input shows
    // the colors `shown`; false where it came out in another color before,
    // which no table says.
    static bool see(vector<seen_key>& seen, uint32_t key, int8_t color, size_t which, uint16_t shown) {
        const uint32_t pair = 1u << min<size_t>(which, 31);
        const bool shows = (shown >> color & 1) != 0;
        const auto known = ranges::find(seen, key, &seen_key::key);
        if (known == seen.end()) {
            seen.push_back({ key, color, pair, shows });
            return true;
        }
        known->pairs |= pair;
        known->shown = known->shown || shows;
        return known->color == color;
    }
    // THE TABLE THE PAIRS SHOW, or one that does not hold. A key seen in one pair
    // only is that pair's own piece, not a rule (b230c067's twins, a new pair in
    // every pair), so the table names the keys seen in two pairs at least -- and
    // beside them a key seen in one pair whose color only the table can say: no
    // other key comes out in it, so the color says the key and not something
    // else about the piece (b230c067's twins all come out 1, whatever their
    // shape), and its input shows that color nowhere, so no piece wearing it
    // says it instead (ef26cbf6's and b7256dcd's markers; 27a28665's plus, 6,
    // beside three shapes each seen twice). The keys seen in one pair and not
    // named may still all come out in one color, itself seen in two pairs at
    // least, and that is then the color of every key the table does not name
    // (25d8a9c8: a full row 5, any other shape 0, though its square and its L
    // are each seen in one pair). A table naming no key at all is one color for
    // everything, which is no table.
    static shape_color_map_rule read_off(const vector<seen_key>& seen) {
        auto in_two_pairs = [](uint32_t pairs) { return popcount(pairs) >= 2; };
        const bool repeats = ranges::any_of(seen, [&](const seen_key& one) { return in_two_pairs(one.pairs); });
        auto named = [&](const seen_key& one) {
            return in_two_pairs(one.pairs) ||
                   (repeats && !one.shown && ranges::count(seen, one.color, &seen_key::color) == 1);
        };
        // The one color the keys not named come out in: -2 until one is read,
        // -1 once two disagree.
        int lone = -2;
        uint32_t wearing = 0;
        for (const seen_key& one : seen)
            if (!named(one)) lone = lone == -2 || lone == one.color ? one.color : -1;
        for (const seen_key& one : seen)
            if (one.color == lone) wearing |= one.pairs;
        if (lone == -1 || (lone >= 0 && !in_two_pairs(wearing))) return {};
        shape_color_map_rule table;
        table.otherwise = (int8_t)max(lone, -1);
        // Every key named that the color of the rest does not already say.
        for (const seen_key& one : seen)
            if (named(one) && one.color != table.otherwise && !table.learn(one.key, one.color))
                return {};
        if (table.holds()) table.claimed_by = body_recolor;
        return table;
    }
};
static_assert(row_of(node_type::ShapeColorMap).high + 1 == (int)shape_key::count,
              "ShapeColorMap takes one value for each key a learned table knows a piece by");

// A repeating pattern carried across the whole output, with its colors remapped.
struct motif_continuation_rule {
    bool held = false;
    recoloring color_map;
    bool holds() const { return held; }

    // Checks that the input is one tiling pattern and reads the recolor off the
    // output. Read off the repeating block's own best reading, elected or not:
    // that the grid repeats is this rule's claim, not a way of seeing objects.
    static motif_continuation_rule detect(const vector<example>& train,
                                          const correspondence_election& correspondence) {
        motif_continuation_rule rule;
        const auto* motif = correspondence.best_reading_of(segmentation_methods::minimal_repeating_set);
        if (train.empty() || !motif || motif->pairs.size() != train.size()) return rule;
        for (size_t i = 0; i < train.size(); ++i) {
            const auto& pair = motif->pairs[i];
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
                                             const correspondence_election& correspondence) {
        // The cycle is read off the cells where they sit, so the pieces have to
        // be the input's own cells in the input's own frame.
        if (train.empty() || !boxed_by_its_cells(correspondence.segmentation)) return {};
        auto redraws_every_output = [&](int8_t lean) {
            bool carried_something = false;
            for (const auto& pair : train) {
                int counts[COLOR_COUNT];
                count_colors(pair.input, counts, TRANSPARENT);
                int colors = 0;
                for (int color = 0; color < COLOR_COUNT; ++color) colors += counts[color] > 0;
                if (colors < 2) return false;
                auto seat_of = [&](int row, int col) { return diagonal_seat(row, col, lean, colors); };
                int8_t cycle[COLOR_COUNT];
                for (int seat = 0; seat < colors; ++seat) cycle[seat] = -1;
                bool agrees = true;
                walk_colored_cells(pair.input, TRANSPARENT, [&](int row, int col, int color) {
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

// A grid that size with nothing on it yet.
inline grid empty_grid(int width, int height) {
    grid made(width, height);
    made.fill(TRANSPARENT);
    return made;
}

// Paint at an absolute grid position. Transparent cells leave the canvas alone.
// Laid under rather than over, as Underlay lays it, a cell only fills what
// nothing has painted yet. Unless clipping is requested, false means some paint
// fell outside; callers must discard that partial drawing.
inline bool paint_piece(grid& canvas, const object& piece, int row0, int col0,
                        bool over = true, bool clip = false) {
    for (int row = 0; row < piece.summary.h; ++row)
        for (int col = 0; col < piece.summary.w; ++col) {
            const int color = piece.cells[row][col];
            if (color == TRANSPARENT) continue;
            if (!canvas.contains(row0 + row, col0 + col)) {
                if (clip) continue;
                return false;
            }
            if (over || canvas.data[row0 + row][col0 + col] == TRANSPARENT)
                canvas.data[row0 + row][col0 + col] = color;
        }
    return true;
}

// THE CANVAS A PROGRAM PAINTS ON, sized and closed as the kernel does it: the
// size rule's box, or under BoundingBox the largest grid there is, cropped at
// the end to what was painted. `paint` lays its cells on it, TRANSPARENT where
// nothing is yet, or says false. What nothing painted stays empty -- what a
// plan fills last, and what an output with a color there refuses when the
// drawing is compared to it.
template <class Paint>
bool draw_on_canvas(const grid& input, const output_dimension_rule& size, grid& drawn, Paint paint) {
    const bool closes = size.Chosen == output_dimension_rule::BoundingBox;
    int width = MAX_GRID_DIM, height = MAX_GRID_DIM;
    if (!size.canvas_size_for(input.width, input.height, width, height)) return false;
    grid canvas = empty_grid(width, height);
    if (!paint(canvas)) return false;
    const cell_box kept = fit_frame((int)(closes ? fit_to::painted : fit_to::canvas), input.width, input.height, canvas);
    if (kept.empty()) return false;
    drawn = grid(kept.width(), kept.height());
    for (int r = 0; r < drawn.height; ++r)
        for (int c = 0; c < drawn.width; ++c) drawn.data[r][c] = canvas.data[kept.row0 + r][kept.col0 + c];
    return true;
}
static_assert(row_of(node_type::Fit).high + 1 == (int)fit_to::count,
              "Fit takes one value for each target it stretches to");

// A REPEATING PICTURE WITH NOISE ON IT, AND THE OUTPUT IS THE PICTURE CLEAN.
// The input is a lattice of copies that noise has scribbled over, so no cut of
// it finds the copies -- 0607ce86 cuts into rubble every way it is read. What
// finds them is the vote: each cell takes the majority of the cells whole
// periods away. Asked by running that vote, VoteDenoise's own arithmetic, on
// every input under the ground the op reads off it (denoise_ground: empty where
// a plan deleted it, else its most common color), holding only where it draws
// every output exactly. The ground is kept in, because it is the evidence the
// period search is built on (notes/4 - To Do/VoteDenoise Needs The Background.md).
//
// AND IT HAS TO CLEAN SOMETHING. An output the input already was is a grid
// that happens to repeat, which says nothing about noise.
struct periodic_denoise_rule {
    bool held = false;
    bool holds() const { return held; }

    static periodic_denoise_rule detect(const vector<example>& train) {
        if (train.empty()) return {};
        bool cleaned_something = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const grid& input = pair.input;
            if (!input.same_size(pair.output)) return false;
            drawn = input;
            // The ground the op reads off the input, as the op reads it.
            const int8_t background = denoise_ground(input.width, input.height,
                [&](int r, int c) { return (int8_t)input.data[r][c]; });
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

// Every foreground color layer grows rays from its connected pieces. Colors
// are selected by cell-count rank, so the same reading survives recoloring.
// Branch order is inferred by peeling matching layers off the output.
struct ray_extension_rule {
    enum : int8_t { Orthogonal, Diagonal };
    struct branch {
        int8_t rank = 0; // ascending cell count
        int8_t direction = Orthogonal;
        int8_t radius = 2;
        bool fill_interior = false;
        bool stamp_body = false;
    };
    using ranked_colors = vector<int8_t>;
    struct candidate {
        branch how;
        vector<grid> ink;
    };
    vector<branch> branches;
    bool holds() const { return !branches.empty(); }

    static bool rank_colors(const grid& input, ranked_colors& ranked) {
        int counts[COLOR_COUNT];
        count_colors(input, counts);
        ranked.clear();
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (counts[color]) ranked.push_back((int8_t)color);
        sort(ranked.begin(), ranked.end(), [&](int8_t x, int8_t y) { return counts[x] < counts[y]; });
        // Equal sizes do not establish a color-independent rank.
        for (size_t i = 1; i < ranked.size(); ++i)
            if (counts[ranked[i - 1]] == counts[ranked[i]]) return false;
        return !ranked.empty();
    }

    static bool draw_layer(const grid& input, int8_t color, const branch& how, grid& ink) {
        grid layer(input.width, input.height);
        ink = grid(input.width, input.height);
        cell_box bounds;
        for (int row = 0; row < input.height; ++row)
            for (int col = 0; col < input.width; ++col) {
                layer.data[row][col] = TRANSPARENT;
                ink.data[row][col] = TRANSPARENT;
                if (input.data[row][col] == color) {
                    layer.data[row][col] = color;
                    bounds.include_cell(row, col);
                }
            }
        if (bounds.empty()) return false;

        if (how.fill_interior) {
            bool outside[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            mark_outside(bounds.height(), bounds.width(),
                         [&](int row, int col) {
                             return layer.data[bounds.row0 + row][bounds.col0 + col] == color;
                         }, outside);
            for (int row = 0; row < bounds.height(); ++row)
                for (int col = 0; col < bounds.width(); ++col)
                    if (!outside[row][col])
                        layer.data[bounds.row0 + row][bounds.col0 + col] = color;
        }

        object_walk_scratch scratch{};
        bool drew = false;
        each_object(layer, TRANSPARENT, grouping::touching_same_color, scratch,
                    [&](const object_summary& box, const int16_t* cells, int count) {
            grid object = empty_grid(box.w, box.h);
            for (int i = 0; i < count; ++i) {
                const int row = cell_row(cells[i]), col = cell_col(cells[i]);
                object.data[row - box.y0][col - box.x0] = color;
                if (how.stamp_body) { ink.data[row][col] = color; drew = true; }
            }
            const auto seed = nearest_ray_seed(box.h, box.w, (box.h - 1) / 2, (box.w - 1) / 2,
                [&](int row, int col) { return object.data[row][col]; });
            each_extended_ray(seed, how.radius, how.direction == Diagonal,
                [&](int down, int across) {
                    const int row = box.y0 + seed.row + down, col = box.x0 + seed.col + across;
                    if (ink.contains(row, col)) { ink.data[row][col] = color; drew = true; }
                });
        });
        return drew;
    }

    static vector<candidate> options_for(const vector<example>& train,
                                         const vector<ranked_colors>& ranked, int8_t rank) {
        vector<candidate> options;
        const auto orthogonal = row_of(node_type::ExtendOrthogonal);
        const auto diagonal = row_of(node_type::ExtendDiagonal);
        for (int radius = min(orthogonal.low, diagonal.low);
             radius <= max(orthogonal.high, diagonal.high); ++radius) {
            for (int direction = Orthogonal; direction <= Diagonal; ++direction) {
                if (!takes_value(direction == Orthogonal ? node_type::ExtendOrthogonal
                                                        : node_type::ExtendDiagonal, radius)) continue;
                for (int fill = 0; fill <= 1; ++fill)
                    for (int stamp = 0; stamp <= 1; ++stamp) {
                        candidate trying{ { rank, (int8_t)direction, (int8_t)radius,
                                            fill != 0, stamp != 0 }, {} };
                        for (size_t i = 0; i < train.size(); ++i) {
                            grid ink;
                            if (!draw_layer(train[i].input, ranked[i][rank], trying.how, ink)) break;
                            trying.ink.push_back(std::move(ink));
                        }
                        if (trying.ink.size() == train.size() &&
                            none_of(options.begin(), options.end(), [&](const candidate& other) {
                                return other.ink == trying.ink;
                            })) options.push_back(std::move(trying));
                    }
            }
        }
        return options;
    }

    // A remaining color's topmost layer must explain every visible cell of
    // that color, and may paint elsewhere only beneath an already peeled layer.
    // This one check handles every position in the compositing order.
    static bool peel(const candidate& layer, const vector<example>& train,
                     const vector<ranked_colors>& ranked, const vector<grid>& covered,
                     vector<grid>& next) {
        next = covered;
        for (size_t i = 0; i < train.size(); ++i) {
            const int color = ranked[i][layer.how.rank];
            for (int row = 0; row < train[i].output.height; ++row)
                for (int col = 0; col < train[i].output.width; ++col) {
                    if (covered[i].data[row][col]) continue;
                    const bool painted = layer.ink[i].data[row][col] != TRANSPARENT;
                    const int output = train[i].output.data[row][col];
                    if (painted && output != color) return false;
                    // A cell of the color the layer did not paint is one it leaves unsaid.
                    if (output == color && !painted) return false;
                    if (painted) next[i].data[row][col] = 1;
                }
        }
        return true;
    }

    static ray_extension_rule detect(const vector<example>& train) {
        if (train.empty() || !pairs_overlay(train)) return {};
        vector<ranked_colors> ranked(train.size());
        vector<grid> covered;
        for (size_t i = 0; i < train.size(); ++i) {
            const auto& pair = train[i];
            if (!rank_colors(pair.input, ranked[i]) ||
                ranked[i].size() != ranked[0].size()) return {};
            for (int row = 0; row < pair.output.height; ++row)
                for (int col = 0; col < pair.output.width; ++col) {
                    const int color = pair.output.data[row][col];
                    if (color != TRANSPARENT && find(ranked[i].begin(), ranked[i].end(), color) == ranked[i].end())
                        return {};
                }
            covered.emplace_back(pair.output.width, pair.output.height);
        }
        vector<vector<candidate>> options;
        for (int rank = 0; rank < (int)ranked[0].size(); ++rank) {
            options.push_back(options_for(train, ranked, (int8_t)rank));
            if (options.back().empty()) return {};
        }
        ray_extension_rule found;
        auto choose = [&](auto&& self, unsigned used, const vector<grid>& mask) -> bool {
            if (found.branches.size() == options.size()) return true;
            for (size_t rank = 0; rank < options.size(); ++rank) {
                if (used & (1u << rank)) continue;
                for (const auto& option : options[rank]) {
                    vector<grid> next;
                    if (!peel(option, train, ranked, mask, next)) continue;
                    found.branches.push_back(option.how);
                    if (self(self, used | (1u << rank), next)) return true;
                    found.branches.pop_back();
                }
            }
            return false;
        };
        if (!choose(choose, 0, covered)) return {};
        reverse(found.branches.begin(), found.branches.end());
        return found;
    }
};

// WHAT A CUT IS ALLOWED TO FOLLOW. Where the cut falls -- which axis, which
// color, how many panes -- is read off the grid; these are only the permissions,
// the way degrees_of_freedom holds them for matching.
struct split_freedom {
    // A full-span line of one color: a wall.
    bool wall = true;
    // A full-span line of empty cells, where the ground was: the same shape
    // of evidence, thinner.
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
        // Its box on the grid.
        cell_box box() const { return { top, left, top + height - 1, left + width - 1 }; }
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
    static void readings_of(const grid& one, split_freedom allowed,
                            vector<pair<pane_cut, vector<pane>>>& found) {
        found.clear();
        vector<int> row_lines, col_lines;
        vector<pane> panes;
        // A color that is absent offers no full line, so it costs nothing to ask.
        // A line with nothing on it is blank, as a line of the ground is: after a
        // plan deleted the ground, that is what its lines are.
        for (int color = TRANSPARENT; color < COLOR_COUNT; ++color) {
            const int8_t cut = color == TRANSPARENT ? Blank : Wall;
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
            const int line = cut == Wall ? wall_of(one) : color;
            if (!is_color(line)) return false;
            full_lines(one, line, row_lines, col_lines);
            rows = spans_between(one.height, row_lines);
            cols = spans_between(one.width, col_lines);
        }
        if ((int)rows.size() != down || (int)cols.size() != across) return false;
        return lay_out(rows, cols, true, found);
    }

    // The wall's color in this grid: the one the pairs agreed on, or when they
    // agreed only on the cut, the one color whose full lines
    // cut this grid the same way. -1 when none does, or several do.
    int wall_of(const grid& one) const {
        if (cut != Wall) return -1;
        if (is_color(color)) return color;
        int wall = -1;
        vector<int> row_lines, col_lines;
        vector<pane> panes;
        for (int line = 0; line < COLOR_COUNT; ++line) {
            full_lines(one, line, row_lines, col_lines);
            const auto rows = spans_between(one.height, row_lines);
            const auto cols = spans_between(one.width, col_lines);
            if ((int)rows.size() != down || (int)cols.size() != across ||
                !lay_out(rows, cols, true, panes)) continue;
            if (wall >= 0) return -1;
            wall = line;
        }
        return wall;
    }

    // Each grid's walls with their color left out, so the pairs can agree on a
    // cut whose color is each grid's own. A grid that two such walls cut the
    // same way has no one wall, so neither is kept.
    static void without_wall_colors(vector<pair<pane_cut, vector<pane>>>& found) {
        for (auto& [named, panes] : found)
            if (named.cut == Wall) named.color = -1;
        vector<pair<pane_cut, vector<pane>>> kept;
        for (const auto& one : found)
            if (ranges::count_if(found, [&](const auto& other) { return other.first == one.first; }) == 1)
                kept.push_back(one);
        found = std::move(kept);
    }

    // The same cut in every pair, wall color and all. Failing that, the same cut
    // with each grid naming its own wall: four areas split by one line each way
    // are the rule, whatever color the lines are drawn in.
    static pane_split_facts detect(const vector<example>& train,
                                   split_freedom allowed = {}) {
        pane_split_facts facts = agreed_on(train, allowed, false);
        if (!facts.holds()) facts = agreed_on(train, allowed, true);
        return facts;
    }

    // Reads every training input, keeps the readings they all agree on, and holds
    // only when exactly one of those survives the precedence.
    static pane_split_facts agreed_on(const vector<example>& train, split_freedom allowed,
                                      bool own_walls) {
        if (train.empty()) return {};
        vector<pair<pane_cut, vector<pane>>> found;
        vector<pane_cut> agreed;
        for (size_t i = 0; i < train.size(); ++i) {
            readings_of(train[i].input, allowed, found);
            if (own_walls) without_wall_colors(found);
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
            readings_of(train_pair.input, allowed, found);
            if (own_walls) without_wall_colors(found);
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
    // otherwise paint is laid down; or the cell is left empty, where the output
    // has nothing because a plan fills it last. Exactly one of the three is set.
    struct answer {
        int8_t paint = -1;
        int8_t copy_pane = -1;
        bool leaves_empty = false;
        bool said() const { return paint >= 0 || copy_pane >= 0 || leaves_empty; }
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
                       int r, int c, int pose) {
        int mask = 0;
        for (int p = 0; p < (int)boxes.size(); ++p)
            if (shown(boxes, p, input, r, c, pose) != TRANSPARENT) mask |= 1 << p;
        return mask;
    }

    // Reads every training pair under one pose and says what each mask means,
    // or says nothing when two cells with the same mask disagree beyond repair.
    static pane_mask_rule read_with(const vector<example>& train,
                                    const pane_split_facts& split, int pose) {
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
        // What each mask laid first -- a color, or nothing -- and whether every
        // later cell of that mask laid the same.
        vector<uint8_t> painted((size_t)1 << panes, 0), agreed((size_t)1 << panes, 1);
        vector<int8_t> laid((size_t)1 << panes, TRANSPARENT);
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
                    const int mask = mask_at(boxes, input, r, c, pose);
                    const int now = output.data[r][c];
                    for (int p = 0; p < panes; ++p)
                        if (shown(boxes, p, input, r, c, pose) != now)
                            may_copy[mask] &= (uint8_t)~(1u << p);
                    if (!painted[mask]) {
                        painted[mask] = 1;
                        laid[mask] = (int8_t)now;
                    } else if (laid[mask] != now) {
                        agreed[mask] = 0;
                    }
                }
        }
        // A mask that always laid the same color, or always left the cell empty,
        // says so; one that did not must name a pane it always copied, or it
        // says nothing and the rule falls.
        for (size_t mask = 0; mask < rule.by_mask.size(); ++mask) {
            if (!painted[mask]) continue;
            if (agreed[mask]) {
                if (laid[mask] == TRANSPARENT) rule.by_mask[mask].leaves_empty = true;
                else rule.by_mask[mask].paint = laid[mask];
                continue;
            }
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
    bool draw(const grid& input, const pane_split_facts& split, grid& drawn) const {
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
                const answer& says = by_mask[mask_at(boxes, input, r, c, posed)];
                if (!says.said()) return false;
                drawn.data[r][c] = says.leaves_empty ? (int8_t)TRANSPARENT
                                 : says.paint >= 0   ? says.paint
                                                     : shown(boxes, says.copy_pane, input, r, c, posed);
            }
        return true;
    }

    // Tries every pose and keeps the plainest one that redraws every output. As
    // is comes first, so a pose is only ever worn when it is doing work.
    // With no ground color, "nothing here" is an empty cell: what is left
    // where a plan deleted the ground.
    static pane_mask_rule detect(const vector<example>& train,
                                 const pane_split_facts& split) {
        if (!split.holds()) return {};
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            const pane_mask_rule rule = read_with(train, split, pose);
            if (rule.holds() && redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    return rule.draw(pair.input, split, drawn);
                })) return rule;
        }
        return {};
    }
};

// WHICH PANE THE ANSWER USES, directly or enlarged behind the divider walls.
//
// pane_mask_rule reads the panes against each other and makes a new drawing.
// This one chooses a pane, either as the answer or as its miniature blueprint.
// The rule keeps the reason for choosing, and the ways to choose
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
    bool fitted = false;
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
    static int carried_by(const grid& one, const pane_split_facts::pane& box) {
        int count = 0;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c)
                count += box.color_at(one, r, c) != TRANSPARENT;
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
                    int8_t way, int8_t which) {
        const int count = (int)boxes.size();
        if (count < 2) return -1;
        switch (way) {
        case AtIndex:
            return which < count ? which : -1;
        case OddOneOut:
            return unique_drawing(count,
                [&](int i) { return boxes[i].width; },
                [&](int i) { return boxes[i].height; },
                [&](int i, int r, int c) { return boxes[i].color_at(one, r, c); });
        case TheyRepeat: {
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
            for (const auto& box : boxes) carried.push_back(carried_by(one, box));
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

    // Hands back the named pane, or fits it behind the original wall color.
    bool draw(const grid& input, const pane_split_facts& split, grid& drawn) const {
        vector<pane_split_facts::pane> boxes;
        if (!holds() || !split.panes_of(input, boxes)) return false;
        const int found = pick(input, boxes, way, which);
        if (found < 0) return false;
        const auto& source = boxes[(size_t)found];
        if (!fitted) {
            drawn = source.drawn(input);
            return true;
        }
        // A cell of the chosen pane describes a pane of the full lattice.
        // Replay Fit's integer sampling, then restore the divider color, just
        // as the emitted sentences do. The existing pick owns ties and counts.
        const int wall = split.wall_of(input);
        if (!is_color(wall) ||
            source.width != split.across || source.height != split.down) return false;
        drawn = grid(input.width, input.height);
        for (int r = 0; r < input.height; ++r)
            for (int c = 0; c < input.width; ++c)
                drawn.data[r][c] = input.data[r][c] == wall ? (int8_t)wall :
                    source.color_at(input, fit_source_cell(r, source.height, input.height),
                                           fit_source_cell(c, source.width, input.width));
        return true;
    }

    // Walks the ways in order and keeps the first that names the right pane in
    // every pair. Order is the whole tiebreak: a reason before a place.
    static pane_pick_rule detect(const vector<example>& train,
                                 const pane_split_facts& split) {
        if (!split.holds() || train.empty()) return {};
        // Preserve direct picks before considering the enlarged reading.
        for (bool fitted : { false, true }) {
            auto names_it = [&](int8_t way, int8_t which) {
                const pane_pick_rule trying{ way, which, true, fitted };
                return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    return trying.draw(pair.input, split, drawn);
                });
            };
            // For a miniature blueprint, occupancy is the simplest reason:
            // its palette can also look odd merely because it has less ink.
            static constexpr int8_t fitted_ways[] = {
                Emptiest, Fullest, OddOneOut, OddColor, TheyRepeat,
                NotSymmetric, OnlySymmetric, MostColored, LeastColored, AtIndex
            };
            for (int8_t index = 0; index < WAYS; ++index) {
                const int8_t way = fitted ? fitted_ways[index] : index;
                if (way == NotSymmetric || way == OnlySymmetric) {
                    for (int8_t fold : FOLDS)
                        if (names_it(way, fold)) return { way, fold, true, fitted };
                } else if (way == AtIndex) {
                    for (int8_t place = 0; place < (int8_t)split.pairs.front().size(); ++place)
                        if (names_it(way, place)) return { way, place, true, fitted };
                } else if (names_it(way, 0)) {
                    return { way, 0, true, fitted };
                }
            }
        }
        return {};
    }
};

// THE BANDS `count` BOXES SIT IN, of rows or of columns, as overlapping_bands
// groups them for Layout, SelectRow and SelectColumn: each box's band into
// `band`, and how many bands there are. `box_of(i)` is box i.
template <class BoxOf>
inline int bands_of(int count, bool rows, BoxOf box_of, int8_t* band) {
    vector<int16_t> lows((size_t)count), highs((size_t)count);
    vector<int> order((size_t)count);
    for (int i = 0; i < count; ++i) {
        const cell_box box = box_of(i);
        lows[(size_t)i] = (int16_t)(rows ? box.row0 : box.col0);
        highs[(size_t)i] = (int16_t)(rows ? box.row1 : box.col1);
    }
    return overlapping_bands(count, lows.data(), highs.data(), band, order.data());
}

// EVERY PANE'S DRAWING, CUT DOWN TO WHAT IT HOLDS AND GATHERED WHERE ITS PANE WAS.
//
// pane_mask_rule makes one drawing out of the panes and pane_pick_rule keeps
// one of them; here each pane keeps its own drawing, cropped, and the drawings
// are laid together the way the panes lay, in any number of rows and columns.
// After explicit ground removal, `Panes Layout[0]` paints this: Panes hands each
// pane back in its own frame, Layout bands the pieces by their frames and draws
// what each holds, and Layout[0] makes every cell as big as the largest drawing.
// Nothing about the drawings' size is in the rule; each grid says its own.
struct pane_gather_rule {
    bool held = false;
    bool holds() const { return held; }

    // Draws what the sentence paints for one input, on the canvas the size rule
    // gives it -- or, under BoundingBox, on the largest canvas there is, closed on
    // what was painted, as the kernel closes it. Declines when no pane holds
    // anything, since then nothing is laid out at all.
    static bool draw(const grid& input, const pane_split_facts& split, const output_dimension_rule& size,
                     grid& drawn) {
        vector<pane_split_facts::pane> boxes;
        if (!split.panes_of(input, boxes) || boxes.empty() ||
            (int)boxes.size() > MAX_OBJECTS_IN_A_LIST) return false;
        const int count = (int)boxes.size();
        // What each pane holds, and the cell that holds the largest of them.
        vector<cell_box> drawings((size_t)count);
        int cell_w = 0, cell_h = 0;
        for (int p = 0; p < count; ++p) {
            const auto& box = boxes[(size_t)p];
            for (int r = 0; r < box.height; ++r)
                for (int c = 0; c < box.width; ++c)
                    if (box.color_at(input, r, c) != TRANSPARENT) drawings[(size_t)p].include_cell(r, c);
            cell_w = max(cell_w, drawings[(size_t)p].width());
            cell_h = max(cell_h, drawings[(size_t)p].height());
        }
        if (cell_w == 0 || cell_h == 0) return false;
        // The bands come off the frames, the same grouping the kernel reads.
        int8_t row_of[MAX_OBJECTS_IN_A_LIST], col_of[MAX_OBJECTS_IN_A_LIST];
        auto frame = [&](int p) { return boxes[(size_t)p].box(); };
        const int rows = bands_of(count, true, frame, row_of), cols = bands_of(count, false, frame, col_of);
        const int laid_w = min(cols * cell_w, MAX_GRID_DIM), laid_h = min(rows * cell_h, MAX_GRID_DIM);
        return draw_on_canvas(input, size, drawn, [&](grid& canvas) {
            for (int p = 0; p < count; ++p) {
                const auto& box = boxes[(size_t)p];
                const cell_box& drawing = drawings[(size_t)p];
                for (int r = 0; r < min(cell_h, drawing.height()); ++r)
                    for (int c = 0; c < min(cell_w, drawing.width()); ++c) {
                        const int row = row_of[p] * cell_h + r, col = col_of[p] * cell_w + c;
                        const int color = box.color_at(input, drawing.row0 + r, drawing.col0 + c);
                        if (color == TRANSPARENT || row >= min(laid_h, canvas.height) ||
                            col >= min(laid_w, canvas.width)) continue;
                        canvas.data[row][col] = (int8_t)color;
                    }
            }
            return true;
        });
    }

    // With no ground color, what a pane holds is what is not empty.
    static pane_gather_rule detect(const vector<example>& train, const pane_split_facts& split,
                                   const output_dimension_rule& size) {
        if (!split.holds()) return {};
        pane_gather_rule rule;
        rule.held = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            return draw(pair.input, split, size, drawn);
        });
        return rule;
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
                                    const correspondence_election& correspondence) {
        if (train.empty()) return {};
        for (const auto& pair : train)
            if (pair.input.empty() ||
                pair.output.width % pair.input.width || pair.output.height % pair.input.height) return {};
        const auto* blocks = correspondence.best_reading_of(segmentation_methods::smallest_of_two_grids);
        if (!blocks || blocks->pairs.size() != train.size()) return {};
        // Background blocks are empty blocks: where a plan deleted the ground
        // the inputs have empty cells, and empty is what the blocks are made of.
        // A puzzle read whole, with no empty cell, has no background block.
        const int ground = TRANSPARENT;
        const bool has_ground = ranges::any_of(train, [](const example& pair) {
            for (int r = 0; r < pair.input.height; ++r)
                for (int c = 0; c < pair.input.width; ++c)
                    if (pair.input.data[r][c] == TRANSPARENT) return true;
            return false; });

        input_blocks_rule rule;
        rule.figure_poses = has_ground ? 0xFF : 0;
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
                if (has_ground) wears[TRANSPARENT] = edge.color_map.apply(TRANSPARENT);
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
                    if (!at.poses[r][c] && !has_ground) return {};
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
            for (int was = TRANSPARENT; was < COLOR_COUNT; ++was) {
                if (!wears.named(was)) continue;
                one_table = one_table && (!table.named(was) || table[was] == wears[was]);
                table[was] = wears[was];
            }
        // A copy that trades its figure with empty is the negative, whatever
        // one table says: no color word says a figure and empty trading places.
        if (one_table && !table.changes(TRANSPARENT)) {
            for (int was = 0; was < COLOR_COUNT; ++was)
                rule.color_map[was] = table.changes(was) ? table[was] : recoloring::UNSET;
        } else {
            if (!has_ground) return {};
            for (size_t which = 0; which < train.size(); ++which) {
                const recoloring& wears = worn[which];
                if (wears == recoloring{}) continue;
                const int8_t figure = the_figure_color(train[which].input, ground);
                if (!is_color(figure) ||
                    (wears.named(ground) && wears[ground] != figure) ||
                    wears[figure] != ground) return {};
            }
            rule.negative = true;
        }
        rule.held = true;
        return rule;
    }
};

// THE PIECES AS A CUT'S WORD HANDS THEM OVER, each in the box its own cells
// make, which is what they are once Trim has had them. pieces_of reads the
// same cut in matching's order; an op that works down its list in turn needs
// the word's own order, or a reading lays a later piece under an earlier one.
// Split hands them over in the order the walk meets them. Colors walks the
// palette rather than the grid, so its layers come in colour order.
inline vector<object> pieces_as_said(const grid& input, segmentation_methods segmentation) {
    vector<object> pieces;
    if (!boxed_by_its_cells(segmentation)) return pieces;
    object_walk_scratch scratch;
    each_object(input, TRANSPARENT, grouping_of(segmentation), scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
        pieces.push_back(object::from_cells(input, box, cells, count));
    });
    if (segmentation == segmentation_methods::all_cells_of_one_color)
        stable_sort(pieces.begin(), pieces.end(), [](const object& one, const object& other) {
            return one.first_color() < other.first_color(); });
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
                                  const correspondence_election& correspondence) {
        // The slide happens in each piece's own box, and the cells it leaves are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !pairs_overlay(train)) return {};
        // Draws each output from its input's pieces slid that way, noting whether any
        // cell moved at all.
        auto draws_every_output = [&](bool rows, int step, bool& moved) {
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                const auto pieces = pieces_of(pair.input, correspondence.segmentation);
                drawn = empty_grid(pair.output.width, pair.output.height);
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

// A BODY TURNED ONE OF THE EIGHT WAYS: the offset (row, col) from the cell it
// turns about, `way` & 3 quarter turns clockwise, then from 4 on flipped top to
// bottom -- the turns SnapToAnchors tries.
ARC_HOST_DEVICE inline void snap_turn(int way, int row, int col, int& to_row, int& to_col) {
    for (int turn = 0; turn < (way & 3); ++turn) {
        const int turned = col;
        col = -row;
        row = turned;
    }
    if (way >= 4) row = -row;
    to_row = row;
    to_col = col;
}

// THE BODIES SNAPPED ONTO THEIR ANCHORS, placed the way SnapToAnchors places
// them -- one home for the placing, so the op and the rule that replays it
// cannot set a body down two ways. The cells `at` hands over, TRANSPARENT where
// empty, are read as they lie:
//   - the body color is the color most cells wear, the lowest on a tie;
//   - a body is an 8-connected run of painted cells, of any colors, that holds
//     the body color, and its keys are its cells of the other colors, the first
//     SNAP_MOST_KEYS of them in reading order;
//   - an anchor is a cell of a run that does not hold it, SNAP_MOST_ANCHORS of
//     each color at most.
// Each body, in the order the reading meets its run, is turned one of the eight
// ways (snap_turn) and set down where the most of its keys land on anchors of
// their own colors -- the first placing tried wins a tie, and one key at least
// has to land. `label` hands back each cell's run, -1 for none. `place` is told
// the body's run, the turn, the key the turn is taken about and the anchor that
// key lands on, and says whether it set anything down; only then are the
// anchors its keys cover used up, so no later body can take them.
constexpr int SNAP_MOST_RUNS = 64, SNAP_MOST_KEYS = 12, SNAP_MOST_ANCHORS = 16;
template <class At, class Place>
ARC_HOST_DEVICE inline void snap_to_anchors(int width, int height, At at,
                                            int8_t (&label)[MAX_GRID_DIM][MAX_GRID_DIM], Place place) {
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) label[row][col] = -1;
    color_tally worn;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) worn.add(at(row, col));
    const int body = worn.most();
    if (body < 0) return;

    // The runs, each walked eight ways out from the first cell the reading meets.
    bool holds_body[SNAP_MOST_RUNS] = {};
    int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
    int runs = 0;
    for (int row = 0; row < height && runs < SNAP_MOST_RUNS; ++row)
        for (int col = 0; col < width && runs < SNAP_MOST_RUNS; ++col) {
            if (at(row, col) == TRANSPARENT || label[row][col] != -1) continue;
            const int run = runs++;
            label[row][col] = (int8_t)run;
            queue_row[0] = (int16_t)row;
            queue_col[0] = (int16_t)col;
            walk_out<8>(height, width, queue_row, queue_col, 1,
                [&](int next_row, int next_col) {
                    if (at(next_row, next_col) == TRANSPARENT || label[next_row][next_col] != -1) return false;
                    label[next_row][next_col] = (int8_t)run;
                    return true;
                },
                [&](int seen_row, int seen_col) { if (at(seen_row, seen_col) == body) holds_body[run] = true; });
        }

    // The anchors: where each lies, and a list of them by color.
    int8_t loose[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t anchor_row[COLOR_COUNT][SNAP_MOST_ANCHORS], anchor_col[COLOR_COUNT][SNAP_MOST_ANCHORS];
    int anchors[COLOR_COUNT] = {};
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) loose[row][col] = (int8_t)TRANSPARENT;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int color = at(row, col);
            const int run = label[row][col];
            if (color == TRANSPARENT || run < 0 || holds_body[run]) continue;
            loose[row][col] = (int8_t)color;
            if (is_color(color) && anchors[color] < SNAP_MOST_ANCHORS) {
                anchor_row[color][anchors[color]] = (int16_t)row;
                anchor_col[color][anchors[color]] = (int16_t)col;
                ++anchors[color];
            }
        }

    // Each body, set down where the most of its keys meet anchors of their colors.
    for (int run = 0; run < runs; ++run) {
        if (!holds_body[run]) continue;
        int key_color[SNAP_MOST_KEYS], key_row[SNAP_MOST_KEYS], key_col[SNAP_MOST_KEYS], keys = 0;
        for (int row = 0; row < height && keys < SNAP_MOST_KEYS; ++row)
            for (int col = 0; col < width && keys < SNAP_MOST_KEYS; ++col) {
                const int color = at(row, col);
                if (label[row][col] != run || color == body || color == TRANSPARENT) continue;
                key_color[keys] = color;
                key_row[keys] = row;
                key_col[keys] = col;
                ++keys;
            }
        if (keys == 0) continue;
        int best = 0, best_key = -1, best_row = 0, best_col = 0, best_way = 0;
        for (int key = 0; key < keys; ++key) {
            const int color = key_color[key];
            // A key is a color; said in full where it indexes the lists by color.
            if (color < 0 || color >= COLOR_COUNT) continue;
            for (int anchor = 0; anchor < anchors[color]; ++anchor)
                for (int way = 0; way < 8; ++way) {
                    int met = 0;
                    for (int other = 0; other < keys; ++other) {
                        int down, across;
                        snap_turn(way, key_row[other] - key_row[key], key_col[other] - key_col[key], down, across);
                        const int row = anchor_row[color][anchor] + down, col = anchor_col[color][anchor] + across;
                        met += row >= 0 && row < height && col >= 0 && col < width && loose[row][col] == key_color[other];
                    }
                    if (met > best) {
                        best = met;
                        best_key = key;
                        best_row = anchor_row[color][anchor];
                        best_col = anchor_col[color][anchor];
                        best_way = way;
                    }
                }
        }
        if (best_key < 0 || !place(run, best_way, key_row[best_key], key_col[best_key], best_row, best_col)) continue;
        for (int other = 0; other < keys; ++other) {
            int down, across;
            snap_turn(best_way, key_row[other] - key_row[best_key], key_col[other] - key_col[best_key], down, across);
            const int row = best_row + down, col = best_col + across, color = key_color[other];
            if (row < 0 || row >= height || col < 0 || col >= width || loose[row][col] != color) continue;
            loose[row][col] = (int8_t)TRANSPARENT;
            for (int anchor = 0; anchor < anchors[color]; ++anchor)
                if (anchor_row[color][anchor] == row && anchor_col[color][anchor] == col) {
                    anchor_row[color][anchor] = anchor_row[color][anchors[color] - 1];
                    anchor_col[color][anchor] = anchor_col[color][anchors[color] - 1];
                    --anchors[color];
                    break;
                }
        }
    }
}

// BODIES SNAPPED ONTO THEIR ANCHORS: each output is the input's bodies, turned
// and set down on the anchors their keys match, over the ground and nothing
// else -- snap_to_anchors run on the pieces the cut hands over, painted into the
// one grid the op reads them as. Held only where that draws every output
// exactly, and something other than the input in one pair at least: like the
// slam and the chain, it redraws each output whole and picks nothing, so a pair
// it leaves as it was is one it explains, not one it stands idle in.
struct snap_to_anchors_rule {
    bool held = false;
    bool holds() const { return held; }

    static snap_to_anchors_rule detect(const vector<example>& train, const correspondence_election& correspondence,
                                       const output_dimension_rule& size) {
        if (train.empty()) return {};
        bool at_work = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            // The pieces laid into one grid as large as a grid gets, a later one over an earlier.
            int8_t cells[MAX_GRID_DIM][MAX_GRID_DIM];
            for (auto& row : cells)
                for (int8_t& cell : row) cell = (int8_t)TRANSPARENT;
            for (const object& piece : pieces_as_said(pair.input, correspondence.segmentation))
                piece.each_cell([&](int row, int col, int8_t color) { cells[row][col] = color; });
            const bool painted = draw_on_canvas(pair.input, size, drawn, [&](grid& canvas) {
                int8_t label[MAX_GRID_DIM][MAX_GRID_DIM];
                bool set_down = false;
                snap_to_anchors(MAX_GRID_DIM, MAX_GRID_DIM, [&](int row, int col) { return (int)cells[row][col]; },
                                label, [&](int run, int way, int key_row, int key_col, int to_row, int to_col) {
                    // The body drawn turned about its key, over what the ones before it drew.
                    bool any = false;
                    for (int row = 0; row < MAX_GRID_DIM; ++row)
                        for (int col = 0; col < MAX_GRID_DIM; ++col) {
                            if (label[row][col] != run) continue;
                            int down, across;
                            snap_turn(way, row - key_row, col - key_col, down, across);
                            const int at_row = to_row + down, at_col = to_col + across;
                            if (at_row < 0 || at_row >= MAX_GRID_DIM || at_col < 0 || at_col >= MAX_GRID_DIM) continue;
                            any = true;
                            if (canvas.contains(at_row, at_col)) canvas.data[at_row][at_col] = cells[row][col];
                        }
                    set_down = set_down || any;
                    return any;
                });
                return set_down;
            });
            if (!painted) return false;
            at_work = at_work || drawn != pair.input;
            return true;
        });
        return { redraws && at_work };
    }
};

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
                                      const correspondence_election& correspondence) {
        // Each piece is hung off the last one's box, so the pieces have to be
        // measured in their own, and the cells no piece reaches are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            train.empty()) return {};
        // OverlapCorner lays out as many objects as one list holds; a chain this
        // detector verified past that point is one the op would not draw.
        constexpr size_t most_the_op_lays_out = MAX_OBJECTS_IN_A_LIST;
        bool laid_any_piece_somewhere_new = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const auto pieces = pieces_of(pair.input, correspondence.segmentation);
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
            drawn = empty_grid(pair.output.width, pair.output.height);
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
                                      const correspondence_election& correspondence) {
        // The pieces slide in grid coordinates and the cells they leave behind
        // are ground, so the two grids have to lie over one another; and the
        // boxes measured here are the boxes Trim hands the op.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !pairs_overlay(train)) return {};

        // The colours the pieces wear: two of them, the same two in every pair,
        // or there is no pair of picks that can say this.
        int8_t colors[2] = { -1, -1 };
        for (const auto& pair : train) {
            vector<object_segmentation::object_info> info;
            object_segmentation::segment_for_correspondence(pair.input,
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
                object_segmentation::segment_for_correspondence(pair.input,
                                                                correspondence.segmentation, info, &pieces);
                drawn = empty_grid(pair.output.width, pair.output.height);
                // WHAT THE OP SEES OF THE CANVAS is the box around everything
                // already painted, read once, before the first piece slides --
                // so what slides is measured against the pieces that stay, and
                // never against another piece that is still on its way.
                cell_box target;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != stays) continue;
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.y0, pieces[i].summary.x0)) return false;
                    target.include(pieces[i].summary.box());
                }
                // An empty canvas is nothing to touch, and the op answers that
                // by leaving every piece exactly where it stood.
                if (target.empty()) return false;
                bool anything_slid = false;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != slides) continue;
                    anything_slid = true;
                    const auto [down, across] = shift_to_touch(pieces[i].summary.box(), target);
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

// PIECES THAT SLIDE TO AN EDGE, AND THE REST LEFT WHERE IT LIES. Every piece,
// or one color's pieces, go one of the eight ways until they meet the edge of
// the canvas, passing over whatever is in the way, and land over or under what
// stays. That is MoveToEdge on a pick, so the replay steps each piece as the
// kernel does (steps_to_edge), and how far is the canvas's to say -- it differs
// from piece to piece, and no count of cells is written into the sentence.
//
// THE ELECTED CUT FIRST, THEN THE OTHERS. The election keeps the shortest
// reading, which can join the pieces that slide to the ones that stay: in
// 3618c87e it joins each blue dot to the gray under it and reads a recolor.
// Plainest first after that: every piece before one color, the sides before
// the diagonals, over before under.
//
// A SLIDE THE CANVAS HIDES IS NO SLIDE. Where the pieces laid where they lie
// already draw every output, nothing has to go anywhere -- 1cf80156 is a crop,
// and every piece pushed up and then cropped reads the same, falsely.
struct edge_slide_rule {
    segmentation_methods segmentation = segmentation_methods::unset;
    // The color whose pieces slide, or -1 when every piece does.
    int8_t color = -1;
    int8_t way = heading::up;
    // Laid over what stays, as Union lays it, or under it, as Underlay does.
    bool over = true;
    bool held = false;
    bool holds() const { return held; }

    // SelectColor picks a piece that carries the color anywhere.
    bool slides(const object& piece) const { return !is_color(color) || (palette(piece) >> color & 1) != 0; }

    // One input's output as the sentences paint it: what stays, where it lies,
    // then each sliding piece in the cut's own order, later over earlier.
    bool draw(const grid& input, const vector<object>& pieces, const output_dimension_rule& size,
              grid& drawn) const {
        return draw_on_canvas(input, size, drawn, [&](grid& canvas) {
            for (const object& piece : pieces)
                if (!slides(piece) && !paint_piece(canvas, piece, piece.summary.y0, piece.summary.x0))
                    return false;
            const heading_step step = step_of(way);
            for (const object& piece : pieces) {
                if (!slides(piece)) continue;
                const int steps = steps_to_edge(piece.summary.box(), canvas.width, canvas.height, way);
                if (!paint_piece(canvas, piece, piece.summary.y0 + steps * step.down,
                                 piece.summary.x0 + steps * step.across, over)) return false;
            }
            return true;
        });
    }

    static edge_slide_rule detect(const vector<example>& train, segmentation_methods elected,
                                  const output_dimension_rule& size) {
        if (train.empty()) return {};
        vector<segmentation_methods> cuts{ elected };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::all_cells_of_one_color,
                                segmentation_methods::connected_multicolor })
            if (cut != elected) cuts.push_back(cut);
        for (const auto cut : cuts) {
            if (!boxed_by_its_cells(cut)) continue;
            // Each pair cut once; every reading below is asked of the same pieces.
            vector<vector<object>> pieces;
            uint16_t colors = 0;
            for (const example& pair : train) {
                pieces.push_back(pieces_as_said(pair.input, cut));
                for (const object& piece : pieces.back()) colors = (uint16_t)(colors | palette(piece));
            }
            bool as_they_lie = true;
            for (size_t i = 0; i < train.size() && as_they_lie; ++i) {
                grid drawn;
                as_they_lie = draw_on_canvas(train[i].input, size, drawn, [&](grid& canvas) {
                    for (const object& piece : pieces[i])
                        if (!paint_piece(canvas, piece, piece.summary.y0, piece.summary.x0)) return false;
                    return true;
                }) && drawn == train[i].output;
            }
            if (as_they_lie) continue;
            for (int color = -1; color < COLOR_COUNT; ++color) {
                if (color >= 0 && !(colors >> color & 1)) continue;
                for (int at = 0; at < heading::count; ++at)
                    for (bool over : { true, false }) {
                        // With every piece sliding nothing stays to lie under.
                        if (!over && color < 0) continue;
                        const edge_slide_rule trying{ cut, (int8_t)color, (int8_t)heading_asked(at), over, true };
                        bool every = true;
                        for (size_t i = 0; i < train.size() && every; ++i) {
                            grid drawn;
                            every = trying.draw(train[i].input, pieces[i], size, drawn) &&
                                    drawn == train[i].output;
                        }
                        if (every) return trying;
                    }
            }
        }
        return {};
    }
};

// ONE FIGURE COPIED TO THE OTHER PIECES. The largest connected figure supplies
// the drawing, the others supply frames, and the examples agree on one anchor.
// Try the input's pieces first, then the already-elected panes without their
// separators. No color, template size, destination count or coordinates are
// baked into the claim. Trim gives the op the same tight frames read here.
// The op itself accepts any selection of source figures;
// this reading only names the unambiguous largest one.
struct copy_to_rule {
    bool held = false;
    bool panes = false;
    int8_t anchor = 4;
    bool holds() const { return held; }

    static vector<object> pieces(const grid& input, const pane_split_facts& split, bool use_panes) {
        grid source = input;
        if (use_panes) {
            vector<pane_split_facts::pane> frames;
            if (!split.panes_of(input, frames) || frames.size() > MAX_OBJECTS_IN_A_LIST) return {};
            for (int r = 0; r < source.height; ++r)
                for (int c = 0; c < source.width; ++c) source.data[r][c] = TRANSPARENT;
            for (const auto& frame : frames)
                for (int r = frame.top; r < frame.top + frame.height; ++r)
                    for (int c = frame.left; c < frame.left + frame.width; ++c)
                        source.data[r][c] = input.data[r][c];
        }
        return pieces_as_said(source, segmentation_methods::connected_multicolor);
    }

    static int source_index(const vector<object>& objects) {
        if (objects.size() < 2 || objects.size() > MAX_OBJECTS_IN_A_LIST) return -1;
        int largest = 0;
        for (int i = 1; i < (int)objects.size(); ++i)
            if (objects[i].summary.count > objects[largest].summary.count) largest = i;
        for (int i = 0; i < (int)objects.size(); ++i)
            if (i != largest && objects[i].summary.count == objects[largest].summary.count) return -1;
        return largest;
    }

    static grid draw(const grid& input, const vector<object>& objects, int source, int anchor) {
        grid drawn = input;
        const auto& pattern = objects[source];
        for (int i = 0; i < (int)objects.size(); ++i) {
            if (i == source) continue;
            const auto& target = objects[i].summary;
            const int row = aligned_copy_start(pattern.summary.h, target.y0, target.h, anchor / 3);
            const int col = aligned_copy_start(pattern.summary.w, target.x0, target.w, anchor % 3);
            paint_piece(drawn, pattern, row, col, /*over=*/true, /*clip=*/true);
        }
        return drawn;
    }

    static copy_to_rule detect(const vector<example>& train,
                               const pane_split_facts& split) {
        if (train.empty()) return {};
        bool changed = false;
        for (const auto& pair : train) {
            if (pair.input.width != pair.output.width || pair.input.height != pair.output.height) return {};
            changed |= pair.input != pair.output;
        }
        if (!changed) return {};
        for (bool use_panes : { false, true }) {
            if (use_panes && !split.holds()) continue;
            vector<vector<object>> cut;
            vector<int> sources;
            bool readable = true;
            for (const auto& pair : train) {
                cut.push_back(pieces(pair.input, split, use_panes));
                sources.push_back(source_index(cut.back()));
                if (sources.back() < 0) { readable = false; break; }
            }
            if (!readable) continue;
            for (int anchor = 0; anchor < 9; ++anchor) {
                bool every = true;
                for (size_t i = 0; i < train.size() && every; ++i)
                    every = draw(train[i].input, cut[i], sources[i], anchor) == train[i].output;
                if (every) return { true, use_panes, (int8_t)anchor };
            }
        }
        return {};
    }
};

// THE SEEDS THE OP WILL BE HANDED. The DSL fills its seed list off the input
// before a program runs (17_fixed_dsl.hpp), and this says the same list from
// this side of the fence, so a replay below can be checked against what the op
// will really do. The rule itself is not restated -- 4_objects.hpp's read_seeds
// is the one copy of it -- only the grid it is asked about.
struct marker_piece {
    object_summary summary;
    int8_t color = 0;
};

inline vector<marker_piece> seeds_the_op_will_see(const grid& input) {
    vector<marker_piece> pieces;
    object_walk_scratch scratch;
    each_object(input, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int found) {
        if (found <= 0) return;
        pieces.push_back({ box, (int8_t)input.data[cell_row(cells[0])][cell_col(cells[0])] });
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
//
// OR EVERY STAMP WEARS ONE COLOUR, AND THE INPUT IS LAID BACK OVER THEM, so a
// stamp shows only where the input had ground. That is a run carried on to the
// edge in a colour of its own: 0b17323b's dots step on at their own spacing
// until they leave the grid, and the new ones come out red. The colour is read
// off the first stamp that shows and held to every other, as the interior fill
// reads its own. The markers' colours are asked first, being the plainer claim.
struct project_on_seeds_rule {
    bool held = false;
    // The one colour every stamp wears when the input is laid back over them;
    // not a colour when each stamp wears its marker's, with the input beneath.
    int8_t worn = TRANSPARENT;
    bool holds() const { return held; }

    // Calls land(row, col, marker colour) for each cell of each stamp the words
    // lay on this pair's canvas, and says how many stamps they laid -- none when
    // there is no template or no marker to lay one along.
    template <class Land>
    static int stamp(const example& pair, segmentation_methods cut, Land land) {
        const auto pieces = pieces_of(pair.input, cut);
        if (pieces.empty()) return 0;
        // The template is whichever piece SortBySize[0] puts first, and that
        // word ranks by cells -- not the box that picked the markers. The op
        // reads the two off different lists, so this has to as well.
        // Tied for most cells: the op's list order picks, which we can't see.
        const object& tpl = pieces[0];
        if (pieces.size() > 1 && pieces[1].summary.count == tpl.summary.count) return 0;
        // ProjectOnSeeds hands back the copy of the input it draws on plus one
        // object per stamp, and they share one list, so the drawing runs out one
        // stamp before the list does.
        constexpr int most_the_op_stamps = MAX_OBJECTS_IN_A_LIST - 1;
        int stamped = 0;
        for (const marker_piece& marker : seeds_the_op_will_see(pair.input))
            project_along(tpl.summary.box(), marker.summary.box(),
                          pair.input.width, pair.input.height, [&](int at_row, int at_col) {
                if (stamped >= most_the_op_stamps) return false;
                // A stamp hanging over the edge is drawn as far as it fits.
                for (int r = 0; r < tpl.summary.h; ++r)
                    for (int c = 0; c < tpl.summary.w; ++c) {
                        const int row = at_row + r, col = at_col + c;
                        if (tpl.cells[r][c] != TRANSPARENT && pair.input.contains(row, col))
                            land(row, col, marker.color);
                    }
                ++stamped;
                return true;
            });
        return stamped;
    }

    static project_on_seeds_rule detect(const vector<example>& train,
                                        const correspondence_election& correspondence) {
        // The template is handed over in the frame its cells make, which is
        // what the Trim in front of the word gives it. Stamping paints over a
        // copy of the input, so a canvas of another size is a different drawing
        // than the one replayed here.
        const segmentation_methods cut = correspondence.segmentation;
        if (!boxed_by_its_cells(cut) || !pairs_overlay(train)) return {};
        // Nothing stamped is the input handed straight back, which every other
        // reading already says without a word this long.
        const bool in_the_markers_colors = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = pair.input;
            return stamp(pair, cut, [&](int row, int col, int8_t color) {
                drawn.data[row][col] = color;
            }) > 0;
        });
        if (in_the_markers_colors) return { true, TRANSPARENT };

        // With the input laid back over the stamps, its ground shows wherever no
        // stamp went, and the outputs' elected ground is what fills it -- or,
        // where a plan deleted the ground, nothing does until the plan's fill.
        int8_t worn = TRANSPARENT;
        bool one_color = true;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = empty_grid(pair.input.width, pair.input.height);
            walk_colored_cells(pair.input, TRANSPARENT, [&](int row, int col, int color) {
                drawn.data[row][col] = (int8_t)color;
            });
            const int stamped = stamp(pair, cut, [&](int row, int col, int8_t) {
                if (pair.input.data[row][col] != TRANSPARENT) return;
                const int8_t shown = (int8_t)pair.output.data[row][col];
                if (!is_color(worn)) worn = shown;
                one_color = one_color && shown == worn;
                drawn.data[row][col] = worn;
            });
            return stamped > 0 && one_color;
        });
        // A stamp in the ground's own colour draws nothing at all.
        if (!redraws || !is_color(worn) || worn == TRANSPARENT) return {};
        return { true, worn };
    }
};

// WHAT BRIDGE DRAWS, said once for the op and for the reading that replays it.
// Each piece reaches for the pieces of its own colour whose centres share its
// row or its column, and the box between two is filled in that colour. HOW IT
// REACHES is Bridge's value, `way`: along rows and columns both, along rows
// alone, or along columns alone (way % 3), and to the nearest piece that way or
// to every one (way / 3) -- Bridge[0] reaching both ways for the nearest.
// `count` pieces in the op's own order: box_of_piece(i) is piece i's cell_box,
// color_of(i) the colour the op reads off its first cell (-1 for none).
// land(bar, colour) sees each bar in the order the op lays them into its one
// drawing of every bar, and may say false to stop. gpu_exec_bridge calls it, so
// a reading and the op cannot draw different bars.
constexpr int BRIDGE_WAYS = 6;
template <class BoxOf, class ColorOf, class Land>
ARC_HOST_DEVICE inline void bridge_bars(int way, int count, BoxOf box_of_piece, ColorOf color_of, Land land) {
    if (way < 0 || way >= BRIDGE_WAYS) return;
    const bool rows = way % 3 != 2, columns = way % 3 != 1, every = way >= 3;
    constexpr int most_pairs = 64;
    int16_t lows[most_pairs], highs[most_pairs];
    int pairs = 0;
    auto across = [&](int i) { const cell_box box = box_of_piece(i); return (box.col0 + box.col1) / 2; };
    auto down = [&](int i) { const cell_box box = box_of_piece(i); return (box.row0 + box.row1) / 2; };
    auto pair_up = [&](int i, int partner) {
        const int low = min(i, partner), high = max(i, partner);
        bool said = false;
        for (int p = 0; p < pairs && !said; ++p) said = lows[p] == low && highs[p] == high;
        if (said || pairs >= most_pairs) return;
        lows[pairs] = (int16_t)low;
        highs[pairs] = (int16_t)high;
        ++pairs;
    };
    for (int i = 0; i < count; ++i) {
        const int8_t color = color_of(i);
        if (color < 0) continue;
        const int my_across = across(i), my_down = down(i);
        // Ties keep the first found, as the op's list has them.
        int partner = -1, nearest = 0x7FFF;
        for (int j = 0; j < count; ++j) {
            if (j == i || color_of(j) != color) continue;
            const int their_across = across(j), their_down = down(j);
            if (!(rows && their_down == my_down) && !(columns && their_across == my_across)) continue;
            if (every) { pair_up(i, j); continue; }
            const int distance = abs(their_across - my_across) + abs(their_down - my_down);
            if (distance < nearest) { nearest = distance; partner = j; }
        }
        if (partner >= 0) pair_up(i, partner);
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
// THE WAYS A BRIDGE IS ASKED IN, the narrowest first -- one direction before
// both, the nearest before every one -- so the bars are said no wider than the
// pairs show them: a699fb00 bridges along rows, and nothing in it says a
// column would be bridged too.
constexpr int bridge_ways_narrowest_first[BRIDGE_WAYS] = { 1, 2, 0, 4, 5, 3 };
static_assert(row_of(node_type::Bridge).high + 1 == BRIDGE_WAYS, "Bridge takes one value for each way it reaches");

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
    // How the bridge reaches, Bridge's value (bridge_bars).
    int8_t way = 0;
    bool holds() const { return held; }

    static bridge_rule detect(const vector<example>& train,
                              const correspondence_election& correspondence) {
        // The bars fill cells that were ground, and the rest of the grid stays
        // where it was, so the two grids lie over one another.
        if (!pairs_overlay(train)) return {};
        vector<segmentation_methods> cuts{ correspondence.segmentation };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::connected_multicolor })
            if (cut != correspondence.segmentation) cuts.push_back(cut);
        // Each cut the bridge can be read under, each way it can reach, the
        // narrowest first.
        for (const auto cut : cuts) {
            if (cut != segmentation_methods::connected_monochrome &&
                cut != segmentation_methods::connected_multicolor) continue;
            for (const int way : bridge_ways_narrowest_first) {
                bool bridged = false;
                const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    // The pieces as Split hands them over: the one walk, in its order.
                    const vector<object> pieces = pieces_as_said(pair.input, cut);
                    // One list is all the op is handed, and the bars take one place
                    // more in it, all of them one drawing.
                    if (pieces.empty() || pieces.size() >= MAX_OBJECTS_IN_A_LIST) return false;
                    vector<int8_t> colors;
                    for (const object& piece : pieces) colors.push_back(piece.first_color());
                    drawn = empty_grid(pair.output.width, pair.output.height);
                    bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                    for (const object& piece : pieces)
                        piece.each_cell([&](int row, int col, int8_t color) {
                            drawn.data[row][col] = color;
                            painted[row][col] = true;
                        });
                    // Each bar is laid under whatever is already drawn.
                    bridge_bars(way, (int)pieces.size(),
                                [&](int which) { return pieces[which].summary.box(); },
                                [&](int which) { return colors[which]; },
                                [&](const cell_box& bar, int8_t color) {
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
                if (redraws && bridged) return { true, cut, (int8_t)way };
            }
        }
        return {};
    }
};

// TILE[3]'S LATTICE, READ OFF THE MARKS THEMSELVES. They stand one smallest
// gap apart and take turns, so each comes round again after as many gaps as
// there are marks, starting on its own row and never above it. Said once for
// the op and for the reading that replays it, so the two cannot read a
// different lattice off the same marks.
//
// The period over `count` marks whose top rows row_of(i) gives, or 0 when
// there is no lattice to read -- fewer than two marks, or all of them on one
// row -- and the op then hands its marks back as they were. No two rows of one
// grid stand the largest grid apart, so that gap doubles as none found.
template <class RowOf>
ARC_HOST_DEVICE inline int mark_lattice_period(int count, RowOf row_of) {
    int gap = MAX_GRID_DIM;
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j) {
            const int apart = row_of(i) > row_of(j) ? row_of(i) - row_of(j) : row_of(j) - row_of(i);
            if (apart > 0 && apart < gap) gap = apart;
        }
    return gap < MAX_GRID_DIM ? count * gap : 0;
}

// Calls lay(row, source) for every row of a `rows`-tall canvas that a mark
// `height` rows tall, standing at `top`, comes round on, with which of its own
// rows lands there.
template <class Lay>
ARC_HOST_DEVICE inline void each_lattice_row(int top, int height, int period, int rows, Lay lay) {
    if (period <= 0) return;
    for (int row = top > 0 ? top : 0; row < rows; ++row) {
        const int source = (row - top) % period;
        if (source < height) lay(row, source);
    }
}

// THE MARKS AS LINES THAT TAKE TURNS DOWN THE GRID. Every mark becomes a line
// across the grid, and the lines come round again in the order the marks
// stand, to the far edge. That is TileX[1], which spreads a mark's first
// column into its line, then Tile[3], which lays the lattice; with
// NormalizeOrientation[0] in front, a pair lying on its side is read turned,
// so its lines cross the short way. Only held when replaying those words
// redraws every output exactly, which is what makes the lattice a rule read
// off each input rather than rows remembered from the pairs.
//
// THE CUT IS THE ELECTED ONE WHEN IT CAN SAY THIS, and otherwise the first
// that can, as for the bridge: a mark is whatever the word hands over, and it
// is replayed in the order the word hands it, since a later line paints over
// an earlier one where two come round on the same row.
//
// A TURN IS WORN ONLY WHEN IT DOES WORK. The grid as given is asked first, so
// pairs that all stand upright never say it, and a pair lying on its side only
// says it when its lines cross the short way.
//
// AND IT HAS TO LAY SOMETHING. An output the input already was is marks
// sitting where their own lattice would put them, which says nothing about one.
struct mark_lattice_rule {
    bool held = false;
    segmentation_methods segmentation = segmentation_methods::unset;
    // Whether a grid lying on its side is read turned: NormalizeOrientation[0].
    bool portrait = false;
    bool holds() const { return held; }

    // What the words draw on one input: the cut's marks, trimmed and read the
    // way the pair is turned, each laid as its line wherever the lattice puts
    // it, the rest left empty.
    static bool draw(const grid& input, segmentation_methods cut, bool portrait, grid& drawn) {
        const vector<object> marks = pieces_as_said(input, cut);
        // One list is all the op is handed.
        if (marks.size() > MAX_OBJECTS_IN_A_LIST) return false;
        // Read turned, the marks' rows are their columns, and the other way round.
        const bool turned = portrait && lies_on_its_side(input.width, input.height);
        auto top = [&](const object& mark) { return turned ? mark.summary.x0 : mark.summary.y0; };
        const int period = mark_lattice_period((int)marks.size(),
                                               [&](int i) { return (int)top(marks[i]); });
        if (period <= 0) return false;
        drawn = empty_grid(input.width, input.height);
        const int rows = turned ? input.width : input.height;
        const int across = turned ? input.height : input.width;
        auto at = [&](int row, int col) -> int& {
            return turned ? drawn.data[col][row] : drawn.data[row][col];
        };
        for (const object& mark : marks)
            each_lattice_row(top(mark), turned ? mark.summary.w : mark.summary.h, period, rows,
                             [&](int row, int source) {
                // The line wears the mark's first column, as TileX[1] spreads it.
                const int8_t color = turned ? mark.cells[0][source] : mark.cells[source][0];
                if (color == TRANSPARENT) return;
                for (int col = 0; col < across; ++col) at(row, col) = color;
            });
        return true;
    }

    static mark_lattice_rule detect(const vector<example>& train,
                                    const correspondence_election& correspondence) {
        // Lines go down on an empty canvas the size of the input, and whatever
        // they miss stays empty, for the plan to fill.
        if (!pairs_overlay(train)) return {};
        vector<segmentation_methods> cuts{ correspondence.segmentation };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::connected_multicolor,
                                segmentation_methods::all_cells_of_one_color })
            if (cut != correspondence.segmentation) cuts.push_back(cut);
        for (const auto cut : cuts) {
            if (!boxed_by_its_cells(cut)) continue;
            for (const bool portrait : { false, true }) {
                bool laid = false;
                const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    if (!draw(pair.input, cut, portrait, drawn))
                        return false;
                    laid = laid || drawn != pair.input;
                    return true;
                });
                if (redraws && laid) return { true, cut, portrait };
            }
        }
        return {};
    }
};

// A shape keeps its boundary and spreads its own rarest color over its core.
// Read each marker from the input, so neither colors nor shapes are memorized.
struct interior_spread_rule {
    bool held = false;
    bool holds() const { return held; }

    static interior_spread_rule detect(const vector<example>& train) {
        if (train.empty() || !pairs_overlay(train)) return {};
        bool changed = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = pair.input;
            object_walk_scratch scratch;
            int pieces = 0;
            each_object(pair.input, TRANSPARENT, grouping::touching_any_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    ++pieces;
                    const object piece = object::from_cells(pair.input, box, cells, count);
                    const int8_t color = object_paint_color(box.h, box.w, COLOR_COUNT,
                        [&](int row, int col) { return (int8_t)piece.cells[row][col]; });
                    piece.each_cell([&](int row, int col, int8_t was) {
                        if (on_object_edge(row, col, [&](int r, int c) { return piece.holds(r, c); })) return;
                        drawn.data[row][col] = color;
                        changed = changed || color != was;
                    });
                });
            return pieces > 0 && pieces <= MAX_OBJECTS_IN_A_LIST;
        });
        return {redraws && changed};
    }
};

// Every enclosed space takes one color, independently of the boundary's shape.
// Read that color from the interiors, then check the whole drawing, including
// unchanged bodies and the elected outside color. No exact-shape keys are needed.
struct interior_fill_rule {
    int8_t color = -1;
    bool holds() const { return is_color(color); }

    static interior_fill_rule detect(const vector<example>& train) {
        if (train.empty() || !pairs_overlay(train)) return {};
        interior_fill_rule rule;
        bool changed = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = empty_grid(pair.input.width, pair.input.height);
            object_walk_scratch scratch;
            bool consistent = true;
            int pieces = 0;
            each_object(pair.input, TRANSPARENT, grouping::touching_any_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    if (!consistent) return;
                    ++pieces;
                    const object piece = object::from_cells(pair.input, box, cells, count);
                    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
                    mark_outside(box.h, box.w,
                        [&](int r, int c) { return is_color(piece.cells[r][c]); }, outside);
                    // Separate marker objects inside a hole keep their own
                    // colors. They are not evidence for the interior's fill.
                    for (int r = 0; r < box.h; ++r)
                        for (int c = 0; c < box.w; ++c) {
                            const int y = box.y0 + r, x = box.x0 + c;
                            if (is_color(piece.cells[r][c]) || outside[r][c] ||
                                pair.input.data[y][x] != TRANSPARENT) continue;
                            const int fill = pair.output.data[y][x];
                            if (!is_color(fill) || (rule.holds() && rule.color != fill)) {
                                consistent = false;
                                return;
                            }
                            rule.color = (int8_t)fill;
                            changed = changed || fill != pair.input.data[y][x];
                        }
                    // Replay in object order, as Union does: a later marker
                    // can restore its body over an earlier object's fill.
                    for (int r = 0; r < box.h; ++r)
                        for (int c = 0; c < box.w; ++c) {
                            const int y = box.y0 + r, x = box.x0 + c;
                            if (is_color(piece.cells[r][c]))
                                drawn.data[y][x] = piece.cells[r][c];
                            else if (!outside[r][c] && rule.holds())
                                drawn.data[y][x] = rule.color;
                        }
                });
            return consistent && pieces > 0 && pieces <= MAX_OBJECTS_IN_A_LIST;
        });
        return redraws && changed ? rule : interior_fill_rule{};
    }
};

// A grid of trimmed connected pieces, selected by removing its most common colors.
// Only held when the shared Layout and ScaleDown geometry redraws every output.
struct ranked_layout_rule {
    bool held = false;
    int8_t discarded_colors = 0;
    int8_t split_mode = 0;
    int8_t cell_size = 0;
    bool holds() const { return held; }

    bool draw(const example& pair, grid& drawn) const {
        if (!holds() || pair.input.empty() || discarded_colors < 0 || discarded_colors > COLOR_COUNT ||
            cell_size < 2 || cell_size > 10 || (split_mode != 0 && split_mode != 1)) return false;

        int frequencies[COLOR_COUNT];
        count_colors(pair.input, frequencies);
        bool removed[COLOR_COUNT] = {};
        for (int rank = 0; rank < discarded_colors; ++rank) {
            int common = -1;
            for (int color = 0; color < COLOR_COUNT; ++color) {
                if (removed[color] || frequencies[color] == 0) continue;
                if (common < 0 || frequencies[color] > frequencies[common]) common = color;
            }
            if (common >= 0) removed[common] = true;
        }

        grid remaining = empty_grid(pair.input.width, pair.input.height);
        for (int row = 0; row < pair.input.height; ++row)
            for (int col = 0; col < pair.input.width; ++col) {
                const int color = pair.input.data[row][col];
                if (is_color(color) && !removed[color])
                    remaining.data[row][col] = color;
            }

        vector<object> pieces;
        object_walk_scratch scratch;
        const grouping groups = split_mode == 0 ? grouping::touching_any_color
                                                 : grouping::touching_same_color;
        each_object(remaining, TRANSPARENT, groups, scratch,
            [&](const object_summary& box, const int16_t* cells, int count) {
                if ((int)pieces.size() < MAX_OBJECTS_IN_A_LIST)
                    pieces.push_back(object::from_cells(remaining, box, cells, count));
            });
        if (pieces.empty()) return false;

        int8_t row_of[MAX_OBJECTS_IN_A_LIST], col_of[MAX_OBJECTS_IN_A_LIST];
        auto piece_box = [&](int i) { return pieces[(size_t)i].summary.box(); };
        const int rows = bands_of((int)pieces.size(), true, piece_box, row_of);
        const int cols = bands_of((int)pieces.size(), false, piece_box, col_of);
        const int width = min(cols * (int)cell_size, MAX_GRID_DIM);
        const int height = min(rows * (int)cell_size, MAX_GRID_DIM);
        if (width <= 0 || height <= 0) return false;

        object arranged;
        arranged.summary = {0, 0, (int8_t)width, (int8_t)height, 0};
        arranged.cells.clear();
        for (int i = 0; i < (int)pieces.size(); ++i) {
            const object& piece = pieces[i];
            const int top = row_of[i] * cell_size;
            const int left = col_of[i] * cell_size;
            for (int row = 0; row < piece.summary.h && row < cell_size; ++row)
                for (int col = 0; col < piece.summary.w && col < cell_size; ++col) {
                    const int destination_row = top + row, destination_col = left + col;
                    if (destination_row < height && destination_col < width &&
                        piece.cells[row][col] != TRANSPARENT)
                        arranged.cells[destination_row][destination_col] = piece.cells[row][col];
                }
        }

        const int reduced_width = max((width + cell_size - 1) / cell_size, 1);
        const int reduced_height = max((height + cell_size - 1) / cell_size, 1);
        grid painted = empty_grid(MAX_GRID_DIM, MAX_GRID_DIM);
        for (int row = 0; row < reduced_height; ++row)
            for (int col = 0; col < reduced_width; ++col)
                painted.data[row][col] = majority_vote(arranged.cells, row, col, cell_size, height, width);

        cell_box bounds;
        for (int row = 0; row < reduced_height; ++row)
            for (int col = 0; col < reduced_width; ++col)
                if (painted.data[row][col] != TRANSPARENT) bounds.include_cell(row, col);
        if (bounds.empty()) return false;

        drawn = grid(bounds.width(), bounds.height());
        for (int row = 0; row < drawn.height; ++row)
            for (int col = 0; col < drawn.width; ++col) {
                const int color = painted.data[bounds.row0 + row][bounds.col0 + col];
                drawn.data[row][col] = color;
            }
        return true;
    }

    static ranked_layout_rule detect(const vector<example>& train, const output_dimension_rule& size) {
        if (train.empty() || size.Chosen != output_dimension_rule::BoundingBox) return {};
        for (int discarded = 0; discarded <= COLOR_COUNT; ++discarded)
            for (int mode = 0; mode <= 1; ++mode)
                for (int factor = 2; factor <= 10; ++factor) {
                    ranked_layout_rule candidate;
                    candidate.held = true;
                    candidate.discarded_colors = (int8_t)discarded;
                    candidate.split_mode = (int8_t)mode;
                    candidate.cell_size = (int8_t)factor;
                    const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                        return candidate.draw(pair, drawn);
                    });
                    if (redraws) return candidate;
                }
        return {};
    }
};

// The output is the least common color, where it lies. Nothing crops it but the
// size rule: under BoundingBox the canvas closes on what was painted. Read per
// grid, so the color may change from pair to pair; a tie for least common, or a
// grid with one color and nothing to rank it against, is no reading at all.
struct rarest_color_rule {
    bool held = false;
    bool holds() const { return held; }

    static bool draw(const grid& input, grid& drawn) {
        int counts[COLOR_COUNT];
        count_colors(input, counts);
        int rarest = -1, colors = 0;
        bool tied = false;
        for (int color = 0; color < COLOR_COUNT; ++color) {
            if (!counts[color]) continue;
            ++colors;
            if (rarest >= 0 && counts[color] == counts[rarest]) tied = true;
            else if (rarest < 0 || counts[color] < counts[rarest]) { rarest = color; tied = false; }
        }
        if (colors < 2 || tied) return false;
        cell_box bounds;
        for (int row = 0; row < input.height; ++row)
            for (int col = 0; col < input.width; ++col)
                if (input.data[row][col] == rarest) bounds.include_cell(row, col);
        drawn = grid(bounds.width(), bounds.height());
        for (int row = 0; row < drawn.height; ++row)
            for (int col = 0; col < drawn.width; ++col)
                drawn.data[row][col] = input.data[bounds.row0 + row][bounds.col0 + col] == rarest
                                     ? (int8_t)rarest : (int8_t)TRANSPARENT;
        return true;
    }

    static rarest_color_rule detect(const vector<example>& train, const output_dimension_rule& size) {
        if (train.empty() || size.Chosen != output_dimension_rule::BoundingBox) return {};
        rarest_color_rule rule;
        rule.held = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            return draw(pair.input, drawn);
        });
        return rule;
    }
};

// THE OUTPUT IS THE INPUT SEEN THROUGH THE SIZE RULE'S BOX, AT ONE CORNER.
// The canvas is the box, so painting the input crops it, and pushing the input
// down or right is what brings its bottom or right side into view: `Input
// SelectAll MoveToEdge[..] Union`. No lattice, no pieces and no count of cells
// -- the box says how much. A box larger than the input pads instead, which is
// the same sentence. Corners are tried with the fewest pushes first, and one
// corner must hold for every pair.
struct corner_crop_rule {
    bool held = false;
    // The pushes that bring this corner into view.
    bool down = false, right = false;
    bool holds() const { return held; }

    bool draw(const grid& input, const output_dimension_rule& size, grid& drawn) const {
        if (input.empty()) return false;
        return draw_on_canvas(input, size, drawn, [&](grid& canvas) {
            // The input is one piece as big as itself, so its cells are its box.
            const cell_box whole{ 0, 0, input.height - 1, input.width - 1 };
            const int top = down ? steps_to_edge(whole, canvas.width, canvas.height, heading::down) : 0;
            const int left = right ? steps_to_edge(whole, canvas.width, canvas.height, heading::right) : 0;
            for (int r = 0; r < input.height; ++r)
                for (int c = 0; c < input.width; ++c)
                    if (canvas.contains(top + r, left + c)) canvas.data[top + r][left + c] = input.data[r][c];
            return true;
        });
    }

    static corner_crop_rule detect(const vector<example>& train,
                                   const output_dimension_rule& size) {
        if (train.empty()) return {};
        for (bool down : { false, true })
            for (bool right : { false, true }) {
                const corner_crop_rule trying{ true, down, right };
                if (redraws_every_output(train, [&](const example& pair, grid& drawn) {
                        return trying.draw(pair.input, size, drawn);
                    })) return trying;
            }
        return {};
    }
};

// THE FRAME A SIZE CHANGE IS READ THROUGH: how the output's grid is made from
// the input's, so that what is left is a same-size puzzle. Either the piece a
// pick takes from a cut, cut out and moved to the corner of the canvas the size
// rule gives -- the pick, every other piece deleted, then Origin -- or the
// whole input scaled up `factor` times, scaled down `factor` times by the
// vote, tiled over the canvas, turned to the pose `factor` names, made into a
// fractal of itself, or reduced to the unit it repeats; or the pieces a pick
// takes laid out on graph paper of cells `factor` wide, by the bands they sit in;
// or the canvas filled with the one color the pieces a pick takes come out in.
struct piece_frame_rule {
    enum kind_of : int8_t { cut_out, scaled_up, scaled_down, tiled, posed, fractal, repeat_unit,
                            laid_out, filled } kind = cut_out;
    int8_t factor = 0;
    bool held = false;
    // Whether the pick takes several pieces in some pair, cut out together:
    // merged into one first (UnionList), so they keep their places to each other.
    bool merged = false;
    segmentation_methods cut = segmentation_methods::unset;
    piece_pick pick;
    // What the filled canvas's color is read off: each piece's own color, -1, or
    // the color the puzzle's table names for its key of this kind (shape_key).
    int8_t keyed = -1;
    bool holds() const { return held; }
};

// Collects facts, rules, and elections about one puzzle.
struct inference {
    output_dimension_rule size;
    color_facts colors;
    shape_color_map_rule shape_color_map;
    hole_color_rule hole_colors;
    unchanged_cells_facts unchanged;
    color_diff_facts color_diff;
    color_swap_candidates color_swaps;
    motif_continuation_rule motif_continuation;
    diagonal_continuation_rule diagonal_continuation;
    periodic_denoise_rule periodic_denoise;
    piece_slam_rule piece_slam;
    overlap_corner_rule overlap_corner;
    slide_to_touch_rule slide_to_touch;
    edge_slide_rule edge_slide;
    project_on_seeds_rule project_on_seeds;
    snap_to_anchors_rule snap_to_anchors;
    copy_to_rule copy_to;
    bridge_rule bridge;
    mark_lattice_rule mark_lattice;
    interior_spread_rule interior_spread;
    interior_fill_rule interior_fill;
    input_blocks_rule input_blocks;
    ranked_layout_rule ranked_layout;
    rarest_color_rule rarest_color;
    corner_crop_rule corner_crop;
    ray_extension_rule ray_extension;
    pane_split_facts pane_split;
    pane_mask_rule pane_mask;
    pane_pick_rule pane_pick;
    pane_gather_rule pane_gather;
    correspondence_election correspondence;
    // A SIZE CHANGE, READ THROUGH ITS FRAME: the frame, and everything read
    // from each framed input to its output, as a same-size puzzle of its own.
    piece_frame_rule frame;
    std::shared_ptr<const inference> framed;
    // Whether the cut's pieces, painted where they lie on the canvas the size
    // rule gives, draw every output: what a line with no words claims.
    bool pieces_redraw = true;
    // THE GROUND, OWNED ONCE (infer): what the program
    // deletes before anything else and fills after everything else. Everything
    // above was read from the figures that plan leaves, with no ground at all;
    // `ground_election` is the election the plan came from, for the readout.
    ground_plan plan;
    background_election ground_election;
};

// A SWAP A FIRED RULE ALREADY SAYS. Every cell it changes is one a rule on the
// elected reading accounts for, so saying it too would do that work twice -- on
// 009d5c81 the 0<->1 swap only ever painted away the key ShapeColorMap uses up.
// A swap that changes nothing at all is not one this can speak for.
inline bool swap_already_claimed(const inference& inf, int variant) {
    const auto [first, second] = color_pair::from_index(variant);
    const auto& pairs = inf.correspondence.chosen_pairs();
    const auto& lost = inf.color_diff.lost;
    bool changes_anything = false;
    for (size_t which = 0; which < pairs.size() && which < lost.size(); ++which)
        for (const int color : { first, second })
            for (const auto& [row, col] : lost[which][color]) {
                changes_anything = true;
                if (!ranges::any_of(pairs[which].left_over, [&](const auto& cell) {
                        return cell.from_input && cell.row == row && cell.col == col &&
                               cell.claimed_by >= 0; }))
                    return false;
            }
    return changes_anything;
}

// THE RULES THAT SAY A WHOLE PROGRAM OF THEIR OWN, each an explanation of every
// pair by itself and so a rival to the cut's own line rather than a part of it.
// Listed in this order, which is the order a tie between them goes by: a split
// is a fact about the frame, so the panes lead, and a pane chosen only for where
// it sits is the corner that crops it, so the corner comes before the pick.
enum class own_sentences : int8_t { panes, corner_crop, pane_pick, pane_gather, ray_extension,
    mark_lattice, input_blocks, periodic_denoise, bridge, interior_fill, interior_spread,
    hole_colors, ranked_layout, edge_slide, copy_to, rarest_color, motif_continuation, count };

// One rule for the elected reading's list: its name, what it says, how it is said.
inline correspondence_election::fired_rule listed(const char* name, const char* says,
                                                  correspondence_election::fired_rule::said_as as,
                                                  int8_t value = 0) {
    correspondence_election::fired_rule one;
    one.name = name;
    one.says = says;
    one.as = as;
    one.value = value;
    return one;
}

// EVERY RULE THAT SAYS A WHOLE PROGRAM, ON THE LIST. Each holds over every pair
// by itself, so it claims no cells from the cut's line: what it explains is
// everything, and it is weighed against that line whole.
inline void list_the_rivals(inference& inf) {
    auto& election = inf.correspondence;
    if (election.readings.empty()) return;
    auto& read = election.readings[election.chosen];
    using rule = correspondence_election::fired_rule;
    auto rival = [&](bool holds, own_sentences which, const char* name, const char* says) {
        if (!holds) return;
        rule one = listed(name, says, rule::own_sentences, (int8_t)which);
        one.stands_alone = true;
        read.fired.push_back(std::move(one));
    };
    const bool panes = inf.pane_split.holds();
    rival(panes && inf.pane_mask.holds(), own_sentences::panes, "Panes", "what the panes come to together");
    rival(inf.corner_crop.holds(), own_sentences::corner_crop, "Corner Crop", "the input seen through one corner");
    rival(panes && inf.pane_pick.holds(), own_sentences::pane_pick, "Pane Pick", "one pane is the answer");
    rival(panes && inf.pane_gather.holds(), own_sentences::pane_gather, "Pane Gather",
          "every pane's drawing where its pane was");
    rival(inf.ray_extension.holds(), own_sentences::ray_extension, "Rays", "pieces grow rays");
    rival(inf.mark_lattice.holds(), own_sentences::mark_lattice, "Mark Lattice", "marks laid as lines that take turns");
    rival(inf.input_blocks.holds(), own_sentences::input_blocks, "Copies Of The Input",
          "the output is the input laid out in blocks");
    rival(inf.periodic_denoise.holds(), own_sentences::periodic_denoise, "Denoise", "the repeats voted clean");
    rival(inf.bridge.holds(), own_sentences::bridge, "Bridge", "the gaps between pieces of one color filled");
    rival(inf.interior_fill.holds(), own_sentences::interior_fill, "Interior Fill", "every hollow filled with one color");
    rival(inf.interior_spread.holds(), own_sentences::interior_spread, "Interior Spread", "cores painted, outlines kept");
    rival(inf.hole_colors.holds(), own_sentences::hole_colors, "Hole Colors", "each piece recolored by its holes");
    rival(inf.ranked_layout.holds(), own_sentences::ranked_layout, "Ranked Layout", "pieces laid out by rank");
    rival(inf.edge_slide.holds(), own_sentences::edge_slide, "Edge Slide", "pieces slid to an edge");
    rival(inf.copy_to.holds(), own_sentences::copy_to, "Copy To", "the largest figure copied onto the others");
    rival(inf.rarest_color.holds(), own_sentences::rarest_color, "Rarest Color", "the least common color alone");
    rival(inf.motif_continuation.holds(), own_sentences::motif_continuation, "The Input Repeats",
          "the output carries the repeating block");
}

// THE PIECES A PICK BY HOLES OR BY SIZE TAKES, as the op takes them from the
// pieces as the cut hands them over: every one with `value` holes
// (object_hole_count, the count SelectObjectsWithHoles makes), or the one `value`
// down the order of painted cells, largest or smallest first, ties keeping the
// order the pieces came in -- the insertion sort SelectLargestObject and
// SelectSmallestObject run -- or the first `value` of that order, one selector
// a rank, or every one painting at least `value` cells, or every one with a
// cell on the row or the column `value` lines past the middle of all the
// pieces paint (line_from_middle). A pick's but_those turns any of them round.
inline vector<bool> picked_pieces(const vector<object>& pieces, int8_t how, int value) {
    vector<bool> picked(pieces.size(), false);
    if (how == piece_pick::middle_row || how == piece_pick::middle_column) {
        const bool rows = how == piece_pick::middle_row;
        int lo = MAX_GRID_DIM * 2, hi = -MAX_GRID_DIM * 2;
        for (const object& piece : pieces)
            piece.each_cell([&](int row, int col, int8_t) {
                lo = min(lo, rows ? row : col);
                hi = max(hi, rows ? row : col);
            });
        const int line = line_from_middle(lo, hi, value);
        for (size_t i = 0; i < pieces.size() && line >= 0; ++i)
            pieces[i].each_cell([&](int row, int col, int8_t) { picked[i] = picked[i] || (rows ? row : col) == line; });
        return picked;
    }
    if (how == piece_pick::most_common) {
        // The color most of the pieces' cells wear, the lowest on a tie, and every
        // piece wearing it -- the count SelectMostCommonColor makes.
        color_tally worn;
        for (const object& piece : pieces) piece.each_cell([&](int, int, int8_t color) { worn.add(color); });
        const int most = worn.most();
        if (most >= 0)
            for (size_t i = 0; i < pieces.size(); ++i)
                pieces[i].each_cell([&](int, int, int8_t color) { picked[i] = picked[i] || color == most; });
        return picked;
    }
    if (how == piece_pick::at_least_cells) {
        // Every piece painting at least `value` cells (paints_at_least, the count
        // SelectAtLeastCells makes).
        for (size_t i = 0; i < pieces.size(); ++i) {
            const object& piece = pieces[i];
            picked[i] = paints_at_least(piece.summary.h, piece.summary.w, [&](int row, int col) {
                return piece.cells[row][col] != (int8_t)TRANSPARENT; }, value);
        }
        return picked;
    }
    if (how == piece_pick::row_band || how == piece_pick::column_band) {
        // The band of rows or of columns each piece's box sits in, as
        // overlapping_bands groups them for SelectRow, SelectColumn and Layout.
        const int count = (int)pieces.size();
        if (count == 0) return picked;
        vector<int8_t> band(count);
        bands_of(count, how == piece_pick::row_band, [&](int i) { return pieces[(size_t)i].summary.box(); },
                 band.data());
        for (int i = 0; i < count; ++i) picked[i] = band[i] == value;
        return picked;
    }
    if (how == piece_pick::by_holes) {
        for (size_t i = 0; i < pieces.size(); ++i) {
            const object& piece = pieces[i];
            picked[i] = object_hole_count(piece.summary.h, piece.summary.w, [&](int row, int col) {
                return piece.cells[row][col] != (int8_t)TRANSPARENT; }) == value;
        }
        return picked;
    }
    vector<int> order(pieces.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    const bool largest = how == piece_pick::largest || how == piece_pick::largest_ranks;
    stable_sort(order.begin(), order.end(), [&](int one, int other) {
        return largest ? pieces[one].summary.count > pieces[other].summary.count
                       : pieces[one].summary.count < pieces[other].summary.count;
    });
    if (how == piece_pick::largest || how == piece_pick::smallest) {
        if (value >= 0 && value < (int)order.size()) picked[order[value]] = true;
        return picked;
    }
    for (int rank = 0; rank < value && rank < (int)order.size(); ++rank) picked[order[rank]] = true;
    return picked;
}

static_assert(row_of(node_type::WouldFit).high + 1 == (int)partner_by::count,
              "WouldFit takes one value for each way partner_by finds a partner");

inline vector<bool> picked_pieces(const vector<object>& pieces, const piece_pick& pick) {
    vector<bool> picked(pieces.size(), false);
    if (pick.how >= 0) {
        picked = picked_pieces(pieces, pick.how, pick.value);
        // Every piece but those, as InvertSelection turns them round.
        if (pick.but_those) picked.flip();
    } else
        for (size_t i = 0; i < pieces.size(); ++i) picked[i] = (palette(pieces[i]) & pick.colors) != 0;
    if (pick.fits < 0) return picked;
    // WouldFit keeps those that would fit into a gap of their partner among the
    // pieces not picked, every partner read off the pick as it stood, as the op
    // reads its marks before it drops any.
    const partner_pieces places(pieces, picked);
    for (size_t i = 0; i < pieces.size(); ++i)
        if (places.picked[i]) picked[i] = places.fits((partner_by)pick.fits, (int)i);
    return picked;
}

inline partner_pieces::partner_pieces(const vector<object>& cut, const piece_pick& pick)
    : partner_pieces(cut, picked_pieces(cut, pick)) {}

static_assert(row_of(node_type::SelectLargestObject).low == row_of(node_type::SelectSmallestObject).low &&
              row_of(node_type::SelectLargestObject).high == row_of(node_type::SelectSmallestObject).high,
              "the largest and the smallest are asked as far down the order");

// THE CELLS OF EVERY PAIR, a list each, (row, col): what a replay sees.
using cells_by_pair = vector<vector<std::pair<int, int>>>;

// Whether `cells` holds the cell at (row, col) of pair `which`.
inline bool on_cells(const cells_by_pair& cells, size_t which, int row, int col) {
    return which < cells.size() &&
           ranges::find(cells[which], std::pair<int, int>{ row, col }) != cells[which].end();
}

// WHAT EVERY REPLAY BELOW IS HANDED: the training pairs, each input's pieces as
// the cut hands them over, cut once, and the grounds the pairs were read on.
//
// AND WHAT EVERY ONE ASKS OF A WORD: that it is seen at work in every pair. A
// word that does nothing in some pair holds there only by standing idle -- a
// band of rows that pair's pieces do not make, a color it does not show -- and
// a word that holds by standing idle is a word for the pairs, not a rule: the
// test lays its bands and colors out afresh (22a4bbc2's rows, e3fe1151's
// colors). At work is the one thing every replay asks the same way: at some
// cell the word acts on, the output shows what its picked pieces did not show
// there before -- a piece recolored, moved, copied, grown or cut back, never
// only drawn again as it was. A copy laid on its own likeness is at work all
// the same: 4938f0c2's last pair is already whole.
struct replay_facts {
    const vector<example>& train;
    vector<vector<object>> cuts;
    bool overlays = true;
    // Whether `cuts` is still the hand the line leaves: false once a word on it
    // is one carry_the_hand cannot carry the pieces through.
    bool hand_known = true;
    // The pieces a named cut hands over, rather than the elected one.
    replay_facts(const vector<example>& pairs, segmentation_methods cut) : train(pairs) {
        for (const example& pair : train) {
            cuts.push_back(pieces_as_said(pair.input, cut));
            overlays = overlays && pair.input.same_size(pair.output);
        }
    }
    replay_facts(const inference& inf, const vector<example>& pairs)
        : replay_facts(pairs, inf.correspondence.segmentation) {}
};

// THE PIECES A PICK TAKES, pair by pair, made ready once for a word asked many
// ways over -- every length, both ways round, every lay -- whose tries differ in
// everything but the pieces. Kept only while that word is asked, and only while
// nothing carries the hand: `facts` must not change under it.
inline vector<partner_pieces> handed_pieces(const replay_facts& facts, const piece_pick& pick) {
    vector<partner_pieces> handed;
    handed.reserve(facts.cuts.size());
    for (const vector<object>& cut : facts.cuts) handed.emplace_back(cut, pick);
    return handed;
}

// Whether a pick by `colors` picks by the same colors in every pair: each pair's
// pieces hold the same ones of them. A pick whose colors change from pair to pair
// names each pair's own colors rather than a rule, and a test brings its own.
inline bool same_colors_every_pair(const replay_facts& facts, uint16_t colors) {
    bool first = true;
    uint16_t seen = 0;
    for (const auto& cut : facts.cuts) {
        uint16_t held = 0;
        for (const object& piece : cut) held = (uint16_t)(held | (palette(piece) & colors));
        if (first) seen = held;
        else if (held != seen) return false;
        first = false;
    }
    return true;
}

// Every pick a word is asked with: the colors an unsaid change starts from, when
// there are any, then every pick by holes, size, band, the color most cells
// wear and how many cells the ops can make, then
// each of `fit_colors` alone kept to its pieces that would fit into a gap of
// their partner, each partner way. Colors that change from pair to pair are
// asked last, so a pick by holes or size that accounts for as much wins the tie
// (aabf363d's shape, a new color in its test).
inline vector<piece_pick> picks_to_ask(const replay_facts& facts, uint16_t colors, uint16_t fit_colors = 0) {
    vector<piece_pick> picks;
    const bool steady = colors && same_colors_every_pair(facts, colors);
    if (steady) picks.push_back({ colors });
    const auto holes = row_of(node_type::SelectObjectsWithHoles), ranks = row_of(node_type::SelectLargestObject);
    for (int value = holes.low; value <= holes.high; ++value) picks.push_back({ 0, piece_pick::by_holes, (int8_t)value });
    // By size, the nearer an end the plainer: the largest and the smallest, then
    // the second of each, so where two take the same pieces in every pair the
    // plainer is said (62ab2642's "smallest", not "ninth largest", which only
    // the pairs' own counts made the same).
    for (int value = ranks.low; value <= ranks.high; ++value)
        for (const int8_t how : { (int8_t)piece_pick::largest, (int8_t)piece_pick::smallest }) picks.push_back({ 0, how, (int8_t)value });
    // The bands, as many as some pair's pieces make.
    for (const int8_t how : { (int8_t)piece_pick::row_band, (int8_t)piece_pick::column_band }) {
        int bands = 0;
        for (const auto& cut : facts.cuts)
            for (int value = 0; value < (int)cut.size(); ++value)
                if (ranges::any_of(picked_pieces(cut, how, value), [](bool one) { return one; }))
                    bands = max(bands, value + 1);
        for (int value = 0; value < bands && value <= row_of(node_type::SelectRow).high; ++value)
            picks.push_back({ 0, how, (int8_t)value });
    }
    for (const bool but_those : { false, true }) picks.push_back({ 0, piece_pick::most_common, 0, -1, but_those });
    // By how many cells, at least so many or fewer, each count asked only where
    // it splits some pair's pieces: one at least taken there and one left.
    const auto cells = row_of(node_type::SelectAtLeastCells);
    for (int value = cells.low; value <= cells.high; ++value) {
        if (ranges::none_of(facts.cuts, [&](const vector<object>& cut) {
                const vector<bool> taken = picked_pieces(cut, piece_pick::at_least_cells, value);
                return ranges::find(taken, true) != taken.end() && ranges::find(taken, false) != taken.end();
            }))
            continue;
        for (const bool but_those : { false, true })
            picks.push_back({ 0, piece_pick::at_least_cells, (int8_t)value, -1, but_those });
    }
    // The first so many by size, largest or smallest first, and every piece but
    // those (9caf5b84: every color but the two most common becomes 7), each asked
    // only where in every pair the order parts there, the last piece taken
    // painting more cells than the first left -- or fewer -- so no tie the order
    // the pieces came in breaks says which. The first one alone is the pick by
    // size above, so the first so many are asked from two.
    for (const bool largest : { true, false })
        for (int value = 1; value <= ranks.high + 1; ++value) {
            if (!ranges::all_of(facts.cuts, [&](const vector<object>& cut) {
                    if ((int)cut.size() <= value) return false;
                    vector<int> counts;
                    for (const object& piece : cut) counts.push_back(piece.summary.count);
                    if (largest) ranges::sort(counts, greater<>());
                    else ranges::sort(counts);
                    return counts[(size_t)value - 1] != counts[(size_t)value];
                }))
                continue;
            const int8_t first_so_many = largest ? piece_pick::largest_ranks : piece_pick::smallest_ranks;
            if (value >= 2) picks.push_back({ 0, first_so_many, (int8_t)value });
            picks.push_back({ 0, first_so_many, (int8_t)value, -1, true });
        }
    // The row or the column so many lines past the middle of all the pieces
    // paint, and every piece but those (d23f8c26 keeps its middle column), the
    // middle first and then out a line at a time, each asked only where every
    // pair has that line and it splits some pair's pieces: one at least taken
    // there and one left.
    constexpr auto from_middle = row_of(node_type::SelectRowFromMiddle);
    static_assert(from_middle.high == (MAX_GRID_DIM - 1) / 2 && from_middle.low == -from_middle.high &&
                  row_of(node_type::SelectColumnFromMiddle).low == from_middle.low &&
                  row_of(node_type::SelectColumnFromMiddle).high == from_middle.high,
                  "a line from the middle reaches either edge of the largest grid");
    for (const bool rows : { true, false })
        for (int at = 0; at <= 2 * from_middle.high; ++at) {
            // 0, -1, 1, -2, 2, ...
            const int value = (at + 1) / 2 * (at % 2 ? -1 : 1);
            const int8_t how = rows ? piece_pick::middle_row : piece_pick::middle_column;
            bool every = true, splits = false;
            for (const auto& cut : facts.cuts) {
                const vector<bool> taken = picked_pieces(cut, how, value);
                const bool some = ranges::find(taken, true) != taken.end();
                every = every && some;
                splits = splits || (some && ranges::find(taken, false) != taken.end());
            }
            if (!every || !splits) continue;
            for (const bool but_those : { false, true }) picks.push_back({ 0, how, (int8_t)value, -1, but_those });
        }
    // A fit is asked only where it is at work in every pair: it keeps one piece
    // of the color at least and drops one, or the color alone says as much.
    for (int color = 0; color < COLOR_COUNT; ++color) {
        if (!(fit_colors >> color & 1)) continue;
        const piece_pick alone = piece_pick::of_color(color);
        for (int way = 0; way < (int)partner_by::count; ++way) {
            const piece_pick fit{ alone.colors, -1, 0, (int8_t)way };
            if (ranges::all_of(facts.cuts, [&](const vector<object>& cut) {
                    const vector<bool> kept = picked_pieces(cut, fit);
                    return ranges::find(kept, true) != kept.end() && kept != picked_pieces(cut, alone);
                }))
                picks.push_back(fit);
        }
    }
    if (colors && !steady) picks.push_back({ colors });
    return picks;
}

// Whether two pieces draw the same thing: the same cells in the same colors,
// wherever each lies.
inline bool same_drawing(const object& one, const object& other) {
    if (one.summary.h != other.summary.h || one.summary.w != other.summary.w) return false;
    for (int row = 0; row < one.summary.h; ++row)
        for (int col = 0; col < one.summary.w; ++col)
            if (one.cells[row][col] != other.cells[row][col]) return false;
    return true;
}

// WHETHER A CANVAS SHOWS WHAT IT FRAMES: all of `drawing`, or the corner of it
// that says the rest -- half of it as tall or as wide, rounded up, on every
// side the canvas cuts, the drawing coming back on itself there: mirrored
// across the cut, or a half turn for a half, a quarter turn for a quarter, as
// 2013d3e2's figure does in its corner. Any other cut shows whatever happened
// to lie in the corner: 39a8645d's shapes, laid out, with one of them there, or
// its layer, scattered, with one copy in it.
inline bool shows_what_it_frames(const grid& drawing, int height, int width) {
    const int tall = drawing.height, wide = drawing.width;
    const bool cut_down = height < tall, cut_across = width < wide;
    if (!cut_down && !cut_across) return true;
    auto half = [](int whole, int kept) { return (whole + 1) / 2 == kept; };
    if ((cut_down && !half(tall, height)) || (cut_across && !half(wide, width))) return false;
    // Whether the drawing is the same read through `at`, cell for cell.
    auto comes_back = [&](auto at) {
        for (int row = 0; row < tall; ++row)
            for (int col = 0; col < wide; ++col) {
                const auto [from_row, from_col] = at(row, col);
                if (drawing.data[row][col] != drawing.data[from_row][from_col]) return false;
            }
        return true;
    };
    const bool across_mirror = comes_back([&](int row, int col) { return std::pair{ row, wide - 1 - col }; });
    const bool down_mirror = comes_back([&](int row, int col) { return std::pair{ tall - 1 - row, col }; });
    const bool half_turn = comes_back([&](int row, int col) { return std::pair{ tall - 1 - row, wide - 1 - col }; });
    if (cut_down && cut_across)
        return (across_mirror && down_mirror) ||
               (tall == wide && comes_back([&](int row, int col) { return std::pair{ col, wide - 1 - row }; }));
    return half_turn || (cut_across ? across_mirror : down_mirror);
}

// Whether a pick by size took piece `one` on a tie with a piece that draws
// something else: the op breaks the tie by where the pieces lie, so what it
// takes is the order's to say and not the rule's.
inline bool size_pick_ties(const vector<object>& pieces, const piece_pick& pick, int one) {
    if (pick.how != piece_pick::largest && pick.how != piece_pick::smallest) return false;
    for (size_t i = 0; i < pieces.size(); ++i)
        if ((int)i != one && pieces[i].summary.count == pieces[one].summary.count &&
            !same_drawing(pieces[i], pieces[one]))
            return true;
    return false;
}

// THE INPUT FRAMED: the pieces a pick takes, moved as one to the corner of the
// canvas the size rule gives, and ground everywhere else -- what the frame's
// words leave on the canvas. Several go together, keeping their places to each
// other, as UnionList merges them; `several` says whether this pair's pick took
// more than one. False when the pick takes no piece; when the canvas does not
// show what it frames (shows_what_it_frames); and when a pick by size lands on a
// tie (size_pick_ties), since the crop would be of whichever came first.
inline bool framed_input(const vector<object>& pieces, const piece_pick& pick, const grid& input,
                         const output_dimension_rule& size, grid& framed, bool& several) {
    const vector<bool> picked = picked_pieces(pieces, pick);
    cell_box box;
    int taken = 0, one = -1;
    for (size_t i = 0; i < picked.size(); ++i) {
        if (!picked[i]) continue;
        ++taken;
        one = (int)i;
        box.include(pieces[i].summary.box());
    }
    several = taken > 1;
    if (taken == 0 || size_pick_ties(pieces, pick, one)) return false;
    grid drawing = empty_grid(box.width(), box.height());
    for (size_t i = 0; i < pieces.size(); ++i)
        if (picked[i])
            pieces[i].each_cell([&](int row, int col, int8_t color) {
                drawing.data[row - box.row0][col - box.col0] = color;
            });
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        if (!shows_what_it_frames(drawing, canvas.height, canvas.width)) return false;
        for (int row = 0; row < drawing.height && row < canvas.height; ++row)
            for (int col = 0; col < drawing.width && col < canvas.width; ++col)
                if (drawing.data[row][col] != TRANSPARENT) canvas.data[row][col] = drawing.data[row][col];
        return true;
    });
}

// THE PIECES LAID OUT: the ones a pick takes, as many as one list holds, each
// drawn unresized into its own cell of graph paper `cell` wide -- or, for 0, as
// wide as the widest and as tall as the tallest -- in the row band and column
// band its box sits in (overlapping_bands, the grouping Layout and the band
// picks read), clipped to the cell, the paper laid at the corner of the canvas
// -- where each cell shows the piece it frames, and the canvas the paper
// (shows_what_it_frames).
inline bool laid_out_input(const vector<object>& pieces, const piece_pick& pick, int cell, const grid& input,
                           const output_dimension_rule& size, grid& framed) {
    const vector<bool> picked = picked_pieces(pieces, pick);
    vector<const object*> kept;
    for (size_t i = 0; i < pieces.size() && kept.size() < (size_t)MAX_OBJECTS_IN_A_LIST; ++i)
        if (picked[i]) kept.push_back(&pieces[i]);
    const int count = (int)kept.size();
    if (count == 0) return false;
    int cell_width = cell, cell_height = cell;
    if (cell == 0)
        for (const object* piece : kept) {
            cell_width = max(cell_width, (int)piece->summary.w);
            cell_height = max(cell_height, (int)piece->summary.h);
        }
    if (cell_width <= 0 || cell_height <= 0) return false;
    vector<int8_t> row_band(count), col_band(count);
    auto kept_box = [&](int i) { return kept[(size_t)i]->summary.box(); };
    const int rows = bands_of(count, true, kept_box, row_band.data());
    const int cols = bands_of(count, false, kept_box, col_band.data());
    const int paper_width = min(cols * cell_width, MAX_GRID_DIM), paper_height = min(rows * cell_height, MAX_GRID_DIM);
    grid paper = empty_grid(paper_width, paper_height);
    for (int i = 0; i < count; ++i) {
        const object& piece = *kept[i];
        grid drawing(piece.summary.w, piece.summary.h);
        for (int row = 0; row < piece.summary.h; ++row)
            for (int col = 0; col < piece.summary.w; ++col) drawing.data[row][col] = piece.cells[row][col];
        if (!shows_what_it_frames(drawing, cell_height, cell_width)) return false;
        const int top = row_band[i] * cell_height, left = col_band[i] * cell_width;
        for (int row = 0; row < cell_height && row < piece.summary.h; ++row)
            for (int col = 0; col < cell_width && col < piece.summary.w; ++col) {
                const int at_row = top + row, at_col = left + col;
                const int8_t color = piece.cells[row][col];
                if (at_row < paper_height && at_col < paper_width && color != (int8_t)TRANSPARENT)
                    paper.data[at_row][at_col] = color;
            }
    }
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        if (!shows_what_it_frames(paper, canvas.height, canvas.width)) return false;
        for (int row = 0; row < paper.height && row < canvas.height; ++row)
            for (int col = 0; col < paper.width && col < canvas.width; ++col)
                if (paper.data[row][col] != TRANSPARENT) canvas.data[row][col] = paper.data[row][col];
        return true;
    });
}

// THE CANVAS FILLED, as the filled frame's words leave it: the pieces a pick
// takes, each recolored by the puzzle's table where `keyed` names a key
// (ShapeColorMap[keyed]), filled to its box with its most common color
// (AddColor[11]) and stretched over the canvas the size rule gives (Fit[1]),
// laid in turn. Held only where every piece the pick takes comes out in one
// color, the same for all, so the canvas is that color whichever lies last;
// false when the pick takes no piece, or a pick by size lands on a tie
// (size_pick_ties).
inline bool filled_input(const vector<object>& pieces, const piece_pick& pick, int8_t keyed,
                         const shape_color_map_rule& table, const grid& input, const output_dimension_rule& size,
                         grid& framed) {
    const vector<bool> picked = picked_pieces(pieces, pick);
    int color = -1, one = -1;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (!picked[i]) continue;
        one = (int)i;
        // The one color the piece comes out in: -2 until a cell is read, -1 once
        // two disagree.
        int worn = -2;
        if (keyed >= 0) worn = table.color_of((shape_key)keyed, pieces[i]);
        else pieces[i].each_cell([&](int, int, int8_t was) { worn = worn == -2 || worn == was ? was : -1; });
        if (!is_color(worn) || (color >= 0 && worn != color)) return false;
        color = worn;
    }
    if (!is_color(color) || size_pick_ties(pieces, pick, one)) return false;
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        canvas.fill((int8_t)color);
        return true;
    });
}

// THE WHOLE INPUT FRAMED: scaled up `factor` times, each cell a block, as ScaleUp
// scales the input's one object; scaled down `factor` times, each block the
// color most of it wears, as ScaleDown votes (majority_vote); or tiled from the
// corner over the canvas the size rule gives, a copy per input, as Tile[0] lays
// them -- as many as one list holds; turned to a pose (posed_place, the pose the
// turns and flips said for it make); a fractal, each cell of the input scaled
// up to a copy of the input where it holds a figure, as Fractal draws the input
// without its ground; or the unit it repeats (repeat_periods), each cell the
// color most of its places wear, the first to lead on a tie, as Pattern votes.
// What falls off the canvas is clipped. Where a plan deleted the ground, the
// input framed is its figures alone, and every frame acts on them as they are.
inline bool framed_whole(const grid& input, const piece_frame_rule& frame, const output_dimension_rule& size,
                         grid& framed) {
    using kind = piece_frame_rule::kind_of;
    const int factor = frame.factor;
    const bool drew = draw_on_canvas(input, size, framed, [&](grid& canvas) {
        switch (frame.kind) {
        case kind::posed:
            for (int row = 0; row < input.height; ++row)
                for (int col = 0; col < input.width; ++col) {
                    const auto [at_row, at_col] = posed_place(factor, input.width, input.height, row, col);
                    if (canvas.contains(at_row, at_col)) canvas.data[at_row][at_col] = input.data[row][col];
                }
            return true;
        case kind::fractal: {
            // A figure is what is not empty, as a plan that deleted the ground
            // leaves the input.
            auto figure = [&](int row, int col) { return input.data[row][col] != TRANSPARENT; };
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    if (figure(row * input.height / canvas.height, col * input.width / canvas.width) &&
                        figure(row % input.height, col % input.width))
                        canvas.data[row][col] = input.data[row % input.height][col % input.width];
            return true;
        }
        case kind::repeat_unit: {
            int unit_width = 0, unit_height = 0;
            repeat_periods(input, TRANSPARENT, unit_width, unit_height, /*must_fit_twice=*/false);
            for (int row = 0; row < unit_height; ++row)
                for (int col = 0; col < unit_width; ++col) {
                    if (!canvas.contains(row, col)) return false;
                    int counts[COLOR_COUNT] = {}, best = 0, worn = TRANSPARENT;
                    for (int at_row = row; at_row < input.height; at_row += unit_height)
                        for (int at_col = col; at_col < input.width; at_col += unit_width) {
                            const int color = input.data[at_row][at_col];
                            if (!is_color(color)) continue;
                            if (++counts[color] > best) { best = counts[color]; worn = color; }
                        }
                    canvas.data[row][col] = (int8_t)worn;
                }
            return true;
        }
        case kind::scaled_up:
            if (factor < 2) return false;
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    if (row / factor < input.height && col / factor < input.width)
                        canvas.data[row][col] = input.data[row / factor][col / factor];
            return true;
        case kind::scaled_down:
            if (factor < 2) return false;
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    canvas.data[row][col] = majority_vote(input.data, row, col, factor, input.height, input.width);
            return true;
        case kind::tiled: {
            const int down = (canvas.height + input.height - 1) / input.height;
            const int across = (canvas.width + input.width - 1) / input.width;
            if (down * across > MAX_OBJECTS_IN_A_LIST) return false;
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    canvas.data[row][col] = input.data[row % input.height][col % input.width];
            return true;
        }
        default: return false;
        }
    });
    // A TILED INPUT IS COPIES OF THE INPUT'S PIECES, each piece cut on its own
    // and laid down again in every copy, as its words cut the input before they
    // tile it (frame_the_sentences): no piece joins one in the next copy.
    if (drew && frame.kind == kind::tiled) {
        framed.tile_width = input.width;
        framed.tile_height = input.height;
    }
    return drew;
}

// THE FRAME A SIZE CHANGE IS READ THROUGH, chosen. Of every cut with a word and
// every pick a word is asked with -- each color on its own, then the picks by
// holes, size, band and the color most cells wear -- the one whose framed
// inputs share the most cells with their outputs, counted in the pair that
// shares the fewest; a tie goes to the cut and the pick asked first. And ahead
// of them the whole input scaled or tiled, where the size rule scales it; and
// after them all the canvas filled in one color. Asked only of a size change,
// and held only where the frame lands on the output's size in every pair. A
// filled frame that reads its color off a table read off the pairs makes that
// the puzzle's table (inf.shape_color_map), the one ShapeColorMap looks up.
inline piece_frame_rule choose_piece_frame(const vector<example>& train, inference& inf) {
    if (train.empty() || inf.size.Chosen == output_dimension_rule::Identity) return {};
    // The framed grid is of figures: no ground is filled into it, so what a
    // frame leaves unpainted is empty (draw_on_canvas).
    // A frame that leaves a cell empty is right only where the output is
    // empty -- what a plan fills last. An output with no empty cell at all has
    // nothing to fill one, so there such a frame is no frame; elsewhere an empty
    // cell facing a color is one more cell the frame does not share, which the
    // reading through it may still explain (28bf18c6's copies).
    auto shared = [](const grid& one, const grid& other) {
        bool other_has_empty = false;
        for (int row = 0; row < other.height && !other_has_empty; ++row)
            for (int col = 0; col < other.width && !other_has_empty; ++col)
                other_has_empty = other.data[row][col] == TRANSPARENT;
        int same = 0;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col) {
                if (one.data[row][col] == TRANSPARENT && !other_has_empty) return 0.0;
                same += one.data[row][col] == other.data[row][col];
            }
        return (double)same / (one.width * one.height);
    };
    piece_frame_rule best;
    double best_share = 0;
    auto weigh = [&](const piece_frame_rule& frame, auto framing) {
        double share = 1;
        for (size_t which = 0; which < train.size() && share > best_share; ++which) {
            grid framed;
            if (!framing(which, framed) || !framed.same_size(train[which].output)) {
                share = 0;
                break;
            }
            share = min(share, shared(framed, train[which].output));
        }
        if (share > best_share) {
            best_share = share;
            best = frame;
        }
    };
    // The whole input, where the size rule scales it: up or down by one whole
    // factor both ways, which ScaleUp and ScaleDown take; tiled by any whole count.
    vector<piece_frame_rule> wholes;
    if (inf.size.Chosen == output_dimension_rule::Scale) {
        const auto& ratio = inf.size;
        const bool square = ratio.width_numerator * ratio.height_denominator ==
                            ratio.height_numerator * ratio.width_denominator;
        const auto up = row_of(node_type::ScaleUp), down = row_of(node_type::ScaleDown);
        if (square && ratio.width_denominator == 1 && ratio.width_numerator >= up.low &&
            ratio.width_numerator <= up.high)
            wholes.push_back({ piece_frame_rule::scaled_up, (int8_t)ratio.width_numerator, true });
        if (square && ratio.width_numerator == 1 && ratio.width_denominator >= down.low &&
            ratio.width_denominator <= down.high)
            wholes.push_back({ piece_frame_rule::scaled_down, (int8_t)ratio.width_denominator, true });
        if (ratio.width_denominator == 1 && ratio.height_denominator == 1)
            wholes.push_back({ piece_frame_rule::tiled, 0, true });
        for (const piece_frame_rule& frame : wholes)
            weigh(frame, [&](size_t which, grid& framed) {
                return framed_whole(train[which].input, frame, inf.size, framed);
            });
    }
    // The whole input turned, where every output's sides are its input's traded;
    // a fractal of itself, where every output is its input times its own size;
    // or the unit it repeats. Asked whatever the size rule reads, since the rule
    // cannot say "the input's own size" and the canvas it gives is checked anyway.
    vector<piece_frame_rule> reads;
    if (ranges::all_of(train, [](const example& pair) {
            return pair.output.width == pair.input.height && pair.output.height == pair.input.width; }))
        for (const int pose : { 1, 3, 5, 7 }) reads.push_back({ piece_frame_rule::posed, (int8_t)pose, true });
    if (ranges::all_of(train, [](const example& pair) {
            return pair.output.width == pair.input.width * pair.input.width &&
                   pair.output.height == pair.input.height * pair.input.height; }))
        reads.push_back({ piece_frame_rule::fractal, 0, true });
    reads.push_back({ piece_frame_rule::repeat_unit, 0, true });
    for (const piece_frame_rule& frame : reads)
        weigh(frame, [&](size_t which, grid& framed) {
            return framed_whole(train[which].input, frame, inf.size, framed);
        });
    static constexpr segmentation_methods frame_cuts[] = { segmentation_methods::connected_multicolor,
                                                           segmentation_methods::connected_monochrome,
                                                           segmentation_methods::all_cells_of_one_color };
    // Every piece first, then each color on its own, then the picks by kind.
    auto picks_of = [&](const replay_facts& facts) {
        vector<piece_pick> picks = { piece_pick::all() };
        for (int color = 0; color < COLOR_COUNT; ++color) picks.push_back(piece_pick::of_color(color));
        for (const piece_pick& pick : picks_to_ask(facts, 0)) picks.push_back(pick);
        return picks;
    };
    for (const segmentation_methods cut : frame_cuts) {
        const replay_facts facts(train, cut);
        const vector<piece_pick> picks = picks_of(facts);
        for (const piece_pick& pick : picks) {
            piece_frame_rule frame;
            frame.held = true;
            frame.cut = cut;
            frame.pick = pick;
            for (size_t which = 0; which < train.size(); ++which) {
                grid framed;
                bool several = false;
                if (!framed_input(facts.cuts[which], pick, train[which].input, inf.size, framed,
                                  several))
                    break;
                frame.merged = frame.merged || several;
            }
            weigh(frame, [&](size_t which, grid& framed) {
                bool several = false;
                return framed_input(facts.cuts[which], pick, train[which].input, inf.size, framed,
                                    several);
            });
        }
        // Laid out on graph paper: every piece, or the pieces of one color, in
        // cells of every size Layout takes, the size read off the pieces first.
        const auto cells = row_of(node_type::Layout);
        for (size_t which_pick = 0; which_pick <= (size_t)COLOR_COUNT; ++which_pick) {
            const piece_pick pick = picks[which_pick];
            for (int cell = cells.low; cell <= cells.high; ++cell) {
                piece_frame_rule frame;
                frame.kind = piece_frame_rule::laid_out;
                frame.factor = (int8_t)cell;
                frame.held = true;
                frame.cut = cut;
                frame.pick = pick;
                weigh(frame, [&](size_t which, grid& framed) {
                    return laid_out_input(facts.cuts[which], pick, cell, train[which].input, inf.size, framed);
                });
            }
        }
    }
    // THE CANVAS IN ONE COLOR, the color a word names off the input, where every
    // output is one color all over: the pieces a pick takes, each in one color --
    // its own, or the one its shape or its symmetry names in a table read off the
    // pairs (ShapeColorMap, read_off) -- filled to its box and stretched over the
    // canvas, as a plan lays its fill (filled_input). Each piece's own color is
    // asked before a table; a table only while the puzzle has none, and only
    // where it is at work in every pair: some piece it takes comes out in a
    // color it did not wear (44f52bb0's picture, 1 where a turn or flip leaves
    // it as it was, else 7). The size rule gives the canvas, so not under
    // BoundingBox.
    const bool one_color = ranges::all_of(train, [](const example& pair) {
        const int first = pair.output.data[0][0];
        for (int row = 0; row < pair.output.height; ++row)
            for (int col = 0; col < pair.output.width; ++col)
                if (pair.output.data[row][col] != first) return false;
        return is_color(first);
    });
    shape_color_map_rule best_table;
    if (one_color && inf.size.Chosen != output_dimension_rule::BoundingBox) {
        const int last_kind = inf.shape_color_map.holds() ? -1 : (int)shape_key::count - 1;
        vector<uint16_t> shown;
        for (const example& pair : train) shown.push_back(shape_color_map_rule::colors_shown(pair.input));
        for (const segmentation_methods cut : frame_cuts) {
            const replay_facts facts(train, cut);
            for (const piece_pick& pick : picks_of(facts))
                for (int keyed = -1; keyed <= last_kind; ++keyed) {
                    shape_color_map_rule table;
                    if (keyed >= 0) {
                        vector<shape_color_map_rule::seen_key> seen;
                        bool learned = true;
                        for (size_t which = 0; which < train.size() && learned; ++which) {
                            const int8_t wanted = (int8_t)train[which].output.data[0][0];
                            const vector<object>& pieces = facts.cuts[which];
                            const vector<bool> picked = picked_pieces(pieces, pick);
                            bool at_work = false;
                            for (size_t i = 0; i < pieces.size() && learned; ++i) {
                                if (!picked[i]) continue;
                                learned = shape_color_map_rule::see(
                                    seen, shape_color_map_rule::key_of((shape_key)keyed, pieces[i]), wanted, which,
                                    shown[which]);
                                pieces[i].each_cell([&](int, int, int8_t was) { at_work = at_work || was != wanted; });
                            }
                            learned = learned && at_work;
                        }
                        if (!learned) continue;
                        table = shape_color_map_rule::read_off(seen);
                        if (!table.holds()) continue;
                    }
                    piece_frame_rule frame;
                    frame.kind = piece_frame_rule::filled;
                    frame.held = true;
                    frame.cut = cut;
                    frame.pick = pick;
                    frame.keyed = (int8_t)keyed;
                    const double before = best_share;
                    weigh(frame, [&](size_t which, grid& framed) {
                        return filled_input(facts.cuts[which], pick, (int8_t)keyed, table, train[which].input,
                                            inf.size, framed);
                    });
                    if (best_share > before) best_table = table;
                }
        }
    }
    if (best.kind == piece_frame_rule::filled && best.keyed >= 0) inf.shape_color_map = best_table;
    return best;
}

// WHAT IS DOWN, pair by pair, where a paint lands: whether each cell has
// something on it already.
struct cells_down { bool at[MAX_GRID_DIM][MAX_GRID_DIM] = {}; };

// The pieces `facts` hands over, where they lie, as what is down.
inline vector<cells_down> pieces_down(const replay_facts& facts) {
    vector<cells_down> down(facts.cuts.size());
    for (size_t which = 0; which < facts.cuts.size(); ++which)
        for (const object& piece : facts.cuts[which])
            piece.each_cell([&](int row, int col, int8_t) {
                if (facts.train[which].input.contains(row, col)) down[which].at[row][col] = true;
            });
    return down;
}

// Whether a paint laid `lay` lands on a cell that has something down or not:
// over it lands everywhere, under it only where nothing is, as Underlay paints,
// and onto it only where something is, as MaskBy[10] keeps.
inline bool lands(int8_t lay, bool down) {
    using rule = correspondence_election::fired_rule;
    return lay == rule::laid_under ? !down : lay == rule::laid_onto ? down : true;
}

// HOW A PAINT IS LAID: the way `lay` says (lands) on what `canvas` holds down,
// pair by pair -- nothing, where there is no canvas -- and in the colors it
// draws, or all in one color -- the one the output shows where it first lands,
// the line's Recolor after the word, whatever the colors drawn -- or in the
// colors it draws where the output agrees, and else in one.
struct laying {
    int8_t lay = correspondence_election::fired_rule::laid_over;
    const vector<cells_down>* canvas = nullptr;
    enum colors : int8_t { as_drawn, in_one, own_else_one } in = as_drawn;
};

// WHAT A REPLAY SAW, where its word holds -- every replay below hands one back,
// or nothing where the word does not hold. The cells the word lays down on each
// output, pair by pair: where its paint landed, where its pieces landed, what
// they came out in or kept. The cells its picked pieces stood on that it took
// them off or left as they were: where they slid from, what they lost, the
// pieces a copy was drawn from. How many pieces it changed; the one color it
// paints in, -1 for the colors it draws; and whether it picked every piece.
struct replayed {
    cells_by_pair drawn, left;
    int pieces = 0;
    int8_t paint = -1;
    bool every_piece = true;
};

// A PAINT LAID ON WHAT IS DOWN, replayed: what every replay of a word that
// draws asks, once. On each pair, `draw(which, cut, put)` hands every cell its
// picked pieces paint, in the color it paints it, to `put(row, col, color)`,
// which answers whether that pair's output shows what has landed so far;
// `draw` answers false for a pair it cannot draw at all. A cell lands the way
// the paint is laid -- under what is down, the first paint to reach a cell
// keeps it, this one's own earlier cells too, as Underlay paints -- and laid in
// one color, every cell is the color the output shows where the first lands --
// a paint is never no color -- the same in every pair. It
// explains the drawing only if the output shows every cell that lands, and in
// each pair the word is at work (replay_facts): one cell at least lands in a
// color its picked pieces did not show there. Sees that one color (-1 laid in
// the colors drawn) and the cells that landed. Handed the pieces its pick takes
// (handed_pieces), made ready once for every way the word is asked.
template <class Draw>
inline optional<replayed> paint_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      const laying& how, Draw draw) {
    using rule = correspondence_election::fired_rule;
    if (how.in == laying::own_else_one) {
        if (auto seen = paint_holds(facts, handed, { how.lay, how.canvas }, draw)) return seen;
        return paint_holds(facts, handed, { how.lay, how.canvas, laying::in_one }, draw);
    }
    if (!facts.overlays || facts.train.empty()) return nullopt;
    replayed seen;
    seen.drawn.assign(facts.train.size(), {});
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const example& pair = facts.train[which];
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        cells_down down = how.canvas ? (*how.canvas)[which] : cells_down{};
        // What the picked pieces show, where they lie: drawn again as it is, a
        // cell of them is no work.
        int8_t before[MAX_GRID_DIM][MAX_GRID_DIM];
        for (auto& row : before) ranges::fill(row, (int8_t)TRANSPARENT);
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i])
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    if (pair.input.contains(row, col)) before[row][col] = color;
                });
        bool shows = true, at_work = false;
        const bool drew = draw(which, cut, [&](int row, int col, int8_t color) {
            if (!lands(how.lay, down.at[row][col])) return shows;
            if (how.lay == rule::laid_under) down.at[row][col] = true;
            if (how.in == laying::in_one) {
                if (seen.paint < 0) seen.paint = (int8_t)pair.output.data[row][col];
                color = seen.paint;
                shows = shows && is_color(seen.paint);
            }
            shows = shows && pair.output.data[row][col] == color;
            seen.drawn[which].push_back({ row, col });
            at_work = at_work || before[row][col] != color;
            return shows;
        });
        if (!drew || !shows || !at_work) return nullopt;
    }
    if (how.in != laying::as_drawn && !is_color(seen.paint)) return nullopt;
    return seen;
}

// The color a recolor's replay names for a piece the op leaves as it is.
constexpr int8_t LEFT_AS_IT_IS = -2;

// PICKED PIECES REPAINTED WHERE THEY LIE, replayed: what every recolor's replay
// asks, once. On each pair, `color_of(which, cut, i, color)` says the color
// picked piece i comes out in -- LEFT_AS_IT_IS where the word names it none and
// the op leaves it as it is (ColorFrom's piece with no partner; ShapeColorMap's
// key the table does not name) -- or false where the word cannot say; the
// piece is then a paint laid over itself in that color, or in its own colors
// (paint_holds), so a piece the word leaves as it is has to come out as it
// was: the word picked it, and does not get to leave it to another (4f537728's
// blocks in the marker's column, left by "takes the color of a piece in its
// rows", came out recolored). Sees the cells of the pieces it repainted and
// how many pieces changed color.
template <class ColorOf>
inline optional<replayed> recolors_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                        ColorOf color_of) {
    int pieces = 0;
    auto seen = paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            int8_t color = -1;
            if (!color_of(which, cut, i, color)) return false;
            bool changed = false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t was) {
                const int8_t now = color == LEFT_AS_IT_IS ? was : color;
                put(row, col, now);
                changed = changed || was != now;
            });
            pieces += changed;
        }
        return true;
    });
    if (seen) seen->pieces = pieces;
    return seen;
}

// A PIECE THAT TAKES ITS COLOR FROM ANOTHER, replayed. ColorFrom paints every
// picked piece in the color of its partner among the rest, and leaves one with
// no partner as it is (recolors_hold: b7256dcd's shape with no dot at its side
// stays 6). A piece whose partners disagree the op leaves too, but the word
// cannot say it: which color the rule means there, no pair shows (d23f8c26's
// middle column, left by "takes the color of a piece in its columns", read as
// a rule that kept it, and was wrong on its test).
inline optional<replayed> color_from_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                           partner_by how) {
    return recolors_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, int8_t& color) {
        const int worn = cut.color_from(how, i);
        color = worn >= 0 ? (int8_t)worn : LEFT_AS_IT_IS;
        return worn >= 0 || !cut.partnered(how, i);
    });
}
static_assert(row_of(node_type::ColorFrom).high + 1 == (int)partner_by::count,
              "ColorFrom takes one value for each way partner_by finds a partner");

// A RECOLOR A PICK DECIDES, replayed: every piece picked comes out wholly in
// `color` (recolors_hold).
inline optional<replayed> picked_recolor_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                               int8_t color) {
    return recolors_hold(facts, handed, [&](size_t, const partner_pieces&, int, int8_t& to) {
        to = color;
        return true;
    });
}

// SIZE RANK, replayed. SizeRankColorMap takes the pieces as the line hands them
// over -- a layer of one color split into its connected pieces first -- sorts
// them biggest first, ties keeping that order, and paints the first the
// output's most common color, the next the next, and the rest the ground: a
// recolor of every piece (recolors_hold).
inline optional<replayed> size_rank_holds(const replay_facts& facts, const color_facts& palette,
                                          segmentation_methods cut) {
    if (palette.output_colors_by_size_count <= 0) return nullopt;
    replay_facts ranked = facts;
    if (cut == segmentation_methods::all_cells_of_one_color)
        for (size_t which = 0; which < ranked.cuts.size(); ++which) {
            const grid& input = ranked.train[which].input;
            vector<object> split;
            for (const object& piece : ranked.cuts[which]) {
                grid layer = empty_grid(input.width, input.height);
                paint_piece(layer, piece, piece.summary.y0, piece.summary.x0, true, /*clip=*/true);
                object_walk_scratch scratch;
                each_object(layer, -1, grouping::touching_any_color, scratch,
                            [&](const object_summary& box, const int16_t* cells, int count) {
                    split.push_back(object::from_cells(layer, box, cells, count));
                });
            }
            ranked.cuts[which] = std::move(split);
        }
    // Each piece's place in the order, pair by pair.
    vector<vector<int>> places;
    for (const vector<object>& hand : ranked.cuts) {
        vector<int> order(hand.size());
        iota(order.begin(), order.end(), 0);
        stable_sort(order.begin(), order.end(), [&](int one, int other) {
            return hand[one].summary.count > hand[other].summary.count; });
        vector<int> place(hand.size());
        for (size_t rank = 0; rank < order.size(); ++rank) place[order[rank]] = (int)rank;
        places.push_back(std::move(place));
    }
    return recolors_hold(ranked, handed_pieces(ranked, piece_pick::all()),
                         [&](size_t which, const partner_pieces&, int i, int8_t& color) {
        const int rank = places[which][i];
        color = rank < palette.output_colors_by_size_count ? palette.output_colors_by_size[rank]
                                                           : (int8_t)TRANSPARENT;
        return true;
    });
}

// PICKED PIECES SLID, replayed: each by its own `down` and `across`. It explains
// the moves only if the pair's output shows each moved piece whole where it
// lands, and, where it left that nothing else now covers, anything but the
// piece itself -- a cell still showing it was a copy left behind, not a move;
// what does show there is another word's to say (18286ef8's box closing behind
// its 9). And the word only if one piece at least moved in each pair, which its
// caller asks. Adds the cells each one left and the cells it landed on to the
// pair's lists, and counts the pieces that moved.
inline bool slides_show(const example& pair, const partner_pieces& cut,
                        const vector<int>& down, const vector<int>& across,
                        vector<std::pair<int, int>>& left, vector<std::pair<int, int>>& landed, int& pieces_moved) {
    bool covered[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (int i = 0; i < cut.count(); ++i) {
        bool fits = true;
        cut.pieces[i].each_cell([&](int row, int col, int8_t) {
            const int to_row = row + down[i], to_col = col + across[i];
            if (!pair.output.contains(to_row, to_col)) { fits = false; return; }
            covered[to_row][to_col] = true;
        });
        if (!fits) return false;
    }
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i] || (down[i] == 0 && across[i] == 0)) continue;
        bool shows = true;
        cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
            const int to_row = row + down[i], to_col = col + across[i];
            shows = shows && pair.output.data[to_row][to_col] == color &&
                    (covered[row][col] || pair.output.data[row][col] != color);
            left.push_back({ row, col });
            landed.push_back({ to_row, to_col });
        });
        if (!shows) return false;
        ++pieces_moved;
    }
    return true;
}

// PICKED PIECES SLID, EACH ITS OWN WAY, replayed: what every move word's replay
// asks, once. On each pair, `where(which, cut, i, by)` says how far picked piece
// i goes, down and across, or false where the word cannot say; the pieces then
// slide as slides_show reads them, and the word holds only if one piece at least
// moved in each pair. Sees the cells each one left and the cells it landed on,
// and how many pieces moved. Handed the pieces its pick takes (handed_pieces).
template <class Where>
inline optional<replayed> slides_hold(const replay_facts& facts, const vector<partner_pieces>& handed, Where where) {
    if (!facts.overlays) return nullopt;
    replayed seen;
    seen.left.assign(facts.train.size(), {});
    seen.drawn.assign(facts.train.size(), {});
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const int before = seen.pieces;
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        vector<int> downs(cut.pieces.size(), 0), acrosses(cut.pieces.size(), 0);
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            heading_step by{ 0, 0 };
            if (!where(which, cut, i, by)) return nullopt;
            downs[i] = by.down;
            acrosses[i] = by.across;
        }
        if (!slides_show(facts.train[which], cut, downs, acrosses, seen.left[which], seen.drawn[which], seen.pieces))
            return nullopt;
        if (seen.pieces == before) return nullopt;
    }
    if (seen.pieces == 0) return nullopt;
    return seen;
}

// A PIECE THAT MOVES LEVEL WITH ANOTHER, replayed. AlignTo slides every picked
// piece, down or across alone, until its box starts on the row -- or with
// `columns`, the column -- its one partner's box starts on (goes_level).
inline optional<replayed> align_holds(const replay_facts& facts, const vector<partner_pieces>& handed, bool columns,
                                      partner_by how) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, heading_step& by) {
        return cut.goes_level(i, columns, how, by);
    });
}
static_assert(row_of(node_type::AlignTo).high + 1 == 2 * (int)partner_by::count,
              "AlignTo takes rows then columns, each by every way partner_by finds a partner");

// A PIECE THAT MOVES TOWARD ANOTHER, replayed. MoveToward moves every picked
// piece toward its one partner of the kind `value` % PARTNER_KINDS names -- each
// way partner_by finds one, then each color -- by what `value` / PARTNER_KINDS
// names: one step, its own size, or until it touches it (goes_toward,
// move_toward in 1_readings.hpp, the move the op makes).
inline optional<replayed> toward_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int value) {
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, heading_step& by) {
        const grid& output = facts.train[which].output;
        return cut.goes_toward(i, value, output.width, output.height, by);
    });
}
static_assert(row_of(node_type::MoveToward).high + 1 == (int)toward_by::count * PARTNER_KINDS,
              "MoveToward takes each way of going toward each kind of partner");

// A PICKED PIECE SLID THE SAME WAY IN EVERY PAIR, replayed: `down` and `across`,
// the MoveColumn and MoveRow a pick by holes or by size is said with.
inline optional<replayed> move_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int down,
                                     int across) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces&, int, heading_step& by) {
        by = { down, across };
        return true;
    });
}

// A PICKED PIECE SLID TO THE EDGE OF THE GRID, replayed: that way, until its box
// meets the edge (steps_to_edge, 1_readings.hpp, the walk MoveToEdge takes).
inline optional<replayed> to_edge_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int way) {
    const heading_step step = step_of(way);
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, heading_step& by) {
        const grid& output = facts.train[which].output;
        const int steps = steps_to_edge(cut.boxes[i], output.width, output.height, way);
        by = { steps * step.down, steps * step.across };
        return true;
    });
}

// A PICKED PIECE MOVED ITS OWN SIZE, replayed: the way heading `way` names
// (shift_by, 1_readings.hpp, the step the op takes).
inline optional<replayed> shift_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int way) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, heading_step& by) {
        by = shift_by(way, cut.boxes[i].width(), cut.boxes[i].height());
        return true;
    });
}

// Where the first piece a pick takes could have gone in the first pair: every
// offset at which the output shows it whole, it not standing still. Moves are
// read off the output, not guessed, and every one is then replayed on every pair.
inline vector<std::pair<int, int>> offsets_to_ask(const replay_facts& facts, const vector<partner_pieces>& handed) {
    vector<std::pair<int, int>> offsets;
    if (facts.train.empty() || !facts.overlays) return offsets;
    const example& pair = facts.train[0];
    const partner_pieces& cut = handed[0];
    int first = -1;
    for (int i = 0; i < cut.count() && first < 0; ++i)
        if (cut.picked[i]) first = i;
    if (first < 0) return offsets;
    for (int down = -pair.output.height + 1; down < pair.output.height; ++down)
        for (int across = -pair.output.width + 1; across < pair.output.width; ++across) {
            if (down == 0 && across == 0) continue;
            bool shows = true;
            cut.pieces[first].each_cell([&](int row, int col, int8_t color) {
                shows = shows && pair.output.contains(row + down, col + across) &&
                        pair.output.data[row + down][col + across] == color;
            });
            if (shows) offsets.push_back({ down, across });
        }
    return offsets;
}

// PICKED PIECES THAT LOSE CELLS WHERE THEY LIE, replayed: what the removal's
// and the outline's replays ask, once. Each picked piece keeps the cells
// `keeps(cut, i, row, col)` names and loses the rest. It explains the change
// only if every pair's output shows each kept cell in the piece's own color,
// and each lost one bare -- the ground -- unless a piece still there covers it:
// one not picked, or a picked one that keeps that cell. And in each pair one
// piece at least lost a cell that now shows something else. Sees the cells the
// picked pieces keep (drawn) and lose (left), and how many pieces lost a cell
// that shows it.
template <class Keeps>
inline optional<replayed> drops_hold(const replay_facts& facts, const vector<partner_pieces>& handed, Keeps keeps) {
    if (!facts.overlays || facts.train.empty()) return nullopt;
    replayed seen;
    seen.drawn.assign(facts.train.size(), {});
    seen.left.assign(facts.train.size(), {});
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const example& pair = facts.train[which];
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        auto still_there = [&](int i, int row, int col) {
            for (int j = 0; j < cut.count(); ++j)
                if (j != i && cut.painted(j, row, col) && (!cut.picked[j] || keeps(cut, j, row, col))) return true;
            return false;
        };
        const int before = seen.pieces;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool shows = true, changed = false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                const int8_t there = (int8_t)pair.output.data[row][col];
                if (keeps(cut, i, row, col)) {
                    shows = shows && there == color;
                    seen.drawn[which].push_back({ row, col });
                    return;
                }
                shows = shows && (there == TRANSPARENT || still_there(i, row, col));
                seen.left[which].push_back({ row, col });
                changed = changed || there != color;
            });
            if (!shows) return nullopt;
            seen.pieces += changed;
        }
        if (seen.pieces == before) return nullopt;
    }
    return seen;
}

// PICKED PIECES REMOVED, replayed. DeleteSelection takes the pieces a pick takes
// out of the hand, so nothing paints them: they keep no cell (drops_hold), and
// the cells they sat on are left.
inline optional<replayed> removal_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return drops_hold(facts, handed, [](const partner_pieces&, int, int, int) { return false; });
}

// A PIECE HOLLOWED TO ITS EDGE, replayed. Outline keeps the cells of each picked
// piece that have, among their eight neighbours, a cell that is not the piece's
// (on_object_edge, the test the op takes), and drops the rest (drops_hold).
inline optional<replayed> outline_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return drops_hold(facts, handed, [](const partner_pieces& cut, int i, int row, int col) {
        const object& piece = cut.pieces[i];
        return on_object_edge(row - piece.summary.y0, col - piece.summary.x0, [&](int down, int across) {
            return down >= 0 && across >= 0 && down < piece.summary.h && across < piece.summary.w &&
                   piece.cells[down][across] != (int8_t)TRANSPARENT;
        });
    });
}

// WHERE MirrorAround DRAWS, said once for the op and for the reading that
// replays it. The middle a picked piece is mirrored around, twice its row and
// column so a middle between two cells is whole: its one partner's by `how`
// among the pieces not picked (partner_of), or past those the grid's own --
// false for a piece with no one partner, which gets no copy.
template <class Pieces>
ARC_HOST_DEVICE inline bool mirror_middle(int how, int mine, const Pieces& pieces, int width, int height,
                                          int& row2, int& col2) {
    row2 = height - 1;
    col2 = width - 1;
    if (how >= (int)partner_by::count) return true;
    const int partner = partner_of((partner_by)how, mine, pieces);
    if (partner < 0) return false;
    const cell_box there = pieces.box(partner);
    row2 = there.row0 + there.row1;
    col2 = there.col0 + there.col1;
    return true;
}
// Each copy the mirror draws, handed to `copy(down, across)`, the ways it flips:
// across for left to right, down for top to bottom, and all four ways each
// alone and then both.
template <class Copy>
ARC_HOST_DEVICE inline void each_mirror(mirror_by mirror, Copy copy) {
    for (int way = 0; way < 3; ++way)
        if (mirror == mirror_by::four_ways || way == (int)mirror) copy(way != 0, way != 1);
}
// Where a copy lies, its far side where the piece's near side was, and the pose
// its cells are read through (pose_source): flipped top to bottom where it is
// mirrored down, left to right where across.
ARC_HOST_DEVICE inline cell_box mirrored_box(const cell_box& box, int row2, int col2, bool down, bool across) {
    return { down ? row2 - box.row1 : box.row0, across ? col2 - box.col1 : box.col0,
             down ? row2 - box.row0 : box.row1, across ? col2 - box.col0 : box.col1 };
}
ARC_HOST_DEVICE constexpr int mirror_pose(bool down, bool across) { return (down ? 2 : 0) | (across ? 4 : 0); }
// One copy of a piece, mirrored so.
inline object mirrored(const object& piece, int row2, int col2, bool down, bool across) {
    object copy = piece.posed(mirror_pose(down, across));
    const cell_box to = mirrored_box(piece.summary.box(), row2, col2, down, across);
    copy.summary.y0 = (int8_t)to.row0;
    copy.summary.x0 = (int8_t)to.col0;
    return copy;
}

// A PIECE DRAWN AGAIN MIRRORED AROUND ANOTHER'S MIDDLE, replayed. MirrorAround
// draws every picked piece again mirrored around the middle of its one partner
// among the rest -- or, for the last of MIRROR_PARTNERS, around the middle of the
// grid itself (mirror_middle) -- laid over what is down (paint_holds); a copy
// that falls off the grid cannot be drawn, and a row mirrored end to end onto
// itself is no work. Sees the cells the picked pieces sit on (left) and the
// cells their copies landed on, and how many copies it drew.
inline optional<replayed> mirror_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       mirror_by mirror, int how) {
    cells_by_pair kept(facts.train.size());
    int copies = 0;
    auto seen = paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& input = facts.train[which].input;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            int row2 = 0, col2 = 0;
            if (!mirror_middle(how, i, cut, input.width, input.height, row2, col2)) return false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t) { kept[which].push_back({ row, col }); });
            bool on_grid = true;
            each_mirror(mirror, [&](bool down, bool across) {
                mirrored(cut.pieces[i], row2, col2, down, across).each_cell([&](int row, int col, int8_t color) {
                    if (!input.contains(row, col)) on_grid = false;
                    else put(row, col, color);
                });
            });
            if (!on_grid) return false;
            copies += mirror == mirror_by::four_ways ? 3 : 1;
        }
        return true;
    });
    if (seen) {
        seen->left = std::move(kept);
        seen->pieces = copies;
    }
    return seen;
}
static_assert(row_of(node_type::MirrorAround).high + 1 == (int)mirror_by::count * MIRROR_PARTNERS,
              "MirrorAround takes each way of mirroring around each partner, then the grid");

// THE TRAIL ExtendTo GROWS, said once for the op and for the reading that
// replays it: from the cell (row, col) of a piece, step by step that way to the
// edge of a canvas `width` by `height` or to a cell another piece holds,
// through the piece's own cells -- `own(row, col)` and `taken(row, col)` say
// which -- and each cell passed that is not its own handed to `lay(row, col)`.
template <class Own, class Taken, class Lay>
ARC_HOST_DEVICE inline void march_trail(int row, int col, heading_step step, int width, int height,
                                        Own own, Taken taken, Lay lay) {
    for (int at_row = row + step.down, at_col = col + step.across;
         at_row >= 0 && at_row < height && at_col >= 0 && at_col < width;
         at_row += step.down, at_col += step.across) {
        if (own(at_row, at_col)) continue;
        if (taken(at_row, at_col)) break;
        lay(at_row, at_col);
    }
}

// A TRAIL GROWN OFF A PIECE, replayed. ExtendTo marches from every cell of each
// picked piece that way (march_trail) and fills what it passes in the piece's
// color, laid over what is down (paint_holds). The op hands back the body with
// its trail, so the paint lays the body down again where it lies: the output
// must show it there too (66e6c45b's line removed a piece whose trail then drew
// it back).
inline optional<replayed> trail_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int way) {
    const heading_step step = step_of(way);
    return paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        bool occupied[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        for (const object& piece : cut.pieces)
            piece.each_cell([&](int row, int col, int8_t) { occupied[row][col] = true; });
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const int8_t color = cut.pieces[i].first_color();
            cut.pieces[i].each_cell([&](int row, int col, int8_t body) {
                if (output.contains(row, col)) put(row, col, body);
                march_trail(row, col, step, output.width, output.height,
                            [&](int at_row, int at_col) { return cut.painted(i, at_row, at_col); },
                            [&](int at_row, int at_col) { return occupied[at_row][at_col]; },
                            [&](int at_row, int at_col) { put(at_row, at_col, color); });
            });
        }
        return true;
    });
}

// THE PICKED PIECES COPIED ONTO THE OTHERS, replayed. CopyTo draws each picked
// piece again onto every piece not picked, lined up the way `anchor` numbers
// CopyTo's nine -- top, middle or bottom by left, middle or right,
// aligned_copy_start (1_readings.hpp), the step the op takes -- laid over what is
// down (paint_holds), each copied cell that lands on the grid. Picking none, or
// every piece, or more copies than a list holds, it draws nothing.
inline optional<replayed> copy_onto_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                          int anchor) {
    return paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        int picked = 0;
        for (const bool one : cut.picked) picked += one;
        if (picked == 0 || picked == cut.count() || picked * (cut.count() - picked) > MAX_OBJECTS_IN_A_LIST)
            return false;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const auto& pattern = cut.pieces[i].summary;
            for (int j = 0; j < cut.count(); ++j) {
                if (cut.picked[j]) continue;
                const auto& target = cut.pieces[j].summary;
                const int row0 = aligned_copy_start(pattern.h, target.y0, target.h, anchor / 3);
                const int col0 = aligned_copy_start(pattern.w, target.x0, target.w, anchor % 3);
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    const int at_row = row0 + row - pattern.y0, at_col = col0 + col - pattern.x0;
                    if (output.contains(at_row, at_col)) put(at_row, at_col, color);
                });
            }
        }
        return true;
    });
}

// A PIECE REPEATED ALONG ITS ROWS OR DOWN ITS COLUMNS, replayed. TileX lays each
// picked piece again every `period` cells across the whole canvas, in its own
// rows; TileY the same down the canvas, in its own columns -- counted from where
// the piece stands, both ways, tiled_source (4_objects.hpp), the step the op
// takes -- laid `lay` on what is down (paint_holds): one cell at least new in
// each pair is, the first copy being the piece itself, a second copy in each.
inline optional<replayed> tile_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                     const vector<cells_down>& canvas, bool down, int period, int8_t lay) {
    return paint_holds(facts, handed, { lay, &canvas }, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int along = down ? output.height : output.width;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            const auto& box = piece.summary;
            for (int at = 0; at < along; ++at) {
                const int source = tiled_source(at, down ? box.y0 : box.x0, period, down ? box.h : box.w);
                if (source < 0) continue;
                for (int other = 0; other < (down ? box.w : box.h); ++other) {
                    const int row = down ? at : box.y0 + other, col = down ? box.x0 + other : at;
                    const int8_t color = down ? piece.cells[source][other] : piece.cells[other][source];
                    if (color == (int8_t)TRANSPARENT || !output.contains(row, col)) continue;
                    if (!put(row, col, color)) return false;
                }
            }
        }
        return true;
    });
}

// RAYS GROWN OFF A PIECE, replayed. Extend grows rays from the cell of each
// picked piece nearest its middle -- up, down and to the sides, or along the
// diagonals -- `radius` cells long, in that cell's color (nearest_ray_seed and
// each_extended_ray, 4_objects.hpp, the walk the op takes), and the line may
// paint them over in one color of its own: laid `lay` on what is down, in one
// color (paint_holds), which it sees too.
inline optional<replayed> rays_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                    const vector<cells_down>& canvas, bool diagonal, int radius, int8_t lay) {
    return paint_holds(facts, handed, { lay, &canvas, laying::in_one },
                       [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            const auto& box = piece.summary;
            const auto seed = nearest_ray_seed(box.h, box.w, (box.h - 1) / 2, (box.w - 1) / 2,
                                               [&](int row, int col) { return piece.cells[row][col]; });
            each_extended_ray(seed, radius, diagonal, [&](int down, int across) {
                const int row = box.y0 + seed.row + down, col = box.x0 + seed.col + across;
                if (output.contains(row, col)) put(row, col, (int8_t)TRANSPARENT);
            });
        }
        return true;
    });
}

// THE PICKED PIECES, AS ONE FIGURE, DRAWN AGAIN TURNED in their own box,
// replayed: UnionList folds them into one, Trim fits its box to them, and
// TurnRight, twice or TurnLeft -- `pose` 3, 6 or 5, read through pose_source as
// the ops read it, posed_place here -- turns the figure in that box, its
// top-left corner where it was. Laid over what is down (paint_holds), and the
// figure itself is down where it lies, so the output must show it there too
// wherever the turned copy does not cover it: the copy and the figure are drawn,
// both, since the matching read the figure turned (11852cab's design, the same
// from all four sides). Picking nothing, it draws nothing.
inline optional<replayed> turned_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int pose) {
    return paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        cell_box figure;
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) figure.include(cut.boxes[i]);
        if (figure.empty()) return false;
        int8_t copy[MAX_GRID_DIM][MAX_GRID_DIM];
        for (auto& row : copy) ranges::fill(row, (int8_t)TRANSPARENT);
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i])
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    const auto [down, across] = posed_place(pose, figure.width(), figure.height(),
                                                            row - figure.row0, col - figure.col0);
                    if (output.contains(figure.row0 + down, figure.col0 + across))
                        copy[figure.row0 + down][figure.col0 + across] = color;
                });
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i])
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    if (copy[row][col] == (int8_t)TRANSPARENT) put(row, col, color);
                });
        for (int row = 0; row < output.height; ++row)
            for (int col = 0; col < output.width; ++col)
                if (copy[row][col] != (int8_t)TRANSPARENT) put(row, col, copy[row][col]);
        return true;
    });
}

// A PIECE'S BOX FILLED, replayed. AddColor paints every cell of each picked
// piece's box in its one color -- laid under what is down (paint_holds), so it
// fills only what the piece and every paint before it leave empty.
inline optional<replayed> box_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                    const vector<cells_down>& canvas) {
    using rule = correspondence_election::fired_rule;
    return paint_holds(facts, handed, { rule::laid_under, &canvas, laying::in_one },
                       [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const cell_box& box = cut.boxes[i];
            for (int row = box.row0; row <= box.row1; ++row)
                for (int col = box.col0; col <= box.col1; ++col) put(row, col, (int8_t)TRANSPARENT);
        }
        return true;
    });
}

// BARS BETWEEN PIECES OF ONE COLOR, replayed. Bridge reaches from each piece to
// those of its own color whose middles share its row or its column, the `way`
// it is told (bridge_bars, the walk the op takes), and fills the box between
// two -- laid under what is down (paint_holds), so a bar only fills what nothing
// has painted -- in the bar's own color, or all of them in one.
inline optional<replayed> bridge_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       const vector<cells_down>& canvas, int way) {
    using rule = correspondence_election::fired_rule;
    return paint_holds(facts, handed, { rule::laid_under, &canvas, laying::own_else_one },
                       [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        // The bars take one place more in the pieces' list, all of them one drawing.
        if (cut.count() >= MAX_OBJECTS_IN_A_LIST) return false;
        bridge_bars(way, cut.count(), [&](int i) { return cut.boxes[i]; },
                    [&](int i) { return cut.pieces[i].first_color(); },
                    [&](const cell_box& bar, int8_t color) {
            for (int row = bar.row0; row <= bar.row1; ++row)
                for (int col = bar.col0; col <= bar.col1; ++col)
                    if (output.contains(row, col)) put(row, col, color);
            return true;
        });
        return true;
    });
}

// THE CELLS A PIECE ENCLOSES: those its own paint walls off from the edge of its
// box (mark_outside, the four-way flood the ops flood), each handed to
// `see(row, col)` where it lies.
template <class See>
inline void each_hollow(const object& piece, See see) {
    const int height = piece.summary.h, width = piece.summary.w;
    auto occupied = [&](int row, int col) { return piece.cells[row][col] != (int8_t)TRANSPARENT; };
    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
    mark_outside(height, width, occupied, outside);
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            if (!occupied(row, col) && !outside[row][col]) see(piece.summary.y0 + row, piece.summary.x0 + col);
}

// THE HOLLOWS OF A PIECE, FILLED, replayed. InteriorColor fills each picked
// piece's hollows (each_hollow) -- laid under what is down (paint_holds), so a
// marker sitting in a hollow keeps its own -- in each piece's own color, or all
// of them in one.
inline optional<replayed> fill_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                     const vector<cells_down>& canvas) {
    using rule = correspondence_election::fired_rule;
    return paint_holds(facts, handed, { rule::laid_under, &canvas, laying::own_else_one },
                       [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) {
                const int8_t worn = cut.pieces[i].first_color();
                each_hollow(cut.pieces[i], [&](int row, int col) { put(row, col, worn); });
            }
        return true;
    });
}

// THE HOLLOWS, FILLED BY SHAPE, replayed. ShapeToInteriorColorMap takes the
// pieces as the line hands them over, looks each one's shape up in the learned
// table, and fills its hollows (each_hollow) with that color -- laid under the
// hand, so a marker already sitting in a hollow keeps its own (paint_holds).
// Sees the cells it fills and how many pieces it filled.
inline optional<replayed> interior_fills_hold(const replay_facts& facts, const shape_color_map_rule& table) {
    using rule = correspondence_election::fired_rule;
    const vector<cells_down> hand = pieces_down(facts);
    int pieces_filled = 0;
    auto seen = paint_holds(facts, handed_pieces(facts, piece_pick::all()), { rule::laid_under, &hand },
                       [&](size_t, const partner_pieces& cut, auto&& put) {
        for (const object& piece : cut.pieces) {
            const int8_t fill = (int8_t)table.color_of(shape_key::shape, piece);
            if (fill < 0) continue;
            bool painted = false;
            each_hollow(piece, [&](int row, int col) { put(row, col, fill); painted = true; });
            pieces_filled += painted;
        }
        return true;
    });
    if (seen) seen->pieces = pieces_filled;
    return seen;
}

// THE PIECES AS THE LINE LEAVES THEM. Each word on the line acts on the hand the
// words before it left, as the program runs them, so a word asked after them is
// replayed on that hand and not on the cut: the pieces it picks, where they
// stand and what they wear. Carried here pair by pair, one word at a time, with
// the arithmetic its op runs -- the pick read off the hand as it is, each
// partner found before any piece is touched, copies joining the end of the list.
// False for a word the hand cannot be carried through yet -- a table learned off
// the pairs, the size rank -- after which the words below are asked the way
// they were before the hand was carried.
inline bool carry_the_hand(const correspondence_election::fired_rule& rule, replay_facts& facts) {
    using said = correspondence_election::fired_rule;
    const piece_pick& pick = rule.pick;
    auto recolored = [](object& piece, int color) { piece.repaint([&](int8_t) { return color; }); };
    auto moved = [](object& piece, int down, int across) {
        piece.summary.y0 = (int8_t)(piece.summary.y0 + down);
        piece.summary.x0 = (int8_t)(piece.summary.x0 + across);
    };
    for (size_t which = 0; which < facts.cuts.size(); ++which) {
        vector<object>& hand = facts.cuts[which];
        const example& pair = facts.train[which];
        const vector<bool> picked = picked_pieces(hand, pick);
        switch (rule.as) {
        case said::turn:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i] = hand[i].posed(rule.value);
            break;
        case said::move_down:
        case said::move_across:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) moved(hand[i], rule.as == said::move_down ? rule.value : 0,
                                     rule.as == said::move_across ? rule.value : 0);
            break;
        case said::recolor:
        case said::recolor_picked:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) recolored(hand[i], rule.value);
            break;
        case said::swap_colors: {
            const auto [first, second] = color_pair::from_index(rule.value);
            for (object& piece : hand)
                piece.repaint([&](int8_t now) { return now == first ? second : now == second ? first : now; });
            break;
        }
        case said::color_from:
        case said::align_to:
        case said::move_toward: {
            // Every partner is found on the hand as it was, before any piece
            // changes; a piece with no one partner stays as it is.
            const partner_pieces cut(hand, pick);
            constexpr int ways = (int)partner_by::count;
            vector<int> found(hand.size(), -1);
            vector<heading_step> by(hand.size(), heading_step{ 0, 0 });
            for (int i = 0; i < cut.count(); ++i) {
                if (!cut.picked[i]) continue;
                if (rule.as == said::color_from) found[i] = cut.color_from((partner_by)rule.value, i);
                else if (rule.as == said::align_to)
                    cut.goes_level(i, rule.value >= ways, (partner_by)(rule.value % ways), by[i]);
                else cut.goes_toward(i, rule.value, pair.output.width, pair.output.height, by[i]);
            }
            for (size_t i = 0; i < hand.size(); ++i) {
                if (found[i] >= 0) recolored(hand[i], found[i]);
                moved(hand[i], by[i].down, by[i].across);
            }
            break;
        }
        case said::to_edge:
        case said::shifted: {
            const heading_step step = step_of(rule.value);
            for (size_t i = 0; i < hand.size(); ++i) {
                if (!picked[i]) continue;
                const cell_box box = hand[i].painted_box();
                if (rule.as == said::to_edge) {
                    const int steps = steps_to_edge(box, pair.output.width, pair.output.height, rule.value);
                    moved(hand[i], steps * step.down, steps * step.across);
                } else {
                    const heading_step by = shift_by(rule.value, box.width(), box.height());
                    moved(hand[i], by.down, by.across);
                }
            }
            break;
        }
        case said::removed: {
            vector<object> kept;
            for (size_t i = 0; i < hand.size(); ++i)
                if (!picked[i]) kept.push_back(hand[i]);
            hand.swap(kept);
            break;
        }
        case said::outlined:
            for (size_t i = 0; i < hand.size(); ++i) {
                if (!picked[i]) continue;
                object& piece = hand[i];
                const object was = piece;
                auto occupied = [&](int row, int col) {
                    return row >= 0 && col >= 0 && row < was.summary.h && col < was.summary.w &&
                           was.cells[row][col] != (int8_t)TRANSPARENT;
                };
                piece.summary.count = 0;
                for (int row = 0; row < was.summary.h; ++row)
                    for (int col = 0; col < was.summary.w; ++col) {
                        if (!occupied(row, col)) continue;
                        if (on_object_edge(row, col, occupied)) ++piece.summary.count;
                        else piece.cells[row][col] = (int8_t)TRANSPARENT;
                    }
            }
            break;
        case said::mirror_around: {
            const partner_pieces cut(hand, pick);
            const auto mirror = (mirror_by)(rule.value / MIRROR_PARTNERS);
            const int how = rule.value % MIRROR_PARTNERS;
            vector<object> copies;
            for (int i = 0; i < cut.count(); ++i) {
                if (!cut.picked[i]) continue;
                int row2 = 0, col2 = 0;
                if (!mirror_middle(how, i, cut, pair.input.width, pair.input.height, row2, col2)) continue;
                each_mirror(mirror, [&](bool down, bool across) {
                    copies.push_back(mirrored(hand[i], row2, col2, down, across));
                });
            }
            for (const object& copy : copies)
                if (hand.size() < (size_t)MAX_OBJECTS_IN_A_LIST) hand.push_back(copy);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

// Each word belongs to one group. Acts change the hand; paints draw from the
// cut again; whole-program rules have their own spelling. The act groups say
// which later picks remain usable. No group carries another group's fields.
enum class word_group { unsaid, recolors, changes_pieces, removes, paints, whole_program };
inline constexpr word_group group_of(correspondence_election::fired_rule::said_as kind) {
    using rule = correspondence_election::fired_rule;
    switch (kind) {
        // Acts: only the colors change.
        case rule::recolor: case rule::swap_colors: case rule::size_rank:
        case rule::color_from: case rule::recolor_picked: case rule::shape_recolor:
            return word_group::recolors;
        // Acts: pieces move, turn, change shape, or gain copies. Learned-table
        // compositions keep the old gate: their hand cannot yet be carried.
        case rule::turn: case rule::move_down: case rule::move_across:
        case rule::learned_table: case rule::align_to: case rule::mirror_around:
        case rule::outlined: case rule::to_edge: case rule::shifted: case rule::move_toward:
            return word_group::changes_pieces;
        case rule::removed: return word_group::removes;
        // Paints: the hand on the line is left alone.
        case rule::extend_rays: case rule::bridged: case rule::filled: case rule::trailed:
        case rule::boxed: case rule::copied_onto: case rule::tiled_along: case rule::turned_copy:
            return word_group::paints;
        // These rules redraw the whole output and spell their own programs.
        case rule::diagonal_tile: case rule::piece_slam: case rule::corner_chain:
        case rule::slide_to_touch: case rule::project_on_seeds: case rule::snap_to_anchors:
        case rule::own_sentences: return word_group::whole_program;
        case rule::unsaid: case rule::kind_count: return word_group::unsaid;
    }
    return word_group::unsaid;
}
static_assert([] {
    using rule = correspondence_election::fired_rule;
    for (int kind = rule::unsaid + 1; kind < rule::kind_count; ++kind)
        if (group_of((rule::said_as)kind) == word_group::unsaid) return false;
    return true;
}(), "every word belongs to a group; duplicate entries fail the switch");

// Rays and repeats keep a length/period below 32, plus 32 for the second
// direction. This is their stored value, not the argument written in the DSL.
struct paint_value {
    static constexpr int direction_bit = 32;
    static constexpr int packed(bool second, int number) { return (second ? direction_bit : 0) + number; }
    static constexpr bool second(int stored) { return stored >= direction_bit; }
    static constexpr int number(int stored) { return stored % direction_bit; }
};

// THE REST OF THE ELECTED CUT'S LINE, after its own acts: a rule read off the
// cut's own pieces that drew every output by itself, or else the words its
// recolor is said with and what it has no word for. Listed on the reading under
// the rule whose evidence each one reads, and put on the line in the order it is
// said.
inline void finish_the_line(inference& inf, const vector<example>& train) {
    using rule = correspondence_election::fired_rule;
    auto& election = inf.correspondence;
    if (election.chosen_pairs().empty()) return;
    auto& read = election.readings[election.chosen];
    const int8_t colors = read.fired_named(degrees_of_freedom::freedom_names[2]);

    // A RULE THAT DREW EVERY OUTPUT BY ITSELF from the cut's own pieces is the
    // whole rest of the line: the acts have nothing left to explain, so they stay
    // listed and are not said, and it accounts for every cell there is. Asked in
    // this order, the first that holds.
    auto whole_line = [&](bool holds, rule::said_as as, const char* name, const char* says) {
        if (!holds) return false;
        read.line.clear();
        read.say_fired(listed(name, says, as), -1, [](size_t, const auto&) { return true; });
        return true;
    };
    if (whole_line(inf.diagonal_continuation.holds(), rule::diagonal_tile, "Diagonal Tile",
                   "the colors cycle along the diagonals") ||
        whole_line(inf.piece_slam.holds(), rule::piece_slam, "Slam", "each piece piled up inside its own box") ||
        whole_line(inf.overlap_corner.holds(), rule::corner_chain, "Corner Chain",
                   "each piece hung off the last one's corner") ||
        whole_line(inf.slide_to_touch.holds(), rule::slide_to_touch, "Slide To Touch",
                   "pieces slid up against the ones that stay") ||
        whole_line(inf.project_on_seeds.holds(), rule::project_on_seeds, "Project On Seeds",
                   "the template stamped along every marker") ||
        whole_line(inf.snap_to_anchors.holds(), rule::snap_to_anchors, "Snap To Anchors",
                   "each body turned and set down on the anchors its keys match"))
        return;
    using missing = correspondence_election::missing;
    const auto& shapes = inf.shape_color_map;
    const bool recolors = read.allowed.recolor && ranges::any_of(read.pairs, [](const auto& pair) {
        return pair.found_recolored > 0; });

    // THE KEY'S SHAPE NAMES THE TARGET'S COLOR. The rule read the elected
    // reading's recolor, so it says so on that reading, under the recolor --
    // freedom 2, Color -- accounting for the target's recolored cells and for
    // the keys it used up: their cells are the ones the reading had left over
    // as gone. Said ahead of the swaps, since a swap is only said where no rule
    // already accounts for what it would change.
    const bool by_shape = recolors && shapes.holds() && shapes.claimed_by == shape_color_map_rule::body_recolor;
    if (by_shape) {
        rule one = listed("Shape Picks Color", "the shape of the key names the color", rule::learned_table);
        one.pieces = (int)shapes.read.size();
        read.say_fired(std::move(one), colors, [&](size_t which, const auto& cell) {
            if (which >= shapes.read.size()) return false;
            const auto& seen = shapes.read[which];
            return (cell.from_input && cell.waits < 0 && cell.piece == seen.key) ||
                   (cell.waits_for(missing::a_recolor) && cell.about == seen.target);
        });
    }

    // A swap is said over the pieces, so it only has to hold where a piece is.
    // Candidates that fail even that remain available to search.
    //
    // AND ONLY WHERE THE READING LET A COLOUR CHANGE AT ALL. A swap IS a
    // recolour, so a reading that matched every piece without one has already
    // said these pieces kept their colours -- and over_the_pieces is in no
    // position to argue, because the cells it reads are the ones a piece LEFT.
    // 05f2a901 is the puzzle that shows it: the shape slides up against the
    // block, [Move] matches all six pieces exactly, 0 recoloured, and 0<->2
    // holds over the pieces for one reason only -- a shape that moves away
    // leaves ground where it stood and covers ground where it lands, which
    // reads cell for cell as the two trading places. Said, it paints the shape
    // away and the reading draws an empty grid, while its own list has already
    // named the thing it cannot say: where each piece went.
    // Nor where a fired rule already accounts for every cell it would change.
    // A swap with the ground cannot come up: the plan deleted it, so no piece
    // wears it (aabf363d's marker was painted away by one, before the shape could
    // take its color). It accounts for every recolored cell that traded one of
    // its two colors for the other.
    // Nor a swap with a color a word already said paints: the swap would trade
    // back what that word laid down (9565186b: 8 recolored to 5, then 1 and 5
    // trading places turns those 5s to 1s, while each word alone claimed its
    // cells). A swap paints both its colors, so a second swap of either one
    // trades back what the first laid down (9565186b again: 1 and 5, then 2
    // and 5, turns the 5s the first laid down to 2s).
    auto painted_by_the_line = [&](int color) {
        return ranges::any_of(read.fired, [&](const auto& one) {
            if (one.as == rule::swap_colors) {
                const auto [traded, other] = color_pair::from_index(one.value);
                return traded == color || other == color;
            }
            return ((one.as == rule::recolor || one.as == rule::recolor_picked) && one.value == color) ||
                   one.paint == color;
        });
    };
    if (read.allowed.recolor)
        for (int variant = 0; variant < COLOR_PAIR_COUNT; ++variant)
            if (const auto [first, second] = color_pair::from_index(variant);
                inf.color_swaps.over_the_pieces[variant] &&
                !painted_by_the_line(first) && !painted_by_the_line(second) &&
                !swap_already_claimed(inf, variant)) {
                const string says = to_string(first) + " and " + to_string(second) + " trade places";
                read.say_fired(listed("Swap", says.c_str(), rule::swap_colors, (int8_t)variant), colors,
                               [&](size_t, const auto& cell) {
                    return cell.waits_for(missing::a_recolor) &&
                           ((cell.color == first && cell.other == second) ||
                            (cell.color == second && cell.other == first));
                });
            }

    // THE HAND AS THE LINE LEAVES IT: the cut's pieces carried through every word
    // already on the line, the hand the words asked below are replayed on.
    replay_facts facts(inf, train);
    for (const int8_t which : read.line)
        facts.hand_known = facts.hand_known && carry_the_hand(read.fired[which], facts);
    // Carries the hand through a word just said on the line.
    auto carry = [&](int8_t which) {
        facts.hand_known = facts.hand_known && carry_the_hand(read.fired[which], facts);
    };

    // THE HOLLOWS FILLED BY SHAPE, a step of its own: it paints space the pieces
    // enclose, not the pieces, so it is no word for a recolor. Said where a replay
    // on the hand shows every fill it paints, and it accounts for those cells.
    if (shapes.holds() && shapes.claimed_by == shape_color_map_rule::interior_fill && facts.hand_known)
        if (const auto filled = interior_fills_hold(facts, shapes)) {
            rule one = listed("Shape Picks Fill", "the shape of a piece names the color inside it",
                              rule::learned_table);
            one.pieces = filled->pieces;
            carry(read.say_fired(std::move(one), -1, [&](size_t which, const auto& cell) {
                return !cell.from_input && on_cells(filled->drawn, which, cell.row, cell.col);
            }));
        }

    // A RECOLOR THE ACTS DID NOT SAY, said by size rank where the palette ranks
    // the same in every output. The replay on the hand showed every piece in its
    // rank's color, so it accounts for every recolored cell.
    const bool a_recolor_was_said = ranges::any_of(read.line, [&](int8_t which) {
        return read.fired[which].as == rule::recolor; });
    if (recolors && !by_shape && !a_recolor_was_said && facts.hand_known)
        if (const auto ranked = size_rank_holds(facts, inf.colors, election.segmentation)) {
            rule one = listed("Size Picks Color", "the colors go by size", rule::size_rank);
            one.pieces = ranked->pieces;
            carry(read.say_fired(std::move(one), colors,
                                 [](size_t, const auto& cell) { return cell.waits_for(missing::a_recolor); }));
        }

    // THE WORDS READ OFF THE PIECES: a piece takes its color from its partner,
    // moves level with it or by one step every pair agrees on, is drawn again
    // mirrored around a middle, is hollowed to its edge or recolored, grows rays,
    // or has its hollows filled; the gaps between pieces of one color are bridged.
    // Each is asked the same way. It picks the pieces the ways every word picks --
    // by the colors an unsaid change starts from, by holes, by size -- and replays
    // each value it takes on every pair. Of the picks and values every pair bears
    // out, the one that accounts for the most cells still unexplained is said, a
    // tie going to the plainer: the colors before holes and size, the smaller
    // value before the larger. Where every piece of every pair is picked, it picks
    // every piece, since a test can hold a color no pair did (496994bd).
    static constexpr const char* partners[] = { "the piece around it", "the piece inside it",
        "a piece in its rows", "a piece in its columns", "the nearest piece", "the piece it shares a side with",
        "the piece it touches", "the piece of its shape", "the piece of its shape, turned" };
    static_assert(size(partners) == (size_t)partner_by::count);
    static constexpr const char* mirrors[] = { "left to right", "top to bottom", "all four ways" };
    static_assert(size(mirrors) == (size_t)mirror_by::count);
    // Each kind of pick by name, then turned round, every piece but those.
    static constexpr const char* pick_names[][2] = { { "Holes", "" }, { "Largest", "" }, { "Smallest", "" },
        { "Row Band", "" }, { "Column Band", "" }, { "Most Common Color", "Not Most Common Color" },
        { "Cells At Least", "Cells Fewer Than" }, { "The Largest", "All But The Largest" },
        { "The Smallest", "All But The Smallest" }, { "Row From The Middle", "Off The Row From The Middle" },
        { "Column From The Middle", "Off The Column From The Middle" } };
    static_assert(size(pick_names) == (size_t)piece_pick::pick_kinds);
    using left_over_cell = correspondence_election::left_over_cell;
    using claim = std::function<bool(size_t, const left_over_cell&)>;
    // The best a word heard: its value and pick, how many cells nothing else
    // claims it accounts for, what its replay saw, and the cells it claims.
    struct heard {
        int value = -1, said = 0;
        piece_pick pick;
        replayed seen;
        claim claims;
    };
    // Whether the line already says `as` with this pick and value. Said again of a
    // hand that changed under it, the pick takes some other piece each time --
    // 1a07d186 removed "the third largest" four times over, peeling pieces off
    // one by one -- so it names nothing steady, and it is not asked again.
    auto already_said = [&](rule::said_as as, const piece_pick& pick, int value) {
        return ranges::any_of(read.line, [&](int8_t which) {
            const rule& step = read.fired[which];
            return step.as == as && step.value == value && step.pick == pick;
        });
    };
    // What a paint said after the line reads: the cut opened again, as it is said.
    const replay_facts opened(inf, train);
    // The colors an unsaid change of `word` starts from, a bit each.
    auto starts_from = [&](missing word) {
        uint16_t found = 0;
        for (const auto& pair : read.pairs)
            for (const auto& cell : pair.left_over)
                if (cell.claimed_by < 0 && cell.from_input && cell.answers_to(word))
                    found = (uint16_t)(found | 1u << cell.color);
        return found;
    };
    // Keeps `value` when it accounts for more cells nothing claims yet than the
    // best so far, with what its replay saw.
    auto weigh = [&](heard& best, int value, const piece_pick& pick, const replayed& seen, claim claims) {
        int said = 0;
        for (size_t which = 0; which < read.pairs.size(); ++which)
            for (const auto& cell : read.pairs[which].left_over)
                said += cell.claimed_by < 0 && claims(which, cell);
        if (said > best.said) best = { value, said, pick, seen, std::move(claims) };
    };
    // Says what a word heard best, under the rule whose freedom it needs, named and
    // picking by every piece, by its colors, or by its pick of holes or size.
    auto say = [&](const heard& best, rule::said_as as, const string& says, int8_t under) {
        string name;
        const bool every_piece = best.seen.every_piece;
        if (every_piece) name = "Every Piece";
        else if (best.pick.how >= 0) {
            name = pick_names[best.pick.how][best.pick.but_those];
            if (best.pick.how != piece_pick::most_common) name += " " + to_string(best.pick.value);
        }
        else {
            name = "Colors";
            for (int color = 0; color < COLOR_COUNT; ++color)
                if (best.pick.colors >> color & 1) name += " " + to_string(color);
        }
        if (!every_piece && best.pick.fits >= 0)
            name += string(" That Would Fit ") + partners[best.pick.fits];
        rule one = listed(name.c_str(), says.c_str(), as, (int8_t)best.value);
        one.pick = every_piece ? piece_pick::all() : best.pick;
        one.pieces = best.seen.pieces;
        one.paint = best.seen.paint;
        return read.say_fired(std::move(one), under, best.claims);
    };
    const int8_t moves = read.fired_named(degrees_of_freedom::freedom_names[0]);
    constexpr int ways = (int)partner_by::count;
    // Claims for a move: the cells the pieces left, on the input, and landed on.
    auto move_claims = [](const replayed& seen) -> claim {
        return [left = seen.left, landed = seen.drawn](size_t which, const left_over_cell& cell) {
            return cell.answers_to(missing::where_it_went) &&
                   on_cells(cell.from_input ? left : landed, which, cell.row, cell.col);
        };
    };
    // Claims for a recolor: the cells it repainted that wait on one.
    auto recolor_claims = [](const replayed& seen) -> claim {
        return [recolored = seen.drawn](size_t which, const left_over_cell& cell) {
            return cell.answers_to(missing::a_recolor) && on_cells(recolored, which, cell.row, cell.col);
        };
    };

    // What changes a pick by color: a word on the line that moves, turns or
    // recolors the pieces of that color, or turns another color into it.
    auto changes = [](const rule& step, int color) {
        switch (step.as) {
            case rule::turn:
            case rule::move_down:
            case rule::move_across: return step.pick.may_take_color(color);
            case rule::recolor:     return step.pick.may_take_color(color) || step.value == color;
            case rule::swap_colors: {
                const auto [first, second] = color_pair::from_index(step.value);
                return first == color || second == color;
            }
            default:                return group_of(step.as) != word_group::paints;
        }
    };

    // THE WORDS READ OFF THE PIECES, THE ONE THAT SAYS THE MOST FIRST. Each kind --
    // a color taken from a partner, a move, a mirrored copy, an outline, a recolor
    // of a pick, a removal -- hears its best of the hand the line leaves, and the
    // kind whose best accounts for the most cells nothing claims yet is said, a tie
    // going in that order; then every kind is asked again, of the hand that word
    // leaves, since a word said late can leave the pieces as another needs them.
    // Asked in a fixed order instead, the first kind took what a later one says
    // better: 18286ef8's 9 was recolored to the 5 around it before its step toward
    // the 6 was weighed. Every word said claims cells nothing had, and each kind
    // says at most `rounds` words, as many as the rounds it was once asked in --
    // a line of a hundred words, each claiming a cell or two, says nothing (and a
    // reading lists at most 127 rules); where the hand could not be carried, only
    // the kinds that need no hand are asked. An offer is what a kind would say:
    // how much it accounts for, and the saying.
    struct offer { int said = 0; std::function<void()> say; };

    // A RECOLOR ANOTHER PIECE DECIDES: every picked piece takes its partner's color.
    // It accounts for the cells it recolored.
    auto color_from_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (!facts.hand_known || !from) return {};
        heard best;
        for (const piece_pick& pick : picks_to_ask(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int how = 0; how < ways; ++how)
                if (const auto seen = color_from_holds(facts, handed, (partner_by)how))
                    weigh(best, how, pick, *seen, recolor_claims(*seen));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            carry(say(best, rule::color_from, string("takes the color of ") + partners[best.value], colors)); } };
    };

    // A MOVE THE PIECES DECIDE: every picked piece slides level with its partner,
    // into its rows or its columns; slides to an edge of the grid; moves its own
    // size one of eight ways; or, picked by holes, size or band, the same step in
    // every pair, which a pick by color already says as an act; or moves toward
    // its partner. Of the kinds, the one that accounts for the most is said, a tie
    // going in that order -- a move toward a partner, the newest word, after every
    // older one. It accounts for the cells each piece left and the cells it landed
    // on.
    static_assert(heading::count == row_of(node_type::MoveToEdge).high + 1 &&
                  heading::count == row_of(node_type::Shift).high + 1);
    static constexpr const char* how_far[] = { "one step", "its own size", "until it touches" };
    static_assert(size(how_far) == (size_t)toward_by::count);
    auto move_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::where_it_went);
        if (!facts.hand_known || !from) return {};
        heard aligned, edged, sized, stepped, toward;
        int best_down = 0, best_across = 0;
        for (const piece_pick& pick : picks_to_ask(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int value = 0; value < 2 * ways; ++value)
                if (const auto seen = align_holds(facts, handed, value >= ways, (partner_by)(value % ways)))
                    weigh(aligned, value, pick, *seen, move_claims(*seen));
            for (int value = 0; value <= row_of(node_type::MoveToward).high; ++value)
                if (const auto seen = toward_holds(facts, handed, value))
                    weigh(toward, value, pick, *seen, move_claims(*seen));
            for (int at = 0; at < heading::count; ++at) {
                const int way = heading_asked(at);
                if (const auto seen = to_edge_holds(facts, handed, way)) weigh(edged, way, pick, *seen, move_claims(*seen));
                if (const auto seen = shift_holds(facts, handed, way)) weigh(sized, way, pick, *seen, move_claims(*seen));
            }
            if (pick.how < 0) continue;
            for (const auto& [down, across] : offsets_to_ask(facts, handed)) {
                const auto seen = move_holds(facts, handed, down, across);
                if (!seen) continue;
                const int before = stepped.said;
                weigh(stepped, 0, pick, *seen, move_claims(*seen));
                if (stepped.said > before) { best_down = down; best_across = across; }
            }
        }
        const int most = max({ aligned.said, edged.said, sized.said, stepped.said, toward.said });
        if (most == 0) return {};
        return { most, [&, aligned, edged, sized, stepped, toward, best_down, best_across, most]() mutable {
            if (aligned.said == most)
                carry(say(aligned, rule::align_to,
                          string("moves level with the ") + (aligned.value >= ways ? "columns" : "rows") + " of " +
                              partners[aligned.value % ways], moves));
            else if (edged.said == most)
                carry(say(edged, rule::to_edge, string("slides ") + heading_names[edged.value] + " to the edge", moves));
            else if (sized.said == most)
                carry(say(sized, rule::shifted, string("moves its own size ") + heading_names[sized.value], moves));
            else if (stepped.said < most) {
                const int kind = toward.value % PARTNER_KINDS;
                carry(say(toward, rule::move_toward,
                          string("moves ") + how_far[toward.value / PARTNER_KINDS] + " toward " +
                              (kind < ways ? string(partners[kind]) : "the piece of color " + to_string(kind - ways)),
                          moves));
            } else {
                // Said as the acts a pick by color says: down, then across, each only
                // when it moves at all -- the pick by holes or size is not changed by
                // a move, so the second finds what the first moved.
                heard across = stepped;
                across.claims = [](size_t, const left_over_cell&) { return false; };
                if (best_down) {
                    stepped.value = best_down;
                    carry(say(stepped, rule::move_down, best_down > 0 ? "moves " + to_string(best_down) + " down"
                                                                      : "moves " + to_string(-best_down) + " up",
                              moves));
                }
                if (best_across) {
                    across.value = best_across;
                    if (!best_down) across.claims = stepped.claims;
                    carry(say(across, rule::move_across,
                              best_across > 0 ? "moves " + to_string(best_across) + " right"
                                              : "moves " + to_string(-best_across) + " left", moves));
                }
            }
        } };
    };

    // A COPY ANOTHER PIECE DECIDES: every picked piece is drawn again mirrored
    // around its partner's middle, or the grid's -- left to right, top to bottom,
    // or all four ways. The copies are drawn turned, so it accounts for the copies
    // and the turns of the pieces it mirrors, and for the cells no match covered
    // that the copies land on.
    auto mirror_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::where_it_was_copied);
        if (!facts.hand_known || !from) return {};
        heard best;
        for (const piece_pick& pick : picks_to_ask(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int value = 0; value < (int)mirror_by::count * MIRROR_PARTNERS; ++value) {
                const auto seen = mirror_holds(facts, handed, (mirror_by)(value / MIRROR_PARTNERS),
                                               value % MIRROR_PARTNERS);
                if (!seen) continue;
                weigh(best, value, pick, *seen, [kept = seen->left, landed = seen->drawn](size_t which,
                                                                                          const left_over_cell& cell) {
                    const bool mine = on_cells(kept, which, cell.row, cell.col) || on_cells(landed, which, cell.row, cell.col);
                    if (cell.waits_for(missing::where_it_was_copied) || cell.waits_for(missing::how_it_turned))
                        return mine;
                    return cell.waits < 0 && !cell.from_input && on_cells(landed, which, cell.row, cell.col);
                });
            }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const int how = best.value % MIRROR_PARTNERS;
            carry(say(best, rule::mirror_around,
                      string("drawn again mirrored ") + mirrors[best.value / MIRROR_PARTNERS] + " around the middle of " +
                          (how < ways ? string(partners[how]) : string("the grid")), moves)); } };
    };

    // A PIECE HOLLOWED TO ITS EDGE: every picked piece keeps only its outline. It
    // accounts for the cells those pieces keep and drop, whatever they waited on.
    auto outline_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::how_it_changed_shape);
        if (!facts.hand_known || !from) return {};
        heard best;
        for (const piece_pick& pick : picks_to_ask(facts, from)) {
            const auto seen = outline_holds(facts, handed_pieces(facts, pick));
            if (!seen) continue;
            weigh(best, 0, pick, *seen, [kept = seen->drawn, dropped = seen->left](size_t which,
                                                                                   const left_over_cell& cell) {
                return (cell.waits < 0 || cell.waits_for(missing::how_it_changed_shape)) &&
                       (on_cells(kept, which, cell.row, cell.col) || on_cells(dropped, which, cell.row, cell.col));
            });
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            carry(say(best, rule::outlined, "keeps only its outline",
                      read.fired_named(correspondence_election::sits_on_name))); } };
    };

    // A RECOLOR A PICK BY HOLES OR SIZE DECIDES: the pieces it picks all come out
    // in one color -- the pick, then Recolor; a pick by color already says so as
    // an act, unless only its pieces that would fit into a gap of their partner
    // change, so the colors an unsaid recolor starts from are asked with a fit.
    // Asked of the hand the line leaves -- or, where the hand could not be carried,
    // only of a line nothing on it moves or reshapes a piece, and the color most
    // cells wear, which a recolor changes, only of a bare line.
    auto recolor_word = [&]() -> offer {
        if (!facts.hand_known && ranges::any_of(read.line, [&](int8_t which) {
                return group_of(read.fired[which].as) == word_group::changes_pieces; }))
            return {};
        heard best;
        for (const piece_pick& pick : picks_to_ask(facts, 0, starts_from(missing::a_recolor))) {
            if (pick.how == piece_pick::most_common &&
                !facts.hand_known && !read.line.empty())
                continue;
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int8_t color = 0; color < COLOR_COUNT; ++color) {
                if (already_said(rule::recolor_picked, pick, color)) continue;
                if (const auto seen = picked_recolor_holds(facts, handed, color))
                    weigh(best, color, pick, *seen, recolor_claims(*seen));
            }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { carry(say(best, rule::recolor_picked, "becomes " + to_string(best.value), colors)); } };
    };

    // A RECOLOR THE PIECES' OWN KEYS DECIDE: every picked piece takes the color
    // its key names -- its shape, the same whichever way round (e8593010's holes:
    // one cell 3, two 2, three 1), or its symmetry -- ShapeColorMap, its table
    // read off the pairs (read_off), one color a key, the color of the rest for
    // every key it does not name. The puzzle has one table, so this is asked
    // only while nothing else holds it. It accounts for the cells it recolored.
    static constexpr const char* keys_named[] = { "shape", "symmetry" };
    static_assert(size(keys_named) == (size_t)shape_key::count);
    auto shape_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (!facts.hand_known || !from || !facts.overlays || inf.shape_color_map.holds()) return {};
        heard best;
        shape_color_map_rule best_table;
        vector<uint16_t> shown;
        for (const example& pair : facts.train) shown.push_back(shape_color_map_rule::colors_shown(pair.input));
        for (const piece_pick& pick : picks_to_ask(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int kind = 0; kind < (int)shape_key::count; ++kind) {
                // Each picked piece's key and the one color it comes out in.
                vector<shape_color_map_rule::seen_key> seen;
                bool reads = true;
                for (size_t which = 0; which < handed.size() && reads; ++which) {
                    const grid& output = facts.train[which].output;
                    const partner_pieces& cut = handed[which];
                    for (int i = 0; i < cut.count() && reads; ++i) {
                        if (!cut.picked[i]) continue;
                        // -2 until a cell is read, -3 once two cells disagree.
                        int worn = -2;
                        cut.pieces[i].each_cell([&](int row, int col, int8_t) {
                            const int there = output.data[row][col];
                            worn = worn == -2 || worn == there ? there : -3;
                        });
                        reads = is_color(worn) &&
                                shape_color_map_rule::see(seen, shape_color_map_rule::key_of((shape_key)kind,
                                                                                             cut.pieces[i]),
                                                          (int8_t)worn, which, shown[which]);
                    }
                }
                if (!reads) continue;
                const shape_color_map_rule table = shape_color_map_rule::read_off(seen);
                if (!table.holds()) continue;
                const auto repainted = recolors_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i,
                                                                        int8_t& color) {
                    const int named = table.color_of((shape_key)kind, cut.pieces[i]);
                    color = named >= 0 ? (int8_t)named : LEFT_AS_IT_IS;
                    return true;
                });
                if (!repainted) continue;
                const int before = best.said;
                weigh(best, kind, pick, *repainted, recolor_claims(*repainted));
                if (best.said > before) best_table = table;
            }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best, best_table] {
            inf.shape_color_map = best_table;
            const string rest = best_table.otherwise >= 0 ? ", any other " + to_string(best_table.otherwise) : "";
            carry(say(best, rule::shape_recolor,
                      string("takes the color its ") + keys_named[best.value] + " names" + rest, colors)); } };
    };

    // A PIECE REMOVED: every piece the pick takes is gone from every pair's output
    // -- the pick, then DeleteSelection. A tie goes to every other kind first, so
    // the line uses a piece as a partner before it goes. Where the hand could not
    // be carried, a pick by color is asked only of colors nothing on the line
    // changes; a pick by holes, size or band only where nothing on the line adds,
    // removes or reshapes a piece, since either moves the pick; the pick of the
    // color most cells wear only of a bare line, since any of them or a recolor
    // can change which color that is. It accounts for the cells the removed pieces
    // sat on.
    auto removal_word = [&]() -> offer {
        uint16_t from = 0;
        for (const auto& pair : read.pairs)
            for (const auto& cell : pair.left_over)
                if (cell.claimed_by < 0 && cell.from_input) from = (uint16_t)(from | 1u << cell.color);
        if (!from) return {};
        heard best;
        for (const piece_pick& pick : picks_to_ask(facts, from)) {
            bool may = true;
            if (facts.hand_known)
                may = true;
            else if (pick.how == piece_pick::most_common)
                may = read.line.empty();
            else if (pick.how >= 0)
                may = ranges::none_of(read.line, [&](int8_t which) {
                    const auto group = group_of(read.fired[which].as);
                    return group == word_group::changes_pieces || group == word_group::removes; });
            else
                for (int color = 0; color < COLOR_COUNT && may; ++color)
                    may = !(pick.colors >> color & 1) ||
                          ranges::none_of(read.line, [&](int8_t which) { return changes(read.fired[which], color); });
            if (!may) continue;
            if (already_said(rule::removed, pick, 0)) continue;
            const auto seen = removal_holds(facts, handed_pieces(facts, pick));
            if (!seen || seen->every_piece) continue;
            weigh(best, 0, pick, *seen, [gone = seen->left](size_t which, const left_over_cell& cell) {
                return cell.from_input && on_cells(gone, which, cell.row, cell.col);
            });
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { carry(say(best, rule::removed, "is removed", -1)); } };
    };

    const std::function<offer()> kinds[] = { color_from_word, move_word, mirror_word, outline_word, recolor_word,
                                             shape_word, removal_word };
    constexpr int rounds = 8;
    int words[size(kinds)] = {};
    while (true) {
        offer best;
        size_t chosen = 0;
        for (size_t kind = 0; kind < size(kinds); ++kind)
            if (words[kind] < rounds)
                if (offer one = kinds[kind](); one.said > best.said) {
                    best = std::move(one);
                    chosen = kind;
                }
        if (best.said == 0) break;
        best.say();
        ++words[chosen];
    }

    // WHAT THE LINE DRAWS, CELL BY CELL: the hand as the words leave it, laid as
    // its sentence lays it -- over what is down, or under it where a piece sits
    // on another -- explains every cell where it shows what the output shows,
    // however the matching paired the pieces. d23f8c26's middle column, which
    // the matching paired with pieces the words remove, is drawn by the line
    // where it stands. Asked only of a hand the words carried, on pairs that lie
    // over their outputs.
    if (facts.hand_known && facts.overlays) {
        const bool under = ranges::any_of(read.pairs, [](const auto& pair) { return pair.sits_on > 0; });
        vector<grid> drawn;
        for (size_t which = 0; which < facts.cuts.size(); ++which) {
            const grid& output = facts.train[which].output;
            grid canvas = empty_grid(output.width, output.height);
            for (const object& piece : facts.cuts[which])
                paint_piece(canvas, piece, piece.summary.y0, piece.summary.x0, !under, /*clip=*/true);
            drawn.push_back(std::move(canvas));
        }
        read.list_claims(listed("Drawn By The Line", "the words draw it as the output shows it", rule::unsaid),
                         [&](size_t which, const left_over_cell& cell) {
            return which < drawn.size() && drawn[which].contains(cell.row, cell.col) &&
                   drawn[which].data[cell.row][cell.col] == facts.train[which].output.data[cell.row][cell.col];
        });
    }

    // The picks a draw word is asked with: each color on its own, then the picks by
    // holes, size and band.
    vector<piece_pick> draw_picks;
    for (int color = 0; color < COLOR_COUNT; ++color) draw_picks.push_back(piece_pick::of_color(color));
    for (const piece_pick& pick : picks_to_ask(opened, 0)) draw_picks.push_back(pick);

    // WHAT A PAINT SAID AFTER THE LINE READS: it opens the cut again, so it reads
    // the pieces as the cut hands them over, as its replay does, and paints only
    // what it draws. Laid over what is down -- rays, trails, copies, repeats -- it needs
    // nothing of the line at all. Laid under it -- bars, filled hollows, filled
    // boxes -- it needs the line to cover what the cut's pieces cover, so nothing
    // on the line may move, reshape or remove a piece. Each accounts for the cells
    // it draws, and is asked for every pick that accounts for any.
    auto drawn_claims = [](const cells_by_pair& drawn) -> claim {
        return [drawn](size_t which, const left_over_cell& cell) {
            return !cell.from_input && on_cells(drawn, which, cell.row, cell.col);
        };
    };
    // Laid over what is down, a paint explains what it covers as well: an input
    // cell under it came out in the paint's color, its replay saw to that, so a
    // piece under 4f537728's repeated marker needs no recolor of its own.
    auto covered_claims = [](const cells_by_pair& drawn) -> claim {
        return [drawn](size_t which, const left_over_cell& cell) { return on_cells(drawn, which, cell.row, cell.col); };
    };
    // Whether the line leaves every piece where the cut finds it: none of its
    // words moves, reshapes or removes one, though it may recolor them.
    const bool still = ranges::all_of(read.line, [&](int8_t which) {
        const auto group = group_of(read.fired[which].as);
        return group == word_group::recolors || group == word_group::paints; });
    // WHAT IS DOWN WHERE A PAINT LANDS: the cut's pieces, which a still line
    // leaves where they lie, and every paint said before it. Laid under it or
    // only onto it, a paint reads it, so those two ways are asked only of a still
    // line; laid over it, a paint needs nothing of it.
    vector<cells_down> canvas = pieces_down(opened);
    // Says a paint, and lays what it draws on the canvas.
    auto say_paint = [&](const heard& best, rule::said_as as, const string& says) {
        const int8_t said = say(best, as, says, -1);
        for (size_t which = 0; which < best.seen.drawn.size() && which < canvas.size(); ++which)
            for (const auto& [row, col] : best.seen.drawn[which])
                canvas[which].at[row][col] = true;
        return said;
    };
    vector<int8_t> lays = { rule::laid_over };
    if (still) lays.insert(lays.end(), { rule::laid_under, rule::laid_onto });
    static constexpr const char* laid_words[] = { "", ", laid under what is down", ", only onto what is down" };

    // THE PAINTS, THE ONE THAT SAYS THE MOST FIRST. Each kind -- rays, trails,
    // copies, repeats, the turned figure, bars, filled hollows, filled boxes --
    // offers its best over every pick, the kind whose best accounts for the most
    // cells nothing claims yet is said, a tie going in that order, and then every
    // kind is asked again of the claims and the canvas that paint leaves, so a
    // kind can be said again: a piece grows rays to the sides and then along the
    // diagonals. Asked in a fixed order instead, the first kind took what a later
    // one says whole: e729b7be's row from the middle repeated down the grid, a
    // coincidence of the pairs, was said ahead of the half turn that draws every
    // output. Each kind's pieces are handed over once, the cut opened again.
    // A PAINT IS WEIGHED ONLY WHERE IT EXPLAINS A CELL IN EVERY PAIR that
    // nothing claims yet. One that explains a cell or two of one pair redoes, by
    // coincidence, what the line already did in the others: 91714a58's rays one
    // cell long, in the ground's color, three of them, finished a line of color
    // coincidences that is wrong on its test.
    auto weigh_paint = [&](heard& best, int value, const piece_pick& pick, const replayed& seen, claim claims) {
        for (size_t which = 0; which < read.pairs.size(); ++which)
            if (ranges::none_of(read.pairs[which].left_over, [&](const left_over_cell& cell) {
                    return cell.claimed_by < 0 && claims(which, cell); }))
                return;
        weigh(best, value, pick, seen, std::move(claims));
    };

    vector<vector<partner_pieces>> handed_by_pick;
    for (const piece_pick& pick : draw_picks) handed_by_pick.push_back(handed_pieces(opened, pick));

    // RAYS A PIECE GROWS: up, down and to the sides, or along the diagonals, as
    // long as every pair bears out, painted in one color, its own or one the line
    // names, which is a question of its own. A tie goes to the longer ray: a ray
    // that explains no more for being longer has run into the edge, and a larger
    // test grid is exactly where the shorter reading stops too soon (ea786f4a).
    // Asked again while a ray accounts for any cell, so one piece can grow both
    // kinds: dc1df850's ring is the four to the sides and the four along the
    // diagonals, and said once, the diagonals were left to any other pick that
    // happened to take the same cells.
    const auto rays = row_of(node_type::ExtendOrthogonal);
    static_assert(row_of(node_type::ExtendOrthogonal).low == row_of(node_type::ExtendDiagonal).low &&
                  row_of(node_type::ExtendOrthogonal).high == row_of(node_type::ExtendDiagonal).high &&
                  row_of(node_type::ExtendDiagonal).high < paint_value::direction_bit,
                  "one ray length is said the same both ways round, and fits under the diagonal's 32");
    auto ray_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (size_t at = 0; at < draw_picks.size(); ++at)
            for (const int8_t lay : lays)
                for (const bool diagonal : { false, true })
                    for (int radius = rays.high; radius >= rays.low; --radius) {
                        const auto seen = rays_hold(opened, handed_by_pick[at], canvas, diagonal, radius, lay);
                        if (!seen) continue;
                        const int before = best.said;
                        weigh_paint(best, paint_value::packed(diagonal, radius), draw_picks[at], *seen,
                                    covered_claims(seen->drawn));
                        if (best.said > before) best_lay = lay;
                    }
        if (best.value < 0) return {};
        return { best.said, [&, best, best_lay] {
            const bool own = best.pick.how < 0 && best.pick.colors == (uint16_t)(1u << best.seen.paint);
            const int8_t said = say_paint(best, rule::extend_rays,
                string("grows rays ") + (paint_value::second(best.value) ? "along the diagonals" : "up, down and to the sides") + ", " +
                    to_string(paint_value::number(best.value)) + " long, in " +
                    (own ? string("its own color") : "color " + to_string(best.seen.paint)) + laid_words[best_lay]);
            read.fired[said].lay = best_lay; } };
    };

    // A TRAIL A PIECE GROWS one of the eight ways, in its own color, until it meets
    // the edge of the grid or another piece -- asked again, like the rays, while a
    // trail accounts for any cell.
    static_assert(heading::count == row_of(node_type::ExtendTo).high + 1);
    auto trail_paint = [&]() -> offer {
        heard best;
        for (size_t at = 0; at < draw_picks.size(); ++at)
            for (int asked = 0; asked < heading::count; ++asked) {
                const int way = heading_asked(asked);
                if (const auto seen = trail_holds(opened, handed_by_pick[at], way))
                    weigh_paint(best, way, draw_picks[at], *seen, covered_claims(seen->drawn));
            }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::trailed, string("grows a trail ") + heading_names[best.value] + " until it meets something");
        } };
    };

    // THE PICKED PIECES COPIED ONTO THE OTHERS, lined up one of CopyTo's nine ways.
    static_assert(row_of(node_type::CopyTo).high + 1 == 9);
    static constexpr const char* lined_up[] = { "top-left", "top", "top-right", "left", "middle", "right",
                                                "bottom-left", "bottom", "bottom-right" };
    auto copy_paint = [&]() -> offer {
        heard best;
        for (size_t at = 0; at < draw_picks.size(); ++at)
            for (int anchor = 0; anchor < (int)size(lined_up); ++anchor) {
                if (const auto seen = copy_onto_holds(opened, handed_by_pick[at], anchor))
                    weigh_paint(best, anchor, draw_picks[at], *seen, covered_claims(seen->drawn));
            }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::copied_onto, string("is copied onto the other pieces, lined up ") + lined_up[best.value]);
        } };
    };

    // A PIECE REPEATED ALONG ITS ROWS OR DOWN ITS COLUMNS, every so many cells, to
    // the edges -- asked again, like the rays, while a repeat accounts for any
    // cell, so one piece can be repeated both ways. A tie goes to the shorter
    // period, the tighter spacing the pairs bear out.
    const auto tiles = row_of(node_type::TileX);
    static_assert(row_of(node_type::TileX).low == row_of(node_type::TileY).low &&
                  row_of(node_type::TileX).high == row_of(node_type::TileY).high &&
                  row_of(node_type::TileY).high < paint_value::direction_bit,
                  "one period is said the same both ways round, and fits under the columns' 32");
    auto repeat_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (size_t at = 0; at < draw_picks.size(); ++at)
            for (const int8_t lay : lays)
                for (const bool down : { false, true })
                    for (int period = tiles.low; period <= tiles.high; ++period) {
                        const auto seen = tile_holds(opened, handed_by_pick[at], canvas, down, period, lay);
                        if (!seen) continue;
                        const int before = best.said;
                        weigh_paint(best, paint_value::packed(down, period), draw_picks[at], *seen,
                                    covered_claims(seen->drawn));
                        if (best.said > before) best_lay = lay;
                    }
        if (best.value < 0) return {};
        return { best.said, [&, best, best_lay] {
            const int8_t said = say_paint(best, rule::tiled_along,
                "is repeated every " + to_string(paint_value::number(best.value)) + " cells " +
                    (paint_value::second(best.value) ? "down its columns" : "along its rows") + laid_words[best_lay]);
            read.fired[said].lay = best_lay; } };
    };

    // THE WHOLE FIGURE DRAWN AGAIN TURNED in its own box -- a quarter turn
    // right, a half turn, a quarter turn left, each a copy of its own -- asked
    // again while a turn accounts for any cell, so it can be turned all three
    // ways (11852cab's design, the same from all four sides). Every piece, as one
    // figure: a part of the drawing turned in a box of its own fit the pairs by
    // accident (2546ccf6's quadrants, which mirror across walls; 52df9849's
    // hidden corner). It accounts for the figure where it lies as well as for its
    // copies, so it is asked only of a still line.
    const int right = pose_of(node_type::TurnRight);
    const std::pair<int, const char*> turns[] = { { right, "a quarter turn right" },
                                                  { compose_poses(right, right), "a half turn" },
                                                  { pose_of(node_type::TurnLeft), "a quarter turn left" } };
    const piece_pick figure = piece_pick::all();
    const vector<partner_pieces> figure_handed = still ? handed_pieces(opened, figure) : vector<partner_pieces>{};
    auto turned_paint = [&]() -> offer {
        if (!still) return {};
        heard best;
        for (const auto& [pose, name] : turns) {
            if (const auto seen = turned_holds(opened, figure_handed, pose))
                weigh_paint(best, pose, figure, *seen, covered_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const auto turn = ranges::find_if(turns, [&](const auto& one) { return one.first == best.value; });
            say_paint(best, rule::turned_copy, string("is drawn again as one figure, ") + turn->second + " in its box");
        } };
    };

    // BARS BETWEEN PIECES OF ONE COLOR, in their own color or one of the line's,
    // reaching each way a bridge can, the narrowest first, so a wider way is said
    // only where it accounts for more.
    static constexpr const char* reaches[BRIDGE_WAYS] = {
        "both ways to the nearest", "along rows to the nearest", "along columns to the nearest",
        "both ways to every one", "along rows to every one", "along columns to every one" };
    const vector<partner_pieces> bar_handed = still ? handed_pieces(opened, {}) : vector<partner_pieces>{};
    auto bar_paint = [&]() -> offer {
        if (!still) return {};
        heard best;
        for (const int way : bridge_ways_narrowest_first) {
            auto seen = bridge_holds(opened, bar_handed, canvas, way);
            if (!seen) continue;
            // Bridge reads every piece, whatever is picked, so it is said of every one.
            seen->every_piece = true;
            weigh_paint(best, way, {}, *seen, drawn_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::bridged,
                string("bars filled between pieces of one color, reaching ") + reaches[best.value] + ", in " +
                    (best.seen.paint < 0 ? string("their own color") : "color " + to_string(best.seen.paint)));
        } };
    };

    // THE HOLLOWS OF THE PICKED PIECES, FILLED, in their own color or one of the line's.
    auto fill_paint = [&]() -> offer {
        if (!still) return {};
        heard best;
        for (size_t at = 0; at < draw_picks.size(); ++at) {
            if (const auto seen = fill_holds(opened, handed_by_pick[at], canvas))
                weigh_paint(best, 0, draw_picks[at], *seen, drawn_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::filled,
                      string("fills its hollows in ") +
                          (best.seen.paint < 0 ? string("its own color") : "color " + to_string(best.seen.paint)));
        } };
    };

    // THE BOXES OF THE PICKED PIECES, FILLED where the pieces leave them empty, in
    // one color.
    auto box_paint = [&]() -> offer {
        if (!still) return {};
        heard best;
        for (size_t at = 0; at < draw_picks.size(); ++at) {
            if (const auto seen = box_holds(opened, handed_by_pick[at], canvas))
                weigh_paint(best, 0, draw_picks[at], *seen, drawn_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { say_paint(best, rule::boxed, "fills its box in color " + to_string(best.seen.paint)); } };
    };

    const std::function<offer()> paints[] = { ray_paint, trail_paint, copy_paint, repeat_paint, turned_paint,
                                              bar_paint, fill_paint, box_paint };
    while (true) {
        offer best;
        for (const auto& kind : paints)
            if (offer one = kind(); one.said > best.said) best = std::move(one);
        if (best.said == 0) break;
        best.say();
    }
}

// THE ACTS LEFT UNSAID: the line the matching opened with its acts, cleared.
// They stay listed, claiming nothing, and every cell they claimed is the words'
// to claim again.
inline void leave_the_acts_unsaid(inference& inf) {
    auto& read = inf.correspondence.readings[inf.correspondence.chosen];
    for (auto& pair : read.pairs)
        for (auto& cell : pair.left_over)
            if (cell.claimed_by >= 0 && ranges::find(read.line, cell.claimed_by) != read.line.end())
                cell.claimed_by = -1;
    for (const int8_t which : read.line) read.fired[which].cells = 0;
    read.line.clear();
}

// WHETHER THE CUT'S PIECES GO UNSAID WHOLE: matched, they do not redraw the
// outputs where they lie, and nothing on the line says what else happened.
inline bool pieces_go_unsaid(const inference& read) {
    const auto& facts = read.correspondence;
    return !facts.chosen_pairs().empty() && !read.pieces_redraw && facts.readings[facts.chosen].line.empty();
}

// HOW MUCH OF THE PAIRS A READING LEAVES UNSAID: what its chosen cut's line
// leaves unclaimed -- every cell, where its pieces go unsaid whole -- or its
// framed line where that says more; none where a replayed standalone rule
// already redraws every output.
inline int cells_left_unsaid(const inference& read) {
    const auto& readings = read.correspondence.readings;
    if (readings.empty()) return numeric_limits<int>::max();
    const auto& chosen = readings[read.correspondence.chosen];
    if (ranges::any_of(chosen.fired, [](const auto& rule) { return rule.stands_alone; })) return 0;
    int unsaid = pieces_go_unsaid(read) ? read.correspondence.cells : chosen.unclaimed();
    if (read.framed && !read.framed->correspondence.readings.empty()) {
        const auto& inside = read.framed->correspondence;
        const auto& line = inside.readings[inside.chosen];
        if (read.framed->pieces_redraw || !line.line.empty()) unsaid = min(unsaid, line.unclaimed());
    }
    return unsaid;
}

// THE CANDIDATE WHOSE WORDS SAY THE MOST: each one read in the order given, and
// the one whose reading leaves the fewest cells unsaid kept, a tie going to the
// one read first -- so once one leaves none, no later one is read. Every choice
// the reading makes is made here and judged this one way, by its words: the
// ground plan (infer) and the cut (read_figures), each asked just above the
// first word that reads it. Whatever orders the candidates only breaks a tie.
// Hands back the one kept, where it stood, and the cells each one left unsaid,
// -1 for one never read.
struct said_most {
    inference reading;
    int kept = -1;
    vector<int> left;
};
template <class Read>
inline said_most says_the_most(int count, Read read) {
    said_most most;
    most.left.assign((size_t)count, -1);
    for (int i = 0; i < count; ++i) {
        inference one = read(i);
        most.left[i] = cells_left_unsaid(one);
        if (most.kept < 0 || most.left[i] < most.left[most.kept]) {
            most.reading = std::move(one);
            most.kept = i;
        }
        if (most.left[most.kept] == 0) break;
    }
    return most;
}

inline void read_the_cut(inference& inf, const vector<example>& train);

// THE READING OF FIGURES: pairs with no ground in them -- a plan deleted it,
// or keeps every color as a figure -- read with no election and no ground in
// any reader, so every detector sees figures and empty cells, and an empty cell
// is nothing (where a plan deleted the ground, what it fills last). What reads
// no cut is asked once; what reads one is asked of each cut the words weigh.
inline inference read_figures(const vector<example>& train) {
    inference inf;
    inf.size = output_dimension_rule::infer_rule(train);
    inf.colors = color_facts::detect(train);  // rule-independent
    // No reader is handed a ground color: the ground is empty where a plan
    // deleted it, and a color like any other where a plan kept it.
    inf.colors.detect_output_colors_by_size(train);
    // The frame before the figures: whether this input is really several grids
    // side by side, walls and blank lines told apart by what is on them -- an
    // empty line is blank -- ahead of everything that would read the panes.
    inf.pane_split = pane_split_facts::detect(train);
    // What the panes come to, once there are panes. It reads the masks, not the
    // colors, so a combination the pairs never showed is still one it can answer.
    inf.pane_mask = pane_mask_rule::detect(train, inf.pane_split);
    // And when the answer is a pane rather than something made from the panes,
    // the only question left is which one. Asked after the mask, because making
    // a drawing is the stronger claim and should get first refusal.
    inf.pane_pick = pane_pick_rule::detect(train, inf.pane_split);
    // And when no one pane is the answer and nothing is made of them together,
    // every pane may be: its drawing kept, cropped, and laid where the pane was.
    inf.pane_gather = pane_gather_rule::detect(train, inf.pane_split, inf.size);
    // A noisy lattice cleaned by the vote. It reads the whole input, not the
    // pieces, and the ground the op reads off it (denoise_ground).
    inf.periodic_denoise = periodic_denoise_rule::detect(train);
    inf.copy_to = copy_to_rule::detect(train, inf.pane_split);
    inf.interior_fill = interior_fill_rule::detect(train);
    inf.interior_spread = interior_spread_rule::detect(train);
    inf.ranked_layout = ranked_layout_rule::detect(train, inf.size);
    inf.rarest_color = rarest_color_rule::detect(train, inf.size);
    inf.corner_crop = corner_crop_rule::detect(train, inf.size);

    // Dims is detector #0: its rule decides which questions even apply
    // (Backward Deduction step 1, applied to detectors instead of ops).
    // Identity = input and output overlay cell-for-cell, so the whole
    // pair-relation toolbox below is meaningful. Each detector re-asks the
    // same question itself (pairs_overlay), so this gate states the intent
    // and none of them depends on being called from here.
    if (inf.size.Chosen == output_dimension_rule::Identity) {
        inf.hole_colors = hole_color_rule::detect(train);
        inf.ray_extension = ray_extension_rule::detect(train);
        inf.unchanged   = unchanged_cells_facts::detect(train);
        inf.color_diff  = color_diff_facts::detect(train);
        inf.color_swaps = color_swap_candidates::detect(train, inf.color_diff);
    }
    // Scale / Fixed / BoundingBox: no rule-specific detectors yet.
    // Scale could later reuse unchanged_cells through the ratio (cell<->block).

    // Edge level of correspondence: the figures cut up every way that competes,
    // ranked by how well the pieces match.
    inf.correspondence = correspondence_election::infer(train);
    // THE CUT IS CHOSEN HERE, just above the first word that reads it: each cut
    // in the matching's order is read to the end of its line, and the one whose
    // words say the most is said (says_the_most).
    const vector<int8_t> ranked = inf.correspondence.ranked;
    if (ranked.empty()) {
        read_the_cut(inf, train);
        return inf;
    }
    said_most said = says_the_most((int)ranked.size(), [&](int rank) {
        inference read = inf;
        read.correspondence.choose((size_t)rank);
        read_the_cut(read, train);
        return read;
    });
    inference best = std::move(said.reading);
    best.correspondence.left_by_rank = said.left;
    // A tied opening the words ruled out -- read, and leaving more cells
    // unsaid -- is no longer one the reading could not rule out.
    erase_if(best.correspondence.tied_with_chosen, [&](int8_t tie) {
        const auto at = ranges::find(ranked, tie);
        const int left = at == ranked.end() ? -1 : said.left[(size_t)(at - ranked.begin())];
        return left > said.left[(size_t)said.kept];
    });
    return best;
}

// THE WORDS THAT READ THE CUT, asked of the one `inf` reads on: the rules read
// off its pieces, then -- once every rule is known -- the rivals, the rest of
// its line, what no rule on it says, and a size change read through its frame.
inline void read_the_cut(inference& inf, const vector<example>& train) {
    inf.motif_continuation = motif_continuation_rule::detect(train, inf.correspondence);
    // The same continuation read along the diagonals. Asked after the motif
    // because a repeating box is the plainer claim and should get first refusal.
    inf.diagonal_continuation = diagonal_continuation_rule::detect(train, inf.correspondence);
    inf.piece_slam = piece_slam_rule::detect(train, inf.correspondence);
    inf.overlap_corner = overlap_corner_rule::detect(train, inf.correspondence);
    // Pieces that slid up against the ones that stayed. Asked after the chain
    // because a layout that puts every piece somewhere new is the larger claim
    // and should get first refusal; this one leaves most of the grid alone.
    inf.slide_to_touch = slide_to_touch_rule::detect(train, inf.correspondence);
    // Pieces that slide to the canvas edge, over whatever is in the way. It reads
    // its own cut when the elected one joins the slider to what it passes over.
    inf.edge_slide = edge_slide_rule::detect(train, inf.correspondence.segmentation, inf.size);
    inf.project_on_seeds = project_on_seeds_rule::detect(train, inf.correspondence);
    inf.snap_to_anchors = snap_to_anchors_rule::detect(train, inf.correspondence, inf.size);
    // Gaps between pieces of one colour, filled. It reads its own cut when the
    // elected one cannot say it, so it does not wait on the election's verdict.
    inf.bridge = bridge_rule::detect(train, inf.correspondence);
    // Marks laid as lines that take turns down the grid. Like the bridge, it
    // falls back on a cut of its own when the elected one cannot say it.
    inf.mark_lattice = mark_lattice_rule::detect(train, inf.correspondence);
    inf.input_blocks = input_blocks_rule::detect(train, inf.correspondence);
    if (inf.size.Chosen == output_dimension_rule::Identity) {
        inf.shape_color_map = shape_color_map_rule::detect_body_recolor(train, inf.correspondence);
        // Fallback: same shape->color table, but keyed on each object's own
        // shape and applied as an interior fill (ShapeToInteriorColorMap)
        // instead of a body recolor. Only run when the body-recolor
        // hypothesis didn't validate; the winner is recorded in claimed_by,
        // which is what tells the two ops apart downstream.
        if (!inf.shape_color_map.holds())
            inf.shape_color_map = shape_color_map_rule::detect_interior_fill(train);
    }

    // PIECES THAT DO NOT REDRAW THE OUTPUT where they lie: the cut's pieces,
    // matched, can leave nothing over and still not be the output -- 2013d3e2's
    // figure matched whole while the output is its corner, cce03e0d's copies
    // matched where half of them are gone -- so a line with no words is held to
    // what it would draw. Asked of the cut alone, before the line is chosen.
    inf.pieces_redraw = redraws_every_output(train, [&](const example& pair, grid& drawn) {
        const auto pieces = pieces_as_said(pair.input, inf.correspondence.segmentation);
        return !pieces.empty() && draw_on_canvas(pair.input, inf.size, drawn, [&](grid& canvas) {
            for (const object& piece : pieces)
                paint_piece(canvas, piece, piece.summary.y0, piece.summary.x0, true, /*clip=*/true);
            return true;
        });
    });

    // Last, once every rule is known: the rivals, the rest of the cut's line,
    // and only then what no rule on it says. THE ACTS, OR THE WORDS ALONE: the
    // acts the matching read off the pieces open the line and the words read
    // off the pieces go on from the hand they leave, or the acts are left
    // unsaid and the words say every change themselves. A word that says what
    // an act says and more is seen at work only on a hand the act has not
    // changed yet: 9565186b's "8 becomes 5" left its first pair nothing for
    // "every color but the most common becomes 5" to change. Whichever says
    // the most is said (says_the_most), a tie keeping the acts.
    list_the_rivals(inf);
    const auto& election = inf.correspondence;
    const bool acts = !election.readings.empty() && !election.readings[election.chosen].line.empty();
    inf = says_the_most(acts ? 2 : 1, [&](int alone) {
        inference read = inf;
        if (alone) leave_the_acts_unsaid(read);
        finish_the_line(read, train);
        return read;
    }).reading;
    inf.correspondence.list_what_is_missing();

    // A SIZE CHANGE, READ THROUGH ITS FRAME: once the frame cuts the output's
    // grid out of the input's, what is left is a same-size puzzle, read by every
    // word a same-size puzzle has -- from each framed input to its output.
    inf.frame = choose_piece_frame(train, inf);
    if (inf.frame.holds()) {
        using kind = piece_frame_rule::kind_of;
        // The frames that take the pieces a pick takes off a cut of their own;
        // the rest take the whole input.
        const bool picks = inf.frame.kind == kind::cut_out || inf.frame.kind == kind::laid_out ||
                           inf.frame.kind == kind::filled;
        // The framed grid is what the frame leaves of the figures: nothing is
        // filled in it, and the reading inside it is a reading of figures too.
        const replay_facts facts(train, picks ? inf.frame.cut : inf.correspondence.segmentation);
        const piece_pick& pick = inf.frame.pick;
        vector<example> framed(train.size());
        bool all = true;
        for (size_t which = 0; which < train.size() && all; ++which) {
            framed[which].output = train[which].output;
            const grid& input = train[which].input;
            grid& into = framed[which].input;
            bool several = false;
            switch (inf.frame.kind) {
            case kind::laid_out:
                all = laid_out_input(facts.cuts[which], pick, inf.frame.factor, input, inf.size, into);
                break;
            case kind::filled:
                all = filled_input(facts.cuts[which], pick, inf.frame.keyed, inf.shape_color_map, input, inf.size,
                                   into);
                break;
            case kind::cut_out:
                all = framed_input(facts.cuts[which], pick, input, inf.size, into, several);
                break;
            default: all = framed_whole(input, inf.frame, inf.size, into);
            }
        }
        if (!all) inf.frame = {};
        else inf.framed = std::make_shared<const inference>(read_figures(framed));
    }
}

// Asks every question about the training pairs, in order, and returns the answers.
// THE GROUND IS OWNED ONCE: the election names the plans a puzzle allows
// (ground_plans), each plan's figures are read (read_figures), and the plan
// whose words say the most is said (says_the_most), in the order the plans are
// named. Nothing after this knows a ground but the plan.
inline inference infer(const vector<example>& train) {
    const background_election election = elect_background(train);
    const vector<ground_plan> plans = ground_plans(election);
    inference best = says_the_most((int)plans.size(), [&](int which) {
        inference read = read_figures(figures_only(train, plans[(size_t)which]));
        read.plan = plans[(size_t)which];
        return read;
    }).reading;
    best.ground_election = election;
    return best;
}

}  // namespace arc
