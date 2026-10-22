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
#include "8b_op_math.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace arc {

// Diagnostic work counts, local to the worker; never read by inference decisions.
// arc_said resets these for each puzzle. Count calls, not cells or clock reads.
struct inference_work_counts {
    unsigned long long preparations = 0, readings = 0, later_readings = 0, lines = 0;
    unsigned long long paint_replays = 0, slide_replays = 0, drop_replays = 0, walks = 0, lays = 0;
};
inline thread_local inference_work_counts inference_work;
// TEMPORARY, to find where the slowest puzzles spend their reading: elapsed
// milliseconds by reading, by stage, and by each kind of word and paint a line
// asks, in finish_the_line's order. Nested: a reading holds its stages.
struct inference_times {
    static constexpr const char* names[] = { "strict", "tolerant", "partners", "preparing",
        "lines_first", "lines_later", "reading_ahead", "color_from", "move", "mirror", "outline",
        "recolor", "shape", "key", "removal", "ray", "trail", "copy", "repeat", "turned", "bar",
        "fill", "box", "room", "symmetry", "picture" };
    enum at { strict, tolerant, partners, preparing, lines_first, lines_later, reading_ahead, first_kind };
    double ms[size(names)] = {};
};
inline thread_local inference_times inference_time;
// Adds the time from here to the end of the scope to `total`.
struct time_into {
    double& total;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    ~time_into() { total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); }
};
// Optional runner-owned audit; not consulted by any inference decision.
inline thread_local std::function<void(const std::vector<example>&, const tiled_pieces&,
                                      double, unsigned long long, unsigned long long)> preparation_audit;
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
            const color_tally counts(out);
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
            // Cells of each color; a color is present where its count is not 0.
            const color_tally in(ex.input), out(ex.output);
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
            ex.output.for_each_colored_cell(TRANSPARENT, [&](int, int, int color) {
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
    // encoding BidiColorMap decodes
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

    static hole_color_rule detect(const vector<example>& train, const tiled_pieces& tiles) {
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
                        occupied[r - box.row][c - box.col] = true;
                        const int target = ex.output.data[r][c];
                        if (!is_color(target) || (color >= 0 && color != target)) {
                            consistent = false;
                            return;
                        }
                        color = target;
                        changed = changed || target != ex.input.data[r][c];
                    }
                    const int key = object_hole_count(box.height, box.width,
                        [&](int r, int c) { return occupied[r][c]; });
                    if (key >= key_count || (rule.colors[key] >= 0 && rule.colors[key] != color)) {
                        consistent = false;
                        return;
                    }
                    rule.colors[key] = (int8_t)color;
                }, tiles.of(ex));
            if (!consistent) return {};
        }
        rule.held = changed;
        return rule;
    }
};

// Read just the pieces, in the same size-and-position order used by matching.
inline vector<object> pieces_of(const grid& input, segmentation_methods segmentation, tile_size tiles = {}) {
    vector<object_segmentation::object_info> info;
    vector<object> pieces;
    object_segmentation::segment_for_correspondence(input, segmentation, info, &pieces, 0, 0, tiles);
    return pieces;
}

// What two pieces of a hand answer, worked out once: touching [0], touching at a
// side [1], then each likeness of shape.
struct hand_answers {
    static constexpr int QUESTIONS = 2 + SHAPE_LIKENESSES;
    // Per question, piece by piece: -1 not asked yet.
    array<vector<int8_t>, QUESTIONS> kept;
    int pieces = 0;
    template <class Work> bool answer(int question, int mine, int other, int count, Work work) {
        if (count != pieces) forget(count);
        vector<int8_t>& asked = kept[(size_t)question];
        if (asked.empty()) asked.assign((size_t)count * (size_t)count, -1);
        int8_t& known = asked[(size_t)mine * (size_t)count + (size_t)other];
        if (known < 0) known = work() ? 1 : 0;
        return known != 0;
    }
    void forget(int count = 0) {
        for (vector<int8_t>& asked : kept) asked.clear();
        pieces = count;
    }
};

// THE PIECES A WORD IS HANDED, as the cut's word hands them over, cut once per
// pair and read by every replay below: where each lies, and whether the word's
// pick takes it -- the questions the partner helpers ask (each_partner and kin,
// 8b_op_math.hpp), asked here as the ops ask them of their objects, so a replay
// and the op it replays cannot find different partners or fits.
struct partner_pieces {
    const vector<object>& pieces;
    vector<cell_box> boxes;
    vector<bool> picked;
    // The hand's, shared by every pick.
    hand_answers& answers;
    // The pieces of a cut, the ones `marks` names picked.
    partner_pieces(const vector<object>& cut, vector<bool> marks, hand_answers& kept)
        : pieces(cut), boxes(cut.size()), picked(std::move(marks)), answers(kept) {
        for (size_t i = 0; i < pieces.size(); ++i) boxes[i] = pieces[i].painted_box();
    }
    // The pieces of a cut, the ones `pick` takes picked (picked_pieces, below),
    // on a canvas the size of `canvas`, where a pick reads the parts walls fence.
    partner_pieces(const vector<object>& cut, const piece_pick& pick, tile_size canvas, hand_answers& kept);
    // WHAT THE PICKED PIECES SHOW where they lie on their grid, `width` by
    // `height`, cell by cell, TRANSPARENT where none does: a cell a paint lays
    // again as it was is no work (paint_holds). Drawn the first time a replay
    // asks and kept, for every length and lay a word is asked with: neither the
    // pieces, the pick nor their grid change under it.
    const cells& shown(int width, int height) const {
        if (!shown_once) {
            auto made = std::make_shared<cells>();
            for (size_t i = 0; i < pieces.size(); ++i)
                if (picked[i])
                    pieces[i].each_cell([&](int row, int col, int8_t color) {
                        if (row >= 0 && col >= 0 && row < height && col < width) (*made)[row][col] = color;
                    });
            shown_once = std::move(made);
        }
        return *shown_once;
    }
    // The blocks piece i's shape is drawn in (blocks_of), its colors aside:
    // read the first time a partner way at another size asks, and kept.
    shape_blocks blocks(int i) const {
        if (blocks_read.empty()) blocks_read.assign(pieces.size(), shape_blocks{ 0, 0 });
        shape_blocks& kept = blocks_read[(size_t)i];
        if (kept.down == 0) {
            const cell_box own = boxes[(size_t)i];
            kept = blocks_of(own.width(), own.height(),
                             [&](int row, int col) { return pieces[(size_t)i].color_at(own.row0 + row, own.col0 + col); },
                             AS_IT_LIES);
        }
        return kept;
    }
    mutable vector<shape_blocks> blocks_read;
    // What the partner helpers ask.
    int count() const { return (int)pieces.size(); }
    cell_box box(int i) const { return boxes[i]; }
    bool painted(int i, int row, int col) const { return pieces[i].holds(row, col); }
    int color_at(int i, int row, int col) const { return pieces[i].color_at(row, col); }
    bool is_picked(int i) const { return picked[i]; }
    int color(int i) const { return pieces[i].first_color(); }
    // Whether piece i holds a cell of `color`.
    bool holds(int i, int color) const { return (palette(pieces[i]) >> color & 1) != 0; }
    // What each_partner asks, worked out once a hand.
    bool touching(int mine, int other, bool sides_only) const {
        return answers.answer(sides_only ? 1 : 0, mine, other, count(),
                              [&] { return touches(*this, mine, other, { 0, 0 }, sides_only); });
    }
    bool alike(int mine, int other, int like) const {
        return answers.answer(2 + like, mine, other, count(), [&] { return shapes_alike(*this, mine, other, like); });
    }
    // key_piece_at, worked out once a cell.
    int key_piece(int row, int col) const {
        if (row < 0 || col < 0 || row >= MAX_GRID_DIM || col >= MAX_GRID_DIM) return key_piece_at(*this, row, col);
        if (!keys_once) {
            keys_once = std::make_shared<key_drawing>();
            for (auto& line : keys_once->at) ranges::fill(line, NOT_ASKED);
        }
        int16_t& known = keys_once->at[row][col];
        if (known == NOT_ASKED) known = (int16_t)key_piece_at(*this, row, col);
        return known;
    }
    bool every_piece_picked() const { return ranges::all_of(picked, [](bool one) { return one; }); }
    // Piece i's one partner by `how` among the pieces not picked, or -1.
    int partner(partner_by how, int i) const { return partner_of(how, i, *this); }
    // Whether piece i has any partner by `how` among them, one or several.
    bool partnered(partner_by how, int i) const { return any_partner(how, i, *this); }
    // Whether piece i would fit into a gap of its one partner by `how`.
    bool fits(partner_by how, int i) const {
        const int other = partner(how, i);
        return other >= 0 && would_fit(i, other, *this);
    }
    // Piece i's one partner of `kind` (partner_of_kind) among the pieces not picked, or -1.
    int partner_of_kind(int kind, int i) const { return arc::partner_of_kind(kind, i, *this); }
    // HOW FAR PIECE i GOES, as the partner moves take it (the ops AlignTo and
    // MoveToward, through the same helpers), or false when it has no one partner.
    // Level with its partner's rows -- or, with `columns`, columns:
    bool goes_level(int i, bool columns, partner_by how, direction_step& by) const {
        const int other = partner(how, i);
        if (other < 0) return false;
        by = columns ? direction_step{ 0, boxes[other].col0 - boxes[i].col0 }
                     : direction_step{ boxes[other].row0 - boxes[i].row0, 0 };
        return true;
    }
    // Toward its partner of the kind `value` % PARTNER_KINDS names, by what
    // `value` / PARTNER_KINDS names (move_toward), on a canvas `width` by `height`:
    bool goes_toward(int i, int value, int width, int height, direction_step& by) const {
        const int other = partner_of_kind(value % PARTNER_KINDS, i);
        if (other < 0) return false;
        by = move_toward((toward_by)(value / PARTNER_KINDS), i, other, *this, width, height);
        return true;
    }
    // The color piece i takes from its partners by `how`, or -1.
    int color_from(partner_by how, int i) const { return partner_color(how, i, *this); }
    // The color a key drawn in the grid gives each picked piece by its place, -1
    // for none -- every one where the key does not say (key_place_colors).
    vector<int8_t> key_places() const {
        const size_t count = pieces.size();
        vector<int8_t> color(count, -1), band(count);
        vector<int16_t> lows(count), highs(count);
        vector<int> order(count);
        if (count) key_place_colors(*this, color.data(), lows.data(), highs.data(), band.data(), order.data());
        return color;
    }
    // THE CELL EACH PIECE GROWS ITS RAYS FROM, in its own box (nearest_ray_seed,
    // the cell the op takes): found the first time a replay asks and kept, for
    // every length and lay the rays are asked with.
    const vector<ray_seed>& ray_seeds() const {
        if (!seeds_once) {
            auto made = std::make_shared<vector<ray_seed>>(pieces.size());
            for (size_t i = 0; i < pieces.size(); ++i) {
                const object& piece = pieces[i];
                (*made)[i] = nearest_ray_seed(piece.summary.height, piece.summary.width, (piece.summary.height - 1) / 2,
                                              (piece.summary.width - 1) / 2,
                                              [&](int row, int col) { return piece.data[row][col]; });
            }
            seeds_once = std::move(made);
        }
        return *seeds_once;
    }
    // What shown, ray_seeds and key_piece keep, shared by a copy of the cut.
    struct key_drawing { int16_t at[MAX_GRID_DIM][MAX_GRID_DIM]; };
    static constexpr int16_t NOT_ASKED = -2;
    mutable std::shared_ptr<const cells> shown_once;
    mutable std::shared_ptr<const vector<ray_seed>> seeds_once;
    mutable std::shared_ptr<key_drawing> keys_once;
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
                                                    const correspondence_election& correspondence,
                                                    const tiled_pieces& tiles) {
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
                if (!edge.from_input || edge.from != target || !edge.same_shape || edge.scale != 1) continue;
                const int8_t worn = edge.exact ? pieces[target].color : edge.recolored_to(pieces[target].color);
                if (!is_color(worn) || (became >= 0 && became != worn)) return {};
                became = worn;
            }
            if (became < 0) return {};
            changed = changed || became != pieces[target].color;

            const auto cut = pieces_of(train[which].input, correspondence.segmentation, tiles.of(train[which]));
            if (cut.size() != pieces.size()) return {};
            // The largest takes its color from its one nearest piece, as ColorFrom
            // finds it among the pieces not picked: that must be the key.
            vector<bool> largest(cut.size(), false);
            largest[(size_t)target] = true;
            hand_answers fresh;
            if (partner_pieces(cut, largest, fresh).partner(partner_by::nearest, target) != key) return {};
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
    // at least has to be filled with a color the ground is not. Nor is a table
    // whose every key is seen in one pair alone, all filled in one color: that
    // is the color for everything, and no shape decides it (272f95fa's lattice,
    // a new shape in every pair, its middle always 6).
    static shape_color_map_rule detect_interior_fill(const vector<example>& train, const tiled_pieces& tiles) {
        shape_color_map_rule rule;
        if (!pairs_overlay(train)) return {};
        bool fills_something = false;
        // The pairs each key is seen in, a bit each.
        vector<std::pair<uint32_t, uint32_t>> seen_in;

        object_walk_scratch scratch;
        bool occupied[MAX_GRID_CELLS] = {};
        bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
        for (size_t which = 0; which < train.size(); ++which) {
            const example& ex = train[which];
            const grid& in = ex.input;
            const grid& out = ex.output;
            bool consistent = true;
            each_object(in, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int found) {
                if (!consistent) return;
                const int left = box.col, top = box.row, width = box.width, height = box.height;
                fill_n(occupied, width * height, false);
                for (int i = 0; i < found; ++i)
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

                const uint32_t object_shape = key_of_shape(width, height,
                    [&](int row, int col) { return occupied[row * width + col] ? 0 : TRANSPARENT; }, ANY_WAY_ROUND);
                consistent = rule.learn(object_shape, (int8_t)fill);
                fills_something = fills_something || fill != TRANSPARENT;
                const uint32_t pair_bit = 1u << min<size_t>(which, 31);
                const auto known = ranges::find(seen_in, object_shape, &std::pair<uint32_t, uint32_t>::first);
                if (known == seen_in.end()) seen_in.push_back({ object_shape, pair_bit });
                else known->second |= pair_bit;
            }, tiles.of(ex));
            if (!consistent) return {};
        }
        if (!fills_something) return {};
        const bool repeats = ranges::any_of(seen_in, [](const auto& one) { return popcount(one.second) >= 2; });
        const bool one_color = all_of(rule.colors, rule.colors + rule.count,
                                      [&](int8_t color) { return color == rule.colors[0]; });
        if (!repeats && one_color) return {};
        if (rule.holds()) rule.claimed_by = shape_color_map_rule::interior_fill;
        return rule;
    }

    // THE TABLE'S KEY FOR A PIECE, of the kind ShapeColorMap's value names: its
    // shape, whichever way round it lies, or its symmetry (shape_key_of, the key
    // the ops look up).
    static uint32_t key_of(shape_key kind, const object& piece) {
        return shape_key_of((int)kind, piece.summary.width, piece.summary.height,
                            [&](int row, int col) { return piece.data[row][col]; });
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
            if (!edge.from_input || !edge.same_shape || edge.pose || edge.scale != 1 || edge.hidden || edge.outside ||
                edge.from < 0 || edge.to < 0 || edge.from >= (int)pair.input_info.size() ||
                edge.to >= (int)pair.output_info.size()) return {};
            const auto& from = pair.input_info[edge.from].summary;
            const auto& into = pair.output_info[edge.to].summary;
            if (from.col || from.row || into.col || into.row ||
                from.width != into.width || from.height != into.height) return {};
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
            if (width != from.width || height != from.height) return {};
            for (int r = 0; r < input.height; ++r)
                for (int c = 0; c < input.width; ++c)
                    if (input.data[r][c] != input.data[r % height][c % width]) return {};
            for (int r = 0; r < output.height; ++r)
                for (int c = 0; c < output.width; ++c) {
                    const int was = input.data[r % height][c % width];
                    const int now = output.data[r][c];
                    // An empty cell of the motif stays empty.
                    if (is_color(was) != is_color(now)) return {};
                    if (!is_color(was)) continue;
                    if (rule.color_map[was] >= 0 && rule.color_map[was] != now) return {};
                    rule.color_map[was] = (int8_t)now;
                }
        }
        rule.held = true;
        return rule;
    }
};

// THE INPUT TILED BY ITS OWN SIZE, from where it stands onward, every copy
// taking both steps again: what a tiled frame, the input's blocks and a motif
// carried over the grid say.
inline constexpr tile_value tiled_by_its_own_size{
    .steps = { { .down = measured(tile_measure::own_height), .across = { 0 } },
               { .down = { 0 }, .across = measured(tile_measure::own_width) } },
    .start = {}, .end = { .edge = true }, .spread = tile_spread::grid };

// THE SAME CONTINUATION ALONG THE DIAGONALS. The colors cycle as the diagonals
// are crossed, the input paints part of that cycle, and the output is the cycle
// carried over every cell -- the motif above with diagonals in place of a box.
//
// THE PERIOD IS HOW MANY COLORS THERE ARE, a measure the Tile reads on each
// grid, not a choice made here: every cell steps along its diagonal, and across
// by as many cells as there are colors. Diagonals that repeat on any other
// period are a reading this has no way to say, so it does not claim them.
//
// ONE COLOR IS NO CYCLE. With a single color every diagonal is the same
// diagonal, the claim comes out true of any grid painted all one color, and it
// would say "diagonals" about a puzzle that never mentioned them.
//
// AND IT HAS TO CARRY SOMETHING. An output the input already was is the cycle
// redrawn where it already sat, which says nothing about diagonals either.
struct diagonal_continuation_rule {
    bool held = false;
    // Which diagonals: 0 those of constant row + column, stepped down and to
    // the left; 1 those of constant row - column, down and to the right.
    int8_t lean = 0;
    bool holds() const { return held; }

    // The Tile that says it, on the input's own cells, one piece each (Split[2]).
    static constexpr tile_value tile_of(int8_t lean) {
        return { .steps = { { .down = { 1 }, .across = { (int8_t)(lean ? 1 : -1) } },
                            { .down = { 0 }, .across = measured(tile_measure::color_count) } },
                 .start = { .edge = true }, .end = { .edge = true }, .spread = tile_spread::grid };
    }

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
            const tile_value tile = tile_of(lean);
            bool carried_something = false;
            for (const auto& pair : train) {
                // The input's own cells in reading order, each laid again where
                // the Tile lays it, a later cell's copies over an earlier one's.
                struct cell_at { int row, col, color; };
                vector<cell_at> cells;
                pair.input.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
                    cells.push_back({ row, col, color });
                });
                const int colors = color_count((int)cells.size(), [](int) { return true; },
                                               [&](int i) { return 1u << cells[(size_t)i].color; });
                if (colors < 2) return false;
                const tile_moves moves = moves_of(tile, { 1, 1, 0, 0, colors });
                grid drawn(pair.output.width, pair.output.height);
                for (const cell_at& cell : cells)
                    each_tile_offset(tile, moves, cell.row, cell.col, 1, 1, drawn.width, drawn.height, [&](int down, int across) {
                        drawn.data[cell.row + down][cell.col + across] = (int8_t)cell.color;
                        return true;
                    });
                if (drawn != pair.output) return false;
                for (int row = 0; row < pair.output.height; ++row)
                    for (int col = 0; col < pair.output.width; ++col)
                        carried_something = carried_something ||
                            row >= pair.input.height || col >= pair.input.width ||
                            pair.input.data[row][col] != pair.output.data[row][col];
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

// Paint at an absolute grid position. Transparent cells leave the canvas alone.
// Laid under rather than over, as Underlay lays it, a cell only fills what
// nothing has painted yet. Unless clipping is requested, false means some paint
// fell outside; callers must discard that partial drawing.
inline bool paint_piece(grid& canvas, const object& piece, int row0, int col0,
                        bool over = true, bool clip = false) {
    for (int row = 0; row < piece.summary.height; ++row)
        for (int col = 0; col < piece.summary.width; ++col) {
            const int color = piece.data[row][col];
            if (color == TRANSPARENT) continue;
            if (!canvas.contains(row0 + row, col0 + col)) {
                if (clip) continue;
                return false;
            }
            if (over || canvas.data[row0 + row][col0 + col] == TRANSPARENT)
                canvas.data[row0 + row][col0 + col] = (int8_t)color;
        }
    return true;
}

// THE HAND DRAWN AS ITS SENTENCE LAYS IT: every piece, moved by `by(i)`, in the
// order the list holds them, as the ops Union and Underlay draw a list
// (each_picked_cell, src/ops.cpp) -- each over what the pieces before it put down,
// or, laid `under`, only where nothing is yet. Where two pieces meet, Union
// shows the later one and Underlay the earlier, whatever the output wants there
// (833966f4's 9, slid level with the 0 after it in the list, lay under the 0).
// False where a piece falls off the canvas, unless `clip`.
template <class By>
inline bool draw_the_hand(grid& canvas, const vector<object>& hand, bool under, bool clip, By by) {
    for (size_t i = 0; i < hand.size(); ++i) {
        const direction_step step = by(i);
        if (!paint_piece(canvas, hand[i], hand[i].summary.row + step.down, hand[i].summary.col + step.across,
                         !under, clip))
            return false;
    }
    return true;
}

// THE CANVAS A PROGRAM PAINTS ON, sized and closed as the ops do it: the
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
    grid canvas(width, height);
    if (!paint(canvas)) return false;
    const cell_box kept = fit_frame((int)(closes ? fit_to::painted : fit_to::canvas), input.width, input.height, canvas, {});
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
        int8_t ray_kind = Orthogonal;
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
        const color_tally counts(input);
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
            grid object(box.width, box.height);
            for (int i = 0; i < count; ++i) {
                const int row = cell_row(cells[i]), col = cell_col(cells[i]);
                object.data[row - box.row][col - box.col] = color;
                if (how.stamp_body) { ink.data[row][col] = color; drew = true; }
            }
            const auto seed = nearest_ray_seed(box.height, box.width, (box.height - 1) / 2, (box.width - 1) / 2,
                [&](int row, int col) { return object.data[row][col]; });
            each_extended_ray(seed, how.radius, how.ray_kind == Diagonal,
                [&](int down, int across) {
                    const int row = box.row + seed.row + down, col = box.col + seed.col + across;
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
            for (int ray_kind = Orthogonal; ray_kind <= Diagonal; ++ray_kind) {
                if (!takes_value(ray_kind == Orthogonal ? node_type::ExtendOrthogonal
                                                        : node_type::ExtendDiagonal, radius)) continue;
                for (int fill = 0; fill <= 1; ++fill)
                    for (int stamp = 0; stamp <= 1; ++stamp) {
                        candidate trying{ { rank, (int8_t)ray_kind, (int8_t)radius,
                                            fill != 0, stamp != 0 }, {} };
                        for (size_t i = 0; i < train.size(); ++i) {
                            grid ink;
                            if (!draw_layer(train[i].input, ranked[i][rank], trying.how, ink)) break;
                            trying.ink.push_back(std::move(ink));
                        }
                        if (trying.ink.size() != train.size()) continue;
                        const auto same = find_if(options.begin(), options.end(), [&](const candidate& other) {
                            return other.ink == trying.ink;
                        });
                        if (same == options.end()) options.push_back(std::move(trying));
                        // The same drawing with a longer ray: the shorter ran into the
                        // edge, and a larger test grid is where it stops too soon
                        // (ea786f4a), so the longer one takes its place, as on the line.
                        else if (same->how.ray_kind == trying.how.ray_kind &&
                                 same->how.fill_interior == trying.how.fill_interior &&
                                 same->how.stamp_body == trying.how.stamp_body)
                            same->how.radius = trying.how.radius;
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
            covered.back().fill(0);
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
                for (int col = 0; col < width; ++col) out.data[row][col] = (int8_t)color_at(source, row, col);
            return out;
        }
    };
    vector<vector<pane>> pairs;

    bool held = false;
    bool holds() const { return held; }

    // The size every one of `panes` shares, none where they differ or there
    // are none: one pane's box, as Fit stretches over it (fit_to::pane).
    static tile_size shared_size(const vector<pane>& panes) {
        if (panes.empty()) return {};
        for (const pane& one : panes)
            if (one.width != panes[0].width || one.height != panes[0].height) return {};
        return { panes[0].width, panes[0].height };
    }

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
                drawn.data[r][c] = (int8_t)(says.leaves_empty ? TRANSPARENT
                                     : says.paint >= 0   ? says.paint
                                                         : shown(boxes, says.copy_pane, input, r, c, posed));
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
        return same_shape(left.width, left.height, [&](int row, int col) { return left.color_at(one, row, col); },
                          right.width, right.height, [&](int row, int col) { return right.color_at(one, row, col); },
                          DRAWING);
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
            return unique_piece((int)twin_by::drawing, count,
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
                drawn.data[r][c] = (int8_t)(input.data[r][c] == wall ? wall :
                    source.color_at(input, fit_source_cell(r, source.height, input.height),
                                           fit_source_cell(c, source.width, input.width)));
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

// Most objects a list can hold: one per cell of the biggest grid.
constexpr int MAX_OBJECTS_IN_A_LIST = MAX_GRID_CELLS;

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
    // what was painted, as the ops close it. Declines when no pane holds
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
        // The bands come off the frames, the same grouping Layout reads.
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

// The one color every cell off the ground wears, or TRANSPARENT when there is no
// figure or it wears several. What a person means by "the shape's color": read
// off the grid in hand, so a rule that asks for it names a color no example
// showed as readily as one they all did.
inline int8_t the_figure_color(const grid& one_grid, int ground) {
    const color_tally counts(one_grid, ground);
    int8_t figure = TRANSPARENT;
    for (int color = 0; color < COLOR_COUNT; ++color) {
        if (!counts[color]) continue;
        if (figure != TRANSPARENT) return TRANSPARENT;
        figure = (int8_t)color;
    }
    return figure;
}

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
                if (!edge.from_input || edge.scale != 1 || edge.to < 0 || edge.to >= (int)read.output_info.size())
                    continue;
                const auto& landing = read.output_info[edge.to].summary;
                at.poses[landing.row / height][landing.col / width] |= edge.poses;
                recoloring wears;
                pair.input.for_each_colored_cell(TRANSPARENT, [&](int, int, int was) {
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
inline vector<object> pieces_as_said(const grid& input, segmentation_methods segmentation, tile_size tiles = {}) {
    vector<object> pieces;
    if (!boxed_by_its_cells(segmentation)) return pieces;
    object_walk_scratch scratch;
    each_object(input, TRANSPARENT, grouping_of(segmentation), scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
        pieces.push_back(object::from_cells(input, box, cells, count));
    }, tiles);
    if (segmentation == segmentation_methods::all_cells_of_one_color)
        stable_sort(pieces.begin(), pieces.end(), [](const object& one, const object& other) {
            return one.first_color() < other.first_color(); });
    return pieces;
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
                                  const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // The slide happens in each piece's own box, and the cells it leaves are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !pairs_overlay(train)) return {};
        // Draws each output from its input's pieces slid that way, noting whether any
        // cell moved at all.
        auto draws_every_output = [&](bool rows, int step, bool& moved) {
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                const auto pieces = pieces_of(pair.input, correspondence.segmentation, tiles.of(pair));
                drawn = grid(pair.output.width, pair.output.height);
                bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                for (const object& piece : pieces) {
                    bool collision = false;
                    moved |= slam_cells(piece.summary.width, piece.summary.height, rows, step,
                        [&](int row, int col) { return (int)piece.data[row][col]; },
                        [&](int row, int col, int color) {
                            row += piece.summary.row;
                            col += piece.summary.col;
                            // Two pieces landing on one cell would leave paint order to decide.
                            collision = collision || painted[row][col];
                            painted[row][col] = true;
                            drawn.data[row][col] = (int8_t)color;
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
                                       const output_dimension_rule& size, const tiled_pieces& tiles) {
        if (train.empty()) return {};
        bool at_work = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            // The pieces laid into one grid as large as a grid gets, a later one over an earlier.
            int8_t cells[MAX_GRID_DIM][MAX_GRID_DIM];
            for (auto& row : cells)
                for (int8_t& cell : row) cell = (int8_t)TRANSPARENT;
            for (const object& piece : pieces_as_said(pair.input, correspondence.segmentation, tiles.of(pair)))
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
                                      const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // Each piece is hung off the last one's box, so the pieces have to be
        // measured in their own, and the cells no piece reaches are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            train.empty()) return {};
        // OverlapCorner lays out as many objects as one list holds; a chain this
        // detector verified past that point is one the op would not draw.
        constexpr size_t most_the_op_lays_out = MAX_OBJECTS_IN_A_LIST;
        bool laid_any_piece_somewhere_new = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const auto pieces = pieces_of(pair.input, correspondence.segmentation, tiles.of(pair));
            // One piece put in the corner says nothing about chaining.
            if (pieces.size() < 2 || pieces.size() > most_the_op_lays_out) return false;
            // The chain runs left to right, ties top to bottom -- the order the
            // op itself walks the objects in, so the reading and the word agree
            // about which piece is first.
            vector<const object*> chain;
            for (const object& piece : pieces) chain.push_back(&piece);
            stable_sort(chain.begin(), chain.end(), [](const object* one, const object* other) {
                return one->summary.col != other->summary.col
                     ? one->summary.col < other->summary.col
                     : one->summary.row < other->summary.row;
            });
            // Tied corners leave the order to the op's list, which we can't see.
            for (size_t i = 1; i < chain.size(); ++i)
                if (same_corner(chain[i]->summary, chain[i - 1]->summary)) return false;
            drawn = grid(pair.output.width, pair.output.height);
            // The head of the chain sits in the corner; each piece after it puts
            // its top-left on the last one's bottom-right and paints over it,
            // which is the paint order the closing Union gives them for free.
            int x = 0, y = 0;
            for (const object* piece : chain) {
                if (!paint_piece(drawn, *piece, y, x)) return false;
                laid_any_piece_somewhere_new = laid_any_piece_somewhere_new ||
                    x != piece->summary.col || y != piece->summary.row;
                x += piece->summary.width - 1;
                y += piece->summary.height - 1;
            }
            return true;
        });
        // Pieces already sitting in the chain were never laid into one.
        return { redraws && laid_any_piece_somewhere_new };
    }
};

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
// shift_to_touch is shared with the op, so both choose the same landing.
struct slide_to_touch_rule {
    int8_t stays  = -1;   // the colour that is already where it belongs
    int8_t slides = -1;   // the colour that closes the gap until they touch
    bool holds() const { return is_color(stays) && is_color(slides); }

    static slide_to_touch_rule detect(const vector<example>& train,
                                      const correspondence_election& correspondence, const tiled_pieces& tiles) {
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
                                                            correspondence.segmentation, info, nullptr,
                                                                0, 0, tiles.of(pair));
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
                                                                correspondence.segmentation, info, &pieces,
                                                                0, 0, tiles.of(pair));
                drawn = grid(pair.output.width, pair.output.height);
                // WHAT THE OP SEES OF THE CANVAS is the box around everything
                // already painted, read once, before the first piece slides --
                // so what slides is measured against the pieces that stay, and
                // never against another piece that is still on its way.
                cell_box target;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != stays) continue;
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.row, pieces[i].summary.col)) return false;
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
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.row + down,
                                     pieces[i].summary.col + across)) return false;
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

// THE ORDER THE DIRECTIONS ARE ASKED IN, so the order a tie between two that say as
// much is broken in: the four sides before the diagonals, the plainer move
// first, each clockwise from up.
constexpr int direction_asked(int at) { return at < 4 ? 2 * at : 2 * (at - 4) + 1; }

// PIECES THAT SLIDE TO AN EDGE, AND THE REST LEFT WHERE IT LIES. Every piece,
// or one color's pieces, go one of the eight ways until they meet the edge of
// the canvas, passing over whatever is in the way, and land over or under what
// stays. That is MoveToEdge on a pick, so the replay steps each piece as the
// op does (steps_to_edge), and how far is the canvas's to say -- it differs
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
    int8_t slide_direction = direction::up;
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
                if (!slides(piece) && !paint_piece(canvas, piece, piece.summary.row, piece.summary.col))
                    return false;
            const direction_step step = step_of(slide_direction);
            for (const object& piece : pieces) {
                if (!slides(piece)) continue;
                const int steps = steps_to_edge(piece.summary.box(), canvas.width, canvas.height, slide_direction);
                if (!paint_piece(canvas, piece, piece.summary.row + steps * step.down,
                                 piece.summary.col + steps * step.across, over)) return false;
            }
            return true;
        });
    }

    static edge_slide_rule detect(const vector<example>& train, segmentation_methods elected,
                                  const output_dimension_rule& size, const tiled_pieces& tiles) {
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
                pieces.push_back(pieces_as_said(pair.input, cut, tiles.of(pair)));
                for (const object& piece : pieces.back()) colors = (uint16_t)(colors | palette(piece));
            }
            bool as_they_lie = true;
            for (size_t i = 0; i < train.size() && as_they_lie; ++i) {
                grid drawn;
                as_they_lie = draw_on_canvas(train[i].input, size, drawn, [&](grid& canvas) {
                    for (const object& piece : pieces[i])
                        if (!paint_piece(canvas, piece, piece.summary.row, piece.summary.col)) return false;
                    return true;
                }) && drawn == train[i].output;
            }
            if (as_they_lie) continue;
            for (int color = -1; color < COLOR_COUNT; ++color) {
                if (color >= 0 && !(colors >> color & 1)) continue;
                for (int at = 0; at < direction::count; ++at)
                    for (bool over : { true, false }) {
                        // With every piece sliding nothing stays to lie under.
                        if (!over && color < 0) continue;
                        const edge_slide_rule trying{ cut, (int8_t)color, (int8_t)direction_asked(at), over, true };
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

    static vector<object> pieces(const grid& input, const pane_split_facts& split, bool use_panes, tile_size tiles = {}) {
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
        return pieces_as_said(source, segmentation_methods::connected_multicolor, tiles);
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
            const int row = aligned_copy_start(pattern.summary.height, target.row, target.height, anchor / 3);
            const int col = aligned_copy_start(pattern.summary.width, target.col, target.width, anchor % 3);
            paint_piece(drawn, pattern, row, col, /*over=*/true, /*clip=*/true);
        }
        return drawn;
    }

    static copy_to_rule detect(const vector<example>& train,
                               const pane_split_facts& split, const tiled_pieces& tiles) {
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
                cut.push_back(pieces(pair.input, split, use_panes, tiles.of(pair)));
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

inline vector<marker_piece> seeds_the_op_will_see(const grid& input, tile_size tiles = {}) {
    vector<marker_piece> pieces;
    object_walk_scratch scratch;
    each_object(input, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int found) {
        if (found <= 0) return;
        pieces.push_back({ box, (int8_t)input.data[cell_row(cells[0])][cell_col(cells[0])] });
    }, tiles);
    vector<object_summary> boxes;
    for (const marker_piece& piece : pieces) boxes.push_back(piece.summary);
    vector<marker_piece> seeds;
    for (int which : read_seeds(boxes).seeds) seeds.push_back(pieces[which]);
    return seeds;
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
    static int stamp(const example& pair, segmentation_methods cut, Land land, tile_size tiles) {
        const auto pieces = pieces_of(pair.input, cut, tiles);
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
        for (const marker_piece& marker : seeds_the_op_will_see(pair.input, tiles))
            project_along(tpl.summary.box(), marker.summary.box(),
                          pair.input.width, pair.input.height, [&](int at_row, int at_col) {
                if (stamped >= most_the_op_stamps) return false;
                // A stamp hanging over the edge is drawn as far as it fits.
                for (int r = 0; r < tpl.summary.height; ++r)
                    for (int c = 0; c < tpl.summary.width; ++c) {
                        const int row = at_row + r, col = at_col + c;
                        if (tpl.data[r][c] != TRANSPARENT && pair.input.contains(row, col))
                            land(row, col, marker.color);
                    }
                ++stamped;
                return true;
            });
        return stamped;
    }

    static project_on_seeds_rule detect(const vector<example>& train,
                                        const correspondence_election& correspondence,
                                        const tiled_pieces& tiles) {
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
            }, tiles.of(pair)) > 0;
        });
        if (in_the_markers_colors) return { true, TRANSPARENT };

        // With the input laid back over the stamps, its ground shows wherever no
        // stamp went, and the outputs' elected ground is what fills it -- or,
        // where a plan deleted the ground, nothing does until the plan's fill.
        int8_t worn = TRANSPARENT;
        bool one_color = true;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = grid(pair.input.width, pair.input.height);
            pair.input.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
                drawn.data[row][col] = (int8_t)color;
            });
            const int stamped = stamp(pair, cut, [&](int row, int col, int8_t) {
                if (pair.input.data[row][col] != TRANSPARENT) return;
                const int8_t shown = (int8_t)pair.output.data[row][col];
                if (!is_color(worn)) worn = shown;
                one_color = one_color && shown == worn;
                drawn.data[row][col] = worn;
            }, tiles.of(pair));
            return stamped > 0 && one_color;
        });
        // A stamp in the ground's own colour draws nothing at all.
        if (!redraws || !is_color(worn) || worn == TRANSPARENT) return {};
        return { true, worn };
    }
};

// THE WAYS A BRIDGE IS ASKED IN, the narrowest first -- one direction before
// both; only across the grid, then the nearest, then every one -- so the bars
// are said no wider than the pairs show them: a699fb00 bridges along rows, and
// nothing in it says a column would be bridged too; 0d87d2a6's marks face each
// other from opposite edges, and nothing in its pairs says two marks on one
// edge would be joined, as the nearest joins them on its test.
constexpr int bridge_ways_narrowest_first[BRIDGE_WAYS] = { 7, 8, 6, 1, 2, 0, 4, 5, 3 };
static_assert(row_of(node_type::Bridge).high + 1 == BRIDGE_VALUES, "Bridge takes one value for each way it reaches");

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
                              const correspondence_election& correspondence, const tiled_pieces& tiles) {
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
                    const vector<object> pieces = pieces_as_said(pair.input, cut, tiles.of(pair));
                    // One list is all the op is handed, and the bars take one place
                    // more in it, all of them one drawing.
                    if (pieces.empty() || pieces.size() >= MAX_OBJECTS_IN_A_LIST) return false;
                    drawn = grid(pair.output.width, pair.output.height);
                    bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                    for (const object& piece : pieces)
                        piece.each_cell([&](int row, int col, int8_t color) {
                            drawn.data[row][col] = color;
                            painted[row][col] = true;
                        });
                    // Each bar is laid under whatever is already drawn. Every
                    // piece is picked, as SelectAll picks them before Bridge.
                    hand_answers fresh;
                    bridge_bars(way, pair.output.width, pair.output.height,
                                partner_pieces(pieces, vector<bool>(pieces.size(), true), fresh),
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

// THE MARKS AS LINES, REPEATED DOWN THE GRID. Every mark becomes a line
// across the grid, and the lines together come round again, the same gap
// after them as between them, to the far edge. That is two Tiles: `lines`,
// which lays a mark's first column across the grid, then, the lines made one
// drawing, `lattice`, which lays it down by its own height and its gap; with
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

    // The Tiles that say it: each mark's first column laid across the grid
    // both ways, a step at a time; then those lines, made one drawing
    // (UnionList), laid down from where it stands onward, by its own height and
    // the gap between its lines -- the whole thing again, its gap after it.
    static constexpr tile_value lines{ .steps = { { .down = { 0 }, .across = { 1 } } },
                                       .start = { .edge = true }, .end = { .edge = true }, .spread = tile_spread::lines };
    static constexpr tile_value lattice{
        .steps = { { .down = measured(tile_measure::own_height, tile_measure::row_gap), .across = { 0 } } },
        .start = {}, .end = { .edge = true }, .spread = tile_spread::lines };

    // What the words draw on one input: the cut's marks, trimmed and read the
    // way the pair is turned, each laid as its line, the lines laid again
    // wherever the lattice puts them, the rest left empty. False where the
    // lines would not fit the one list the op is handed.
    static bool draw(const grid& input, segmentation_methods cut, bool portrait, grid& drawn, tile_size tiles = {}) {
        const vector<object> marks = pieces_as_said(input, cut, tiles);
        // Read turned, the marks' rows are their columns, and the other way round.
        const bool turned = portrait && lies_on_its_side(input.width, input.height);
        auto top = [&](const object& mark) { return (int)(turned ? mark.summary.col : mark.summary.row); };
        auto left = [&](const object& mark) { return (int)(turned ? mark.summary.row : mark.summary.col); };
        auto tall = [&](const object& mark) { return (int)(turned ? mark.summary.width : mark.summary.height); };
        auto wide = [&](const object& mark) { return (int)(turned ? mark.summary.height : mark.summary.width); };
        const int rows = turned ? input.width : input.height;
        const int across = turned ? input.height : input.width;
        // The lines, as the one drawing UnionList makes of them, and the rows it spans.
        int8_t picture[MAX_GRID_DIM][MAX_GRID_DIM];
        for (auto& row : picture) std::fill(std::begin(row), std::end(row), (int8_t)TRANSPARENT);
        int first_row = rows, last_row = -1, laid_lines = 0;
        for (const object& mark : marks) {
            const tile_moves spread = moves_of(lines, {});
            const auto [line_rows, line_columns] = copy_size(spread, tall(mark), wide(mark));
            each_tile_offset(lines, spread, top(mark), left(mark), line_rows, line_columns, across, rows,
                             [&](int, int sideways) {
                ++laid_lines;
                for (int row = 0; row < line_rows; ++row)
                    for (int col = 0; col < line_columns; ++col) {
                        const int8_t color = turned ? mark.data[col][row] : mark.data[row][col];
                        const int at_row = top(mark) + row, at_col = left(mark) + sideways + col;
                        if (color != TRANSPARENT && at_row >= 0 && at_row < rows && at_col >= 0 && at_col < across)
                            picture[at_row][at_col] = color;
                    }
                return true;
            });
            first_row = min(first_row, top(mark));
            last_row = max(last_row, top(mark) + tall(mark) - 1);
        }
        if (last_row < first_row || laid_lines > MAX_OBJECTS_IN_A_LIST) return false;
        const int height = last_row - first_row + 1;
        const tile_moves down = moves_of(lattice, measures_of(lattice, height, across, [&](int row, int col) {
            return picture[first_row + row][col] != TRANSPARENT; }, 0));
        const auto [copy_rows, copy_columns] = copy_size(down, height, across);
        drawn = grid(input.width, input.height);
        auto at = [&](int row, int col) -> int8_t& {
            return turned ? drawn.data[col][row] : drawn.data[row][col];
        };
        each_tile_offset(lattice, down, first_row, 0, copy_rows, copy_columns, across, rows, [&](int lower, int) {
            for (int row = 0; row < copy_rows && first_row + lower + row < rows; ++row)
                for (int col = 0; col < copy_columns; ++col)
                    if (const int8_t color = picture[first_row + row][col]; color != TRANSPARENT)
                        at(first_row + lower + row, col) = color;
            return true;
        });
        return true;
    }

    static mark_lattice_rule detect(const vector<example>& train,
                                    const correspondence_election& correspondence, const tiled_pieces& tiles) {
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
                    if (!draw(pair.input, cut, portrait, drawn, tiles.of(pair)))
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

    static interior_spread_rule detect(const vector<example>& train, const tiled_pieces& tiles) {
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
                    const int8_t color = object_paint_color(box.height, box.width, COLOR_COUNT,
                        [&](int row, int col) { return (int8_t)piece.data[row][col]; });
                    piece.each_cell([&](int row, int col, int8_t was) {
                        if (on_object_edge(row, col, [&](int r, int c) { return piece.holds(r, c); })) return;
                        drawn.data[row][col] = color;
                        changed = changed || color != was;
                    });
                }, tiles.of(pair));
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

    static interior_fill_rule detect(const vector<example>& train, const tiled_pieces& tiles) {
        if (train.empty() || !pairs_overlay(train)) return {};
        interior_fill_rule rule;
        bool changed = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = grid(pair.input.width, pair.input.height);
            object_walk_scratch scratch;
            bool consistent = true;
            int pieces = 0;
            each_object(pair.input, TRANSPARENT, grouping::touching_any_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    if (!consistent) return;
                    ++pieces;
                    const object piece = object::from_cells(pair.input, box, cells, count);
                    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
                    mark_outside(box.height, box.width,
                        [&](int r, int c) { return is_color(piece.data[r][c]); }, outside);
                    // Separate marker objects inside a hole keep their own
                    // colors. They are not evidence for the interior's fill.
                    for (int r = 0; r < box.height; ++r)
                        for (int c = 0; c < box.width; ++c) {
                            const int y = box.row + r, x = box.col + c;
                            if (is_color(piece.data[r][c]) || outside[r][c] ||
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
                    for (int r = 0; r < box.height; ++r)
                        for (int c = 0; c < box.width; ++c) {
                            const int y = box.row + r, x = box.col + c;
                            if (is_color(piece.data[r][c]))
                                drawn.data[y][x] = piece.data[r][c];
                            else if (!outside[r][c] && rule.holds())
                                drawn.data[y][x] = rule.color;
                        }
                }, tiles.of(pair));
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

        const color_tally frequencies(pair.input);
        bool removed[COLOR_COUNT] = {};
        for (int rank = 0; rank < discarded_colors; ++rank) {
            int common = -1;
            for (int color = 0; color < COLOR_COUNT; ++color) {
                if (removed[color] || frequencies[color] == 0) continue;
                if (common < 0 || frequencies[color] > frequencies[common]) common = color;
            }
            if (common >= 0) removed[common] = true;
        }

        grid remaining(pair.input.width, pair.input.height);
        for (int row = 0; row < pair.input.height; ++row)
            for (int col = 0; col < pair.input.width; ++col) {
                const int color = pair.input.data[row][col];
                if (is_color(color) && !removed[color])
                    remaining.data[row][col] = (int8_t)color;
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
        for (int i = 0; i < (int)pieces.size(); ++i) {
            const object& piece = pieces[i];
            const int top = row_of[i] * cell_size;
            const int left = col_of[i] * cell_size;
            for (int row = 0; row < piece.summary.height && row < cell_size; ++row)
                for (int col = 0; col < piece.summary.width && col < cell_size; ++col) {
                    const int destination_row = top + row, destination_col = left + col;
                    if (destination_row < height && destination_col < width &&
                        piece.data[row][col] != TRANSPARENT)
                        arranged.data[destination_row][destination_col] = piece.data[row][col];
                }
        }

        const int reduced_width = max((width + cell_size - 1) / cell_size, 1);
        const int reduced_height = max((height + cell_size - 1) / cell_size, 1);
        grid painted(MAX_GRID_DIM, MAX_GRID_DIM);
        for (int row = 0; row < reduced_height; ++row)
            for (int col = 0; col < reduced_width; ++col)
                painted.data[row][col] = majority_vote(arranged.data, row, col, cell_size, height, width);

        cell_box bounds;
        for (int row = 0; row < reduced_height; ++row)
            for (int col = 0; col < reduced_width; ++col)
                if (painted.data[row][col] != TRANSPARENT) bounds.include_cell(row, col);
        if (bounds.empty()) return false;

        drawn = grid(bounds.width(), bounds.height());
        for (int row = 0; row < drawn.height; ++row)
            for (int col = 0; col < drawn.width; ++col) {
                const int color = painted.data[bounds.row0 + row][bounds.col0 + col];
                drawn.data[row][col] = (int8_t)color;
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
        const color_tally counts(input);
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
            const int top = down ? steps_to_edge(whole, canvas.width, canvas.height, direction::down) : 0;
            const int left = right ? steps_to_edge(whole, canvas.width, canvas.height, direction::right) : 0;
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
    // No cut means Input's whole box, including its empty margins.
    vector<object> pieces(const grid& input, tile_size tiles = {}) const {
        if (cut != segmentation_methods::unset) return pieces_as_said(input, cut, tiles);
        object whole;
        whole.summary = { 0, 0, (int8_t)input.width, (int8_t)input.height, 0 };
        for (int row = 0; row < input.height; ++row)
            for (int col = 0; col < input.width; ++col) {
                whole.data[row][col] = (int8_t)input.data[row][col];
                whole.summary.count += input.data[row][col] != TRANSPARENT;
            }
        return { whole };
    }
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
    // WHICH PASS THIS READING IS (read_the_rest): the first reads the input,
    // and each pass after it the picture the passes before it drew.
    int8_t pass = 1;
    // WHETHER ITS WORDS MAY LAY CELLS WRONG, each where it brings every pair
    // nearer its output, for the passes after it to set right (brings_nearer):
    // a choice infer makes, every reading that lays none read first. Its
    // passes and its framed reading read the same way.
    bool tolerant = false;
    // WHETHER ITS WORDS AND PAINTS ARE ASKED WITH THE PARTNER PICKS too
    // (partner_picks): a last resort infer takes only where no reading
    // without them says every cell. Its passes and its framed reading read
    // the same way.
    bool partners = false;
    // WHAT THE LINE DRAWS, pair by pair, before the plan fills the ground: the
    // hand as its words leave it and every cell a paint drew, in the color the
    // output shows there (finish_the_line). None where the line cannot know
    // it: a hand it could not carry, or pairs that do not lie over each other.
    vector<grid> picture;
    // THE REST OF THE CHANGE, READ ON THAT PICTURE: the next pass, a puzzle of
    // its own from each picture to its output (read_the_rest), and whether it
    // leaves fewer cells unsaid -- else it is only the readout's, which names
    // the words the rest lacks.
    std::shared_ptr<const inference> rest;
    bool rest_says_more = false;
    // Whether the cut's pieces, painted where they lie on the canvas the size
    // rule gives, draw every output: what a line with no words claims.
    bool pieces_redraw = true;
    // THE GROUND, OWNED ONCE (infer): what the program
    // deletes before anything else and fills after everything else. Everything
    // above was read from the figures that plan leaves, with no ground at all;
    // `ground_election` is the election the plan came from, for the readout.
    ground_plan plan;
    background_election ground_election;
    // EACH PLAN READ, in the order infer reads them (says_the_most): which plan,
    // with words that may lay cells wrong or not, the cells it left unsaid, the
    // passes it took, and the words its line said -- the readout's [plans
    // read], which shows why one plan was kept over another (a61f2674's lost
    // line was found this way).
    struct plan_read { int plan = 0; bool tolerant = false; int left = 0, passes = 0; string words; bool partners = false; };
    vector<plan_read> plans_read;
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
    // What piece i counts, as count_of or twin_key reads it, read once a piece.
    auto keys_of = [&](auto key) {
        vector<int> keys;
        for (const object& piece : pieces)
            keys.push_back(key(piece.summary.height, piece.summary.width,
                               [&](int row, int col) { return (int)piece.data[row][col]; }));
        return keys;
    };
    if (how == piece_pick::only_one) {
        // The one piece with no twin, as `value` reads twins (unique_piece, the
        // count SelectUnique makes): by a key read once a piece where twins
        // share one (twin_key), else shape against shape.
        int found = -1;
        if (value >= (int)twin_by::colors) {
            const vector<int> keys = keys_of([&](int height, int width, auto cell) {
                return twin_key(value, height, width, cell); });
            found = lone_one((int)keys.size(), [&](int i, int j) { return keys[(size_t)i] == keys[(size_t)j]; });
        } else
            found = unique_piece(value, (int)pieces.size(),
                [&](int i) { return (int)pieces[(size_t)i].summary.width; },
                [&](int i) { return (int)pieces[(size_t)i].summary.height; },
                [&](int i, int row, int col) { return (int)pieces[(size_t)i].data[row][col]; });
        if (found >= 0) picked[(size_t)found] = true;
        return picked;
    }
    if ((how == piece_pick::largest || how == piece_pick::smallest) && value >= SIZE_RANKS) {
        // Every piece with the most -- or the fewest -- of a count (count_of),
        // ties and all, as the op picks them past its ranks.
        const vector<int> counts = keys_of([&](int height, int width, auto cell) {
            return count_of(value - SIZE_RANKS, height, width, cell); });
        each_extreme((int)counts.size(), how == piece_pick::largest, [&](int i) { return counts[(size_t)i]; },
                     [&](int i) { picked[(size_t)i] = true; });
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
            picked[i] = paints_at_least(piece.summary.height, piece.summary.width, [&](int row, int col) {
                return piece.data[row][col] != (int8_t)TRANSPARENT; }, value);
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
            picked[i] = object_hole_count(piece.summary.height, piece.summary.width, [&](int row, int col) {
                return piece.data[row][col] != (int8_t)TRANSPARENT; }) == value;
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
static_assert(row_of(node_type::SelectLikeKey).high + 1 == KEY_LIKENESSES,
              "SelectLikeKey takes one value for each likeness to the key");
static_assert(row_of(node_type::SelectPartner).high + 1 == SELECT_PARTNER_VALUES,
              "SelectPartner takes each partner way on any side, then on each of the eight directions");

// THE PIECES A PICK TAKES, as its words take them on a canvas the size of
// `canvas`, in the order pick_words says them: its colors or its kind; the
// pieces in the part those fence off, where it names a part; their partners,
// where it names a partner; every piece but those, where it turns them round;
// then WouldFit's and SelectLikeKey's keeps.
inline vector<bool> picked_pieces(const vector<object>& pieces, const piece_pick& pick, tile_size canvas,
                                  hand_answers& answers) {
    vector<bool> picked(pieces.size(), false);
    if (pick.how >= 0) picked = picked_pieces(pieces, pick.how, pick.value);
    else
        for (size_t i = 0; i < pieces.size(); ++i) picked[i] = (palette(pieces[i]) & pick.colors) != 0;
    if (pick.part >= 0) {
        // SelectPartition: the pick as it stands is the walls, and the pieces
        // lying in the part `part` they fence off are picked instead.
        const partner_pieces walls(pieces, picked, answers);
        int16_t part[MAX_GRID_DIM][MAX_GRID_DIM];
        parts_the_pick_fences(walls, canvas.width, canvas.height, part);
        for (size_t i = 0; i < pieces.size(); ++i) picked[i] = lies_in_part(walls, (int)i, pick.part, part);
    }
    if (pick.partner >= 0) {
        // SelectPartner: the pick moves onto every partner the pieces it took
        // have, among every piece, by the way and on the side it names.
        const partner_pieces taken(pieces, picked, answers);
        vector<bool> partners(pieces.size(), false);
        each_partner_picked(pick.partner, taken, [&](int other) { partners[(size_t)other] = true; });
        picked = std::move(partners);
    }
    // Every piece but those, as InvertSelection turns them round.
    if (pick.but_those) picked.flip();
    if (pick.fits >= 0) {
        // WouldFit keeps those that would fit into a gap of their partner among
        // the pieces not picked, every partner read off the pick as it stood, as
        // the op reads its marks before it drops any.
        const partner_pieces places(pieces, picked, answers);
        for (size_t i = 0; i < pieces.size(); ++i)
            if (places.picked[i]) picked[i] = places.fits((partner_by)pick.fits, (int)i);
    }
    if (pick.like >= 0) {
        // SelectLikeKey keeps those like a piece of the key, the pieces not
        // picked, read off the pick as it stood.
        const partner_pieces key(pieces, picked, answers);
        for (size_t i = 0; i < pieces.size(); ++i)
            if (key.picked[i]) picked[i] = like_the_key(pick.like, (int)i, key);
    }
    return picked;
}

inline partner_pieces::partner_pieces(const vector<object>& cut, const piece_pick& pick, tile_size canvas,
                                      hand_answers& kept)
    : partner_pieces(cut, picked_pieces(cut, pick, canvas, kept), kept) {}

static_assert(row_of(node_type::SelectLargestObject).low == row_of(node_type::SelectSmallestObject).low &&
              row_of(node_type::SelectLargestObject).high == row_of(node_type::SelectSmallestObject).high,
              "the largest and the smallest are asked as far down the order");

// THE CELLS OF EVERY PAIR, a list each, (row, col): what a replay sees.
using cells_by_pair = vector<vector<std::pair<int, int>>>;

// A cell laid down, in the color laid there; and every pair's, a list each.
struct cell_color { int8_t row, col, color; };
using colors_by_pair = vector<vector<cell_color>>;

// Whether `cells` holds the cell at (row, col) of pair `which`.
inline bool on_cells(const cells_by_pair& cells, size_t which, int row, int col) {
    return which < cells.size() &&
           ranges::find(cells[which], std::pair<int, int>{ row, col }) != cells[which].end();
}

// THE CELLS OF EVERY PAIR, MARKED where they lie: what on_cells answers, at a
// look rather than a search, for a claim that asks it of every cell left over.
// A cell past the edge of the largest grid -- a piece the words moved off it --
// is no cell left over, and is not marked.
struct cell_marks {
    vector<std::bitset<MAX_GRID_CELLS>> at;
    cell_marks() = default;
    explicit cell_marks(const cells_by_pair& cells) : at(cells.size()) {
        for (size_t which = 0; which < cells.size(); ++which)
            for (const auto& [row, col] : cells[which])
                if (row >= 0 && col >= 0 && row < MAX_GRID_DIM && col < MAX_GRID_DIM)
                    at[which].set((size_t)cell_index(row, col));
    }
};
inline bool on_cells(const cell_marks& cells, size_t which, int row, int col) {
    return which < cells.at.size() && cells.at[which].test((size_t)cell_index(row, col));
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
//
// BUT A FINISHED PAIR, whose grid read already shows its output, has nothing
// left for any word to do, so it asks only that a word lays nothing wrong
// there (the user): a pass after the first finds some pairs done and others
// not (1190bc91's pair 2, whole once its rays are drawn, while each marker of
// the other pairs still fills its part).
struct replay_facts {
    const vector<example>& train;
    // Whether each pair's grid read is already its output, and how many pairs'
    // are not (finished).
    vector<bool> done;
    int open = 0;
    bool overlays = true;
    // Whether `cuts` is still the hand the line leaves: false once a word on it
    // is one carry_the_hand cannot carry the pieces through.
    bool hand_known = true;
    // Whether the line lays its hand under what is down, as Underlay does, where
    // some piece sits on another -- else over it, as Union does (draw_the_hand).
    bool laid_under = false;
    // Whether a word may lay cells wrong where it brings every pair nearer
    // (brings_nearer): only on a line that draws its picture, so the next pass
    // can read the cells it left wrong. Off everywhere else.
    bool tolerant = false;
    // The pieces a named cut hands over, rather than the elected one.
    replay_facts(const vector<example>& pairs, segmentation_methods cut,
        const tiled_pieces& tiles) : train(pairs) {
        for (const example& pair : train) {
            hands.push_back(pieces_as_said(pair.input, cut, tiles.of(pair)));
            done.push_back(pair.input == pair.output);
            open += !done.back();
            overlays = overlays && pair.input.same_size(pair.output);
        }
        answers.resize(hands.size());
    }
    replay_facts(const inference& inf, const vector<example>& pairs, const tiled_pieces& tiles)
        : replay_facts(pairs, inf.correspondence.segmentation, tiles) {}
    // The pieces a named cut hands over of `grids`, one a pair, rather than of
    // the inputs: the picture the line has drawn so far, cut again.
    replay_facts(const vector<example>& pairs, const vector<grid>& grids, segmentation_methods cut,
        const tiled_pieces& tiles) : train(pairs) {
        for (size_t which = 0; which < train.size() && which < grids.size(); ++which) {
            hands.push_back(pieces_as_said(grids[which], cut, tiles.of(train[which])));
            done.push_back(grids[which] == train[which].output);
            open += !done.back();
            overlays = overlays && train[which].input.same_size(train[which].output);
        }
        answers.resize(hands.size());
    }
    // The canvas pair `which`'s pieces lie on: its input's, as the ops take it.
    tile_size canvas(size_t which) const { return { train[which].input.width, train[which].input.height }; }
    // Whether pair `which` is finished: its grid read already its output,
    // while some pair's is not. Where every pair's is, no word has anything to
    // do anywhere, and none is let off (cd3c21df's framed pieces, each already
    // its output: a mirror that drew them again was said).
    bool finished(size_t which) const { return open > 0 && which < done.size() && done[which]; }
    bool any_finished() const { return open > 0 && ranges::find(done, true) != done.end(); }
    // WHETHER A GRID HOLDS ONE SHAPE AT TWO SIZES: two pieces of one pair's
    // hand whose boxes differ while their shape at any size -- scaled
    // [0], or stretched [1] -- is one (key_of_shape). Only there does a
    // likeness at any size, or stretched, say what the likeness at its own
    // size does not: elsewhere the two find the same partners, and the one at
    // its own size is asked first. Read once, and again once the hand changes.
    bool shape_at_two_sizes(bool stretched) const {
        if (sizes_seen < 0) {
            sizes_seen = 0;
            for (const vector<object>& cut : hands)
                for (const shape_freedom free : { shape_freedom{ true, true }, shape_freedom{ true, false, true } }) {
                    vector<uint32_t> keys;
                    for (const object& piece : cut)
                        keys.push_back(key_of_shape(piece.summary.width, piece.summary.height,
                                                    [&](int row, int col) { return (int)piece.data[row][col]; }, free));
                    for (size_t i = 0; i < cut.size(); ++i)
                        for (size_t j = i + 1; j < cut.size(); ++j)
                            if (keys[i] == keys[j] && (cut[i].summary.width != cut[j].summary.width ||
                                                       cut[i].summary.height != cut[j].summary.height))
                                sizes_seen |= (int8_t)(free.stretch ? 2 : 1);
                }
        }
        return (sizes_seen >> (stretched ? 1 : 0) & 1) != 0;
    }
    // Whether a likeness, a partner way, or a partner of any kind
    // (partner_of_kind) is worth asking of this hand (shape_at_two_sizes).
    bool asks_likeness(int like) const { return !sized_likeness(like) || shape_at_two_sizes(like / 2 == 2); }
    bool asks_way(partner_by how) const { return likeness_of(how) < 0 || asks_likeness(likeness_of(how)); }
    bool asks_kind(int kind) const { return kind >= (int)partner_by::count || asks_way((partner_by)kind); }
    mutable int8_t sizes_seen = -1;
    // A hand changes only through change(), so nothing read off it outlives it.
    const vector<vector<object>>& cuts() const { return hands; }
    hand_answers& answers_of(size_t which) const { return answers[which]; }
    vector<bool> picked(size_t which, const piece_pick& pick) const {
        return picked_pieces(hands[which], pick, canvas(which), answers[which]);
    }
    partner_pieces handed(size_t which, const piece_pick& pick) const {
        return { hands[which], pick, canvas(which), answers[which] };
    }
    // Pair `which`'s hand, open to change; what was read off it is forgotten after.
    class hand_change {
    public:
        vector<object>& hand;
        hand_change(replay_facts& facts, size_t which) : hand(facts.hands[which]), facts(facts), which(which) {}
        hand_change(const hand_change&) = delete;
        hand_change& operator=(const hand_change&) = delete;
        ~hand_change() { facts.forget(which); }
    private:
        replay_facts& facts;
        size_t which;
    };
    hand_change change(size_t which) { return { *this, which }; }
private:
    vector<vector<object>> hands;
    mutable vector<hand_answers> answers;
    void forget(size_t which) {
        answers[which].forget();
        sizes_seen = -1;
    }
};

// THE PIECES A PICK TAKES, pair by pair, made ready once for a word asked many
// ways over -- every length, both ways round, every lay -- whose tries differ in
// everything but the pieces. Kept only while that word is asked, and only while
// nothing carries the hand: `facts` must not change under it.
inline vector<partner_pieces> handed_pieces(const replay_facts& facts, const piece_pick& pick) {
    vector<partner_pieces> handed;
    handed.reserve(facts.cuts().size());
    for (size_t which = 0; which < facts.cuts().size(); ++which) handed.push_back(facts.handed(which, pick));
    return handed;
}

// Whether two picks take the same pieces in every pair: each replay is handed
// the same pieces by both, and sees the same.
inline bool takes_the_same(const vector<partner_pieces>& one, const vector<partner_pieces>& other) {
    return ranges::equal(one, other, [](const partner_pieces& a, const partner_pieces& b) { return a.picked == b.picked; });
}

// Local to one word on one unchanged hand. Record only eligible picks, so the
// first pick allowed to ask a value keeps its tie, even if another was skipped.
struct asked_piece_sets {
    vector<vector<partner_pieces>> asked;
    bool before(const vector<partner_pieces>& handed) {
        if (ranges::any_of(asked, [&](const auto& prior) { return takes_the_same(prior, handed); })) return true;
        asked.push_back(handed);
        return false;
    }
};

// Whether a pick by `colors` picks by the same colors in every pair: each pair's
// pieces hold the same ones of them. A pick whose colors change from pair to pair
// names each pair's own colors rather than a rule, and a test brings its own.
inline bool same_colors_every_pair(const replay_facts& facts, uint16_t colors) {
    bool first = true;
    uint16_t seen = 0;
    for (const auto& cut : facts.cuts()) {
        uint16_t held = 0;
        for (const object& piece : cut) held = (uint16_t)(held | (palette(piece) & colors));
        if (first) seen = held;
        else if (held != seen) return false;
        first = false;
    }
    return true;
}

// THE PARTNERS OF A PICK (SelectPartner): of every piece, and of each color
// every pair holds, every partner each has by a way partner_by finds one -- on
// any side, then wholly on each side and each corner in turn (direction_asked) --
// and every piece but those, each asked only where it takes a piece in every
// pair and leaves one in some, and takes other pieces somewhere than its base
// and every partner pick asked before it (4364c1c4: of every piece, the
// partners above, the upper piece of each stack). Each way's partners are found
// once a pair (each_partner, among every piece, as each_partner_picked finds
// them), for every base and side.
// A LAST RESORT: a line asks them only once nothing else it asks says any more
// (finish_the_line), so a line the other picks finish reads as it did -- asked
// with the rest, they said more at once in lines the others finish, and steered
// them wrong (22168020, 272f95fa, 62b74c02, 9344f635, a61f2674, f823c43c).
inline vector<piece_pick> partner_picks(const replay_facts& facts) {
    vector<piece_pick> picks;
    // Each partner found with the sides of its piece it lies on, a bit for each direction.
    struct found_partner { int mine, other; uint8_t partner_directions; };
    vector<vector<vector<found_partner>>> partners_by_way((size_t)partner_by::count);
    for (int way = 0; way < (int)partner_by::count; ++way) {
        if (!facts.asks_way((partner_by)way)) continue;
        for (size_t which = 0; which < facts.cuts().size(); ++which) {
            const vector<object>& cut = facts.cuts()[which];
            const partner_pieces every(cut, vector<bool>(cut.size(), true), facts.answers_of(which));
            vector<found_partner>& found = partners_by_way[(size_t)way].emplace_back();
            for (int mine = 0; mine < every.count(); ++mine)
                each_partner((partner_by)way, mine, every, [&](int other) {
                    uint8_t partner_directions = 0;
                    for (int partner_direction = 0; partner_direction < direction::count; ++partner_direction)
                        if (lies_toward(every.box(mine), every.box(other), partner_direction)) partner_directions = (uint8_t)(partner_directions | 1u << partner_direction);
                    found.push_back({ mine, other, partner_directions });
                }, true);
        }
    }
    vector<uint16_t> bases = { piece_pick::all().colors };
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (ranges::all_of(facts.cuts(), [&](const vector<object>& cut) {
                return ranges::any_of(cut, [&](const object& piece) { return (palette(piece) >> color & 1) != 0; });
            }))
            bases.push_back(piece_pick::of_color(color).colors);
    // The ways asked touch first, at a side and then at a corner, and then in
    // partner_by's order: a piece it touches is its partner whatever lies in
    // line with it, which a move can change (4364c1c4's test: once each upper
    // piece stepped left, the lower stack's top piece came into the columns of
    // the upper stack's bottom one, which "a piece in its columns" then took for
    // its partner too, and it did not step right).
    vector<int> ways_asked = { (int)partner_by::touching_sides, (int)partner_by::touching };
    for (int way = 0; way < (int)partner_by::count; ++way)
        if (ranges::find(ways_asked, way) == ways_asked.end()) ways_asked.push_back(way);
    vector<vector<vector<bool>>> partner_picks_taken;
    for (const uint16_t base : bases) {
        vector<vector<bool>> base_taken;
        for (size_t which = 0; which < facts.cuts().size(); ++which)
            base_taken.push_back(facts.picked(which, piece_pick{ base }));
        for (const int way : ways_asked) {
            if (partners_by_way[(size_t)way].empty()) continue;
            for (int at = -1; at < direction::count; ++at) {
                const int partner_direction = at < 0 ? -1 : direction_asked(at);
                vector<vector<bool>> taken;
                bool every = true, splits = false;
                for (size_t which = 0; which < facts.cuts().size(); ++which) {
                    vector<bool>& one = taken.emplace_back(facts.cuts()[which].size(), false);
                    for (const found_partner& found : partners_by_way[(size_t)way][which])
                        if (base_taken[which][(size_t)found.mine] && (partner_direction < 0 || (found.partner_directions >> partner_direction & 1u)))
                            one[(size_t)found.other] = true;
                    const bool some = ranges::find(one, true) != one.end();
                    every = every && some;
                    splits = splits || (some && ranges::find(one, false) != one.end());
                }
                if (!every || !splits || taken == base_taken) continue;
                vector<vector<bool>> turned = taken;
                for (vector<bool>& one : turned) one.flip();
                if (ranges::find(partner_picks_taken, taken) != partner_picks_taken.end() ||
                    ranges::find(partner_picks_taken, turned) != partner_picks_taken.end())
                    continue;
                partner_picks_taken.push_back(std::move(taken));
                partner_picks_taken.push_back(std::move(turned));
                piece_pick partnered{ base };
                partnered.partner = (int8_t)((partner_direction + 1) * (int)partner_by::count + way);
                for (const bool but_those : { false, true }) {
                    partnered.but_those = but_those;
                    picks.push_back(partnered);
                }
            }
        }
    }
    return picks;
}

// Every pick a word is asked with: the colors an unsaid change starts from, when
// there are any, then every pick by holes, size, band, the color most cells
// wear and how many cells the ops can make, then
// each of `fit_colors` alone kept to its pieces that would fit into a gap of
// their partner, each partner way. Colors that change from pair to pair are
// asked after those, so a pick by holes or size that accounts for as much wins
// the tie (aabf363d's shape, a new color in its test); then each of the colors
// alone; then the parts one color's walls fence off, and the keys a pick turned
// round leaves to be like.
inline vector<piece_pick> picks_to_ask(const replay_facts& facts, uint16_t colors, uint16_t fit_colors = 0,
                                       bool partners = false) {
    vector<piece_pick> picks;
    const bool steady = colors && same_colors_every_pair(facts, colors);
    if (steady) picks.push_back({ colors });
    const auto holes = row_of(node_type::SelectObjectsWithHoles);
    for (int value = holes.low; value <= holes.high; ++value) picks.push_back({ 0, piece_pick::by_holes, (int8_t)value });
    // Sort each pair once for both size orders and every rank asked below.
    vector<vector<int>> sizes;
    for (const auto& cut : facts.cuts()) {
        vector<int> counts;
        for (const object& piece : cut) counts.push_back(piece.summary.count);
        ranges::sort(counts);
        sizes.push_back(std::move(counts));
    }
    auto size_at = [](const vector<int>& counts, size_t at, bool largest) {
        return counts[largest ? counts.size() - 1 - at : at];
    };
    // By size, the nearer an end the plainer: the largest and the smallest, then
    // the second of each, so where two take the same pieces in every pair the
    // plainer is said (62ab2642's "smallest", not "ninth largest", which only
    // the pairs' own counts made the same). Each asked only where no pair's
    // pieces tie at that place in the order: a tie there is broken by the order
    // the pieces came in, a pick by where they lie and not by size (833966f4's
    // cells, one cell each, the "second largest" only the second scanned). Nor
    // anywhere before it: pieces of one size ahead of it make its place count
    // how many share a size, not how large it is (272f95fa's center, "the fifth
    // smallest" only past three panes of 8 cells; three of 18 on its test put
    // another pane fifth).
    for (int value = 0; value < SIZE_RANKS; ++value)
        for (const int8_t how : { (int8_t)piece_pick::largest, (int8_t)piece_pick::smallest })
            if (ranges::all_of(sizes, [&](const vector<int>& counts) {
                    const size_t at = (size_t)value;
                    if (at >= counts.size()) return true;
                    const bool largest = how == piece_pick::largest;
                    for (size_t i = 0; i <= at && i + 1 < counts.size(); ++i)
                        if (size_at(counts, i, largest) == size_at(counts, i + 1, largest)) return false;
                    return true;
                }))
                picks.push_back({ 0, how, (int8_t)value });
    // The bands, as many as some pair's pieces make.
    for (const int8_t how : { (int8_t)piece_pick::row_band, (int8_t)piece_pick::column_band }) {
        int bands = 0;
        for (const auto& cut : facts.cuts())
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
        if (ranges::none_of(facts.cuts(), [&](const vector<object>& cut) {
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
        for (int value = 1; value <= SIZE_RANKS; ++value) {
            if (!ranges::all_of(sizes, [&](const vector<int>& counts) {
                    if ((int)counts.size() <= value) return false;
                    return size_at(counts, (size_t)value - 1, largest) != size_at(counts, (size_t)value, largest);
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
            for (const auto& cut : facts.cuts()) {
                const vector<bool> taken = picked_pieces(cut, how, value);
                const bool some = ranges::find(taken, true) != taken.end();
                every = every && some;
                splits = splits || (some && ranges::find(taken, false) != taken.end());
            }
            if (!every || !splits) continue;
            for (const bool but_those : { false, true }) picks.push_back({ 0, how, (int8_t)value, -1, but_those });
        }
    // Whether holds(which, the pieces `pick` takes) in every pair.
    auto every_pair = [&](const piece_pick& pick, auto holds) {
        for (size_t which = 0; which < facts.cuts().size(); ++which)
            if (!holds(which, facts.picked(which, pick))) return false;
        return true;
    };
    // A fit is asked only where it is at work in every pair: it keeps one piece
    // of the color at least and drops one, or the color alone says as much.
    for (int color = 0; color < COLOR_COUNT; ++color) {
        if (!(fit_colors >> color & 1)) continue;
        const piece_pick alone = piece_pick::of_color(color);
        for (int way = 0; way < (int)partner_by::count; ++way) {
            if (!facts.asks_way((partner_by)way)) continue;
            const piece_pick fit{ alone.colors, -1, 0, (int8_t)way };
            if (every_pair(fit, [&](size_t which, const vector<bool>& kept) {
                    return ranges::find(kept, true) != kept.end() && kept != facts.picked(which, alone);
                }))
                picks.push_back(fit);
        }
    }
    if (colors && !steady) picks.push_back({ colors });
    // Then each of those colors alone, where there are several and every pair
    // holds it or none does: a change of one of them is said by its own pick,
    // where the matching's acts are left unsaid (18286ef8's 9 steps toward the
    // 6 while the 5 it steps onto and the 6 change too).
    if (popcount(colors) > 1)
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (const piece_pick alone = piece_pick::of_color(color);
                colors >> color & 1 && same_colors_every_pair(facts, alone.colors))
                picks.push_back(alone);
    // THE PARTS ONE COLOR'S WALLS FENCE OFF (parts_the_pick_fences): the pieces
    // lying in one part, and every piece but those, asked only where every pair
    // holds the color and its walls fence two parts at least in every pair, and
    // only for a part every pair has that holds a piece in every pair
    // (63613498's grey corner, its key alone in part 0). Then, on every pick
    // that turns its pieces round, SelectLikeKey: the ones it keeps that are
    // like a piece of the key, the pieces it turned away, asked only where it
    // keeps one in every pair and drops one in some -- a key fenced off by walls
    // first, the surer key, so it takes a tie (63613498's corner, not the column
    // three left of the middle, which leaves its key and an 8 besides). The
    // newest picks, asked last, so an older pick that says as much takes the tie.
    auto takes_one = [](size_t, const vector<bool>& taken) { return ranges::find(taken, true) != taken.end(); };
    for (int color = 0; color < COLOR_COUNT && !facts.cuts().empty(); ++color) {
        const piece_pick walls = piece_pick::of_color(color);
        if (!every_pair(walls, takes_one)) continue;
        int parts = row_of(node_type::SelectPartition).high + 1;
        for (size_t which = 0; which < facts.cuts().size(); ++which) {
            const tile_size canvas = facts.canvas(which);
            int16_t part[MAX_GRID_DIM][MAX_GRID_DIM];
            parts = min(parts, parts_the_pick_fences(facts.handed(which, walls), canvas.width, canvas.height, part));
        }
        if (parts < 2) continue;
        for (int which_part = 0; which_part < parts; ++which_part) {
            piece_pick in_part = walls;
            in_part.part = (int8_t)which_part;
            if (!every_pair(in_part, takes_one)) continue;
            for (const bool but_those : { false, true }) {
                in_part.but_those = but_those;
                picks.push_back(in_part);
            }
        }
    }
    const size_t asked = picks.size();
    for (const bool fenced : { true, false })
        for (size_t at = 0; at < asked; ++at) {
            if (!picks[at].but_those || picks[at].fits >= 0 || (picks[at].part >= 0) != fenced) continue;
            for (int like = 0; like < KEY_LIKENESSES; ++like) {
                if (!facts.asks_likeness(like)) continue;
                piece_pick liked = picks[at];
                liked.like = (int8_t)like;
                bool drops = false;
                const bool keeps = every_pair(liked, [&](size_t which, const vector<bool>& kept) {
                    drops = drops || kept != facts.picked(which, picks[at]);
                    return takes_one(which, kept);
                });
                if (keeps && drops) picks.push_back(liked);
            }
        }
    // THE NEWEST PICKS, ASKED LAST, so an older pick that says as much takes the
    // tie (a61f2674's bars: a count asked ahead of the ranks by size took
    // their words and left its line unfinished).
    // The one piece with no twin, each way twins are read (unique_piece): asked
    // only where it takes a piece in every pair, a twin being what tells the
    // rest apart from it.
    for (int value = 0; value < (int)twin_by::count; ++value)
        if (ranges::all_of(facts.cuts(), [&](const vector<object>& cut) {
                return ranges::any_of(picked_pieces(cut, piece_pick::only_one, value), [](bool one) { return one; });
            }))
            for (const bool but_those : { false, true })
                picks.push_back({ 0, piece_pick::only_one, (int8_t)value, -1, but_those });
    // The most and the fewest of each count past size (count_of): the pieces
    // with the most colors, holes or cells of one color, or the fewest, and
    // every piece but those, asked only where the count parts some pair's
    // pieces: one at least taken there and one left. A count of one color's
    // cells only where some piece mixes that color with another: among pieces
    // of one color each it is the size of the ones of that color, which the
    // ranks by size say -- asked anyway, it peeled a61f2674's bars off one at
    // a time and left its line unfinished.
    auto mixes = [&](int color) {
        return ranges::any_of(facts.cuts(), [&](const vector<object>& cut) {
            return ranges::any_of(cut, [&](const object& piece) {
                const uint16_t worn = palette(piece);
                return (worn >> color & 1) != 0 && (worn & ~(1u << color)) != 0;
            });
        });
    };
    for (int kind = 0; kind < (int)count_kind::count; ++kind) {
        if (kind == (int)count_kind::cells) continue;
        if (kind >= (int)count_kind::of_color && !mixes(kind - (int)count_kind::of_color)) continue;
        for (const int8_t how : { (int8_t)piece_pick::largest, (int8_t)piece_pick::smallest }) {
            const int value = SIZE_RANKS + kind;
            if (ranges::none_of(facts.cuts(), [&](const vector<object>& cut) {
                    const vector<bool> taken = picked_pieces(cut, how, value);
                    return ranges::find(taken, true) != taken.end() && ranges::find(taken, false) != taken.end();
                }))
                continue;
            for (const bool but_those : { false, true })
                picks.push_back({ 0, how, (int8_t)value, -1, but_those });
        }
    }
    if (partners)
        for (const piece_pick& pick : partner_picks(facts)) picks.push_back(pick);
    return picks;
}

// Picks depend on the hand and the requested colors, not on claims or the
// canvas painted beside it. Keep their order, and forget them when carried.
// With `partners`, once the line has turned to its last resort, the partner
// picks too (partner_picks).
struct picks_on_hand {
    bool partners = false;
    std::map<std::tuple<uint16_t, uint16_t, bool>, vector<piece_pick>> asked;
    const vector<piece_pick>& get(const replay_facts& facts, uint16_t colors, uint16_t fit_colors = 0) {
        const auto [at, fresh] = asked.try_emplace({ colors, fit_colors, partners });
        if (fresh) at->second = picks_to_ask(facts, colors, fit_colors, partners);
        return at->second;
    }
    void clear() { asked.clear(); }
};

// Whether two pieces draw the same thing: the same cells in the same colors,
// wherever each lies.
inline bool same_drawing(const object& one, const object& other) {
    return same_shape(one.summary.width, one.summary.height, [&](int row, int col) { return (int)one.data[row][col]; },
                      other.summary.width, other.summary.height,
                      [&](int row, int col) { return (int)other.data[row][col]; }, DRAWING);
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
    if ((pick.how != piece_pick::largest && pick.how != piece_pick::smallest) || pick.value >= SIZE_RANKS) return false;
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
    hand_answers fresh;
    const vector<bool> picked = picked_pieces(pieces, pick, { input.width, input.height }, fresh);
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
    grid drawing(box.width(), box.height());
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
    hand_answers fresh;
    const vector<bool> picked = picked_pieces(pieces, pick, { input.width, input.height }, fresh);
    vector<const object*> kept;
    for (size_t i = 0; i < pieces.size() && kept.size() < (size_t)MAX_OBJECTS_IN_A_LIST; ++i)
        if (picked[i]) kept.push_back(&pieces[i]);
    const int count = (int)kept.size();
    if (count == 0) return false;
    int cell_width = cell, cell_height = cell;
    if (cell == 0)
        for (const object* piece : kept) {
            cell_width = max(cell_width, (int)piece->summary.width);
            cell_height = max(cell_height, (int)piece->summary.height);
        }
    if (cell_width <= 0 || cell_height <= 0) return false;
    vector<int8_t> row_band(count), col_band(count);
    auto kept_box = [&](int i) { return kept[(size_t)i]->summary.box(); };
    const int rows = bands_of(count, true, kept_box, row_band.data());
    const int cols = bands_of(count, false, kept_box, col_band.data());
    const int paper_width = min(cols * cell_width, MAX_GRID_DIM), paper_height = min(rows * cell_height, MAX_GRID_DIM);
    grid paper(paper_width, paper_height);
    for (int i = 0; i < count; ++i) {
        const object& piece = *kept[i];
        grid drawing(piece.summary.width, piece.summary.height);
        for (int row = 0; row < piece.summary.height; ++row)
            for (int col = 0; col < piece.summary.width; ++col) drawing.data[row][col] = piece.data[row][col];
        if (!shows_what_it_frames(drawing, cell_height, cell_width)) return false;
        const int top = row_band[i] * cell_height, left = col_band[i] * cell_width;
        for (int row = 0; row < cell_height && row < piece.summary.height; ++row)
            for (int col = 0; col < cell_width && col < piece.summary.width; ++col) {
                const int at_row = top + row, at_col = left + col;
                const int8_t color = piece.data[row][col];
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
    hand_answers fresh;
    const vector<bool> picked = picked_pieces(pieces, pick, { input.width, input.height }, fresh);
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
// corner over the canvas the size rule gives, a copy per input, as Tile lays
// them by its own size (tiled_by_its_own_size) -- as many as one list holds;
// turned to a pose (posed_place, the pose the
// turns and flips said for it make); a fractal, each cell of the input scaled
// up to a copy of the input where it holds a figure, as Fractal draws the input
// without its ground; or the unit it repeats (repeat_periods), as Pattern cuts it.
// What falls off the canvas is clipped. Where a plan deleted the ground, the
// input framed is its figures alone, and every frame acts on them as they are.
inline bool framed_whole(const grid& input, const piece_frame_rule& frame, const output_dimension_rule& size,
                         grid& framed) {
    using kind = piece_frame_rule::kind_of;
    const int factor = frame.factor;
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
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
            repeat_periods(input, unit_width, unit_height);
            for (int row = 0; row < unit_height; ++row)
                for (int col = 0; col < unit_width; ++col) {
                    if (!canvas.contains(row, col)) return false;
                    canvas.data[row][col] = input.data[row][col];
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
            const tile_moves moves = moves_of(tiled_by_its_own_size, { input.width, input.height });
            int copies = 0;
            each_tile_offset(tiled_by_its_own_size, moves, 0, 0, input.height, input.width, canvas.width, canvas.height,
                             [&](int, int) { return ++copies <= MAX_OBJECTS_IN_A_LIST; });
            if (copies > MAX_OBJECTS_IN_A_LIST) return false;
            each_tile_offset(tiled_by_its_own_size, moves, 0, 0, input.height, input.width, canvas.width, canvas.height,
                             [&](int down, int across) {
                for (int row = 0; row < input.height; ++row)
                    for (int col = 0; col < input.width; ++col)
                        if (canvas.contains(down + row, across + col))
                            canvas.data[down + row][across + col] = input.data[row][col];
                return true;
            });
            return true;
        }
        default: return false;
        }
    });
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
inline piece_frame_rule choose_piece_frame(const vector<example>& train, inference& inf, const tiled_pieces& tiles) {
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
    // Every piece first, then each color on its own, then the picks by kind --
    // and the partner picks, where the reading asks them (inference::partners).
    auto picks_of = [&](const replay_facts& facts) {
        vector<piece_pick> picks = { piece_pick::all() };
        for (int color = 0; color < COLOR_COUNT; ++color) picks.push_back(piece_pick::of_color(color));
        for (const piece_pick& pick : picks_to_ask(facts, 0, 0, inf.partners)) picks.push_back(pick);
        return picks;
    };
    for (const segmentation_methods cut : frame_cuts) {
        const replay_facts facts(train, cut, tiles);
        const vector<piece_pick> picks = picks_of(facts);
        for (const piece_pick& pick : picks) {
            piece_frame_rule frame;
            frame.held = true;
            frame.cut = cut;
            frame.pick = pick;
            for (size_t which = 0; which < train.size(); ++which) {
                grid framed;
                bool several = false;
                if (!framed_input(facts.cuts()[which], pick, train[which].input, inf.size, framed,
                                  several))
                    break;
                frame.merged = frame.merged || several;
            }
            weigh(frame, [&](size_t which, grid& framed) {
                bool several = false;
                return framed_input(facts.cuts()[which], pick, train[which].input, inf.size, framed,
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
                    return laid_out_input(facts.cuts()[which], pick, cell, train[which].input, inf.size, framed);
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
        for (size_t which_cut = 0; which_cut <= size(frame_cuts); ++which_cut) {
            piece_frame_rule source;
            source.cut = which_cut < size(frame_cuts) ? frame_cuts[which_cut] : segmentation_methods::unset;
            replay_facts facts(train, source.cut, tiles);
            const bool whole = source.cut == segmentation_methods::unset;
            if (whole)
                for (size_t which = 0; which < train.size(); ++which)
                    facts.change(which).hand = source.pieces(train[which].input);
            const vector<piece_pick> picks = whole ? vector<piece_pick>{ piece_pick::all() } : picks_of(facts);
            for (const piece_pick& pick : picks)
                for (int keyed = -1; keyed <= last_kind; ++keyed) {
                    shape_color_map_rule table;
                    if (keyed >= 0) {
                        vector<shape_color_map_rule::seen_key> seen;
                        bool learned = true;
                        for (size_t which = 0; which < train.size() && learned; ++which) {
                            const int8_t wanted = (int8_t)train[which].output.data[0][0];
                            const vector<object>& pieces = facts.cuts()[which];
                            const vector<bool> picked = facts.picked(which, pick);
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
                    frame.cut = source.cut;
                    frame.pick = pick;
                    frame.keyed = (int8_t)keyed;
                    const double before = best_share;
                    weigh(frame, [&](size_t which, grid& framed) {
                        return filled_input(facts.cuts()[which], pick, (int8_t)keyed, table, train[which].input,
                                            inf.size, framed);
                    });
                    if (best_share > before) best_table = table;
                }
        }
    }
    if (best.kind == piece_frame_rule::filled && best.keyed >= 0) inf.shape_color_map = best_table;
    return best;
}

// WHAT IS DOWN, pair by pair, where a paint lands: the color each cell shows
// already, TRANSPARENT where nothing is -- here the pieces `facts` hands over,
// where they lie.
inline vector<cells> pieces_down(const replay_facts& facts) {
    vector<cells> down(facts.cuts().size());
    for (size_t which = 0; which < facts.cuts().size(); ++which)
        for (const object& piece : facts.cuts()[which])
            piece.each_cell([&](int row, int col, int8_t color) {
                if (facts.train[which].input.contains(row, col)) down[which][row][col] = color;
            });
    return down;
}

// Whether a paint laid `lay` lands on a cell where `down` shows: over it lands
// everywhere; under it only where nothing is, as Underlay paints; onto it only
// where something is, as MaskBy[10] keeps; onto one color only where that color
// is, as MaskBy of it keeps; and off one color everywhere else, as MaskOut of
// it keeps (mask_keeps, 8b_op_math.hpp, what the ops keep).
inline bool lands(int8_t lay, int8_t down) {
    using rule = correspondence_election::fired_rule;
    if (lay == rule::laid_under) return down == (int8_t)TRANSPARENT;
    if (lay == rule::laid_onto) return mask_keeps(down, 0, /*any=*/true, /*keep_match=*/true);
    if (lay >= rule::laid_off_color) return mask_keeps(down, (int8_t)(lay - rule::laid_off_color), false, false);
    if (lay >= rule::laid_onto_color) return mask_keeps(down, (int8_t)(lay - rule::laid_onto_color), false, true);
    return true;
}

// HOW A PAINT IS LAID: the way `lay` says (lands) on what `canvas` holds down,
// pair by pair -- nothing, where there is no canvas -- and in the colors it
// draws, or all in one color -- the one the output shows where it first lands,
// the line's Recolor after the word, whatever the colors drawn -- or in the
// colors it draws where the output agrees, and else in one.
struct laying {
    int8_t lay = correspondence_election::fired_rule::laid_over;
    const vector<cells>* canvas = nullptr;
    enum colors : int8_t { as_drawn, in_one, own_else_one } in = as_drawn;
};

// WHAT A REPLAY SAW, where its word holds -- every replay below hands one back,
// or nothing where the word does not hold. The cells the word lays down right
// on each output, pair by pair: where its paint landed, where its pieces
// landed, what they came out in or kept. The cells its picked pieces stood on
// that it took them off or left as they were: where they slid from, what they
// lost, the pieces a copy was drawn from. The cells it lays wrong, in what it
// lays there (brings_nearer). How many pieces it changed; the one color it
// paints in, -1 for the colors it draws; and whether it picked every piece.
struct replayed {
    cells_by_pair drawn, left;
    colors_by_pair wrong;
    int pieces = 0;
    int8_t paint = -1;
    bool every_piece = true;
    int wrong_cells() const {
        int count = 0;
        for (const auto& cells : wrong) count += (int)cells.size();
        return count;
    }
    int right_cells() const {
        int count = 0;
        for (const auto& cells : drawn) count += (int)cells.size();
        return count;
    }
};

// THE ONE JUDGE OF A WORD, PAIR BY PAIR (the user: a word thrown out for one
// wrong cell is thrown out too early). In a pair a word makes cells right --
// `gained`: the output shows what it laid there, and the cell did not show it
// before -- and lays cells wrong, where the output shows something else. It
// holds in that pair where it gains more than it lays wrong, so it brings the
// pair nearer its output, one cell gained at least: at work (replay_facts).
// Its wrong cells are left for a word said after it, or the next pass on the
// picture so far, to set right (bdad9b1f's two lines, each wrong only where
// they cross, the crossing a color of its own). Where the line does not draw
// its picture (replay_facts::tolerant), no pass could see a cell left wrong,
// and a word holds only where it lays none. A move or a removal is at work
// only where it changed a piece too (`acted`). In a finished pair
// (replay_facts::finished) there is nothing to gain: a word holds there where
// it lays nothing wrong.
struct pair_tally { int gained = 0, wrong = 0; bool acted = true; };
inline bool brings_nearer(const replay_facts& facts, size_t which, const pair_tally& tally) {
    if (facts.finished(which)) return tally.wrong == 0;
    return tally.acted && tally.gained > tally.wrong && (facts.tolerant || tally.wrong == 0);
}

// A PAINT, WALKED: on each pair, every cell its picked pieces paint, once each
// -- `draw(which, cut, put)` hands each one, in the color it paints it, to
// `put(row, col, color)`, which answers whether the paint may still hold, and
// answers false for a pair it cannot draw at all -- with the color laid there
// first and last, whether two of its own disagree there (torn), and what the
// picked pieces showed there before. Every way the paint is laid reads the
// walk (lay_paint), so a paint asked several ways is walked once, and laid on
// a new canvas without walking it again.
struct paint_walk {
    struct cell { int8_t row, col, first, last, was; bool torn; };
    vector<vector<cell>> pairs;
    bool every_piece = true;
};

// A LAY KEPT FOR SCORING: marks of its right cells and a count of its wrong
// ones. Its walk and laying can supply the full lists once this kind offers
// it. The walk belongs to the cut; the canvas is unchanged while offers are
// weighed, and a winning lay is materialized before that canvas changes.
struct marked_paint {
    cell_marks drawn;
    int wrong = 0;
    int8_t paint = -1;
    bool every_piece = true;
    const paint_walk* walk = nullptr;
    laying how;
    int wrong_cells() const { return wrong; }
};

// What a paint laid `how` reads where it lands on pair `which`: nothing, laid
// over what is down, where it lands everywhere; else its canvas's pair, and
// nothing down where there is no canvas.
inline const cells* landing_on(const laying& how, size_t which) {
    using rule = correspondence_election::fired_rule;
    static const cells nothing_down;
    return how.lay == rule::laid_over ? nullptr : how.canvas ? &(*how.canvas)[which] : &nothing_down;
}

// WHAT LANDS OF A WALK laid the way `how` says on its canvas (lands): over what
// is down, every cell, in the last color laid there; under it, where nothing
// is, in the first -- the first paint to reach a cell keeps it, this one's own
// earlier cells too, as Underlay paints; onto or off what is down, or one color
// of it, the cells the canvas lets through. Each handed to `see(cell, color,
// torn)`, in the order the walk first reached them.
template <class See>
inline void each_landing(const paint_walk& walk, size_t which, const laying& how, See see) {
    using rule = correspondence_election::fired_rule;
    const cells* down = landing_on(how, which);
    const bool first_wins = how.lay == rule::laid_under;
    for (const paint_walk::cell& cell : walk.pairs[which])
        if (!down || lands(how.lay, (*down)[cell.row][cell.col]))
            see(cell, first_wins ? cell.first : cell.last, !first_wins && cell.torn);
}

// A WALKED PAINT, LAID: what lands of it (each_landing) -- laid in one color,
// every cell in the color the output shows most where it lands, over every
// pair, the first to land of those on a tie: a paint is never no color -- and
// each pair judged (brings_nearer): a cell is gained where the output shows
// what landed and the picked pieces did not show it there, and a cell its own
// paint laid twice in two colors is wrong whatever the output shows. Sees that
// one color (-1 laid in the colors drawn), the cells that landed right and the
// ones that landed wrong, and whether it picked every piece.
template <class Seen = replayed>
inline optional<Seen> lay_paint(const replay_facts& facts, const paint_walk& walk, const laying& how) {
    ++inference_work.lays;
    Seen seen;
    seen.every_piece = walk.every_piece;
    const size_t pairs = walk.pairs.size();
    if (how.in == laying::in_one) {
        int shown[COLOR_COUNT] = {};
        for (size_t which = 0; which < pairs; ++which)
            each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t, bool) {
                if (const int color = facts.train[which].output.data[cell.row][cell.col]; is_color(color)) ++shown[color];
            });
        const int most = *ranges::max_element(shown);
        for (size_t which = 0; which < pairs && seen.paint < 0 && most > 0; ++which)
            each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t, bool) {
                if (const int color = facts.train[which].output.data[cell.row][cell.col];
                    seen.paint < 0 && is_color(color) && shown[color] == most)
                    seen.paint = (int8_t)color;
            });
        if (!is_color(seen.paint)) return nullopt;
    }
    if constexpr (std::is_same_v<Seen, marked_paint>) {
        seen.drawn.at.resize(pairs);
        seen.walk = &walk;
        seen.how = how;
    } else {
        seen.drawn.assign(pairs, {});
        seen.wrong.assign(pairs, {});
    }
    for (size_t which = 0; which < pairs; ++which) {
        const grid& output = facts.train[which].output;
        pair_tally tally;
        each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t color, bool torn) {
            const int8_t laid = how.in == laying::in_one ? seen.paint : color;
            if (!torn && output.data[cell.row][cell.col] == laid) {
                if constexpr (std::is_same_v<Seen, marked_paint>)
                    seen.drawn.at[which].set((size_t)cell_index(cell.row, cell.col));
                else seen.drawn[which].push_back({ cell.row, cell.col });
                tally.gained += cell.was != laid;
            } else {
                ++tally.wrong;
                if constexpr (std::is_same_v<Seen, marked_paint>) ++seen.wrong;
                else seen.wrong[which].push_back({ cell.row, cell.col, laid });
            }
        });
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    return seen;
}

// A WALK COUNTED BY WHAT IS DOWN WHERE ITS CELLS LIE, pair by pair: for each
// thing a cell can show on the canvas -- nothing, or each color -- how many of
// the walk's cells lie on it, and of those, laid in the last color laid there
// (over what is down, or onto it, and not torn) or the first (under it), how
// many the output shows so and how many of those the picked pieces did not
// show already; and how many the output shows in each color, and of those how
// many the picked pieces showed so already (laid in one color). Each way the
// walk is laid on that canvas takes a sum of these (may_hold), so a lay that
// cannot hold is known without going over its cells again.
struct walk_tally {
    // 0 where nothing is down, 1 + c where color c is.
    static constexpr int downs = COLOR_COUNT + 1;
    struct of_pair {
        int cells[downs] = {}, right_last[downs] = {}, gained_last[downs] = {}, right_first[downs] = {},
            gained_first[downs] = {};
        int shows[downs][COLOR_COUNT] = {}, shown_before[downs][COLOR_COUNT] = {};
    };
    vector<of_pair> pairs;
};
inline walk_tally tally_walk(const replay_facts& facts, const paint_walk& walk, const vector<cells>& canvas) {
    walk_tally tally;
    tally.pairs.resize(walk.pairs.size());
    for (size_t which = 0; which < walk.pairs.size(); ++which) {
        walk_tally::of_pair& counted = tally.pairs[which];
        const grid& output = facts.train[which].output;
        for (const paint_walk::cell& cell : walk.pairs[which]) {
            const int8_t down = which < canvas.size() ? canvas[which][cell.row][cell.col] : (int8_t)TRANSPARENT;
            const int at = down == (int8_t)TRANSPARENT ? 0 : 1 + down;
            const int shows = output.data[cell.row][cell.col];
            ++counted.cells[at];
            if (!cell.torn && shows == cell.last) {
                ++counted.right_last[at];
                counted.gained_last[at] += cell.was != cell.last;
            }
            if (shows == cell.first) {
                ++counted.right_first[at];
                counted.gained_first[at] += cell.was != cell.first;
            }
            if (is_color(shows)) {
                ++counted.shows[at][shows];
                counted.shown_before[at][shows] += cell.was == shows;
            }
        }
    }
    return tally;
}

// WHETHER A WALK LAID `how` MAY HOLD, read off its tally on the canvas `how`
// names (walk_tally) as lay_paint judges it, cell by cell: false where it
// cannot; true where it may, and then it is laid cell by cell -- as it is
// where, laid in one color, two colors tie for the one the output shows most,
// which only the order the cells landed in breaks.
inline bool may_hold(const replay_facts& facts, const walk_tally& tally, const laying& how) {
    using rule = correspondence_election::fired_rule;
    constexpr int downs = walk_tally::downs;
    bool lands_on[downs];
    for (int at = 0; at < downs; ++at) lands_on[at] = lands(how.lay, at == 0 ? (int8_t)TRANSPARENT : (int8_t)(at - 1));
    const bool first_wins = how.lay == rule::laid_under;
    int paint = -1;
    if (how.in == laying::in_one) {
        long long shown[COLOR_COUNT] = {};
        for (const walk_tally::of_pair& counted : tally.pairs)
            for (int at = 0; at < downs; ++at)
                if (lands_on[at])
                    for (int color = 0; color < COLOR_COUNT; ++color) shown[color] += counted.shows[at][color];
        const long long most = *ranges::max_element(shown);
        if (most == 0) return false;
        if (ranges::count(shown, most) > 1) return true;
        paint = (int)(ranges::find(shown, most) - std::begin(shown));
    }
    for (size_t which = 0; which < tally.pairs.size(); ++which) {
        const walk_tally::of_pair& counted = tally.pairs[which];
        pair_tally said;
        int cells = 0, right = 0;
        for (int at = 0; at < downs; ++at) {
            if (!lands_on[at]) continue;
            cells += counted.cells[at];
            if (paint >= 0) {
                right += counted.shows[at][paint];
                said.gained += counted.shows[at][paint] - counted.shown_before[at][paint];
            } else if (first_wins) {
                right += counted.right_first[at];
                said.gained += counted.gained_first[at];
            } else {
                right += counted.right_last[at];
                said.gained += counted.gained_last[at];
            }
        }
        said.wrong = cells - right;
        if (!brings_nearer(facts, which, said)) return false;
    }
    return true;
}

// THE WALK `draw` MAKES (paint_walk), or nothing where a pair cannot be drawn
// -- or where, laid only the way `only` says, the paint cannot hold, done with
// at once: allowed no wrong cell, at the first -- laid in the colors drawn, a
// cell the output shows in another; laid in one, a cell the output shows in
// another than where the first landed -- and else once as many cells are sure
// to be wrong as the pair could gain. THE MOST A PAIR COULD GAIN is every cell
// where the output shows anything but what the picked pieces show there; a
// cell first laid in a color the output does not show stays wrong -- laid
// again in another, it is torn -- and laid in one color, a cell the output
// shows bare is wrong whatever the color.
template <class Draw>
inline optional<paint_walk> walk_paint(const replay_facts& facts, const vector<partner_pieces>& handed, Draw draw,
                                       const laying* only = nullptr) {
    ++inference_work.walks;
    using rule = correspondence_election::fired_rule;
    if (!facts.overlays || facts.train.empty()) return nullopt;
    paint_walk walk;
    walk.pairs.resize(facts.train.size());
    const bool exact = only && !facts.tolerant;
    int8_t first_shown = (int8_t)TRANSPARENT;
    bool first = true;
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const example& pair = facts.train[which];
        const partner_pieces& cut = handed[which];
        walk.every_piece = walk.every_piece && cut.every_piece_picked();
        // What the picked pieces show, where they lie: drawn again as it is, a
        // cell of them is no work.
        const auto& before = cut.shown(pair.input.width, pair.input.height);
        // Each cell's place in the walk, -1 for none yet: every cell, since a
        // piece the words moved may lay one past the grid's edge.
        int16_t index[MAX_GRID_DIM][MAX_GRID_DIM];
        std::memset(index, 0xFF, sizeof index);
        auto& cells = walk.pairs[which];
        // Laid only one way, what is down where it lands (landing_on); laid
        // under, its own cells go down as they land, on a copy of its own.
        optional<arc::cells> under;
        const arc::cells* down = only ? landing_on(*only, which) : nullptr;
        if (only && only->lay == rule::laid_under) {
            under.emplace(*down);
            down = &*under;
        }
        bool holds = true;
        int could_gain = -1, sure_wrong = 0;
        const bool drew = draw(which, cut, [&](int row, int col, int8_t color) {
            if (!holds) return false;
            int16_t& at = index[row][col];
            const bool fresh = at < 0;
            if (fresh) {
                at = (int16_t)cells.size();
                cells.push_back({ (int8_t)row, (int8_t)col, color, color, before[row][col], false });
            } else {
                paint_walk::cell& cell = cells[(size_t)at];
                cell.torn = cell.torn || cell.last != color;
                cell.last = color;
            }
            if (!only || (down && !lands(only->lay, (*down)[row][col]))) return holds;
            // Under what is down, its own cell is down now: something, whatever its color.
            if (under) (*under)[row][col] = is_color(color) ? color : (int8_t)0;
            const int8_t shows = (int8_t)pair.output.data[row][col];
            if (exact) {
                if (only->in == laying::as_drawn) holds = shows == color;
                else {
                    if (first) first_shown = shows;
                    first = false;
                    holds = is_color(first_shown) && shows == first_shown;
                }
            } else if (fresh && (only->in == laying::in_one ? shows == (int8_t)TRANSPARENT : shows != color)) {
                if (could_gain < 0) {
                    could_gain = 0;
                    for (int r = 0; r < pair.output.height; ++r)
                        for (int c = 0; c < pair.output.width; ++c)
                            could_gain += pair.output.data[r][c] != before[r][c];
                }
                holds = ++sure_wrong < could_gain;
            }
            return holds;
        });
        if (!drew || !holds) return nullopt;
    }
    return walk;
}

// A PAINT LAID ON WHAT IS DOWN, replayed: what every replay of a word that
// draws asks, once -- its walk (walk_paint), laid the one way `how` says
// (lay_paint), done with as soon as it cannot hold so. Laid in its own colors
// or else in one, it is walked once and laid both ways: in its own colors where
// they lay no cell wrong, else whichever way says more, right cells less wrong
// ones, its own colors on a tie. Handed the pieces its pick takes
// (handed_pieces), made ready once for every way the word is asked.
template <class Draw>
inline optional<replayed> paint_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      const laying& how, Draw draw) {
    ++inference_work.paint_replays;
    if (how.in == laying::own_else_one) {
        const auto walk = walk_paint(facts, handed, draw);
        if (!walk) return nullopt;
        auto own = lay_paint(facts, *walk, { how.lay, how.canvas });
        if (own && own->wrong_cells() == 0) return own;
        auto one = lay_paint(facts, *walk, { how.lay, how.canvas, laying::in_one });
        if (one && (!own || one->right_cells() - one->wrong_cells() > own->right_cells() - own->wrong_cells()))
            return one;
        return own;
    }
    const auto walk = walk_paint(facts, handed, draw, &how);
    return walk ? lay_paint(facts, *walk, how) : nullopt;
}

// The color a recolor's replay names for a piece the op leaves as it is.
constexpr int8_t LEFT_AS_IT_IS = -2;

// PICKED PIECES REPAINTED WHERE THEY LIE, replayed: what every recolor's replay
// asks, once. On each pair, `colors_of(which, cut, i, to)` fills `to`, the color
// each color picked piece i wears comes out in -- LEFT_AS_IT_IS where the word
// leaves it as it is (ColorFrom's piece with no partner; ShapeColorMap's key the
// table does not name; a color no key pairs) -- or answers false where the word
// cannot say; the piece is then a paint laid over itself in those colors
// (paint_holds), so a piece the word leaves as it is has to come out as it
// was: the word picked it, and does not get to leave it to another (4f537728's
// blocks in the marker's column, left by "takes the color of a piece in its
// rows", came out recolored). Sees the cells of the pieces it repainted and
// how many pieces changed color.
// Cell by cell: `cells_of(which, cut, i, repaint)` hands each cell of picked
// piece i to repaint(row, col, was, now), or answers false where the word
// cannot say.
template <class CellsOf>
inline optional<replayed> cells_repainted_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                               CellsOf cells_of) {
    int pieces = 0;
    auto seen = paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool changed = false;
            if (!cells_of(which, cut, i, [&](int row, int col, int8_t was, int8_t now) {
                    put(row, col, now);
                    changed = changed || was != now;
                }))
                return false;
            pieces += changed;
        }
        return true;
    });
    if (seen) seen->pieces = pieces;
    return seen;
}
template <class ColorsOf>
inline optional<replayed> repaints_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                        ColorsOf colors_of) {
    return cells_repainted_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, auto&& repaint) {
        int8_t to[COLOR_COUNT];
        ranges::fill(to, LEFT_AS_IT_IS);
        if (!colors_of(which, cut, i, to)) return false;
        cut.pieces[i].each_cell([&](int row, int col, int8_t was) {
            repaint(row, col, was, to[was] == LEFT_AS_IT_IS ? was : to[was]);
        });
        return true;
    });
}

// A RECOLOR OF WHOLE PIECES, replayed: `color_of(which, cut, i, color)` says the
// one color picked piece i comes out in, every cell of it, or LEFT_AS_IT_IS
// (repaints_hold).
template <class ColorOf>
inline optional<replayed> recolors_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                        ColorOf color_of) {
    return repaints_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, int8_t* to) {
        int8_t color = -1;
        if (!color_of(which, cut, i, color)) return false;
        std::fill(to, to + COLOR_COUNT, color);
        return true;
    });
}

// COLORS THAT TRADE PLACES AS A KEY PAIRS THEM, replayed: BidiColorMap past its
// fixed pairs reads the key on each grid -- every piece not picked, along its
// rows or its columns (key_swaps) -- and every cell of a picked piece in a color
// the key pairs comes out in its partner's (repaints_hold). A key that names no
// swap leaves them as they were, so that pair shows the word at no work.
inline optional<replayed> key_swap_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                         int value) {
    vector<array<int8_t, COLOR_COUNT>> swaps(handed.size());
    for (size_t which = 0; which < handed.size(); ++which) swaps_of(value, handed[which], swaps[which].data());
    return repaints_hold(facts, handed, [&](size_t which, const partner_pieces&, int, int8_t* to) {
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (swaps[which][color] >= 0) to[color] = swaps[which][color];
        return true;
    });
}

// A COLOR TRADING PLACES WITH EMPTY, replayed: BidiColorMap past the key's two
// over every cell of each picked piece's box (traded_with_empty, the op's own
// rule) -- that color's cells left empty, the box's empty cells that color --
// the box laid over what is down as it comes out (paint_holds).
inline optional<replayed> empty_swap_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                           int color) {
    return paint_holds(facts, handed, {}, [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            for (int row = 0; row < piece.summary.height; ++row)
                for (int col = 0; col < piece.summary.width; ++col)
                    put(piece.summary.row + row, piece.summary.col + col,
                        traded_with_empty((int8_t)piece.data[row][col], color));
        }
        return true;
    });
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
static_assert(row_of(node_type::ColorFrom).high == COLORS_FROM_ITS_SHAPE + SHAPE_LIKENESSES - 1,
              "ColorFrom takes one value for each way partner_by finds a partner, then the key by place, "
              "then each likeness of shape cell by cell");

// A PIECE COLORED CELL BY CELL BY ANOTHER OF ITS SHAPE, replayed: ColorFrom
// past the key by place paints each cell of every picked piece in the color of
// the cell of its one partner of its shape it lies on (colors_from_its_shape,
// the walk the op takes); a piece with no partner of its shape is left as it
// is, one with several the word cannot say.
inline optional<replayed> colors_from_its_shape_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                                      int like) {
    const partner_by way = (partner_by)((int)partner_by::same_shape + like);
    return cells_repainted_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, auto&& repaint) {
        const bool laid = colors_from_its_shape(like, i, cut, [&](int row, int col, int color) {
            repaint(row, col, (int8_t)cut.color_at(i, row, col), is_color(color) ? (int8_t)color
                                                                                 : (int8_t)cut.color_at(i, row, col));
        });
        if (laid) return true;
        if (cut.partnered(way, i)) return false;
        cut.pieces[i].each_cell([&](int row, int col, int8_t was) { repaint(row, col, was, was); });
        return true;
    });
}

// A PIECE THAT TAKES ITS COLOR OFF A KEY BY ITS PLACE, replayed: ColorFrom past
// its partner ways reads the key on each grid -- every piece not picked -- and
// paints the n-th picked piece in reading order in the key's n-th entry
// (key_place_colors: 33b52de3's shapes, its key cell by cell). Where the key
// does not say, every piece is left as it is, so that pair shows the word at no
// work (recolors_hold).
inline optional<replayed> key_place_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    vector<vector<int8_t>> places;
    for (const partner_pieces& cut : handed) places.push_back(cut.key_places());
    return recolors_hold(facts, handed, [&](size_t which, const partner_pieces&, int i, int8_t& color) {
        const int8_t worn = places[which][(size_t)i];
        color = worn >= 0 ? worn : LEFT_AS_IT_IS;
        return true;
    });
}

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
        for (size_t which = 0; which < ranked.cuts().size(); ++which) {
            const grid& input = ranked.train[which].input;
            vector<object> split;
            for (const object& piece : ranked.cuts()[which]) {
                grid layer(input.width, input.height);
                paint_piece(layer, piece, piece.summary.row, piece.summary.col, true, /*clip=*/true);
                object_walk_scratch scratch;
                each_object(layer, -1, grouping::touching_any_color, scratch,
                            [&](const object_summary& box, const int16_t* cells, int count) {
                    split.push_back(object::from_cells(layer, box, cells, count));
                });
            }
            ranked.change(which).hand = std::move(split);
        }
    // Each piece's place in the order, pair by pair.
    vector<vector<int>> places;
    for (const vector<object>& hand : ranked.cuts()) {
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

// PICKED PIECES SLID, replayed: each by its own `down` and `across`, and the
// hand then drawn as the line lays it (draw_the_hand), every piece landing on
// the output. A moved piece's cell lands right where it shows -- in the
// drawing, not covered by a piece the line lays over it, and in the output --
// and leaves its cell right where nothing else now covers it and the output
// shows anything but the piece itself -- a cell still showing it was a copy
// left behind, not a move; what does show there is another word's to say
// (18286ef8's box closing behind its 9). Each cell otherwise is wrong. A right
// cell is gained where the drawing changed there and now shows what the output
// shows (brings_nearer, which its caller asks, with one piece at least moved).
// Adds the cells each one left and the cells it landed on, right, and the ones
// wrong to the pair's lists, and counts the pieces that moved.
inline bool slides_show(const example& pair, const partner_pieces& cut, bool under,
                        const vector<int>& down, const vector<int>& across, pair_tally& tally,
                        vector<std::pair<int, int>>& left, vector<std::pair<int, int>>& landed,
                        vector<cell_color>& wrong, int& pieces_moved) {
    grid drawn(pair.output.width, pair.output.height), was = drawn;
    if (!draw_the_hand(drawn, cut.pieces, under, /*clip=*/false,
                       [&](size_t i) { return direction_step{ down[i], across[i] }; }))
        return false;
    draw_the_hand(was, cut.pieces, under, /*clip=*/true, [](size_t) { return direction_step{ 0, 0 }; });
    auto gains = [&](int row, int col) {
        return drawn.data[row][col] != was.data[row][col] && drawn.data[row][col] == pair.output.data[row][col];
    };
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i] || (down[i] == 0 && across[i] == 0)) continue;
        cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
            const int to_row = row + down[i], to_col = col + across[i];
            if (drawn.data[to_row][to_col] == color && pair.output.data[to_row][to_col] == color) {
                landed.push_back({ to_row, to_col });
                tally.gained += gains(to_row, to_col);
            } else {
                ++tally.wrong;
                wrong.push_back({ (int8_t)to_row, (int8_t)to_col, (int8_t)drawn.data[to_row][to_col] });
            }
            if (drawn.data[row][col] != TRANSPARENT || pair.output.data[row][col] != color) {
                left.push_back({ row, col });
                tally.gained += gains(row, col);
            } else {
                ++tally.wrong;
                wrong.push_back({ (int8_t)row, (int8_t)col, (int8_t)TRANSPARENT });
            }
        });
        ++pieces_moved;
    }
    return true;
}

// PICKED PIECES SLID, EACH ITS OWN WAY, replayed: what every move word's replay
// asks, once. On each pair, `where(which, cut, i, by)` says how far picked piece
// i goes, down and across, or false where the word cannot say; the pieces then
// slide as slides_show reads them, and the word holds only where one piece at
// least moved and it brings the pair nearer (brings_nearer). Sees the cells each
// one left and the cells it landed on, right and wrong, and how many pieces
// moved. Handed the pieces its pick takes (handed_pieces).
template <class Where>
inline optional<replayed> slides_hold(const replay_facts& facts, const vector<partner_pieces>& handed, Where where) {
    ++inference_work.slide_replays;
    if (!facts.overlays) return nullopt;
    replayed seen;
    seen.left.assign(facts.train.size(), {});
    seen.drawn.assign(facts.train.size(), {});
    seen.wrong.assign(facts.train.size(), {});
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const int before = seen.pieces;
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        vector<int> downs(cut.pieces.size(), 0), acrosses(cut.pieces.size(), 0);
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            direction_step by{ 0, 0 };
            if (!where(which, cut, i, by)) return nullopt;
            downs[i] = by.down;
            acrosses[i] = by.across;
        }
        pair_tally tally;
        if (!slides_show(facts.train[which], cut, facts.laid_under, downs, acrosses, tally, seen.left[which],
                         seen.drawn[which], seen.wrong[which], seen.pieces))
            return nullopt;
        tally.acted = seen.pieces > before;
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    if (seen.pieces == 0) return nullopt;
    return seen;
}

// A PIECE THAT MOVES LEVEL WITH ANOTHER, replayed. AlignTo slides every picked
// piece, down or across alone, until its box starts on the row -- or with
// `columns`, the column -- its one partner's box starts on (goes_level).
inline optional<replayed> align_holds(const replay_facts& facts, const vector<partner_pieces>& handed, bool columns,
                                      partner_by how) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, direction_step& by) {
        return cut.goes_level(i, columns, how, by);
    });
}
static_assert(row_of(node_type::AlignTo).high + 1 == 2 * (int)partner_by::count,
              "AlignTo takes rows then columns, each by every way partner_by finds a partner");

// A PIECE THAT MOVES TOWARD ANOTHER, replayed. MoveToward moves every picked
// piece toward its one partner of the kind `value` % PARTNER_KINDS names -- each
// way partner_by finds one, then each color -- by what `value` / PARTNER_KINDS
// names: one step, its own size, or until it touches it (goes_toward,
// move_toward in 8b_op_math.hpp, the move the op makes).
inline optional<replayed> toward_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int value) {
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, direction_step& by) {
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
    return slides_hold(facts, handed, [&](size_t, const partner_pieces&, int, direction_step& by) {
        by = { down, across };
        return true;
    });
}

// A PICKED PIECE SLID TO THE EDGE OF THE GRID, replayed: that way, until its box
// meets the edge (steps_to_edge, 8b_op_math.hpp, the walk MoveToEdge takes).
inline optional<replayed> to_edge_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int move_direction) {
    const direction_step step = step_of(move_direction);
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, direction_step& by) {
        const grid& output = facts.train[which].output;
        const int steps = steps_to_edge(cut.boxes[i], output.width, output.height, move_direction);
        by = { steps * step.down, steps * step.across };
        return true;
    });
}

// A PICKED PIECE MOVED ITS OWN SIZE, replayed: the direction `move_direction` names
// (shift_by, 8b_op_math.hpp, the step the op takes).
inline optional<replayed> shift_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int move_direction) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, direction_step& by) {
        by = shift_by(move_direction, cut.boxes[i].width(), cut.boxes[i].height());
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
// `keeps(cut, i, row, col)` names and loses the rest. A kept cell is right
// where the output shows it in the piece's own color, and a lost one where the
// output shows it bare -- the ground -- or a piece still there covers it: one
// not picked, or a picked one that keeps that cell. Each cell otherwise is
// wrong. A lost cell right is gained where the output shows something else
// there now, and in each pair one piece at least lost such a cell, and the
// word brings the pair nearer (brings_nearer). Sees the cells the picked pieces
// keep (drawn) and lose (left), right, the ones wrong, and how many pieces lost
// a cell that shows it.
template <class Keeps>
inline optional<replayed> drops_hold(const replay_facts& facts, const vector<partner_pieces>& handed, Keeps keeps) {
    ++inference_work.drop_replays;
    if (!facts.overlays || facts.train.empty()) return nullopt;
    replayed seen;
    seen.drawn.assign(facts.train.size(), {});
    seen.left.assign(facts.train.size(), {});
    seen.wrong.assign(facts.train.size(), {});
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
        pair_tally tally;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool changed = false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                const int8_t there = (int8_t)pair.output.data[row][col];
                if (keeps(cut, i, row, col)) {
                    if (there == color) seen.drawn[which].push_back({ row, col });
                    else {
                        ++tally.wrong;
                        seen.wrong[which].push_back({ (int8_t)row, (int8_t)col, color });
                    }
                    return;
                }
                if (there == TRANSPARENT || still_there(i, row, col)) {
                    seen.left[which].push_back({ row, col });
                    tally.gained += there != color;
                    changed = changed || there != color;
                } else {
                    ++tally.wrong;
                    seen.wrong[which].push_back({ (int8_t)row, (int8_t)col, (int8_t)TRANSPARENT });
                }
            });
            seen.pieces += changed;
        }
        tally.acted = seen.pieces > before;
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    if (seen.pieces == 0) return nullopt;
    return seen;
}

// PICKED PIECES REMOVED, replayed. DeleteSelection takes the pieces a pick takes
// out of the hand, so nothing paints them: they keep no cell (drops_hold), and
// the cells they sat on are left.
inline optional<replayed> removal_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return drops_hold(facts, handed, [](const partner_pieces&, int, int, int) { return false; });
}

// A PIECE HOLLOWED TO ITS EDGE, as Outline leaves it: the cells that have, among
// their eight neighbours, a cell that is not the piece's (on_object_edge, the
// test the op takes), the rest dropped.
inline object outlined(const object& piece) {
    object rim = piece;
    auto occupied = [&](int row, int col) {
        return row >= 0 && col >= 0 && row < piece.summary.height && col < piece.summary.width &&
               piece.data[row][col] != (int8_t)TRANSPARENT;
    };
    rim.summary.count = 0;
    for (int row = 0; row < piece.summary.height; ++row)
        for (int col = 0; col < piece.summary.width; ++col) {
            if (!occupied(row, col)) continue;
            if (on_object_edge(row, col, occupied)) ++rim.summary.count;
            else rim.data[row][col] = (int8_t)TRANSPARENT;
        }
    return rim;
}

// A PIECE HOLLOWED TO ITS EDGE, replayed. Outline keeps the cells of each picked
// piece that have, among their eight neighbours, a cell that is not the piece's
// (on_object_edge, the test the op takes), and drops the rest (drops_hold).
inline optional<replayed> outline_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return drops_hold(facts, handed, [](const partner_pieces& cut, int i, int row, int col) {
        const object& piece = cut.pieces[i];
        return on_object_edge(row - piece.summary.row, col - piece.summary.col, [&](int down, int across) {
            return down >= 0 && across >= 0 && down < piece.summary.height && across < piece.summary.width &&
                   piece.data[down][across] != (int8_t)TRANSPARENT;
        });
    });
}

// The object turned or flipped to `pose` in its own box, its top-left
// corner where it was: each cell read through pose_source, as the ops read it.
inline object posed(const object& piece, int pose) {
    object turned = piece;
    const auto [width, height] = posed_size(pose, piece.summary.width, piece.summary.height);
    turned.summary.width = (int8_t)width;
    turned.summary.height = (int8_t)height;
    turned.data.clear();
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const auto [from_row, from_col] = pose_source(pose, piece.summary.width, piece.summary.height, row, col);
            turned.data[row][col] = piece.data[from_row][from_col];
        }
    return turned;
}

// One copy of a piece, mirrored so.
inline object mirrored(const object& piece, int row2, int col2, bool down, bool across) {
    object copy = posed(piece, mirror_pose(down, across));
    const cell_box to = mirrored_box(piece.summary.box(), row2, col2, down, across);
    copy.summary.row = (int8_t)to.row0;
    copy.summary.col = (int8_t)to.col0;
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

// PIECES GROWN IN STEP UNTIL THEY MEET, walked: ExtendTo past its eight ways,
// every picked piece growing along its track at once (grow_in_step, the growth
// the op runs), each cell laid in the color of the piece that took it, and the
// bodies laid down again where they lie, as the op hands them back. Refused
// where a ray stops at a meeting but the output carries it on past (23581191).
inline optional<paint_walk> grown_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       grown_along track) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int width = output.width, height = output.height;
        // Each cell's piece, the first in the cut that holds it, as the op reads its list.
        int16_t owner[MAX_GRID_CELLS], turn[MAX_GRID_CELLS], queue[MAX_GRID_CELLS];
        uint8_t cell_directions[MAX_GRID_CELLS];
        std::fill_n(owner, width * height, (int16_t)-1);
        for (int i = 0; i < cut.count(); ++i)
            cut.pieces[i].each_cell([&](int row, int col, int8_t) {
                if (output.contains(row, col) && owner[row * width + col] < 0) owner[row * width + col] = (int16_t)i;
            });
        vector<array<int16_t, 3>> meetings;
        grow_in_step(width, height, track, [&](int piece) { return cut.is_picked(piece); }, owner, turn, queue, cell_directions,
                     [&](int at, int grow_direction, int piece) { meetings.push_back({ (int16_t)at, (int16_t)grow_direction, (int16_t)piece }); });
        for (const auto& [at, grow_direction, piece] : meetings) {
            const direction_step step = step_of(grow_direction);
            const int row = at / width + step.down, col = at % width + step.across;
            if (row >= 0 && col >= 0 && row < height && col < width && turn[row * width + col] < 0 &&
                output.data[row][col] == cut.color(piece))
                return false;
        }
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool laid = true;
            cut.pieces[i].each_cell([&](int row, int col, int8_t body) {
                if (laid && output.contains(row, col)) laid = put(row, col, body);
            });
            if (!laid) return false;
        }
        for (int at = 0; at < width * height; ++at)
            if (turn[at] > 0 && !put(at / width, at % width, (int8_t)cut.color(owner[at]))) return false;
        return true;
    });
}

// A TRAIL GROWN OFF A PIECE, walked. ExtendTo marches from every cell of each
// picked piece that way (march_trail) and fills what it passes in the piece's
// color (walk_paint), laid every way the line asks (lay_paint). The op hands
// back the body with its trail, so the paint lays the body down again where it
// lies: the output must show it there too (66e6c45b's line removed a piece
// whose trail then drew it back). Past the eight ways, the pieces grow in step
// until they meet (grown_walk).
inline optional<paint_walk> trail_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int grow_direction) {
    if (grow_direction >= direction::count) return grown_walk(facts, handed, (grown_along)(grow_direction - direction::count));
    const direction_step step = step_of(grow_direction);
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
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

// THE PICKED PIECES COPIED ONTO THE OTHERS, walked. CopyTo draws each picked
// piece again onto every piece not picked, lined up the way `anchor` numbers
// CopyTo's nine -- top, middle or bottom by left, middle or right,
// aligned_copy_start (8b_op_math.hpp), the step the op takes -- each copied
// cell that lands on the grid (walk_paint), laid every way the line asks
// (lay_paint). Picking none, or every piece, or more copies than a list holds,
// it draws nothing.
inline optional<paint_walk> copy_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int anchor) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
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
                const int row0 = aligned_copy_start(pattern.height, target.row, target.height, anchor / 3);
                const int col0 = aligned_copy_start(pattern.width, target.col, target.width, anchor % 3);
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    const int at_row = row0 + row - pattern.row, at_col = col0 + col - pattern.col;
                    if (output.contains(at_row, at_col)) put(at_row, at_col, color);
                });
            }
        }
        return true;
    });
}

// A PIECE REPEATED, walked: each picked piece laid again where its Tile lays it
// (each_tile_offset, 8b_op_math.hpp, the arithmetic the op runs), each step read
// off the pieces the pair's cut hands over, a copy one step of the piece where
// a step is shorter (copy_size) -- walked (walk_paint) and laid every way the
// line asks (lay_paint). A piece no step moves is drawn where it lies, as the
// op hands it on.
inline optional<paint_walk> tile_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      const tile_value& tile) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int colors = color_count(cut.count(), [&](int i) { return (bool)cut.picked[(size_t)i]; },
                                       [&](int i) { return (uint32_t)palette(cut.pieces[(size_t)i]); });
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            const auto& box = piece.summary;
            const tile_moves moves = moves_of(tile, measures_of(tile, box.height, box.width, [&](int row, int col) {
                return piece.data[row][col] != (int8_t)TRANSPARENT; }, colors));
            const auto [rows, columns] = copy_size(moves, box.height, box.width);
            bool laid = true;
            auto lay_copy = [&](int down, int across) {
                for (int row = 0; row < rows && laid; ++row)
                    for (int col = 0; col < columns && laid; ++col) {
                        const int8_t color = piece.data[row][col];
                        const int at_row = box.row + down + row, at_col = box.col + across + col;
                        if (color != (int8_t)TRANSPARENT && output.contains(at_row, at_col))
                            laid = put(at_row, at_col, color);
                    }
                return laid;
            };
            if (moves.count == 0) lay_copy(0, 0);
            else each_tile_offset(tile, moves, box.row, box.col, rows, columns, output.width, output.height, lay_copy);
            if (!laid) return false;
        }
        return true;
    });
}

// EACH CELL THE RAYS OF A PAIR'S PICKED PIECES REACH on its output, `radius`
// cells long, from the cell of each piece nearest its middle -- up, down and to
// the sides, or along the diagonals (nearest_ray_seed and each_extended_ray,
// 8b_op_math.hpp, the walk the op takes) -- handed to `visit(row, col, along)`,
// `along` cells out along its ray, in the order the op draws them.
template <class Visit>
inline void each_ray_cell(const replay_facts& facts, size_t which, const partner_pieces& cut, bool diagonal,
                          int radius, Visit visit) {
    const grid& output = facts.train[which].output;
    const vector<ray_seed>& seeds = cut.ray_seeds();
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i]) continue;
        const auto& box = cut.pieces[i].summary;
        const ray_seed seed = seeds[(size_t)i];
        each_extended_ray(seed, radius, diagonal, [&](int down, int across) {
            const int row = box.row + seed.row + down, col = box.col + seed.col + across;
            if (output.contains(row, col)) visit(row, col, max(abs(down), abs(across)));
        });
    }
}

// RAYS GROWN OFF A PIECE, walked. Extend grows rays from the cell of each
// picked piece nearest its middle, `radius` cells long, in that cell's color
// (each_ray_cell; walk_paint), and the line may paint them over in one color of
// its own: laid every way the line asks, in one color (lay_paint), which it
// sees too.
inline optional<paint_walk> rays_walk(const replay_facts& facts, const vector<partner_pieces>& handed, bool diagonal,
                                      int radius) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        each_ray_cell(facts, which, cut, diagonal, radius, [&](int row, int col, int) { put(row, col, (int8_t)TRANSPARENT); });
        return true;
    });
}

// THE RAYS OF EVERY LENGTH, WALKED ONCE. A ray `radius` long reaches the cells
// a longer one reaches within `radius` of its seed, so the longest rays, each
// cell kept with how far along the nearest one reaches it, hold the cells of
// every length. And laid in one color, none torn (rays_walk), a length's cells
// judge the same whatever order they landed in -- but for two colors tied for
// the one the output shows most, which only that order breaks (lay_paint).
struct ray_fan {
    struct cell { int8_t row, col, was; };
    // Pair by pair, the cells the rays reach, nearest first: those first
    // reached `along` cells out are cells[starts[along]] up to cells[starts[along + 1]].
    struct of_pair { vector<cell> cells; vector<int> starts; };
    vector<of_pair> pairs;
    bool every_piece = true;
};
// The rays `reach` cells long fanned out, with what the picked pieces showed
// at each cell before; nothing where rays_walk walks nothing (walk_paint).
inline optional<ray_fan> rays_fan(const replay_facts& facts, const vector<partner_pieces>& handed, bool diagonal,
                                  int reach) {
    if (!facts.overlays || facts.train.empty()) return nullopt;
    ray_fan fan;
    fan.pairs.resize(facts.train.size());
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const example& pair = facts.train[which];
        const partner_pieces& cut = handed[which];
        fan.every_piece = fan.every_piece && cut.every_piece_picked();
        const auto& before = cut.shown(pair.input.width, pair.input.height);
        // How far along the nearest ray reaches each cell, 0 where none does.
        int8_t nearest[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        each_ray_cell(facts, which, cut, diagonal, reach, [&](int row, int col, int along) {
            int8_t& at = nearest[row][col];
            if (at == 0 || along < at) at = (int8_t)along;
        });
        ray_fan::of_pair& fanned = fan.pairs[which];
        fanned.starts.assign((size_t)reach + 2, 0);
        for (const auto& row : nearest)
            for (const int8_t along : row)
                if (along) ++fanned.starts[(size_t)along + 1];
        for (size_t along = 1; along < fanned.starts.size(); ++along) fanned.starts[along] += fanned.starts[along - 1];
        fanned.cells.resize((size_t)fanned.starts.back());
        vector<int> next(fanned.starts.begin(), fanned.starts.end() - 1);
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < MAX_GRID_DIM; ++col)
                if (const int8_t along = nearest[row][col])
                    fanned.cells[(size_t)next[(size_t)along]++] = { (int8_t)row, (int8_t)col, before[row][col] };
    }
    return fan;
}

// EVERY LENGTH OF THE RAYS LAID AT ONCE, `how` says, in one color, as lay_paint
// lays each length's own walk: adding the cells one step further out, length
// by length, it counts what lands (landing_on, lands) and what the output
// shows there, takes the color shown most and judges each pair (brings_nearer)
// -- the cells laid right, as marks, and how many are laid wrong. A length
// whose cells tie for the color is laid off its own walk (lay_paint), and a
// length that holds keeps its walk for its lists (heard::materialize):
// `walk_of(length)` hands it over, walked once. Lengths 0 to `reach`.
template <class WalkOf>
inline vector<optional<marked_paint>> lay_every_length(const replay_facts& facts, const ray_fan& fan,
                                                       const laying& how, int reach, WalkOf walk_of) {
    // What has landed so far on a pair: how many cells, of those how many the
    // output shows in each color, how many the picked pieces showed so already,
    // and where.
    struct landed_so_far {
        int cells = 0, shows[COLOR_COUNT] = {}, shown_before[COLOR_COUNT] = {};
        std::bitset<MAX_GRID_CELLS> right[COLOR_COUNT];
    };
    const size_t pairs = fan.pairs.size();
    vector<landed_so_far> landed(pairs);
    vector<optional<marked_paint>> laid((size_t)reach + 1);
    for (int length = 1; length <= reach; ++length) {
        for (size_t which = 0; which < pairs; ++which) {
            const ray_fan::of_pair& fanned = fan.pairs[which];
            if ((size_t)length + 1 >= fanned.starts.size()) continue;
            const grid& output = facts.train[which].output;
            const cells* down = landing_on(how, which);
            landed_so_far& so_far = landed[which];
            for (int at = fanned.starts[(size_t)length]; at < fanned.starts[(size_t)length + 1]; ++at) {
                const ray_fan::cell& cell = fanned.cells[(size_t)at];
                if (down && !lands(how.lay, (*down)[cell.row][cell.col])) continue;
                ++so_far.cells;
                const int shows = output.data[cell.row][cell.col];
                if (!is_color(shows)) continue;
                ++so_far.shows[shows];
                so_far.shown_before[shows] += cell.was == shows;
                so_far.right[shows].set((size_t)cell_index(cell.row, cell.col));
            }
        }
        long long shown[COLOR_COUNT] = {};
        for (const landed_so_far& so_far : landed)
            for (int color = 0; color < COLOR_COUNT; ++color) shown[color] += so_far.shows[color];
        const long long most = *ranges::max_element(shown);
        if (most == 0) continue;
        if (ranges::count(shown, most) > 1) {
            if (const paint_walk* walk = walk_of(length)) laid[(size_t)length] = lay_paint<marked_paint>(facts, *walk, how);
            continue;
        }
        const int8_t paint = (int8_t)(ranges::find(shown, most) - std::begin(shown));
        int wrong = 0;
        bool holds = true;
        for (size_t which = 0; which < pairs; ++which) {
            const landed_so_far& so_far = landed[which];
            const pair_tally tally{ so_far.shows[paint] - so_far.shown_before[paint], so_far.cells - so_far.shows[paint] };
            holds = holds && brings_nearer(facts, which, tally);
            wrong += tally.wrong;
        }
        const paint_walk* walk = holds ? walk_of(length) : nullptr;
        if (!walk) continue;
        marked_paint& seen = laid[(size_t)length].emplace();
        seen.drawn.at.resize(pairs);
        for (size_t which = 0; which < pairs; ++which) seen.drawn.at[which] = landed[which].right[paint];
        seen.wrong = wrong;
        seen.paint = paint;
        seen.every_piece = fan.every_piece;
        seen.walk = walk;
        seen.how = how;
    }
    return laid;
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
        cells copy;
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

// THE SYMMETRIES A PICTURE IS MADE TO KEEP, the narrowest first: the poses
// (1_readings.hpp's three bits) laid over the picture in turn, each taken of
// what is drawn by then, so the last pose laid makes the whole symmetry -- one
// mirror or the half turn; then both mirrors, or every quarter turn; then all
// eight ways round.
struct symmetry { const char* says; int8_t poses[3]; };
constexpr symmetry symmetries[] = {
    { "mirrored left to right", { 4, -1, -1 } },      { "mirrored top to bottom", { 2, -1, -1 } },
    { "mirrored across its diagonal", { 1, -1, -1 } }, { "mirrored across its other diagonal", { 7, -1, -1 } },
    { "turned a half turn", { 6, -1, -1 } },          { "mirrored both ways", { 4, 2, -1 } },
    { "turned every quarter turn", { 3, 6, -1 } },    { "kept all eight ways round", { 4, 2, 3 } } };

// THE PICTURE MADE SYMMETRIC, replayed: the picture so far -- the one piece
// `handed` holds in each pair, the grid's own size -- laid over itself posed
// each way the symmetry names, in turn, every pose taken of what is drawn by
// then, read through pose_source as the ops read it, as `Output SelectAll`
// and the pose's words lay it around the grid's middle. A pose that trades
// rows for columns fits only a square grid.
inline optional<replayed> symmetric_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                          const symmetry& kept) {
    return paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        if (cut.count() != 1) return false;
        grid drawn(output.width, output.height);
        cut.pieces[0].each_cell([&](int row, int col, int8_t color) {
            if (drawn.contains(row, col)) drawn.data[row][col] = color;
        });
        for (const int8_t pose : kept.poses) {
            if (pose < 0) break;
            const auto [width, height] = posed_size(pose, output.width, output.height);
            if (width != output.width || height != output.height) return false;
            grid layer(width, height);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const auto [from_row, from_col] = pose_source(pose, output.width, output.height, row, col);
                    layer.data[row][col] = drawn.data[from_row][from_col];
                }
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    if (layer.data[row][col] != (int8_t)TRANSPARENT) {
                        drawn.data[row][col] = layer.data[row][col];
                        put(row, col, (int8_t)layer.data[row][col]);
                    }
        }
        return true;
    });
}

// A PIECE'S BOX FILLED, replayed. AddColor paints every cell of each picked
// piece's box in its one color -- laid under what is down (paint_holds), so it
// fills only what the piece and every paint before it leave empty.
inline optional<replayed> box_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                    const vector<cells>& canvas) {
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

// WHICH ROOM A ROOM PAINT FILLS (filled_room's value), in the order a tie goes:
// every room (d511f180's ground, all of it 8); by size, ranked among the rooms
// alone, as SelectLargestObject and SelectSmallestObject rank what is left once
// every other piece is deleted -- the largest, the smallest, the 2nd largest,
// the 2nd smallest... -- then the room of the part the picked pieces fence
// off, numbered as SelectPartition numbers the parts.
// Past those, every room at once, each in the color its own key names -- its
// shape or its symmetry (shape_key) -- as a table read off the pairs says
// (rooms_table).
constexpr int ROOM_EVERY = 0, ROOM_SIZES = 1, ROOM_PARTS = ROOM_SIZES + 2 * SIZE_RANKS,
              ROOM_CHOICES = ROOM_PARTS + row_of(node_type::SelectPartition).high + 1;
constexpr int ROOM_TABLE = ROOM_CHOICES, ROOM_VALUES = ROOM_TABLE + (int)shape_key::count;
constexpr bool room_by_size(int chosen) { return chosen >= ROOM_SIZES && chosen < ROOM_PARTS; }
constexpr bool room_largest(int chosen) { return (chosen - ROOM_SIZES) % 2 == 0; }
constexpr int room_rank(int chosen) { return (chosen - ROOM_SIZES) / 2; }
constexpr int room_part(int chosen) { return chosen - ROOM_PARTS; }

// A PART'S ROOM: the part it lies in, its cells, and whether another piece
// shares its part.
struct fenced_room {
    int part = -1;
    vector<int16_t> cells;
    bool shared = false;
};

// THE ROOMS THE PICKED PIECES FENCE OFF on each pair, in the order Partition
// hands them back (each_partition_piece, the op's own walk), on the canvas the
// pieces lie on, as Partition reads it.
inline vector<vector<fenced_room>> rooms_fenced(const replay_facts& facts, const vector<partner_pieces>& handed) {
    vector<vector<fenced_room>> rooms(handed.size());
    for (size_t which = 0; which < handed.size() && which < facts.train.size(); ++which) {
        const tile_size canvas = facts.canvas(which);
        const partner_pieces& pieces = handed[which];
        // Whether the part walked now holds a piece clipped to it, which
        // SelectPartition picks with its room.
        bool clipped = false;
        each_partition_piece(pieces, canvas.width, canvas.height,
            [&](int i, const cell_box& box, auto in_part) {
                const cell_box own = pieces.box(i);
                for (int row = max(box.row0, own.row0); row <= min(box.row1, own.row1) && !clipped; ++row)
                    for (int col = max(box.col0, own.col0); col <= min(box.col1, own.col1) && !clipped; ++col)
                        clipped = in_part(row, col) && pieces.painted(i, row, col);
            },
            [&](int part, const cell_box& box, auto empty) {
                fenced_room room{ part, {}, clipped };
                clipped = false;
                for (int row = box.row0; row <= box.row1; ++row)
                    for (int col = box.col0; col <= box.col1; ++col)
                        if (empty(row, col)) room.cells.push_back((int16_t)cell_index(row, col));
                if (!room.cells.empty()) rooms[which].push_back(std::move(room));
            });
    }
    return rooms;
}

// THE ROOM `chosen` NAMES among one pair's rooms (ROOM_PARTS), by size or by
// part, or none: the part's own room, where no other piece lies in that part;
// or, by size, the room at that rank, where no other room has its size, so the
// rank names it whatever order the op ranks ties in.
inline const fenced_room* room_chosen(const vector<fenced_room>& rooms, int chosen) {
    if (!room_by_size(chosen)) {
        const auto found = ranges::find(rooms, room_part(chosen), &fenced_room::part);
        return found == rooms.end() || found->shared ? nullptr : &*found;
    }
    const bool largest = room_largest(chosen);
    const size_t rank = (size_t)room_rank(chosen);
    if (rank >= rooms.size()) return nullptr;
    vector<const fenced_room*> ranked;
    for (const fenced_room& room : rooms) ranked.push_back(&room);
    ranges::stable_sort(ranked, [&](const fenced_room* one, const fenced_room* other) {
        return largest ? one->cells.size() > other->cells.size() : one->cells.size() < other->cells.size();
    });
    const size_t size = ranked[rank]->cells.size();
    const bool tied = (rank > 0 && ranked[rank - 1]->cells.size() == size) ||
                      (rank + 1 < ranked.size() && ranked[rank + 1]->cells.size() == size);
    return tied ? nullptr : ranked[rank];
}

// A ROOM FILLED, replayed. Partition hands back each part the picked pieces
// fence off, its empty cells as a room (rooms_fenced), and Recolor paints the
// room `chosen` names -- or every room -- in one color, laid the way `lay`
// says on what is down (paint_holds): over it, or under it, filling only what
// nothing has painted.
inline optional<replayed> room_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                     const vector<vector<fenced_room>>& rooms, const vector<cells>& canvas,
                                     int chosen, int8_t lay) {
    return paint_holds(facts, handed, { lay, &canvas, laying::in_one },
                       [&](size_t which, const partner_pieces&, auto&& put) {
        if (which >= rooms.size()) return false;
        auto fill = [&](const fenced_room& room) {
            for (const int16_t cell : room.cells) put(cell_row(cell), cell_col(cell), (int8_t)TRANSPARENT);
        };
        if (chosen == ROOM_EVERY) {
            ranges::for_each(rooms[which], fill);
            return !rooms[which].empty();
        }
        const fenced_room* room = room_chosen(rooms[which], chosen);
        if (!room) return false;
        fill(*room);
        return true;
    });
}

// A ROOM'S KEY of the kind `kind` names (shape_key_of), read as ShapeColorMap
// reads a room Partition hands back: its cells, empty but its own (BLANK), in
// the box they fill.
inline uint32_t room_key(const fenced_room& room, int kind) {
    cell_box box;
    bool in_room[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (const int16_t cell : room.cells) {
        box.include_cell(cell_row(cell), cell_col(cell));
        in_room[cell_row(cell)][cell_col(cell)] = true;
    }
    return shape_key_of(kind, box.width(), box.height(), [&](int row, int col) {
        return in_room[box.row0 + row][box.col0 + col] ? (int8_t)BLANK : (int8_t)TRANSPARENT;
    });
}

// THE ROOMS' TABLE, read off the pairs (read_off) as the shape word reads its
// pieces': each room's key of the kind `kind` names, and the one color the
// output shows over it. A room the output leaves empty stays as it is and
// says nothing. None where a room comes out in two colors, or none holds.
inline shape_color_map_rule rooms_table(const replay_facts& facts, const vector<vector<fenced_room>>& rooms,
                                        int kind) {
    vector<shape_color_map_rule::seen_key> seen;
    for (size_t which = 0; which < rooms.size() && which < facts.train.size(); ++which) {
        const grid& output = facts.train[which].output;
        const uint16_t shown = shape_color_map_rule::colors_shown(facts.train[which].input);
        for (const fenced_room& room : rooms[which]) {
            // -2 until a cell is read, -3 once two cells disagree.
            int worn = -2;
            for (const int16_t cell : room.cells) {
                const int there = output.data[cell_row(cell)][cell_col(cell)];
                worn = worn == -2 || worn == there ? there : -3;
            }
            if (worn == TRANSPARENT) continue;
            if (!is_color(worn) ||
                !shape_color_map_rule::see(seen, room_key(room, kind), (int8_t)worn, which, shown))
                return {};
        }
    }
    return shape_color_map_rule::read_off(seen);
}

// THE ROOMS, EACH IN THE COLOR ITS KEY NAMES, replayed: ShapeColorMap over
// every room Partition hands back (rooms_fenced), in the color `table` names
// for its key -- a room it names none for left as it is -- laid the way `lay`
// says on what is down (paint_holds).
inline optional<replayed> room_table_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                           const vector<vector<fenced_room>>& rooms, const vector<cells>& canvas,
                                           const shape_color_map_rule& table, int kind, int8_t lay) {
    return paint_holds(facts, handed, { lay, &canvas }, [&](size_t which, const partner_pieces&, auto&& put) {
        if (which >= rooms.size()) return false;
        for (const fenced_room& room : rooms[which]) {
            const int color = table.color_of(room_key(room, kind));
            if (color < 0) continue;
            for (const int16_t cell : room.cells) put(cell_row(cell), cell_col(cell), (int8_t)color);
        }
        return true;
    });
}

// BARS BETWEEN PIECES OF ONE COLOR, replayed. Bridge reaches from each picked
// piece to the picked pieces of its own color whose middles share its row or its
// column, the `way` it is told (bridge_bars, the walk the op takes), and fills
// the box between two -- laid under what is down (paint_holds), so a bar only
// fills what nothing has painted -- in the bar's own color, or all of them in one.
// Past those ways, the L from each picked piece to its partner (bridge_bars),
// SAID ONLY WHERE THE PAIRS SHOW IT TURN, in one pair at least: an L that never
// turns is a bar, and which way it turns would be a guess on a test, as a bar is
// said no wider than the pairs show it (833966f4's column of cells, where the
// top one ran straight down to the 6 in every pair, a coincidence).
inline optional<replayed> bridge_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       const vector<cells>& canvas, int way) {
    using rule = correspondence_election::fired_rule;
    bool turned = false;
    auto seen = paint_holds(facts, handed, { rule::laid_under, &canvas, laying::own_else_one },
                            [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        // The bars take one place more in the pieces' list, all of them one drawing.
        if (cut.count() >= MAX_OBJECTS_IN_A_LIST) return false;
        turned = bridge_bars(way, output.width, output.height, cut,
                             [&](const cell_box& bar, int8_t color) {
            for (int row = bar.row0; row <= bar.row1; ++row)
                for (int col = bar.col0; col <= bar.col1; ++col)
                    if (output.contains(row, col)) put(row, col, color);
            return true;
        }) || turned;
        return true;
    });
    if (way >= BRIDGE_TO_PARTNER && !turned) return nullopt;
    return seen;
}

// THE CELLS A PIECE ENCLOSES: those its own paint walls off from the edge of its
// box (mark_outside, the four-way flood the ops flood), each handed to
// `see(row, col)` where it lies.
template <class See>
inline void each_hollow(const object& piece, See see) {
    const int height = piece.summary.height, width = piece.summary.width;
    auto occupied = [&](int row, int col) { return piece.data[row][col] != (int8_t)TRANSPARENT; };
    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
    mark_outside(height, width, occupied, outside);
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            if (!occupied(row, col) && !outside[row][col]) see(piece.summary.row + row, piece.summary.col + col);
}

// THE HOLLOWS OF A PIECE, FILLED, replayed. InteriorColor fills each picked
// piece's hollows (each_hollow), in each piece's own color or all of them in
// one, and its sentence lays the whole piece under what is down (paint_holds):
// its own cells as they are -- a piece the line removed comes back with it
// (d5d6de2d's ring) -- and a marker sitting in a hollow keeps its own. In its
// own colors where they lay no cell wrong, else whichever says more, right
// cells less wrong ones, its own colors on a tie; in one, the color a paint in
// one color reads off the output where the hollows land (lay_paint).
inline optional<replayed> fill_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                     const vector<cells>& canvas) {
    using rule = correspondence_election::fired_rule;
    // Each picked piece, its own cells as they are and its hollows in `fill`,
    // or in its own color where `fill` is no color.
    auto filled = [&](int8_t fill) {
        return paint_holds(facts, handed, { rule::laid_under, &canvas },
                           [&](size_t, const partner_pieces& cut, auto&& put) {
            for (int i = 0; i < cut.count(); ++i)
                if (cut.picked[i]) {
                    const object& piece = cut.pieces[i];
                    piece.each_cell([&](int row, int col, int8_t color) { put(row, col, color); });
                    const int8_t worn = is_color(fill) ? fill : piece.first_color();
                    each_hollow(piece, [&](int row, int col) { put(row, col, worn); });
                }
            return true;
        });
    };
    auto own = filled((int8_t)TRANSPARENT);
    if (own && own->wrong_cells() == 0) return own;
    const auto hollows = paint_holds(facts, handed, { rule::laid_under, &canvas, laying::in_one },
                                     [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) each_hollow(cut.pieces[i], [&](int row, int col) { put(row, col, (int8_t)TRANSPARENT); });
        return true;
    });
    if (!hollows) return own;
    auto one = filled(hollows->paint);
    if (one) one->paint = hollows->paint;
    if (one && (!own || one->right_cells() - one->wrong_cells() > own->right_cells() - own->wrong_cells()))
        return one;
    return own;
}

// THE INSIDE OF THE PICKED PIECES, IN ONE COLOR, replayed: Outline hollows each
// picked piece to its edge (outlined), then InteriorColor fills what the edge
// encloses (each_hollow) -- the piece's core and any hollow it had -- in one
// color, the same in every pair, read off the output where the first lands.
// Laid over what is down, the edge as it was: bb43febb's blocks, red inside a
// gray rim; 694f12f3's, red inside the larger and blue inside the smaller.
inline optional<replayed> inside_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    int8_t inside = (int8_t)TRANSPARENT;
    for (size_t which = 0; which < handed.size() && inside == (int8_t)TRANSPARENT; ++which)
        for (int i = 0; i < handed[which].count() && inside == (int8_t)TRANSPARENT; ++i)
            if (handed[which].picked[i])
                each_hollow(outlined(handed[which].pieces[i]), [&](int row, int col) {
                    if (inside == (int8_t)TRANSPARENT && facts.train[which].output.contains(row, col))
                        inside = (int8_t)facts.train[which].output.data[row][col];
                });
    if (!is_color(inside)) return nullopt;
    auto seen = paint_holds(facts, handed, {}, [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object rim = outlined(cut.pieces[i]);
            rim.each_cell([&](int row, int col, int8_t color) { put(row, col, color); });
            each_hollow(rim, [&](int row, int col) { put(row, col, inside); });
        }
        return true;
    });
    if (seen) seen->paint = inside;
    return seen;
}

// THE HOLLOWS, FILLED BY SHAPE, replayed. ShapeToInteriorColorMap takes the
// pieces as the line hands them over, looks each one's shape up in the learned
// table, and fills its hollows (each_hollow) with that color -- laid under the
// hand, so a marker already sitting in a hollow keeps its own (paint_holds).
// Sees the cells it fills and how many pieces it filled.
inline optional<replayed> interior_fills_hold(const replay_facts& facts, const shape_color_map_rule& table) {
    using rule = correspondence_election::fired_rule;
    const vector<cells> hand = pieces_down(facts);
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
        piece.summary.row = (int8_t)(piece.summary.row + down);
        piece.summary.col = (int8_t)(piece.summary.col + across);
    };
    for (size_t which = 0; which < facts.cuts().size(); ++which) {
        const auto changing = facts.change(which);
        vector<object>& hand = changing.hand;
        const example& pair = facts.train[which];
        const vector<bool> picked = facts.picked(which, pick);
        switch (rule.as) {
        case said::turn:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i] = posed(hand[i], rule.value);
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
        case said::swap_colors:
        case said::key_swap: {
            // What trades places, read off the hand as it is before any piece
            // changes: a fixed pair, or the pairs the key the pick leaves names.
            int8_t to[COLOR_COUNT];
            swaps_of(rule.value, facts.handed(which, pick), to);
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i].repaint([&](int8_t now) { return to[now] >= 0 ? to[now] : now; });
            break;
        }
        case said::empty_swap:
            // Every cell of each picked piece's box comes out as the op's own
            // rule says (traded_with_empty), the box kept.
            for (size_t i = 0; i < hand.size(); ++i) {
                if (!picked[i]) continue;
                object& piece = hand[i];
                int count = 0;
                for (int row = 0; row < piece.summary.height; ++row)
                    for (int col = 0; col < piece.summary.width; ++col) {
                        int8_t& cell = piece.data[row][col];
                        cell = traded_with_empty(cell, rule.value - EMPTY_PAIRS);
                        count += cell != (int8_t)TRANSPARENT;
                    }
                piece.summary.count = (int16_t)count;
            }
            break;
        case said::color_from:
        case said::align_to:
        case said::move_toward: {
            // Every partner is found on the hand as it was, before any piece
            // changes; a piece with no one partner stays as it is.
            const partner_pieces cut = facts.handed(which, pick);
            if (rule.as == said::color_from && rule.value >= COLORS_FROM_ITS_SHAPE) {
                // Cell by cell from a piece of its shape, every cell read off the
                // hand as it was and laid on a copy of it.
                vector<object> laid = hand;
                for (int i = 0; i < cut.count(); ++i)
                    if (cut.picked[i])
                        colors_from_its_shape(rule.value - COLORS_FROM_ITS_SHAPE, i, cut, [&](int row, int col, int color) {
                            object& piece = laid[(size_t)i];
                            if (is_color(color)) piece.data[row - piece.summary.row][col - piece.summary.col] = (int8_t)color;
                        });
                hand = std::move(laid);
                break;
            }
            constexpr int ways = (int)partner_by::count;
            vector<int> found(hand.size(), -1);
            vector<direction_step> by(hand.size(), direction_step{ 0, 0 });
            const bool by_place = rule.as == said::color_from && rule.value == KEY_BY_PLACE;
            const vector<int8_t> places = by_place ? cut.key_places() : vector<int8_t>{};
            for (int i = 0; i < cut.count(); ++i) {
                if (!cut.picked[i]) continue;
                if (by_place) found[i] = places[(size_t)i];
                else if (rule.as == said::color_from) found[i] = cut.color_from((partner_by)rule.value, i);
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
            const direction_step step = step_of(rule.value);
            for (size_t i = 0; i < hand.size(); ++i) {
                if (!picked[i]) continue;
                const cell_box box = hand[i].painted_box();
                if (rule.as == said::to_edge) {
                    const int steps = steps_to_edge(box, pair.output.width, pair.output.height, rule.value);
                    moved(hand[i], steps * step.down, steps * step.across);
                } else {
                    const direction_step by = shift_by(rule.value, box.width(), box.height());
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
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i] = outlined(hand[i]);
            break;
        case said::mirror_around: {
            const partner_pieces cut = facts.handed(which, pick);
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
        case rule::color_from: case rule::recolor_picked: case rule::shape_recolor: case rule::key_swap:
            return word_group::recolors;
        // Acts: pieces move, turn, change shape, or gain copies. Learned-table
        // compositions keep the old gate: their hand cannot yet be carried.
        case rule::turn: case rule::move_down: case rule::move_across:
        case rule::learned_table: case rule::align_to: case rule::mirror_around:
        case rule::outlined: case rule::to_edge: case rule::shifted: case rule::move_toward:
        case rule::empty_swap:
            return word_group::changes_pieces;
        case rule::removed: return word_group::removes;
        // Paints: the hand on the line is left alone.
        case rule::extend_rays: case rule::bridged: case rule::filled: case rule::trailed:
        case rule::boxed: case rule::filled_room: case rule::copied_onto: case rule::tiled_along:
        case rule::turned_copy: case rule::symmetric:
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

// Rays keep their length below 32, plus 32 for the diagonals. This is their
// stored value, not the argument written in the DSL.
struct paint_value {
    static constexpr int second_bit = 32;
    static constexpr int packed(bool second, int number) { return (second ? second_bit : 0) + number; }
    static constexpr bool second(int stored) { return stored >= second_bit; }
    static constexpr int number(int stored) { return stored % second_bit; }
};

// The directions' names, as the reading says them in words.
inline constexpr const char* const direction_names[direction::count] = {
    "up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left" };

// The one pose that does `first` and then `then`. Read off where the corners of
// a square land, since the eight ways round are exactly the ways to move them.
inline int compose_poses(int first, int then) {
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        bool same = true;
        for (int corner = 0; corner < 4 && same; ++corner) {
            const auto [row, col] = posed_place(first, 2, 2, corner / 2, corner % 2);
            same = posed_place(then, 2, 2, row, col) == posed_place(pose, 2, 2, corner / 2, corner % 2);
        }
        if (same) return pose;
    }
    return 0;
}

// THE CUT'S PAINTS, READ ONCE FOR EVERY LINE READ ON IT: what a paint reads
// is the cut opened again, the same for each line read on one cut -- with its
// acts or the words alone, plainly or reading ahead (read_the_cut) -- so the
// cut is opened once, the pieces each draw pick takes off it handed over once,
// and each way of asking a paint -- the kind, the pick, the value -- walked
// once (paint_walk) for all of them. A walk is counted once for each canvas
// it is laid on (walk_tally), each canvas of every line its own number.
// `asked_picks` are the draw picks a paint asks, in order: each takes, in some
// pair, pieces no pick before it takes (takes_the_same); `asked_partner_picks`
// the partner picks past them, which a line asks last (partner_picks). The
// rays of each pick, one way round, are fanned out once for every length
// (rays_fan).
struct walked_paint { optional<paint_walk> walk; int counted_on = -1; walk_tally tally; };
struct cut_paints {
    optional<replay_facts> opened;
    vector<piece_pick> draw_picks;
    vector<vector<partner_pieces>> handed_by_pick;
    vector<vector<uint16_t>> colors_by_pick;
    // The draw picks asked, and past them the partner picks a line asks last
    // (partner_picks): indices into draw_picks.
    vector<size_t> asked_picks, asked_partner_picks;
    size_t plain_draws = 0;
    std::map<std::tuple<int, size_t, int>, walked_paint> walks;
    std::map<std::pair<size_t, bool>, optional<ray_fan>> fans;
    int canvases = 0;
};

// THE REST OF THE ELECTED CUT'S LINE, after its own acts: a rule read off the
// cut's own pieces that drew every output by itself, or else the words its
// recolor is said with and what it has no word for. Listed on the reading under
// the rule whose evidence each one reads, and put on the line in the order it is
// said. Reading ahead, a paint is weighed with what the picture lets be read
// right after it (read_figures asks both ways).
inline void finish_the_line(inference& inf, const vector<example>& train, const tiled_pieces& boundaries,
                            bool read_ahead, cut_paints& paints) {
    ++inference_work.lines;
    const time_into clock{ inference_time.ms[inf.pass > 1 ? inference_times::lines_later : inference_times::lines_first] };
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
    // A pass after the first says one change in one sentence, and these spell
    // programs of their own, some reading the input again; the first pass alone
    // asks them.
    if (inf.pass == 1 &&
        (whole_line(inf.diagonal_continuation.holds(), rule::diagonal_tile, "Diagonal Tile",
                   "the colors cycle along the diagonals") ||
        whole_line(inf.piece_slam.holds(), rule::piece_slam, "Slam", "each piece piled up inside its own box") ||
        whole_line(inf.overlap_corner.holds(), rule::corner_chain, "Corner Chain",
                   "each piece hung off the last one's corner") ||
        whole_line(inf.slide_to_touch.holds(), rule::slide_to_touch, "Slide To Touch",
                   "pieces slid up against the ones that stay") ||
        whole_line(inf.project_on_seeds.holds(), rule::project_on_seeds, "Project On Seeds",
                   "the template stamped along every marker") ||
        whole_line(inf.snap_to_anchors.holds(), rule::snap_to_anchors, "Snap To Anchors",
                   "each body turned and set down on the anchors its keys match")))
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
    // already on the line, the hand the words asked below are replayed on, laid
    // under what is down where a piece sits on another, as the line is said.
    replay_facts facts(inf, train, boundaries);
    picks_on_hand hand_picks;
    facts.laid_under = ranges::any_of(read.pairs, [](const auto& pair) { return pair.sits_on > 0; });
    for (const int8_t which : read.line)
        facts.hand_known = facts.hand_known && carry_the_hand(read.fired[which], facts);
    // Carries the hand through a word just said on the line. Once it cannot,
    // the line no longer draws its picture, and no word after lays a cell wrong.
    auto carry = [&](int8_t which) {
        hand_picks.clear();
        facts.hand_known = facts.hand_known && carry_the_hand(read.fired[which], facts);
        facts.tolerant = facts.tolerant && facts.hand_known;
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
        "the piece it touches", "the piece of its shape", "the piece of its shape, turned",
        "the piece of its shape at any size", "the piece of its shape at any size, turned",
        "the piece of its shape stretched", "the piece of its shape stretched, turned" };
    static_assert(size(partners) == (size_t)partner_by::count);
    // How a piece is like the key (SelectLikeKey), each likeness of shape in turn.
    static constexpr const char* likenesses[] = { "", " Any Way Round", " At Any Size", " At Any Size, Any Way Round",
                                                  " Stretched", " Stretched, Any Way Round" };
    static_assert(size(likenesses) == (size_t)KEY_LIKENESSES);
    // A partner of any kind (partner_of_kind) by name: each way partner_by finds
    // one, then the piece of each color.
    auto partner_named = [](int kind) {
        return kind < (int)partner_by::count ? string(partners[kind])
                                             : "the piece of color " + to_string(kind - (int)partner_by::count);
    };
    static constexpr const char* mirrors[] = { "left to right", "top to bottom", "all four ways" };
    static_assert(size(mirrors) == (size_t)mirror_by::count);
    // Each kind of pick by name, then turned round, every piece but those.
    static constexpr const char* pick_names[][2] = { { "Holes", "" }, { "Largest", "" }, { "Smallest", "" },
        { "Row Band", "" }, { "Column Band", "" }, { "Most Common Color", "Not Most Common Color" },
        { "Cells At Least", "Cells Fewer Than" }, { "The Largest", "All But The Largest" },
        { "The Smallest", "All But The Smallest" }, { "Row From The Middle", "Off The Row From The Middle" },
        { "Column From The Middle", "Off The Column From The Middle" },
        { "The One With No Twin By", "All But The One With No Twin By" } };
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
        const marked_paint* marked = nullptr;
        heard() = default;
        heard(int value, int said, const piece_pick& pick, const replayed& seen, claim claims)
            : value(value), said(said), pick(pick), seen(seen), claims(std::move(claims)) {}
        heard(int value, int said, const piece_pick& pick, const marked_paint& seen, claim claims)
            : value(value), said(said), pick(pick), claims(std::move(claims)), marked(&seen) {}
        // Only the best lay of a kind needs lists, for saying it or reading
        // ahead on its drawing. Every candidate used the same lay_paint judge.
        bool materialize(const replay_facts& facts) {
            if (value < 0) return false;
            if (marked) {
                seen = *lay_paint(facts, *marked->walk, marked->how);
                marked = nullptr;
            }
            return true;
        }
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
    // What a paint said after the line reads: the cut opened again, as it is said,
    // once for every line read on the cut (cut_paints). It lays cells wrong where
    // the words may (replay_facts::tolerant).
    const bool cut_opened = paints.opened.has_value();
    if (!cut_opened) paints.opened.emplace(inf, train, boundaries);
    replay_facts& opened = *paints.opened;
    opened.tolerant = false;
    // THE CELLS A WORD SAID LAID WRONG, every word's: where the picture the line
    // draws still shows them wrong at its end, they are left unsaid.
    colors_by_pair laid_wrong(facts.train.size());
    // The colors of the input cells nothing claims yet that `says` takes, a bit each.
    auto colors_left = [&](auto says) {
        uint16_t found = 0;
        for (const auto& pair : read.pairs)
            for (const auto& cell : pair.left_over)
                if (cell.claimed_by < 0 && cell.from_input && says(cell))
                    found = (uint16_t)(found | 1u << cell.color);
        return found;
    };
    // The colors an unsaid change of `word` starts from.
    auto starts_from = [&](missing word) {
        return colors_left([word](const auto& cell) { return cell.answers_to(word); });
    };
    // Keeps `value` when it accounts for more cells nothing claims yet than the
    // best so far, less the cells it lays wrong, with what its replay saw. A
    // finished pair (replay_facts::finished) lets every word off, so none is
    // credited with a cell there: the cells left over in it are the line's to
    // draw (cd3c21df's framed pieces, each already its output).
    auto weigh = [&](heard& best, int value, const piece_pick& pick, const auto& seen, claim claims) {
        if (facts.any_finished())
            claims = [&facts, claims = std::move(claims)](size_t which, const left_over_cell& cell) {
                return !facts.finished(which) && claims(which, cell);
            };
        int said = -seen.wrong_cells();
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
            const bool by_count = (best.pick.how == piece_pick::largest || best.pick.how == piece_pick::smallest)
                                      ? best.pick.value >= SIZE_RANKS
                                      : best.pick.how == piece_pick::only_one && best.pick.value >= (int)twin_by::by_count;
            if (by_count) {
                const int kind = best.pick.value - (best.pick.how == piece_pick::only_one ? (int)twin_by::by_count : SIZE_RANKS);
                name += string(best.pick.how == piece_pick::smallest ? " Fewest " : best.pick.how == piece_pick::largest
                                                                                         ? " Most " : " Count Of ") +
                        (kind == (int)count_kind::cells ? "Cells" : kind == (int)count_kind::colors ? "Colors"
                         : kind == (int)count_kind::holes ? "Holes"
                         : "Cells Of Color " + to_string(kind - (int)count_kind::of_color));
            } else if (best.pick.how != piece_pick::most_common) name += " " + to_string(best.pick.value);
        }
        else if (best.pick.partner >= 0) {
            // SelectPartner: the partners its base's pieces have, on a side or any.
            const int partner_direction = partner_direction_of(best.pick.partner);
            name = best.pick.but_those ? "All But " : "";
            if (best.pick.colors == piece_pick::all().colors) name += "Every Piece's";
            else {
                name += "Colors";
                for (int color = 0; color < COLOR_COUNT; ++color)
                    if (best.pick.colors >> color & 1) name += " " + to_string(color);
                name += "'";
            }
            name += string(" Partners ") + (partner_direction < 0 ? "Any Side" : direction_names[partner_direction]) + " (" +
                    partners[(int)partner_way_of(best.pick.partner)] + ")";
        }
        else {
            name = best.pick.part < 0 ? "Colors"
                 : string(best.pick.but_those ? "Off Part " : "Part ") + to_string(best.pick.part) + " Fenced By";
            for (int color = 0; color < COLOR_COUNT; ++color)
                if (best.pick.colors >> color & 1) name += " " + to_string(color);
        }
        if (!every_piece && best.pick.fits >= 0)
            name += string(" That Would Fit ") + partners[best.pick.fits];
        if (!every_piece && best.pick.like >= 0)
            name += string(" Shaped Like The Key") + likenesses[best.pick.like];
        rule one = listed(name.c_str(), says.c_str(), as, (int8_t)best.value);
        one.pick = every_piece ? piece_pick::all() : best.pick;
        one.pieces = best.seen.pieces;
        one.paint = best.seen.paint;
        for (size_t which = 0; which < best.seen.wrong.size() && which < laid_wrong.size(); ++which)
            laid_wrong[which].insert(laid_wrong[which].end(), best.seen.wrong[which].begin(),
                                     best.seen.wrong[which].end());
        return read.say_fired(std::move(one), under, best.claims);
    };
    const int8_t moves = read.fired_named(degrees_of_freedom::freedom_names[0]);
    constexpr int ways = (int)partner_by::count;
    // Claims for a move: the cells the pieces left, on the input, and landed on,
    // on both grids -- an input cell a landing covers is one the move drew over
    // (18286ef8's 9 stepping onto the 5 beside it).
    auto move_claims = [](const replayed& seen) -> claim {
        return [left = cell_marks(seen.left), landed = cell_marks(seen.drawn)](size_t which, const left_over_cell& cell) {
            return cell.answers_to(missing::where_it_went) &&
                   (on_cells(landed, which, cell.row, cell.col) ||
                    (cell.from_input && on_cells(left, which, cell.row, cell.col)));
        };
    };
    // Claims for a recolor: the cells it repainted that wait on one.
    auto recolor_claims = [](const replayed& seen) -> claim {
        return [recolored = cell_marks(seen.drawn)](size_t which, const left_over_cell& cell) {
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
    // What a kind offers: how many cells it says, how to say it, and for a
    // paint, the cells it would lay, in what it lays there.
    struct offer { int said = 0; std::function<void()> say; colors_by_pair drawn; };

    // A RECOLOR ANOTHER PIECE DECIDES: every picked piece takes its partner's color
    // -- or, past the partner ways, the color a key drawn in the grid shows at its
    // place (key_place_holds). It accounts for the cells it recolored.
    auto color_from_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            for (int how = 0; how <= row_of(node_type::ColorFrom).high; ++how)
                if (how >= COLORS_FROM_ITS_SHAPE ? !facts.asks_likeness(how - COLORS_FROM_ITS_SHAPE)
                                                 : how < KEY_BY_PLACE && !facts.asks_way((partner_by)how))
                    continue;
                else if (const auto seen = how >= COLORS_FROM_ITS_SHAPE
                                          ? colors_from_its_shape_holds(facts, handed, how - COLORS_FROM_ITS_SHAPE)
                                      : how == KEY_BY_PLACE ? key_place_holds(facts, handed)
                                                            : color_from_holds(facts, handed, (partner_by)how))
                    weigh(best, how, pick, *seen, recolor_claims(*seen));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            carry(say(best, rule::color_from,
                      best.value >= COLORS_FROM_ITS_SHAPE
                          ? string("takes its colors cell by cell from ") +
                                partners[(int)partner_by::same_shape + best.value - COLORS_FROM_ITS_SHAPE]
                          : string("takes the color of ") +
                                (best.value == KEY_BY_PLACE ? "the key at its place" : partners[best.value]),
                      colors)); } };
    };

    // A MOVE THE PIECES DECIDE: every picked piece slides level with its partner,
    // into its rows or its columns; slides to an edge of the grid; moves its own
    // size one of eight ways; or, picked by anything but its colors alone -- holes,
    // size, band, part, partner -- the same step in every pair, which a pick by
    // color already says as an act; or moves toward
    // its partner. Of the kinds, the one that accounts for the most is said, a tie
    // going in that order -- a move toward a partner, the newest word, after every
    // older one. It accounts for the cells each piece left and the cells it landed
    // on.
    static_assert(direction::count == row_of(node_type::MoveToEdge).high + 1 &&
                  direction::count == row_of(node_type::Shift).high + 1);
    static constexpr const char* how_far[] = { "one step", "its own size", "until it touches" };
    static_assert(size(how_far) == (size_t)toward_by::count);
    auto move_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::where_it_went);
        if (!facts.hand_known || !from) return {};
        heard aligned, edged, sized, stepped, toward;
        asked_piece_sets asked, axis_steps, diagonal_steps;
        int best_down = 0, best_across = 0;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (!asked.before(handed)) {
                for (int value = 0; value < 2 * ways; ++value)
                    if (!facts.asks_way((partner_by)(value % ways))) continue;
                    else if (const auto seen = align_holds(facts, handed, value >= ways, (partner_by)(value % ways)))
                        weigh(aligned, value, pick, *seen, move_claims(*seen));
                for (int value = 0; value <= row_of(node_type::MoveToward).high; ++value)
                    if (!facts.asks_kind(value % PARTNER_KINDS)) continue;
                    else if (const auto seen = toward_holds(facts, handed, value))
                        weigh(toward, value, pick, *seen, move_claims(*seen));
                for (int at = 0; at < direction::count; ++at) {
                    const int move_direction = direction_asked(at);
                    if (const auto seen = to_edge_holds(facts, handed, move_direction)) weigh(edged, move_direction, pick, *seen, move_claims(*seen));
                    if (const auto seen = shift_holds(facts, handed, move_direction)) weigh(sized, move_direction, pick, *seen, move_claims(*seen));
                }
            }
            if (pick.by_color()) continue;
            const bool axis_before = axis_steps.before(handed);
            const bool diagonal_before = pick.partner >= 0 || diagonal_steps.before(handed);
            if (axis_before && diagonal_before) continue;
            for (const auto& [down, across] : offsets_to_ask(facts, handed)) {
                // A step both ways is said as two sentences, the second picking
                // again on the hand the first moved: a pick that reads a partner
                // may take other pieces there, so it steps one way only.
                if (down != 0 && across != 0 ? diagonal_before : axis_before) continue;
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
                carry(say(edged, rule::to_edge, string("slides ") + direction_names[edged.value] + " to the edge", moves));
            else if (sized.said == most)
                carry(say(sized, rule::shifted, string("moves its own size ") + direction_names[sized.value], moves));
            else if (stepped.said < most) {
                const int kind = toward.value % PARTNER_KINDS;
                carry(say(toward, rule::move_toward,
                          string("moves ") + how_far[toward.value / PARTNER_KINDS] + " toward " + partner_named(kind),
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
    // that the copies land on. All four ways is asked first, so it takes a tie:
    // it holds only where both its halves hold, every copy borne out, and one way
    // alone that accounts for as much says only as much as the pairs' own gaps
    // needed -- as the longer ray takes a tie (b8825c91's test hides a patch on
    // its middle column, which only the copy top to bottom restores).
    static constexpr mirror_by mirrors_asked[] = { mirror_by::four_ways, mirror_by::left_right,
                                                   mirror_by::top_bottom };
    static_assert(size(mirrors_asked) == (size_t)mirror_by::count);
    // Claims for a copy: the cells waiting on a copy or a turn where what it
    // copies lies (`kept`) or its copies land, and the cells no match covered
    // that its copies land on.
    auto copy_claims = [](cells_by_pair kept, cells_by_pair landed) -> claim {
        return [kept = cell_marks(kept), landed = cell_marks(landed)](size_t which, const left_over_cell& cell) {
            const bool mine = on_cells(kept, which, cell.row, cell.col) || on_cells(landed, which, cell.row, cell.col);
            if (cell.waits_for(missing::where_it_was_copied) || cell.waits_for(missing::how_it_turned))
                return mine;
            return cell.waits < 0 && !cell.from_input && on_cells(landed, which, cell.row, cell.col);
        };
    };
    auto mirror_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::where_it_was_copied);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            for (const mirror_by mirror : mirrors_asked)
                for (int how = 0; how < MIRROR_PARTNERS; ++how) {
                    if (how < (int)partner_by::count && !facts.asks_way((partner_by)how)) continue;
                    const int value = (int)mirror * MIRROR_PARTNERS + how;
                    const auto seen = mirror_holds(facts, handed, mirror, how);
                    if (!seen) continue;
                    weigh(best, value, pick, *seen, copy_claims(seen->left, seen->drawn));
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
    // accounts for the cells those pieces keep and drop that wait on a change of
    // shape or on no word at all, and it is asked wherever such a cell is left:
    // asked only for a change of shape, it never saw 4347f46a, whose rectangles'
    // cores are simply gone, with no word named for them.
    auto outline_word = [&]() -> offer {
        auto says = [](const left_over_cell& cell) {
            return cell.waits < 0 || cell.waits_for(missing::how_it_changed_shape); };
        const uint16_t from = colors_left(says);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const auto handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            const auto seen = outline_holds(facts, handed);
            if (!seen) continue;
            weigh(best, 0, pick, *seen, [says, kept = cell_marks(seen->drawn), dropped = cell_marks(seen->left)](size_t which,
                                                                                         const left_over_cell& cell) {
                return says(cell) &&
                       (on_cells(kept, which, cell.row, cell.col) || on_cells(dropped, which, cell.row, cell.col));
            });
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            carry(say(best, rule::outlined, "keeps only its outline",
                      read.fired_named(correspondence_election::sits_on_name))); } };
    };

    // A RECOLOR A PICK DECIDES: the pieces it picks all come out in one color --
    // the pick, then Recolor. The colors an unsaid recolor starts from are asked
    // too, alone and with a fit: an act says a recolor by color only where the
    // line opens with the acts, and the words alone say it themselves (18286ef8's
    // 6 becomes 9 once its 9 has stepped). Asked of the hand the line leaves --
    // or, where the hand could not be carried, only of a line nothing on it moves
    // or reshapes a piece, with no pick by color, which a recolor on it changes,
    // and the color most cells wear, which a recolor changes, only of a bare line.
    auto recolor_word = [&]() -> offer {
        if (!facts.hand_known && ranges::any_of(read.line, [&](int8_t which) {
                return group_of(read.fired[which].as) == word_group::changes_pieces; }))
            return {};
        heard best;
        array<asked_piece_sets, COLOR_COUNT> asked;
        const uint16_t from = starts_from(missing::a_recolor);
        for (const piece_pick& pick : hand_picks.get(facts, facts.hand_known ? from : 0, from)) {
            if (pick.how == piece_pick::most_common &&
                !facts.hand_known && !read.line.empty())
                continue;
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int8_t color = 0; color < COLOR_COUNT; ++color) {
                if (already_said(rule::recolor_picked, pick, color)) continue;
                if (asked[(size_t)color].before(handed)) continue;
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
    // every key it does not name. The puzzle has one table, the first pass's
    // (read_the_cut), so this is asked only there, while nothing else holds it.
    // It accounts for the cells it recolored.
    static constexpr const char* keys_named[] = { "shape", "symmetry" };
    static_assert(size(keys_named) == (size_t)shape_key::count);
    auto shape_word = [&]() -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (inf.pass > 1 || !facts.hand_known || !from || !facts.overlays || inf.shape_color_map.holds()) return {};
        heard best;
        shape_color_map_rule best_table;
        asked_piece_sets asked;
        vector<uint16_t> shown;
        for (const example& pair : facts.train) shown.push_back(shape_color_map_rule::colors_shown(pair.input));
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
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

    // BidiColorMap PAST ITS FIXED PAIRS. A RECOLOR A KEY DECIDES: the colors of
    // every picked piece trade places as a key drawn in the grid pairs them --
    // the pieces the pick leaves, read along their rows or their columns
    // (key_swaps). Each grid brings its own key, so no swap fixed for every pair
    // says it (0becf7df's corner: 1 and 3, 2 and 8, then 4 and 2, 3 and 7). It
    // accounts for the cells it recolored. OR A COLOR TRADING PLACES WITH EMPTY
    // over each picked piece's box (empty_swap_holds), a color an unsaid recolor
    // starts from (d511f180's 8s and the ground the plan deleted). It accounts for
    // the cells it emptied and the ones it drew where the box was empty.
    static constexpr const char* keys_read[] = { "rows", "columns" };
    static_assert(size(keys_read) == (size_t)(KEY_PAIRS_ALONG_COLUMNS - KEY_PAIRS_ALONG_ROWS + 1) &&
                  EMPTY_PAIRS + COLOR_COUNT - 1 == row_of(node_type::BidiColorMap).high);
    auto swap_claims = [](const replayed& seen) -> claim {
        return [changed = cell_marks(seen.drawn)](size_t which, const left_over_cell& cell) {
            return (cell.answers_to(missing::a_recolor) || !cell.from_input) && on_cells(changed, which, cell.row, cell.col);
        };
    };
    auto key_word = [&](bool changes_allowed) -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            for (int value = KEY_PAIRS_ALONG_ROWS; value <= KEY_PAIRS_ALONG_COLUMNS; ++value)
                if (const auto seen = key_swap_holds(facts, handed, value))
                    weigh(best, value, pick, *seen, recolor_claims(*seen));
            for (int color = 0; color < COLOR_COUNT; ++color)
                if (changes_allowed && (from >> color & 1))
                    if (const auto seen = empty_swap_holds(facts, handed, color))
                        weigh(best, EMPTY_PAIRS + color, pick, *seen, swap_claims(*seen));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            if (best.value >= EMPTY_PAIRS)
                carry(say(best, rule::empty_swap,
                          "its " + to_string(best.value - EMPTY_PAIRS) + "s and the empty cells of its box trade places",
                          colors));
            else
                carry(say(best, rule::key_swap,
                          string("trades colors as the key's ") + keys_read[best.value - KEY_PAIRS_ALONG_ROWS] +
                              " pair them", colors)); } };
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
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            bool may = true;
            if (facts.hand_known)
                may = true;
            else if (pick.how == piece_pick::most_common)
                may = read.line.empty();
            else if (!pick.by_color())
                may = ranges::none_of(read.line, [&](int8_t which) {
                    const auto group = group_of(read.fired[which].as);
                    return group == word_group::changes_pieces || group == word_group::removes; });
            else
                for (int color = 0; color < COLOR_COUNT && may; ++color)
                    may = !(pick.colors >> color & 1) ||
                          ranges::none_of(read.line, [&](int8_t which) { return changes(read.fired[which], color); });
            if (!may) continue;
            if (already_said(rule::removed, pick, 0)) continue;
            const auto handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            const auto seen = removal_holds(facts, handed);
            if (!seen || seen->every_piece) continue;
            weigh(best, 0, pick, *seen, [gone = cell_marks(seen->left)](size_t which, const left_over_cell& cell) {
                return cell.from_input && on_cells(gone, which, cell.row, cell.col);
            });
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { carry(say(best, rule::removed, "is removed", -1)); } };
    };

    // The picks a draw word is asked with: each color on its own, then the picks by
    // holes, size and band.
    vector<piece_pick>& draw_picks = paints.draw_picks;
    if (!cut_opened) {
        for (int color = 0; color < COLOR_COUNT; ++color) draw_picks.push_back(piece_pick::of_color(color));
        for (const piece_pick& pick : picks_to_ask(opened, 0)) draw_picks.push_back(pick);
        paints.plain_draws = draw_picks.size();
        for (const piece_pick& pick : partner_picks(opened)) draw_picks.push_back(pick);
    }

    // WHAT A PAINT SAID AFTER THE LINE READS: it opens the cut again, so it reads
    // the pieces as the cut hands them over, as its replay does, and paints only
    // what it draws. Laid over what is down -- rays, trails, copies, repeats -- it needs
    // nothing of the line at all. Laid under it -- bars, filled hollows, filled
    // boxes -- it needs to know what the line covers: the cut's pieces, where
    // nothing on the line moves, reshapes or removes one, or else the hand the
    // line leaves, where it was carried. Each accounts for the cells it draws, and
    // is asked for every pick that accounts for any.
    auto drawn_claims = [](const cells_by_pair& drawn) -> claim {
        return [drawn = cell_marks(drawn)](size_t which, const left_over_cell& cell) {
            return !cell.from_input && on_cells(drawn, which, cell.row, cell.col);
        };
    };
    // Laid over what is down, a paint explains what it covers as well: an input
    // cell under it came out in the paint's color, its replay saw to that, so a
    // piece under 4f537728's repeated marker needs no recolor of its own.
    auto covered_claims = [](const auto& drawn) -> claim {
        if constexpr (std::is_same_v<std::decay_t<decltype(drawn)>, cell_marks>)
            // These marks live in the line's cache, through weighing and saying.
            return [&drawn](size_t which, const left_over_cell& cell) {
                return on_cells(drawn, which, cell.row, cell.col);
            };
        else return [drawn = cell_marks(drawn)](size_t which, const left_over_cell& cell) {
            return on_cells(drawn, which, cell.row, cell.col);
        };
    };
    // Whether the line leaves every piece where the cut finds it: none of its
    // words moves, reshapes or removes one, though it may recolor them.
    auto still = [&] {
        return ranges::all_of(read.line, [&](int8_t which) {
            const auto group = group_of(read.fired[which].as);
            return group == word_group::recolors || group == word_group::paints; });
    };
    // Whether what is down is known: the cut's pieces, where a still line leaves
    // them where they lie, or the hand the words carried.
    auto down_known = [&] { return still() || facts.hand_known; };
    // WHAT IS DOWN WHERE A PAINT LANDS: the pieces as the line leaves them so far
    // -- the cut's, where the line is still, else the hand the words carried
    // (18286ef8's 5s paint, laid under, the hollow their 9 stepped off) -- and
    // every paint said before it, each cell in the color it shows: in the colors
    // the line draws, where it draws its picture, and what each paint laid
    // there. Laid under it or only onto it, a paint reads it, so those ways are
    // asked only where it is known; laid over it, a paint needs nothing of it.
    // Laid again before every word is asked, and counted each time it changes.
    vector<cells> canvas(opened.cuts().size());
    // WHAT THE PAINTS SAID LAY, in the order said: each cell in what it laid
    // there, right or wrong.
    colors_by_pair painted(opened.cuts().size());
    // How often it changed, and how often where something is; and its number
    // among every canvas a line on the cut laid (cut_paints).
    int canvases = 0, occupancies = 0, canvas_number = 0;
    // WHAT THE LINE DRAWS SO FAR, CELL BY CELL: the hand as the words leave it,
    // laid as its sentence lays it -- over what is down, or under it where a
    // piece sits on another (draw_the_hand) -- pair by pair, where the words
    // carried the hand, on pairs that lie over their outputs; and whether it
    // shows a cell's place as the output shows it, which the line then says
    // itself, however the matching paired the pieces.
    vector<grid> line_drawn;
    auto draw_the_line = [&] {
        line_drawn.clear();
        if (!facts.hand_known || !facts.overlays) return;
        for (size_t which = 0; which < facts.cuts().size(); ++which) {
            const grid& output = facts.train[which].output;
            grid drawn(output.width, output.height);
            draw_the_hand(drawn, facts.cuts()[which], facts.laid_under, /*clip=*/true,
                          [](size_t) { return direction_step{ 0, 0 }; });
            line_drawn.push_back(std::move(drawn));
        }
    };
    auto line_shows = [&](size_t which, const left_over_cell& cell) {
        return which < line_drawn.size() && line_drawn[which].contains(cell.row, cell.col) &&
               line_drawn[which].data[cell.row][cell.col] == facts.train[which].output.data[cell.row][cell.col];
    };
    auto lay_the_canvas = [&] {
        draw_the_line();
        vector<cells> now = pieces_down(still() ? opened : facts);
        for (size_t which = 0; which < now.size() && which < line_drawn.size(); ++which)
            for (int row = 0; row < line_drawn[which].height; ++row)
                for (int col = 0; col < line_drawn[which].width; ++col)
                    if (now[which][row][col] != (int8_t)TRANSPARENT &&
                        line_drawn[which].data[row][col] != TRANSPARENT)
                        now[which][row][col] = (int8_t)line_drawn[which].data[row][col];
        for (size_t which = 0; which < painted.size() && which < now.size(); ++which)
            for (const cell_color& cell : painted[which]) now[which][cell.row][cell.col] = cell.color;
        if (now.size() != canvas.size() || memcmp(now.data(), canvas.data(), now.size() * sizeof(cells)) != 0) {
            // Laid under or onto anything, a paint reads only where something
            // is, so it is read again only where that changed.
            bool moved = now.size() != canvas.size();
            for (size_t which = 0; which < now.size() && !moved; ++which)
                for (int row = 0; row < MAX_GRID_DIM && !moved; ++row)
                    for (int col = 0; col < MAX_GRID_DIM && !moved; ++col)
                        moved = (now[which][row][col] != (int8_t)TRANSPARENT) !=
                                (canvas[which][row][col] != (int8_t)TRANSPARENT);
            occupancies += moved;
            canvas = std::move(now);
            ++canvases;
            canvas_number = ++paints.canvases;
        }
    };
    // What a paint lays, pair by pair: the cells it lays right, in the colors
    // the output shows there, and the cells it lays wrong, in what it lays.
    auto laid_by = [&](const replayed& seen) {
        colors_by_pair laid(seen.drawn.size());
        for (size_t which = 0; which < seen.drawn.size(); ++which) {
            for (const auto& [row, col] : seen.drawn[which])
                laid[which].push_back({ (int8_t)row, (int8_t)col, (int8_t)facts.train[which].output.data[row][col] });
            if (which < seen.wrong.size())
                laid[which].insert(laid[which].end(), seen.wrong[which].begin(), seen.wrong[which].end());
        }
        return laid;
    };
    // Says a paint, and keeps what it lays for the canvas.
    auto say_paint = [&](const heard& best, rule::said_as as, const string& says, int8_t under = -1) {
        const int8_t said = say(best, as, says, under);
        const colors_by_pair laid = laid_by(best.seen);
        for (size_t which = 0; which < laid.size() && which < painted.size(); ++which)
            painted[which].insert(painted[which].end(), laid[which].begin(), laid[which].end());
        return said;
    };
    // THE WAYS A PAINT MAY BE LAID on what is down now: over it; where it is
    // known, under it and only onto it; and where the line draws its picture,
    // only onto each color down in every pair, or only off it (MaskBy and
    // MaskOut of the color: bdad9b1f's crossing, one line laid only onto the
    // other's color).
    auto lays = [&] {
        vector<int8_t> ways = { rule::laid_over };
        if (!down_known()) return ways;
        ways.insert(ways.end(), { rule::laid_under, rule::laid_onto });
        if (line_drawn.size() != facts.train.size()) return ways;
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (ranges::all_of(canvas, [&](const cells& down) {
                    return ranges::find(down.values, (int8_t)color) != ranges::end(down.values);
                }))
                ways.insert(ways.end(), { (int8_t)(rule::laid_onto_color + color), (int8_t)(rule::laid_off_color + color) });
        return ways;
    };
    // EACH PAINT REPLAYED ONCE: it reads only the cut opened again and, laid under
    // or onto what is down, the canvas, so one way of asking it -- the kind, the
    // pick, the lay, the value -- is replayed once laid over, and once for each
    // canvas otherwise -- where something is, laid under or onto anything, and
    // in what colors, laid onto or off one -- however often the words around it
    // are asked; and again once it may no longer lay a cell wrong.
    using replay_key = std::tuple<int, size_t, int, int, int, bool>;
    auto replay_key_of = [&](int kind, size_t pick, int8_t lay, int value) -> replay_key {
        const int read_on = lay == rule::laid_over ? 0
                          : lay == rule::laid_under || lay == rule::laid_onto ? occupancies : canvases;
        return { kind, pick, (int)lay, value, read_on, opened.tolerant };
    };
    std::map<replay_key, optional<replayed>> replays;
    std::map<replay_key, optional<marked_paint>> marked_replays;
    auto replayed_once = [&](auto& cache, int kind, size_t pick, int8_t lay, int value, auto replay) -> const auto& {
        const auto [at, fresh] = cache.try_emplace(replay_key_of(kind, pick, lay, value));
        if (fresh) at->second = replay();
        return at->second;
    };
    // EACH PAINT WALKED ONCE (paint_walk): its walk reads only the cut opened
    // again, so one way of asking it -- the kind, the pick, the value -- is
    // walked once for every line on the cut (cut_paints), however often the
    // words around it are asked, and laid from that walk every way, on every
    // canvas, in `in` (lay_paint) -- counted once for each canvas it is laid on
    // (walk_tally), so a lay that cannot hold is known from the counts, and only
    // one that may is laid cell by cell.
    auto laid_once = [&](int kind, size_t pick, int8_t lay, int value, laying::colors in,
                         auto walk) -> const optional<marked_paint>& {
        const auto [walked, fresh] = paints.walks.try_emplace({ kind, pick, value });
        if (fresh) walked->second.walk = walk();
        return replayed_once(marked_replays, kind, pick, lay, value, [&]() -> optional<marked_paint> {
            walked_paint& paint = walked->second;
            if (!paint.walk) return nullopt;
            if (paint.counted_on != canvas_number) {
                paint.tally = tally_walk(opened, *paint.walk, canvas);
                paint.counted_on = canvas_number;
            }
            if (!may_hold(opened, paint.tally, { lay, &canvas, in })) return nullopt;
            return lay_paint<marked_paint>(opened, *paint.walk, { lay, &canvas, in });
        });
    };
    // How a paint was laid, in words.
    auto laid_words = [](int8_t lay) -> string {
        if (lay == rule::laid_under) return ", laid under what is down";
        if (lay == rule::laid_onto) return ", only onto what is down";
        if (lay >= rule::laid_off_color) return ", only off the " + to_string(lay - rule::laid_off_color) + "s down";
        if (lay >= rule::laid_onto_color) return ", only onto the " + to_string(lay - rule::laid_onto_color) + "s down";
        return "";
    };

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
    // coincidences that is wrong on its test. A pair finished before the line
    // (replay_facts::finished) has no cell to explain, and asks none.
    // AND A PAINT ACCOUNTS ONLY FOR WHAT THE LINE DOES NOT SHOW ALREADY: a cell
    // the line so far draws as the output shows it is the line's own to say. A
    // paint asked beside the words could take it otherwise: 22168020's pieces,
    // each repeated along its rows only onto what is down, redrew the pieces
    // the line leaves where they lie -- and the bars beside them -- and did
    // nothing else, a word for the pairs, wrong on its test.
    auto weigh_paint = [&](heard& best, int value, const piece_pick& pick, const auto& seen, claim claims) {
        claim unshown = [&line_shows, claims = std::move(claims)](size_t which, const left_over_cell& cell) {
            return claims(which, cell) && !line_shows(which, cell);
        };
        for (size_t which = 0; which < read.pairs.size(); ++which)
            if (!opened.finished(which) && ranges::none_of(read.pairs[which].left_over, [&](const left_over_cell& cell) {
                    return cell.claimed_by < 0 && unshown(which, cell); }))
                return;
        weigh(best, value, pick, seen, std::move(unshown));
    };

    // Own-color paints cannot claim a target color absent from their sources.
    // Read actual output colors: an input-side leftover carries its old color.
    vector<uint16_t> paint_colors(read.pairs.size());
    auto colors_of_pick = [](const vector<partner_pieces>& handed) {
        vector<uint16_t> colors(handed.size());
        for (size_t which = 0; which < handed.size(); ++which)
            for (int i = 0; i < handed[which].count(); ++i)
                if (handed[which].picked[i])
                    colors[which] |= palette(handed[which].pieces[i]);
        return colors;
    };
    auto paint_has_color = [&](const vector<uint16_t>& colors) {
        for (size_t which = 0; which < read.pairs.size(); ++which)
            if (!opened.finished(which) && !(colors[which] & paint_colors[which])) return false;
        return true;
    };
    vector<vector<partner_pieces>>& handed_by_pick = paints.handed_by_pick;
    if (!cut_opened)
        for (const piece_pick& pick : draw_picks) {
            handed_by_pick.push_back(handed_pieces(opened, pick));
            paints.colors_by_pick.push_back(colors_of_pick(handed_by_pick.back()));
        }
    // THE DRAW PICKS A PAINT ASKS, in order. A pick that takes the same pieces
    // as one before it, in every pair, is handed the same pieces: every paint
    // draws for it what it drew for that one, which keeps a tie, so it is not
    // asked.
    vector<size_t>& asked_picks = paints.asked_picks;
    if (!cut_opened)
        for (size_t at = 0; at < draw_picks.size(); ++at) {
            auto same = [&](size_t before) { return takes_the_same(handed_by_pick[before], handed_by_pick[at]); };
            if (ranges::any_of(asked_picks, same) || ranges::any_of(paints.asked_partner_picks, same)) continue;
            (at < paints.plain_draws ? asked_picks : paints.asked_partner_picks).push_back(at);
        }
    // The draw picks a paint is asked with now: the partner picks too, once the
    // line has turned to its last resort (partner_picks).
    vector<size_t> asked_now = asked_picks;
    vector<size_t> own_color_picks;

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
                  row_of(node_type::ExtendDiagonal).high < paint_value::second_bit,
                  "one ray length is said the same both ways round, and fits under the diagonal's 32");
    // THE FARTHEST A RAY RUNS on any pair's grid: past it, every longer ray draws
    // as the longest, asked first, so it is replayed once.
    int farthest = 0;
    for (const example& pair : opened.train) farthest = max(farthest, max(pair.output.height, pair.output.width) - 1);
    // EVERY RAY LENGTH LAID AT ONCE (lay_every_length): the rays of a pick, one
    // way round, fanned out once for every line on the cut (rays_fan), every
    // length laid the first time one is asked on a canvas, and each kept as
    // laid_once keeps one. A length is walked on its own (rays_walk), once for
    // every line, only where it ties for its color or holds.
    auto rays_laid_once = [&](size_t at, int8_t lay, bool diagonal, int reach) -> const optional<marked_paint>& {
        const replay_key key = replay_key_of(0, at, lay, paint_value::packed(diagonal, reach));
        if (const auto found = marked_replays.find(key); found != marked_replays.end()) return found->second;
        const auto [fanned, fresh] = paints.fans.try_emplace({ at, diagonal });
        if (fresh) fanned->second = rays_fan(opened, handed_by_pick[at], diagonal, farthest);
        vector<optional<marked_paint>> laid;
        if (fanned->second)
            laid = lay_every_length(opened, *fanned->second, { lay, &canvas, laying::in_one }, farthest,
                                    [&](int length) -> const paint_walk* {
                const auto [walked, unwalked] = paints.walks.try_emplace({ 0, at, paint_value::packed(diagonal, length) });
                if (unwalked) walked->second.walk = rays_walk(opened, handed_by_pick[at], diagonal, length);
                return walked->second.walk ? &*walked->second.walk : nullptr;
            });
        laid.resize((size_t)farthest + 1);
        for (int length = 0; length <= farthest; ++length)
            marked_replays.try_emplace(replay_key_of(0, at, lay, paint_value::packed(diagonal, length)),
                                       std::move(laid[(size_t)length]));
        return marked_replays.at(key);
    };
    auto ray_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : asked_now)
            for (const int8_t lay : lays())
                for (const bool diagonal : { false, true })
                    for (int radius = rays.high; radius >= rays.low; --radius) {
                        if (radius < rays.high && radius > farthest) continue;
                        const int reach = min(radius, farthest);
                        const auto& seen = rays_laid_once(at, lay, diagonal, reach);
                        if (!seen) continue;
                        const int before = best.said;
                        weigh_paint(best, paint_value::packed(diagonal, radius), draw_picks[at], *seen,
                                    covered_claims(seen->drawn));
                        if (best.said > before) best_lay = lay;
                    }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const bool own = best.pick.by_color() && best.pick.colors == (uint16_t)(1u << best.seen.paint);
            const int8_t said = say_paint(best, rule::extend_rays,
                string("grows rays ") + (paint_value::second(best.value) ? "along the diagonals" : "up, down and to the sides") + ", " +
                    to_string(paint_value::number(best.value)) + " long, in " +
                    (own ? string("its own color") : "color " + to_string(best.seen.paint)) + laid_words(best_lay));
            read.fired[said].lay = best_lay; }, laid_by(best.seen) };
    };

    // A TRAIL A PIECE GROWS one of the eight ways, in its own color, until it meets
    // the edge of the grid or another piece -- or, past those, every picked piece
    // grown in step with the others along its track until they meet, the eight
    // ways asked first and taking a tie -- asked again, like the rays, while a
    // trail accounts for any cell, laid each way a paint may be.
    static_assert(direction::count + (int)grown_along::count == row_of(node_type::ExtendTo).high + 1);
    static constexpr const char* grown_names[] = { "along its rows", "along its columns", "to every side",
                                                   "along the grid's edge", "along its diagonals down to the right",
                                                   "along its diagonals up to the right", "along both its diagonals",
                                                   "along its rows and columns", "every way, straight and diagonal",
                                                   "to every side and corner" };
    static_assert(size(grown_names) == (size_t)grown_along::count);
    auto trail_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : own_color_picks)
            for (const int8_t lay : lays())
                for (int asked = 0; asked <= row_of(node_type::ExtendTo).high; ++asked) {
                    const int grow_direction = asked < direction::count ? direction_asked(asked) : asked;
                    const auto& seen = laid_once(1, at, lay, grow_direction, laying::as_drawn, [&] {
                        return trail_walk(opened, handed_by_pick[at], grow_direction); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, grow_direction, draw_picks[at], *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const string grows = best.value < direction::count
                ? string("grows a trail ") + direction_names[best.value] + " until it meets something"
                : string("grows ") + grown_names[best.value - direction::count] + " in step with the others until they meet";
            const int8_t said = say_paint(best, rule::trailed, grows + laid_words(best_lay));
            read.fired[said].lay = best_lay;
        }, laid_by(best.seen) };
    };

    // THE PICKED PIECES COPIED ONTO THE OTHERS, lined up one of CopyTo's nine ways.
    static_assert(row_of(node_type::CopyTo).high + 1 == 9);
    static constexpr const char* lined_up[] = { "top-left", "top", "top-right", "left", "middle", "right",
                                                "bottom-left", "bottom", "bottom-right" };
    auto copy_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : own_color_picks)
            for (const int8_t lay : lays())
                for (int anchor = 0; anchor < (int)size(lined_up); ++anchor) {
                    const auto& seen = laid_once(2, at, lay, anchor, laying::as_drawn, [&] {
                        return copy_walk(opened, handed_by_pick[at], anchor); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, anchor, draw_picks[at], *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const int8_t said = say_paint(best, rule::copied_onto,
                string("is copied onto the other pieces, lined up ") + lined_up[best.value] + laid_words(best_lay));
            read.fired[said].lay = best_lay;
        }, laid_by(best.seen) };
    };

    // A PIECE REPEATED ALONG ITS ROWS OR DOWN ITS COLUMNS, to the edges both ways
    // -- as far apart as its own width along its rows, or its own height down
    // its columns, read on each grid, or every so many cells -- asked again,
    // like the rays, while a repeat accounts for any cell, so one piece can be
    // repeated both ways. A tie goes to its own width or height, which grows
    // with the piece on a test, then to the tighter spacing the pairs bear out.
    // THE NARROWEST PAIR'S GRID, across and down: so many cells apart as that
    // lays no copy beside the piece itself there, and is not at work.
    int narrowest[2] = { MAX_GRID_DIM, MAX_GRID_DIM };
    for (const example& pair : opened.train) {
        narrowest[0] = min(narrowest[0], pair.output.width);
        narrowest[1] = min(narrowest[1], pair.output.height);
    }
    // The repeats asked, in the order asked; a repeat's value is its place here.
    // ITS OWN WIDTH AND HEIGHT ARE A SHAPE'S: a whole color layer's are only the
    // span its scattered cells happen to cover, and a layer repeated by them fits
    // the pairs by coincidence (62b74c02's last nine cells, wrong on its test).
    const bool cut_into_shapes = election.segmentation != segmentation_methods::all_cells_of_one_color;
    vector<tile_value> repeats;
    for (const bool down : { false, true }) {
        auto apart = [&](tile_number cells) {
            tile_value repeat{ .start = { .edge = true }, .end = { .edge = true },
                               .spread = tile_spread::lines };
            repeat.steps[0] = down ? tile_value::step{ cells, { 0 } } : tile_value::step{ { 0 }, cells };
            repeats.push_back(repeat);
        };
        if (cut_into_shapes) apart(measured(down ? tile_measure::own_height : tile_measure::own_width));
        for (int cells = 1; cells < narrowest[down]; ++cells) apart({ (int8_t)cells });
    }
    auto repeat_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : own_color_picks)
            for (const int8_t lay : lays())
                for (int asked = 0; asked < (int)repeats.size(); ++asked) {
                    const auto& seen = laid_once(3, at, lay, asked, laying::as_drawn, [&] {
                        return tile_walk(opened, handed_by_pick[at], repeats[(size_t)asked]); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, asked, draw_picks[at], *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const tile_value& repeat = repeats[(size_t)best.value];
            const bool down = repeat.steps[0].down != tile_number{};
            const tile_number& cells = down ? repeat.steps[0].down : repeat.steps[0].across;
            const int8_t said = say_paint(best, rule::tiled_along,
                "is repeated " +
                    (cells.terms[0].measure == tile_measure::none
                         ? "every " + to_string(cells.count) + " cells "
                         : "its " + string(tile_measure_names[(int)cells.terms[0].measure]) + " apart ") +
                    (down ? "down its columns" : "along its rows") + laid_words(best_lay));
            read.fired[said].lay = best_lay;
            read.fired[said].tile = repeat; }, laid_by(best.seen) };
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
    const vector<partner_pieces> figure_handed = handed_pieces(opened, figure);
    const vector<uint16_t> figure_colors = colors_of_pick(figure_handed);
    auto turned_paint = [&]() -> offer {
        if (!still() || !paint_has_color(figure_colors)) return {};
        heard best;
        for (const auto& [pose, name] : turns) {
            if (const auto& seen = replayed_once(replays, 4, 0, rule::laid_over, pose, [&] {
                    return turned_holds(opened, figure_handed, pose); }))
                weigh_paint(best, pose, figure, *seen, covered_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const auto turn = ranges::find_if(turns, [&](const auto& one) { return one.first == best.value; });
            say_paint(best, rule::turned_copy, string("is drawn again as one figure, ") + turn->second + " in its box");
        }, laid_by(best.seen) };
    };

    // BARS BETWEEN PIECES OF ONE COLOR, in their own color or one of the line's,
    // reaching each way a bridge can, the narrowest first, so a wider way is said
    // only where it accounts for more. Between every piece first, then between
    // the pieces each pick takes, as the other paints are asked: Bridge joins
    // only the picked pieces (0d87d2a6's blue marks, past red blocks that line
    // up with one another and must not be joined).
    // THEN AN L from each picked piece to its one partner of each kind
    // (partner_of_kind), every pick asked for every straight bar first, so an
    // L is said only where it accounts for more. It leaves along the picked
    // piece's row, so the pick says which way it turns (0e671a1a: the 2 runs to
    // the piece of color 4, and the 4 to the piece of color 3).
    static constexpr const char* reaches[BRIDGE_WAYS] = {
        "both ways to the nearest", "along rows to the nearest", "along columns to the nearest",
        "both ways to every one", "along rows to every one", "along columns to every one",
        "both ways to the one facing it across the grid", "along rows to the one facing it across the grid",
        "along columns to the one facing it across the grid" };
    const vector<partner_pieces> every_piece_handed = handed_pieces(opened, piece_pick::all());
    auto bar_paint = [&]() -> offer {
        if (!down_known()) return {};
        heard best;
        for (const bool ells : { false, true })
            for (size_t at = 0; at <= draw_picks.size(); ++at) {
                if (at > 0 && !ranges::binary_search(asked_now, at - 1)) continue;
                const piece_pick pick = at == 0 ? piece_pick::all() : draw_picks[at - 1];
                const vector<partner_pieces>& handed = at == 0 ? every_piece_handed : handed_by_pick[at - 1];
                auto ask = [&](int way) {
                    if (const auto& seen = replayed_once(replays, 5, at, rule::laid_under, way, [&] {
                            return bridge_holds(opened, handed, canvas, way); }))
                        weigh_paint(best, way, pick, *seen, drawn_claims(seen->drawn));
                };
                if (!ells) ranges::for_each(bridge_ways_narrowest_first, ask);
                else for (int way = BRIDGE_TO_PARTNER; way < BRIDGE_VALUES; ++way)
                    if (opened.asks_kind(way - BRIDGE_TO_PARTNER)) ask(way);
            }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::bridged,
                (best.value < BRIDGE_TO_PARTNER
                     ? string("bars filled between pieces of one color, reaching ") + reaches[best.value]
                     : "an L run along its row to the column of " + partner_named(best.value - BRIDGE_TO_PARTNER) +
                           ", then along that column to it") +
                    ", in " + (best.seen.paint < 0 ? string("their own color") : "color " + to_string(best.seen.paint)));
        }, laid_by(best.seen) };
    };

    // THE REST OF THE CHANGE, READ OFF THE PICTURE SO FAR. Once a paint is said,
    // what it drew is in the picture the line has drawn -- the hand as the words
    // leave it, and every cell a paint drew, in the color its replay saw the
    // output show there -- and the rest of the change may read it: the picture
    // is cut again as the line cuts the input, and each picked piece takes the
    // color of its partner, a piece a paint drew among them (0d87d2a6's red
    // blocks, blue where a bar the bridge drew touches one). Said after the
    // paints said before it, in a sentence of its own that opens on the picture
    // (Output), it lays only the pieces it picked, over what is down, where they
    // lie. Only a recolor is read so: a piece moved or removed would leave its
    // old cells drawn. And every word on the hand is drawn before the paints,
    // under the picture this read, so none is asked after it.
    // The picture with `also` drawn in as well: the cells a paint not said yet
    // would lay, in what it lays there. Every paint lays its cells in the order
    // said, the ones it lays wrong in what it lays, so a cell one lays wrong and
    // a later one sets right shows right.
    auto the_picture = [&](const colors_by_pair& also) {
        vector<grid> pictures = line_drawn;
        for (size_t which = 0; which < pictures.size(); ++which) {
            auto shown = [&](const vector<cell_color>& cells) {
                for (const cell_color& cell : cells) pictures[which].data[cell.row][cell.col] = cell.color;
            };
            if (which < painted.size()) shown(painted[which]);
            if (which < also.size()) shown(also[which]);
        }
        return pictures;
    };
    // A PIECE OF THE PICTURE THE LINE KNOWS: one of the hand's own pieces, the
    // same cells where they lie. A paint may draw a partner beside a piece, but
    // a piece a paint cut into or ran into is not one the line cut: 63613498's
    // diagonal rays, laid onto a shape, left fragments of it for a recolor to
    // finish, a coincidence of its pairs that is wrong on its test.
    auto whole = [&](const vector<partner_pieces>& handed) {
        for (size_t which = 0; which < handed.size(); ++which)
            for (int i = 0; i < handed[which].count(); ++i) {
                if (!handed[which].is_picked(i)) continue;
                const object& piece = handed[which].pieces[i];
                const bool known = ranges::any_of(facts.cuts()[which], [&](const object& other) {
                    const object_summary& one = piece.summary, & two = other.summary;
                    if (one.row != two.row || one.col != two.col || one.width != two.width ||
                        one.height != two.height || one.count != two.count) return false;
                    bool same = true;
                    piece.each_cell([&](int row, int col, int8_t) { same = same && other.holds(row, col); });
                    return same;
                });
                if (!known) return false;
            }
        return true;
    };
    // The best recolor read off the picture, with `also` drawn in: none before
    // any paint, since the picture is then the hand the words read already.
    auto read_off_the_picture = [&](const colors_by_pair& also) {
        heard best;
        const bool painted_yet = !also.empty() || ranges::any_of(read.line, [&](int8_t which) {
            return group_of(read.fired[which].as) == word_group::paints; });
        const uint16_t from = starts_from(missing::a_recolor);
        if (!painted_yet || !from || line_drawn.size() != facts.train.size()) return best;
        const replay_facts picture(facts.train, the_picture(also), election.segmentation, boundaries);
        for (const piece_pick& pick : picks_to_ask(picture, from)) {
            const vector<partner_pieces> handed = handed_pieces(picture, pick);
            if (!whole(handed)) continue;
            for (int how = 0; how < ways; ++how)
                if (!picture.asks_way((partner_by)how)) continue;
                else if (const auto seen = color_from_holds(picture, handed, (partner_by)how))
                    weigh(best, how, pick, *seen, recolor_claims(*seen));
        }
        return best;
    };
    auto picture_word = [&]() -> offer {
        const heard best = read_off_the_picture({});
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const int8_t said = say_paint(best, rule::color_from,
                string("takes the color of ") + partners[best.value] + ", read off the picture so far", colors);
            read.fired[said].on_the_picture = true;
        } };
    };

    // THE PICTURE MADE SYMMETRIC (symmetric_holds): the picture so far laid over
    // itself mirrored or turned around the grid's middle, the narrowest symmetry
    // first, a wider one said only where it accounts for more. One word for the
    // whole symmetry: said a pose at a time, an early pose can finish one pair
    // and leave a later one nothing to do there (9ddd00f0's first pair needs the
    // mirrors alone, its second the quarter turn too). It reads the picture, so
    // it is said after the paints before it, and no word on the hand follows it.
    // A copy, it accounts as the mirror word does (copy_claims), what it copies
    // being the whole picture: weighed as a paint, it could not count the cells
    // of the pieces it copies, which the line draws already, and lost them to
    // a mirror of some of the pieces. It completes the input, asked only before
    // anything else is said: asked after 21f83797's rays, it mirrored them, a
    // symmetry its pairs' marks share by coincidence, and went wrong on its test.
    auto symmetry_paint = [&]() -> offer {
        if (!read.line.empty() || line_drawn.size() != facts.train.size()) return {};
        replay_facts as_one = facts;
        const vector<grid> pictures = the_picture({});
        for (size_t which = 0; which < as_one.cuts().size(); ++which) {
            const grid& picture = pictures[which];
            object piece;
            int count = 0;
            for (int row = 0; row < picture.height; ++row)
                for (int col = 0; col < picture.width; ++col)
                    if (picture.data[row][col] != (int8_t)TRANSPARENT) {
                        piece.data[row][col] = (int8_t)picture.data[row][col];
                        ++count;
                    }
            piece.summary = { 0, 0, (int8_t)picture.width, (int8_t)picture.height, (int16_t)count };
            as_one.change(which).hand = { piece };
        }
        const vector<partner_pieces> handed = handed_pieces(as_one, piece_pick::all());
        cells_by_pair copied(as_one.cuts().size());
        for (size_t which = 0; which < as_one.cuts().size(); ++which)
            as_one.cuts()[which][0].each_cell([&](int row, int col, int8_t) { copied[which].push_back({ row, col }); });
        heard best;
        for (int kept = 0; kept < (int)size(symmetries); ++kept)
            if (const auto seen = symmetric_holds(as_one, handed, symmetries[kept]))
                weigh(best, kept, piece_pick::all(), *seen, copy_claims(copied, seen->drawn));
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const int8_t said = say_paint(best, rule::symmetric,
                                          string("the picture so far ") + symmetries[best.value].says);
            read.fired[said].on_the_picture = true;
        }, laid_by(best.seen) };
    };

    // THE HOLLOWS OF THE PICKED PIECES, FILLED, in their own color or one of the
    // line's -- or, with value 1, THEIR INSIDE: hollowed to the edge first, the
    // core and the hollows in one color, laid over (inside_holds).
    auto fill_paint = [&]() -> offer {
        if (!down_known()) return {};
        heard best;
        for (const size_t at : asked_now) {
            if (const auto& seen = replayed_once(replays, 6, at, rule::laid_under, 0, [&] {
                    return fill_holds(opened, handed_by_pick[at], canvas); }))
                weigh_paint(best, 0, draw_picks[at], *seen, drawn_claims(seen->drawn));
            if (const auto& seen = replayed_once(replays, 6, at, rule::laid_over, 1, [&] {
                    return inside_holds(opened, handed_by_pick[at]); }))
                weigh_paint(best, 1, draw_picks[at], *seen, covered_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::filled,
                      string(best.value == 1 ? "fills its inside in " : "fills its hollows in ") +
                          (best.seen.paint < 0 ? string("its own color") : "color " + to_string(best.seen.paint)));
        }, laid_by(best.seen) };
    };

    // THE BOXES OF THE PICKED PIECES, FILLED where the pieces leave them empty, in
    // one color.
    auto box_paint = [&]() -> offer {
        if (!down_known()) return {};
        heard best;
        for (const size_t at : asked_now) {
            if (const auto& seen = replayed_once(replays, 7, at, rule::laid_under, 0, [&] {
                    return box_holds(opened, handed_by_pick[at], canvas); }))
                weigh_paint(best, 0, draw_picks[at], *seen, drawn_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { say_paint(best, rule::boxed, "fills its box in color " + to_string(best.seen.paint)); }, laid_by(best.seen) };
    };

    // THE ROOMS THE PICKED PIECES FENCE OFF, ONE FILLED in one color: the
    // largest or smallest room, or a part's room (room_holds). The walls are
    // every piece, or the pieces of one color, which the program can pick again
    // once Partition hands them back. A room is cut from the input, not from what is down, so it is
    // laid over what is down, and under it where that is known, a tie going
    // over. A pick's rooms are walked once, for every room and canvas the line
    // asks.
    vector<optional<vector<vector<fenced_room>>>> rooms_by_pick(draw_picks.size());
    auto room_named = [](int chosen) {
        if (chosen == ROOM_EVERY) return string("every room");
        if (!room_by_size(chosen)) return "the room of part " + to_string(room_part(chosen));
        const int rank = room_rank(chosen);
        return string("the ") + (rank > 0 ? to_string(rank + 1) + (rank == 1 ? "nd " : rank == 2 ? "rd " : "th ") : "") +
               (room_largest(chosen) ? "largest" : "smallest") + " room";
    };
    // EVERY ROOM AT ONCE, EACH IN THE COLOR ITS OWN KEY NAMES (rooms_table): the
    // puzzle's one table, so asked only in the first pass while nothing else
    // holds one (shape_word), and said, it is the table the program carries.
    optional<vector<vector<fenced_room>>> rooms_of_every_piece;
    auto room_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        shape_color_map_rule best_table;
        const bool tables = inf.pass == 1 && !inf.shape_color_map.holds();
        // The rooms the walls `pick` takes fence off, `at` its place among the
        // picks a paint is replayed for.
        auto ask = [&](size_t at, const piece_pick& pick, const vector<partner_pieces>& handed,
                       optional<vector<vector<fenced_room>>>& rooms) {
            if (!rooms) rooms = rooms_fenced(opened, handed);
            for (const int8_t lay : { rule::laid_over, rule::laid_under }) {
                if (lay == rule::laid_under && !down_known()) continue;
                for (int chosen = 0; chosen < (tables ? ROOM_VALUES : ROOM_CHOICES); ++chosen) {
                    shape_color_map_rule table;
                    if (chosen >= ROOM_TABLE) {
                        table = rooms_table(opened, *rooms, chosen - ROOM_TABLE);
                        if (!table.holds()) continue;
                    } else if (ranges::any_of(*rooms, [&](const auto& pair) {
                                   return chosen == ROOM_EVERY ? pair.empty() : !room_chosen(pair, chosen); }))
                        continue;
                    const auto& seen = replayed_once(replays, 8, at, lay, chosen, [&] {
                        return chosen >= ROOM_TABLE
                            ? room_table_holds(opened, handed, *rooms, canvas, table, chosen - ROOM_TABLE, lay)
                            : room_holds(opened, handed, *rooms, canvas, chosen, lay); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, chosen, pick, *seen,
                                lay == rule::laid_over ? covered_claims(seen->drawn) : drawn_claims(seen->drawn));
                    if (best.said > before) {
                        best_lay = lay;
                        best_table = table;
                    }
                }
            }
        };
        // The walls: every piece -- 83302e8f's are 1s, 8s and 9s, one color a
        // pair -- then the pieces of each one color that are not every piece.
        ask(draw_picks.size(), piece_pick::all(), every_piece_handed, rooms_of_every_piece);
        for (const size_t at : asked_now) {
            const piece_pick& pick = draw_picks[at];
            if (has_single_bit(pick.colors) && pick == piece_pick::of_color(countr_zero(pick.colors)) &&
                !takes_the_same(handed_by_pick[at], every_piece_handed))
                ask(at, pick, handed_by_pick[at], rooms_by_pick[at]);
        }
        if (best.value < 0) return {};
        return { best.said, [&, best, best_lay, best_table] {
            const bool by_table = best.value >= ROOM_TABLE;
            if (by_table) inf.shape_color_map = best_table;
            const int8_t said = say_paint(best, rule::filled_room,
                (by_table ? "fences off its rooms, each taking the color its " +
                                string(keys_named[best.value - ROOM_TABLE]) + " names"
                          : "fences off " + room_named(best.value) + ", filled in color " + to_string(best.seen.paint)) +
                    laid_words(best_lay));
            read.fired[said].lay = best_lay;
        }, laid_by(best.seen) };
    };

    // THE WORDS AND THE PAINTS, THE ONE THAT SAYS THE MOST FIRST. Every kind of
    // word and of paint offers its best on the hand and the canvas the line
    // leaves so far, the one accounting for the most cells nothing claims yet is
    // said, a tie going in the order below -- the words before the paints -- and
    // every kind is asked again. Asked after the words instead, a paint lost what
    // it says whole to the words said first: 178fcbfb's rows and columns, each
    // repeated across the grid, came out as a string of mirrored cells the
    // repeats then finished. Spelled, every paint follows the line, laid over
    // what it leaves -- so a paint laid over needs nothing of the words around
    // it -- or under it or onto it, reading the canvas: once a paint that reads
    // it is said, no word that moves, copies, reshapes or removes a piece is
    // asked after it, so the line leaves the canvas as that paint read it. Each
    // kind of word says at most so many words (rounds). A word read off the
    // picture so far is said after the line too, last on a tie; once one is
    // said, no word on the hand is asked: every word is drawn before the
    // paints, under the picture it read.
    bool frozen = false, picture_read = false;
    struct asked { std::function<offer()> offers; bool paint, changes_pieces; };
    const asked kinds[] = { { color_from_word, false, false }, { move_word, false, true },
                            { mirror_word, false, true }, { outline_word, false, true },
                            { recolor_word, false, false }, { shape_word, false, false },
                            { [&] { return key_word(!frozen); }, false, false },
                            { removal_word, false, true }, { ray_paint, true, false },
                            { trail_paint, true, false }, { copy_paint, true, false },
                            { repeat_paint, true, false }, { turned_paint, true, false },
                            { bar_paint, true, false }, { fill_paint, true, false }, { box_paint, true, false },
                            { room_paint, true, false }, { symmetry_paint, true, false }, { picture_word, true, false } };
    static_assert(size(kinds) == size(inference_times::names) - inference_times::first_kind);
    // Whether a paint said reads what is down: laid under it or onto it, or the
    // whole figure where it lies.
    auto reads_what_is_down = [](const rule& said) {
        return group_of(said.as) == word_group::paints &&
               (said.lay != rule::laid_over || said.as == rule::bridged || said.as == rule::filled ||
                said.as == rule::boxed || said.as == rule::turned_copy || said.as == rule::symmetric);
    };
    constexpr int rounds = 8;
    int words[size(kinds)] = {};
    // A PAINT IS WORTH WHAT IT SAYS AND WHAT IT LETS BE READ OFF THE PICTURE
    // RIGHT AFTER IT: 0d87d2a6's bars say 37 cells, and with them drawn, the
    // blocks they touch take their color, 224 more; weighed alone, the bars
    // came after "the largest becomes 1" (140), which left the first pair's
    // one touched block nothing for the picture to change.
    // A paint said for what it lets be read is followed at once by that reading,
    // so the line keeps the worth it was chosen for: 833966f4's rays, said ahead
    // on that promise, were followed by words on the hand instead, a coincidence
    // of its pairs that is wrong on its test. The picture can say no more than
    // every recolored cell still unclaimed, so a paint that could not win even
    // so is not read ahead.
    // A PASS AFTER THE FIRST SAYS ONE CHANGE (read_the_rest): the words on its
    // hand, which lay the hand in one sentence, or else one paint. Its sentence
    // reads the picture the passes before it drew, as the program hands it over
    // (Output); a second sentence would read a picture the first had changed,
    // one this reading never looked at, and the next pass looks at it instead.
    const bool one_change = inf.pass > 1;
    auto hand_said = [&] {
        return ranges::any_of(read.line, [&](int8_t which) {
            const auto group = group_of(read.fired[which].as);
            return group == word_group::recolors || group == word_group::changes_pieces ||
                   group == word_group::removes;
        });
    };
    auto said_last = [&] {
        frozen = frozen || reads_what_is_down(read.fired[read.line.back()]);
        picture_read = picture_read || read.fired[read.line.back()].on_the_picture;
    };
    // A word may lay cells wrong, where the reading lets it, while the line
    // draws its picture (brings_nearer).
    facts.tolerant = inf.tolerant && facts.hand_known && facts.overlays;
    // THE PARTNER PICKS, where the reading asks them (inference::partners):
    // every word and every paint is asked with them too.
    if (inf.partners) {
        hand_picks.partners = true;
        asked_now.insert(asked_now.end(), paints.asked_partner_picks.begin(), paints.asked_partner_picks.end());
    }
    while (true) {
        opened.tolerant = facts.tolerant;
        lay_the_canvas();
        ranges::fill(paint_colors, (uint16_t)0);
        for (size_t which = 0; which < read.pairs.size(); ++which) {
            const grid& output = opened.train[which].output;
            for (const auto& cell : read.pairs[which].left_over)
                if (cell.claimed_by < 0 && !line_shows(which, cell) && output.contains(cell.row, cell.col)) {
                    const int color = output.data[cell.row][cell.col];
                    if (is_color(color)) paint_colors[which] |= (uint16_t)(1u << color);
                }
        }
        own_color_picks.clear();
        for (const size_t at : asked_now)
            if (paint_has_color(paints.colors_by_pick[at])) own_color_picks.push_back(at);
        int recolors_left = 0;
        for (const auto& pair : read.pairs)
            for (const auto& cell : pair.left_over)
                recolors_left += cell.claimed_by < 0 && cell.answers_to(missing::a_recolor);
        offer best;
        int best_worth = 0, best_unlocks = 0;
        size_t chosen = 0;
        for (size_t kind = 0; kind < size(kinds); ++kind) {
            if ((!kinds[kind].paint && (words[kind] >= rounds || picture_read)) ||
                (frozen && kinds[kind].changes_pieces) || (one_change && kinds[kind].paint && hand_said()))
                continue;
            offer one;
            {
                const time_into offer_clock{ inference_time.ms[inference_times::first_kind + kind] };
                one = kinds[kind].offers();
            }
            const bool may_unlock = read_ahead && one.said > 0 && !one.drawn.empty() &&
                                    one.said + recolors_left > best_worth;
            int unlocks = 0;
            if (may_unlock) {
                const time_into ahead_clock{ inference_time.ms[inference_times::reading_ahead] };
                unlocks = read_off_the_picture(one.drawn).said;
            }
            if (one.said + unlocks > best_worth) {
                best = std::move(one);
                best_worth = best.said + unlocks;
                best_unlocks = unlocks;
                chosen = kind;
            }
        }
        if (best.said == 0) break;
        best.say();
        ++words[chosen];
        said_last();
        if (one_change && kinds[chosen].paint) break;
        if (best_unlocks > 0)
            if (offer then = picture_word(); then.said > 0) {
                then.say();
                said_last();
            }
    }

    // WHAT THE LINE DRAWS, as the words leave it (draw_the_line): it explains
    // every cell where it shows what the output shows. d23f8c26's middle column,
    // which the matching paired with pieces the words remove, is drawn by the
    // line where it stands.
    draw_the_line();
    read.drawn_wrong = 0;
    // Where a word said laid a cell wrong, pair by pair.
    vector<array<array<bool, MAX_GRID_DIM>, MAX_GRID_DIM>> laid_wrong_at(laid_wrong.size());
    for (size_t which = 0; which < laid_wrong.size(); ++which) {
        for (auto& row : laid_wrong_at[which]) row.fill(false);
        for (const cell_color& cell : laid_wrong[which]) laid_wrong_at[which][cell.row][cell.col] = true;
    }
    if (!line_drawn.empty()) {
        // THE PICTURE THE LINE DRAWS: the hand as the words leave it, and every
        // paint laid over it in the order said.
        const vector<grid> pictures = the_picture({});
        auto shows_right = [&](size_t which, int row, int col) {
            return pictures[which].data[row][col] == facts.train[which].output.data[row][col];
        };
        // WHAT THE WORDS CLAIMED, HELD TO WHAT THE LINE DRAWS. Each word claimed
        // its cells on the hand as it was said, and a word said after it can undo
        // them: 50c07299's diagonal, slid its own size and then a step more,
        // leaves the cell the first slide landed on. A cell a word on the line
        // claimed stays its own only where the picture bears it out -- on the
        // output, the picture shows the cell's color; on the input, the picture
        // shows what the output shows there, or nothing, which a paint may say.
        // So does a cell a paint claimed, where a word laid that cell wrong.
        for (size_t which = 0; which < read.pairs.size() && which < pictures.size(); ++which)
            for (auto& cell : read.pairs[which].left_over) {
                if (cell.claimed_by < 0 || ranges::find(read.line, cell.claimed_by) == read.line.end()) continue;
                const auto group = group_of(read.fired[cell.claimed_by].as);
                if (group != word_group::recolors && group != word_group::changes_pieces && group != word_group::removes &&
                    !(which < laid_wrong_at.size() && laid_wrong_at[which][cell.row][cell.col]))
                    continue;
                const int8_t there = (int8_t)pictures[which].data[cell.row][cell.col];
                if (there == (int8_t)facts.train[which].output.data[cell.row][cell.col] ||
                    (cell.from_input && there == (int8_t)TRANSPARENT))
                    continue;
                --read.fired[cell.claimed_by].cells;
                cell.claimed_by = -1;
            }
        read.list_claims(listed("Drawn By The Line", "the words draw it as the output shows it", rule::unsaid),
                         [&](size_t which, const left_over_cell& cell) {
                             return line_shows(which, cell) && shows_right(which, cell.row, cell.col);
                         });
        // THE CELLS THE LINE DRAWS WRONG: each one a word laid wrong that the
        // picture still shows wrong, where no cell left over on the output
        // stands for it already.
        for (size_t which = 0; which < laid_wrong.size() && which < pictures.size(); ++which) {
            bool counted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            for (const auto& cell : read.pairs[which].left_over)
                if (!cell.from_input) counted[cell.row][cell.col] = true;
            for (const cell_color& cell : laid_wrong[which])
                if (!counted[cell.row][cell.col] && !shows_right(which, cell.row, cell.col)) {
                    counted[cell.row][cell.col] = true;
                    ++read.drawn_wrong;
                }
        }
    }
    else
        // No picture to look at: every cell a word laid wrong stays wrong.
        for (const auto& wrong_here : laid_wrong_at)
            for (const auto& row : wrong_here) read.drawn_wrong += (int)ranges::count(row, true);
    // What the line draws, for the pass after it to read (read_the_rest).
    inf.picture.clear();
    if (line_drawn.size() == facts.train.size()) inf.picture = the_picture({});
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
// framed line where that says more, or the passes after it (read_the_rest);
// none where a replayed standalone rule already redraws every output.
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
        if (read.framed->rest_says_more) unsaid = min(unsaid, cells_left_unsaid(*read.framed->rest));
    }
    // What the passes after the line leave unsaid, read on the picture it drew.
    if (read.rest_says_more) unsaid = min(unsaid, cells_left_unsaid(*read.rest));
    return unsaid;
}

// HOW MANY PASSES A READING SAYS: its own, then each after it that says more
// (read_the_rest), the last one's number.
inline int passes_said(const inference& read) {
    return read.rest_says_more && read.rest ? passes_said(*read.rest) : read.pass;
}

// THE CANDIDATE WHOSE WORDS SAY THE MOST: each one read in the order given, and
// the one whose reading leaves the fewest cells unsaid kept -- of those, the
// one said in the fewest passes, the plainer saying: bdad9b1f's two lines and
// their crossing, said in one, where five passes of mirrored copies drew its
// pairs by coincidence -- a tie going to the one read first. So once one
// leaves none with no pass after its own, no later one is read. Every choice
// the reading makes is made here and judged this one way, by its words: the
// ground plan and whether a word may lay cells wrong (infer), the cut
// (read_figures), each asked just above the first word that reads it; the
// passes are read once a cut is chosen, so a cut is judged by its line alone.
// Whatever orders the candidates only breaks a tie. Hands back the one kept,
// where it stood, and the cells each one left unsaid, -1 for one never read.
struct said_most {
    inference reading;
    int kept = -1;
    vector<int> left;
};
template <class Read>
inline said_most says_the_most(int count, Read read) {
    said_most most;
    most.left.assign((size_t)count, -1);
    int kept_passes = 0;
    for (int i = 0; i < count; ++i) {
        inference one = read(i);
        most.left[i] = cells_left_unsaid(one);
        const int passes = passes_said(one);
        if (most.kept < 0 || most.left[i] < most.left[most.kept] ||
            (most.left[i] == most.left[most.kept] && passes < kept_passes)) {
            most.reading = std::move(one);
            most.kept = i;
            kept_passes = passes;
        }
        if (most.left[most.kept] == 0 && kept_passes <= most.reading.pass) break;
    }
    return most;
}

// Exact, self-delimiting picture data; tile boundaries affect segmentation too.
inline string figures_key(const vector<example>& train, const tiled_pieces& tiles) {
    string key;
    for (const auto& pair : train) {
        for (const grid* picture : { &pair.input, &pair.output }) {
            key.push_back((char)picture->width);
            key.push_back((char)picture->height);
            for (int row = 0; row < picture->height; ++row)
                for (int col = 0; col < picture->width; ++col)
                    key.push_back((char)picture->data[row][col]);
        }
        const auto tile = tiles.of(pair);
        key.push_back((char)tile.width);
        key.push_back((char)tile.height);
    }
    return key;
}

// One puzzle owns completed results, not in-progress readings or mutable claims.
// Eviction changes reuse only; readers still owning a result keep it alive.
struct figure_readings {
    map<string, shared_ptr<const inference>> completed;
    vector<string> oldest_first;
    void keep(string key, shared_ptr<const inference> read) {
        constexpr size_t capacity = 8;
        if (completed.contains(key)) return;
        if (oldest_first.size() == capacity) {
            completed.erase(oldest_first.front());
            oldest_first.erase(oldest_first.begin());
        }
        oldest_first.push_back(key);
        completed.emplace(std::move(key), std::move(read));
    }
};

inline void read_the_cut(inference& inf, const vector<example>& train, const tiled_pieces& tiles,
                         figure_readings& readings);
inline void read_the_rest(inference& inf, const vector<example>& train, figure_readings& readings);

// THE READING OF FIGURES: pairs with no ground in them -- a plan deleted it,
// or keeps every color as a figure -- read with no election and no ground in
// any reader, so every detector sees figures and empty cells, and an empty cell
// is nothing (where a plan deleted the ground, what it fills last). What reads
// no cut is asked once; what reads one is asked of each cut the words weigh.
// `pass` is which pass this reading is, and once its cut is chosen the rest of
// the change is read on the picture its line draws (read_the_rest); `tolerant`
// whether its words may lay cells wrong (inference::tolerant).
inline inference prepare_figures(const vector<example>& train, const tiled_pieces& tiles) {
    const time_into clock{ inference_time.ms[inference_times::preparing] };
    const auto started = preparation_audit ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    const auto matching_before = matching_work;
    ++inference_work.preparations;
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
    inf.copy_to = copy_to_rule::detect(train, inf.pane_split, tiles);
    inf.interior_fill = interior_fill_rule::detect(train, tiles);
    inf.interior_spread = interior_spread_rule::detect(train, tiles);
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
        inf.hole_colors = hole_color_rule::detect(train, tiles);
        inf.ray_extension = ray_extension_rule::detect(train);
        inf.unchanged   = unchanged_cells_facts::detect(train);
        inf.color_diff  = color_diff_facts::detect(train);
        inf.color_swaps = color_swap_candidates::detect(train, inf.color_diff);
    }
    // Scale / Fixed / BoundingBox: no rule-specific detectors yet.
    // Scale could later reuse unchanged_cells through the ratio (cell<->block).

    // Edge level of correspondence: the figures cut up every way that competes,
    // ranked by how well the pieces match.
    inf.correspondence = correspondence_election::infer(train, tiles);
    if (preparation_audit)
        preparation_audit(train, tiles,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
            matching_work.calls - matching_before.calls,
            matching_work.fitting_edges - matching_before.fitting_edges);
    return inf;
}

// The facts and matching belong to the figures, before any word changes them.
// Each reading takes its own copy; its claims, frames and later pictures stay its own.
inline inference read_figures(const vector<example>& train, const tiled_pieces& tiles, int8_t pass,
                              bool tolerant, bool partners, inference inf, figure_readings& readings) {
    ++inference_work.readings;
    inference_work.later_readings += pass > 1;
    inf.pass = pass;
    inf.tolerant = tolerant;
    inf.partners = partners;
    // THE CUT IS CHOSEN HERE, just above the first word that reads it: each cut
    // in the matching's order is read to the end of its line, and the one whose
    // words say the most is said (says_the_most).
    const vector<correspondence_election::reading_index> ranked = inf.correspondence.ranked;
    if (ranked.empty()) {
        read_the_cut(inf, train, tiles, readings);
        read_the_rest(inf, train, readings);
        return inf;
    }
    said_most said = says_the_most((int)ranked.size(), [&](int rank) {
        inference read = inf;
        read.correspondence.choose((size_t)rank);
        read_the_cut(read, train, tiles, readings);
        return read;
    });
    inference best = std::move(said.reading);
    best.correspondence.left_by_rank = said.left;
    // A tied opening the words ruled out -- read, and leaving more cells
    // unsaid -- is no longer one the reading could not rule out.
    erase_if(best.correspondence.tied_with_chosen, [&](correspondence_election::reading_index tie) {
        const auto at = ranges::find(ranked, tie);
        const int left = at == ranked.end() ? -1 : said.left[(size_t)(at - ranked.begin())];
        return left > said.left[(size_t)said.kept];
    });
    read_the_rest(best, train, readings);
    return best;
}

inline shared_ptr<const inference> read_figures_once(const vector<example>& train, const tiled_pieces& tiles,
        int8_t pass, bool tolerant, bool partners, figure_readings& readings,
        optional<inference>* prepared = nullptr) {
    string key = figures_key(train, tiles);
    key.push_back((char)pass);
    key.push_back((char)tolerant);
    key.push_back((char)partners);
    if (const auto found = readings.completed.find(key); found != readings.completed.end())
        return found->second;
    optional<inference> local;
    auto& facts = prepared ? *prepared : local;
    if (!facts) facts = prepare_figures(train, tiles);
    auto read = make_shared<const inference>(read_figures(train, tiles, pass, tolerant, partners,
        prepared ? *facts : std::move(*facts), readings));
    readings.keep(std::move(key), read);
    return read;
}

// THE PASSES A READING MAKES AT MOST: the first, and nine after it.
constexpr int8_t MOST_PASSES = 10;

// THE REST OF THE CHANGE, READ ON THE PICTURE SO FAR (the user: build on the
// output so far). Where a pass leaves cells unsaid and its line draws something
// new, the picture it draws is the input of a puzzle of its own, from each
// picture to its output, read by every word a puzzle has: the next pass. It is
// said where it leaves fewer cells unsaid, and otherwise kept for the readout
// alone; a pass that draws nothing new ends the passes, as the tenth does.
// Every word is read on the picture the program hands it, so a pass after the
// first says one change (finish_the_line).
inline void read_the_rest(inference& inf, const vector<example>& train, figure_readings& readings) {
    if (inf.pass >= MOST_PASSES || inf.picture.size() != train.size() || cells_left_unsaid(inf) == 0) return;
    vector<example> rest = train;
    bool drew = false;
    for (size_t which = 0; which < train.size(); ++which) {
        rest[which].input = inf.picture[which];
        drew = drew || !(inf.picture[which] == train[which].input);
    }
    if (!drew) return;
    inf.rest = read_figures_once(rest, {}, (int8_t)(inf.pass + 1), inf.tolerant, inf.partners, readings);
    inf.rest_says_more = cells_left_unsaid(*inf.rest) < cells_left_unsaid(inf);
}

// THE WORDS THAT READ THE CUT, asked of the one `inf` reads on: the rules read
// off its pieces, then -- once every rule is known -- the rivals, the rest of
// its line, what no rule on it says, and a size change read through its frame.
inline void read_the_cut(inference& inf, const vector<example>& train, const tiled_pieces& tiles,
                         figure_readings& readings) {
    inf.motif_continuation = motif_continuation_rule::detect(train, inf.correspondence);
    // The same continuation read along the diagonals. Asked after the motif
    // because a repeating box is the plainer claim and should get first refusal.
    inf.diagonal_continuation = diagonal_continuation_rule::detect(train, inf.correspondence);
    inf.piece_slam = piece_slam_rule::detect(train, inf.correspondence, tiles);
    inf.overlap_corner = overlap_corner_rule::detect(train, inf.correspondence, tiles);
    // Pieces that slid up against the ones that stayed. Asked after the chain
    // because a layout that puts every piece somewhere new is the larger claim
    // and should get first refusal; this one leaves most of the grid alone.
    inf.slide_to_touch = slide_to_touch_rule::detect(train, inf.correspondence, tiles);
    // Pieces that slide to the canvas edge, over whatever is in the way. It reads
    // its own cut when the elected one joins the slider to what it passes over.
    inf.edge_slide = edge_slide_rule::detect(train, inf.correspondence.segmentation, inf.size, tiles);
    inf.project_on_seeds = project_on_seeds_rule::detect(train, inf.correspondence, tiles);
    inf.snap_to_anchors = snap_to_anchors_rule::detect(train, inf.correspondence, inf.size, tiles);
    // Gaps between pieces of one colour, filled. It reads its own cut when the
    // elected one cannot say it, so it does not wait on the election's verdict.
    inf.bridge = bridge_rule::detect(train, inf.correspondence, tiles);
    // Marks laid as lines that take turns down the grid. Like the bridge, it
    // falls back on a cut of its own when the elected one cannot say it.
    inf.mark_lattice = mark_lattice_rule::detect(train, inf.correspondence, tiles);
    inf.input_blocks = input_blocks_rule::detect(train, inf.correspondence);
    // THE PUZZLE HAS ONE TABLE, the first pass's: the program carries that one
    // (environment::draw), so a pass after it reads none (e760a62e's third pass
    // named its pieces' colors by a symmetry table the program never had).
    if (inf.pass == 1 && inf.size.Chosen == output_dimension_rule::Identity) {
        inf.shape_color_map = shape_color_map_rule::detect_body_recolor(train, inf.correspondence, tiles);
        // Fallback: same shape->color table, but keyed on each object's own
        // shape and applied as an interior fill (ShapeToInteriorColorMap)
        // instead of a body recolor. Only run when the body-recolor
        // hypothesis didn't validate; the winner is recorded in claimed_by,
        // which is what tells the two ops apart downstream.
        if (!inf.shape_color_map.holds())
            inf.shape_color_map = shape_color_map_rule::detect_interior_fill(train, tiles);
    }

    // PIECES THAT DO NOT REDRAW THE OUTPUT where they lie: the cut's pieces,
    // matched, can leave nothing over and still not be the output -- 2013d3e2's
    // figure matched whole while the output is its corner, cce03e0d's copies
    // matched where half of them are gone -- so a line with no words is held to
    // what it would draw. Asked of the cut alone, before the line is chosen.
    inf.pieces_redraw = redraws_every_output(train, [&](const example& pair, grid& drawn) {
        const auto pieces = pieces_as_said(pair.input, inf.correspondence.segmentation, tiles.of(pair));
        return !pieces.empty() && draw_on_canvas(pair.input, inf.size, drawn, [&](grid& canvas) {
            for (const object& piece : pieces)
                paint_piece(canvas, piece, piece.summary.row, piece.summary.col, true, /*clip=*/true);
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
    // the most is said (says_the_most), a tie keeping the acts. EACH READ
    // AGAIN AHEAD, a paint weighed with what the picture lets be read after it
    // (finish_the_line), and kept only where it leaves fewer cells unsaid: read
    // ahead, 0d87d2a6's bars come before "the largest becomes 1", which left the
    // picture nothing to change in its first pair; but a line read ahead can
    // say less -- ae3edfdc's rays and the reading after them stopped its moves.
    // The picture reads recolors alone, so a line is read ahead only where read
    // plainly it left a recolor unsaid.
    // A PASS AFTER THE FIRST (read_the_rest) says one change in one sentence:
    // no rival, whose program is its own, and no reading ahead, which says a
    // second change after the first.
    if (inf.pass == 1) list_the_rivals(inf);
    const auto& election = inf.correspondence;
    const bool acts = !election.readings.empty() && !election.readings[election.chosen].line.empty();
    const int openings = acts ? 2 : 1;
    vector<inference> plain((size_t)openings);
    auto recolor_unsaid = [](const inference& one) {
        const auto& said = one.correspondence;
        return !said.readings.empty() && ranges::any_of(said.readings[said.chosen].pairs, [](const auto& pair) {
            return ranges::any_of(pair.left_over, [](const auto& cell) {
                return cell.claimed_by < 0 && cell.answers_to(correspondence_election::missing::a_recolor); });
        });
    };
    cut_paints paints;
    inf = says_the_most(inf.pass == 1 ? 2 * openings : openings, [&](int which) {
        const int opening = which % openings;
        const bool ahead = which >= openings;
        if (ahead && !recolor_unsaid(plain[(size_t)opening])) return plain[(size_t)opening];
        inference read = inf;
        if (opening == 1) leave_the_acts_unsaid(read);
        finish_the_line(read, train, tiles, ahead, paints);
        if (!ahead) plain[(size_t)opening] = read;
        return read;
    }).reading;
    inf.correspondence.list_what_is_missing();

    // A SIZE CHANGE, READ THROUGH ITS FRAME: once the frame cuts the output's
    // grid out of the input's, what is left is a same-size puzzle, read by every
    // word a same-size puzzle has -- from each framed input to its output. A
    // pass after the first reads a picture its output's size.
    if (inf.pass > 1) return;
    inf.frame = choose_piece_frame(train, inf, tiles);
    if (inf.frame.holds()) {
        using kind = piece_frame_rule::kind_of;
        // The frames that take the pieces a pick takes off a cut of their own;
        // the rest take the whole input.
        const bool picks = inf.frame.kind == kind::cut_out || inf.frame.kind == kind::laid_out ||
                           inf.frame.kind == kind::filled;
        // The framed grid is what the frame leaves of the figures: nothing is
        // filled in it, and the reading inside it is a reading of figures too.
        const piece_pick& pick = inf.frame.pick;
        vector<example> framed(train.size());
        bool all = true;
        for (size_t which = 0; which < train.size() && all; ++which) {
            framed[which].output = train[which].output;
            const grid& input = train[which].input;
            grid& into = framed[which].input;
            const vector<object> pieces = picks ? inf.frame.pieces(input, tiles.of(train[which])) : vector<object>{};
            bool several = false;
            switch (inf.frame.kind) {
            case kind::laid_out:
                all = laid_out_input(pieces, pick, inf.frame.factor, input, inf.size, into);
                break;
            case kind::filled:
                all = filled_input(pieces, pick, inf.frame.keyed, inf.shape_color_map, input, inf.size,
                                   into);
                break;
            case kind::cut_out:
                all = framed_input(pieces, pick, input, inf.size, into, several);
                break;
            default: all = framed_whole(input, inf.frame, inf.size, into);
            }
        }
        if (!all) inf.frame = {};
        else {
            // Split before Tile: each copy keeps the original input's boundaries.
            const tiled_pieces copies = inf.frame.kind == kind::tiled
                ? tiled_pieces{ framed.data(), train.data() } : tiled_pieces{};
            inf.framed = read_figures_once(framed, copies, 1, inf.tolerant, inf.partners, readings);
        }
    }
}

// Asks every question about the training pairs, in order, and returns the answers.
// THE GROUND IS OWNED ONCE: the election names the plans a puzzle allows
// (ground_plans), each plan's figures are read (read_figures), and the plan
// whose words say the most is said (says_the_most), in the order the plans are
// named. Nothing after this knows a ground but the plan.
inline inference infer(const vector<example>& train) {
    figure_readings readings;
    const background_election election = elect_background(train);
    const vector<ground_plan> plans = ground_plans(election);
    // EVERY PLAN READ WITH WORDS THAT LAY NO CELL WRONG FIRST, then with words
    // that may (inference::tolerant), so a reading that lays one wrong is kept
    // only where it leaves fewer cells unsaid, its passes and all: a word that
    // lays cells wrong can say so much that it is said first and crowds out
    // the words that would have said every cell (332efdb3's grid, every cell 1
    // at once and its holes wrong, where rays laid it right).
    const int count = 1;
    vector<optional<inference>> prepared((size_t)count);
    auto read_plan = [&](int at, bool tolerant, bool partners) {
        const vector<example> figures = figures_only(train, plans[(size_t)at]);
        return *read_figures_once(figures, {}, 1, tolerant, partners, readings, &prepared[(size_t)at]);
    };
    vector<inference::plan_read> plans_read;
    // Each plan read, as the readout shows it: the words its line said.
    auto note_read = [&](int plan, const inference& read) {
        string words;
        if (!read.correspondence.readings.empty()) {
            const auto& line = read.correspondence.readings[read.correspondence.chosen];
            for (const int8_t said : line.line)
                words += (words.empty() ? "" : " | ") + string(line.fired[said].name) +
                         (line.fired[said].says.empty() ? "" : " " + line.fired[said].says);
        }
        plans_read.push_back({ plan, read.tolerant, cells_left_unsaid(read), passes_said(read), words, read.partners });
    };
    inference best = says_the_most(2 * count, [&](int which) {
        const ground_plan& plan = plans[(size_t)(which % count)];
        inference read;
        {
            const time_into clock{ inference_time.ms[which >= count ? inference_times::tolerant : inference_times::strict] };
            read = read_plan(which % count, which >= count, false);
        }
        read.plan = plan;
        note_read(which % count, read);
        return read;
    }).reading;
    // THE PARTNER PICKS, A LAST RESORT (partner_picks): where the reading kept
    // leaves cells unsaid, its plan read again, the same way, with every word,
    // paint and frame asked with the partner picks too, and kept only where it
    // leaves fewer. Not says_the_most's tie on passes: that weighs readings of
    // the same picks, and with more picks a line packs more into a pass by
    // coincidence (62b74c02, f823c43c: every cell in fewer passes than the
    // plain reading that solved them, and wrong on their tests). Asked with the
    // rest in every reading, they said more at once in lines the other picks
    // finish, and steered six solved puzzles wrong (22168020, 272f95fa,
    // 62b74c02, 9344f635, a61f2674, f823c43c); asked only once a line had
    // nothing else to say, they came after picks that left the line no way on
    // (4364c1c4's bands, its first stack only).
    if (const int left = cells_left_unsaid(best); left > 0) {
        const int at = (int)(ranges::find(plans, best.plan) - plans.begin());
        inference with;
        {
            const time_into clock{ inference_time.ms[inference_times::partners] };
            with = read_plan(at, best.tolerant, true);
        }
        with.plan = best.plan;
        note_read(at, with);
        if (cells_left_unsaid(with) < left) best = std::move(with);
    }
    best.ground_election = election;
    best.plans_read = std::move(plans_read);
    return best;
}

}  // namespace arc
