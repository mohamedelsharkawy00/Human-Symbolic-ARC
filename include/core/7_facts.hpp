// The reading, 7_facts.hpp to 13_infer.hpp, reads the examples like a human.
//
// Three kinds of answer:
//   FACTS      observations computed from the examples.
//   RULES      claims checked against the examples.
//   ELECTIONS  ranked interpretations, supplied by upstream detectors.
//
// Each answer owns the detector that fills it (color_facts::detect,
// piece_slam_rule::detect). Shared drawing helpers support those detectors;
// infer() asks them all in order.
//
// This file holds what every later one stands on: the reading's work counters,
// the color and cell facts, the pieces a word is handed, the shape table, and
// the drawing helpers the rules replay with.
#pragma once
#include "1_readings.hpp"
#include "2_size.hpp"
#include "3_background.hpp"
#include "6_correspondence.hpp"
#include "14_vocabulary.hpp"
#include "15_op_math.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
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

// Read just the pieces, in the same size-and-position order used by matching.
inline vector<object> pieces_of(const grid& input, segmentation_methods segmentation, tile_size tiles = {}) {
    vector<object_segmentation::object_info> info;
    vector<object> pieces;
    object_segmentation::segment_for_correspondence(input, segmentation, info, &pieces, 0, 0, tiles);
    return pieces;
}

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
// 15_op_math.hpp), asked here as the ops ask them of their objects, so a replay
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
    // THE KEY IS USED UP (009d5c81), said in three words (16_sentences.hpp): the
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

}  // namespace arc
