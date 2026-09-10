// 5_infer.hpp — read the examples like a human.
// Design story, rationale, and experiment records
// in notes/1 - Overall Design/Inference Master.md
//
// infer() lives here, and so do the channels small enough to have no file of
// their own. Each thing is explained where it is defined; infer() at the bottom
// carries the reading order and the family of files.

#pragma once
#include "1_readings.hpp"
#include "2_size.hpp"
#include "3_background.hpp"
#include "4_correspondence.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// ──────────────────────────────────────────────────────────────────────────
// COLORS · which colors appear where, and in what order
// ──────────────────────────────────────────────────────────────────────────
// The one channel with no file of its own, and the one that cannot sit in one
// place: the presence facts need no background, while by_size ranks only once
// the election has named one.
// Color facts — each answers exactly one question along one direction.
// A bool[COLOR_COUNT] is a color set.
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
    
    // Background is a ROLE, not a statistic: the ground figures sit on.
    // Elected by elect_background, NOT computed here.
    // -1 = "no background" is a legitimate verdict — tiny full grids are all
    // figure — and segmentation then excludes nothing.
    int8_t background_color = -1;

    // in_any_input and in_any_output are safe for LITERAL pinning only, and one
    // set per side — never a global whitelist (lethal, measured).
};

// A set of colors said as digits, "0,3,4", or "-" when it holds none. Beside
// the sets it reads, so the screen and the log say a color set the same way.
inline string colors_as_text(const bool member[COLOR_COUNT]) {
    string listed;
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (member[color]) {
            if (!listed.empty()) listed += ',';
            listed += (char)('0' + color);
        }
    return listed.empty() ? string("-") : listed;
}

// One pass over the train pairs fills every presence set in color_facts.
inline void detect_color_presence(const vector<example>& train, color_facts& facts) {
    if (train.empty()) return;

    auto present = [](const grid& one_grid, bool seen[COLOR_COUNT]) {
        for (int i = 0; i < COLOR_COUNT; ++i) seen[i] = false;
        walk_colored_cells(one_grid, TRANSPARENT,
                           [&](int, int, int color) { seen[color] = true; });
    };

    for (int i = 0; i < COLOR_COUNT; ++i) {
        facts.in_every_input[i]  = true;
        facts.in_every_output[i] = true;
        facts.in_every_grid[i]   = true;
        facts.consumed[i] = true; facts.introduced[i] = true; facts.preserved[i] = true;
    }

    for (const auto& ex : train) {
        bool in[COLOR_COUNT], out[COLOR_COUNT];
        present(ex.input, in);
        present(ex.output, out);
        for (int i = 0; i < COLOR_COUNT; ++i) {
            facts.in_every_input[i]  = facts.in_every_input[i]  && in[i];
            facts.in_any_input[i]    = facts.in_any_input[i]    || in[i];
            facts.in_every_output[i] = facts.in_every_output[i] && out[i];
            facts.in_any_output[i]   = facts.in_any_output[i]   || out[i];
            facts.in_every_grid[i]   = facts.in_every_grid[i]   && in[i] && out[i];
            facts.in_any_grid[i]     = facts.in_any_grid[i]     || in[i] || out[i];
            facts.consumed[i]   = facts.consumed[i]   && (in[i] && !out[i]);
            facts.introduced[i] = facts.introduced[i] && (!in[i] && out[i]);
            facts.preserved[i]  = facts.preserved[i]  && (in[i] && out[i]);
        }
    }

}

// Color · across_outputs — "which colors appear in the outputs, in what
// order?" First-appearance order (load-bearing).
inline void detect_output_colors(const vector<example>& train, color_facts& facts) {
    bool seen[COLOR_COUNT] = {};
    facts.output_colors_count = 0;
    for (const auto& ex : train) {
        const grid& out = ex.output;
        if (out.width <= 0 || out.height <= 0) continue;
        walk_colored_cells(out, TRANSPARENT, [&](int, int, int color) {
            if (seen[color]) return;
            seen[color] = true;
            facts.output_colors[facts.output_colors_count++] = (int8_t)color;
        });
    }
}

// Color · across_outputs — "do all outputs rank their colors by pixel count the
// same way?" The background is a PARAMETER, not read off `facts`, so the
// dependency on the election having already run is visible at the call site
// instead of being a rule someone has to remember.
inline void detect_output_colors_by_size(const vector<example>& train,
                                         int8_t background, color_facts& facts) {
    facts.output_colors_by_size_count = 0;
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
        facts.output_colors_by_size[rank] = first_ranking[rank];
    facts.output_colors_by_size_count = first_count;
}

// Runs the two color detectors that need nothing but the pairs. by_size is not
// among them: it needs the elected background, so infer() calls it later.
//
// A DETECTOR RETURNS ITS FACT. Read the call and you know where the answer came
// out; nothing you passed in is quietly rewritten. The only functions taking a
// non-const reference are the ones whose whole job is to write it, and they say
// so in their names — count_edges_leaving, measure_area_and_reach — or they are
// the several passes that fill ONE shared fact, whose public entry still
// returns: these two, behind this one.
inline color_facts detect_color_facts(const vector<example>& train) {
    color_facts facts;
    detect_color_presence(train, facts);
    detect_output_colors(train, facts);
    return facts;
}

// ──────────────────────────────────────────────────────────────────────────
// SHAPE -> COLOR · which shape says to paint which color
// ──────────────────────────────────────────────────────────────────────────
// Learned shape -> color lookup for the ShapeColorMap op, rejected whole if one
// shape ever wants two colors. Story in
// notes/1 - Overall Design/Inference Master.md, section 3c.
struct shape_color_map_rule {
    static constexpr int MAX_KEYS = 16;
    uint32_t keys[MAX_KEYS]   = {};  // FNV-1a hash of the key glyph's occupancy
    int8_t   colors[MAX_KEYS] = {};  // color the target is repainted to
    int      count = 0;
    bool     valid = false;
    // WHICH DETECTOR FILLED IT. Two of them share this storage and mean
    // different things by it: body_recolor repaints an object, interior_fill
    // paints the hole inside one. A consumer that cannot tell them apart is
    // feeding one op the other's table, so the table says which it is.
    enum : int8_t { claimed_by_nobody, body_recolor, interior_fill } claimed_by
        = claimed_by_nobody;
};

// Add one shape -> color to the table, or say the table cannot hold it. False
// means this shape already wants a different color, or the table is full —
// either way the whole rule is off, which is what both detectors do with it.
inline bool learn_shape_color(shape_color_map_rule& rule, uint32_t shape, int8_t color) {
    for (int i = 0; i < rule.count; ++i)
        if (rule.keys[i] == shape) return rule.colors[i] == color;
    if (rule.count >= shape_color_map_rule::MAX_KEYS) return false;
    rule.keys[rule.count] = shape;
    rule.colors[rule.count] = color;
    rule.count++;
    return true;
}

inline shape_color_map_rule detect_shape_color_map(const vector<example>& train,
                                                   int8_t background) {
    shape_color_map_rule rule;
    if (!pairs_overlay(train)) return rule;

    for (const auto& ex : train) {
        const grid& in = ex.input;
        const grid& out = ex.output;

        // Per-color box over the input (background excluded). An absent color
        // leaves its box empty, so the boxes alone say who is here.
        color_box box[COLOR_COUNT];
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
        if (key < 0 || target < 0 || key == target) return rule;  // need two distinct classes

        // The color the target wears in the output (read at its first input pixel).
        int8_t repainted_to = -1;
        for (int row = box[target].row0; row <= box[target].row1 && repainted_to < 0; ++row)
            for (int col = box[target].col0; col <= box[target].col1; ++col)
                if (in.data[row][col] == target) {
                    repainted_to = (int8_t)out.data[row][col]; break;
                }
        if (!is_color(repainted_to)) return rule;

        // Hash the key glyph's occupancy.
        const int key_width = box[key].width();
        const int key_height = box[key].height();
        bool occupied[MAX_GRID_DIM * MAX_GRID_DIM];
        for (int row = 0; row < key_height; ++row)
            for (int col = 0; col < key_width; ++col)
                occupied[row * key_width + col] =
                    (in.data[box[key].row0 + row][box[key].col0 + col] == key);
        uint32_t key_shape = shape_occupancy_hash(key_width, key_height, occupied);

        // Insert with the consistency gate.
        if (!learn_shape_color(rule, key_shape, repainted_to)) return rule;
    }
    rule.valid = (rule.count > 0);
    if (rule.valid) rule.claimed_by = shape_color_map_rule::body_recolor;
    return rule;
}

// Shape -> INTERIOR-fill color, per-object. Sibling of detect_shape_color_map:
// same occupancy-hash key, but keyed on each object's OWN shape and applied as
// an interior fill. Writes the SHARED shape_color_map_rule storage,
// so infer() must run it only as a FALLBACK after the body-recolor detector
// rejects. Story in notes/1 - Overall Design/Inference Master.md, section 3c.
inline shape_color_map_rule detect_shape_to_interior_color_map(
        const vector<example>& train, int8_t background) {
    shape_color_map_rule rule;
    if (!pairs_overlay(train)) return rule;

    bool visited[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_DIM * MAX_GRID_DIM], queue_col[MAX_GRID_DIM * MAX_GRID_DIM];
    // Zeroed once here, then the used prefix is reset per object below. The reset
    // is what the code relies on; this only says so where a reader starts.
    bool occupied[MAX_GRID_DIM * MAX_GRID_DIM] = {};  // object footprint over its box
    bool outside[MAX_GRID_DIM * MAX_GRID_DIM] = {};   // box cells reachable from the rim

    for (const auto& ex : train) {
        const grid& in = ex.input;
        const grid& out = ex.output;

        memset(visited, 0, sizeof(visited));
        for (int start_row = 0; start_row < in.height && start_row < MAX_GRID_DIM; ++start_row)
        for (int start_col = 0; start_col < in.width  && start_col < MAX_GRID_DIM; ++start_col) {
            if (visited[start_row][start_col]) continue;
            int color = in.data[start_row][start_col];
            if (color < 0 || color == background) {
                visited[start_row][start_col] = true; continue;
            }

            // One object: same-color cells that touch, corners counting (this is
            // what the device's gpu_flood_fill does too).
            queue_row[0] = (int16_t)start_row; queue_col[0] = (int16_t)start_col;
            visited[start_row][start_col] = true;
            int left = start_col, top = start_row, right = start_col, bottom = start_row;
            const int object_size = walk_out(in.height, in.width, queue_row, queue_col, 1,
                [&](int next_row, int next_col) {
                    if (visited[next_row][next_col] ||
                        in.data[next_row][next_col] != color) return false;
                    visited[next_row][next_col] = true; return true;
                },
                [&](int row, int col) {
                    left = min(left, col);  top = min(top, row);
                    right = max(right, col); bottom = max(bottom, row);
                });
            const int width = right - left + 1, height = bottom - top + 1;

            for (int i = 0; i < width * height; ++i) { occupied[i] = false; outside[i] = false; }
            for (int i = 0; i < object_size; ++i)
                occupied[(queue_row[i] - top) * width + (queue_col[i] - left)] = true;

            // Outside = empty cells you can reach from the box's rim without
            // going through a corner, mirroring gpu_interior_paint_warp.
            // Interior = neither the piece nor outside: the hole inside it.
            for (bool changed = true; changed; ) {
                changed = false;
                for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    int id = row * width + col;
                    if (occupied[id] || outside[id]) continue;
                    bool on_box_rim = (row == 0 || row == height - 1 ||
                                         col == 0 || col == width - 1);
                    bool next_to_outside =
                        (row > 0          && outside[id - width]) ||
                        (row < height - 1 && outside[id + width]) ||
                        (col > 0          && outside[id - 1]) ||
                        (col < width - 1  && outside[id + 1]);
                    if (on_box_rim || next_to_outside) { outside[id] = true; changed = true; }
                }
            }

            // Fill color = the output color on interior cells whose INPUT was
            // background — the cells the fill actually paints. Cells that are
            // non-background in the input (inside-noise like a marker pixel)
            // survive unchanged, so they are skipped. Must be one color across
            // the painted interior.
            int fill = -1; bool any_painted = false;
            for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) {
                int id = row * width + col;
                if (occupied[id] || outside[id]) continue;         // not interior
                int in_row = top + row, in_col = left + col;
                if (in.data[in_row][in_col] != background) continue;  // inside-noise
                int out_color = out.data[in_row][in_col];
                if (!is_color(out_color)) return rule;                  // bad fill -> reject
                if (!any_painted) { fill = out_color; any_painted = true; }
                else if (out_color != fill) return rule;                // non-uniform -> reject
            }
            if (!any_painted) continue;                            // nothing painted -> skip

            uint32_t object_shape = shape_occupancy_hash(width, height, occupied);
            if (!learn_shape_color(rule, object_shape, (int8_t)fill)) return rule;
        }
    }
    rule.valid = (rule.count > 0);
    if (rule.valid) rule.claimed_by = shape_color_map_rule::interior_fill;
    return rule;
}

// ──────────────────────────────────────────────────────────────────────────
// UNCHANGED · how much of the input survives untouched
// ──────────────────────────────────────────────────────────────────────────
// Unchanged · across_input_output — "how much of the input is still there,
// untouched, in the output?" One map per pair (per-example values, not just a
// fold). Only meaningful when a pair's input and output are the same size.
// Consumer idea: high kept fraction => layer 1 is "copy the input"; search
// only the cells that changed.
struct unchanged_cells_facts {
    // One map per train pair: true = the cell kept its color
    vector<array<bool, MAX_GRID_DIM * MAX_GRID_DIM>> kept;
    // Per pair: share of cells kept
    vector<float> kept_fraction;
    // The worst pair — the safe number to gate on
    float min_kept_fraction = 0.0f;
    // False until computed (infer() only runs this on Identity puzzles)
    bool valid = false;
};

inline unchanged_cells_facts detect_unchanged_cells(const vector<example>& train) {
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
                    kept[row * MAX_GRID_DIM + col] = true;
                    ++same;
                }
        facts.kept_fraction[i] = (float)same / (float)(in.width * in.height);
        facts.min_kept_fraction = min(facts.min_kept_fraction, facts.kept_fraction[i]);
    }
    facts.valid = true;
    return facts;
}

// ──────────────────────────────────────────────────────────────────────────
// COLOR DIFF · which cells each color gained and lost
// ──────────────────────────────────────────────────────────────────────────
// color_diff · across_input_output · per pair, per color — the raw diff split by
// color: which cells did each color GAIN and LOSE? No interpretation; this is
// the material later detectors read. Complements unchanged_cells, which is
// colorless.
struct color_diff_facts {
    // One map per train pair per color, indexed [pair][color]
    // gained: output cell is this color, input cell wasn't
    vector<array<vector<pair<int8_t, int8_t>>, COLOR_COUNT>> gained;
    // Lost: input cell was this color, output cell isn't
    vector<array<vector<pair<int8_t, int8_t>>, COLOR_COUNT>> lost;
    // Per color, across all pairs: total gained / lost cell counts
    int gained_count[COLOR_COUNT] = {};
    int lost_count[COLOR_COUNT] = {};
    bool valid = false;
};

inline color_diff_facts detect_color_diff(const vector<example>& train) {
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
    facts.valid = true;
    return facts;
}

// Every unordered pair of colors, 10 choose 2: how many variants the swap op
// fans out to, and the size of the table that pins them.
constexpr int COLOR_PAIR_COUNT = COLOR_COUNT * (COLOR_COUNT - 1) / 2;

// ──────────────────────────────────────────────────────────────────────────
// COLOR SWAP · whether part of the change is two colors trading places
// ──────────────────────────────────────────────────────────────────────────
// color_swap · across_input_output — "is (part of) the change a color SWAP?" A
// pair is learned when, in EVERY train pair, every cell that lost one became the
// other and back. It need not explain the whole diff, and a pair that never
// fires says nothing, so it is not learned. Consumer: candidate generation pins
// BidiColorMap's fan-out to the learned variants (solver_beam.hpp).
struct color_swap_rule {
    // Indexed by the op's variant k = row-major pair {a,b}, a<b — the same
    // encoding gpu_exec_bidi_color_map decodes
    bool variant[COLOR_PAIR_COUNT] = {};
    int  count = 0;
};

inline color_swap_rule detect_color_swaps(const vector<example>& train,
                                          const color_diff_facts& diff) {
    color_swap_rule rule;
    if (!diff.valid || !pairs_overlay(train)) return rule;

    int variant = 0;
    for (int first = 0; first < COLOR_COUNT; ++first)
        for (int second = first + 1; second < COLOR_COUNT; ++second, ++variant) {
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
            if (swaps) { rule.variant[variant] = true; rule.count++; }
        }
    return rule;
}

// ──────────────────────────────────────────────────────────────────────────
// THE PRODUCT · the inference struct, and infer() — the reading order
// ──────────────────────────────────────────────────────────────────────────
// The product of infer(): one facts struct per channel, plus the engine's
// size rule read straight off the train pairs.
//
// ONE CHANNEL PER QUESTION about the train pairs. Each channel owns a rule (what
// it claims; some can fail) and a detector (the one pass that reads that rule
// off the pairs), and nothing else — one field below apiece.
//
// FACTS, NOT VERDICTS: consumers fold and gate downstream. One exception —
// background_hypotheses keeps its runners-up ranked, because something later
// revises the choice.
struct inference {
    output_dimension_rule size;
    color_facts colors;
    shape_color_map_rule shape_color_map;
    unchanged_cells_facts unchanged;
    color_diff_facts color_diff;
    color_swap_rule color_swaps;
    correspondence_facts correspondence;
    background_hypotheses background;
};

// infer() is PURE: train pairs in, inference out. Engine reads via puzzle_loader
// bridge. Analysis phase, called once at puzzle load. propose() deferred (see notes).
//
// THE READING ORDER, the one every file number and every include follows, and
// the one the body below walks top to bottom:
//
//   SIZE            what decides the output's width and height
//   COLORS          which colors appear where, and in what order
//   BACKGROUND      which color the figures sit on, elected per side
//   CORRESPONDENCE  where each object went, and how to cut the grid up
//   UNCHANGED       how much of the input survives untouched   ┐ only when
//   COLOR DIFF      which cells each color gained and lost     │ input and
//   COLOR SWAP      whether part of the change is two trading  │ output
//   SHAPE -> COLOR  which shape says to paint which color      ┘ overlay
//
// THE FAMILY. A channel that grew big enough took ALL of itself into a file —
// facts, machinery, and the detector that fills them. infer() calls in and gets
// its answer back. The numbers ARE the reading order above, as far as the files
// go: each may include only what is numbered below it.
//
//   1_readings.hpp        the shared base: count it, box it, walk it
//   2_size.hpp            infer_rule
//   3_background.hpp      elect_background
//   4_correspondence.hpp  select_segmentation
//   6_sentences.hpp       say_the_reading
//   7_log.hpp             the per-puzzle readout
inline inference infer(const vector<example>& train) {
    inference inf;
    inf.size = output_dimension_rule::infer_rule(train);
    inf.colors = detect_color_facts(train);  // rule-independent
    // Background: input-side and output-side elected independently by role,
    // ranked lists kept alive (see step 2 note on background_hypotheses).
    // background_color is the INPUT side's verdict — what segmentation excludes.
    inf.background = elect_background(train);
    inf.colors.background_color = inf.background.input_elected();
    // by_size excludes the background, so it can only rank after the election.
    detect_output_colors_by_size(train, inf.colors.background_color, inf.colors);
    // Edge level of correspondence, under the elected background, with the way
    // of cutting the grid up chosen by what it manages to explain.
    inf.correspondence = select_segmentation(train, inf.colors.background_color);

    // Dims is detector #0: its rule decides which questions even apply
    // (Backward Deduction step 1, applied to detectors instead of ops).
    // Identity = input and output overlay cell-for-cell, so the whole
    // pair-relation toolbox below is meaningful. Each detector re-asks the
    // same question itself (pairs_overlay), so this gate states the intent
    // and none of them depends on being called from here.
    if (inf.size.Chosen == output_dimension_rule::Identity) {
        inf.unchanged   = detect_unchanged_cells(train);
        inf.color_diff  = detect_color_diff(train);
        inf.color_swaps = detect_color_swaps(train, inf.color_diff);
        inf.shape_color_map =
            detect_shape_color_map(train, inf.colors.background_color);
        // Fallback: same shape->color table, but keyed on each object's own
        // shape and applied as an interior fill (ShapeToInteriorColorMap)
        // instead of a body recolor. Only run when the body-recolor
        // hypothesis didn't validate; the winner is recorded in claimed_by,
        // which is what tells the two ops apart downstream.
        if (!inf.shape_color_map.valid)
            inf.shape_color_map = detect_shape_to_interior_color_map(
                train, inf.colors.background_color);
    }
    // Scale / Fixed / BoundingBox: no rule-specific detectors yet.
    // Scale could later reuse unchanged_cells through the ratio (cell<->block).
    return inf;
}

}  // namespace arc
