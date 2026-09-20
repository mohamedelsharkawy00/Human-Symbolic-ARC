// This file reads the examples like a human.
//
// Three kinds of answer:
//   FACTS      observations computed from the examples.
//   RULES      claims checked against the examples.
//   ELECTIONS  ranked interpretations, supplied by upstream detectors.
#pragma once
#include "1_readings.hpp"
#include "2_size.hpp"
#include "3_background.hpp"
#include "6_correspondence.hpp"
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
};

// Formats a color set as comma-separated digits for display and logging, or "-" if empty.
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

// Records the output color palette in order of first appearance for logs and the GUI.
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

// Records colors from most-used to least-used only when every output agrees
// on the ranking, excluding the supplied background.
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

// Collects all basic color facts that can be learned without knowing the background yet.
inline color_facts detect_color_facts(const vector<example>& train) {
    color_facts facts;
    detect_color_presence(train, facts);
    detect_output_colors(train, facts);
    return facts;
}

// Which cells kept their color from input to output, one map per example.
struct unchanged_cells_facts {
    // One map per train pair: true = the cell kept its color
    vector<array<bool, MAX_GRID_DIM * MAX_GRID_DIM>> kept;
    // Per pair: share of cells kept
    vector<float> kept_fraction;
    // The worst pair — the safe number to gate on
    float min_kept_fraction = 0.0f;
    // False until filled in (infer() only runs this on same-size puzzles)
    bool computed = false;
};

// Fills the maps above by comparing each input to its output, cell by cell.
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
    facts.computed = true;
    return facts;
}

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
};

// Records every changed cell under the color it lost and the color it gained.
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
    facts.computed = true;
    return facts;
}

// How many two-color pairs exist — 45 for the 10 colors.
constexpr int COLOR_PAIR_COUNT = COLOR_COUNT * (COLOR_COUNT - 1) / 2;

// Two colors that trade places: every cell of color A becomes color B, and B becomes A.
struct color_swap_rule {
    // Indexed by the op's variant k = row-major pair {a,b}, a<b — the same
    // encoding gpu_exec_bidi_color_map decodes
    bool variant[COLOR_PAIR_COUNT] = {};
    int  count = 0;
    bool holds() const { return count > 0; }
};

// Tries every color pair and keeps the ones that swap in all examples.
inline color_swap_rule detect_color_swaps(const vector<example>& train,
                                          const color_diff_facts& diff) {
    color_swap_rule rule;
    if (!diff.computed || !pairs_overlay(train)) return rule;

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
};

// Remembers which color goes with this shape. Returns false if the shape gets conflicting colors.
inline bool learn_shape_color(shape_color_map_rule& rule, uint32_t shape, int8_t color) {
    for (int i = 0; i < rule.count; ++i)
        if (rule.keys[i] == shape) return rule.colors[i] == color;
    if (rule.count >= shape_color_map_rule::MAX_KEYS) return false;
    rule.keys[rule.count] = shape;
    rule.colors[rule.count] = color;
    rule.count++;
    return true;
}

// Looks at small key shapes and learns what color they say to paint the target shapes.
inline shape_color_map_rule detect_shape_color_map(const vector<example>& train,
                                                   int8_t background) {
    shape_color_map_rule rule;
    if (!pairs_overlay(train)) return {};

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
        bool occupied[MAX_GRID_DIM * MAX_GRID_DIM];
        for (int row = 0; row < key_height; ++row)
            for (int col = 0; col < key_width; ++col)
                occupied[row * key_width + col] =
                    (in.data[box[key].row0 + row][box[key].col0 + col] == key);
        uint32_t key_shape = shape_occupancy_hash(key_width, key_height, occupied);

        // Insert with the consistency gate.
        if (!learn_shape_color(rule, key_shape, repainted_to)) return {};
    }
    if (rule.holds()) rule.claimed_by = shape_color_map_rule::body_recolor;
    return rule;
}

// Learns which color fills the hollow inside of each shape. Shares storage with
// detect_shape_color_map, so infer() runs this only when that one finds nothing.
inline shape_color_map_rule detect_shape_to_interior_color_map(
        const vector<example>& train, int8_t background) {
    shape_color_map_rule rule;
    if (!pairs_overlay(train)) return {};

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
                if (!is_color(out_color)) return {};                  // bad fill -> reject
                if (!any_painted) { fill = out_color; any_painted = true; }
                else if (out_color != fill) return {};                // non-uniform -> reject
            }
            if (!any_painted) continue;                            // nothing painted -> skip

            uint32_t object_shape = shape_occupancy_hash(width, height, occupied);
            if (!learn_shape_color(rule, object_shape, (int8_t)fill)) return {};
        }
    }
    if (rule.holds()) rule.claimed_by = shape_color_map_rule::interior_fill;
    return rule;
}

// A repeating pattern carried across the whole output, with its colors remapped.
struct motif_continuation_rule {
    bool held = false;
    int8_t color_map[COLOR_COUNT] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    bool holds() const { return held; }
};

// Checks that the input is one tiling pattern and reads the recolor off the output.
inline motif_continuation_rule detect_motif_continuation(
        const vector<example>& train, const correspondence_election& correspondence) {
    motif_continuation_rule rule;
    if (train.empty() || correspondence.pairs.size() != train.size() ||
        correspondence.segmentation != segmentation_methods::minimal_repeating_set) return rule;
    for (size_t i = 0; i < train.size(); ++i) {
        const auto& pair = correspondence.pairs[i];
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
        int width, height;
        repeat_periods(input, -2, width, height, /*must_fit_twice=*/false);
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

// The output is a grid of input-sized blocks, each one a turned copy of the
// input or plain background. Every pose that draws a block is kept, not just one.
struct input_blocks_rule {
    bool held = false;
    // One cell-level map shared by every copy; -1 means the color keeps itself.
    int8_t color_map[COLOR_COUNT] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
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
};

// Cuts each output into input-sized blocks and records how every block is drawn.
inline input_blocks_rule detect_input_blocks(const vector<example>& train,
        int input_background, int output_background) {
    if (train.empty()) return {};
    for (const auto& pair : train)
        if (pair.input.width <= 0 || pair.input.height <= 0 ||
            pair.output.width % pair.input.width || pair.output.height % pair.input.height) return {};
    // Background blocks need a ground both sides agree on.
    const int ground = is_color(input_background) && input_background == output_background
                       ? input_background : -1;
    auto is_ground = [&](const grid& output, int top, int left, int width, int height) {
        if (ground < 0) return false;
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                if (output.data[top + r][left + c] != ground) return false;
        return true;
    };
    // Visits each cell of the input posed into the block at (top, left) as
    // (was, now); stops and says false as soon as `cell` does.
    auto each_posed_cell = [](const example& pair, int top, int left, int pose, auto cell) {
        const auto& input = pair.input;
        if (posed_size(pose, input.width, input.height) != make_pair(input.width, input.height))
            return false;
        for (int r = 0; r < input.height; ++r)
            for (int c = 0; c < input.width; ++c) {
                const auto [row, col] = pose_source(pose, input.width, input.height, r, c);
                if (!cell(input.data[row][col], pair.output.data[top + r][left + c])) return false;
            }
        return true;
    };

    auto read_with = [&](const int8_t* map) {
        input_blocks_rule candidate;
        copy_n(map, COLOR_COUNT, candidate.color_map);
        candidate.figure_poses = ground < 0 ? 0 : 0xFF;
        for (const auto& pair : train) {
            const int width = pair.input.width, height = pair.input.height;
            input_blocks_rule::layout at;
            at.rows = (int8_t)(pair.output.height / height);
            at.cols = (int8_t)(pair.output.width / width);
            for (int r = 0; r < at.rows; ++r)
                for (int c = 0; c < at.cols; ++c) {
                    for (int pose = 0; pose < POSE_COUNT; ++pose)
                        if (each_posed_cell(pair, r * height, c * width, pose, [&](int was, int now) {
                                return (map[was] >= 0 ? map[was] : was) == now; }))
                            at.poses[r][c] |= (uint8_t)(1 << pose);
                    if (!at.poses[r][c] && !is_ground(pair.output, r * height, c * width, width, height))
                        return input_blocks_rule{};
                }
            for (int pose = 0; pose < POSE_COUNT; ++pose) {
                const auto [posed_width, posed_height] = posed_size(pose, width, height);
                bool figure = posed_width == at.cols && posed_height == at.rows;
                for (int r = 0; r < at.rows && figure; ++r)
                    for (int c = 0; c < at.cols && figure; ++c) {
                        const auto [row, col] = pose_source(pose, width, height, r, c);
                        figure = pair.input.data[row][col] != ground
                                 ? (at.poses[r][c] >> pose & 1) != 0 : at.poses[r][c] == 0;
                    }
                if (!figure) candidate.figure_poses &= (uint8_t)~(1 << pose);
            }
            candidate.pairs.push_back(at);
        }
        candidate.held = true;
        return candidate;
    };

    // The map is read off the first block that is not background, once per pose
    // it could be; the reading that changes the fewest colors wins.
    const auto& first = train.front();
    int top = 0, left = 0;
    while (is_ground(first.output, top, left, first.input.width, first.input.height)) {
        left += first.input.width;
        if (left >= first.output.width) { left = 0; top += first.input.height; }
        if (top >= first.output.height) return {};
    }
    input_blocks_rule best;
    int fewest_changes = COLOR_COUNT + 1;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        int8_t map[COLOR_COUNT] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
        const bool consistent = each_posed_cell(first, top, left, pose, [&](int was, int now) {
            if (!is_color(was) || !is_color(now) || (map[was] >= 0 && map[was] != now)) return false;
            map[was] = (int8_t)now;
            return true;
        });
        if (!consistent) continue;
        int changes = 0;
        for (int was = 0; was < COLOR_COUNT; ++was) changes += map[was] >= 0 && map[was] != was;
        if (changes >= fewest_changes) continue;
        if (input_blocks_rule candidate = read_with(map); candidate.holds()) {
            best = move(candidate);
            fewest_changes = changes;
        }
    }
    return best;
}

// Every piece slammed its own cells against one side of its own box.
// Only held when sliding every piece that way redraws every output exactly.
struct piece_slam_rule {
    // True when the rows slide sideways, false when the columns slide up or down.
    bool rows = true;
    // +1 toward the right or the bottom, -1 toward the left or the top.
    int8_t step = 0;
    bool holds() const { return step != 0; }
};

// Tries all four slide directions and keeps the one that redraws every output.
inline piece_slam_rule detect_piece_slam(const vector<example>& train,
        const correspondence_election& correspondence, int8_t input_background, int output_background) {
    // The slide happens in each piece's own box, and the cells it leaves are ground.
    if (!boxed_by_its_cells(correspondence.segmentation) ||
        !is_color(output_background) || !pairs_overlay(train)) return {};
    // Draws each output from its input's pieces slid that way, noting whether any
    // cell moved at all; false at the first cell that comes out wrong.
    auto draws_every_output = [&](bool rows, int step, bool& moved) {
        for (const auto& pair : train) {
            vector<object_segmentation::object_info> info;
            vector<object> pieces;
            object_segmentation::segment_for_correspondence(pair.input, input_background,
                                                            correspondence.segmentation, info, &pieces);
            grid drawn;
            drawn.width = pair.output.width;
            drawn.height = pair.output.height;
            for (int r = 0; r < drawn.height; ++r)
                for (int c = 0; c < drawn.width; ++c) drawn.data[r][c] = output_background;
            bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            for (const object& piece : pieces) {
                const int lines = rows ? piece.summary.h : piece.summary.w;
                const int length = rows ? piece.summary.w : piece.summary.h;
                for (int line = 0; line < lines; ++line) {
                    // Scanning from the side the cells slide toward, each takes the
                    // next free spot and becomes the wall for the ones behind it.
                    int wall = step > 0 ? length - 1 : 0;
                    for (int i = 0; i < length; ++i) {
                        const int at = step > 0 ? length - 1 - i : i;
                        const int color = rows ? piece.cells[line][at] : piece.cells[at][line];
                        if (color == TRANSPARENT) continue;
                        const int to = step > 0 ? min(at + step, wall) : max(at + step, wall);
                        wall = to - step;
                        const int row = piece.summary.y0 + (rows ? line : to);
                        const int col = piece.summary.x0 + (rows ? to : line);
                        // Two pieces landing on one cell would leave the paint order to decide.
                        if (painted[row][col]) return false;
                        painted[row][col] = true;
                        drawn.data[row][col] = color;
                        moved = moved || to != at;
                    }
                }
            }
            for (int r = 0; r < drawn.height; ++r)
                for (int c = 0; c < drawn.width; ++c)
                    if (drawn.data[r][c] != pair.output.data[r][c]) return false;
        }
        return true;
    };
    for (const bool rows : {true, false})
        for (const int step : {1, -1}) {
            bool moved = false;
            if (draws_every_output(rows, step, moved) && moved)
                return {rows, (int8_t)step};
        }
    return {};
}

// Collects facts, rules, and elections about one puzzle.
struct inference {
    output_dimension_rule size;
    color_facts colors;
    shape_color_map_rule shape_color_map;
    unchanged_cells_facts unchanged;
    color_diff_facts color_diff;
    color_swap_rule color_swaps;
    motif_continuation_rule motif_continuation;
    piece_slam_rule piece_slam;
    input_blocks_rule input_blocks;
    correspondence_election correspondence;
    background_election background;
};

// Asks every question about the training pairs, in order, and returns the answers.
inline inference infer(const vector<example>& train) {
    inference inf;
    inf.size = output_dimension_rule::infer_rule(train);
    inf.colors = detect_color_facts(train);  // rule-independent
    // Background: input-side and output-side elected independently by role,
    // ranked lists kept alive (see step 2 note on background_election).
    // input_elected() is the INPUT side's verdict — what segmentation excludes.
    inf.background = elect_background(train);
    // by_size excludes the background, so it can only rank after the election.
    detect_output_colors_by_size(train, inf.background.input_elected(), inf.colors);
    // Edge level of correspondence, under the elected background, with the way
    // of cutting the grid up chosen by what it manages to explain.
    inf.correspondence = select_segmentation(train, inf.background.input_elected());
    inf.motif_continuation = detect_motif_continuation(train, inf.correspondence);
    inf.piece_slam = detect_piece_slam(train, inf.correspondence,
        inf.background.input_elected(), inf.background.output_elected());
    inf.input_blocks = detect_input_blocks(train,
        inf.background.input_elected(), inf.background.output_elected());

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
            detect_shape_color_map(train, inf.background.input_elected());
        // Fallback: same shape->color table, but keyed on each object's own
        // shape and applied as an interior fill (ShapeToInteriorColorMap)
        // instead of a body recolor. Only run when the body-recolor
        // hypothesis didn't validate; the winner is recorded in claimed_by,
        // which is what tells the two ops apart downstream.
        if (!inf.shape_color_map.holds())
            inf.shape_color_map = detect_shape_to_interior_color_map(
                train, inf.background.input_elected());
    }
    // Scale / Fixed / BoundingBox: no rule-specific detectors yet.
    // Scale could later reuse unchanged_cells through the ratio (cell<->block).
    return inf;
}

}  // namespace arc
