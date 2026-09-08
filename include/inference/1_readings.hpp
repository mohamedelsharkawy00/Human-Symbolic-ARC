// 1_readings.hpp — how to read a grid at all: count it, box it, walk it, name its shape.
// Design story in notes/1 - Overall Design/Inference Master.md
//
// The shared base every channel stands on. Nothing here asks a question about
// the train pairs; these are the tools the questions get asked with.

#pragma once
#include "core_types.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace arc {
using namespace std;

// Human reading order.
// Plural S = whole train set; no S = one pair.
enum class direction {
    across_input_output,
    across_inputs,
    across_outputs,
    across_inputs_outputs,
};

// Every cell wearing a color, in reading order, with one color left out if
// asked. Single home of the walk: the bounds are the grid's own, held to what a
// grid can hold, so no two readings can disagree about which cells count.
template <class See>
inline void walk_colored_cells(const grid& one_grid, int skip, See see) {
    for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
            const int color = one_grid.data[row][col];
            if (is_color(color) && color != skip) see(row, col, color);
        }
}

// How many cells of each color a grid holds. `skip` leaves one color out of the
// tally — the background, wherever a caller wants it ignored; TRANSPARENT
// counts all ten.
inline void count_colors(const grid& one_grid, int counts[COLOR_COUNT],
                         int skip = TRANSPARENT) {
    for (int i = 0; i < COLOR_COUNT; ++i) counts[i] = 0;
    walk_colored_cells(one_grid, skip, [&](int, int, int color) { counts[color]++; });
}

// The smallest box each color's cells fit in. A color with no cells keeps the
// empty box it starts with, and empty() is how you ask.
struct color_box {
    int row0, col0, row1, col1;

    // Its own size, so no caller has to spell the +1 out and none can spell it
    // differently. A box nobody wrote to has no cells, and says so itself
    // rather than leaving the reader to check the tally beside it.
    bool empty() const { return row1 < row0 || col1 < col0; }
    int height() const { return empty() ? 0 : row1 - row0 + 1; }
    int width()  const { return empty() ? 0 : col1 - col0 + 1; }
    int area()   const { return width() * height(); }
};

inline void color_boxes(const grid& one_grid, color_box boxes[COLOR_COUNT],
                        int skip = TRANSPARENT) {
    for (int i = 0; i < COLOR_COUNT; ++i) boxes[i] = { MAX_GRID_DIM, MAX_GRID_DIM, -1, -1 };
    walk_colored_cells(one_grid, skip, [&](int row, int col, int color) {
        color_box& box = boxes[color];
        box.row0 = min(box.row0, row); box.row1 = max(box.row1, row);
        box.col0 = min(box.col0, col); box.col1 = max(box.col1, col);
    });
}

// Do input and output overlay cell-for-cell in EVERY pair? The channels below
// that read out.data at an input coordinate need this to be true, and each one
// says so itself rather than trusting the caller — infer() gates on the size
// rule, but a detector called from anywhere else must still be safe.
inline bool pairs_overlay(const vector<example>& train) {
    if (train.empty()) return false;
    return ranges::all_of(train, [](const example& ex) {
        // Non-empty as well as equal: an empty pair overlays itself vacuously,
        // and a channel that divides by the cell count would hand back a
        // nonsense number instead of declining.
        return ex.input.width > 0 && ex.input.height > 0 &&
               ex.input.width  == ex.output.width &&
               ex.input.height == ex.output.height;
    });
}

// Walk out from the cells already sitting in the queue until nothing new can be
// reached, corners always counting. `claim` both decides and marks: it answers
// "is this cell part of the walk", and a yes means the walk has taken it, so the
// caller's own mark is what keeps a cell from being queued twice. `see` gets
// every cell once, in the order the walk reached it. Returns how many cells the
// walk ended with, left in the queue for the caller to read back.
template <class Claim, class See>
inline int walk_out(int rows, int cols, int queue_row[], int queue_col[], int seeded,
                    Claim claim, See see) {
    int head = 0, tail = seeded;
    while (head < tail) {
        const int row = queue_row[head], col = queue_col[head];
        ++head;
        see(row, col);
        for (int dir = 0; dir < 8; ++dir) {
            const int next_row = row + NEIGHBOUR_ROW_8[dir];
            const int next_col = col + NEIGHBOUR_COL_8[dir];
            if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) continue;
            if (!claim(next_row, next_col)) continue;
            queue_row[tail] = next_row; queue_col[tail] = next_col;
            ++tail;
        }
    }
    return tail;
}

// The hash recipe lives in core_types.hpp (single home); this adapts a packed
// bool occupancy array to it.
inline uint32_t shape_occupancy_hash(int width, int height, const bool* occupied) {
    return ::shape_occupancy_hash(width, height, [&](int i) { return occupied[i]; });
}

// Canonical (orientationless, positionless) shape identity: the smallest
// shape_occupancy_hash over the eight ways round, done by index arithmetic —
// no pose grids are built. Two objects share a shape, in any orientation, iff
// their canonical hashes are equal.
inline uint32_t canonical_shape_hash(int width, int height, const bool* occupied) {
    uint32_t best = 0xFFFFFFFFu;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, width, height);
        uint32_t hash = ::shape_occupancy_hash(posed_width, posed_height, [&](int i) {
            const auto [row, col] = pose_source(pose, width, height, i / posed_width,
                                                i % posed_width);
            return occupied[row * width + col];
        });
        best = min(best, hash);
    }
    return best;
}

}  // namespace arc
