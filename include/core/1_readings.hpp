// This file describes how to read a grid.
#pragma once
#include "0_core.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>

namespace arc {
using namespace std;

// An object is a related group of cells.
// Humans mostly decompose an ARC grid into objects when reading the grid,
// so the code will match that level of abstraction
// vs carrying each cell as its own entity.

// An object's bounding box and cell count, usable without storing its cells.
struct object_summary {
    int8_t  x0 = 0, y0 = 0;  // top-left in grid coordinates
    int8_t  w = 0, h = 0;    // box width and height
    int16_t count = 0;       // number of cells in the object
};

// The sole drawing with no equal in the list. Compare full frames and cells,
// including transparent cells, in local coordinates; placement does not matter.
// Ambiguous lists (including two different objects) add no selection.
template<class Width, class Height, class ColorAt>
ARC_HOST_DEVICE inline int unique_drawing(int count, Width width, Height height, ColorAt color_at) {
    if (count < 3) return -1;
    int found = -1;
    for (int i = 0; i < count; ++i) {
        bool repeated = false;
        for (int j = 0; j < count && !repeated; ++j) {
            if (i == j || width(i) != width(j) || height(i) != height(j)) continue;
            bool same = true;
            for (int r = 0; r < height(i) && same; ++r)
                for (int c = 0; c < width(i) && same; ++c)
                    same = color_at(i, r, c) == color_at(j, r, c);
            repeated = same;
        }
        if (!repeated) {
            if (found >= 0) return -1;
            found = i;
        }
    }
    return found;
}

// Align like anchors of two spans: 0 = first cell, 1 = middle cell,
// 2 = last cell. Even spans use the earlier of the two middle cells.
// Copy placement shares this arithmetic between the reading and the GPU.
ARC_HOST_DEVICE constexpr int aligned_copy_start(int source_size, int target_start,
                                                 int target_size, int anchor) {
    return target_start + (target_size - 1) * anchor / 2 - (source_size - 1) * anchor / 2;
}

// A cell's place in the largest grid as one number, counted row by row. Lists
// of cells hold these, so a cell costs two bytes rather than a pair, and the
// packed store below finds its cells the same way.
ARC_HOST_DEVICE constexpr int cell_index(int row, int col) { return row * MAX_GRID_DIM + col; }
ARC_HOST_DEVICE constexpr int cell_row(int index) { return index / MAX_GRID_DIM; }
ARC_HOST_DEVICE constexpr int cell_col(int index) { return index % MAX_GRID_DIM; }

// A cell is 4 bits, so two of them share a byte,
// in fixed size storage held by both CPU and GPU.
// This optimization is needed to lower the consumption
// of registers and the spilling to memory, which hurts performance.
// Also, parallel writes by GPU lanes must
// take separate whole bytes each to avoid racing.
// Four-byte aligned, and the store rounded up to a whole number of words, so
// that the 32-bit word holding any cell lies wholly inside `bytes`. The device
// has no byte-wide atomic, so a lane painting a cell that shares its byte with
// another lane's cell has to compare-and-swap the enclosing word; without this
// that word could reach past the store into whatever field sits next door.
struct alignas(4) object_cells {
    static constexpr int CELLS = MAX_GRID_CELLS;
    static constexpr int BYTES = ((CELLS + 1) / 2 + 3) / 4 * 4;
    uint8_t bytes[BYTES];

    // A cell occupies half a byte, so reads and writes need a proxy,
    // because C++ can't give a reference to half a byte.
    struct const_cell_ref {
        const uint8_t* byte;
        int shift;

        ARC_HOST_DEVICE operator int8_t() const { return dec(*byte, shift); }
    };

    struct cell_ref {
        uint8_t* byte;
        int shift;

        ARC_HOST_DEVICE operator int8_t() const { return dec(*byte, shift); }

        ARC_HOST_DEVICE void operator=(int8_t value) {
            const uint8_t encoded = object_cells::enc(value);
            *byte = (uint8_t)((*byte & ~(0xF << shift)) | (encoded << shift));
        }

        // Without these, `a[r][c] = b[r][c]` would take the compiler-written copy
        // assignment and rebind the proxy -- moving where this cell_ref points
        // instead of writing a cell. Both spellings write the value through.
        ARC_HOST_DEVICE void operator=(const cell_ref& other) { *this = (int8_t)other; }
        ARC_HOST_DEVICE void operator=(const const_cell_ref& other) { *this = (int8_t)other; }
    };

    // Row proxies allow cells[row][col]; invalid flat indices fall back to cell 0.
    // One template makes the writable row and the read-only one, so the two
    // cannot find a cell differently.
    template <class Byte, class Cell>
    struct row_of_cells {
        Byte* base;
        int row;

        ARC_HOST_DEVICE Cell operator[](int col) const {
            int index = cell_index(row, col);
            if (index < 0 || index >= CELLS) index = 0;
            return { base + (index >> 1), (index & 1) * 4 };
        }
    };

    ARC_HOST_DEVICE row_of_cells<uint8_t, cell_ref> operator[](int row) { return { bytes, row }; }
    ARC_HOST_DEVICE row_of_cells<const uint8_t, const_cell_ref> operator[](int row) const {
        return { bytes, row };
    }

    // Four bits are unsigned, so TRANSPARENT (-1) has nowhere to stay. What goes
    // in the byte is therefore the real value plus one: TRANSPARENT becomes 0,
    // black becomes 1, and so on. dec reads the half at `shift` back.
    ARC_HOST_DEVICE static constexpr uint8_t enc(int8_t value) {
        return (uint8_t)(((int)value + 1) & 0xF);
    }
    ARC_HOST_DEVICE static constexpr int8_t dec(uint8_t byte, int shift) {
        return (int8_t)(((byte >> shift) & 0xF) - 1);
    }

    // A byte of two TRANSPARENT cells. Named because clearing by whole bytes is
    // how lanes wipe a canvas without two of them ever sharing one. A function
    // rather than a constant: its body is compiled once the class is complete,
    // where a member initializer could not yet call enc.
    ARC_HOST_DEVICE static constexpr uint8_t empty_byte() {
        return (uint8_t)(enc(TRANSPARENT) | (enc(TRANSPARENT) << 4));
    }

    ARC_HOST_DEVICE void clear() {
        for (int i = 0; i < (int)sizeof(bytes); ++i) bytes[i] = empty_byte();
    }
};

template <class Mask>
ARC_HOST_DEVICE inline int8_t majority_vote(const Mask& mask, int row, int col,
                                             int factor, int height, int width) {
    int counts[COLOR_COUNT] = {};
    int non_transparent = 0;
    for (int dr = 0; dr < factor; ++dr)
        for (int dc = 0; dc < factor; ++dc) {
            const int source_row = row * factor + dr;
            const int source_col = col * factor + dc;
            if (source_row >= height || source_col >= width) continue;
            const int8_t color = (int8_t)mask[source_row][source_col];
            if (color == TRANSPARENT) continue;
            ++counts[color % COLOR_COUNT];
            ++non_transparent;
        }
    if (non_transparent == 0) return TRANSPARENT;
    int winner = -1, best = -1;
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (counts[color] > best) { best = counts[color]; winner = color; }
    return (int8_t)winner;
}

// MSVC C4324: the device only grabs memory 4 bytes at a time, so object_cells
// has to start where the 4s start. Lining it up leaves 2 bytes empty. The
// compiler says "you left 2 empty". We say "yes, on purpose".
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
struct object {
    object_summary summary;
    // Stored relative to the box's top-left.
    object_cells cells;

    // Build from the grid-cell indices returned by each_object.
    // Cells outside the group stay transparent, including holes in its box.
    static object from_cells(const grid& source, const object_summary& box,
                             const int16_t* indices, int count) {
        object made;
        made.summary = box;
        made.cells.clear();
        for (int i = 0; i < count; ++i) {
            const int row = cell_row(indices[i]), col = cell_col(indices[i]);
            made.cells[row - box.y0][col - box.x0] = (int8_t)source.data[row][col];
        }
        return made;
    }

    // Object cell accessors here take grid coordinates, not box-local coordinates; they do the shift.
    // Gaps or holes within the box read as TRANSPARENT, and so does anything outside it.
    ARC_HOST_DEVICE int8_t color_at(int row, int col) const {
        const int local_row = row - summary.y0;
        const int local_col = col - summary.x0;
        if (local_row < 0 || local_row >= summary.h ||
            local_col < 0 || local_col >= summary.w) return (int8_t)TRANSPARENT;
        return cells[local_row][local_col];
    }

    ARC_HOST_DEVICE bool holds(int row, int col) const {
        return color_at(row, col) != (int8_t)TRANSPARENT;
    }

    // The color of its first cell in reading order, which is the cell the walk
    // started from and the one an op reads a piece's color off. TRANSPARENT
    // when it has no cells.
    ARC_HOST_DEVICE int8_t first_color() const {
        for (int row = 0; row < summary.h; ++row)
            for (int col = 0; col < summary.w; ++col)
                if (cells[row][col] != (int8_t)TRANSPARENT) return cells[row][col];
        return (int8_t)TRANSPARENT;
    }

    // Every cell the object fills, in reading order: see(row, col, color), in
    // grid coordinates.
    template <class See>
    void each_cell(See see) const {
        for (int row = 0; row < summary.h; ++row)
            for (int col = 0; col < summary.w; ++col) {
                const int8_t color = cells[row][col];
                if (color != TRANSPARENT) see(summary.y0 + row, summary.x0 + col, color);
            }
    }

    // The object on its own, as a grid just big enough to hold it: its own
    // colors where it reaches, TRANSPARENT in the gaps and nowhere else's cells.
    // The cells are already this, packed four bits each; this spells them out
    // for the code that walks a grid row by row. Host only -- it allocates a
    // whole grid, which is what the packing exists to avoid on the device.
    grid drawn() const {
        grid out(summary.w, summary.h);
        for (int row = 0; row < summary.h; ++row)
            for (int col = 0; col < summary.w; ++col)
                out.data[row][col] = cells[row][col];
        return out;
    }
};

// End of warning suppression
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// What counts as one object. The walk answers all three the same way, so no
// two readings can disagree about which cells belong together.
enum class grouping : int8_t {
    // Cells that touch, all of them the same color. The usual object.
    touching_same_color,
    // Cells that touch, whatever their colors. One object may be many colors.
    touching_any_color,
    // Every cell of a color at once, touching or not. One object per color.
    all_of_one_color,
};

// Caller-owned so host and device can lend memory they already hold.
struct object_walk_scratch {
    bool taken[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_CELLS];
    int16_t queue_col[MAX_GRID_CELLS];
    int16_t cells[MAX_GRID_CELLS];
};

// Grouping of cells into an object is 8 way, "is this one object?".
//
// Worked out rather than written down, because a constexpr array lives in
// host memory and device code cannot index one -- and the walk below runs on
// both sides. The eight are the 3x3 around a cell with the cell itself left
// out, so dir 0..7 steps over the missing middle and reads off row and column.
ARC_HOST_DEVICE constexpr int neighbour_row_8(int dir) {
    return (dir + (dir >= 4 ? 1 : 0)) / 3 - 1;
}
ARC_HOST_DEVICE constexpr int neighbour_col_8(int dir) {
    return (dir + (dir >= 4 ? 1 : 0)) % 3 - 1;
}

// Adjacency between two cells is 4 way, "what shares an edge with this cell?".
// Up, down, left, right. A diagonal cell touches only at a corner.
//
// Why four here and eight for grouping. Picture a ring of blocks meeting only
// at their corners. Ask "is this one ring" and the answer is yes, corners
// count, which is the eight. Ask "can the water inside get out" and corners
// must not count, or the water escapes through the very tip just called a
// wall and the ring has no inside at all. So anything measuring a border --
// what frays against what, what surrounds what, whether background leaks to
// the grid edge -- takes the four. Whatever the shapes use, the space around
// them uses the other one. Let both use eight and no shape has an inside any
// more, so holes stop existing.
ARC_HOST_DEVICE constexpr int neighbour_row_4(int dir) {
    return dir < 2 ? dir * 2 - 1 : 0;
}
ARC_HOST_DEVICE constexpr int neighbour_col_4(int dir) {
    return dir < 2 ? 0 : (dir - 2) * 2 - 1;
}

// Walks connected cells: eight directions for objects, four for surrounding space.
// `claim` decides membership into that group and must mark what it accepts, or the walk never ends.
template <int Directions = 8, class Claim, class See>
ARC_HOST_DEVICE inline int walk_out(int rows, int cols, int16_t queue_row[], int16_t queue_col[], int seeded,
                    Claim claim, See see) {
    static_assert(Directions == 4 || Directions == 8);
    int head = 0, tail = seeded;
    while (head < tail) {
        const int row = queue_row[head], col = queue_col[head];
        ++head;
        see(row, col);
        for (int dir = 0; dir < Directions; ++dir) {
            const int next_row = row + (Directions == 4 ? neighbour_row_4(dir) : neighbour_row_8(dir));
            const int next_col = col + (Directions == 4 ? neighbour_col_4(dir) : neighbour_col_8(dir));
            if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) continue;
            if (!claim(next_row, next_col)) continue;
            queue_row[tail] = (int16_t)next_row; queue_col[tail] = (int16_t)next_col;
            ++tail;
        }
    }
    return tail;
}

// Whether a cell sits in the outermost ring of a height x width grid.
ARC_HOST_DEVICE constexpr bool on_border(int row, int col, int height, int width) {
    return row == 0 || col == 0 || row == height - 1 || col == width - 1;
}

// The boundary used by Outline: any missing eight-way neighbour makes an edge.
// The caller supplies occupancy, including false outside its own frame.
template <class Occupied>
ARC_HOST_DEVICE inline bool on_object_edge(int row, int col, Occupied occupied) {
    for (int dir = 0; dir < 8; ++dir)
        if (!occupied(row + neighbour_row_8(dir), col + neighbour_col_8(dir))) return true;
    return false;
}

// Recolor and AddColor's relative color sources, shared by reading and execution.
// Ties choose the lowest color, and an empty object stays transparent.
template <class ColorAt>
ARC_HOST_DEVICE inline int8_t object_paint_color(int height, int width, int8_t value, ColorAt color_at) {
    if (value < COLOR_COUNT) return value;
    int counts[COLOR_COUNT] = {};
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int color = color_at(row, col);
            if (color >= 0 && color < COLOR_COUNT) ++counts[color];
        }
    int best = TRANSPARENT;
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (counts[color] && (best < 0 || (value == COLOR_COUNT
            ? counts[color] < counts[best] : counts[color] > counts[best]))) best = color;
    return (int8_t)best;
}

// Mark cells reachable from the border without crossing a blocked cell.
// Space uses four-way connectivity, so it cannot leak through touching corners.
template <class Blocked>
ARC_HOST_DEVICE inline void mark_outside(int height, int width, Blocked blocked,
                         bool outside[MAX_GRID_DIM][MAX_GRID_DIM]) {
    int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) outside[row][col] = false;
    auto claim = [&](int row, int col) {
        if (outside[row][col] || blocked(row, col)) return false;
        outside[row][col] = true;
        return true;
    };
    int seeded = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            if (!on_border(row, col, height, width)) continue;
            if (!claim(row, col)) continue;
            queue_row[seeded] = (int16_t)row;
            queue_col[seeded++] = (int16_t)col;
        }
    walk_out<4>(height, width, queue_row, queue_col, seeded, claim, [](int, int) {});
}

// Count four-connected enclosed spaces in an occupancy mask. Shared by
// inference and SelectObjectsWithHoles so learned keys mean the same at runtime.
template <class Occupied>
ARC_HOST_DEVICE inline int object_hole_count(int mh, int mw, Occupied occupied) {
    if (mw <= 0 || mh <= 0) return 0;
    // mark: 1 = wall (painted cell), 2 = outside (reached from bbox edge), 3 = counted hole.
    uint8_t mark[MAX_GRID_DIM][MAX_GRID_DIM];
    for (int r = 0; r < mh; ++r)
        for (int c = 0; c < mw; ++c)
            mark[r][c] = occupied(r, c) ? 1 : 0;

    // Outside = monotonic sweeps from the bbox edge through transparent cells.
    for (bool changed = true; changed; ) {
        changed = false;
        for (int r = 0; r < mh; ++r)
            for (int c = 0; c < mw; ++c) {
                if (mark[r][c]) continue;
                bool edge = (r == 0 || r == mh - 1 || c == 0 || c == mw - 1);
                bool nb = (r > 0      && mark[r - 1][c] == 2) ||
                          (r < mh - 1 && mark[r + 1][c] == 2) ||
                          (c > 0      && mark[r][c - 1] == 2) ||
                          (c < mw - 1 && mark[r][c + 1] == 2);
                if (edge || nb) { mark[r][c] = 2; changed = true; }
            }
    }

    // Holes = remaining unmarked components; flood each with an explicit stack.
    int holes = 0;
    int16_t stack[MAX_GRID_DIM * MAX_GRID_DIM];
    for (int r = 0; r < mh; ++r)
        for (int c = 0; c < mw; ++c) {
            if (mark[r][c]) continue;
            ++holes;
            mark[r][c] = 3;
            int top = 0;
            stack[top++] = (int16_t)(r * MAX_GRID_DIM + c);
            while (top > 0) {
                int p = stack[--top];
                int pr = p / MAX_GRID_DIM, pc = p % MAX_GRID_DIM;
                const int dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
                for (int d = 0; d < 4; ++d) {
                    int nr = pr + dr[d], nc = pc + dc[d];
                    if (nr < 0 || nr >= mh || nc < 0 || nc >= mw || mark[nr][nc]) continue;
                    mark[nr][nc] = 3;
                    stack[top++] = (int16_t)(nr * MAX_GRID_DIM + nc);
                }
            }
        }
    return holes;
}

// A box in grid coordinates, with inclusive bottom and right edges.
// An empty box has its last row or column before its first.
struct cell_box {
    int row0 = 0, col0 = 0, row1 = -1, col1 = -1;

    // Its own size, so no caller has to spell the +1 out and none can spell it
    // differently. A box nobody wrote to has no cells, and says so itself
    // rather than leaving the reader to check the tally beside it.
    ARC_HOST_DEVICE bool empty() const { return row1 < row0 || col1 < col0; }
    ARC_HOST_DEVICE int height() const { return empty() ? 0 : row1 - row0 + 1; }
    ARC_HOST_DEVICE int width()  const { return empty() ? 0 : col1 - col0 + 1; }
    ARC_HOST_DEVICE int area()   const { return width() * height(); }

    ARC_HOST_DEVICE void include_cell(int row, int col) {
        if (empty()) {
            row0 = row1 = row;
            col0 = col1 = col;
            return;
        }
        if (row < row0) row0 = row;
        if (row > row1) row1 = row;
        if (col < col0) col0 = col;
        if (col > col1) col1 = col;
    }
};

// Fit names its target: input frame, live output frame, or painted output
// bounds. Reading and execution ask the same helper; empty paint has no bounds.
template <class Canvas>
ARC_HOST_DEVICE inline cell_box fit_frame(int mode, int input_width, int input_height,
                                         const Canvas& canvas) {
    if (mode == 0) return {0, 0, input_height - 1, input_width - 1};
    if (mode == 1) return {0, 0, canvas.height - 1, canvas.width - 1};
    cell_box box;
    if (mode == 2)
        for (int r = 0; r < canvas.height; ++r)
            for (int c = 0; c < canvas.width; ++c)
                if (is_color(canvas.data[r][c])) box.include_cell(r, c);
    return box;
}
ARC_HOST_DEVICE inline int fit_source_cell(int target_cell, int source_size, int target_size) {
    return target_cell * source_size / target_size;
}

// An object's box as cell_box has it, and back again with its cell count.
inline cell_box box_of(const object_summary& summary) {
    return { summary.y0, summary.x0, summary.y0 + summary.h - 1, summary.x0 + summary.w - 1 };
}
ARC_HOST_DEVICE inline object_summary summary_of(const cell_box& box, int count) {
    return { (int8_t)box.col0, (int8_t)box.row0, (int8_t)box.width(), (int8_t)box.height(),
             (int16_t)count };
}

// Share the executor's span grouping between inference and Layout/SelectRow/SelectColumn.
ARC_HOST_DEVICE inline int overlapping_bands(int count, const int16_t* lows,
                                               const int16_t* highs, int8_t* band_of,
                                               int* order) {
    for (int i = 0; i < count; ++i) order[i] = i;
    for (int i = 1; i < count; ++i) {
        int j = i;
        while (j > 0) {
            const int a = order[j], b = order[j - 1];
            if (lows[a] >= lows[b]) break;
            order[j] = b;
            order[j - 1] = a;
            --j;
        }
    }
    int bands = 0, band_hi = -1;
    for (int m = 0; m < count; ++m) {
        const int i = order[m];
        const int need = (highs[i] - lows[i] + 2) / 2;
        if (bands == 0 || band_hi - lows[i] + 1 < need) {
            ++bands;
            band_hi = highs[i];
        } else if (highs[i] > band_hi) {
            band_hi = highs[i];
        }
        band_of[i] = (int8_t)(bands - 1);
    }
    return bands;
}

// THE EIGHT WAYS A PIECE CAN GO, numbered as MoveToEdge and ExtendTo take them:
// the four sides, then the diagonals.
struct heading {
    enum : int8_t { up, down, left, right, up_left, up_right, down_left, down_right, count };
};
inline constexpr const char* const heading_names[heading::count] = {
    "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" };

// One step that way. A switch and not a table, so the kernel reads the same
// steps the host does; a heading out of range goes up.
struct heading_step { int down, across; };
ARC_HOST_DEVICE inline heading_step step_of(int way) {
    switch (way) {
    case heading::down:       return {  1,  0 };
    case heading::left:       return {  0, -1 };
    case heading::right:      return {  0,  1 };
    case heading::up_left:    return { -1, -1 };
    case heading::up_right:   return { -1,  1 };
    case heading::down_left:  return {  1, -1 };
    case heading::down_right: return {  1,  1 };
    default:                  return { -1,  0 };
    }
}

// How far a box goes that way before its far side meets the edge of a canvas
// this size, as MoveToEdge takes it: negative when that side already lies past
// the edge, and on a diagonal whichever edge comes first.
ARC_HOST_DEVICE inline int steps_to_edge(const cell_box& box, int width, int height, int way) {
    const heading_step step = step_of(way);
    const int room_down = step.down > 0 ? height - 1 - box.row1 : box.row0;
    const int room_across = step.across > 0 ? width - 1 - box.col1 : box.col0;
    if (step.across == 0) return room_down;
    if (step.down == 0) return room_across;
    return room_down < room_across ? room_down : room_across;
}

// WHERE SHIFT MOVES A BOX, its own size away, the ways Shift numbers them: up,
// right, down, left, then down-right, down-left, up-right, up-left. The kernel
// and the reading that replays it take the step from here.
ARC_HOST_DEVICE inline heading_step shift_by(int way, int width, int height) {
    switch (way) {
    case 0:  return { -height, 0 };
    case 1:  return { 0, width };
    case 2:  return { height, 0 };
    case 3:  return { 0, -width };
    case 4:  return { height, width };
    case 5:  return { height, -width };
    case 6:  return { -height, width };
    case 7:  return { -height, -width };
    default: return { 0, 0 };
    }
}

// Find every touching group and hand it to `take`: summary, cells as pointer
// and length. Grid is anything with width, height and data[row][col], so host
// and device share one answer about what one object is.
//
// The cells point into `scratch`, which the next group overwrites -- `take`
// must COPY anything it keeps. That is what lets this run without allocating.
//
// `background` is the color the grid was found to sit on, read off inference,
// not assumed. Pass -1 to exclude nothing but TRANSPARENT.
template <class Grid, class Take>
ARC_HOST_DEVICE inline void each_object(const Grid& one_grid, int background,
                                 grouping how, object_walk_scratch& scratch, Take take) {
    const bool any_color = (how == grouping::touching_any_color);
    // One object per color takes every cell of the color at once, wherever it
    // is; the touching kinds walk out from the seed cell only.
    const bool whole_color = (how == grouping::all_of_one_color);
    if (one_grid.width <= 0 || one_grid.height <= 0) return;
    bool (&taken)[MAX_GRID_DIM][MAX_GRID_DIM] = scratch.taken;
    for (int r = 0; r < MAX_GRID_DIM; ++r)
        for (int c = 0; c < MAX_GRID_DIM; ++c) taken[r][c] = false;
    int16_t* queue_row = scratch.queue_row;
    int16_t* queue_col = scratch.queue_col;
    int16_t* found_cells = scratch.cells;
    for (int row = 0; row < one_grid.height; ++row)
        for (int col = 0; col < one_grid.width; ++col) {
            if (taken[row][col]) continue;
            const int color = one_grid.data[row][col];
            // TRANSPARENT is not a color at all; `background` is the ground this
            // grid was found to sit on, and -1 from the election means the grid
            // is all figure and nothing is excluded.
            if (color < 0 || color == background) { taken[row][col] = true; continue; }

            taken[row][col] = true;
            queue_row[0] = (int16_t)row; queue_col[0] = (int16_t)col;
            int seen = 0;
            int seeded = 1;
            cell_box box;
            // Every other cell of this color joins before the walk starts, and
            // the walk below then claims nothing, so the group is the color.
            if (whole_color)
                for (int other_row = 0; other_row < one_grid.height; ++other_row)
                    for (int other_col = 0; other_col < one_grid.width; ++other_col) {
                        if (taken[other_row][other_col] ||
                            one_grid.data[other_row][other_col] != color) continue;
                        taken[other_row][other_col] = true;
                        queue_row[seeded] = (int16_t)other_row;
                        queue_col[seeded] = (int16_t)other_col;
                        ++seeded;
                    }
            // One flood fill for the whole project, so no two readings can
            // disagree about which cells belong to one group.
            const int count = walk_out(one_grid.height, one_grid.width,
                queue_row, queue_col, seeded,
                [&](int next_row, int next_col) {
                    if (whole_color || taken[next_row][next_col]) return false;
                    const int there = one_grid.data[next_row][next_col];
                    if (any_color ? (there < 0 || there == background)
                                  : (there != color)) return false;
                    taken[next_row][next_col] = true;
                    return true;
                },
                [&](int seen_row, int seen_col) {
                    found_cells[seen++] = (int16_t)cell_index(seen_row, seen_col);
                    box.include_cell(seen_row, seen_col);
                });
            take(summary_of(box, count), found_cells, seen);
        }
}

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

// Tallies; how much of each color, and where it sits

// Every cell wearing a color, in reading order, with one color left out if
// asked. Single home of the walk: the bounds are the grid's own, held to what a
// grid can hold, so no two readings can disagree about which cells count.
template <class Grid, class See>
ARC_HOST_DEVICE inline void walk_colored_cells(const Grid& one_grid, int skip, See see) {
    for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
            const int color = one_grid.data[row][col];
            if (is_color(color) && color != skip) see(row, col, color);
        }
}

// How many cells of each color a grid holds. `skip` leaves one color out of the
// tally — the background, wherever a caller wants it ignored; TRANSPARENT
// counts all ten.
template <class Grid>
ARC_HOST_DEVICE inline void count_colors(const Grid& one_grid, int counts[COLOR_COUNT],
                         int skip = TRANSPARENT) {
    for (int i = 0; i < COLOR_COUNT; ++i) counts[i] = 0;
    walk_colored_cells(one_grid, skip, [&](int, int, int color) { counts[color]++; });
}

// Which colors occur, using the same cell filtering as counts and color boxes.
inline void present_colors(const grid& one_grid, bool seen[COLOR_COUNT]) {
    ranges::fill_n(seen, COLOR_COUNT, false);
    walk_colored_cells(one_grid, TRANSPARENT,
                       [&](int, int, int color) { seen[color] = true; });
}

// The one color every cell off the ground wears, or TRANSPARENT when there is no
// figure or it wears several. What a person means by "the shape's color": read
// off the grid in hand, so a rule that asks for it names a color no example
// showed as readily as one they all did.
inline int8_t the_figure_color(const grid& one_grid, int ground) {
    int counts[COLOR_COUNT];
    count_colors(one_grid, counts, ground);
    int8_t figure = TRANSPARENT;
    for (int color = 0; color < COLOR_COUNT; ++color) {
        if (!counts[color]) continue;
        if (figure != TRANSPARENT) return TRANSPARENT;
        figure = (int8_t)color;
    }
    return figure;
}

template <class Grid>
ARC_HOST_DEVICE inline void color_boxes(const Grid& one_grid, cell_box boxes[COLOR_COUNT],
                        int skip = TRANSPARENT) {
    for (int i = 0; i < COLOR_COUNT; ++i) boxes[i] = {};
    walk_colored_cells(one_grid, skip, [&](int row, int col, int color) {
        boxes[color].include_cell(row, col);
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
        return !ex.input.empty() && ex.input.same_size(ex.output);
    });
}

// POSES · the ways a grid can sit, and naming the shape that results

// THE EIGHT WAYS A GRID CAN SIT: four turns by two mirrors, packed in three
// bits, pose 0 being the grid as given.
//   1  rows and columns trade places, transpose
//   2  flipped top to bottom
//   4  flipped left to right
// Single home of what the bits mean. Every reader of a pose goes through the
// three below, so none of them can drift on what "the same shape, turned" is.
constexpr int POSE_COUNT = 8;

// Where a cell of the ORIGINAL lands once posed, so a place can be carried into
// the frame a pair is compared in. The exact inverse of pose_source.
ARC_HOST_DEVICE inline std::pair<int, int> posed_place(int pose, int width, int height,
                                  int row, int col) {
    const int down   = (pose & 2) ? height - 1 - row : row;
    const int across = (pose & 4) ? width  - 1 - col : col;
    return (pose & 1) ? std::pair<int, int>{ across, down } : std::pair<int, int>{ down, across };
}

// How big a grid reads once posed. Only the trade changes the size.
inline std::pair<int, int> posed_size(int pose, int width, int height) {
    return (pose & 1) ? std::pair<int, int>{ height, width } : std::pair<int, int>{ width, height };
}

// Which cell of the ORIGINAL a posed cell reads from. Width and height are the
// original's own, which is what the mirrors are measured against.
ARC_HOST_DEVICE inline std::pair<int, int> pose_source(int pose, int width, int height,
                                  int posed_row, int posed_col) {
    int row = (pose & 1) ? posed_col : posed_row;
    int col = (pose & 1) ? posed_row : posed_col;
    if (pose & 2) row = height - 1 - row;
    if (pose & 4) col = width  - 1 - col;
    return { row, col };
}

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

// A GRID LYING ON ITS SIDE is wider than it is tall. NormalizeOrientation[0]
// reads one transposed, so its long way runs down, and a square one as it
// stands. Asked here by the op and by any reading that replays it, so the two
// cannot turn different grids.
ARC_HOST_DEVICE constexpr bool lies_on_its_side(int width, int height) { return width > height; }

// Turn a shape into one number, so "is this the same shape?" becomes "are these
// two numbers equal?". Shape here means which cells of the box around it are
// filled, plus w and h. Not the color, not where it sits on the grid.
// The CPU stores these numbers and the GPU looks them up, so if the two ever
// hashed differently every lookup would miss. Hence one function, called by both.
//
// occ(i) says whether cell i is filled, counting across each row in turn. Every
// caller must count the same way or the two sides hash the same shape apart.
template <typename OccupancyAt>
ARC_HOST_DEVICE inline uint32_t shape_occupancy_hash(int w, int h, OccupancyAt occ) {
    uint32_t hsh = 2166136261u;
    auto mix = [&](uint8_t b) { hsh = (hsh ^ b) * 16777619u; };
    mix((uint8_t)w); mix((uint8_t)h);
    for (int i = 0; i < w * h; ++i) mix(occ(i) ? (uint8_t)1 : (uint8_t)0);
    return hsh;
}

// Canonical (orientationless, positionless) shape identity: the smallest
// shape_occupancy_hash over the eight ways round, done by index arithmetic —
// no pose grids are built. Equal shapes have equal hashes; collisions are possible.
// occupied(row, col) reads the original shape without copying its storage.
template <typename OccupancyAt>
inline uint32_t canonical_shape_hash(int width, int height, OccupancyAt occupied) {
    uint32_t best = 0xFFFFFFFFu;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, width, height);
        uint32_t hash = arc::shape_occupancy_hash(posed_width, posed_height, [&](int i) {
            const auto [row, col] = pose_source(pose, width, height, i / posed_width,
                                                i % posed_width);
            return occupied(row, col);
        });
        best = min(best, hash);
    }
    return best;
}

// WHO A PIECE TAKES ITS COLOR FROM: another piece of the same grid, found by how
// the two sit -- the nearest one, one sharing its rows or its columns, the one
// whose box holds it or that its box holds, one it touches, or one of its very
// shape, as it lies or turned any of the eight ways round. The rule that reads
// it off the pairs and the op that does it both ask here, so the two cannot
// disagree about which piece is the partner. Numbered in the order they are
// asked.
enum class partner_by : int8_t { nearest, same_row, same_column, around_it, inside_it, touching,
                                 same_shape, same_shape_turned, count };

// HOW A COPY MIRRORS ITS PIECE around a middle: left to right, top to bottom,
// or all four ways at once. The middle is a partner's, one of the partner_by
// ways, or past those the grid's own -- so MIRROR_PARTNERS counts one more.
enum class mirror_by : int8_t { left_right, top_bottom, four_ways, count };
constexpr int MIRROR_PARTNERS = (int)partner_by::count + 1;

// The gap between two boxes, the empty rows plus the empty columns between
// them; 0 when they touch or overlap.
ARC_HOST_DEVICE inline int box_gap(const cell_box& one, const cell_box& other) {
    const int rows = one.row0 > other.row1 ? one.row0 - other.row1 - 1
                   : other.row0 > one.row1 ? other.row0 - one.row1 - 1 : 0;
    const int cols = one.col0 > other.col1 ? one.col0 - other.col1 - 1
                   : other.col0 > one.col1 ? other.col0 - one.col1 - 1 : 0;
    return rows + cols;
}

// Every partner the piece `mine` has by `how` among the pieces not `picked`,
// handed to `visit(other)` -- for the nearest, only the ones no other is nearer
// than. The pieces are read through two questions, so one walk serves a grid's
// pieces and the op's alike: `box_of(i)`, the box its paint fills on the grid,
// and `painted(i, row, col)`, whether it paints that cell of the grid, false
// off it.
template <class BoxOf, class Painted, class Picked, class Visit>
ARC_HOST_DEVICE inline void each_partner(partner_by how, int mine, int count, BoxOf box_of,
                                         Painted painted, Picked picked, Visit visit) {
    const cell_box own = box_of(mine);
    if (own.empty()) return;
    // Its paint read against another's turned by `pose`: the same cells painted.
    auto same_paint = [&](int other, const cell_box& box, int pose) {
        const int height = (pose & 1) ? box.width() : box.height();
        const int width = (pose & 1) ? box.height() : box.width();
        if (height != own.height() || width != own.width()) return false;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) {
                const auto [from_row, from_col] = pose_source(pose, box.width(), box.height(), row, col);
                if (painted(mine, own.row0 + row, own.col0 + col) !=
                    painted(other, box.row0 + from_row, box.col0 + from_col))
                    return false;
            }
        return true;
    };
    int nearest = 1 << 30;
    if (how == partner_by::nearest)
        for (int other = 0; other < count; ++other) {
            if (other == mine || picked(other)) continue;
            const cell_box box = box_of(other);
            if (box.empty()) continue;
            const int gap = box_gap(own, box);
            if (gap < nearest) nearest = gap;
        }
    for (int other = 0; other < count; ++other) {
        if (other == mine || picked(other)) continue;
        const cell_box box = box_of(other);
        if (box.empty()) continue;
        bool partner = false;
        switch (how) {
            case partner_by::nearest:     partner = box_gap(own, box) == nearest; break;
            case partner_by::same_row:    partner = own.row0 <= box.row1 && box.row0 <= own.row1; break;
            case partner_by::same_column: partner = own.col0 <= box.col1 && box.col0 <= own.col1; break;
            case partner_by::around_it:
                partner = box.row0 <= own.row0 && box.col0 <= own.col0 && own.row1 <= box.row1 && own.col1 <= box.col1;
                break;
            case partner_by::inside_it:
                partner = own.row0 <= box.row0 && own.col0 <= box.col0 && box.row1 <= own.row1 && box.col1 <= own.col1;
                break;
            case partner_by::touching:
                for (int row = own.row0; row <= own.row1 && !partner; ++row)
                    for (int col = own.col0; col <= own.col1 && !partner; ++col) {
                        if (!painted(mine, row, col)) continue;
                        for (int down = -1; down <= 1 && !partner; ++down)
                            for (int across = -1; across <= 1 && !partner; ++across)
                                partner = painted(other, row + down, col + across);
                    }
                break;
            case partner_by::same_shape: partner = same_paint(other, box, 0); break;
            case partner_by::same_shape_turned:
                for (int pose = 0; pose < POSE_COUNT && !partner; ++pose) partner = same_paint(other, box, pose);
                break;
            default: break;
        }
        if (partner) visit(other);
    }
}

// The color the piece `mine` takes from its partners: the one color every
// partner wears, `color_of(i)` telling each one's, or -1 when it has none or
// they disagree.
template <class BoxOf, class Painted, class ColorOf, class Picked>
ARC_HOST_DEVICE inline int partner_color(partner_by how, int mine, int count, BoxOf box_of,
                                         Painted painted, ColorOf color_of, Picked picked) {
    int color = -1;
    bool clash = false;
    each_partner(how, mine, count, box_of, painted, picked, [&](int other) {
        const int worn = color_of(other);
        if (color < 0) color = worn;
        else if (worn != color) clash = true;
    });
    return clash ? -1 : color;
}

// The piece `mine`'s one partner, or -1 when it has none or several: a place
// is read off one piece, never agreed among many.
template <class BoxOf, class Painted, class Picked>
ARC_HOST_DEVICE inline int partner_of(partner_by how, int mine, int count, BoxOf box_of,
                                      Painted painted, Picked picked) {
    int found = -1, partners = 0;
    each_partner(how, mine, count, box_of, painted, picked, [&](int other) {
        found = other;
        ++partners;
    });
    return partners == 1 ? found : -1;
}

// WHETHER THE PIECE `mine` WOULD FIT INTO `partner`: turned or mirrored any of
// the eight ways and set down anywhere, it fills one of the partner's gaps
// whole, paints no cell the partner paints, and whatever it has over hangs out
// past the partner's box on the side the gap opens to. A GAP, for now: a run of
// the cells in the partner's box the partner leaves empty, joined side to side,
// that meets the edge of that box on one side alone -- a side the piece lies
// past. Nothing moves: this is the question WouldFit asks, and moving a piece
// into its gap would be the same question answered with the place. The pieces
// are read through the two questions each_partner asks.
template <class BoxOf, class Painted>
ARC_HOST_DEVICE inline bool would_fit(int mine, int partner, BoxOf box_of, Painted painted) {
    const cell_box own = box_of(mine), box = box_of(partner);
    if (own.empty() || box.empty()) return false;
    constexpr uint8_t top = 1, bottom = 2, left = 4, right = 8;
    const uint8_t past = (uint8_t)((own.row1 < box.row0 ? top : 0) | (own.row0 > box.row1 ? bottom : 0) |
                                   (own.col1 < box.col0 ? left : 0) | (own.col0 > box.col1 ? right : 0));
    if (!past) return false;
    int cells = 0;
    for (int row = own.row0; row <= own.row1; ++row)
        for (int col = own.col0; col <= own.col1; ++col) cells += painted(mine, row, col) ? 1 : 0;
    const int tall = box.height(), wide = box.width();
    // Each empty cell of the box: 0 not walked yet, 1 walked, 2 in the run walked now.
    uint8_t seen[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    int16_t run[MAX_GRID_DIM * MAX_GRID_DIM];
    // Whether the piece, some way round, fills whole the run of `count` cells
    // starting at `first`, hanging out past `side` with whatever it has over.
    auto fills = [&](int first, int count, uint8_t side) {
        const int anchor_row = box.row0 + first / MAX_GRID_DIM, anchor_col = box.col0 + first % MAX_GRID_DIM;
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            const int high = (pose & 1) ? own.width() : own.height(), broad = (pose & 1) ? own.height() : own.width();
            auto at = [&](int row, int col) {
                const auto [from_row, from_col] = pose_source(pose, own.width(), own.height(), row, col);
                return (bool)painted(mine, own.row0 + from_row, own.col0 + from_col);
            };
            // The run's first cell is filled by one of the piece's cells: each in turn.
            for (int lands_row = 0; lands_row < high; ++lands_row)
                for (int lands_col = 0; lands_col < broad; ++lands_col) {
                    if (!at(lands_row, lands_col)) continue;
                    const int down = anchor_row - lands_row, across = anchor_col - lands_col;
                    int covered = 0;
                    bool sits = true;
                    for (int row = 0; row < high && sits; ++row)
                        for (int col = 0; col < broad && sits; ++col) {
                            if (!at(row, col)) continue;
                            const int on_row = row + down, on_col = col + across;
                            if (painted(partner, on_row, on_col)) { sits = false; break; }
                            if (on_row >= box.row0 && on_row <= box.row1 && on_col >= box.col0 && on_col <= box.col1) {
                                if (seen[on_row - box.row0][on_col - box.col0] == 2) ++covered;
                                else sits = false;
                            } else {
                                sits = side == top ? on_row < box.row0 : side == bottom ? on_row > box.row1
                                     : side == left ? on_col < box.col0 : on_col > box.col1;
                            }
                        }
                    if (sits && covered == count) return true;
                }
        }
        return false;
    };
    for (int start_row = 0; start_row < tall; ++start_row)
        for (int start_col = 0; start_col < wide; ++start_col) {
            if (seen[start_row][start_col] || painted(partner, box.row0 + start_row, box.col0 + start_col)) continue;
            // Walk the run from its first cell in reading order, noting the box's sides it meets.
            int count = 0;
            uint8_t meets = 0;
            seen[start_row][start_col] = 2;
            run[count++] = (int16_t)(start_row * MAX_GRID_DIM + start_col);
            for (int next = 0; next < count; ++next) {
                const int row = run[next] / MAX_GRID_DIM, col = run[next] % MAX_GRID_DIM;
                if (row == 0) meets |= top;
                if (row == tall - 1) meets |= bottom;
                if (col == 0) meets |= left;
                if (col == wide - 1) meets |= right;
                const int steps[4][2] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
                for (const auto& step : steps) {
                    const int to_row = row + step[0], to_col = col + step[1];
                    if (to_row < 0 || to_col < 0 || to_row >= tall || to_col >= wide || seen[to_row][to_col] ||
                        painted(partner, box.row0 + to_row, box.col0 + to_col))
                        continue;
                    seen[to_row][to_col] = 2;
                    run[count++] = (int16_t)(to_row * MAX_GRID_DIM + to_col);
                }
            }
            const bool one_side = meets == top || meets == bottom || meets == left || meets == right;
            if (one_side && (meets & past) && count <= cells && fills(run[0], count, meets)) return true;
            for (int i = 0; i < count; ++i) seen[run[i] / MAX_GRID_DIM][run[i] % MAX_GRID_DIM] = 1;
        }
    return false;
}

}  // namespace arc
