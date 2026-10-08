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
    // Grown to hold another box as well; an empty one adds nothing.
    ARC_HOST_DEVICE void include(const cell_box& other) {
        if (other.empty()) return;
        include_cell(other.row0, other.col0);
        include_cell(other.row1, other.col1);
    }
};

// An object's bounding box and cell count, usable without storing its cells.
// Packed small for the device; box() reads it as a cell_box.
struct object_summary {
    int8_t  x0 = 0, y0 = 0;  // top-left in grid coordinates
    int8_t  w = 0, h = 0;    // box width and height
    int16_t count = 0;       // number of cells in the object
    ARC_HOST_DEVICE cell_box box() const { return { y0, x0, y0 + h - 1, x0 + w - 1 }; }
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

// HOW MANY CELLS WEAR EACH COLOR, counted a cell at a time -- what is no color
// is not counted -- and the color the most of them wear, or the fewest, the
// lowest on a tie: TRANSPARENT where nothing was counted. The one tie rule for
// every op and every reading that asks which color leads.
struct color_tally {
    int counts[COLOR_COUNT] = {};
    ARC_HOST_DEVICE void add(int color) { if (is_color(color)) ++counts[color]; }
    ARC_HOST_DEVICE int8_t most() const { return leading(true); }
    ARC_HOST_DEVICE int8_t least() const { return leading(false); }

private:
    ARC_HOST_DEVICE int8_t leading(bool most) const {
        int8_t best = TRANSPARENT;
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (counts[color] && (best < 0 || (most ? counts[color] > counts[best] : counts[color] < counts[best])))
                best = (int8_t)color;
        return best;
    }
};

// The color most of a `factor` by `factor` block wears, as ScaleDown votes.
template <class Mask>
ARC_HOST_DEVICE inline int8_t majority_vote(const Mask& mask, int row, int col,
                                             int factor, int height, int width) {
    color_tally tally;
    for (int dr = 0; dr < factor; ++dr)
        for (int dc = 0; dc < factor; ++dc) {
            const int source_row = row * factor + dr;
            const int source_col = col * factor + dc;
            if (source_row < height && source_col < width) tally.add((int8_t)mask[source_row][source_col]);
        }
    return tally.most();
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
ARC_HOST_DEVICE inline std::pair<int, int> posed_size(int pose, int width, int height) {
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

    // The box its cells fill -- which the summary's box need not be: a block
    // keeps its whole box, empty cells and all.
    cell_box painted_box() const {
        cell_box box;
        each_cell([&](int row, int col, int8_t) { box.include_cell(row, col); });
        return box;
    }

    // Every cell it fills repainted in the color `color_of(was)` names.
    template <class ColorOf>
    void repaint(ColorOf color_of) {
        for (int row = 0; row < summary.h; ++row)
            for (int col = 0; col < summary.w; ++col)
                if (cells[row][col] != (int8_t)TRANSPARENT) cells[row][col] = (int8_t)color_of((int8_t)cells[row][col]);
    }

    // The object turned or flipped to `pose` in its own box, its top-left
    // corner where it was: each cell read through pose_source, as the ops read it.
    object posed(int pose) const {
        object turned = *this;
        const auto [width, height] = posed_size(pose, summary.w, summary.h);
        turned.summary.w = (int8_t)width;
        turned.summary.h = (int8_t)height;
        turned.cells.clear();
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) {
                const auto [from_row, from_col] = pose_source(pose, summary.w, summary.h, row, col);
                turned.cells[row][col] = cells[from_row][from_col];
            }
        return turned;
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

// What counts as one object. The walk answers all four the same way, so no
// two readings can disagree about which cells belong together.
enum class grouping : int8_t {
    // Cells that touch, all of them the same color. The usual object.
    touching_same_color,
    // Cells that touch, whatever their colors. One object may be many colors.
    touching_any_color,
    // Every cell of a color at once, touching or not. One object per color.
    all_of_one_color,
    // Cells that share a side, all of them the same color: the usual object cut
    // finer, a corner joining nothing (aedd82e4's lone cells).
    touching_sides_same_color,
};

// Caller-owned so host and device can lend memory they already hold.
struct object_walk_scratch {
    bool taken[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_CELLS];
    int16_t queue_col[MAX_GRID_CELLS];
    int16_t cells[MAX_GRID_CELLS];
};

// THE EIGHT WAYS, the one numbering of directions in the project: the cells
// around a cell, and the ways a piece can go, numbered as a compass is,
// clockwise from up, as MoveToEdge, ExtendTo and Shift take them. The way back
// is four on, a quarter turn two on, and the sides are the even ways.
struct heading {
    enum : int8_t { up, up_right, right, down_right, down, down_left, left, up_left, count };
};
inline constexpr const char* const heading_names[heading::count] = {
    "up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left" };

// THE ORDER THE WAYS ARE ASKED IN, so the order a tie between two that say as
// much is broken in: the four sides before the diagonals, the plainer move
// first, each clockwise from up.
constexpr int heading_asked(int at) { return at < 4 ? 2 * at : 2 * (at - 4) + 1; }

// One step that way. A switch and not a table, so the kernel reads the same
// steps the host does; a heading out of range goes up.
struct heading_step { int down, across; };
ARC_HOST_DEVICE inline heading_step step_of(int way) {
    switch (way) {
    case heading::up_right:   return { -1,  1 };
    case heading::right:      return {  0,  1 };
    case heading::down_right: return {  1,  1 };
    case heading::down:       return {  1,  0 };
    case heading::down_left:  return {  1, -1 };
    case heading::left:       return {  0, -1 };
    case heading::up_left:    return { -1, -1 };
    default:                  return { -1,  0 };
    }
}

// THE CELLS THAT JOIN A CELL: all eight ways round, or only the four sides,
// the even ways -- the step between the ways asked, for 8 or for 4.
//
// Grouping cells into an object is eight ways, "is this one object?" -- save
// the one finer cut that asks it of sides alone (touching_sides_same_color).
// Adjacency is four ways, "what shares an edge with this cell?": a diagonal
// cell touches only at a corner.
//
// Why four there and eight for grouping. Picture a ring of blocks meeting only
// at their corners. Ask "is this one ring" and the answer is yes, corners
// count, which is the eight. Ask "can the water inside get out" and corners
// must not count, or the water escapes through the very tip just called a
// wall and the ring has no inside at all. So anything measuring a border --
// what frays against what, what surrounds what, whether background leaks to
// the grid edge -- takes the four. Whatever the shapes use, the space around
// them uses the other one. Let both use eight and no shape has an inside any
// more, so holes stop existing.
template <int Ways>
constexpr int every_nth_way = Ways == 4 ? 2 : 1;

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
        for (int way = 0; way < heading::count; way += every_nth_way<Directions>) {
            const int next_row = row + step_of(way).down, next_col = col + step_of(way).across;
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
    for (int way = 0; way < heading::count; ++way)
        if (!occupied(row + step_of(way).down, col + step_of(way).across)) return true;
    return false;
}

// Recolor and AddColor's relative color sources, shared by reading and execution.
// Ties choose the lowest color, and an empty object stays transparent.
template <class ColorAt>
ARC_HOST_DEVICE inline int8_t object_paint_color(int height, int width, int8_t value, ColorAt color_at) {
    if (value < COLOR_COUNT) return value;
    color_tally tally;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) tally.add(color_at(row, col));
    // COLOR_COUNT names the color the fewest cells wear, past it the most.
    return value == COLOR_COUNT ? tally.least() : tally.most();
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

// WHETHER SelectAtLeastCells[at_least] PICKS A PIECE: it paints at least that
// many cells of its `height` by `width` box. Counted here for the op and for
// the reading that replays it, so the two cannot pick different pieces.
template <class Occupied>
ARC_HOST_DEVICE inline bool paints_at_least(int height, int width, Occupied occupied, int at_least) {
    int cells = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) cells += occupied(row, col) ? 1 : 0;
    return cells >= at_least;
}

// Count four-connected enclosed spaces in an occupancy mask: the runs of empty
// cells no four-way walk in from the edge of the box reaches (mark_outside).
// Shared by inference and SelectObjectsWithHoles so learned keys mean the same
// at runtime.
template <class Occupied>
ARC_HOST_DEVICE inline int object_hole_count(int height, int width, Occupied occupied) {
    if (width <= 0 || height <= 0) return 0;
    // Walked: outside, or in a hole already counted.
    bool walked[MAX_GRID_DIM][MAX_GRID_DIM];
    mark_outside(height, width, occupied, walked);
    auto claim = [&](int row, int col) {
        if (walked[row][col] || occupied(row, col)) return false;
        walked[row][col] = true;
        return true;
    };
    int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
    int holes = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            if (!claim(row, col)) continue;
            ++holes;
            queue_row[0] = (int16_t)row;
            queue_col[0] = (int16_t)col;
            walk_out<4>(height, width, queue_row, queue_col, 1, claim, [](int, int) {});
        }
    return holes;
}

// WHAT Fit STRETCHES TO, its value: the input's frame, the canvas the output is
// drawn on, or the box of what is painted on it -- none where nothing is.
// Reading and execution ask the same helper.
enum class fit_to : int8_t { input, canvas, painted, count };
template <class Canvas>
ARC_HOST_DEVICE inline cell_box fit_frame(int to, int input_width, int input_height,
                                         const Canvas& canvas) {
    if (to == (int)fit_to::input) return {0, 0, input_height - 1, input_width - 1};
    if (to == (int)fit_to::canvas) return {0, 0, canvas.height - 1, canvas.width - 1};
    cell_box box;
    if (to == (int)fit_to::painted)
        for (int r = 0; r < canvas.height; ++r)
            for (int c = 0; c < canvas.width; ++c)
                if (is_color(canvas.data[r][c])) box.include_cell(r, c);
    return box;
}
ARC_HOST_DEVICE inline int fit_source_cell(int target_cell, int source_size, int target_size) {
    return target_cell * source_size / target_size;
}

// A box as an object's summary has it, with its cell count (object_summary::box
// reads it back).
ARC_HOST_DEVICE inline object_summary summary_of(const cell_box& box, int count) {
    return { (int8_t)box.col0, (int8_t)box.row0, (int8_t)box.width(), (int8_t)box.height(),
             (int16_t)count };
}

// THE LINE `from` STEPS PAST THE MIDDLE OF A SPAN of lines lo..hi, the middle
// being the one line as many lines from either end: -1 where the span has two
// middle lines, or the line falls outside it. SelectRowFromMiddle and
// SelectColumnFromMiddle read the span every piece's paint covers, and the
// reading replays them here (d23f8c26 keeps the middle column).
ARC_HOST_DEVICE inline int line_from_middle(int lo, int hi, int from) {
    if (lo > hi || (hi - lo) % 2 != 0) return -1;
    const int line = (lo + hi) / 2 + from;
    return line >= lo && line <= hi ? line : -1;
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

// WHERE SHIFT MOVES A BOX, its own size away: one step that way (step_of), its
// height a step down or up and its width a step across. The kernel and the
// reading that replays it take the move from here; a way out of range stays.
ARC_HOST_DEVICE inline heading_step shift_by(int way, int width, int height) {
    if (way < 0 || way >= heading::count) return { 0, 0 };
    const heading_step step = step_of(way);
    return { step.down * height, step.across * width };
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
    // A grid laid out of copies (tiles_of): a group never leaves its seed's copy.
    const tile_size tiles = tiles_of(one_grid);
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
            auto in_this_copy = [&](int other_row, int other_col) {
                return (tiles.height <= 0 || other_row / tiles.height == row / tiles.height) &&
                       (tiles.width <= 0 || other_col / tiles.width == col / tiles.width);
            };
            // Every other cell of this color joins before the walk starts, and
            // the walk below then claims nothing, so the group is the color.
            if (whole_color)
                for (int other_row = 0; other_row < one_grid.height; ++other_row)
                    for (int other_col = 0; other_col < one_grid.width; ++other_col) {
                        if (taken[other_row][other_col] || !in_this_copy(other_row, other_col) ||
                            one_grid.data[other_row][other_col] != color) continue;
                        taken[other_row][other_col] = true;
                        queue_row[seeded] = (int16_t)other_row;
                        queue_col[seeded] = (int16_t)other_col;
                        ++seeded;
                    }
            // One flood fill for the whole project, so no two readings can
            // disagree about which cells belong to one group: eight ways round,
            // or four where only a side joins.
            auto claim = [&](int next_row, int next_col) {
                if (whole_color || taken[next_row][next_col] || !in_this_copy(next_row, next_col)) return false;
                const int there = one_grid.data[next_row][next_col];
                if (any_color ? (there < 0 || there == background)
                              : (there != color)) return false;
                taken[next_row][next_col] = true;
                return true;
            };
            auto see = [&](int seen_row, int seen_col) {
                found_cells[seen++] = (int16_t)cell_index(seen_row, seen_col);
                box.include_cell(seen_row, seen_col);
            };
            const int count = how == grouping::touching_sides_same_color
                ? walk_out<4>(one_grid.height, one_grid.width, queue_row, queue_col, seeded, claim, see)
                : walk_out<8>(one_grid.height, one_grid.width, queue_row, queue_col, seeded, claim, see);
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
// shape_occupancy_hash over the eight ways round -- turned either way, mirrored
// either way -- done by index arithmetic, no pose grids built. Equal shapes have
// equal hashes; collisions are possible. occupied(row, col) reads the original
// shape without copying its storage. THE KEY OF THE LEARNED SHAPE TABLE: the
// reading learns it and the ops that read the table look it up, both here, so
// a shape the pairs showed one way round is the same shape the test shows
// another (e8593010's L, turned).
template <typename OccupancyAt>
ARC_HOST_DEVICE inline uint32_t canonical_shape_hash(int width, int height, OccupancyAt occupied) {
    uint32_t best = 0xFFFFFFFFu;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, width, height);
        const uint32_t hash = arc::shape_occupancy_hash(posed_width, posed_height, [&](int i) {
            const auto [row, col] = pose_source(pose, width, height, i / posed_width,
                                                i % posed_width);
            return occupied(row, col);
        });
        if (hash < best) best = hash;
    }
    return best;
}

// WHICH OF THE EIGHT WAYS ROUND LEAVE A DRAWING AS IT WAS, colors and empty
// cells and all, a bit each (pose_source, the turns' and flips' own reading):
// 1 for a drawing only the same as it lies, 255 for one the same every way.
template <class ColorAt>
ARC_HOST_DEVICE inline uint32_t symmetry_of(int width, int height, ColorAt color_at) {
    uint32_t poses = 0;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, width, height);
        bool same = posed_width == width && posed_height == height;
        for (int row = 0; row < height && same; ++row)
            for (int col = 0; col < width && same; ++col) {
                const auto [from_row, from_col] = pose_source(pose, width, height, row, col);
                same = color_at(from_row, from_col) == color_at(row, col);
            }
        if (same) poses |= 1u << pose;
    }
    return poses;
}

// WHAT A LEARNED TABLE KNOWS A PIECE BY, ShapeColorMap's value: its shape,
// whichever way round it lies (canonical_shape_hash), or its symmetry, which
// ways round leave its drawing as it was (symmetry_of: 44f52bb0's picture, 7
// where no turn or flip leaves it as it was, else 1). color_at(row, col) reads
// the piece's drawing in its own box. The reading keys its table here and the
// op looks it up here, so the two cannot know a piece by different keys.
enum class shape_key : int8_t { shape, symmetry, count };
template <class ColorAt>
ARC_HOST_DEVICE inline uint32_t shape_key_of(int kind, int width, int height, ColorAt color_at) {
    if (kind == (int)shape_key::symmetry) return symmetry_of(width, height, color_at);
    return canonical_shape_hash(width, height, [&](int row, int col) { return color_at(row, col) != TRANSPARENT; });
}

// A LEARNED TABLE OF KEYS (shape_key_of): the color each key it knows names,
// and the color every other key takes -- -1 where a piece it does not know
// stays as it was (25d8a9c8: the full row 5, any other shape 0). The reading
// learns it and the ops look it up, both through color_of, so the two cannot
// name a key's color differently.
struct shape_table {
    static constexpr int MAX_KEYS = 16;
    uint32_t keys[MAX_KEYS] = {};
    int8_t colors[MAX_KEYS] = {};
    int count = 0;
    int8_t otherwise = -1;
    ARC_HOST_DEVICE bool holds() const { return count > 0; }
    ARC_HOST_DEVICE int color_of(uint32_t key) const {
        for (int i = 0; i < count && i < MAX_KEYS; ++i)
            if (keys[i] == key) return colors[i];
        return otherwise;
    }
    // Remembers which color goes with this key. Returns false if the key gets
    // conflicting colors.
    bool learn(uint32_t key, int8_t color) {
        for (int i = 0; i < count; ++i)
            if (keys[i] == key) return colors[i] == color;
        if (count >= MAX_KEYS) return false;
        keys[count] = key;
        colors[count] = color;
        count++;
        return true;
    }
};

// WHO A PIECE TAKES ITS COLOR FROM: another piece of the same grid, found by how
// the two sit. The rule that reads it off the pairs and the op that does it
// both ask here, so the two cannot disagree about which piece is the partner.
// NUMBERED IN THE ORDER THE WAYS ARE ASKED IN, so the order a tie between two
// that say as much is broken in: the surer a way names one partner, the
// sooner. The piece around it or inside it first -- one box holds another --
// then the piece lined up with it, in its rows or its columns, which a second
// piece in the same band leaves without one (18419cfa's test shape, under two
// frames), then the nearest, which ties as soon as two stand as far (ddf7fa4f's
// test block, between two dots), then the piece it touches at a side, the
// four-connected touch, before the one it touches at a side or a corner,
// four-connected before eight until eight is needed (b7256dcd's pair meets the
// 3 at a side and the 4 only at a corner; 4938f0c2's shape meets its block only
// at a corner), then one of its very shape, as it lies or turned any of the
// eight ways round.
enum class partner_by : int8_t { around_it, inside_it, same_row, same_column, nearest, touching_sides,
                                 touching, same_shape, same_shape_turned, count };

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

// WHETHER THE PIECE `mine`, MOVED `by`, TOUCHES THE PIECE `other`: a cell of it
// on a cell of the other, or beside one at a side -- or, unless `sides_only`,
// at a corner. The pieces are asked as each_partner asks them, below.
template <class Pieces>
ARC_HOST_DEVICE inline bool touches(const Pieces& pieces, int mine, int other, heading_step by, bool sides_only) {
    const cell_box own = pieces.box(mine);
    for (int row = own.row0; row <= own.row1; ++row)
        for (int col = own.col0; col <= own.col1; ++col) {
            if (!pieces.painted(mine, row, col)) continue;
            const int at_row = row + by.down, at_col = col + by.across;
            if (pieces.painted(other, at_row, at_col)) return true;
            for (int way = 0; way < heading::count; way += sides_only ? every_nth_way<4> : every_nth_way<8>)
                if (pieces.painted(other, at_row + step_of(way).down, at_col + step_of(way).across)) return true;
        }
    return false;
}

// Every partner the piece `mine` has by `how` among the pieces not picked,
// handed to `visit(other)` -- for the nearest, only the ones no other is nearer
// than. THE PIECES ARE ASKED, so one walk serves a grid's pieces and the op's
// alike: `count()`; `box(i)`, the box piece i's paint fills on the grid;
// `painted(i, row, col)`, whether it paints that cell of the grid, false off
// it; `is_picked(i)`; and, for the color questions below, `color(i)`, the color
// an op reads off its first cell, and `holds(i, color)`, whether it holds a
// cell of that color.
template <class Pieces, class Visit>
ARC_HOST_DEVICE inline void each_partner(partner_by how, int mine, const Pieces& pieces, Visit visit) {
    const cell_box own = pieces.box(mine);
    if (own.empty()) return;
    auto painted = [&](int i, int row, int col) { return pieces.painted(i, row, col); };
    // Its paint read against another's turned by `pose`: the same cells painted.
    auto same_paint = [&](int other, const cell_box& box, int pose) {
        const auto [width, height] = posed_size(pose, box.width(), box.height());
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
        for (int other = 0; other < pieces.count(); ++other) {
            if (other == mine || pieces.is_picked(other)) continue;
            const cell_box box = pieces.box(other);
            if (box.empty()) continue;
            const int gap = box_gap(own, box);
            if (gap < nearest) nearest = gap;
        }
    for (int other = 0; other < pieces.count(); ++other) {
        if (other == mine || pieces.is_picked(other)) continue;
        const cell_box box = pieces.box(other);
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
            case partner_by::touching_sides:
                partner = touches(pieces, mine, other, { 0, 0 }, how == partner_by::touching_sides);
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
// partner wears, or -1 when it has none or they disagree.
template <class Pieces>
ARC_HOST_DEVICE inline int partner_color(partner_by how, int mine, const Pieces& pieces) {
    int color = -1;
    bool clash = false;
    each_partner(how, mine, pieces, [&](int other) {
        const int worn = pieces.color(other);
        if (color < 0) color = worn;
        else if (worn != color) clash = true;
    });
    return clash ? -1 : color;
}

// The piece `mine`'s one partner, or -1 when it has none or several: a place
// is read off one piece, never agreed among many.
template <class Pieces>
ARC_HOST_DEVICE inline int partner_of(partner_by how, int mine, const Pieces& pieces) {
    int found = -1, partners = 0;
    each_partner(how, mine, pieces, [&](int other) {
        found = other;
        ++partners;
    });
    return partners == 1 ? found : -1;
}

// A PARTNER OF ANY KIND: the ways partner_by finds one by how two pieces sit,
// then, past those, by color -- the one piece holding a cell of color `kind -
// partner_by::count` (18286ef8's 9 steps toward the 6, wherever the 6 is).
// Asked among the pieces not picked, as partner_of asks. -1 when there is
// none, or several.
constexpr int PARTNER_KINDS = (int)partner_by::count + COLOR_COUNT;
template <class Pieces>
ARC_HOST_DEVICE inline int partner_of_kind(int kind, int mine, const Pieces& pieces) {
    if (kind < (int)partner_by::count) return partner_of((partner_by)kind, mine, pieces);
    int found = -1;
    for (int other = 0; other < pieces.count(); ++other) {
        if (other == mine || pieces.is_picked(other) || !pieces.holds(other, kind - (int)partner_by::count)) continue;
        if (found >= 0) return -1;
        found = other;
    }
    return found;
}

// THE WAY FROM ONE BOX TOWARD ANOTHER: the one of the eight headings whose
// compass sector the line between their middles points into, each sector 45
// degrees wide, its edges 22.5 degrees off the axes. A line steeper than that
// off the rows goes up or down, and one flatter than that off the columns goes
// across; tan 22.5 is sqrt 2 - 1, irrational, so with whole cells no line lies
// on an edge, and the test needs no fractions. -1 when the middles coincide.
ARC_HOST_DEVICE inline int heading_toward(const cell_box& from, const cell_box& to) {
    // Twice the middles' distance, so a middle between two cells is whole.
    const int down = (to.row0 + to.row1) - (from.row0 + from.row1);
    const int across = (to.col0 + to.col1) - (from.col0 + from.col1);
    const int tall = down < 0 ? -down : down, wide = across < 0 ? -across : across;
    const int reach = (tall + wide) * (tall + wide);
    const int vertical = reach > 2 * wide * wide ? (down > 0 ? 1 : -1) : 0;
    const int horizontal = reach > 2 * tall * tall ? (across > 0 ? 1 : -1) : 0;
    if (vertical == 0 && horizontal == 0) return -1;
    for (int way = 0; way < heading::count; ++way)
        if (step_of(way).down == vertical && step_of(way).across == horizontal) return way;
    return -1;
}

// HOW FAR MoveToward GOES toward its partner: one step, its own size, or until
// it touches the partner.
enum class toward_by : int8_t { a_step, its_size, touching, count };

// WHERE MoveToward MOVES A PIECE: that way toward its partner (heading_toward,
// between their boxes), by `far` -- one step; its own height or width each way
// it goes; or step by step until a cell of it touches a cell of the partner, at
// a side or a corner, as partner_by::touching asks, on a canvas `width` by
// `height`. A piece that touches already, or that would leave the canvas first,
// stays. Hands back how far down and across; the kernel and the reading's replay
// both take the move from here. The pieces are asked as each_partner asks them.
template <class Pieces>
ARC_HOST_DEVICE inline heading_step move_toward(toward_by far, int mine, int partner, const Pieces& pieces,
                                                int width, int height) {
    const cell_box own = pieces.box(mine);
    const int way = heading_toward(own, pieces.box(partner));
    if (way < 0 || own.empty()) return { 0, 0 };
    const heading_step step = step_of(way);
    if (far == toward_by::a_step) return step;
    if (far == toward_by::its_size) return { step.down * own.height(), step.across * own.width() };
    for (int steps = 0;; ++steps) {
        const int down = steps * step.down, across = steps * step.across;
        if (own.row0 + down < 0 || own.row1 + down >= height || own.col0 + across < 0 || own.col1 + across >= width)
            return { 0, 0 };
        if (touches(pieces, mine, partner, { down, across }, false)) return { down, across };
    }
}

// WHETHER THE PIECE `mine` WOULD FIT INTO `partner`: turned or mirrored any of
// the eight ways and set down anywhere, it fills one of the partner's gaps
// whole, paints no cell the partner paints, and whatever it has over hangs out
// past the partner's box on the side the gap opens to. A GAP, for now: a run of
// the cells in the partner's box the partner leaves empty, joined side to side,
// that meets the edge of that box on one side alone -- a side the piece lies
// past. Nothing moves: this is the question WouldFit asks, and moving a piece
// into its gap would be the same question answered with the place. The pieces
// are asked as each_partner asks them.
template <class Pieces>
ARC_HOST_DEVICE inline bool would_fit(int mine, int partner, const Pieces& pieces) {
    const cell_box own = pieces.box(mine), box = pieces.box(partner);
    if (own.empty() || box.empty()) return false;
    auto painted = [&](int i, int row, int col) { return pieces.painted(i, row, col); };
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
    int16_t run_row[MAX_GRID_CELLS], run_col[MAX_GRID_CELLS];
    // Whether the piece, some way round, fills whole the run of `count` cells
    // starting at (first_row, first_col), hanging out past `side` with whatever
    // it has over.
    auto fills = [&](int first_row, int first_col, int count, uint8_t side) {
        const int anchor_row = box.row0 + first_row, anchor_col = box.col0 + first_col;
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            const auto [broad, high] = posed_size(pose, own.width(), own.height());
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
            uint8_t meets = 0;
            seen[start_row][start_col] = 2;
            run_row[0] = (int16_t)start_row;
            run_col[0] = (int16_t)start_col;
            const int count = walk_out<4>(tall, wide, run_row, run_col, 1,
                [&](int row, int col) {
                    if (seen[row][col] || painted(partner, box.row0 + row, box.col0 + col)) return false;
                    seen[row][col] = 2;
                    return true;
                },
                [&](int row, int col) {
                    if (row == 0) meets |= top;
                    if (row == tall - 1) meets |= bottom;
                    if (col == 0) meets |= left;
                    if (col == wide - 1) meets |= right;
                });
            const bool one_side = meets == top || meets == bottom || meets == left || meets == right;
            if (one_side && (meets & past) && count <= cells && fills(start_row, start_col, count, meets)) return true;
            for (int i = 0; i < count; ++i) seen[run_row[i]][run_col[i]] = 1;
        }
    return false;
}

}  // namespace arc
