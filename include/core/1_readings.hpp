// This file describes how to read a grid: its cells, colors, objects, poses,
// walks and shapes, each built on the ones before. What an op does to a grid
// is 8b_op_math.hpp's.
#pragma once
#include "0_core.hpp"
#include <algorithm>
#include <cstdint>
#include <utility>

namespace arc {
using namespace std;

// CELLS AND BOXES · where a cell sits, and the box around cells

// A cell's place in the largest grid as one number, counted row by row. Lists
// of cells hold these, so a cell costs two bytes rather than a pair, and the
// packed store below finds its cells the same way.
ARC_HOST_DEVICE constexpr int cell_index(int row, int col) { return row * MAX_GRID_DIM + col; }
ARC_HOST_DEVICE constexpr int cell_row(int index) { return index / MAX_GRID_DIM; }
ARC_HOST_DEVICE constexpr int cell_col(int index) { return index % MAX_GRID_DIM; }

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

// COLORS · how many cells wear each color, and where each sits

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

// The box each color's cells fill, with one color left out if asked; a color
// no cell wears keeps an empty box.
template <class Grid>
ARC_HOST_DEVICE inline void color_boxes(const Grid& one_grid, cell_box boxes[COLOR_COUNT],
                        int skip = TRANSPARENT) {
    for (int i = 0; i < COLOR_COUNT; ++i) boxes[i] = {};
    walk_colored_cells(one_grid, skip, [&](int row, int col, int color) {
        boxes[color].include_cell(row, col);
    });
}

// OBJECTS · a group of cells read as one thing

// An object is a related group of cells.
// Humans mostly decompose an ARC grid into objects when reading the grid,
// so the code will match that level of abstraction
// vs carrying each cell as its own entity.

// An object's bounding box and cell count, usable without storing its cells.
// Packed small for the device; box() reads it as a cell_box.
struct object_summary {
    int8_t  x0 = 0, y0 = 0;  // top-left in grid coordinates
    int8_t  w = 0, h = 0;    // box width and height
    int16_t count = 0;       // number of cells in the object
    ARC_HOST_DEVICE cell_box box() const { return { y0, x0, y0 + h - 1, x0 + w - 1 }; }
};

// A box as an object's summary has it, with its cell count (object_summary::box
// reads it back).
ARC_HOST_DEVICE inline object_summary summary_of(const cell_box& box, int count) {
    return { (int8_t)box.col0, (int8_t)box.row0, (int8_t)box.width(), (int8_t)box.height(),
             (int16_t)count };
}

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

// POSES · the eight ways a grid can sit

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

// The object turned or flipped to `pose` in its own box, its top-left
// corner where it was: each cell read through pose_source, as the ops read it.
inline object posed(const object& piece, int pose) {
    object turned = piece;
    const auto [width, height] = posed_size(pose, piece.summary.w, piece.summary.h);
    turned.summary.w = (int8_t)width;
    turned.summary.h = (int8_t)height;
    turned.cells.clear();
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const auto [from_row, from_col] = pose_source(pose, piece.summary.w, piece.summary.h, row, col);
            turned.cells[row][col] = piece.cells[from_row][from_col];
        }
    return turned;
}

// WALKS · the ways round a cell, and the walk that finds objects

// THE EIGHT WAYS, the one numbering of directions in the project: the cells
// around a cell, and the ways a piece can go, numbered as a compass is,
// clockwise from up, as MoveToEdge, ExtendTo and Shift take them. The way back
// is four on, a quarter turn two on, and the sides are the even ways.
struct heading {
    enum : int8_t { up, up_right, right, down_right, down, down_left, left, up_left, count };
};

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

// Boundaries for reading copies separately; zero means one whole picture.
struct tile_size { int width = 0, height = 0; };

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
                                 grouping how, object_walk_scratch& scratch, Take take, tile_size tiles = {}) {
    const bool any_color = (how == grouping::touching_any_color);
    // One object per color takes every cell of the color at once, wherever it
    // is; the touching kinds walk out from the seed cell only.
    const bool whole_color = (how == grouping::all_of_one_color);
    if (one_grid.width <= 0 || one_grid.height <= 0) return;
    // A group never leaves its seed's copy when boundaries are supplied.
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

// SHAPES · a shape as one number, and the ways round that leave it be

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

}  // namespace arc
