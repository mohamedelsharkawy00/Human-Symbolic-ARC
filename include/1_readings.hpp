// This file describes how to read a grid.
#pragma once
#include "0_core.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>

// Allow a function to run on both CPU and GPU when compiled with CUDA.
#ifdef __CUDACC__
#define ARC_HOST_DEVICE __host__ __device__
#else
#define ARC_HOST_DEVICE
#endif

namespace arc {
using namespace std;

// An object is a related group of cells.
// Humans mostly decompose an ARC grid into objects when reading the grid,
// so the code will match that level of abstraction
// vs carrying each cell as its own entity.

// A sentinel for an empty non-colored cell,
// that will be used for masks and marks later.
constexpr int TRANSPARENT = -1;

// Only puzzle colors count; TRANSPARENT is excluded.
ARC_HOST_DEVICE constexpr bool is_color(int value) {
    return value >= 0 && value < COLOR_COUNT;
}

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

// Finds one group of connected cells by walking the given bounds, 8-way.
// `claim` decides membership into that group and must mark what it accepts, or the walk never ends.
template <class Claim, class See>
ARC_HOST_DEVICE inline int walk_out(int rows, int cols, int16_t queue_row[], int16_t queue_col[], int seeded,
                    Claim claim, See see) {
    int head = 0, tail = seeded;
    while (head < tail) {
        const int row = queue_row[head], col = queue_col[head];
        ++head;
        see(row, col);
        for (int dir = 0; dir < 8; ++dir) {
            const int next_row = row + neighbour_row_8(dir);
            const int next_col = col + neighbour_col_8(dir);
            if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) continue;
            if (!claim(next_row, next_col)) continue;
            queue_row[tail] = (int16_t)next_row; queue_col[tail] = (int16_t)next_col;
            ++tail;
        }
    }
    return tail;
}

// An object's bounding box and cell count, usable without storing its cells.
struct object_summary {
    int8_t  x0 = 0, y0 = 0;  // top-left in grid coordinates
    int8_t  w = 0, h = 0;    // box width and height
    int16_t count = 0;       // number of cells in the object
};

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
    static constexpr int CELLS = MAX_GRID_DIM * MAX_GRID_DIM;
    static constexpr int BYTES = ((CELLS + 1) / 2 + 3) / 4 * 4;
    uint8_t bytes[BYTES];

    // A cell occupies half a byte, so reads and writes need a proxy,
    // because C++ can't give a reference to half a byte.
    struct const_cell_ref {
        const uint8_t* byte;
        int shift;

        ARC_HOST_DEVICE operator int8_t() const {
            return (int8_t)((*byte >> shift) & 0xF) - 1;
        }
    };

    struct cell_ref {
        uint8_t* byte;
        int shift;

        ARC_HOST_DEVICE operator int8_t() const {
            return (int8_t)((*byte >> shift) & 0xF) - 1;
        }

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
    struct row_ref {
        uint8_t* base;
        int row_index;

        ARC_HOST_DEVICE cell_ref operator[](int col) const {
            int index = row_index * MAX_GRID_DIM + col;
            if (index < 0 || index >= CELLS) index = 0;
            return { base + (index >> 1), (index & 1) * 4 };
        }
    };

    struct const_row_ref {
        const uint8_t* base;
        int row_index;

        ARC_HOST_DEVICE const_cell_ref operator[](int col) const {
            int index = row_index * MAX_GRID_DIM + col;
            if (index < 0 || index >= CELLS) index = 0;
            return { base + (index >> 1), (index & 1) * 4 };
        }
    };

    ARC_HOST_DEVICE row_ref operator[](int row) { return { bytes, row }; }
    ARC_HOST_DEVICE const_row_ref operator[](int row) const { return { bytes, row }; }

    // Four bits are unsigned, so TRANSPARENT (-1) has nowhere to stay. What goes
    // in the byte is therefore the real value plus one: TRANSPARENT becomes 0,
    // black becomes 1, and so on.
    ARC_HOST_DEVICE static constexpr uint8_t enc(int8_t value) {
        return (uint8_t)(((int)value + 1) & 0xF);
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
};

// End of warning suppression
#ifdef _MSC_VER
#pragma warning(pop)
#endif


// Caller-owned so host and device can lend memory they already hold.
struct object_walk_scratch {
    bool taken[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_DIM * MAX_GRID_DIM];
    int16_t queue_col[MAX_GRID_DIM * MAX_GRID_DIM];
    int16_t cells[MAX_GRID_DIM * MAX_GRID_DIM];
};

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
                                 bool any_color, object_walk_scratch& scratch, Take take) {
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
            // TRANSPARENT is not a color at all; `background` is the canvas this
            // grid was found to sit on, and -1 from the election means the grid
            // is all figure and nothing is excluded.
            if (color < 0 || color == background) { taken[row][col] = true; continue; }

            taken[row][col] = true;
            queue_row[0] = (int16_t)row; queue_col[0] = (int16_t)col;
            object_summary found;
            int seen = 0;
            int x0 = col, y0 = row, x1 = col, y1 = row;
            // One flood fill for the whole project, so no two readings can
            // disagree about which cells belong to one group.
            found.count = (int16_t)walk_out(one_grid.height, one_grid.width,
                queue_row, queue_col, 1,
                [&](int next_row, int next_col) {
                    if (taken[next_row][next_col]) return false;
                    const int there = one_grid.data[next_row][next_col];
                    if (any_color ? (there < 0 || there == background)
                                  : (there != color)) return false;
                    taken[next_row][next_col] = true;
                    return true;
                },
                [&](int seen_row, int seen_col) {
                    found_cells[seen++] = (int16_t)(seen_row * MAX_GRID_DIM + seen_col);
                    x0 = min(x0, seen_col); x1 = max(x1, seen_col);
                    y0 = min(y0, seen_row); y1 = max(y1, seen_row);
                });
            found.x0 = (int8_t)x0;
            found.y0 = (int8_t)y0;
            found.w  = (int8_t)(x1 - x0 + 1);
            found.h  = (int8_t)(y1 - y0 + 1);
            take(found, found_cells, seen);
        }
}

// The groups themselves, cells and all, in the order the grid gives them up.
// One allocation per group, which is nothing for the handful of objects a
// normal puzzle breaks into and 900 small allocations for a grid that is all
// single cells. If that worst case ever lands on a hot path, the answer is to
// pack every group's cells into one block and hand out runs of it -- not to
// go back to making the caller hold the cells separately.
inline vector<object> find_objects(const grid& one_grid, int background) {
    vector<object> found;
    object_walk_scratch scratch;
    each_object(one_grid, background, /*any_color=*/false, scratch, [&](const object_summary& one,
                                       const int16_t* cells, int n) {
        found.push_back({});
        object& made = found.back();
        made.summary = one;
        made.cells.clear();  // every cell TRANSPARENT until the group claims it
        for (int i = 0; i < n; ++i) {
            const int row = cells[i] / MAX_GRID_DIM, col = cells[i] % MAX_GRID_DIM;
            made.cells[row - one.y0][col - one.x0] = (int8_t)one_grid.data[row][col];
        }
    });
    return found;
}

// The same walk, forgetting the cells on the way out. Writes at most max_out
// and returns how many; a grid with more groups than that silently loses the
// rest, so callers size the array for the worst case they care about.
inline int find_objects_by_color(const grid& one_grid, int background,
                                 object_summary* out, int max_out) {
    int found = 0;
    object_walk_scratch scratch;
    each_object(one_grid, background, /*any_color=*/false, scratch, [&](const object_summary& one,
                                       const int16_t*, int) {
        if (found < max_out) out[found++] = one;
    });
    return found;
}

// Tallies; how much of each color, and where it sits

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

// POSES · the ways a grid can sit, and naming the shape that results

// THE EIGHT WAYS A GRID CAN SIT: four turns by two mirrors, packed in three
// bits, pose 0 being the grid as given.
//   1  rows and columns trade places, transpose
//   2  flipped top to bottom
//   4  flipped left to right
// Single home of what the bits mean. Every reader of a pose goes through the
// three below, so none of them can drift on what "the same shape, turned" is.
constexpr int POSE_COUNT = 8;

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

// Where a cell of the ORIGINAL lands once posed, so a place can be carried into
// the frame a pair is compared in. The exact inverse of pose_source.
ARC_HOST_DEVICE inline std::pair<int, int> posed_place(int pose, int width, int height,
                                  int row, int col) {
    const int down   = (pose & 2) ? height - 1 - row : row;
    const int across = (pose & 4) ? width  - 1 - col : col;
    return (pose & 1) ? std::pair<int, int>{ across, down } : std::pair<int, int>{ down, across };
}

// Turn a shape into one number, so "is this the same shape?" becomes "are these
// two numbers equal?". Shape here means which cells of the box around it are
// filled, plus w and h. Not the color, not where it sits on the grid.
// The CPU stores these numbers and the GPU looks them up, so if the two ever
// hashed differently every lookup would miss. Hence one function, called by both.
//
// occ(i) says whether cell i is filled, counting across each row in turn. Every
// caller must count the same way or the two sides hash the same shape apart.
// The second template parameter demands a function: without it a plain bool
// array lands here instead of the array-taking one just below.
template <typename OccupancyAt,
          typename = decltype(std::declval<OccupancyAt&>()(0))>
ARC_HOST_DEVICE inline uint32_t shape_occupancy_hash(int w, int h, OccupancyAt occ) {
    uint32_t hsh = 2166136261u;
    auto mix = [&](uint8_t b) { hsh = (hsh ^ b) * 16777619u; };
    mix((uint8_t)w); mix((uint8_t)h);
    for (int i = 0; i < w * h; ++i) mix(occ(i) ? (uint8_t)1 : (uint8_t)0);
    return hsh;
}

// Adapts a packed bool occupancy array to the recipe above.
inline uint32_t shape_occupancy_hash(int width, int height, const bool* occupied) {
    return arc::shape_occupancy_hash(width, height, [&](int i) { return occupied[i]; });
}

// Canonical (orientationless, positionless) shape identity: the smallest
// shape_occupancy_hash over the eight ways round, done by index arithmetic —
// no pose grids are built. Two objects share a shape, in any orientation, iff
// their canonical hashes are equal.
inline uint32_t canonical_shape_hash(int width, int height, const bool* occupied) {
    uint32_t best = 0xFFFFFFFFu;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, width, height);
        uint32_t hash = arc::shape_occupancy_hash(posed_width, posed_height, [&](int i) {
            const auto [row, col] = pose_source(pose, width, height, i / posed_width,
                                                i % posed_width);
            return occupied[row * width + col];
        });
        best = min(best, hash);
    }
    return best;
}

}  // namespace arc
