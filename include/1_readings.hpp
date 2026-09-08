// This file describes how to read a grid at all
#pragma once
#include "0_core.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>

// This is a tag that makes code visible
// to a CUDA device and CUDA host,
// otherwise it is host-only.
#ifdef __CUDACC__
#define ARC_HOST_DEVICE __host__ __device__
#else
#define ARC_HOST_DEVICE
#endif

namespace arc {
using namespace std;

// Grouping is 8, "is this one object?".
constexpr int NEIGHBOUR_ROW_8[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
constexpr int NEIGHBOUR_COL_8[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

// Adjacency is 4, "what is beside this?".
constexpr int NEIGHBOUR_ROW_4[4] = {-1, 1, 0, 0};
constexpr int NEIGHBOUR_COL_4[4] = { 0, 0,-1, 1};

// Not an ARC colour: a cell of the real puzzle is 0..9. This is our own
// stand-in for "nothing here", which masks and glyphs need and the grids
// ARC ships never do.
constexpr int TRANSPARENT = -1;

// ...which is what the >= 0 turns away here.
ARC_HOST_DEVICE constexpr bool is_color(int value) {
    return value >= 0 && value < COLOR_COUNT;
}

// A WHOLE GRID OF CELLS IN HALF THE ROOM, and the one cell store both the CPU
// and the GPU can read. A cell is 4 bits, two to a byte, so a 30x30 grid costs
// 450 bytes instead of 900. It holds no addresses, which is what lets the same
// bytes be handed to the device and still mean what they meant here.
//
// DIAG: the proxy clamps an out-of-bounds packed-mask cell access to cell 0 so it
// can't fault. If clamping makes the kernel illegal-access vanish, the bug is a
// mask OOB. (printf here blows the ptxas budget under launch_bounds, so no log.)
#define PACKED_GRID_OOB(r, c, i) ((void)0)

// 4-bit packed 30x30 cell mask: 2 cells/byte (450B vs 900B). Value v in [-1,9]
// (GPU_TRANSPARENT..colors) stored as nibble (v+1) in [0,10]. Proxies expose
// mask[r][c] read/write so lane-0 ops and all reads are unchanged. WARNING:
// the proxy write is a byte read-modify-write, so it is ONLY safe single-
// threaded; warp ops that write cells in parallel must write whole bytes via
// .data[] directly (two cells/byte) to avoid the shared-byte race.
struct packed_grid {
    static constexpr int CELLS = MAX_GRID_DIM * MAX_GRID_DIM;
    uint8_t data[(CELLS + 1) / 2];

    struct cref {
        uint8_t* b; int sh;
        ARC_HOST_DEVICE operator int8_t() const { return (int8_t)(((*b) >> sh) & 0xF) - 1; }
        ARC_HOST_DEVICE void operator=(int8_t v) {
            uint8_t n = (uint8_t)((int)v + 1) & 0xF;
            *b = (uint8_t)(((*b) & ~(0xF << sh)) | (n << sh));
        }
    };
    struct ccref {
        const uint8_t* b; int sh;
        ARC_HOST_DEVICE operator int8_t() const { return (int8_t)(((*b) >> sh) & 0xF) - 1; }
    };
    struct row  { uint8_t* base; int r;
        ARC_HOST_DEVICE cref  operator[](int c) const { int i = r * MAX_GRID_DIM + c;
            if (i < 0 || i >= CELLS) { PACKED_GRID_OOB(r, c, i); i = 0; }   // DIAG: clamp OOB to prevent the fault
            return cref { base + (i >> 1), (i & 1) * 4 }; } };
    struct crow { const uint8_t* base; int r;
        ARC_HOST_DEVICE ccref operator[](int c) const { int i = r * MAX_GRID_DIM + c;
            if (i < 0 || i >= CELLS) { PACKED_GRID_OOB(r, c, i); i = 0; }   // DIAG: clamp OOB to prevent the fault
            return ccref{ base + (i >> 1), (i & 1) * 4 }; } };

    ARC_HOST_DEVICE row  operator[](int r)       { return row { data, r }; }
    ARC_HOST_DEVICE crow operator[](int r) const { return crow{ data, r }; }
    ARC_HOST_DEVICE void clear() { for (int i = 0; i < (int)sizeof(data); ++i) data[i] = 0; }  // all transparent
    // Pack value v into cell (r,c) — for use inside a single byte by one lane.
    ARC_HOST_DEVICE static uint8_t enc(int8_t v) { return (uint8_t)((int)v + 1) & 0xF; }
};

// Pack an unpacked MAX_GRID_DIM^2 int8 grid into a packed grid, cell by cell.
ARC_HOST_DEVICE inline void pack_grid(packed_grid& m, const int8_t g[][MAX_GRID_DIM]) {
    for (int r = 0; r < MAX_GRID_DIM; ++r)
        for (int c = 0; c < MAX_GRID_DIM; ++c)
            m[r][c] = g[r][c];
}


// OBJECTS · the related groups of cells a grid breaks into

// Walk out from the cells already sitting in the queue until nothing new can be
// reached, corners always counting. `claim` both decides and marks: it answers
// "is this cell part of the walk", and a yes means the walk has taken it, so the
// caller's own mark is what keeps a cell from being queued twice. `see` gets
// every cell once, in the order the walk reached it. Returns how many cells the
// walk ended with, left in the queue for the caller to read back.
template <class Claim, class See>
ARC_HOST_DEVICE inline int walk_out(int rows, int cols, int16_t queue_row[], int16_t queue_col[], int seeded,
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
            queue_row[tail] = (int16_t)next_row; queue_col[tail] = (int16_t)next_col;
            ++tail;
        }
    }
    return tail;
}

// AN OBJECT IS A RELATED GROUP OF CELLS. That much never changes. What changes
// from puzzle to puzzle is what RELATED means, and there are several answers:
// same color and touching; any non-background cells touching, so one object may
// wear several colors; same color anywhere on the grid, touching or not; or one
// period of a repeating block. Those relations are split_kind in
// 3_correspondence.hpp, and choosing between them is inference's job. What is
// below uses one of them: same color, touching.

// A STRUCT OF ITS OWN, AND NOT JUST FIELDS INSIDE `object`, BECAUSE IT HAS TO
// GO WHERE CELLS CANNOT. The scorer keeps an array of these in every puzzle
// and ships them to the GPU; a group that owned its cells would carry a
// vector there, which means a heap allocation for each one and a struct that
// can no longer be copied as plain bytes. So the half without cells is its
// own thing, free to be copied about and uploaded by the thousand.
//
// What it keeps is a color, the box its cells fit in, and how many there are.
// Two different groups sharing those three are equal as far as this can tell,
// which is all the scorer needs: it only asks whether a group this shape and
// this size exists in the answer.
//
// It is written above `object` because a thing must be declared before it can
// be used as part of another, not because it is the more important of the two.
struct object_summary {
    int8_t  color = -1;      // 0..9; -1 would mean a group of mixed colors
    int8_t  x0 = 0, y0 = 0;  // top-left of its box, in grid coordinates
    int8_t  w = 0, h = 0;    // how big that box is
    int16_t count = 0;       // cells in the group, which is <= w * h
};

// THE GROUP ITSELF: the summary above, and the cells it is actually made of,
// kept in a packed_grid so the whole thing is plain bytes -- no addresses, so
// an object means the same on the host and on the device, and one segmenter can
// one day serve both.
//
// The cells are BOX-LOCAL: cell (row, col) of the grid lives at
// cells[row - summary.y0][col - summary.x0], and everything outside the group is
// TRANSPARENT. That is the same convention fixed_object uses on the GPU side.
// Use color_at / holds rather than reaching into `cells`, and pass them ordinary
// grid coordinates.
//
// Knowing the cells and not just the box is what lets `holds` exist at all. A
// box cannot answer it: a ring's box holds the ring's own hole, so anything
// sitting in the hole looks like it is part of the ring.
struct object {
    object_summary summary;
    packed_grid cells;

    // What this group puts at that place on the grid, or TRANSPARENT if the
    // group is not there at all -- which covers both outside its box and inside
    // the box but not part of the shape.
    ARC_HOST_DEVICE int8_t color_at(int row, int col) const {
        const int r = row - summary.y0, c = col - summary.x0;
        if (r < 0 || r >= summary.h || c < 0 || c >= summary.w) return (int8_t)TRANSPARENT;
        return cells[r][c];
    }

    ARC_HOST_DEVICE bool holds(int row, int col) const {
        return color_at(row, col) != (int8_t)TRANSPARENT;
    }
};

// Find every same-color touching group and hand each one to `take` as it is
// found: its summary, and its cells as a plain pointer and length. Both finders
// below are this walk wearing different hats, so neither can drift from the
// other on what counts as one object.
//
// The cells handed over live in scratch space that the next group overwrites,
// so `take` must COPY anything it means to keep -- keeping the pointer itself
// buys a view of whatever group comes next, and of dead stack once the walk
// returns. Nothing enforces that: it is the price of handing out cells without
// allocating, and it is what lets the summary finder below run with no
// allocation at all, which matters because it is called once per program the
// solver tries.
//
// `background` is the color this grid was ELECTED to sit on, which the caller
// reads off inference -- not a guess. Black is the usual answer and not the
// only one: a puzzle can put black figures on a colored ground, and assuming
// otherwise loses them. Pass -1 to exclude nothing but TRANSPARENT.
// Touching is the 8-way kind (see NEIGHBOUR_ROW_8 above): a shape joined only
// at a corner is still one shape.
// The room the walk needs while it works. The caller owns it because the device
// has no stack to spare for ten kilobytes: on the host this is a local like any
// other, but a kernel would hand over a slice of memory it already holds.
struct object_walk_scratch {
    bool taken[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_DIM * MAX_GRID_DIM];
    int16_t queue_col[MAX_GRID_DIM * MAX_GRID_DIM];
    int16_t cells[MAX_GRID_DIM * MAX_GRID_DIM];
};

// Grid is any type with width, height and data[row][col] -- arc::grid on the
// host, the GPU's own smaller grid in a kernel. One walk, so the two sides can
// never come to different answers about what one object is.
template <class Grid, class Take>
ARC_HOST_DEVICE inline void each_object_by_color(const Grid& one_grid, int background,
                                 object_walk_scratch& scratch, Take take) {
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
            found.color = (int8_t)color;
            int seen = 0;
            int x0 = col, y0 = row, x1 = col, y1 = row;
            // One flood fill for the whole project, so no two readings can
            // disagree about which cells belong to one group.
            found.count = (int16_t)walk_out(one_grid.height, one_grid.width,
                queue_row, queue_col, 1,
                [&](int next_row, int next_col) {
                    if (taken[next_row][next_col]) return false;
                    if (one_grid.data[next_row][next_col] != color) return false;
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
    each_object_by_color(one_grid, background, scratch, [&](const object_summary& one,
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
    each_object_by_color(one_grid, background, scratch, [&](const object_summary& one,
                                       const int16_t*, int) {
        if (found < max_out) out[found++] = one;
    });
    return found;
}

// TALLIES · how much of each color, and where it sits

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

// POSES · the ways a grid can sit, and naming the shape that results

// THE EIGHT WAYS A GRID CAN SIT: four turns by two mirrors, packed in three
// bits, pose 0 being the grid as given.
//   1  rows and columns trade places
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
