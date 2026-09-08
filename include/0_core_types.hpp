// Start here. This file depends on no other file in the project, and every other
// file depends on it.
// It sets the words the rest of the project is written in.
#pragma once
#include <cstdint>
#include <utility>

// This tag is what makes code visible to a CUDA device and CUDA host,
// otherwise it is host-only.
#ifdef __CUDACC__
#define ARC_HOST_DEVICE __host__ __device__
#else
#define ARC_HOST_DEVICE
#endif

namespace arc {

// The ten colors a cell can hold.
enum class arc_color : int8_t {
	black = 0, blue = 1, red = 2, green = 3, yellow = 4, gray  = 5,
    magenta = 6, orange = 7, cyan = 8, maroon = 9, count = 10
};

constexpr int COLOR_COUNT = (int)arc_color::count;

// 10 choose 2.
constexpr int COLOR_PAIR_COUNT = COLOR_COUNT * (COLOR_COUNT - 1) / 2;

// Sentinel: "no pixel here" in masks.
constexpr int TRANSPARENT = -1;

// TRANSPARENT is -1, so the >= 0 is what keeps it out: it is not a color.
ARC_HOST_DEVICE constexpr bool is_color(int value) {
	return value >= 0 && value < COLOR_COUNT;
}

// Grouping is 8, "is this one object?".
constexpr int NEIGHBOUR_ROW_8[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
constexpr int NEIGHBOUR_COL_8[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

// Adjacency is 4, "what is beside this?".
constexpr int NEIGHBOUR_ROW_4[4] = {-1, 1, 0, 0};
constexpr int NEIGHBOUR_COL_4[4] = { 0, 0,-1, 1};

// The largest grid size ARC uses is 30 x 30.
constexpr int MAX_GRID_DIM = 30;

// Copying is never written out: every field is plain data, so the compiler's own
// copy of all of them is the right one.
struct grid {
	int width = 0, height = 0;
	int data[MAX_GRID_DIM][MAX_GRID_DIM]{};

	// The three below are only for drawing this grid on the GUI; the solver never
	// reads them.

	// Where its cells were put in the renderer's one big buffer.
	int offset = 0;
	// How far apart two rows sit in that buffer.
	int stride = 0;
	// Whether to draw it at all.
	bool visible = true;
};

// The worst a score can be, so an untried example starts below every real one.
constexpr float SCORE_MIN = -1000000.0f;

struct example {
	grid input, output, current, best;
	float best_score = SCORE_MIN, cur_score = SCORE_MIN;
	int distance_maps[COLOR_COUNT][MAX_GRID_DIM][MAX_GRID_DIM]{};
};

// Turn a shape into one number, so "is this the same shape?" becomes "are these
// two numbers equal?". Shape here means which cells of the box around it are
// filled, plus w and h. Not the color, not where it sits on the grid.
// The CPU stores these numbers and the GPU looks them up, so if the two ever
// hashed differently every lookup would miss. Hence one function, called by both.
//
// occ(i) says whether cell i is filled, counting across each row in turn. Every
// caller must count the same way or the two sides hash the same shape apart.
// The second template parameter demands a function: without it a plain bool
// array lands here instead of the array-taking version in 1_readings.hpp.
template <typename OccupancyAt,
          typename = decltype(std::declval<OccupancyAt&>()(0))>
ARC_HOST_DEVICE inline uint32_t shape_occupancy_hash(int w, int h, OccupancyAt occ) {
    uint32_t hsh = 2166136261u;
    auto mix = [&](uint8_t b) { hsh = (hsh ^ b) * 16777619u; };
    mix((uint8_t)w); mix((uint8_t)h);
    for (int i = 0; i < w * h; ++i) mix(occ(i) ? (uint8_t)1 : (uint8_t)0);
    return hsh;
}

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

// The smallest block a grid repeats: the shortest run of columns that repeats
// across it, and of rows that repeats down it. A grid that repeats nothing gets
// back its own width and height. Grid needs width, height, and data[row][col].
// Single home of the recipe: the Pattern op reads the unit by it and inference
// cuts by it (split_kind::repeating_motif), so the two can never read a
// different period off the same grid.
// `background` is don't care one way round only: a colored cell must be what the
// block puts there, a background cell where the block has a color is forgiven.
// Pass a value no cell can hold (-2) to compare raw.
// `must_fit_twice` is the caller's question, not a setting: inference asks "does
// this repeat?" and needs two whole copies, the Pattern op asks "what unit do I
// continue with?" and needs one copy and a matching stub.
// Why smallest, why there is no biggest, and what each rule costs:
// notes/1 - Overall Design/The Smallest Repeating Block.md
template <typename Grid>
ARC_HOST_DEVICE inline void repeat_periods(const Grid& g, int background,
                                           int& width, int& height,
                                           bool must_fit_twice = true) {
    // Does the grid repeat every `period` cells along one axis? `across` asks
    // about columns, otherwise rows. One test for both, so they cannot drift.
    auto repeats_every = [&](int period, bool across) {
        for (int r = across ? 0 : period; r < g.height; ++r)
            for (int c = across ? period : 0; c < g.width; ++c) {
                const int here = g.data[r][c];
                if (here == background) continue;
                const int folded_row = across ? r : r % period;
                const int folded_col = across ? c % period : c;
                if (here != g.data[folded_row][folded_col]) return false;
            }
        return true;
    };
    // The shortest period that holds, or the whole size when none does. Past
    // halfway there is too little left to compare, so a long period passes on a
    // cell or two and means nothing.
    // Why two whole copies, and what it costs:
    // notes/4 - To Do/A Repeat Has To Fit Twice.md
    auto smallest_period = [&](int size, bool across) {
        const int longest = must_fit_twice ? size / 2 : size - 1;
        for (int period = 1; period <= longest; ++period)
            if (repeats_every(period, across)) return period;
        return size;
    };
    width = smallest_period(g.width, true);
    height = smallest_period(g.height, false);
}

}  // namespace arc
