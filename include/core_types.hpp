#pragma once
#include <cstdint>
#include <utility>

// Host/device marker: expands to nothing for plain C++ compilers.
#ifdef __CUDACC__
#define ARC_HOST_DEVICE __host__ __device__
#else
#define ARC_HOST_DEVICE
#endif

// FNV-1a fingerprint of a glyph's bbox occupancy (1 = glyph pixel, 0 = empty),
// seeded with w,h. Single home of the recipe: the CPU hashes learned keys
// (inference/1_readings.hpp) and the kernel hashes again at apply time
// both call this, so learned keys cannot silently miss.
// occ(i): is pixel i (row-major within the bbox) part of the glyph.
template <typename OccupancyAt>
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

namespace arc {


// Below both the engine and inference, so either can be read without the other.
// Reached through inference/4_infer.hpp, the one door to everything below it.

// The pieces a puzzle is made of: colors, grids, one train pair.
enum class arc_color : int8_t {
	black = 0,
	blue = 1,
	red = 2,
	green = 3,
	yellow = 4,
	gray = 5,
	magenta = 6,
	orange = 7,
	cyan = 8,
	maroon = 9,
	count = 10
};

// --- Domain & Type Limits, single home; the GPU_* names alias these ---
constexpr int MAX_GRID_DIM = 30;
constexpr int COLOR_COUNT = (int)arc_color::count;
// Sentinel: "no pixel here" in masks
constexpr int TRANSPARENT = -1;
// How many pairs of two different colors there are, which is how many ways
// two colors can trade places. The swap op counts them in this same order:
// {a, b} with a < b, row-major.
constexpr int COLOR_PAIR_COUNT = COLOR_COUNT * (COLOR_COUNT - 1) / 2;

// Is this one of the ten colors? Single home of the question, so the floor
// stops being written as a bare 0 and the roof stays COLOR_COUNT. Everything
// else is not a color, TRANSPARENT above all.
ARC_HOST_DEVICE constexpr bool is_color(int value) {
	return value >= 0 && value < COLOR_COUNT;
}

constexpr float SCORE_MIN = -1000000.0f;

// What "next to" means, in the two readings a grid ever uses: all eight
// neighbours with corners counted, and the four that share an edge. Single home
// of the offsets, so no two walks over a grid can disagree about who touches.
// GROUPING IS 8, "is this one thing?". ADJACENCY IS 4, "what is beside this?".
// Which one is not a preference, so a 4 needs no comment and anything else does.
// Why, and what unifying them would silently move:
// notes/4 - To Do/Connectivity Is Two Questions.md
constexpr int NEIGHBOUR_ROW_8[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
constexpr int NEIGHBOUR_COL_8[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
constexpr int NEIGHBOUR_ROW_4[4] = {-1, 1, 0, 0};
constexpr int NEIGHBOUR_COL_4[4] = { 0, 0,-1, 1};

struct grid {
	int width = 0, height = 0;
	int data[MAX_GRID_DIM][MAX_GRID_DIM]{};

	// Where this grid sits in the renderer's cell buffer, and whether to draw it.
	// Set only by whoever puts it there (grid_buffer.hpp); nothing below the
	// screen reads them, so a grid that is never drawn leaves them alone.
	int offset = 0, stride = 0;
	bool visible = true;

	// Nothing here needs hand-copying, so the compiler lists the fields, not us.
	grid() = default;
	grid(int w, int h) : width(w), height(h) {}

	uint64_t hash() const {
		uint64_t h = 0;
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				h ^= data[r][c] + 0x9e3779b9 + (h << 6) + (h >> 2);
			}
		}
		h ^= width + 0x9e3779b9 + (h << 6) + (h >> 2);
		h ^= height + 0x9e3779b9 + (h << 6) + (h >> 2);
		return h;
	}
};

struct example {
	grid input, output, current, best;
	float best_score = SCORE_MIN, cur_score = SCORE_MIN;
	int distance_maps[COLOR_COUNT][MAX_GRID_DIM][MAX_GRID_DIM]{};
};

}
