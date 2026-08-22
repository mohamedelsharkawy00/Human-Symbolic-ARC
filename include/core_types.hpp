#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>

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

// The smallest block a grid repeats: the shortest run of columns that
// repeats across it, and of rows that repeats down it. A grid that repeats
// nothing gets back its own width and height. Single home of the recipe: the
// the Pattern op reads the unit by it and inference cuts by it
// (split_kind::repeating_motif), so the two can never read a different period
// off the same grid.
//
// SMALLEST, and there is no threshold anywhere in it. A grid is a WINDOW on
// a pattern that carries on outside, so a copy that starts inside the frame and
// runs off it still counts — 017c7c7b shows a 4-row motif in a 6-row window.
// Asking instead for the BIGGEST block needs a floor to mean anything (every
// offset above the true period passes a trailing comparison too, so the biggest
// is always one less than the size), and every floor either throws that window
// away or makes the block depend on how much of the grid you were shown.
// The smallest is a property of the pattern itself.
//
// `background` is DON'T CARE, one way round only: a colored cell must be the
// cell the block puts there, while a background cell where the block has a color is
// forgiven — so a motif standing on uneven ground still has a block. Forgiving
// it both ways lets a block of pure background explain a whole grid by never
// being compared to anything. Pass a value no cell can hold (-2) to compare raw.
// Grid needs width, height, and data[row][col].
// THE SMALLEST ONLY, and there is no biggest to ask for. Periods behave like
// multiples — once a grid repeats every p it repeats every multiple of p —
// so the largest is a greatest common multiple, which does not exist. It had an
// answer only under an added rule, must divide the grid exactly, and what it
// found was then always a whole number of the smallest. Removed 2026-08-12.
template <typename Grid>
// `must_fit_twice` is the caller's question, not a setting. Asking "does this
// repeat?" is a claim about the picture and needs two whole copies as evidence
// (inference). Asking "what unit do I continue with?" is a different question --
// the output already says it continues, so one whole copy and a matching stub
// names the unit (the Pattern op). 017c7c7b pair 1 is 6 tall with a 4-tall
// motif: real, and invisible to the stricter question.
ARC_HOST_DEVICE inline void repeat_periods(const Grid& g, int background,
                                           int& width, int& height,
                                           bool must_fit_twice = true) {
    width = g.width; height = g.height;
    auto row_holds = [&](int w) {
        for (int r = 0; r < g.height; ++r)
            for (int c = w; c < g.width; ++c) {
                const int here = g.data[r][c];
                if (here == background) continue;
                if (here != g.data[r][c % w]) return false;
            }
        return true;
    };
    auto col_holds = [&](int h) {
        for (int c = 0; c < g.width; ++c)
            for (int r = h; r < g.height; ++r) {
                const int here = g.data[r][c];
                if (here == background) continue;
                if (here != g.data[r % h][c]) return false;
            }
        return true;
    };
    // IT HAS TO FIT TWICE OR IT DID NOT REPEAT. Past halfway there is almost
    // nothing left to compare against, so a long period passes on one or two
    // cells and means nothing: a 3-wide row of 010 "repeats every 2" on the
    // strength of column 2 alone. That false period then becomes the motif and
    // drives the whole reading — 017c7c7b read as 2x2 blocks when the motif is
    // the full 3 wide. No period fitting twice means the grid does not
    // repeat on that axis, and the width stays whole.
    // The cost of this rule is in notes/4 - To Do/A Repeat Has To Fit Twice.md.
    const int widest = must_fit_twice ? g.width / 2 : g.width - 1;
    const int tallest = must_fit_twice ? g.height / 2 : g.height - 1;
    for (int w = 1; w <= widest; ++w)
        if (row_holds(w)) { width = w; break; }
    for (int h = 1; h <= tallest; ++h)
        if (col_holds(h)) { height = h; break; }
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

constexpr float SCORE_MIN = -1000000.0f;

// What "next to" means, in the two readings a grid ever uses: all eight
// neighbours with corners counted, and the four that share an edge. Single home
// of the offsets, so no two walks over a grid can disagree about who touches.
//
// WHICH ONE TO USE IS NOT A PREFERENCE. The two answer different questions:
//
//   GROUPING IS 8 — "is this one thing?" A diagonal join makes one object out
//   of two, which is what a person sees. Every cut in inference groups through
//   walk_out (inference/1_readings.hpp), which is 8 and offers no alternative.
//
//   ADJACENCY IS 4 — "what is beside this?" A diagonal is a corner touch and
//   not a border, so a wall with a diagonal gap must not leak through it.
//
// So a 4 here is an adjacency question and needs no further comment; anything
// else has to say why. Unifying them for tidiness would silently move every
// score that leans on one — see notes/4 - To Do/Connectivity Is Two Questions.md.
constexpr int NEIGHBOUR_ROW_8[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
constexpr int NEIGHBOUR_COL_8[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
constexpr int NEIGHBOUR_ROW_4[4] = {-1, 1, 0, 0};
constexpr int NEIGHBOUR_COL_4[4] = { 0, 0,-1, 1};

struct grid {
	mutable std::atomic<int> ref_count;
	int width = 0, height = 0, offset = 0, stride = MAX_GRID_DIM;
	bool visible = true;
	int data[MAX_GRID_DIM][MAX_GRID_DIM]{};

	grid() : ref_count(1) { memset(data, 0, sizeof(data)); }
	grid(int w, int h, int off, int str) : ref_count(1), width(w), height(h), offset(off), stride(str) { memset(data, 0, sizeof(data)); }
	grid(const grid& other) : ref_count(1), width(other.width), height(other.height), offset(other.offset), stride(other.stride), visible(other.visible) {
		memcpy(data, other.data, sizeof(data));
	}
	grid& operator=(const grid& other) {
		if (this != &other) {
			width = other.width; height = other.height; offset = other.offset;
			stride = other.stride; visible = other.visible;
			memcpy(data, other.data, sizeof(data));
		}
		return *this;
	}

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

	// Use a thread-local free list to avoid Ntdll heap contention
	// and significantly speed up allocations under heavy churn.
	static void* operator new(size_t size) {
		return ::operator new(size);
	}

	static void operator delete(void* ptr) noexcept {
		::operator delete(ptr);
	}
};

struct example {
	grid input, output, current, best;
	float best_score = SCORE_MIN, cur_score = SCORE_MIN;
	int distance_maps[(int)arc_color::count][MAX_GRID_DIM][MAX_GRID_DIM]{};
};

}
