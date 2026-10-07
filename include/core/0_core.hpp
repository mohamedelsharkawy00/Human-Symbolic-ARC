// Start here.
// This file depends on no other file in the project,
// and every other file depends on it.
// It sets the words the rest of the project is written in.
#pragma once
#include <cstdint>

// Shared helpers can run on the host and on the CUDA device.
#ifdef __CUDACC__
#define ARC_HOST_DEVICE __host__ __device__
#else
#define ARC_HOST_DEVICE
#endif

namespace arc {

// The ARC cell and the ten colors it can hold.
enum class arc_color : int8_t {
    black = 0, blue = 1, red = 2, green = 3,
    yellow = 4, gray = 5, magenta = 6, orange = 7,
    cyan = 8, maroon = 9, count = 10
};
constexpr int COLOR_COUNT = (int)arc_color::count;
constexpr int TRANSPARENT = -1;

ARC_HOST_DEVICE constexpr bool is_color(int value) {
    return value >= 0 && value < COLOR_COUNT;
}

// The ARC grid, its max size is 30 x 30.
// grid(width, height) is a grid of that size with every cell 0.
constexpr int MAX_GRID_DIM = 30;
constexpr int MAX_GRID_CELLS = MAX_GRID_DIM * MAX_GRID_DIM;
struct grid {
    int width = 0, height = 0;
    int data[MAX_GRID_DIM][MAX_GRID_DIM]{};
    // A GRID LAID OUT OF COPIES of one tile_width by tile_height picture, 0 by 0
    // where it is one picture: no piece crosses from one copy into the next
    // (each_object), as the program cuts the picture before it tiles it
    // (10fcaaa3, framed_whole's tiled frame).
    int tile_width = 0, tile_height = 0;

    // No cells at all, so there is nothing to read off it.
    bool empty() const {
        return width <= 0 || height <= 0;
    }

    bool contains(int row, int col) const {
        return row >= 0 && row < height && col >= 0 && col < width;
    }

    bool same_size(const grid& other) const {
        return width == other.width && height == other.height;
    }

    void fill(int color) {
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                data[row][col] = color;
    }

    // Only the drawing counts; unused storage outside its dimensions does not.
    bool operator==(const grid& other) const {
        if (!same_size(other)) return false;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                if (data[row][col] != other.data[row][col]) return false;
        return true;
    }
};

// The size of the copies a grid is laid out of, 0 by 0 for one picture: a
// host grid's own (grid::tile_width); every other kind of grid is one picture.
struct tile_size { int width = 0, height = 0; };
template <class Grid>
ARC_HOST_DEVICE inline tile_size tiles_of(const Grid&) { return {}; }
ARC_HOST_DEVICE inline tile_size tiles_of(const grid& one) { return { one.tile_width, one.tile_height }; }

// The ARC demonstration example.
struct example {
    grid input, output;
};

// HOW MANY OBJECTS ONE LIST HOLDS while a program runs: one for every cell of
// the largest grid, so no cut of a grid runs out of room. The kernel's list is
// this long, so an op handed more than this hands back this many -- and a
// reading that says what an op will draw has to stop counting at the same
// place, or it claims a drawing the op would not make. 12_dsl.cuh names it
// GPU_KERNEL_MAX_OBJECTS on its own side of the fence. It was 32 while search
// ran thousands of programs at once; one reading runs a handful.
constexpr int MAX_OBJECTS_IN_A_LIST = MAX_GRID_CELLS;

}  // namespace arc
