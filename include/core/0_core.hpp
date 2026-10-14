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

// A cell's place in the largest grid as one number, counted row by row. Lists
// of cells hold these, so a cell costs two bytes rather than a pair, and the
// packed store of an object finds its cells the same way.
ARC_HOST_DEVICE constexpr int cell_index(int row, int col) { return row * MAX_GRID_DIM + col; }
ARC_HOST_DEVICE constexpr int cell_row(int index) { return index / MAX_GRID_DIM; }
ARC_HOST_DEVICE constexpr int cell_col(int index) { return index % MAX_GRID_DIM; }

// The ARC demonstration example.
struct example {
    grid input, output;
};

}  // namespace arc
