// Start here.
// This file depends on no other file in the project,
// and every other file depends on it.
// It sets the core concepts the rest of the project is written in.
#pragma once
#include <cstdint>

namespace arc {

// The ARC cell and the ten colors it can hold.
enum class arc_color : int8_t {
    black = 0, blue = 1, red = 2, green = 3,
    yellow = 4, gray = 5, magenta = 6, orange = 7,
    cyan = 8, maroon = 9, count = 10
};

// The number of colors, just to avoid magic numbers in the code.
constexpr int COLOR_COUNT = (int)arc_color::count;

// A special value for an empty cell that is used internally in the solver
constexpr int TRANSPARENT = -1;

// A cell a piece holds with no color: empty, but the piece's. Partition hands a
// part's empty cells back as one piece of these, a room, for a recolor to
// paint; no grid ever shows one.
constexpr int BLANK = COLOR_COUNT;

// Check if a value is a valid color, not anything else.
constexpr bool is_color(int value) {
    return value >= 0 && value < COLOR_COUNT;
}

// The ARC grid, its max size is 30 x 30.
constexpr int MAX_GRID_DIM = 30;
constexpr int MAX_GRID_CELLS = MAX_GRID_DIM * MAX_GRID_DIM;

// A cell's place in the largest grid as one number, counted row by row, and
// back again. One number is for speed: a flat store or set finds a cell at a
// look rather than a search, and a set of cells fits in 900 bits.
constexpr int cell_index(int row, int col) { return row * MAX_GRID_DIM + col; }
constexpr int cell_row(int index) { return index / MAX_GRID_DIM; }
constexpr int cell_col(int index) { return index % MAX_GRID_DIM; }

// A byte a cell, in reading order: its color, or TRANSPARENT where it is empty.
struct cells {
    int8_t values[MAX_GRID_CELLS];

    // A new store is empty.
    cells() { clear(); }

    // A row, so data[row][col] reaches a cell. A place past the store reaches
    // the first cell rather than memory outside it.
    template <class Cell>
    struct row_of_cells {
        Cell* base;
        int row;

        Cell& operator[](int col) const {
            int index = cell_index(row, col);
            if (index < 0 || index >= MAX_GRID_CELLS) index = 0;
            return base[index];
        }
    };

    row_of_cells<int8_t> operator[](int row) { return { values, row }; }
    row_of_cells<const int8_t> operator[](int row) const { return { values, row }; }

    // Empty every cell, including storage outside the grid's current dimensions.
    void clear() {
        for (int8_t& value : values) value = TRANSPARENT;
    }
};

struct grid {
    int width = 0, height = 0;
    cells data;

    // No cells at all, so there is nothing to read off it.
    bool empty() const {
        return width <= 0 || height <= 0;
    }

    bool contains(int row, int col) const {
        return row >= 0 && row < height && col >= 0 && col < width;
    }

    // Shared walk for grid-shaped storage, in reading order.
    template <class Grid, class See>
    static void walk_colored_cells(const Grid& one_grid, int skip, See see) {
        for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
                const int color = one_grid.data[row][col];
                if (is_color(color) && color != skip) see(row, col, color);
            }
    }

    template <class See>
    void for_each_colored_cell(int skip, See see) const {
        walk_colored_cells(*this, skip, see);
    }

    void clear_cells() { data.clear(); }
    void clear() { data.clear(); width = height = 0; }

    bool same_size(const grid& other) const {
        return width == other.width && height == other.height;
    }

    void fill(int color) {
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                data[row][col] = (int8_t)color;
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

// The ARC demonstration example.
struct example {
    grid input, output;
};

}  // namespace arc
