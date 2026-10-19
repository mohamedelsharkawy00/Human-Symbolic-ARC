// Start here.
// This file depends on no other file in the project,
// and every other file depends on it.
// It sets the core concepts the rest of the project is written in.
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

// The number of colors, just to avoid magic numbers in the code.
constexpr int COLOR_COUNT = (int)arc_color::count;

// A special value for an empty cell that is used internally in the solver
constexpr int TRANSPARENT = -1;

// Check if a value is a valid color, not anything else.
ARC_HOST_DEVICE constexpr bool is_color(int value) {
    return value >= 0 && value < COLOR_COUNT;
}

// The ARC grid, its max size is 30 x 30.
constexpr int MAX_GRID_DIM = 30;
constexpr int MAX_GRID_CELLS = MAX_GRID_DIM * MAX_GRID_DIM;

// A cell's place in the largest grid as one number, counted row by row.
ARC_HOST_DEVICE constexpr int cell_index(int row, int col) { return row * MAX_GRID_DIM + col; }
ARC_HOST_DEVICE constexpr int cell_row(int index) { return index / MAX_GRID_DIM; }
ARC_HOST_DEVICE constexpr int cell_col(int index) { return index % MAX_GRID_DIM; }

struct cells {
    // Each cell takes 4 bits, so two share a byte, and a cell's number says
    // which byte and which half.
    static constexpr int CELL_BITS = 4;
    static constexpr int CELL_MASK = (1 << CELL_BITS) - 1;
    static constexpr int BYTE_BITS = 8;
    static constexpr int CELLS_PER_BYTE = BYTE_BITS / CELL_BITS;

    // The GPU updates the store a 4-byte word at a time. The aligned, padded
    // store keeps those updates safe at the end of the storage.
    static constexpr int WORD_BYTES = sizeof(uint32_t);
    static constexpr int PACKED_BYTES = (MAX_GRID_CELLS + CELLS_PER_BYTE - 1) / CELLS_PER_BYTE;
    static constexpr int BYTES = (PACKED_BYTES + WORD_BYTES - 1) / WORD_BYTES * WORD_BYTES;
    alignas(WORD_BYTES) uint8_t bytes[BYTES];

    // A half-byte cannot be returned by reference, so cell access uses proxies.
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
            const uint8_t encoded = cells::enc(value);
            *byte = (uint8_t)((*byte & ~(CELL_MASK << shift)) | (encoded << shift));
        }

        // Copy the cell value, not the proxy's byte pointer and bit position.
        ARC_HOST_DEVICE void operator=(const cell_ref& other) { *this = (int8_t)other; }
        ARC_HOST_DEVICE void operator=(const const_cell_ref& other) { *this = (int8_t)other; }
    };

    // One row proxy handles both read-only and writable cell access.
    template <class Byte, class Cell>
    struct row_of_cells {
        Byte* base;
        int row;

        ARC_HOST_DEVICE Cell operator[](int col) const {
            int index = cell_index(row, col);
            if (index < 0 || index >= MAX_GRID_CELLS) index = 0;
            return { base + index / CELLS_PER_BYTE, shift_in_byte(index) };
        }
    };

    ARC_HOST_DEVICE row_of_cells<uint8_t, cell_ref> operator[](int row) { return { bytes, row }; }
    ARC_HOST_DEVICE row_of_cells<const uint8_t, const_cell_ref> operator[](int row) const {
        return { bytes, row };
    }

    // Store value + 1 so zero bits mean TRANSPARENT and black remains distinct.
    ARC_HOST_DEVICE static constexpr uint8_t enc(int8_t value) {
        return (uint8_t)(((int)value + 1) & CELL_MASK);
    }
    ARC_HOST_DEVICE static constexpr int8_t dec(uint8_t byte, int shift) {
        return (int8_t)(((byte >> shift) & CELL_MASK) - 1);
    }
    // Two side-by-side cells, the first in the low bits.
    ARC_HOST_DEVICE static constexpr uint8_t pair_byte(int8_t first, int8_t second) {
        return (uint8_t)(enc(first) | (enc(second) << CELL_BITS));
    }
    ARC_HOST_DEVICE static constexpr uint8_t empty_byte() { return pair_byte(TRANSPARENT, TRANSPARENT); }

    // Where a cell's bits start, within its byte and within its word.
    ARC_HOST_DEVICE static constexpr int shift_in_byte(int index) { return index % CELLS_PER_BYTE * CELL_BITS; }
    ARC_HOST_DEVICE static constexpr int shift_in_word(int index) {
        return shift_in_byte(index) + index / CELLS_PER_BYTE % WORD_BYTES * BYTE_BITS;
    }

    // The first of the cells a byte holds.
    ARC_HOST_DEVICE static constexpr int first_cell_of_byte(int byte) { return byte * CELLS_PER_BYTE; }

    // Clear every cell, including storage outside the grid's current dimensions.
    ARC_HOST_DEVICE void clear() {
        for (int i = 0; i < (int)sizeof(bytes); ++i) bytes[i] = empty_byte();
    }
};

struct grid {
    int width = 0, height = 0;
    // Zeroed, so a new grid starts transparent.
    cells data{};

    // No cells at all, so there is nothing to read off it.
    bool empty() const {
        return width <= 0 || height <= 0;
    }

    bool contains(int row, int col) const {
        return row >= 0 && row < height && col >= 0 && col < width;
    }

    // Shared walk for grid-shaped storage, in reading order.
    template <class Grid, class See>
    ARC_HOST_DEVICE static void walk_colored_cells(const Grid& one_grid, int skip, See see) {
        for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
                const int color = one_grid.data[row][col];
                if (is_color(color) && color != skip) see(row, col, color);
            }
    }

    template <class See>
    ARC_HOST_DEVICE void for_each_colored_cell(int skip, See see) const {
        walk_colored_cells(*this, skip, see);
    }

    ARC_HOST_DEVICE void clear_cells() { data.clear(); }
    ARC_HOST_DEVICE void clear() { data.clear(); width = height = 0; }

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
