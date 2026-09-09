// Start here.
// This file depends on no other file in the project,
// and every other file depends on it.
// It sets the words the rest of the project is written in.
#pragma once
#include <cstdint>

namespace arc {

// The ARC cell and the ten colors it can hold.
enum class arc_color : int8_t {
	black = 0, blue = 1, red = 2, green = 3,
	yellow = 4, gray = 5, magenta = 6, orange = 7,
	cyan = 8, maroon = 9, count = 10
};
constexpr int COLOR_COUNT = (int)arc_color::count;

// The ARC grid, its max size is 30 x 30.
constexpr int MAX_GRID_DIM = 30;
struct grid {
	int width = 0, height = 0;
	int data[MAX_GRID_DIM][MAX_GRID_DIM]{};
};

// The ARC demonstration example.
struct example {
	grid input, output;
};

// Human reading order.
// Plural S = whole train set; no S = one pair.
// enum class direction {
//	across_input_output,
//	across_inputs,
//	across_outputs,
//	across_inputs_outputs,
// };

}
