#pragma once
// How a grid is laid out inside gpu_buffer, the cell buffer the Vulkan shader
// draws from. The solver never sees it: it says what it painted, and the GUI
// copies that in here (30_solver_session.hpp).
#include "13_environment.hpp"
#include <algorithm>
#include <vector>

namespace arc {
using namespace std;

// WHERE A GRID'S CELLS SIT in gpu_buffer. Kept apart from the grid itself so
// that a core grid -- the puzzle's own input and output -- can be drawn without
// carrying a word about drawing. The GUI stores one of these per grid it paints.
struct placement {
  // Where its cells were put in the renderer's one big buffer.
  int offset = 0;
  // How far apart two rows sit in that buffer.
  int stride = 0;
};

// A grid THE DISPLAY ITSELF OWNS: font glyphs, color sheets, preview slots, and
// the best pictures. Data and placement travel together because nothing
// in core holds these. Core functions taking `const grid&` accept one unchanged;
// assigning one INTO a grid drops the placement, which is what core wants.
struct display_grid : grid {
  placement at;
};

// WHAT THE GUI KEEPS BESIDE AN EXAMPLE, never inside it. The puzzle's own input
// and output live in core and are only pointed at here, by where their cells
// were uploaded. `best` is the picture of what the reading's program drew, for
// a person to watch, so the display owns it outright.
struct example_views {
  placement input, output;
  display_grid best;
};

// One of the above per example, in the same order as puzzle::train and
// puzzle::test, so an index into either side names the same pair in both.
struct puzzle_views {
  vector<example_views> train, test;

  // Empty until a puzzle is loaded, so every reader asks rather than indexes
  // blind.
  example_views *of(bool is_test, size_t i) {
    vector<example_views> &side = is_test ? test : train;
    return i < side.size() ? &side[i] : nullptr;
  }
  const example_views *of(bool is_test, size_t i) const {
    const vector<example_views> &side = is_test ? test : train;
    return i < side.size() ? &side[i] : nullptr;
  }
};

// Helper to align buffer to 4 ints (16 bytes) to prevent GPU read offset issues
inline void pad_buffer(vector<int> &gpu_buffer) {
  while (gpu_buffer.size() % 4 != 0) {
    gpu_buffer.push_back(0);
  }
}

// Copies a grid's cells into the buffer and says where they landed.
inline placement upload_grid(const grid &g, vector<int> &gpu_buffer) {
  pad_buffer(gpu_buffer);
  placement at;
  at.stride = 32;
  at.offset = (int)gpu_buffer.size();
  int max_h = min(g.height, MAX_GRID_DIM);
  for (int r = 0; r < max_h; ++r) {
    for (int c = 0; c < MAX_GRID_DIM; ++c)
      gpu_buffer.push_back((c < g.width) ? g.data[r][c] : 0);
    gpu_buffer.push_back(0); // Pad to 31
    gpu_buffer.push_back(0); // Pad to 32
  }
  return at;
}

inline display_grid allocate_empty_grid(vector<int> &gpu_buffer) {
  pad_buffer(gpu_buffer);
  display_grid g;
  g.at.offset = (int)gpu_buffer.size();
  g.at.stride = 32;
  gpu_buffer.resize(gpu_buffer.size() + MAX_GRID_DIM * 32, 0);
  return g;
}

inline void update_display_buffer(display_grid &g, const grid &s, vector<int> &gb) {
  g.width = s.width;
  g.height = s.height;
  int copy_h = min(s.height, MAX_GRID_DIM);
  for (int r = 0; r < copy_h; ++r)
    for (int c = 0; c < MAX_GRID_DIM; ++c)
      g.data[r][c] = (int8_t)s.data[r][c];
  int max_h = min(g.height, MAX_GRID_DIM);
  for (int r = 0; r < max_h; ++r) {
    int base = g.at.offset + r * g.at.stride;
    for (int c = 0; c < g.at.stride; ++c)
      gb[base + c] = (c < g.width) ? g.data[r][c] : 0;
  }
}

}  // namespace arc
