#pragma once
// How a grid is laid out inside the gpu_buffer the CUDA solver consumes.
// Not rendering code: shared by the real GUI (gui.hpp) and the headless
// Kaggle build (gui_stub.hpp), so it lives here once.
#include "environment.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

namespace arc {
using namespace std;

// Helper to align buffer to 4 ints (16 bytes) to prevent GPU read offset issues
inline void pad_buffer(vector<int> &gpu_buffer) {
  while (gpu_buffer.size() % 4 != 0) {
    gpu_buffer.push_back(0);
  }
}

inline void upload_grid(grid &g, vector<int> &gpu_buffer) {
  pad_buffer(gpu_buffer);
  g.stride = 32;
  g.offset = (int)gpu_buffer.size();
  int max_h = min(g.height, MAX_GRID_DIM);
  for (int r = 0; r < max_h; ++r) {
    for (int c = 0; c < MAX_GRID_DIM; ++c)
      gpu_buffer.push_back((c < g.width) ? g.data[r][c] : 0);
    gpu_buffer.push_back(0); // Pad to 31
    gpu_buffer.push_back(0); // Pad to 32
  }
}

inline grid allocate_empty_grid(vector<int> &gpu_buffer) {
  pad_buffer(gpu_buffer);
  grid g(0, 0, (int)gpu_buffer.size(), 32);
  gpu_buffer.resize(gpu_buffer.size() + MAX_GRID_DIM * 32, 0);
  return g;
}

inline void update_display_buffer(grid &g, const grid &s, vector<int> &gb) {
  g.width = s.width;
  g.height = s.height;
  int copy_h = min(s.height, MAX_GRID_DIM);
  if (copy_h > 0) memcpy(g.data, s.data, copy_h * sizeof(g.data[0]));
  int max_h = min(g.height, MAX_GRID_DIM);
  for (int r = 0; r < max_h; ++r) {
    int base = g.offset + r * g.stride;
    for (int c = 0; c < g.stride; ++c)
      gb[base + c] = (c < g.width) ? g.data[r][c] : 0;
  }
}

}  // namespace arc
