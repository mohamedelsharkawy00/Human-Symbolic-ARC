#pragma once
// Headless (non-MSVC / Kaggle) replacement for gui.hpp. The grid<->gpu_buffer
// layout helpers the CUDA solver consumes live in grid_buffer.hpp, shared with
// the real GUI. Only load_font_atlas (font rendering) is genuinely
// display-only and becomes a no-op here.
// On MSVC the real gui.hpp is used instead; this file is non-MSVC only.
#include "grid_buffer.hpp"
#include <map>

namespace arc {
using namespace std;

// Font atlas is genuinely rendering-only; no-op on headless builds.
inline void load_font_atlas(vector<int> &, map<uint32_t, grid> &) {}

}  // namespace arc
