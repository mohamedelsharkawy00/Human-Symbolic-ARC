#pragma once
// Headless (non-MSVC / Kaggle) replacement for gui.hpp. The grid<->gpu_buffer
// layout helpers live in grid_buffer.hpp, shared with the real GUI: solver_beam
// calls them either way, so here they fill a buffer nothing ever draws. Only
// load_font_atlas is dropped outright, becoming a no-op.
// On MSVC the real gui.hpp is used instead; this file is non-MSVC only.
#include "grid_buffer.hpp"
#include <map>

namespace arc {
using namespace std;

// Font atlas is genuinely rendering-only; no-op on headless builds.
inline void load_font_atlas(vector<int> &, map<uint32_t, display_grid> &) {}

}  // namespace arc
