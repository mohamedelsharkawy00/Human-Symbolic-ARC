#pragma once
// Both builds write submission.json: Kaggle scores it; Windows writes a local
// copy to kaggle/working/ (mirrors the Kaggle layout) for parity / local scoring.

// Kaggle submission.json writer (headless build only).
//
// The benchmark report (logger.hpp) only logs the PROGRAM found + stats, not
// the predicted grids -- so producing a submission is separate work: for each
// finished puzzle we execute its best_tree on every test input to generate the
// predicted output grid, and accumulate them in the competition format:
//
//   { "taskid": [ {"attempt_1": <grid>, "attempt_2": <grid>}, ... ], ... }
//
// Rules (from the competition Evaluation page):
//   - every task_id present, even if unsolved (use a fallback grid)
//   - both attempt_1 and attempt_2 always present (we have 1 attempt -> dup)
//   - one list entry per test input, IN ORDER
// A grid is a list-of-lists of ints 0-9.

#include "environment.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace arc {

class submission_writer {
public:
  // Capture predictions for one finished puzzle. Safe to call once per puzzle
  // as the benchmark loop completes it (solved or timed out). best_tree may be
  // null (nothing found) -> falls back to copying the test input.
  void capture(const puzzle &p) {
    std::string entry = "    \"" + p.id + "\": [";
    for (size_t i = 0; i < p.test.size(); ++i) {
      grid pred = predict(p, p.test[i]);
      std::string g = grid_to_json(pred, p.test[i]);
      entry += "{\"attempt_1\": " + g + ", \"attempt_2\": " + g + "}";
      if (i + 1 < p.test.size()) entry += ", ";
    }
    entry += "]";
    m_entries.push_back(std::move(entry));
  }

  // Write the accumulated submission.json. Returns false on file error.
  bool write(const std::string &path) const {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;
    f << "{\n";
    for (size_t i = 0; i < m_entries.size(); ++i) {
      f << m_entries[i];
      if (i + 1 < m_entries.size()) f << ",";
      f << "\n";
    }
    f << "}\n";
    return true;
  }

  size_t count() const { return m_entries.size(); }

private:
  std::vector<std::string> m_entries;

  // Run best_tree on a test input to get the predicted grid. On failure (no
  // tree, or execution produced an empty grid) fall back to the input grid so
  // the task still has a (wrong but valid) prediction.
  static grid predict(const puzzle &p, const example &test_ex) {
    grid out{};
    if (p.best_tree) {
      environment::score_tree_output(
          p.best_tree, test_ex, out, p, p.best_node_count);
    }
    if (out.width <= 0 || out.height <= 0) {
      return test_ex.input;  // fallback: echo input
    }
    return out;
  }

  static std::string grid_to_json(const grid &g, const example &test_ex) {
    int w = g.width, h = g.height;
    // Guard against malformed dims: never emit an empty grid (invalid format).
    if (w <= 0 || h <= 0) { w = test_ex.input.width; h = test_ex.input.height; }
    if (w <= 0 || h <= 0) { w = 1; h = 1; }
    const grid &src = (g.width > 0 && g.height > 0) ? g : test_ex.input;

    std::ostringstream o;
    o << "[";
    for (int r = 0; r < h; ++r) {
      o << "[";
      for (int c = 0; c < w; ++c) {
        int v = (r < src.height && c < src.width) ? src.data[r][c] : 0;
        // ARC cells must be ints 0-9. The solver uses sentinels (e.g. -1 for
        // transparent/uncomputed) that can leak into a predicted grid; any
        // out-of-range value -> 0, else the scorer rejects the submission
        // ("Submission Scoring Error"). THIS bug caused the v15 scoring error.
        if (!is_color(v)) v = 0;
        o << v;
        if (c + 1 < w) o << ", ";
      }
      o << "]";
      if (r + 1 < h) o << ", ";
    }
    o << "]";
    return o.str();
  }
};

}  // namespace arc
