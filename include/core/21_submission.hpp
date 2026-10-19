#pragma once
// WHAT A SOLVED PUZZLE HANDS BACK: its line of submission.json, and a few lines
// saying how the reading went. Every puzzle is solved in a process of its own
// (see kaggle_main.cpp), so these are also what crosses from that process to
// the one that started it, as a small file.
//
// The submission, in the competition format:
//
//   { "taskid": [ {"attempt_1": <grid>, "attempt_2": <grid>}, ... ], ... }
//
// Rules (from the competition Evaluation page):
//   - every task_id present, even if unsolved (use a fallback grid)
//   - both attempt_1 and attempt_2 always present (we have 1 attempt -> dup)
//   - one list entry per test input, IN ORDER
// A grid is a list-of-lists of ints 0-9.

#include "13_environment.hpp"
#include "14_puzzle_loader.hpp"   // a task's test inputs, for the line of a task that left nothing
#include "15_solver_log.hpp"      // format_elapsed
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace arc {

// Run best_tree on a test input to get the predicted grid. On failure (no
// tree, or execution produced an empty grid) fall back to the input grid so
// the task still has a (wrong but valid) prediction.
inline grid predict_test(const puzzle &p, const example &test_ex) {
  grid out{};
  if (p.best_tree) {
    environment::score_tree_output(
        p.best_tree, test_ex, out, p, environment::count_nodes(p.best_tree));
  }
  if (out.width <= 0 || out.height <= 0) {
    return test_ex.input;  // fallback: echo input
  }
  return out;
}

inline std::string grid_to_json(const grid &g, const example &test_ex) {
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

// One task's line of submission.json, on one line: `answer` gives the grid
// for each test example.
template <typename Answer>
inline std::string submission_line(const puzzle &p, Answer &&answer) {
  std::string entry = "    \"" + p.id + "\": [";
  for (size_t i = 0; i < p.test.size(); ++i) {
    std::string g = grid_to_json(answer(p.test[i]), p.test[i]);
    entry += "{\"attempt_1\": " + g + ", \"attempt_2\": " + g + "}";
    if (i + 1 < p.test.size()) entry += ", ";
  }
  entry += "]";
  return entry;
}

// What the best program predicts. best_tree may be null (nothing found):
// every test input is then echoed back as its own answer.
inline std::string submission_entry(const puzzle &p) {
  return submission_line(p, [&](const example &ex) { return predict_test(p, ex); });
}

// THE RESULT OF ONE PUZZLE, as text. Line 1 is the program alone, so the file
// reads back in as a solution; the rest says how it went. Used for manual
// export and for every entry of a benchmark report.
inline std::string describe_result(const puzzle &p, double elapsed) {
  std::string train_status = p.is_solved ? "Solved" : "Unsolved";
  std::string test_status = p.is_test_solved ? "Solved"
                          : (p.is_solved ? "Failed" : "N/A");

  int node_count = environment::count_nodes(p.best_tree);

  float best_score = 0.0f;
  for (size_t i = 0; i < p.train.size(); ++i)
    best_score += p.best_score_or_worst(false, i);

  std::string sol_str = p.best_tree ? environment::stringify_tree_via_flat(p.best_tree) : "";

  std::string out;
  out += sol_str + "\n";
  out += "Puzzle:       " + p.id + "\n";
  out += "Train:        " + train_status + "\n";
  out += "Test:         " + test_status + "\n";
  out += "Elapsed:      " + format_elapsed(elapsed) + "\n";
  out += to_text("Nodes:        %d\n", node_count);
  out += to_text("Best Score:   %.4f\n", best_score);
  return out;
}

struct puzzle_result {
  std::string id;
  bool solved = false;
  bool test_solved = false;
  double elapsed = 0.0;
  std::string entry;  // its line of submission.json
  std::string text;   // describe_result

  static puzzle_result of(const puzzle &p, double elapsed) {
    puzzle_result r;
    r.id = p.id;
    r.solved = p.is_solved;
    r.test_solved = p.is_test_solved;
    r.elapsed = elapsed;
    r.entry = submission_entry(p);
    r.text = describe_result(p, elapsed);
    return r;
  }

  // A task whose run handed nothing back (it crashed, or was stopped). Still a
  // line in the submission -- every task id must be there -- read straight off
  // the task, with each test input echoed as its answer. Runs nothing, so the
  // GUI, which has no GPU context, can write it.
  static puzzle_result nothing_from(const challenge_task &task, double elapsed) {
    puzzle p;
    load_task_from_content(task.content, task.id, p);
    p.id = task.id;
    puzzle_result r;
    r.id = p.id;
    r.elapsed = elapsed;
    r.entry = submission_line(p, [](const example &ex) { return ex.input; });
    r.text = describe_result(p, elapsed);
    return r;
  }

  // Line by line, the text last because it runs over several.
  bool write(const std::string &path) const {
    std::ofstream f(path, std::ios::trunc | std::ios::binary);
    if (!f) return false;
    f << "id " << id << "\n"
      << "solved " << (int)solved << "\n"
      << "test_solved " << (int)test_solved << "\n"
      << "elapsed " << elapsed << "\n"
      << "entry " << entry << "\n"
      << "text\n" << text;
    return (bool)f;
  }

  // Nothing when the file is missing or cut short.
  static std::optional<puzzle_result> read(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    puzzle_result r;
    std::string key;
    int solved = 0, test_solved = 0;
    if (!(f >> key >> r.id) || key != "id") return std::nullopt;
    if (!(f >> key >> solved) || key != "solved") return std::nullopt;
    if (!(f >> key >> test_solved) || key != "test_solved") return std::nullopt;
    if (!(f >> key >> r.elapsed) || key != "elapsed") return std::nullopt;
    if (!(f >> key) || key != "entry") return std::nullopt;
    f.get();  // the space after "entry"
    if (!std::getline(f, r.entry)) return std::nullopt;
    if (!std::getline(f, key) || key != "text") return std::nullopt;
    r.text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    r.solved = solved != 0;
    r.test_solved = test_solved != 0;
    return r;
  }
};

// submission.json, a line per task.
class submission_writer {
public:
  void add(std::string entry) { m_entries.push_back(std::move(entry)); }

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
};

}  // namespace arc
