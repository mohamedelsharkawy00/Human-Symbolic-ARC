#pragma once
// Puzzles in: the ARC JSON files, parsed into a puzzle and read by inference.
// Core, so it knows nothing about how a puzzle is shown -- the GUI builds its
// own pictures from the puzzle this hands back.
#include "12_environment.hpp"   // pulls in 7_infer.hpp: arc::infer(train) -> inference
#include "9_log.hpp" // the per-puzzle readout, written after infer()

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace arc {
using namespace std;

inline grid parse_grid(const string &str, size_t &pos) {
  grid grid;
  memset(grid.data, 0, sizeof(grid.data));
  size_t start = str.find('[', pos);
  if (start == string::npos)
    return grid;

  int depth = 0;
  size_t end = str.find(']', start);
  for (size_t i = start; i < str.size(); ++i) {
    if (str[i] == '[')
      depth++;
    else if (str[i] == ']') {
      depth--;
      if (depth == 0) {
        end = i;
        break;
      }
    }
  }
  if (end == string::npos)
    return grid;

  pos = end + 1;

  // Parse the grid in place over str[start..end] with an integer accumulator.
  // No substring copy, no per-cell std::string/stoi: zero heap allocation.
  bool in_number = false;
  int num_val = 0;
  bool num_neg = false;
  int r = 0;
  int c = 0;
  int max_w = 0;
  int d = 0;

  auto flush_number = [&]() {
    if (!in_number) return;
    if (r < MAX_GRID_DIM && c < MAX_GRID_DIM)
      grid.data[r][c] = num_neg ? -num_val : num_val;
    c++;
    num_val = 0;
    num_neg = false;
    in_number = false;
  };

  for (size_t i = start; i <= end; ++i) {
    char ch = str[i];
    if (ch == '[') {
      d++;
    } else if (ch == ']') {
      flush_number();
      if (d == 2) {
        if (c > max_w)
          max_w = c;
        c = 0;
        r++;
      }
      d--;
    } else if (ch == '-') {
      num_neg = true;
      in_number = true;
    } else if (isdigit((unsigned char)ch)) {
      num_val = num_val * 10 + (ch - '0');
      in_number = true;
    } else {
      flush_number();
    }
  }

  grid.height = r;
  grid.width = max_w;
  return grid;
}

// One task pulled out of a big ARC challenges JSON: the task id plus the raw
// JSON fragment for its object ({"train":[...],"test":[...]}). The fragment is
// exactly what load_task_from_content() expects, so the solver iterates these
// in-memory entries the same way it iterated per-task files — no on-disk split.
struct challenge_task {
  string id;
  string content;
};

// Reads a whole file into a string; empty string means missing or empty file.
inline string read_whole_file(const string &path) {
  ifstream f(path, ios::binary | ios::ate);
  if (!f) return {};
  streamoff sz = f.tellg();
  if (sz <= 0) return {};
  f.seekg(0);
  string content;
  content.resize((size_t)sz);
  f.read(content.data(), sz);
  return content;
}

// Walks a top-level JSON object ({"key": value, ...}) whose values open with
// `open` and close with the matching balanced `close`, calling
// fn(key, value_start, value_end) with inclusive positions into `content`.
// Empty keys are skipped (the outer object's own brace, not an entry).
template <typename Fn>
inline void for_each_top_level_entry(const string &content, char open, char close, Fn fn) {
  size_t pos = 0;
  while (pos < content.size()) {
    size_t q0 = content.find('"', pos);
    if (q0 == string::npos) break;
    size_t q1 = content.find('"', q0 + 1);
    if (q1 == string::npos) break;
    string key = content.substr(q0 + 1, q1 - q0 - 1);
    pos = q1 + 1;

    auto skip_ws = [&]() {
      while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                                       content[pos] == '\n' || content[pos] == '\r'))
        ++pos;
    };
    skip_ws();
    if (pos >= content.size() || content[pos] != ':') continue;
    ++pos;
    skip_ws();
    if (pos >= content.size() || content[pos] != open) continue;

    size_t start = pos;
    int depth = 0;
    size_t end = string::npos;
    for (size_t k = start; k < content.size(); ++k) {
      if (content[k] == open) ++depth;
      else if (content[k] == close) {
        if (--depth == 0) { end = k; break; }
      }
    }
    if (end == string::npos) break;
    pos = end + 1;

    if (!key.empty()) fn(key, start, end);
  }
}

// Reads a big ARC challenges JSON ({"task_id": {...}, ...}) and slices it into
// one challenge_task per top-level task, taking the file as-is. No files are
// written. Returns the tasks sorted by id (matching the old file-listing order).
inline vector<challenge_task> load_challenges_json(const string &json_path) {
  vector<challenge_task> tasks;
  string content = read_whole_file(json_path);

  for_each_top_level_entry(content, '{', '}', [&](const string &key, size_t start, size_t end) {
    tasks.push_back({key, content.substr(start, end - start + 1)});
  });

  std::sort(tasks.begin(), tasks.end(),
            [](const challenge_task &a, const challenge_task &b) { return a.id < b.id; });
  return tasks;
}

// Local synthetic puzzles use ordinary per-puzzle ARC JSON with inline answers.
inline vector<challenge_task> load_custom_puzzles(const string &directory) {
  vector<challenge_task> tasks;
  if (!filesystem::exists(directory)) return tasks;
  for (const auto &entry : filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
    tasks.push_back({entry.path().stem().string(), read_whole_file(entry.path().string())});
  }
  sort(tasks.begin(), tasks.end(),
       [](const challenge_task &a, const challenge_task &b) { return a.id < b.id; });
  return tasks;
}

// Reads separate test answers, keyed by puzzle id. Missing files yield no answers.
inline map<string, vector<grid>> load_solutions_json(const string &json_path) {
  map<string, vector<grid>> solutions;
  string content = read_whole_file(json_path);

  for_each_top_level_entry(content, '[', ']', [&](const string &key, size_t start, size_t end) {
    // Parse each grid inside the outer array. parse_grid finds the next balanced
    // [..] from its cursor; starting just inside the outer '[' yields grid0,
    // grid1, ... in order. Stop once the cursor passes the outer ']'.
    vector<grid> grids;
    size_t gpos = start + 1;
    while (gpos < end) {
      size_t before = gpos;
      grid g = parse_grid(content, gpos);
      if (gpos <= before || g.width == 0) break;  // no more grids
      grids.push_back(g);
    }
    solutions[key] = std::move(grids);
  });

  return solutions;
}

// Where a puzzle sits in the list it came from. Nothing reads it but the
// header of the inference readout, which says "puzzle 7 of 1000, TRN".
struct puzzle_label {
  int idx = -1;
  int count = 0;
  int dataset_flags = 0;
};

// EVERYTHING INFERENCE SAYS, read once and kept on the puzzle, start to
// finish in this one call: nothing a fact feeds is worked out anywhere else,
// so nothing can be worked out before the fact it rests on. Never at load: a
// puzzle nobody is looking at and nobody is solving does not need it, and
// stepping through a thousand puzzles should cost only the parse. Whoever
// needs the facts asks for them — the pane when it opens, the search before
// it starts. The readout file is written only when asked for, since only a
// person ever reads it.
inline void infer_into_puzzle(puzzle& task, const puzzle_label& label = {},
                              bool write_report = false) {
    task.inf = infer(task.train);
    const inference& inf = task.inf;
    if (write_report)
        log_inference(task.id, inf, label.idx, label.count, label.dataset_flags);
    task.rule = inf.size;                              // size - across_input_output
    if (inf.colors.background_color >= 0)              // color - across_inputs
        task.invariant_colors[inf.colors.background_color] = true;  // (bg only - see CAVEAT)
    // WHAT AN OP CANNOT DO ANYTHING WITH. An op whose backing fact is absent
    // can only no-op, so it leaves the generating pool now, once, rather than
    // being tried and discarded on every candidate. Read off what inference
    // actually said, so a verdict cannot drift from the fact behind it.
    task.dead_ops[(int)node_type::SizeRankColorMap] =
        (inf.colors.output_colors_by_size_count == 0);
    // No elected ground means this op is Input spelled longer.
    task.dead_ops[(int)node_type::InputWithoutBackground] =
        (inf.colors.background_color < 0);
    // The two shape->color ops SHARE one learned table and mean different things
    // by it: one repaints a body, the other fills a hole. Whichever detector
    // claimed it, the other op would be reading the wrong semantics, so an
    // unclaimed op is as dead as an empty table.
    const auto claimed = inf.shape_color_map.claimed_by;
    const bool learned = inf.shape_color_map.valid;
    task.dead_ops[(int)node_type::ShapeColorMap] =
        (!learned || claimed != shape_color_map_rule::body_recolor);
    task.dead_ops[(int)node_type::ShapeToInteriorColorMap] =
        (!learned || claimed != shape_color_map_rule::interior_fill);

    // THE READING, SAID AS PROGRAMS. The election could not separate the tied
    // openings, so each is its own whole program rather than a claim that one
    // of them is right.
    task.suggested_programs = programs_from(say_the_reading(inf));
    task.inferred = true;
}

// Parse a single task from an in-memory JSON string `content` (which must
// contain this task's "train"/"test" sections) into `out`, tagging it with
// `id`. Nothing is read off it yet: that is infer_into_puzzle, asked for
// separately. Returns true on success; on failure `out` is an empty puzzle.
inline bool load_task_from_content(const string &content, const string &id,
                                   puzzle &out,
                                   const vector<grid> *test_solutions = nullptr) {
  // puzzle has a raw node* best_tree; the compiler-generated dtor won't free
  // it, so we have to release it explicitly before overwriting.
  if (out.best_tree) environment::remove(out.best_tree);
  out = puzzle{};

  try {
    puzzle task;
    task.id = id;

    if (content.empty()) return false;

    size_t train_start = content.find("\"train\":");
    size_t test_start = content.find("\"test\":");
    if (train_start == string::npos || test_start == string::npos) return false;

    auto get_section_end = [&](size_t start) {
      int depth = 0;
      bool found_open = false;
      for (size_t k = start; k < content.size(); ++k) {
        if (content[k] == '[') {
          depth++;
          found_open = true;
        } else if (content[k] == ']') {
          depth--;
        }
        if (found_open && depth == 0) return k;
      }
      return content.size();
    };

    auto parse_examples = [&](size_t start, size_t end, vector<example> &out_vec,
                              bool is_train) {
      size_t csr = start;
      while (csr < end) {
        size_t in_pos = content.find("\"input\"", csr);
        if (in_pos == string::npos || in_pos > end) break;

        grid in_g = parse_grid(content, in_pos);
        if (in_g.width == 0 || in_g.height == 0) break;

        if (in_g.width > MAX_GRID_DIM || in_g.height > MAX_GRID_DIM) {
          out_vec.clear();
          return;
        }
        csr = in_pos;

        example ex;
        ex.input = in_g;

        size_t out_pos = content.find("\"output\"", csr);
        bool has_output = (out_pos != string::npos && out_pos < end);

        if (has_output) {
          grid out_g = parse_grid(content, out_pos);
          csr = out_pos;

          if (out_g.width > 0) {
            if (out_g.width > MAX_GRID_DIM || out_g.height > MAX_GRID_DIM) {
              out_vec.clear();
              return;
            }
            ex.output = out_g;
          } else {
            ex.output.width = 0;
          }
        } else {
          ex.output.width = 0;
        }

        // Test outputs are absent from the challenges JSON (the answers live in a
        // separate solutions file). If one was supplied for this test example,
        // attach it here so it behaves exactly like an inline output: scored,
        // distance-mapped, and able to light up "TEST SET SOLVED". One solution
        // grid per test input, in order (out_vec.size() == this example's index).
        if (!is_train && !has_output && test_solutions &&
            out_vec.size() < test_solutions->size()) {
          const grid &sol = (*test_solutions)[out_vec.size()];
          if (sol.width > 0 && sol.width <= MAX_GRID_DIM && sol.height <= MAX_GRID_DIM)
            ex.output = sol;
        }

        out_vec.push_back(ex);
      }
    };

    parse_examples(train_start, get_section_end(train_start), task.train, true);
    parse_examples(test_start, get_section_end(test_start), task.test, false);

    out = std::move(task);
    return true;
  } catch (...) {
  }
  return false;
}

// WHICH PUZZLES. The three competition sets share one layout on disk, the same
// on Kaggle and in this folder (kaggle/input/...), plus the local fixtures.
//   - training:   1000 tasks (has answer keys) — dev / mining.
//   - evaluation:  120 tasks (unseen, has answer keys) — honest local score.
//   - test:        240 tasks (the competition set, no solutions) — submission.
// A bitmask, because the GUI can check several at once.
enum dataset_flag { DS_TRAIN = 1, DS_EVAL = 2, DS_TEST = 4, DS_CUSTOM = 8 };
constexpr int DS_ALL = DS_TRAIN | DS_EVAL | DS_TEST | DS_CUSTOM;

// A dataset by the name a person types, or 0 for a name that is none of them.
inline int dataset_from_name(const string &name) {
  if (name == "train") return DS_TRAIN;
  if (name == "eval") return DS_EVAL;
  if (name == "test") return DS_TEST;
  if (name == "custom") return DS_CUSTOM;
  return 0;
}

// The tasks of every checked set, concatenated, and every answer key found
// beside them. Custom fixtures come first. Duplicate ids across sets are
// dropped (first checked set wins), so an index into `tasks` names one puzzle.
struct task_set {
  vector<challenge_task> tasks;
  map<string, vector<grid>> solutions;

  // Ground-truth test outputs for a task id, or nullptr when none are loaded
  // (no solutions file, or this id isn't in it).
  const vector<grid> *solutions_for(const string &id) const {
    auto it = solutions.find(id);
    return it != solutions.end() ? &it->second : nullptr;
  }
};

// THE FOLDER HOLDING kaggle/input, found by walking up from `start`. This repo
// mirrors Kaggle's layout, so one answer serves both: from out/x64-Release it
// is the repo, from /kaggle/working it is "/". Empty when there is none.
inline filesystem::path find_data_root(filesystem::path start) {
  for (auto p = start; ; p = p.parent_path()) {
    if (filesystem::exists(p / "kaggle" / "input")) return p;
    if (p == p.parent_path()) return {};
  }
}

// Every path is relative to the data root, which both mains make the current
// directory before asking.
inline task_set load_task_set(int flags) {
  const string kaggle_dir = "kaggle/input/competitions/arc-prize-2026-arc-agi-2/";
  struct set_info { int flag; const char *stem; };
  static const set_info sets[] = {
      {DS_TRAIN, "arc-agi_training_challenges.json"},
      {DS_EVAL,  "arc-agi_evaluation_challenges.json"},
      {DS_TEST,  "arc-agi_test_challenges.json"},
  };

  task_set out;
  set<string> seen;
  if (flags & DS_CUSTOM) {
    for (auto &task : load_custom_puzzles("tests/puzzles"))
      if (seen.insert(task.id).second) out.tasks.push_back(std::move(task));
  }
  for (const set_info &s : sets) {
    if (!(flags & s.flag)) continue;
    string challenges_json = kaggle_dir + s.stem;
    for (auto &t : load_challenges_json(challenges_json))
      if (seen.insert(t.id).second)
        out.tasks.push_back(std::move(t));

    // Ground-truth answers: sibling file with "_challenges" -> "_solutions".
    // Missing (e.g. the competition test set) just yields no answers.
    string solutions_json = challenges_json;
    if (size_t at = solutions_json.rfind("_challenges"); at != string::npos)
      solutions_json.replace(at, sizeof("_challenges") - 1, "_solutions");
    for (auto &kv : load_solutions_json(solutions_json))
      out.solutions.emplace(kv.first, std::move(kv.second));
  }
  std::fprintf(stderr, "dataset flags %d: %zu tasks, %zu solutions\n",
               flags, out.tasks.size(), out.solutions.size());
  return out;
}

}
