#pragma once
// Puzzles in: the ARC JSON files, parsed into a puzzle and read by inference.
// Core, so it knows nothing about how a puzzle is shown -- the GUI builds its
// own pictures from the puzzle this hands back.
#include "13_environment.hpp"   // pulls in 7_infer.hpp: arc::infer(train) -> inference
#include "10_log.hpp" // the per-puzzle readout, written after infer()

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
  grid.clear_cells();
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
      grid.data[r][c] = (int8_t)(num_neg ? -num_val : num_val);
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
// needs the facts asks for them — the pane when it opens, the reading when
// it is asked for. The readout file is written only when asked for, since only a
// person ever reads it. The reading itself, infer, is the slow part (seconds
// where it reads its passes); take_the_reading is the rest, so a caller that
// holds a lock others wait on can read outside it and take inside it.
inline void take_the_reading(puzzle& task, inference read, const puzzle_label& label = {},
                             bool write_report = false);
inline void infer_into_puzzle(puzzle& task, const puzzle_label& label = {},
                              bool write_report = false) {
    take_the_reading(task, infer(task.train), label, write_report);
}
inline void take_the_reading(puzzle& task, inference read, const puzzle_label& label, bool write_report) {
    task.inf = std::move(read);
    const inference& inf = task.inf;
    if (write_report)
        log_inference(task.id, inf, label.idx, label.count, label.dataset_flags);
    task.rule = inf.size;                              // size - across_input_output
    if (inf.ground_election.input_elected() >= 0)              // color - across_inputs
        task.invariant_colors[inf.ground_election.input_elected()] = true;  // (bg only - see CAVEAT)

    // THE READING, SAID AS PROGRAMS. The election could not separate the tied
    // openings, so each is its own whole program rather than a claim that one
    // of them is right.
    task.suggested_programs = programs_from(say_the_reading(inf));
    // A sentence that opens with no source carves what the sentence before it
    // left instead of the input. It runs, it draws, and it is wrong, so it is
    // dropped here rather than run.
    erase_if(task.suggested_programs, [](const string& one) {
        return sentence_with_no_source(one) != nullptr; });
    task.inferred = true;
}

// Bounded JSON values: field order and whitespace carry no meaning. In
// particular, an output belongs to its example object, not the next input.
namespace puzzle_json {
inline void space(const string& text, size_t& pos, size_t end) {
  while (pos < end && isspace((unsigned char)text[pos])) ++pos;
}

inline size_t value_end(const string& text, size_t pos, size_t end) {
  if (pos >= end) return string::npos;
  if (text[pos] == '"') {
    for (++pos; pos < end; ++pos) {
      if (text[pos] == '\\') ++pos;
      else if (text[pos] == '"') return pos + 1;
    }
    return string::npos;
  }
  if (text[pos] == '{' || text[pos] == '[') {
    const char close = text[pos] == '{' ? '}' : ']';
    for (++pos; pos < end;) {
      if (text[pos] == close) return pos + 1;
      if (text[pos] == '"' || text[pos] == '{' || text[pos] == '[') {
        pos = value_end(text, pos, end);
        if (pos == string::npos) return pos;
      } else {
        if (text[pos] == '}' || text[pos] == ']') return string::npos;
        ++pos;
      }
    }
    return string::npos;
  }
  const size_t start = pos;
  while (pos < end && !isspace((unsigned char)text[pos]) &&
         text[pos] != ',' && text[pos] != '}' && text[pos] != ']') ++pos;
  return pos == start ? string::npos : pos;
}

template <typename Fn>
bool members(const string& text, size_t begin, size_t end, Fn fn) {
  if (begin >= end || text[begin] != '{' || text[end - 1] != '}') return false;
  size_t pos = begin + 1;
  space(text, pos, end);
  if (pos == end - 1) return true;
  while (pos < end - 1) {
    if (text[pos] != '"') return false;
    const size_t key_end = value_end(text, pos, end);
    if (key_end == string::npos) return false;
    const string key = text.substr(pos + 1, key_end - pos - 2);
    pos = key_end;
    space(text, pos, end);
    if (pos >= end || text[pos++] != ':') return false;
    space(text, pos, end);
    const size_t last = value_end(text, pos, end);
    if (last == string::npos || !fn(key, pos, last)) return false;
    pos = last;
    space(text, pos, end);
    if (pos == end - 1) return true;
    if (pos >= end || text[pos++] != ',') return false;
    space(text, pos, end);
  }
  return false;
}
} // namespace puzzle_json

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

    auto parse_examples = [&](size_t start, size_t end, vector<example> &out_vec,
                              bool is_train) -> bool {
      if (content[start] != '[' || content[end - 1] != ']') return false;
      size_t csr = start + 1;
      puzzle_json::space(content, csr, end);
      while (csr < end - 1) {
        const size_t example_end = puzzle_json::value_end(content, csr, end);
        if (example_end == string::npos) return false;
        example ex;
        bool has_input = false, has_output = false;
        const bool parsed = puzzle_json::members(content, csr, example_end,
            [&](const string& key, size_t first, size_t last) {
          if (key != "input" && key != "output") return true;
          bool& seen = key == "input" ? has_input : has_output;
          if (seen || content[first] != '[') return false;
          grid drawn = parse_grid(content, first);
          if (first != last || drawn.width <= 0 || drawn.height <= 0 ||
              drawn.width > MAX_GRID_DIM || drawn.height > MAX_GRID_DIM) return false;
          (key == "input" ? ex.input : ex.output) = drawn;
          seen = true;
          return true;
        });
        if (!parsed || !has_input || (is_train && !has_output)) return false;

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
        csr = example_end;
        puzzle_json::space(content, csr, end);
        if (csr == end - 1) break;
        if (csr >= end || content[csr++] != ',') return false;
        puzzle_json::space(content, csr, end);
        if (csr == end - 1) return false;
      }
      return !out_vec.empty();
    };

    size_t begin = 0, end = content.size();
    puzzle_json::space(content, begin, end);
    while (end > begin && isspace((unsigned char)content[end - 1])) --end;
    bool has_train = false, has_test = false;
    if (!puzzle_json::members(content, begin, end,
        [&](const string& key, size_t first, size_t last) {
      if (key != "train" && key != "test") return true;
      const bool train = key == "train";
      bool& seen = train ? has_train : has_test;
      if (seen) return false;
      seen = true;
      return parse_examples(first, last, train ? task.train : task.test, train);
    }) || !has_train || !has_test) return false;

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
