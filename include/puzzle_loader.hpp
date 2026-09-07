#pragma once
#include "environment.hpp"   // pulls in inference/4_infer.hpp: arc::infer(train) -> inference
#include "inference/6_log.hpp" // the per-puzzle readout, written after infer()
#include "global_app_state.hpp"
#ifdef _MSC_VER
#include "gui.hpp"        // full GUI/Vulkan rendering
#else
#include "gui_stub.hpp"   // headless: GPU-buffer helpers only, no rendering
#endif
#include "logger.hpp"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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

inline void compute_distance_maps(example &ex) {
  int h = ex.output.height, w = ex.output.width;
  static constexpr int drs[] = {-1, 1, 0, 0};
  static constexpr int dcs[] = {0, 0, -1, 1};
  for (int c = 0; c < (int)arc_color::count; ++c) {
    // BFS from all cells matching color c — O(W*H) instead of O(W²*H²)
    // Queue is static thread_local + clear() (not a local): it grows to the
    // max grid size once and never reallocates again. Do not localize this.
    static thread_local vector<pair<int,int>> q;
    q.clear();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        if (ex.output.data[y][x] == c) {
          ex.distance_maps[c][y][x] = 0;
          q.push_back({y, x});
        } else {
          ex.distance_maps[c][y][x] = DISTANCE_MAX;
        }
      }
    size_t head = 0;
    while (head < q.size()) {
      auto [cy, cx] = q[head++];
      int nd = ex.distance_maps[c][cy][cx] + 1;
      for (int i = 0; i < 4; ++i) {
        int ny = cy + drs[i], nx = cx + dcs[i];
        if (ny >= 0 && ny < h && nx >= 0 && nx < w && nd < ex.distance_maps[c][ny][nx]) {
          ex.distance_maps[c][ny][nx] = nd;
          q.push_back({ny, nx});
        }
      }
    }
  }
}

inline void apply_solution(puzzle &task,
                           vector<int> &gpu_buffer, const string &sol_str, logger &log) {
  // Re-import (or import onto a puzzle that loaded with a solved tree) would
  // otherwise leak the whole prior node subtree.
  if (task.best_tree) environment::remove(task.best_tree);
  // A failed import must not read as solved, so the old verdict goes with the old tree.
  task.is_solved = false;
  task.is_test_solved = false;
  task.best_tree = environment::parse(sol_str);
  if (task.best_tree) {
    task.best_node_count = environment::count_nodes(task.best_tree);

    string tree_str = environment::stringify_tree_via_flat(task.best_tree);
    // Straight from the kernel that scored it, so the printed sum IS the score.
    auto log_score = [&](int ei, const gpu_score_terms &t, const char *phase) {
      log.debug(t.say(tree_str, ei, phase));
    };

    // Evaluate and update GPU buffers
    grid temp;
    bool all_solved = true;
    for (int i = 0; i < (int)task.train.size(); ++i) {
      auto &ex = task.train[i];
      gpu_score_terms terms;
      auto [s, solved] = environment::score_tree_output(
          task.best_tree, ex, temp, task, task.best_node_count, &terms);

      log.debug(to_text("Example score: %s (Solved: %s)", shortest(s).c_str(), solved ? "YES" : "NO"));
      log_score(i, terms, "import_train");

      ex.best_score = s;
      update_display_buffer(ex.best, temp, gpu_buffer);
      if (!solved)
        all_solved = false;
    }

    if (all_solved) {
      task.is_solved = true;
      bool test_all_solved = true;
      for (int i = 0; i < (int)task.test.size(); ++i) {
        auto &ex = task.test[i];
        gpu_score_terms terms;
        auto [s, solved] = environment::score_tree_output(
            task.best_tree, ex, temp, task, task.best_node_count, &terms);
        log_score(i, terms, "import_test");
        ex.best_score = s;
        update_display_buffer(ex.best, temp, gpu_buffer);
        ex.output.visible = true;
        if (!solved) test_all_solved = false;
      }
      if (!task.test.empty() && test_all_solved) {
        task.is_test_solved = true;
      }
    }
  }
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

// Reads an ARC solutions JSON ({"task_id": [grid, grid, ...], ...}) into a map
// from task id to its list of ground-truth test-output grids, in order. These
// answers are absent from the challenges file; load_single_puzzle attaches them
// to the matching test examples so the test row can be scored / "solved".
// Returns an empty map if the file is missing (caller treats that as no answers).
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

inline void init_gui_buffers(vector<int> &gpu_buffer, map<uint32_t, grid> &chars, grid &bg_grid) {
  // Sized for: font atlas + 800x800 bg + 32 preview grids + 32*64 object
  // preview slots + headroom for the largest puzzle's training data.
  // Pre-reserve so init never reallocates (no contiguous-block hunts during
  // startup; pointer stays put once GPU upload begins).
  gpu_buffer.reserve(32 * 1024 * 1024);
  load_font_atlas(gpu_buffer, chars);
  bg_grid.width = 800;
  bg_grid.height = 800;
  bg_grid.stride = 800;
  bg_grid.offset = (int)gpu_buffer.size();
  gpu_buffer.resize(gpu_buffer.size() + 800 * 800, 11);
}

// EVERYTHING INFERENCE SAYS, read once and kept on the puzzle. Held out of the
// load itself because it is the expensive half: a puzzle nobody is looking at
// and nobody is solving does not need it, and stepping through a thousand
// puzzles should cost only the parse. Whoever needs the facts asks for them —
// the pane when it opens, the search before it starts.
inline void infer_into_puzzle(puzzle& task, int idx = -1, int count = 0,
                              int dataset_flags = 0) {
    task.inf = infer(task.train);
    const inference& inf = task.inf;
    log_inference(task.id, inf, idx, count, dataset_flags);
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
// `id`. Shared by the per-file loader (load_single_puzzle) and the big
// challenges-file loader (load_task_from_challenges). Returns true on success.
inline bool load_task_from_content(const string &content, const string &id,
                                   puzzle &out, vector<int> &gpu_buffer,
                                   size_t base_gpu_buffer_size,
                                   const vector<grid> *test_solutions = nullptr,
                                   int idx = -1, int count = 0, int dataset_flags = 0,
                                   bool with_inference = true) {
  // Truncate the GPU buffer back to the base size to drop any old puzzle data
  gpu_buffer.resize(base_gpu_buffer_size);
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

    auto parse_examples = [&](size_t start, size_t end,
                              vector<example> &out_vec, bool is_train) {
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

        upload_grid(in_g, gpu_buffer);
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
            upload_grid(out_g, gpu_buffer);
            ex.output = out_g;

            ex.output.visible = is_train;
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
        bool has_solution = false;
        if (!is_train && !has_output && test_solutions &&
            out_vec.size() < test_solutions->size()) {
          // Copy before uploading: upload_grid rewrites stride/offset, and the
          // solutions map is shared/persistent across loads — mutating it would
          // leave a stale GPU offset behind on the next load of this task.
          grid sol = (*test_solutions)[out_vec.size()];
          if (sol.width > 0 && sol.width <= MAX_GRID_DIM && sol.height <= MAX_GRID_DIM) {
            upload_grid(sol, gpu_buffer);
            ex.output = sol;
            ex.output.visible = false;  // hidden like a normal test target
            has_solution = true;
          }
        }

        ex.current = allocate_empty_grid(gpu_buffer);
        ex.best = allocate_empty_grid(gpu_buffer);

        if (has_output || has_solution) compute_distance_maps(ex);

        out_vec.push_back(ex);
      }
    };

    parse_examples(train_start, get_section_end(train_start), task.train, true);
    parse_examples(test_start, get_section_end(test_start), task.test, false);

    // WHAT LOOKING AT THE PUZZLE SAID. infer() reads every invariant off the
    // examples and hands back one `inference` object, and the puzzle KEEPS it.
    // Nothing is copied out of it: what the search reads and what the kernel is
    // given are both read off the object itself, so a new fact needs no line
    // adding anywhere to reach them. It used to be copied field by field, which
    // meant a fact only arrived if somebody remembered its copy line, and a
    // forgotten one arrived as a zero without saying so.
    if (with_inference) infer_into_puzzle(task, idx, count, dataset_flags);
    // NOT yet migrated — still computed by the engine directly:
    environment::detect_target_objects(task);
    out = std::move(task);
    return true;
  } catch (...) {
  }
  return false;
}

// Per-task loader: parses one challenge_task sliced from a big challenges JSON
// (via load_challenges_json). Thin wrapper over load_task_from_content; no file
// IO — the JSON was already read once when the whole task set was sliced.
inline bool load_single_puzzle(const challenge_task &task, puzzle &out,
                               vector<int> &gpu_buffer,
                               size_t base_gpu_buffer_size,
                               const vector<grid> *test_solutions = nullptr,
                               int idx = -1, int count = 0, int dataset_flags = 0,
                               bool with_inference = true) {
  return load_task_from_content(task.content, task.id, out, gpu_buffer,
                                base_gpu_buffer_size, test_solutions,
                                idx, count, dataset_flags, with_inference);
}

}
