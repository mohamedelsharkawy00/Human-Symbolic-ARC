#pragma once
// A READING WATCHED FROM ANOTHER PROCESS. Every reading the GUI shows runs in a
// process of its own (arc_headless --serve, see kaggle_main.cpp), and the GUI
// watches it: the process says what it paints as lines of text on stdout, and
// the watcher replays them into its own copy of the puzzle. The watcher talks
// back on stdin. Both halves live here so the two sides of the pipe cannot
// drift apart.
//
// What the process says, one message per line, first word saying which:
//   L                                   a puzzle was loaded
//   N train_solved test_solved          the reading is done
//   B solved test_solved | program      the best program and its verdicts
//   V hash examples, then per example:  what a program paints (a hover preview)
//     w h cells...  objects, then per object: x0 y0 x1 y1 w h cells...
//
// What it is told, on stdin:
//   infer | verbose 0|1 | import <program> | preview <hash> <program> | quit
#include "16_solver.hpp"
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace arc {
using namespace std;

// WHAT ONE PROGRAM PAINTS ON EVERY EXAMPLE, train then test: its output, and
// the object list its last op held, each object on a canvas of its own. What
// the GUI shows when a program in the footer is hovered.
struct preview_payload {
  struct bbox { int x0, y0, x1, y1, canvas_w, canvas_h; };
  vector<grid> accumulators;          // size = #examples
  vector<vector<grid>> object_lists;  // [example][object]
  vector<vector<bbox>> object_bboxes; // [example][object] world bbox + tile canvas dims
};

// Runs `target` on every example. The solver's process answers this; the
// GUI only ever reads the result.
inline preview_payload preview_of(const puzzle &task, const node *target) {
  preview_payload payload;
  const int total = (int)(task.train.size() + task.test.size());
  payload.accumulators.resize(total);
  payload.object_lists.resize(total);
  payload.object_bboxes.resize(total);
  int slot = 0;
  auto run_section = [&](const vector<example> &vec) {
    for (const auto &ex : vec) {
      // The object list the hovered op held, beside what it drew. On the
      // heap: a list as long as a grid has cells is too big for the stack.
      static thread_local auto held = std::make_unique<object_list>();
      const grid out = environment::draw(target, ex, task, held.get());
      payload.accumulators[slot] = out;
      const object_list &ol = *held;
      payload.object_lists[slot].reserve(ol.count);
      payload.object_bboxes[slot].reserve(ol.count);
      // Each object on a canvas as big as what was drawn -- which a turn
      // (NormalizeOrientation) leaves on its side, so the background turns with
      // the objects -- else as the example, and grown to hold every object's
      // painted extent: Fractal hands back output-sized masks, and a Tile's
      // copies can hang off the edge.
      int canvas_w = out.width > 0 ? out.width : max(ex.input.width, ex.output.width);
      int canvas_h = out.height > 0 ? out.height : max(ex.input.height, ex.output.height);
      for (int k = 0; k < ol.count; ++k) {
        const auto &o = ol.objects[k];
        canvas_w = max(canvas_w, o.x1 + 1);
        canvas_h = max(canvas_h, o.y1 + 1);
      }
      canvas_w = min(canvas_w, MAX_GRID_DIM);
      canvas_h = min(canvas_h, MAX_GRID_DIM);
      for (int k = 0; k < ol.count; ++k) {
        const auto &o = ol.objects[k];
        grid g;
        g.width = canvas_w;
        g.height = canvas_h;
        int mh = min((int)o.height, MAX_GRID_DIM);
        int mw = min((int)o.width, MAX_GRID_DIM);
        for (int r = 0; r < mh; ++r) {
          int dst_r = o.y0 + r;
          if (dst_r < 0 || dst_r >= canvas_h) continue;
          for (int c = 0; c < mw; ++c) {
            int dst_c = o.x0 + c;
            if (dst_c < 0 || dst_c >= canvas_w) continue;
            int8_t v = o.mask[r][c];
            if (v != TRANSPARENT)
              g.data[dst_r][dst_c] = v;
          }
        }
        payload.object_lists[slot].push_back(std::move(g));
        payload.object_bboxes[slot].push_back({
          (int)o.x0, (int)o.y0, (int)o.x1, (int)o.y1, canvas_w, canvas_h});
      }
      ++slot;
    }
  };
  run_section(task.train);
  run_section(task.test);
  return payload;
}

// A grid on the line: "w h cells...", and back.
inline void write_grid(string &out, const grid &g) {
  out += to_text(" %d %d", g.width, g.height);
  for (int r = 0; r < g.height; ++r)
    for (int c = 0; c < g.width; ++c)
      out += to_text(" %d", (int)g.data[r][c]);
}
inline void read_grid(istream &in, grid &g) {
  in >> g.width >> g.height;
  g.width = min(max(g.width, 0), MAX_GRID_DIM);
  g.height = min(max(g.height, 0), MAX_GRID_DIM);
  for (int r = 0; r < g.height; ++r)
    for (int c = 0; c < g.width; ++c) {
      int color = TRANSPARENT;
      in >> color;
      g.data[r][c] = (int8_t)color;
    }
}

// THE WATCHED SIDE: a solver_observer that writes the lines. The best program
// goes out when it changes, at most every `interval`.
class search_stream_writer : public solver_observer {
public:
  explicit search_stream_writer(FILE *out, chrono::milliseconds interval = chrono::milliseconds(50))
      : m_out(out), m_interval(interval) {}

  void loaded(const puzzle &) override {
    m_best.clear();
    line("L");
  }

  void news(const search_news &n) override {
    line(to_text("N %d %d", (int)n.train_solved, (int)n.test_solved));
  }

  void previewed(uint64_t hash, const preview_payload &p) {
    string text = to_text("V %llu %zu", (unsigned long long)hash, p.accumulators.size());
    for (size_t i = 0; i < p.accumulators.size(); ++i) {
      write_grid(text, p.accumulators[i]);
      text += to_text(" %zu", p.object_lists[i].size());
      for (size_t k = 0; k < p.object_lists[i].size(); ++k) {
        const auto &b = p.object_bboxes[i][k];
        text += to_text(" %d %d %d %d", b.x0, b.y0, b.x1, b.y1);
        write_grid(text, p.object_lists[i][k]);
      }
    }
    line(text);
    fflush(m_out);
  }

  // Called from the loop driving the solver: sends what changed since last
  // time. `now` sends it whatever the interval.
  void tick(const solver &reader, bool now = false) {
    const auto t = chrono::steady_clock::now();
    if (!now && t - m_last < m_interval) {
      fflush(m_out);
      return;
    }
    m_last = t;

    const puzzle &p = reader.get_puzzle();
    string best = to_text("B %d %d", (int)p.is_solved, (int)p.is_test_solved);
    best += " | " + (p.best_tree ? environment::stringify_tree_via_flat(p.best_tree) : string());
    if (best != m_best) line(m_best = best);

    fflush(m_out);
  }

private:
  void line(const string &text) {
    fputs(text.c_str(), m_out);
    fputc('\n', m_out);
  }

  FILE *m_out;
  chrono::milliseconds m_interval;
  chrono::steady_clock::time_point m_last{};
  string m_best;
};

// THE WATCHING SIDE: what a watcher hears besides what the solver told its
// own observer.
struct stream_receiver : solver_observer {
  // The answer to "preview <hash> <program>".
  virtual void previewed(uint64_t /*hash*/, preview_payload && /*p*/) {}
};

// One line, replayed. `p` is the watcher's own copy of the puzzle, parsed from
// the same task; `progress` holds its lock, which guards it the way it guards
// a solver's own.
inline void apply_stream_line(const string &text, puzzle &p, search_progress &progress,
                              stream_receiver &watcher) {
  istringstream in(text);
  string kind;
  if (!(in >> kind)) return;

  // These arrive with the lock NOT held, as a solver's own news does.
  if (kind == "N") {
    int train = 0, test = 0;
    in >> train >> test;
    watcher.news({train != 0, test != 0});
    return;
  }
  if (kind == "V") {
    unsigned long long hash = 0;
    size_t count = 0;
    in >> hash >> count;
    preview_payload payload;
    count = min(count, (size_t)64);
    payload.accumulators.resize(count);
    payload.object_lists.resize(count);
    payload.object_bboxes.resize(count);
    for (size_t i = 0; i < count; ++i) {
      read_grid(in, payload.accumulators[i]);
      size_t objects = 0;
      in >> objects;
      objects = min(objects, (size_t)MAX_OBJECTS_IN_A_LIST);
      for (size_t k = 0; k < objects; ++k) {
        preview_payload::bbox b{};
        in >> b.x0 >> b.y0 >> b.x1 >> b.y1;
        grid g;
        read_grid(in, g);
        b.canvas_w = g.width;
        b.canvas_h = g.height;
        payload.object_bboxes[i].push_back(b);
        payload.object_lists[i].push_back(g);
      }
    }
    watcher.previewed(hash, std::move(payload));
    return;
  }

  scoped_lock lock(progress.mtx);
  if (kind == "L") {
    watcher.loaded(p);
  } else if (kind == "B") {
    int solved = 0, test_solved = 0;
    in >> solved >> test_solved;
    p.is_solved = solved != 0;
    p.is_test_solved = test_solved != 0;
    string bar, program;
    in >> bar;  // the "|" before the program, which may be empty
    getline(in >> ws, program);
    environment::remove(p.best_tree);
    p.best_tree = program.empty() ? nullptr : environment::parse(program);
  }
  watcher.changed();
}

}  // namespace arc
