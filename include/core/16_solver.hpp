#pragma once
// THE SOLVER. Give it a puzzle, ask it to read, read the best program off the
// puzzle. That is the whole contract. It owns no thread, knows no window,
// reads no settings file and plays no sound: a driver does those. The served
// process (kaggle_main.cpp) drives it and watches it through solver_observer.
// Nothing in this folder may include anything from gui/.

#include "13_environment.hpp"
#include "14_puzzle_loader.hpp"
#include "15_solver_log.hpp"

#include "11_text_format.hpp"
#include <string>
#include <vector>

namespace arc {
using namespace std;

// How the solver is set up. Fixed for the solver's life.
struct solver_options {
  // logs/<id>.txt and logs/inference/<id>.txt. For a person reading along;
  // off on Kaggle, where nobody reads them and the disk quota would not hold them.
  bool write_logs = false;
};

// What a reading came to, for a watcher to mark: a sound, a pause.
struct search_news {
  bool train_solved = false;
  bool test_solved = false;
};

// WHAT THE SOLVER SHOWS SOMEBODY WATCHING. The solver never draws: it says what
// it did. Every call arrives on the thread driving the solver, most with
// progress().mtx held, so a watcher drawing under that lock sees a whole update
// or none of it.
struct solver_observer {
  virtual ~solver_observer() = default;
  // A puzzle was just parsed, or failed to parse and is empty.
  virtual void loaded(const puzzle & /*p*/) {}
  // Something visible changed.
  virtual void changed() {}
  // The reading is done. Arrives with the lock NOT held.
  virtual void news(const search_news & /*n*/) {}
};

class solver {
public:
  explicit solver(solver_options options = {}, solver_observer *observer = nullptr)
      : m_options(options), m_observer(observer), m_log(m_progress, options.write_logs) {}

  solver(const solver &) = delete;
  solver &operator=(const solver &) = delete;

  // The watcher, or none. Set before the first load.
  void set_observer(solver_observer *observer) { m_observer = observer; }

  // Debug lines in the log, when it writes at all.
  void set_verbose(bool on) { m_log.verbose = on; }
  void log_debug(const string &msg) { m_log.debug(msg); }

  // Takes no index: one puzzle is loaded at a time, so this is that one.
  // Another thread reads it under the solver's lock.
  const puzzle &get_puzzle() const {
    static puzzle dummy;
    return m_puzzle_loaded ? m_puzzle : dummy;
  }

  // Parse `task` and start over on it: every trace of the last puzzle goes.
  // Only parses: nothing is inferred until infer() asks. `test_solutions` are
  // its answers when known, so the test row can be scored. `label` only heads
  // the inference readout.
  bool load(const challenge_task &task, const vector<grid> *test_solutions,
            const puzzle_label &label = {});

  // Inference, once and whole, then each program it spelled, once. Says what
  // it came to every time it is asked, so a watcher always hears the end.
  void infer();
  // Only the first half: the facts, without trying what they spelled. What
  // running any program on the puzzle needs first.
  void read_facts();

  // A whole program in the solver's own spelling, made the best and checked
  // on every pair. The import path, and how what inference spelled is tried.
  void apply_program(const string &program);

private:
  // apply_program without the lock, for a caller already holding it.
  void check_program(const string &program);
  bool try_what_inference_said(puzzle &p);
  // Gated on the ATTEMPT, not on `inferred`: the pane can infer a puzzle long
  // before anything asks what its reading spelled.
  bool asked_inference_yet = false;

  // What a watcher is told, each a no-op with nobody watching.
  void changed() {
    if (m_observer) m_observer->changed();
  }
  void news(const search_news &n) {
    if (m_observer) m_observer->news(n);
  }

  solver_options m_options;
  solver_observer *m_observer = nullptr;
  search_progress m_progress;
  logger m_log;
  puzzle m_puzzle;
  bool m_puzzle_loaded = false;
  puzzle_label m_label;
};

}

#include "19_solver_executor.hpp"
