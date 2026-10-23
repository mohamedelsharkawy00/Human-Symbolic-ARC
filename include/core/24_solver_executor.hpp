#pragma once

namespace arc {

  inline bool solver::load(const challenge_task &task, const vector<grid> *test_solutions,
                           const puzzle_label &label) {
    m_log.clear();
    scoped_lock lock(m_progress.mtx);

    // Load UNDER the lock: this replaces m_puzzle, which a watcher on another
    // thread reads under this same lock. Loading outside it races that
    // reader, which could draw the new puzzle against a half-swapped one.
    m_puzzle_loaded = load_task_from_content(task.content, task.id, m_puzzle, test_solutions);
    m_label = label;
    if (m_puzzle_loaded) m_log.set_puzzle_id(m_puzzle.id);
    if (m_observer) m_observer->loaded(m_puzzle);

    m_progress.start_time = chrono::steady_clock::now();
    asked_inference_yet = false;
    changed();
    return m_puzzle_loaded;
  }

  inline void solver::read_facts() {
    // THE FACTS ARE READ WHEN SOMEONE WANTS THEM. Stepping through puzzles
    // costs only the parse; reading, or running a program on the puzzle, asks
    // for them, once. Under the lock, since a watcher reads the same puzzle.
    if (!m_puzzle_loaded || m_puzzle.inferred) return;
    scoped_lock lock(m_progress.mtx);
    infer_into_puzzle(m_puzzle, m_label, m_options.write_logs);
    changed();
  }

  inline void solver::infer() {
    if (!m_puzzle_loaded) return;
    read_facts();
    {
      scoped_lock lock(m_progress.mtx);
      if (!asked_inference_yet) {
        asked_inference_yet = true;
        if (!m_puzzle.is_solved) try_what_inference_said(m_puzzle);
        changed();
      }
    }
    // Said with the lock let go: a driver that pauses on news may take it.
    news({m_puzzle.is_solved, m_puzzle.is_test_solved});
  }

  // A WHOLE PROGRAM, MADE THE BEST AND CHECKED on every example. The puzzle is
  // solved if it draws every training output; its test verdict is read the
  // same way. Caller holds the lock.
  inline void solver::check_program(const string &program) {
    puzzle &task = m_puzzle;
    // Re-import (or import onto a puzzle that loaded with a solved tree) would
    // otherwise leak the whole prior node subtree.
    if (task.best_tree) environment::remove(task.best_tree);
    // A failed import must not read as solved, so the old verdict goes with the
    // old tree.
    task.is_solved = false;
    task.is_test_solved = false;
    task.best_tree = environment::parse(program);
    if (!task.best_tree) {
        log_debug("could not read this as a program: " + program);
        return;
    }
    // A program whose sentences do not each open with a source runs and draws
    // and is wrong; say so rather than let it pass.
    if (const char* wrong = sentence_with_no_source(program))
        log_debug(string(wrong) + ": " + program);
    // Whether it draws every output of one side, said pair by pair.
    const auto draws_every = [&](const vector<example> &pairs, const char *side) {
      bool every = true;
      for (size_t i = 0; i < pairs.size(); ++i) {
        const bool fits = environment::draw(task.best_tree, pairs[i], task) == pairs[i].output;
        log_debug(to_text("%s pair %zu %s", side, i, fits ? "fits" : "does not fit"));
        every = every && fits;
      }
      return every;
    };
    task.is_solved = draws_every(task.train, "train");
    task.is_test_solved = task.is_solved && !task.test.empty() && draws_every(task.test, "test");
  }

  inline void solver::apply_program(const string &program) {
    if (!m_puzzle_loaded || program.empty()) return;
    // Checking wants the facts, so it asks for them. Ops that learn a table
    // (ShapeColorMap and kin) have nothing to say about a puzzle nobody has
    // read, and would paint an empty grid for every example.
    read_facts();
    scoped_lock lock(m_progress.mtx);
    check_program(program);
    changed();
    log_debug("Manual solution imported.");
  }

  // WHAT THE LOOKING SPELLED, tried in order. Each suggestion is a whole
  // program, so this either finishes the puzzle or costs one evaluation per
  // suggestion.
  inline bool solver::try_what_inference_said(puzzle &p) {
    if (p.suggested_programs.empty()) return false;
    for (const string &program : p.suggested_programs) {
      check_program(program);
      if (p.is_solved) {
        log_debug("!!!! SOLVED BY WHAT INFERENCE SAID: " + program + " !!!!");
        return true;
      }
    }
    // NOTHING FITTED, so the reading's most likely program (its first) stays on
    // as the best: what inference came to is there to see. The last one tried
    // is no likelier than the rest.
    if (p.suggested_programs.size() > 1) check_program(p.suggested_programs.front());
    return false;
  }

}
