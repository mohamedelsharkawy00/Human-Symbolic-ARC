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

    // A puzzle just parsed has no best and no scores yet; this says so to
    // whoever is watching.
    if (m_puzzle_loaded) {
      clear_views(m_puzzle, false);
      clear_views(m_puzzle, true);
    }
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

  // A WHOLE PROGRAM, MADE THE BEST AND SCORED on every example. The puzzle is
  // solved if it solves the train set; its test verdict is read the same way.
  // Caller holds the lock.
  inline void solver::score_program(const string &program) {
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
    // and is wrong; say so rather than let it score.
    if (const char* wrong = sentence_with_no_source(program))
        log_debug(string(wrong) + ": " + program);
    const int nodes = environment::count_nodes(task.best_tree);
    // Longer than one batch slot never reaches the GPU: every example would come
    // back empty, which reads as a program that draws nothing rather than one
    // that was never run. Say which it is, and keep the old verdict cleared.
    if (too_long_to_run(nodes)) {
        log_debug(to_text("%d ops is past the %d a program may be, so this one was "
                          "not run: %s", nodes, flat_program::MAX_OPS, program.c_str()));
        return;
    }

    string tree_str = environment::stringify_tree_via_flat(task.best_tree);
    // Straight from the kernel that scored it, so the printed sum IS the score.
    auto log_score = [&](int ei, const gpu_score_terms &t, const char *phase) {
      log_debug(t.say(tree_str, ei, phase));
    };

    grid temp;
    bool all_solved = true;
    for (int i = 0; i < (int)task.train.size(); ++i) {
      auto &ex = task.train[i];
      gpu_score_terms terms;
      auto [s, solved] = environment::score_tree_output(
          task.best_tree, ex, temp, task, nodes, &terms);

      log_debug(to_text("Example score: %s (Solved: %s)", shortest(s).c_str(), solved ? "YES" : "NO"));
      log_score(i, terms, "import_train");

      task.best_score(false, i) = s;
      show(false, i, &temp);
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
            task.best_tree, ex, temp, task, nodes, &terms);
        log_score(i, terms, "import_test");
        task.best_score(true, i) = s;
        show(true, i, &temp);
        if (!solved) test_all_solved = false;
      }
      if (!task.test.empty() && test_all_solved) {
        task.is_test_solved = true;
      }
    }
  }

  inline void solver::apply_program(const string &program) {
    if (!m_puzzle_loaded || program.empty()) return;
    // Scoring wants the facts, so it asks for them. Ops that learn a table
    // (ShapeColorMap and kin) have nothing to say about a puzzle nobody has
    // read, and would paint an empty grid for every example.
    read_facts();
    scoped_lock lock(m_progress.mtx);
    score_program(program);
    changed();
    log_debug("Manual solution imported.");
  }

  // WHAT THE LOOKING SPELLED, tried in order. Each suggestion is a whole
  // program, so this either finishes the puzzle or costs one evaluation per
  // suggestion.
  inline bool solver::try_what_inference_said(puzzle &p) {
    if (p.suggested_programs.empty()) return false;
    for (const string &program : p.suggested_programs) {
      score_program(program);
      if (p.is_solved) {
        log_debug("!!!! SOLVED BY WHAT INFERENCE SAID: " + program + " !!!!");
        return true;
      }
    }
    // NOTHING FITTED, so the reading's most likely program (its first) stays on
    // as the best, drawn on every pair: what inference came to is there to see.
    // The last one tried is no likelier than the rest.
    if (p.suggested_programs.size() > 1) score_program(p.suggested_programs.front());
    return false;
  }

  // Blank every picture on one side. The scores go with them: an example with
  // nothing shown has nothing scored either.
  inline void solver::clear_views(puzzle &p, bool is_test) {
    for (size_t i = 0; i < (is_test ? p.test : p.train).size(); ++i) {
      p.best_score(is_test, i) = SCORE_MIN - 1.0f;
      show(is_test, i, nullptr);
    }
  }

}
