#pragma once
// THE BENCHMARK: every puzzle of the checked datasets, from the first, each
// searched in an arc_headless process of its own, like every search the GUI
// shows (30_solver_session.hpp). A solver that has worked through one puzzle
// carries its memory into the next, and runs after the first went measurably
// worse than the same puzzle run first in a clean process -- so no two puzzles
// of a benchmark share one.
//
// The window shows each puzzle's search live as it runs, and its own controls
// do nothing until the run ends. Results go to benchmarks/benchmark_<time>.txt,
// summary on top, and the predictions to kaggle/working/submission.json,
// rewritten after every puzzle so a stopped run still leaves what it finished.
#include "21_submission.hpp"
#include "24_global_app_state.hpp"
#include "30_solver_session.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>

namespace arc {
using namespace std;

class benchmark_run {
public:
  benchmark_run(global_app_state &app, solver_session &session) : m_app(app), m_session(session) {}
  ~benchmark_run() { stop(); }

  // Called every frame: the BENCHMARK switch starts a run and stops one. A run
  // that reaches its end turns the switch off itself.
  void follow_switch() {
    // A run that reached its end is joined first, so switching straight back
    // on starts a new one rather than finding the old thread still there.
    if (m_finished.exchange(false)) m_thread.join();
    const bool wanted = m_app.is_benchmarking;
    if (wanted && !m_thread.joinable()) {
      m_thread = jthread([this] { run(); });
    } else if (!wanted && m_thread.joinable()) {
      stop();
    }
  }

private:
  void stop() {
    m_cancelled = true;
    if (m_thread.joinable()) m_thread.join();
    m_cancelled = false;
    m_finished = false;
  }

  void run() {
    m_session.begin_benchmark();
    const int count = m_session.get_puzzle_count();
    const double timeout = m_app.benchmark_timeout_seconds;
    const int limit = m_app.benchmark_puzzle_limit;

    begin_report();
    submission_writer submission;
    for (int i = 0; i < count && !m_cancelled; ++i) {
      if (limit > 0 && m_app.benchmark_puzzles_run.load() >= limit) break;
      const challenge_task &task = m_session.task(i);

      const auto started = chrono::steady_clock::now();
      optional<puzzle_result> handed_back = m_session.benchmark_puzzle(i, timeout, m_cancelled);
      // A stopped run records nothing of the puzzle it was stopped in.
      if (m_cancelled) break;
      const double took = chrono::duration<double>(chrono::steady_clock::now() - started).count();
      // A child that died still leaves its task a line in the submission.
      const puzzle_result result =
          handed_back ? *handed_back : puzzle_result::nothing_from(task, took);
      if (!handed_back)
        m_session.log_debug("benchmark: " + task.id + " handed back nothing, inputs echoed");

      record(result);
      submission.add(result.entry);
      filesystem::create_directories("kaggle/working");
      submission.write("kaggle/working/submission.json");
    }
    finish_report();
    m_session.end_benchmark();
    m_app.is_benchmarking = false;
    m_finished = true;
  }

  // --- The report: benchmarks/benchmark_<time>.txt ---
  void begin_report() {
    m_app.benchmark_puzzles_run.store(0);
    m_app.benchmark_train_solved.store(0);
    m_app.benchmark_test_solved.store(0);
    m_app.benchmark_total_aps.store(0.0);
    m_app.benchmark_start_time.store(chrono::steady_clock::now());

    time_t now_t = chrono::system_clock::to_time_t(chrono::system_clock::now());
    tm local_tm{};
    localtime_s(&local_tm, &now_t);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d_%I-%M-%S_%p", &local_tm);
    m_report_path = to_text("benchmarks/benchmark_%s.txt", ts);
    filesystem::create_directories("benchmarks");
    ofstream f(m_report_path, ios::trunc);
  }

  void record(const puzzle_result &r) {
    if (r.solved) m_app.benchmark_train_solved.fetch_add(1);
    if (r.test_solved) m_app.benchmark_test_solved.fetch_add(1);
    if (r.elapsed > 0.0) {
      double cur = m_app.benchmark_total_aps.load();
      while (!m_app.benchmark_total_aps.compare_exchange_weak(cur, cur + (double)r.attempts / r.elapsed)) {}
    }
    m_app.benchmark_puzzles_run.fetch_add(1);

    ofstream f(m_report_path, ios::app);
    if (f.is_open()) f << r.text << "\n";
  }

  // Puts the summary at the top of the report, above the per-puzzle lines.
  void finish_report() {
    int puzzles_run = m_app.benchmark_puzzles_run.load();
    int train_solved = m_app.benchmark_train_solved.load();
    int test_solved = m_app.benchmark_test_solved.load();
    double total_aps = m_app.benchmark_total_aps.load();
    auto start_time = m_app.benchmark_start_time.load();

    double wall_time = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
    double avg_aps = puzzles_run > 0 ? total_aps / puzzles_run : 0.0;
    double train_pct = puzzles_run > 0 ? 100.0 * train_solved / puzzles_run : 0.0;
    double test_pct = puzzles_run > 0 ? 100.0 * test_solved / puzzles_run : 0.0;

    string summary;
    summary += "========== BENCHMARK SUMMARY ==========\n";
    summary += to_text("Puzzles Run:    %d\n", puzzles_run);
    summary += to_text("Train Solved:   %d / %d (%.1f%%)\n", train_solved, puzzles_run, train_pct);
    summary += to_text("Test Solved:    %d / %d (%.1f%%)\n", test_solved, puzzles_run, test_pct);
    summary += to_text("Bench Time:     %s\n", format_elapsed(wall_time).c_str());
    summary += to_text("Avg APS:        %.2f\n", avg_aps);
    summary += "========================================\n\n";

    string existing;
    {
      ifstream in(m_report_path);
      if (in.is_open())
        existing.assign(istreambuf_iterator<char>(in), istreambuf_iterator<char>());
    }
    ofstream f(m_report_path, ios::trunc);
    if (f.is_open()) f << summary << existing;
  }

  global_app_state &m_app;
  solver_session &m_session;
  string m_report_path;
  atomic<bool> m_cancelled{false};
  atomic<bool> m_finished{false};
  jthread m_thread;
};

}  // namespace arc
