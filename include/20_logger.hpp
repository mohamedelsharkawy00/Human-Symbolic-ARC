#pragma once
#include "12_environment.hpp"
#include "14_global_app_state.hpp"
#include <fstream>
#include <filesystem>
#include "10_text_format.hpp"
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace arc {
using namespace std;

struct logger {
  global_app_state &app_state;
  const solver_state &solver;

  explicit logger(global_app_state &state, const solver_state &s) : app_state(state), solver(s) {
    flush_thread = jthread([this](stop_token st) {
      while (!st.stop_requested()) {
        this_thread::sleep_for(chrono::milliseconds(200));
        drain();
      }
      drain();
    });
  }

  ~logger() {
    flush_thread.request_stop();
    flush_thread.join();
  }

  void set_puzzle_id(const string &id) {
    lock_guard lock(mtx);
    m_puzzle_id = id;
  }

  void debug(const string &msg) {
    if (app_state.log_mode == logging_mode::none) return;
    lock_guard lock(mtx);
    buffer += format_elapsed(elapsed()) + " [DEBUG] " + msg + "\n";
  }

  // Always recorded: the timing line measures what logging costs, so the
  // logging toggle must not be able to change it.
  void timing(const string &msg) {
    lock_guard lock(mtx);
    buffer += format_elapsed(elapsed()) + " [TIME] " + msg + "\n";
  }

  void tree_record(const node *tree, float score, long long attempts) {
    if (app_state.log_mode == logging_mode::none) return;
    lock_guard lock(mtx);
    buffer += to_text("%s [TREE] %lld | %s | %s\n", format_elapsed(elapsed()).c_str(), attempts,
                      shortest(score).c_str(), environment::stringify_tree_via_flat(tree).c_str());
  }

  void start_benchmark_report(const string &path) {
    lock_guard lock(benchmark_mtx);
    benchmark_path = path;

    filesystem::path dir = filesystem::path(path).parent_path();
    if (!dir.empty())
      filesystem::create_directories(dir);

    ofstream f(benchmark_path, ios::trunc);
  }

  void benchmark_result(const string &result) {
    lock_guard lock(benchmark_mtx);
    if (benchmark_path.empty()) return;

    ofstream f(benchmark_path, ios::app);
    if (f.is_open()) {
      f << result << "\n";
      f.flush();
    }
  }

  void benchmark_record(const puzzle &p, long long attempts, float elapsed) {
    if (p.is_solved) app_state.benchmark_train_solved.fetch_add(1);
    if (p.is_test_solved) app_state.benchmark_test_solved.fetch_add(1);
    if (elapsed > 0.0f) {
      double cur = app_state.benchmark_total_aps.load();
      while (!app_state.benchmark_total_aps.compare_exchange_weak(cur, cur + (double)attempts / elapsed)) {}
    }
    app_state.benchmark_puzzles_run.fetch_add(1);
  }

  void benchmark_begin() {
    app_state.benchmark_puzzles_run.store(0);
    app_state.benchmark_train_solved.store(0);
    app_state.benchmark_test_solved.store(0);
    app_state.benchmark_total_aps.store(0.0);
    app_state.benchmark_start_time.store(chrono::steady_clock::now());
  }

  void finish_benchmark_report() {
    lock_guard lock(benchmark_mtx);
    if (benchmark_path.empty()) return;

    int puzzles_run = app_state.benchmark_puzzles_run.load();
    int train_solved = app_state.benchmark_train_solved.load();
    int test_solved = app_state.benchmark_test_solved.load();
    double total_aps = app_state.benchmark_total_aps.load();
    auto start_time = app_state.benchmark_start_time.load();

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
      ifstream in(benchmark_path);
      if (in.is_open())
        existing.assign(istreambuf_iterator<char>(in), istreambuf_iterator<char>());
    }

    ofstream f(benchmark_path, ios::trunc);
    if (f.is_open()) {
      f << summary;
      f << existing;
      f.flush();
    }

    benchmark_path.clear();
  }

  void clear() {
    lock_guard lock(drain_mtx);
    drain_locked();
  }

private:
  mutex mtx;
  mutex drain_mtx;
  mutex benchmark_mtx;
  string m_puzzle_id;
  string buffer;
  string benchmark_path;
  jthread flush_thread;

  double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - solver.start_time).count();
  }

  void drain() {
    lock_guard lock(drain_mtx);
    drain_locked();
  }

  void drain_locked() {
    string local;
    string pid;
    {
      lock_guard lock(mtx);
      local.swap(buffer);
      pid = m_puzzle_id;
    }

    if (local.empty()) return;

#ifdef _MSC_VER
    // Debug logs go to per-puzzle files (logs/<id>.txt). Skipped headless: on
    // Kaggle this writes 240+ files into /kaggle/working and blows the disk
    // quota. The buffer is still drained above (swapped out), just not written.
    // A line logged before a puzzle is loaded has no file to belong to, and in
    // this repo's whole history nothing has ever logged one.
    if (pid.empty()) return;

    static once_flag dir_flag;
    call_once(dir_flag, [] { filesystem::create_directory("logs"); });

    ofstream f("logs/" + pid + ".txt", ios::app);
    if (f.is_open()) f << local;
#endif
  }
};

}
