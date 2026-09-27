#pragma once
// What the search says about itself as it runs: the live counters another
// thread may read, and the per-puzzle log file a person may read afterwards.
// Core, so neither knows whether anybody is watching. Whoever builds the
// solver says whether the log is written at all.
#include "13_environment.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include "11_text_format.hpp"
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

namespace arc {
using namespace std;

// Shared time formatter: seconds → "HH:MM:SS.ssss"
inline string format_elapsed(double t) {
	int h = (int)t / 3600;
	int m = ((int)t % 3600) / 60;
	double s = t - h * 3600 - m * 60;
	return to_text("%02d:%02d:%07.4f", h, m, s);
}

// WHAT THE SEARCH HAS DONE SO FAR, readable from any thread while it runs.
// The mutex guards the solver's puzzle and current_tree for a reader on
// another thread; a solver driven from one thread never waits on it.
struct search_progress {
	mutable mutex mtx;
	// The program most recently tried, for someone watching to read.
	node *current_tree = nullptr;
	atomic<long long> attempts{0};
	// Nanoseconds the search has spent waiting on the GPU. Read by the render
	// thread: attempts alone cannot tell a running kernel from a stall.
	// gpu_wait_since is the start of the wait in progress (0 = not waiting).
	// Without it a long wait posts its whole duration to the single frame it
	// ends on, which reads as more than 100% busy.
	atomic<long long> gpu_wait_ns{0};
	atomic<long long> gpu_wait_since{0};
	atomic<long long> gpu_wait_count{0};

	// Total GPU wait including the part not yet finished.
	long long gpu_wait_now() const {
		long long since = gpu_wait_since.load();
		long long live = since ? chrono::duration_cast<chrono::nanoseconds>(
			chrono::steady_clock::now().time_since_epoch()).count() - since : 0;
		return gpu_wait_ns.load() + (live > 0 ? live : 0);
	}
	atomic<int> current_max_depth{1};
	atomic<int> current_attempt_depth{0};
	atomic<int> current_attempt_nodes{0};
	// When the puzzle was loaded. A driver that pauses moves it forward by the
	// pause, so time since it is time spent searching.
	chrono::steady_clock::time_point start_time{chrono::steady_clock::now()};
};

struct logger {
  const search_progress &progress;
  // logs/<id>.txt at all. Off for a run nobody reads the logs of: on Kaggle
  // they would be 240+ files in /kaggle/working, over its disk quota.
  const bool write_files;
  // Debug and tree lines too, not just timing. The GUI's logging switch.
  atomic<bool> verbose{false};

  logger(const search_progress &p, bool write) : progress(p), write_files(write) {
    if (!write_files) return;
    flush_thread = jthread([this](stop_token st) {
      while (!st.stop_requested()) {
        this_thread::sleep_for(chrono::milliseconds(200));
        drain();
      }
      drain();
    });
  }

  ~logger() {
    if (flush_thread.joinable()) {
      flush_thread.request_stop();
      flush_thread.join();
    }
  }

  void set_puzzle_id(const string &id) {
    lock_guard lock(mtx);
    m_puzzle_id = id;
  }

  void debug(const string &msg) {
    if (verbose) add("DEBUG", msg);
  }

  // Always recorded: the timing line measures what logging costs, so the
  // logging toggle must not be able to change it.
  void timing(const string &msg) { add("TIME", msg); }

  void tree_record(const node *tree, float score, long long attempts) {
    if (!write_files || !verbose) return;
    add("TREE", to_text("%lld | %s | %s", attempts, shortest(score).c_str(),
                        environment::stringify_tree_via_flat(tree).c_str()));
  }

  // Empties the current puzzle's file, for a log that starts with this run.
  void start_fresh_file() {
    if (!write_files) return;
    string pid;
    {
      lock_guard lock(mtx);
      pid = m_puzzle_id;
    }
    if (pid.empty()) return;
    filesystem::create_directory("logs");
    ofstream("logs/" + pid + ".txt", ios::trunc);
  }

  void clear() {
    lock_guard lock(drain_mtx);
    drain_locked();
  }

private:
  mutex mtx;
  mutex drain_mtx;
  string m_puzzle_id;
  string buffer;
  jthread flush_thread;

  // One line: when, what kind, what.
  void add(const char *tag, const string &msg) {
    if (!write_files) return;
    lock_guard lock(mtx);
    buffer += format_elapsed(elapsed()) + " [" + tag + "] " + msg + "\n";
  }

  double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - progress.start_time).count();
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

    // A line logged before a puzzle is loaded has no file to belong to, and in
    // this repo's whole history nothing has ever logged one.
    if (pid.empty()) return;

    static once_flag dir_flag;
    call_once(dir_flag, [] { filesystem::create_directory("logs"); });

    ofstream f("logs/" + pid + ".txt", ios::app);
    if (f.is_open()) f << local;
  }
};

}
