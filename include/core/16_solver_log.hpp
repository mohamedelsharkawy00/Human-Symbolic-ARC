#pragma once
// What the solver shares while it runs: the lock and clock another thread may
// read by, and the per-puzzle log file a person may read afterwards. Core, so
// neither knows whether anybody is watching. Whoever builds the solver says
// whether the log is written at all.
#include "14_environment.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include "12_text_format.hpp"
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

// The mutex guards the solver's puzzle for a reader on another thread; a
// solver driven from one thread never waits on it.
struct search_progress {
	mutable mutex mtx;
	// When the puzzle was loaded. A driver that pauses moves it forward by the
	// pause, so time since it is time spent reading.
	chrono::steady_clock::time_point start_time{chrono::steady_clock::now()};
};

struct logger {
  const search_progress &progress;
  // logs/<id>.txt at all. Off for a run nobody reads the logs of: on Kaggle
  // they would be 240+ files in /kaggle/working, over its disk quota.
  const bool write_files;
  // Debug lines at all. The GUI's logging switch.
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
    ofstream("logs/" + pid + ".txt", ios::trunc | ios::binary);
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

    ofstream f("logs/" + pid + ".txt", ios::app | ios::binary);
    if (f.is_open()) f << local;
  }
};

}
