#pragma once
// THE GUI'S SIDE OF THE SOLVER. The solver never runs in this process: every
// puzzle on screen has an arc_headless --serve child (see headless_main.cpp), and
// ending it is ending that process, which takes everything it held with it. A
// switch to another puzzle, a reset or a dataset change kills it; the next one
// starts clean.
//
// What a person at a window expects is kept here: the puzzle list, pause,
// settings, sounds, and pictures of what the reading drew. The child streams
// what it says (26_search_stream.hpp); this replays it into its own copy of
// the puzzle, and the frame draws that as if the solver were local. Hovering a
// program asks the child what it paints, since only the child can run one.
//
// A child is started once the window has stayed on a puzzle for a moment, so
// stepping through puzzles costs nothing, and it waits until SPACE or the pane
// asks it to read. A benchmark (35_benchmark.hpp) runs its puzzles through the
// same child, one process each.
#include "21_puzzle_loader.hpp"
#include "22_solver_log.hpp"
#include "25_submission.hpp"   // describe_result, puzzle_result
#include "26_search_stream.hpp"
#include "27_grid_buffer.hpp"
#include "28_global_app_state.hpp"
#include "31_sound_manager.hpp"
#include "32_gui.hpp"   // load_font_atlas
#include "33_os_utilities.hpp"   // search_process

#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

inline void init_gui_buffers(vector<int> &gpu_buffer, map<uint32_t, display_grid> &chars,
                             display_grid &bg_grid) {
  // Sized for: font atlas + 800x800 bg + 32 preview grids + 32*64 object
  // preview slots + headroom for the largest puzzle's training data.
  // Pre-reserve so init never reallocates (no contiguous-block hunts during
  // startup; pointer stays put once GPU upload begins).
  gpu_buffer.reserve(32 * 1024 * 1024);
  load_font_atlas(gpu_buffer, chars);
  bg_grid.width = 800;
  bg_grid.height = 800;
  bg_grid.at.stride = 800;
  bg_grid.at.offset = (int)gpu_buffer.size();
  gpu_buffer.resize(gpu_buffer.size() + 800 * 800, 11);
}

class solver_session : private stream_receiver {
public:
  explicit solver_session(global_app_state &app)
      : m_app(app), m_view(m_search), m_log(m_search, true) {}
  ~solver_session() { stop_thread(); }

  solver_session(const solver_session &) = delete;
  solver_session &operator=(const solver_session &) = delete;

  // Lays out the display buffer, opens the datasets and puzzle last saved,
  // and loads that puzzle. Before start_thread.
  inline void init(vector<int> &gpu_buffer, map<uint32_t, display_grid> &chars,
                   display_grid &bg_grid) {
    m_gpu_buffer = &gpu_buffer;

    init_gui_buffers(gpu_buffer, chars, bg_grid);
    // Pool of preview slots, one per possible example row. Measured against
    // the ARC training+evaluation sets: max train=10, max test=4, max total
    // per puzzle=12. 32 leaves generous headroom for unseen puzzle sets.
    m_view.preview_grids.clear();
    m_view.preview_grids.reserve(32);
    for (int i = 0; i < 32; ++i)
      m_view.preview_grids.push_back(allocate_empty_grid(gpu_buffer));

    // Object-list preview slots: [example][object], as many objects as a
    // list holds (MAX_OBJECTS_IN_A_LIST).
    m_view.preview_object_grids.clear();
    m_view.preview_object_grids.resize(32);
    for (int i = 0; i < 32; ++i) {
      m_view.preview_object_grids[i].reserve(MAX_OBJECTS_IN_A_LIST);
      for (int j = 0; j < MAX_OBJECTS_IN_A_LIST; ++j)
        m_view.preview_object_grids[i].push_back(allocate_empty_grid(gpu_buffer));
    }
    // Everything above stays put; a puzzle's cells go after it and are
    // dropped back to here when the next one loads.
    m_base_gpu_buffer_size = gpu_buffer.size();

    // The longest id of every set, read once, so the ID box keeps one width
    // whichever sets are checked.
    size_t longest = 0;
    for (const challenge_task &task : load_task_set(DS_ALL).tasks) longest = max(longest, task.id.size());
    m_app.longest_id_length.store((int)longest);

    // The saved selection, restored before the saved puzzle index is read below
    // so the index lands in the same list it came from. TRAIN when none is
    // saved; a saved one opens as it was left, even with nothing checked.
    load_dataset(settings_int(read_settings_file(), "datasets", DS_TRAIN) & (DS_ALL | DS_FILTERS));

    if (!m_tasks.tasks.empty()) {
      int saved_idx = settings_int(read_settings_file(), "puzzle", -1);
      if (saved_idx >= 0 && saved_idx < (int)m_tasks.tasks.size())
        m_view.puzzle_idx.store(saved_idx);
      m_view.puzzle_idx = max(0, m_view.puzzle_idx.load());
      m_view.puzzle_idx = min((int)m_tasks.tasks.size() - 1, m_view.puzzle_idx.load());

      load_current();
      // Its log starts with this run; later puzzles keep adding to theirs.
      m_log.start_fresh_file();
    }
    m_app.needs_upload = true;
  }

  inline void start_thread() {
    m_app.stop_threads = false;
    m_thread = jthread(&solver_session::control, this);
  }

  // Stops the control thread and the child with it.
  inline void stop_thread() {
    m_app.stop_threads = true;
    if (m_thread.joinable()) {
      m_thread.request_stop();
      m_thread.join();
    }
    lock_guard lock(m_child_mtx);
    m_child.kill();
  }

  // --- What the solver drew, for the frame and the command port. ---
  inline solver_state &get_solver_state() { return m_view; }
  inline int get_puzzle_count() const { return (int)m_tasks.tasks.size(); }
  inline const challenge_task &task(int idx) const { return m_tasks.tasks[idx]; }
  // Read under get_solver_state().search.mtx.
  inline const puzzle &get_puzzle() const { return m_puzzle; }
  inline int current_puzzle_idx() const { return m_view.puzzle_idx.load(); }

  // ID of the puzzle on screen (for copy-to-clipboard).
  inline string current_puzzle_id() const {
    scoped_lock lock(m_search.mtx);
    if (m_loaded)
      return m_puzzle.id;
    int i = m_view.puzzle_idx.load();
    if (i >= 0 && i < (int)m_tasks.tasks.size())
      return m_tasks.tasks[i].id;
    return string();
  }

  // The current puzzle's result as text, the way a benchmark entry reads.
  inline string result_text() const {
    return describe_result(m_puzzle, get_elapsed(m_app, m_search));
  }

  // --- Asked for from the window; the control thread carries them out. ---
  // A benchmark owns the child while it runs, so these do nothing then.
  inline void change_puzzle(int delta) {
    if (m_benchmarking || m_tasks.tasks.empty())
      return;
    m_queued_puzzle_delta += delta;
  }

  // Jump to a puzzle by ID (the filename stem). Tries an exact match first,
  // then falls back to the first puzzle whose ID starts with `id`. Matching is
  // case-insensitive. Returns true and queues the jump if a match was found.
  inline bool jump_to_puzzle_id(const string &id) {
    if (m_benchmarking || m_tasks.tasks.empty() || id.empty())
      return false;
    auto lower = [](string s) {
      for (char &c : s) c = (char)tolower((unsigned char)c);
      return s;
    };
    string needle = lower(id);
    int exact = -1, prefix = -1;
    for (int i = 0; i < (int)m_tasks.tasks.size(); ++i) {
      string stem = lower(m_tasks.tasks[i].id);
      if (stem == needle) { exact = i; break; }
      if (prefix < 0 && stem.compare(0, needle.size(), needle) == 0)
        prefix = i;
    }
    int target = (exact >= 0) ? exact : prefix;
    if (target < 0)
      return false;
    m_queued_puzzle_delta += target - m_view.puzzle_idx.load();
    return true;
  }

  // Kills the child and starts the puzzle over in a fresh one.
  inline void reset_solver() {
    if (!m_benchmarking) m_reset_requested = true;
  }

  // Dataset selection bitmask, applied by the control thread.
  inline int get_dataset_flags() const { return m_dataset_flags.load(); }
  inline void set_datasets(int flags) {
    if (m_benchmarking) return;
    m_queued_dataset_flags.store(flags);
  }

  // Reads solutions/<id>.txt and hands its program to the child, which scores
  // it and makes it the best.
  inline void import_solution() {
    if (m_benchmarking) return;
    string id = current_puzzle_id();
    if (id.empty()) return;
    string sol_path = "solutions/" + id + ".txt";
    if (!filesystem::exists(sol_path)) {
        m_log.debug("no " + sol_path + " to import (paths are read from where ARC runs)");
        return;
    }
    ifstream sf(sol_path);
    string sol_str, line;
    // How deep in brackets the program is where a line ends: a value may run on.
    int depth = 0;
    while (getline(sf, line)) {
      // Strip CR (Windows line endings).
      if (!line.empty() && line.back() == '\r') line.pop_back();
      // Trim leading/trailing whitespace.
      size_t a = line.find_first_not_of(" \t");
      if (a == string::npos) { if (!sol_str.empty()) break; else continue; }
      size_t b = line.find_last_not_of(" \t");
      line = line.substr(a, b - a + 1);
      // First line of the metadata block ("Puzzle:", "Train:", etc.) ends the
      // program: its "Key:" is a ':' outside every bracket. A program's own
      // colons sit inside its op values ("Tile [steps: ...]").
      bool key = false;
      for (const char c : line) {
        if (c == '[') ++depth;
        else if (c == ']') --depth;
        else if (c == ':' && depth == 0) key = true;
      }
      if (key) break;
      if (!sol_str.empty()) sol_str += ' ';
      sol_str += line;
    }
    // Drop a dangling trailing separator from line continuations.
    while (!sol_str.empty() && sol_str.back() == ' ')
      sol_str.pop_back();
    if (sol_str.empty()) {
        m_log.debug(sol_path + " holds no program above its first Key: value line");
        return;
    }
    send_when_running("import " + sol_str);
  }

  // Writes the best program and how it was found to solutions/<id>.txt, in
  // the shape import_solution reads back.
  inline void export_solution() {
    string id, result;
    {
      scoped_lock lock(m_search.mtx);
      if (!m_loaded || !m_puzzle.best_tree) return;
      id = m_puzzle.id;
      result = result_text();
    }
    filesystem::create_directories("solutions");
    string sol_path = "solutions/" + id + ".txt";
    ofstream sf(sol_path, ios::binary);
    if (sf.is_open()) {
      sf << result;
      m_log.debug("Manual solution exported to " + sol_path);
    }
  }

  // Pausing stops the clock: the time paused is added to the start, so what
  // the clock shows is time spent reading. Both are no-ops when already there,
  // so a reading that ends on an already-paused window leaves it paused.
  // Unpausing asks the child to read, as opening the pane does; the reading
  // pauses the window again when it is done.
  inline void pause() {
    if (m_benchmarking || m_app.is_paused) return;
    m_app.pause_start_time = chrono::steady_clock::now();
    m_app.is_paused = true;
  }
  inline void resume() {
    if (m_benchmarking || !m_app.is_paused) return;
    auto dt = chrono::steady_clock::now() - m_app.pause_start_time;
    {
      scoped_lock lock(m_search.mtx);
      m_search.start_time += dt;
    }
    m_app.is_paused = false;
    send("infer");
  }
  inline void toggle_pause() {
    if (m_app.is_paused) resume();
    else pause();
  }

  void log_debug(const string &msg) { m_log.debug(msg); }

  // --- A benchmark (35_benchmark.hpp). ---
  // Takes over the child: the one running now is killed, and the window's own
  // controls do nothing until end_benchmark.
  inline void begin_benchmark() {
    m_benchmarking = true;
    m_bench_idx = -1;
    lock_guard lock(m_child_mtx);
    m_child.kill();
  }

  // Puzzle `idx`, read in a process of its own while the window shows it.
  // Blocks until that process ends; nothing when it handed nothing back or
  // `cancelled` stopped it.
  inline optional<puzzle_result> benchmark_puzzle(int idx, const atomic<bool> &cancelled) {
    const filesystem::path out =
        filesystem::temp_directory_path() / ("arc_benchmark_" + m_tasks.tasks[idx].id + ".txt");
    error_code gone;
    filesystem::remove(out, gone);
    m_bench_out = out.string();
    m_bench_done = false;
    m_bench_idx = idx;  // the control thread takes it from here
    while (!m_bench_done && !m_app.stop_threads) {
      if (cancelled) {
        lock_guard lock(m_child_mtx);
        m_child.kill();
      }
      this_thread::sleep_for(chrono::milliseconds(10));
    }
    optional<puzzle_result> result = cancelled ? nullopt : puzzle_result::read(out.string());
    filesystem::remove(out, gone);
    return result;
  }

  // Gives the child back to the window, paused, on the puzzle the benchmark
  // ended on.
  inline void end_benchmark() {
    {
      lock_guard lock(m_child_mtx);
      m_child.kill();
    }
    m_bench_idx = -1;
    m_app.pause_start_time = chrono::steady_clock::now();
    m_app.is_paused = true;
    m_benchmarking = false;
    m_reset_requested = true;
  }

private:
  inline void load_dataset(int flags) {
    m_tasks = load_task_set(flags);
    m_dataset_flags.store(flags);
    m_app.dataset_flags.store(flags);
  }

  bool inference_pane_open() const {
    return m_app.inference_view.load() != global_app_state::INFERENCE_OFF;
  }

  // Parses the puzzle puzzle_idx names into this side's copy and blanks every
  // picture, the way a fresh child starts. An empty list names none, and
  // loading nothing empties the copy, so nothing of the last puzzle is drawn
  // or started. Nothing is inferred here: that waits for the pane or SPACE.
  inline void load_current() {
    sound_manager::get().clear_queue();
    int idx = m_view.puzzle_idx;
    int count = (int)m_tasks.tasks.size();
    const challenge_task none{};
    const challenge_task &task = idx >= 0 && idx < count ? m_tasks.tasks[idx] : none;
    {
      scoped_lock lock(m_search.mtx);
      m_loaded = load_task_from_content(task.content, task.id, m_puzzle,
                                        m_tasks.solutions_for(task.id));
      m_puzzle.id = task.id;
      m_label = {idx, count, m_dataset_flags.load()};
      m_search.start_time = chrono::steady_clock::now();
      loaded(m_puzzle);
    }
    m_log.set_puzzle_id(task.id);
    // A paused clock reads pause_start_time - start_time, so both start here.
    m_app.pause_start_time = m_search.start_time;
    m_settled_at = chrono::steady_clock::now();
    m_child_failed = false;
    m_asked_to_infer = false;
    m_announced = false;
    m_preview_sent = 0;
    m_app.needs_upload = true;
  }

  // The command line of a child for the puzzle on screen, from the program
  // beside this one.
  wstring child_command(const wstring &extra) {
    wchar_t self[MAX_PATH];
    GetModuleFileNameW(nullptr, self, MAX_PATH);
    const filesystem::path exe = filesystem::path(self).parent_path() / "arc_headless.exe";
    wstring line = L"\"" + exe.wstring() + L"\" --task " + filesystem::path(m_puzzle.id).wstring() +
                   L" --datasets " + to_wstring(m_dataset_flags.load()) + L" --serve --logs";
    if (m_app.log_mode.load() == logging_mode::stream) line += L" --verbose";
    return line + extra;
  }

  bool start_child(const wstring &extra) {
    lock_guard lock(m_child_mtx);
    filesystem::create_directories("logs");
    // What the child says on stderr, kept for the one running now.
    const bool ok = m_child.start(child_command(extra), "logs/search_process.txt");
    if (!ok) m_log.debug("could not start arc_headless.exe beside ARC.exe");
    m_sent_verbose = m_app.log_mode.load() == logging_mode::stream;
    return ok;
  }

  void send(const string &line) {
    lock_guard lock(m_child_mtx);
    m_child.send(line);
  }

  // Sent to the child as soon as there is one; imports wait for it to start.
  void send_when_running(const string &line) {
    lock_guard lock(m_outbox_mtx);
    m_outbox.push_back(line);
  }

  void kill_child() {
    {
      lock_guard lock(m_child_mtx);
      m_child.kill();
    }
    lock_guard lock(m_outbox_mtx);
    m_outbox.clear();
  }

  // THE CONTROL THREAD: navigation, the child's life, and everything it says.
  inline void control() {
    while (!m_app.stop_threads) {
      const bool verbose = m_app.log_mode.load() == logging_mode::stream;
      m_log.verbose = verbose;
      if (verbose != m_sent_verbose) {
        m_sent_verbose = verbose;
        send(verbose ? "verbose 1" : "verbose 0");
      }

      if (m_benchmarking) {
        run_benchmark_turn();
      } else {
        // A queued dataset selection (from the GUI checkmarks) rebuilds the
        // task list, then reloads the current puzzle. Handled like reset.
        int new_flags = m_queued_dataset_flags.exchange(-1);
        int delta = m_queued_puzzle_delta.exchange(0);
        bool reset = m_reset_requested.exchange(false);
        if (new_flags >= 0 && new_flags != m_dataset_flags.load()) {
          kill_child();
          {
            scoped_lock lock(m_search.mtx);
            load_dataset(new_flags);
            m_view.puzzle_idx = clamp(m_view.puzzle_idx.load(), 0, max(0, (int)m_tasks.tasks.size() - 1));
          }
          load_current();
        } else if (delta != 0 && !m_tasks.tasks.empty()) {
          kill_child();
          {
            scoped_lock lock(m_search.mtx);
            m_view.puzzle_idx = (m_view.puzzle_idx + delta) % (int)m_tasks.tasks.size();
            if (m_view.puzzle_idx < 0)
              m_view.puzzle_idx += static_cast<int>(m_tasks.tasks.size());
          }
          load_current();
        } else if (reset) {
          kill_child();
          load_current();
        }

        // A child once the window has stayed here a moment, so stepping
        // through puzzles starts nothing. It waits until told to read.
        bool started;
        {
          lock_guard lock(m_child_mtx);
          started = m_child.started();
        }
        if (!started && m_loaded && !m_child_failed &&
            chrono::steady_clock::now() - m_settled_at >= chrono::milliseconds(300)) {
          if (start_child(L"")) {
            if (!m_app.is_paused) send("infer");
          } else {
            m_child_failed = true;
          }
        }
      }

      read_facts_for_the_pane();
      ask_for_preview();
      {
        deque<string> outbox;
        {
          lock_guard lock(m_outbox_mtx);
          bool started;
          {
            lock_guard child_lock(m_child_mtx);
            started = m_child.started();
          }
          if (started) outbox.swap(m_outbox);
        }
        for (const string &line : outbox) send(line);
      }
      hear_the_child();
      this_thread::sleep_for(chrono::milliseconds(5));
    }
  }

  // The pane reads the facts off this side's copy, so they are read here too;
  // the child is asked to read its own and try what they spelled, as SPACE
  // asks it.
  void read_facts_for_the_pane() {
    if (!m_loaded || !inference_pane_open()) return;
    if (!m_puzzle.inferred) {
      // READ OUTSIDE THE LOCK THE FRAME DRAWS UNDER, taken in under it. A
      // reading takes seconds where it reads its passes (f823c43c's four), and
      // reading under the lock froze the window for all of them, till Windows
      // called it not responding. Only this thread writes the puzzle, so its
      // pairs hold still while they are read.
      inference read = infer(m_puzzle.train);
      scoped_lock lock(m_search.mtx);
      take_the_reading(m_puzzle, std::move(read), m_label, true);
      m_app.needs_upload = true;
    }
    if (!m_asked_to_infer && !m_benchmarking) {
      m_asked_to_infer = true;
      send_when_running("infer");
    }
  }

  // The frame names the program it wants a preview of; the child runs it.
  void ask_for_preview() {
    uint64_t wanted;
    string program;
    {
      scoped_lock lock(m_search.mtx);
      wanted = m_view.preview_wanted;
      program = m_view.preview_program;
    }
    if (wanted == 0 || wanted == m_preview_sent || program.empty()) return;
    m_preview_sent = wanted;
    send(to_text("preview %llu ", (unsigned long long)wanted) + program);
  }

  // Everything the child has said since last time, replayed; and what to do
  // when it has gone.
  void hear_the_child() {
    vector<string> lines;
    bool alive;
    {
      lock_guard lock(m_child_mtx);
      if (!m_child.started()) return;
      alive = m_child.poll([&](const string &line) { lines.push_back(line); });
      if (!alive) m_child.kill();
    }
    for (const string &line : lines) apply_stream_line(line, m_puzzle, m_search, *this);
    if (alive) return;
    if (m_benchmarking) {
      m_bench_done = true;
    } else {
      // Not killed from here, so it died: say so, and leave it dead until the
      // next puzzle or reset rather than restarting into the same crash.
      m_child_failed = true;
      m_log.debug("the puzzle's process ended by itself; see logs/search_process.txt");
      pause();
    }
  }

  // A benchmark puzzle asked for: loads it on screen and starts its reading,
  // which runs at once and ends by itself.
  void run_benchmark_turn() {
    const int idx = m_bench_idx.exchange(-1);
    if (idx < 0) return;
    kill_child();
    m_view.puzzle_idx = idx;
    load_current();
    m_app.is_paused = false;
    if (!start_child(L" --out \"" + filesystem::path(m_bench_out).wstring() + L"\""))
      m_bench_done = true;
  }

  // --- What the child said, replayed into this side's pictures. ---
  // Everything below arrives on the control thread, most of it under
  // m_search.mtx, which is the lock the frame draws under.
  void loaded(const puzzle &p) override {
    m_view.views = {};
    puzzle_views &views = m_view.views;
    m_view.preview_cache.clear();
    m_view.preview_uploaded_hash = 0;
    m_view.preview_wanted = 0;
    if (!m_gpu_buffer) return;
    // Drop the last puzzle's cells; its views pointed into them.
    m_gpu_buffer->resize(m_base_gpu_buffer_size);
    // Where each pair's cells landed, and a slot for what the best program
    // draws there, one entry per example, same index as the puzzle's own.
    auto build = [&](const vector<example> &side, vector<example_views> &out) {
      for (const example &ex : side) {
        example_views view;
        view.input = upload_grid(ex.input, *m_gpu_buffer);
        if (ex.output.width > 0)
          view.output = upload_grid(ex.output, *m_gpu_buffer);
        view.best = allocate_empty_grid(*m_gpu_buffer);
        out.push_back(view);
      }
    };
    build(p.train, views.train);
    build(p.test, views.test);
  }

  void changed() override { m_app.needs_upload = true; }

  // Kept for the hover; the best program's also goes into the BEST slots,
  // train then test, as the payload holds them.
  void previewed(uint64_t hash, preview_payload &&payload) override {
    scoped_lock lock(m_search.mtx);
    if (m_gpu_buffer && m_puzzle.best_tree && hash == m_puzzle.best_tree->hash) {
      puzzle_views &views = m_view.views;
      size_t at = 0;
      for (vector<example_views> *side : {&views.train, &views.test})
        for (example_views &view : *side)
          if (at < payload.accumulators.size())
            update_display_buffer(view.best, payload.accumulators[at++], *m_gpu_buffer);
      views.best_drawn_by = hash;
      m_app.needs_upload = true;
    }
    m_view.preview_cache[hash] = std::move(payload);
  }

  // The reading is done, so the window's switch goes back to paused. A solve
  // is announced once a puzzle: asked again, the reading says the same thing.
  void news(const search_news &n) override {
    pause();
    if (m_announced || !n.train_solved) return;
    m_announced = true;
    vector<pair<int, int>> seq{{1568, 150}};
    if (n.test_solved)
      seq.push_back({2093, 250});
    sound_manager::get().play_sequence(seq);
  }

  global_app_state &m_app;
  // This side's lock and clock, and its copy of the puzzle, kept up to date
  // from what the child says, both guarded by m_search.mtx.
  search_progress m_search;
  puzzle m_puzzle;
  bool m_loaded = false;
  puzzle_label m_label;
  solver_state m_view;
  logger m_log;
  vector<int> *m_gpu_buffer = nullptr;
  size_t m_base_gpu_buffer_size = 0;

  // One child, for the puzzle on screen.
  mutex m_child_mtx;
  search_process m_child;
  mutex m_outbox_mtx;
  deque<string> m_outbox;
  chrono::steady_clock::time_point m_settled_at{};
  bool m_child_failed = false;
  bool m_asked_to_infer = false;
  bool m_announced = false;
  bool m_sent_verbose = false;
  uint64_t m_preview_sent = 0;

  task_set m_tasks;
  // Which dataset(s) are loaded into m_tasks, as a bitmask of dataset_flag.
  atomic<int> m_dataset_flags{0};
  // A pending dataset-flag change requested from the GUI; -1 means none. The
  // control thread picks it up and rebuilds m_tasks (like reset).
  atomic<int> m_queued_dataset_flags{-1};
  atomic<int> m_queued_puzzle_delta{0};
  atomic<bool> m_reset_requested{false};

  // The benchmark's side of the handoff.
  atomic<bool> m_benchmarking{false};
  atomic<int> m_bench_idx{-1};
  atomic<bool> m_bench_done{false};
  string m_bench_out;

  jthread m_thread;
};

}  // namespace arc
