#pragma once
// Everything the GUI remembers between frames: what the person has switched
// on, where the panes are, and the pictures it keeps of the solver's work.
#include "20_environment.hpp"
#include "22_solver_log.hpp"    // search_progress, format_elapsed
#include "26_search_stream.hpp" // preview_payload
#include "27_grid_buffer.hpp"   // display-side grid layout: placement, display_grid, puzzle_views
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include "18_text_format.hpp"
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace arc {
using namespace std;

// Path where the GUI persists its settings, current puzzle index included.
inline std::filesystem::path get_settings_path() {
	std::filesystem::path dir = "settings";
	std::filesystem::create_directories(dir);
	return dir / "settings.txt";
}

// Read settings.txt into key -> value pairs. A missing file or unknown key is
// simply skipped, so old files never break a newer build.
inline unordered_map<string, string> read_settings_file() {
	unordered_map<string, string> kv;
	ifstream f(get_settings_path());
	string key, value;
	while (f >> key >> value) kv[key] = value;
	return kv;
}

inline int settings_int(const unordered_map<string, string>& kv, const string& key, int fallback) {
	auto it = kv.find(key);
	return it == kv.end() ? fallback : atoi(it->second.c_str());
}

enum class logging_mode { none, stream };
// op_role lives beside the op list in 14_vocabulary.hpp — it is what an op DOES. The
// op_tab_* actions below are clicks on the drawer tab that draws each role.
enum class ui_action { none, toggle_pause, prev_puzzle, next_puzzle, first_puzzle, last_puzzle, edit_id, copy_id, paste_id, toggle_mute, toggle_log, import_sol, export_sol, reset_solver, toggle_benchmark, inc_limit, dec_limit, exit_app, minimize_app, toggle_ds_train, toggle_ds_eval, toggle_ds_test, toggle_ds_custom, toggle_ds_agi_1, toggle_ds_agi_2, toggle_inference_facts, toggle_inference_lines, cycle_coarseness, cycle_coarseness_back, step_edge, step_edge_back, step_piece, step_piece_back, step_dropped, step_dropped_back, step_motif, step_motif_back, step_left_over, step_left_over_back, toggle_op_list, op_tab_get, op_tab_reduce, op_tab_select, op_tab_translate, op_tab_map, op_tab_generate, op_tab_combine, toggle_programs };
// The same actions as words, in the same order: the command port's vocabulary.
// Keep in lockstep with ui_action above.
inline constexpr const char* ui_action_names[] = { "none", "toggle_pause", "prev_puzzle", "next_puzzle", "first_puzzle", "last_puzzle", "edit_id", "copy_id", "paste_id", "toggle_mute", "toggle_log", "import_sol", "export_sol", "reset_solver", "toggle_benchmark", "inc_limit", "dec_limit", "exit_app", "minimize_app", "toggle_ds_train", "toggle_ds_eval", "toggle_ds_test", "toggle_ds_custom", "toggle_ds_agi_1", "toggle_ds_agi_2", "toggle_inference_facts", "toggle_inference_lines", "cycle_coarseness", "cycle_coarseness_back", "step_edge", "step_edge_back", "step_piece", "step_piece_back", "step_dropped", "step_dropped_back", "step_motif", "step_motif_back", "step_left_over", "step_left_over_back", "toggle_op_list", "op_tab_get", "op_tab_reduce", "op_tab_select", "op_tab_translate", "op_tab_map", "op_tab_generate", "op_tab_combine", "toggle_programs" };

struct click_zone {
	int x, y, w, h;
	ui_action action;
};
struct node_zone {
	int x, y, w, h;
	uint64_t node_hash;
};
// EVERY TRACK READS THE SAME WAY, because they all answer the same question
// about a list: none of it at the left, one of them in the middle, all of it
// at the right.
inline constexpr int SHOW_OFF = -2;
inline constexpr int SHOW_ALL = -1;

// One track holds WHAT IS CHOSEN, never where the handle sits. Off and all
// mean the same on any list; an index means something only on the list it was
// picked from. A track that stored its stop would answer a different question
// the moment a new puzzle or split changed the list's length. Only the draw
// knows that length, so it says so on its way past.
struct track {
	atomic<int> at;
	atomic<int> count{0};
	explicit track(int start) : at(start) {}
	// The track reads OFF, then one stop per item, then ALL. That ordering is
	// known here and nowhere else.
	static int to_stop(int at, int n) {
		return at == SHOW_OFF ? 0 : at == SHOW_ALL ? n + 1 : min(at, n - 1) + 1;
	}
	static int from_stop(int stop, int n) {
		return stop == 0 ? SHOW_OFF : stop >= n + 1 ? SHOW_ALL : stop - 1;
	}
	// The draw says how long the list is and reads the choice back in one move,
	// clamped onto what this list has. The clamp is not written back, so a long
	// list you come back to still opens where you left it.
	int pick(int n) {
		count.store(n);
		const int a = at.load();
		return a < 0 ? a : n > 0 ? min(a, n - 1) : SHOW_OFF;
	}
	// A step lands on the stop next door and STOPS at either end. Held down, a
	// wrap ran the list round and round past the place you were steering for,
	// and a stepper you can hold has to come to rest against its own end.
	void step(int by) {
		const int n = count.load();
		const int stop = to_stop(at.load(), n) + by;
		at.store(from_stop(stop < 0 ? 0 : stop > n + 1 ? n + 1 : stop, n));
	}
};

// WHICH READING OF THE SPLIT IS ON SCREEN. It holds a cell of the table, but
// SITTING ON THE ELECTED ONE IS A CHOICE AND NOT A NUMBER: which cell inference
// chose is a different one on every puzzle, so a stepper resting there follows
// it to the next puzzle, and one somebody has moved holds its own cell instead.
// Step back onto the elected cell and it follows again, because being on it is
// the whole of what following means.
struct shown_reading {
	atomic<int> at{-1};        // -1 = nothing picked yet, show the elected one
	atomic<int> elected{-1};   // the elected cell of the puzzle last drawn
	atomic<int> count{0};
	// The draw says which reading inference chose and how many there are, and
	// reads the one to show back in one move — the same trade a track makes.
	int pick(int chosen, int n) {
		count.store(n);
		const int before = elected.exchange(chosen);
		int here = at.load();
		if (here < 0 || here == before) here = chosen;
		here = n > 0 ? min(max(here, 0), n - 1) : 0;
		at.store(here);
		return here;
	}
	// A step lands on the cell next door and STOPS at either end, so a held
	// button rests against the last reading instead of rolling round.
	void step(int by) {
		const int n = count.load();
		at.store(n > 0 ? min(max(at.load() + by, 0), n - 1) : 0);
	}
};

// A track the mouse can scrub. Pressing or dragging anywhere along it puts the
// track on the stop under the cursor, so any place on it is one gesture away
// instead of a press per step. The track lives in this struct below, so the
// pointer stays good for as long as the app does.
struct slider_zone {
	int x, y, w, h;
	track* which;
	int stops;
	int knob; // the handle's width, so the hit test lands where the handle draws
};

// HOW FAR A REGION TALLER THAN ITS ROOM IS SCROLLED, in pixels. The draw says
// what it holds and how much taller than its room that is, and reads back how
// far down to start: the top again for anything new to hold. The wheel moves
// it, stopping at either end.
struct scroll {
	atomic<int> at{0};
	atomic<int> most{0};
	atomic<size_t> holding{0};
	int pick(size_t holds, int taller) {
		if (holding.exchange(holds) != holds) at.store(0);
		most.store(max(0, taller));
		at.store(clamp(at.load(), 0, most.load()));
		return at.load();
	}
	void by(int pixels) { at.store(clamp(at.load() + pixels, 0, most.load())); }
};
// A region the wheel scrolls: where it is, and how far one of its lines is.
struct scroll_zone {
	int x, y, w, h;
	scroll* which;
	int line;
};
// THE GUI'S SIDE OF THE SOLVER: which puzzle is up, and the pictures kept of
// it. `search` holds this side's lock and clock; its mutex guards the puzzle,
// and this side's pictures with it.
struct solver_state {
	search_progress &search;
	explicit solver_state(search_progress &s) : search(s) {}
	atomic<int> puzzle_idx = -1;

	// Hover preview: what the hovered program paints on every example, keyed
	// by its node hash. Only the solver's process can run a program, so the
	// frame names the one it wants (preview_wanted, preview_program) and the
	// answer lands in the cache when it comes back. preview_grids is a fixed
	// pool of GPU display slots, one per example row; preview_object_grids is a
	// 2D pool [example][object_slot] for the object list the hovered op held.
	mutable unordered_map<uint64_t, preview_payload> preview_cache;
	mutable vector<display_grid> preview_grids;
	mutable vector<vector<display_grid>> preview_object_grids;
	// Where the puzzle's own pairs were uploaded. Beside the puzzle, not inside
	// it: core states the facts, the display keeps what it needs to paint them.
	puzzle_views views;
	mutable uint64_t preview_uploaded_hash = 0;
	mutable int preview_last_puzzle_idx = -1;
	mutable uint64_t preview_wanted = 0;
	mutable string preview_program;
};

struct global_app_state {
	atomic<bool> needs_upload{false};
	atomic<bool> is_paused{true};  // start paused (no auto-solve)
	atomic<bool> show_op_list{false}; // overlay: palette of every DSL op (drag its handle open)
	atomic<bool> show_whole_programs{false}; // the facts pane shows every row of a long program, not its first few
	// Overlay: what inference read off the puzzle. Show it, step the coarseness,
	// walk the lines and pieces. Keys live in key_to_action, buttons in the pane.
	// It is either/or: the FACTS view reads out everything inference decided and
	// draws no lines, the LINES view draws the correspondence and the pane says
	// what each thing on the grids is.
	enum : int { INFERENCE_OFF, INFERENCE_FACTS, INFERENCE_LINES, INFERENCE_VIEWS };
	atomic<int> inference_view{INFERENCE_OFF};
	// Which way of cutting the grid up is on screen, each shown at the one
	// reading it kept, so this counts cuts and not readings — and following the
	// elected one is a state of its own, see shown_reading.
	shown_reading inference_rank;
	// Which single line is on screen, which piece of the reading to read off,
	// and which thrown-away match. Lines that land on the same place paint over
	// each other, so stacked ones are only visible one at a time. The other two
	// walk lists the lines cannot reach: a piece with no edge draws nothing, and
	// a dropped match was never drawn at all. The lines are what the view is
	// for, so they start on ALL and the other two start out of the way.
	track inference_edge{SHOW_ALL};
	track inference_piece{SHOW_OFF};
	track inference_dropped{SHOW_OFF};
	// Which set of copies is lit. A grid can hold several motifs and one
	// color cannot say which box belongs with which.
	track inference_motif{SHOW_OFF};
	// Which cell the reading could not account for. A cell belongs to no piece
	// and no match, so it is the one thing none of the walks above can reach.
	track inference_left_over{SHOW_OFF};
	atomic<int> op_tab{(int)op_role::get}; // active tab in the op-palette drawer
	atomic<int> op_drawer_bottom{0}; // y of drawer's bottom edge; clicks above it are swallowed by the drawer
	atomic<int> op_handle_x{0}, op_handle_y{0}, op_handle_w{0}, op_handle_h{0}; // pull-handle rect (published for drag hit-test)
	atomic<bool> op_handle_hot{false}; // mouse is near the handle -> draw it bright
	atomic<int> op_drawer_h{0};      // how far the drawer's bottom edge is pulled down; the draw stops it at full
	atomic<int> op_drawer_h_max{0};  // max height (grid-label row), published for drag clamp
	atomic<bool> gui_minimized{false};
	atomic<bool> needs_resize{false};
	atomic<bool> stop_threads{false};
	atomic<logging_mode> log_mode{logging_mode::none};
	chrono::steady_clock::time_point pause_start_time{chrono::steady_clock::now()};

	// Currently loaded dataset selection (bitmask of dataset_flag),
	// mirrored from the solver so the GUI checkmarks can render their state.
	atomic<int> dataset_flags{0};
	// The longest id on disk, measured once, so the ID box fits every one of
	// them and never changes width.
	atomic<int> longest_id_length{0};

	// Benchmark mode: the switch, and what a run is asked for (35_benchmark.hpp).
	atomic<bool> is_benchmarking{false};
	int benchmark_puzzle_limit{10};

	// Benchmark session stats (reset on (re)start)
	atomic<int> benchmark_puzzles_run{0};
	atomic<int> benchmark_train_solved{0};
	atomic<int> benchmark_test_solved{0};
	atomic<chrono::steady_clock::time_point> benchmark_start_time{};

	// Puzzle-ID inline editor. When `active`, keystrokes append to `buffer`
	// (instead of triggering the normal hotkeys) and the ID field renders the
	// buffer with a cursor. `error` flashes the field when a typed ID matched
	// nothing. Guarded by `id_edit_mtx` since the win32 message thread writes
	// the buffer while the render thread reads it.
	struct id_editor {
		bool active = false;
		bool error = false;
		string buffer;
	} id_edit;
	mutex id_edit_mtx;

	vector<click_zone> click_zones;
	vector<slider_zone> slider_zones;
	// The regions the wheel scrolls, and how far each is: the inference pane,
	// and the BEST program under the grids.
	vector<scroll_zone> scroll_zones;
	scroll pane_scroll, best_scroll;
	mutex node_zones_mtx;
	vector<node_zone> node_zones;
	atomic<uint64_t> hovered_hash{0};
	atomic<ui_action> active_ui_action{ui_action::none};
};

// Snapshot every persisted setting as "key value" lines. The whole file is
// rewritten on any change, so the file and the app can never disagree. Mute
// lives in the sound manager and the puzzle index in the solver state, so
// the caller passes those in.
inline string serialize_settings(const global_app_state& app, bool muted, int puzzle_idx) {
	string s;
	auto line = [&](const char* key, long long value) { s += to_text("%s %lld\n", key, value); };
	line("puzzle", puzzle_idx);
	line("paused", (int)app.is_paused.load());
	line("muted", (int)muted);
	line("datasets", app.dataset_flags.load());
	line("benchmark_limit", app.benchmark_puzzle_limit);
	line("minimized", (int)app.gui_minimized.load());
	line("logging", (int)(app.log_mode.load() == logging_mode::stream));
	line("inference_view", app.inference_view.load());
	line("inference_cut", app.inference_rank.at.load());
	line("inference_edge", app.inference_edge.at.load());
	line("inference_piece", app.inference_piece.at.load());
	line("inference_dropped", app.inference_dropped.at.load());
	line("inference_motif", app.inference_motif.at.load());
	line("inference_left_over", app.inference_left_over.at.load());
	return s;
}

// Restore the saved settings into the app state. Pause, mute, and the dataset
// selection are applied elsewhere: unpausing must go through the solver (it
// fixes up timers), mute lives in the sound manager, and datasets must be
// restored before the saved puzzle index is read.
inline void apply_settings(global_app_state& app, const unordered_map<string, string>& kv) {
	app.benchmark_puzzle_limit = settings_int(kv, "benchmark_limit", app.benchmark_puzzle_limit);
	app.inference_view.store(settings_int(kv, "inference_view", global_app_state::INFERENCE_OFF)
	                         % global_app_state::INFERENCE_VIEWS);
	// -1 says nothing was picked, so a fresh app opens on whatever each
	// puzzle chose rather than on a cell that meant something last time. Saved
	// under its own name since it counts cuts: a reading number saved by an
	// older build would land on a cut that means something else.
	app.inference_rank.at.store(settings_int(kv, "inference_cut", -1));
	// Logging stays where it was left: a run whose log is off records nothing,
	// and finding that out afterwards costs the run.
	app.log_mode.store(settings_int(kv, "logging", 0) ? logging_mode::stream : logging_mode::none);
	// -2 is OFF and -1 is ALL whatever the list, so what was saved still means
	// what it said next launch.
	app.inference_edge.at.store(settings_int(kv, "inference_edge", SHOW_ALL));
	app.inference_piece.at.store(settings_int(kv, "inference_piece", SHOW_OFF));
	app.inference_dropped.at.store(settings_int(kv, "inference_dropped", SHOW_OFF));
	app.inference_motif.at.store(settings_int(kv, "inference_motif", SHOW_OFF));
	app.inference_left_over.at.store(settings_int(kv, "inference_left_over", SHOW_OFF));
}

inline float get_elapsed(const global_app_state &app, const search_progress &search) {
	if (app.is_paused) {
	  // pause_start_time and start_time are initialized independently at
	  // construction, so before the solver has ever run (e.g. start-paused)
	  // their difference can be slightly negative. Clamp to 0.
	  float paused_elapsed = chrono::duration<float>(app.pause_start_time - search.start_time).count();
	  return paused_elapsed > 0.0f ? paused_elapsed : 0.0f;
	}
	return chrono::duration<float>(chrono::steady_clock::now() - search.start_time).count();
}

}
