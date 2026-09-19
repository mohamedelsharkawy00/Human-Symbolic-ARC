// THE GUI: a window onto the solver. The solver never runs in this process:
// each search is an arc_headless child (see 29_solver_session.hpp), and this
// process creates no CUDA context at all -- it is linked without the GPU code,
// so it cannot. The Kaggle run (kaggle_main.cpp) is that same child, alone.
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
// In number order, so the files are read in the order they are numbered.
// 9_log is here for that alone: main calls nothing in it, but without it 8 and
// 9 would be read late, when 13_puzzle_loader first includes them.
#include "9_log.hpp"
#include "10_text_format.hpp"
#include "13_puzzle_loader.hpp"
#include "23_global_app_state.hpp"
#include "25_vulkan_context.hpp"
#include "27_gui.hpp"
#include "28_os_utilities.hpp"
#include "29_solver_session.hpp"
#include "30_benchmark.hpp"
#include "31_command_server.hpp"
#include "32_window_manager.hpp"

using namespace std;
using namespace arc;

int main() {
	// Cap RAM so a runaway search can't hang the machine. Its searches are its
	// children, under the same cap, and die with it.
	enforce_memory_limit();

	// The local tree mirrors the Kaggle layout (kaggle/input/...). The .exe
	// builds into a deep out/ dir, so walk up from the exe to the folder that
	// holds kaggle/input and make it current, so relative data paths resolve
	// wherever the build landed.
	wchar_t buf[MAX_PATH];
	if (GetModuleFileNameW(nullptr, buf, MAX_PATH))
		if (auto root = find_data_root(filesystem::path(buf).parent_path()); !root.empty())
			filesystem::current_path(root);

	// Claim the folder before anything is built or written, so a second app
	// stops here instead of overwriting this one's solutions and settings.
	claim_folder_or_exit();

	// Core State & Logic
	global_app_state app_state;
	solver_session solver(app_state);

	// Window Management. main() has no WinMain params, so fetch the module
	// handle ourselves. Start minimized when the agent drives: either the
	// --minimized launch flag (how Claude starts the app) or the saved state
	// from last run. Read before init so the window never shows full screen.
	bool start_minimized = wcsstr(GetCommandLineW(), L"--minimized") != nullptr ||
	                        settings_int(read_settings_file(), "minimized", 0) != 0;
	window_manager window_manager(app_state, solver);
	window_manager.init(GetModuleHandle(nullptr), start_minimized ? SW_MINIMIZE : SW_SHOW);
	int w = window_manager.get_width();
	int h = window_manager.get_height();

	// Render Context
	vulkan_context context;
	context.init(window_manager.get_hwnd(), w, h);

	// Data Buffers & Solver Setup
	vector<int> gpu_buffer;
	map<uint32_t, display_grid> chars;
	display_grid bg_grid;
	solver.init(gpu_buffer, chars, bg_grid);

	// Experiment knobs, handed to the first search: they are about the puzzle
	// the app opens on.
	// --start-examples N judges candidates against N examples from the first
	// one, instead of starting at one and promoting.
	// --max-attempts N pauses the search once it has tried that many candidates,
	// so two runs compared against each other stopped at the same place.
	if (const wchar_t *arg = wcsstr(GetCommandLineW(), L"--max-attempts "))
		if (long long given = _wtoi64(arg + 15); given > 0)
			solver.set_stop_after_attempts(given);

	if (const wchar_t *arg = wcsstr(GetCommandLineW(), L"--start-examples "))
		if (int given = _wtoi(arg + 17); given > 0)
			solver.set_start_example_count(given);

	// Restore the saved GUI settings. Datasets were already restored inside
	// solver.init; pause and mute go through their owners here. Before the
	// thread starts, so the first search is started already knowing whether
	// to run.
	{
		auto saved = read_settings_file();
		apply_settings(app_state, saved);
		sound_manager::get().set_muted(settings_int(saved, "muted", 0) != 0);
		if (settings_int(saved, "paused", 1) == 0) solver.toggle_pause();
	}

	solver.start_thread();
	benchmark_run benchmark(app_state, solver);

	// Telemetry / Profiling State
	auto last_time = chrono::steady_clock::now();
	long long last_attempts = 0;
	auto last_attempts_at = chrono::steady_clock::now();
	float fps = 0.0f;
	float aps = 0.0f;
	float gpu_ms = 0.0f;
	long long last_gpu_wait = 0;
	long long last_gpu_count = 0;

	// Settings writer: snapshot once a second, write only when something changed.
	string last_settings;
	auto last_settings_check = chrono::steady_clock::now();

	// THE PORT OPENS LAST, after everything a command can reach exists. A
	// screenshot draws the frame where it is asked for, so the handler needs the
	// same buffers and counters the frame loop draws from.
	// Command port: one line in, one line out. Vocabulary is the ui_action
	// names, plus "state" (read where the app is), "puzzle <id>" (jump) and
	// "screenshot <path>" (what the window would look like, written as a PNG).
	// --port N says where to start looking; the server walks up from there, so a
	// second app never fights the first for 2026.
	unsigned short first_port = 2026;
	if (const wchar_t *arg = wcsstr(GetCommandLineW(), L"--port "))
		if (int given = _wtoi(arg + 7); given > 0 && given < 65536)
			first_port = (unsigned short)given;

	command_server cmd_server;
	cmd_server.init(first_port, [&](const string &line) -> string {
		// TAKEN RIGHT HERE, because this handler already runs on the thread that
		// draws — the port is polled from inside the frame loop. Handing the
		// work to that loop and waiting for it was waiting on ourselves.
		// The window is never touched: the frame goes into memory, so a
		// minimized app answers exactly like a visible one.
		if (line.starts_with("screenshot ")) {
			const string path = line.substr(11);
			if (path.empty()) return "error screenshot needs a path";
			// DRAWN TWICE, AND ONLY THE SECOND IS KEPT. A frame fills the buffer
			// it draws from as it goes, so the card only has it on the frame
			// after — which the window never notices, drawing sixty a second,
			// and which a single picture would show as an empty screen. The
			// first pass is how the frame says what it wants; the second draws
			// it. Nothing here waits on the window, so it holds while minimized.
			auto &st = solver.get_solver_state();
			for (int pass = 0; pass < 2; ++pass) {
				context.upload_all_data(gpu_buffer);
				app_state.needs_upload = false;
				if (!context.begin_capture(w, h)) return "error could not draw the frame";
				if (int p_idx = st.puzzle_idx; p_idx >= 0 && p_idx < solver.get_puzzle_count()) {
					lock_guard<mutex> lock(st.search.mtx);
					draw_frame(&context, chars, bg_grid, st, app_state,
						solver.get_puzzle(), w, h, fps, aps, gpu_ms, p_idx,
						solver.get_puzzle_count(), gpu_buffer);
				}
				if (!context.end_capture(pass == 1 ? path : string()))
					return "error could not write " + path;
			}
			return "ok " + path;
		}
		if (line == "state") {
			auto &st = solver.get_solver_state();
			int idx = st.puzzle_idx.load();
			bool solved = idx >= 0 && idx < solver.get_puzzle_count() &&
			              solver.get_puzzle().is_solved;
			// gpu_used says whether the card is shared right now, so nobody reads
			// a contended run as a regression.
			gpu_usage gpu = read_gpu_usage();
			// The three flags read as words, not as 1 and 0: that is the reply
			// every caller of this port already parses.
			auto yes_no = [](bool b) { return b ? "true" : "false"; };
			return to_text(
				"ok id=%s idx=%d/%d paused=%s benchmarking=%s solved=%s elapsed=%s "
				"attempts=%lld gpu_used=%d%% build=%s",
				solver.current_puzzle_id().c_str(), idx, solver.get_puzzle_count(),
				yes_no((bool)app_state.is_paused), yes_no((bool)app_state.is_benchmarking),
				yes_no(solved), format_elapsed(get_elapsed(app_state, st.search)).c_str(),
				st.search.attempts.load(), gpu.used_percent, exe_build_time().c_str());
		}
		if (line.starts_with("puzzle ")) {
			string id = line.substr(7);
			return solver.jump_to_puzzle_id(id) ? "ok " + id : "error unknown id " + id;
		}
		for (size_t i = 1; i < size(ui_action_names); ++i)
			if (line == ui_action_names[i]) {
				window_manager.perform((ui_action)i);
				return "ok";
			}
		return "error unknown command " + line;
	});

	// Say once, at startup, how busy the card already is and whether a sibling
	// app took 2026 before us. Both mean this run shares the GPU, and a shared
	// run is not comparable to one that had the card to itself.
	{
		// Which binary is answering. An app that started before the build after
		// it gives old answers to new questions, and nothing else here says so.
		cmd_server.note(to_text("running the binary built %s", exe_build_time().c_str()));
		gpu_usage gpu = read_gpu_usage();
		cmd_server.note(to_text(
			"gpu %llu MB of %llu MB in use (%d%%)%s", gpu.used_mb, gpu.total_mb,
			gpu.used_percent,
			cmd_server.port() != 2026 ? ", another app already holds 2026" : ""));
	}

	// Program Loop, driven off the window message pump: the benchmark switch,
	// then settings, stats and the frame.
	while (window_manager.process_messages()) {
		window_manager.tick_repeat();
		// Before the minimized early-out, so commands work while minimized.
		cmd_server.poll();

		// Minimized / zero client area: skip all rendering AND swapchain
		// recreation. A zero imageExtent is invalid per the Vulkan spec and
		// AV's inside the NV driver (nvwgf2umx) on alt-tab. DWM keeps the
		// last presented frame for the taskbar thumbnail, so the hover
		// preview is preserved while paused. Gates rendering only: this used
		// to skip the whole loop body, which silently stopped the stats
		// sampling and the benchmark loop whenever the agent drove minimized.
		bool minimized = context.minimized();

		// Recreate the swapchain when acquire/present reported it out of
		// date or suboptimal (set inside vulkan_context). Extent comes from
		// the actual window client rect, not GetSystemMetrics (which returns
		// full-screen size even when the window is small/minimized).
		if (!minimized && (context.needs_recreate() || app_state.needs_resize)) {
			context.client_extent(w, h);
			context.recreate_swap_chain(w, h);
			app_state.needs_resize = false;
		}

		auto now = chrono::steady_clock::now();
		double delta = chrono::duration<double>(now - last_time).count();
		last_time = now;

		auto &state = solver.get_solver_state();
		benchmark.follow_switch();

		// Persist the settings once a second, only when something changed.
		if (chrono::duration<double>(now - last_settings_check).count() >= 1.0) {
			last_settings_check = now;
			string s = serialize_settings(app_state, sound_manager::get().is_muted(), state.puzzle_idx.load());
			if (s != last_settings) {
				last_settings = s;
				ofstream f(get_settings_path());
				f << s;
			}
		}

		// Stats are reads of what the search last said, not rendering — sampled
		// every loop so they stay live while minimized, which is how the agent
		// runs the app.
		fps = (float)(1.0 / delta);
		long long current_attempts = state.search.attempts.load();
		if (current_attempts < last_attempts) {
			last_attempts = 0;
			last_attempts_at = now;
		}
		// MEASURED BETWEEN ONE COUNT AND THE NEXT, not once a frame: the search
		// says where it is on its own beat, and frames that fall between two of
		// its words carry no news. Reading a rate off those gave a zero every
		// time, and the number flickered between zero and a spike.
		if (current_attempts != last_attempts) {
			double since = chrono::duration<double>(now - last_attempts_at).count();
			if (since > 0.0) aps = (float)((current_attempts - last_attempts) / since);
			last_attempts = current_attempts;
			last_attempts_at = now;
		} else if (chrono::duration<double>(now - last_attempts_at).count() > 1.0) {
			// Nothing tried for a second: it is paused, waiting or gone.
			aps = 0.0f;
		}

		// Attempts count candidates handed to the GPU, so a frame spent
		// inside a running kernel scores zero — that is busy, not idle.
		// The old "- Idle" verdict read that as a stall and was always wrong.
		// gpu is how long ONE batch waits on the kernel. Held over between
		// samples, so a batch longer than a frame reads as its own duration
		// instead of the slice of it this frame happened to catch.
		long long current_gpu_wait = state.search.gpu_wait_now();
		long long current_gpu_count = state.search.gpu_wait_count.load();
		if (current_gpu_wait < last_gpu_wait) { last_gpu_wait = 0; last_gpu_count = 0; }
		if (current_gpu_count > last_gpu_count) {
			gpu_ms = (float)((current_gpu_wait - last_gpu_wait) / 1e6
				/ (double)(current_gpu_count - last_gpu_count));
			last_gpu_wait = current_gpu_wait;
			last_gpu_count = current_gpu_count;
		}

		solver.log_debug(to_text("[APS] %.2f  (Attempts: %lld) gpu=%.0fms",
			aps, current_attempts, gpu_ms));

		if (minimized) {
			this_thread::sleep_for(chrono::milliseconds(16));
			continue;
		}

		// Sync solver buffers to GPU
		if (app_state.needs_upload) {
			context.upload_all_data(gpu_buffer);
			app_state.needs_upload = false;
		}

		// Draw frame
		if (context.begin_frame()) {
			int p_idx = state.puzzle_idx;
			int p_count = solver.get_puzzle_count();

			if (p_idx >= 0 && p_idx < p_count) {
				lock_guard<mutex> lock(state.search.mtx);
				draw_frame(&context, chars, bg_grid, state,
					app_state, solver.get_puzzle(), w, h, fps, aps, gpu_ms, p_idx,
					p_count, gpu_buffer);
			}
			context.end_frame();
		}
		// begin_frame() returned false: acquire failed / swapchain out of
		// date. It already set the recreate flag; the top-of-loop guard
		// handles it next iteration. Yield to avoid spinning.
		else {
			this_thread::sleep_for(chrono::milliseconds(16));
		}
	}

	// A benchmark still running is stopped, its puzzle's process with it.
	app_state.is_benchmarking = false;
	benchmark.follow_switch();
	cmd_server.shutdown();
	// Final settings flush so the last second of changes is not lost.
	{
		ofstream f(get_settings_path());
		f << serialize_settings(app_state, sound_manager::get().is_muted(), solver.get_solver_state().puzzle_idx.load());
	}
	// Ends the search, and its process with it.
	solver.stop_thread();
	return 0;
}
