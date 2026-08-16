// First include on the GUI build: pulls winsock2.h, which must precede every
// windows.h (solver_beam.hpp includes one) or the old Winsock wins.
#ifdef _MSC_VER
#include "command_server.hpp"
#endif
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "cuda_solver.cuh"
#include "global_app_state.hpp"
#include "solver_beam.hpp"
#include "submission.hpp"
#include "third_party/fmt/format.h"

// Windows-only GUI/Vulkan/Win32 headers. The headless (Kaggle/gcc) build skips
// these entirely; it runs the solver without a window or renderer.
#ifdef _MSC_VER
#include "os_utilities.hpp"
#include "gui.hpp"
#include "vulkan_context.hpp"
#include "window_manager.hpp"
#endif

using namespace std;
using namespace arc;

int main() {
	// Cap RAM on dev machine so a runaway solver can't hang it, Kaggle enforces limits by default.
#ifdef _MSC_VER
	enforce_memory_limit();
#endif
	cuda_init();

	// Windows: the local tree mirrors the Kaggle layout (kaggle/input/...). The
	// .exe builds into a deep out/ dir, so we walk up from the exe to find the
	// project root (the dir containing kaggle/input) and chdir there so relative
	// data paths resolve regardless of build location. Kaggle mounts its data at
	// fixed absolute paths, so no anchoring is needed there.
#ifdef _MSC_VER
	wchar_t buf[MAX_PATH];
	if (GetModuleFileNameW(nullptr, buf, MAX_PATH)) {
		for (auto p = filesystem::path(buf); p != p.parent_path(); p = p.parent_path()) {
			if (filesystem::exists(p / "kaggle" / "input")) {
				filesystem::current_path(p);
				break;
			}
		}
	}
#endif

	// Core State & Logic
	global_app_state app_state;
	beam_solver solver(app_state);

	// Window + Vulkan rendering: GUI build only. The headless build runs the
	// solver with no window or renderer.
#ifdef _MSC_VER
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

	// Command port: one line in, one line out. Vocabulary is the ui_action
	// names, plus "state" (read where the app is) and "puzzle <id>" (jump).
	// --port N says where to start looking; the server walks up from there, so a
	// second app never fights the first for 2026.
	unsigned short first_port = 2026;
	if (const wchar_t *arg = wcsstr(GetCommandLineW(), L"--port "))
		if (int given = _wtoi(arg + 7); given > 0 && given < 65536)
			first_port = (unsigned short)given;

	command_server cmd_server;
	cmd_server.init(first_port, [&](const string &line) -> string {
		if (line == "state") {
			auto &st = solver.get_solver_state();
			int idx = st.puzzle_idx.load();
			bool solved = idx >= 0 && idx < solver.get_puzzle_count() &&
			              solver.get_puzzle(idx).is_solved;
			// gpu_used says whether the card is shared right now, so nobody reads
			// a contended run as a regression.
			gpu_memory gpu = read_gpu_memory();
			return fmt::format(
				"ok id={} idx={}/{} paused={} benchmarking={} solved={} elapsed={} gpu_used={}%",
				solver.current_puzzle_id(), idx, solver.get_puzzle_count(),
				(bool)app_state.is_paused, (bool)app_state.is_benchmarking,
				solved, format_elapsed(get_elapsed(app_state, st)), gpu.used_percent);
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
		gpu_memory gpu = read_gpu_memory();
		cmd_server.note(fmt::format(
			"gpu {} MB of {} MB in use ({}%){}", gpu.used_mb, gpu.total_mb,
			gpu.used_percent,
			cmd_server.port() != 2026 ? ", another app already holds 2026" : ""));
	}
#endif

	// Data Buffers & Solver Setup
	vector<int> gpu_buffer;
	map<uint32_t, grid> chars;
	grid bg_grid;
	solver.init(gpu_buffer, chars, bg_grid);
	solver.start_thread();

	// Restore the saved GUI settings. Datasets were already restored inside
	// solver.init; pause and mute go through their owners here.
#ifdef _MSC_VER
	{
		auto saved = read_settings_file();
		apply_settings(app_state, saved);
		sound_manager::get().set_muted(settings_int(saved, "muted", 0) != 0);
		if (settings_int(saved, "paused", 1) == 0) solver.toggle_pause();
	}
#endif

	// Headless (Kaggle) startup config. There is no GUI to toggle benchmarking
	// or set options, so we turn it on by default and configure it here.
	//   - is_benchmarking: benchmark mode is the only mode on Kaggle.
	//   - benchmark_puzzle_limit = 0: no limit -> run ALL tasks (not just 10).
	//   - benchmark_timeout_seconds = 120: per-puzzle cap (2 min). Sized to the
	//     Kaggle 9h kernel budget: 240 tasks * 120s = 8h, leaving headroom for
	//     startup/compile/IO. Tune later based on real per-puzzle timings.
	//   - ARC_PUZZLE_LIMIT env var overrides the limit (0/unset = all). Lets the
	//     notebook do a quick N-puzzle validation run without recompiling.
	//   - ARC_TIMEOUT env var overrides per-puzzle seconds (optional).
#ifndef _MSC_VER
	app_state.is_benchmarking = true;
	app_state.is_paused = false;
	app_state.benchmark_puzzle_limit = 0;
	app_state.benchmark_timeout_seconds = 120.0;
	if (const char* lim = std::getenv("ARC_PUZZLE_LIMIT"))
		app_state.benchmark_puzzle_limit = std::atoi(lim);
	if (const char* to = std::getenv("ARC_TIMEOUT"))
		app_state.benchmark_timeout_seconds = std::atof(to);

	// 4-GPU sharding: each process (one per GPU, pinned via CUDA_VISIBLE_DEVICES)
	// solves a contiguous slice of tasks and writes its own submission file.
	//   ARC_SHARD_TOTAL = number of shards (e.g. 4), ARC_SHARD_INDEX = 0..total-1.
	// Slice = [shard*chunk, (shard+1)*chunk) over the (optionally limited) task
	// set; last shard takes the remainder. submission_writer output path gets a
	// per-shard suffix so the 4 files don't clobber; the notebook merges them.
	int shard_index = 0, shard_total = 1;
	if (const char* si = std::getenv("ARC_SHARD_INDEX")) shard_index = std::atoi(si);
	if (const char* st = std::getenv("ARC_SHARD_TOTAL")) shard_total = std::max(1, std::atoi(st));

	int total_tasks = solver.get_puzzle_count();
	if (app_state.benchmark_puzzle_limit > 0)
		total_tasks = std::min(total_tasks, app_state.benchmark_puzzle_limit);
	if (shard_total > 1) {
		int chunk = (total_tasks + shard_total - 1) / shard_total;  // ceil
		int start = shard_index * chunk;
		int count = std::min(chunk, total_tasks - start);
		if (count < 0) count = 0;
		// Point the solver at this shard's first task and cap to its count.
		solver.get_solver_state().puzzle_idx.store(start);
		solver.reload_current_puzzle();
		app_state.benchmark_puzzle_limit = count;
		std::fprintf(stderr, "shard %d/%d: tasks [%d, %d) of %d\n",
		             shard_index, shard_total, start, start + count, total_tasks);
	}

#endif
	submission_writer submission;   // both builds accumulate predictions

	// Telemetry / Profiling State (display-only; harmless on headless)
	auto last_time = chrono::steady_clock::now();
	long long last_attempts = 0;
	float fps = 0.0f;
	float aps = 0.0f;
	float gpu_ms = 0.0f;
	long long last_gpu_wait = 0;
	long long last_gpu_count = 0;

#ifdef _MSC_VER
	// Settings writer: snapshot once a second, write only when something changed.
	string last_settings;
	auto last_settings_check = chrono::steady_clock::now();
#endif

	// Program Loop. Windows drives it off the window message pump and renders
	// each frame; the headless build spins a plain loop until benchmarking is
	// done. The benchmark logic in the body is identical on both -- only the
	// window/render plumbing is compiled out.
#ifdef _MSC_VER
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
#else
	// Headless: run until the benchmark loop signals completion (it clears
	// is_benchmarking when the last puzzle / limit is reached, see body below).
	while (app_state.is_benchmarking) {
#endif

		auto now = chrono::steady_clock::now();
		double delta = chrono::duration<double>(now - last_time).count();
		last_time = now;

		auto &state = solver.get_solver_state();
		// Benchmark execution loop
		if (app_state.is_benchmarking) {
			// Record start time and create the report once per benchmark run.
			// The human-readable timestamped report is GUI-only: it uses
			// the headless build emits submission.json instead.
			if (!app_state.benchmark.started) {
				solver.benchmark_begin();
				app_state.benchmark.started = true;
#ifdef _MSC_VER
				time_t now_t = chrono::system_clock::to_time_t(chrono::system_clock::now());
				tm local_tm{};
				localtime_s(&local_tm, &now_t);
				char ts[32];
				strftime(ts, sizeof(ts), "%Y-%m-%d_%I-%M-%S_%p", &local_tm);
				string benchmark_file = fmt::format("benchmarks/benchmark_{}.txt", ts);
				solver.start_benchmark_report(benchmark_file);
#endif
			}

			int p_idx = state.puzzle_idx;
			int p_count = solver.get_puzzle_count();
			if (app_state.benchmark.waiting_for_advance) {
				if (p_idx != app_state.benchmark.last_idx) app_state.benchmark.waiting_for_advance = false;
			} else if (p_idx >= 0 && p_idx < p_count) {
				const puzzle &p = solver.get_puzzle(p_idx);
				float elapsed = get_elapsed(app_state, state);
				bool timed_out = elapsed >= app_state.benchmark_timeout_seconds;

				// Only act on a finish ONCE per puzzle. A solve/timeout can be
				// observed on several loop iterations before the solver thread
				// actually swaps in the next puzzle; without this guard the same
				// puzzle gets recorded (and captured) twice, inflating the run
				// count and dropping a real puzzle off the end of a limited run.
				bool already_recorded = (app_state.benchmark.recorded_idx.load() == p_idx);
				if ((p.is_solved || timed_out) && !already_recorded) {
					app_state.benchmark.recorded_idx.store(p_idx);
					// Accumulate stats
					long long attempts = state.attempts.load();
					solver.benchmark_record(p, attempts, elapsed);

					// Capture this puzzle's predictions for submission.json (both builds).
					submission.capture(p);

					// Log result
					string result = solver.format_solution_result(p, state, app_state);
					solver.log_benchmark_result(result);

					// Check puzzle limit
					int limit = app_state.benchmark_puzzle_limit;
					bool at_last_puzzle = (p_idx + 1 >= p_count);
					bool hit_limit = (limit > 0 && app_state.benchmark_puzzles_run.load() >= limit);

					if (at_last_puzzle || hit_limit) {
						// Write summary block. GUI: appends the solve-rate summary
						// to benchmarks/benchmark_<ts>.txt. No-op headless.
						solver.finish_benchmark_report();
#ifdef _MSC_VER
						// GUI: write the submission file NOW (on benchmark
						// completion) instead of only on window close, so a
						// limited run (e.g. LIMIT=10) produces submission.json
						// without having to exit the app.
						filesystem::create_directories("kaggle/working");
						bool sub_ok = submission.write("kaggle/working/submission.json");
						solver.log_debug(fmt::format(
							"submission.json: {} ({} tasks)",
							sub_ok ? "written" : "WRITE FAILED", submission.count()));
#endif
						app_state.is_benchmarking = false;
						if (!app_state.is_paused) solver.toggle_pause();
					} else {
						app_state.benchmark.last_idx = p_idx;
						solver.change_puzzle(1);
						app_state.benchmark.waiting_for_advance = true;
					}
				}
			}
		} else {
			app_state.benchmark.reset();
		}

		// --- Rendering tail: GUI build only. Uploads solver buffers to the GPU
		// for display and draws the frame. The headless build does none of this;
		// its loop body ends after the benchmark logic above. ---
#ifdef _MSC_VER
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

		// Stats are thread-state reads, not rendering — sampled every loop so
		// they stay live while minimized, which is how the agent runs the app.
		long long current_attempts = state.attempts.load();
		if (current_attempts < last_attempts) last_attempts = 0;

		fps = (float)(1.0 / delta);
		aps = (float)((current_attempts - last_attempts) / delta);
		last_attempts = current_attempts;

		// Attempts count candidates handed to the GPU, so a frame spent
		// inside a running kernel scores zero — that is busy, not idle.
		// The old "- Idle" verdict read that as a stall and was always wrong.
		// gpu is how long ONE batch waits on the kernel. Held over between
		// samples, so a batch longer than a frame reads as its own duration
		// instead of the slice of it this frame happened to catch.
		long long current_gpu_wait = state.gpu_wait_now();
		long long current_gpu_count = state.gpu_wait_count.load();
		if (current_gpu_wait < last_gpu_wait) { last_gpu_wait = 0; last_gpu_count = 0; }
		if (current_gpu_count > last_gpu_count) {
			gpu_ms = (float)((current_gpu_wait - last_gpu_wait) / 1e6
				/ (double)(current_gpu_count - last_gpu_count));
			last_gpu_wait = current_gpu_wait;
			last_gpu_count = current_gpu_count;
		}

		solver.log_debug(fmt::format("[APS] {:.2f}  (Attempts: {}) gpu={:.0f}ms",
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
				lock_guard<mutex> lock(state.mtx);
				draw_frame(&context, chars, bg_grid, state,
					app_state, solver.get_puzzle(p_idx), w, h, fps, aps, gpu_ms, p_idx,
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
#else
		// Headless: solving runs on the background GPU thread; this loop only
		// watches for each puzzle to finish. Sleep ~7ms between polls so we
		// don't burn a CPU core spinning (no effect on solve throughput).
		this_thread::sleep_for(chrono::milliseconds(7));
#endif  // _MSC_VER (rendering tail)
	}

	// Emit the submission. BEHAVIOUR DIFFERS BY BUILD:
	//   - Windows (MSVC): never sharded (no ARC_SHARD_* env) -> always one plain
	//     kaggle/working/submission.json. A direct, complete file.
	//   - Kaggle (headless): the notebook runs N sharded processes, so each writes
	//     submission_<shard_index>.json and the notebook MERGES them into the final
	//     submission.json. Single-shard runs write submission.json directly.
	// *** CROSS-FILE CONTRACT: the "submission_<n>.json" name below MUST match the
	//     merge glob in kaggle/src/arc_build/build.py (the `submission_{s}.json`
	//     loop). Change one, change both, or the merge silently finds nothing. ***
#ifdef _MSC_VER
	cmd_server.shutdown();
	// Final settings flush so the last second of changes is not lost.
	{
		ofstream f(get_settings_path());
		f << serialize_settings(app_state, sound_manager::get().is_muted(), solver.get_solver_state().puzzle_idx.load());
	}
	filesystem::create_directories("kaggle/working");
	std::string out_path = "kaggle/working/submission.json";
	bool ok = submission.write(out_path);
	std::fprintf(stderr, "%s: %s (%zu tasks)\n",
	             out_path.c_str(), ok ? "written" : "WRITE FAILED", submission.count());
#else
	std::string out_path = (shard_total > 1)
		? "/kaggle/working/submission_" + std::to_string(shard_index) + ".json"
		: "/kaggle/working/submission.json";
	bool ok = submission.write(out_path);
	std::fprintf(stderr, "%s: %s (%zu tasks)\n",
	             out_path.c_str(), ok ? "written" : "WRITE FAILED", submission.count());
#endif

	solver.stop_thread();
	return 0;
}
