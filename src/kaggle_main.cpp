// THE SOLVER ON ITS OWN. No window, no thread, nothing from include/gui --
// this program is built with include/core alone, so it cannot reach anything
// else by accident.
//
// EVERY PUZZLE IN A FRESH PROCESS. A solver that has worked through one puzzle
// carries its memory into the next, and runs after the first went measurably
// worse than the same puzzle run first in a clean process. So a run over many
// puzzles is a parent that starts this same program once per puzzle, and
// never touches the GPU itself.
//
//   arc_headless
//       The Kaggle run: a shard of a dataset, one child per puzzle, merged
//       into a submission. Environment, set by kaggle/src/arc_build/build.py:
//         ARC_DATASET       test | eval | train | custom. Required, with no
//                           default ON PURPOSE: guessing wrong writes a
//                           submission built from the training set that looks
//                           exactly like a real one, and no one is watching.
//         ARC_TIMEOUT       seconds per puzzle (default 120).
//         ARC_PUZZLE_LIMIT  only the first N tasks overall (0 or unset = all).
//         ARC_SHARD_INDEX / ARC_SHARD_TOTAL
//                           which contiguous slice this process solves, one
//                           process per GPU. The last shard takes what is left.
//       Writes kaggle/working/submission.json under the data root, or
//       submission_<shard>.json when there is more than one shard.
//       *** CROSS-FILE CONTRACT: the "submission_<n>.json" name MUST match the
//           merge in kaggle/src/arc_build/build.py. Change one, change both,
//           or the merge silently finds nothing. ***
//
//   arc_headless --task ID --datasets MASK --timeout SECONDS --out FILE
//       One puzzle, searched until solved or out of time, its result written
//       to FILE (a puzzle_result). What the run above starts per puzzle. MASK
//       is the dataset_flag bits to find ID in.
//
//   arc_headless --task ID --datasets MASK --serve [--timeout S --out FILE]
//       One puzzle, watched: the search streams on stdout and takes commands
//       on stdin (22_search_stream.hpp). Every search the GUI shows is one of
//       these. Without --timeout it waits paused for "run" and goes until
//       told to quit or stdin closes; with it (a benchmark) it runs at once
//       and stops like the one above.
//
//   Either takes: --logs (write logs/<id>.txt and the inference readout),
//   --verbose (debug lines in them too), --max-attempts N (stop, paused, after
//   N candidates), --start-examples N (judge on N examples from the first).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// In number order, so the files are read in the order they are numbered.
// 10_log is here for that alone: main calls nothing in it, but without it
// 9_sentences and 10_log would be read late, when 14_puzzle_loader includes them.
#include "10_log.hpp"
#include "11_text_format.hpp"
#include "16_solver_beam.hpp"
#include "20_cuda_solver.cuh"
#include "21_submission.hpp"
#include "22_search_stream.hpp"

using namespace std;
using namespace arc;

static double env_number(const char *name, double fallback) {
	const char *text = getenv(name);
	return text && *text ? atof(text) : fallback;
}

// The value after `flag` on the command line, or empty.
static string arg_after(int argc, char **argv, const string &flag) {
	for (int i = 1; i + 1 < argc; ++i)
		if (argv[i] == flag) return argv[i + 1];
	return {};
}

static bool has_arg(int argc, char **argv, const string &flag) {
	for (int i = 1; i < argc; ++i)
		if (argv[i] == flag) return true;
	return false;
}

// Where this program is, to start it again. Read before anything changes the
// current directory, since argv[0] may be relative to it.
static filesystem::path own_path(const char *argv0) {
	error_code no_proc;
	filesystem::path linux_self = filesystem::canonical("/proc/self/exe", no_proc);
	return no_proc ? filesystem::absolute(argv0) : linux_self;
}

// A command line for std::system, every part quoted. Windows hands it to
// cmd /c, which strips the outer pair of quotes when the line starts with one,
// so there it gets one more pair to lose.
static string shell_line(const vector<string> &parts) {
	string line;
	for (const string &part : parts)
		line += (line.empty() ? "\"" : " \"") + part + "\"";
#ifdef _WIN32
	line = "\"" + line + "\"";
#endif
	return line;
}

// What --serve is told on stdin, read on a thread of its own so the search
// never waits on it. Never freed: the reader may still be blocked on stdin as
// the process exits, and must not find this gone.
struct command_inbox {
	mutex m;
	deque<string> lines;
	bool push(string line) {
		lock_guard lock(m);
		lines.push_back(std::move(line));
		return true;
	}
	deque<string> take() {
		lock_guard lock(m);
		deque<string> out;
		out.swap(lines);
		return out;
	}
};

// ONE PUZZLE: load it, step until it is solved or its time is up (or, served,
// until told to quit), hand back what it found.
static int solve_one(int argc, char **argv) {
	const string id = arg_after(argc, argv, "--task");
	const int flags = atoi(arg_after(argc, argv, "--datasets").c_str());
	const double timeout = atof(arg_after(argc, argv, "--timeout").c_str());
	const string out_path = arg_after(argc, argv, "--out");
	const bool serve = has_arg(argc, argv, "--serve");
	const bool inference_only = has_arg(argc, argv, "--infer-only");
	const bool timed = timeout > 0;
	if (id.empty() || flags == 0 ||
	    (inference_only ? (serve || out_path.empty()) :
	     (timed != !out_path.empty() || (!serve && !timed)))) {
		fprintf(stderr, "--task needs --datasets and --out with --infer-only or --timeout; --serve is interactive\n");
		return 1;
	}

	const task_set tasks = load_task_set(flags);
	auto found = find_if(tasks.tasks.begin(), tasks.tasks.end(),
	                     [&](const challenge_task &t) { return t.id == id; });
	if (found == tasks.tasks.end()) {
		fprintf(stderr, "%s: no such task in datasets %d\n", id.c_str(), flags);
		return 1;
	}
	if (!cuda_init()) {
		fprintf(stderr, "no usable GPU, nothing solved\n");
		return 1;
	}
	if (inference_only) {
		// Same selection as arc_said: only programs emitted by inference, and
		// only the first that draws every training output exactly. No beam.
		const auto started = chrono::steady_clock::now();
		puzzle p;
		const auto* answers = tasks.solutions_for(id);
		if (!load_task_from_content(found->content, id, p, answers)) return 1;
		infer_into_puzzle(p);
		string accepted;
		for (const string& program : p.suggested_programs) {
			node* tree = environment::parse(program);
			if (!tree) continue;
			const int nodes = environment::count_nodes(tree);
			bool fits = !p.train.empty();
			for (const auto& pair : p.train) {
				grid drawn; float score = 0; bool solved = false;
				environment::execute_tree_via_flat(tree, pair, drawn, p, nodes, score, solved);
				if (drawn != pair.output) { fits = false; break; }
			}
			if (fits) {
				p.best_tree = tree;
				p.best_node_count = nodes;
				p.is_solved = true;
				accepted = program;
				break;
			}
			environment::remove(tree);
		}
		bool right = p.is_solved && answers && answers->size() == p.test.size();
		size_t test_index = 0;
		puzzle_result result;
		result.id = id;
		result.solved = p.is_solved;
		result.entry = submission_line(p, [&](const example& ex) {
			grid drawn = ex.input;
			if (p.best_tree) {
				float score = 0; bool solved = false;
				environment::execute_tree_via_flat(p.best_tree, ex, drawn, p,
				                                 p.best_node_count, score, solved);
			}
			if (right && drawn != (*answers)[test_index]) right = false;
			++test_index;
			return drawn;
		});
		result.test_solved = right;
		result.elapsed = chrono::duration<double>(chrono::steady_clock::now() - started).count();
		const char* verdict = !p.is_solved ? "UNSOLVED" : !answers ? "FIT" : right ? "SOLVED" : "WRONG";
		result.text = id + "\t" + verdict + "\t" + accepted + "\nMode: inference only; search attempts: 0\n";
		const bool ok = result.write(out_path);
		environment::remove(p.best_tree);
		p.best_tree = nullptr;
		cuda_runtime_shutting_down();
		return ok ? 0 : 1;
	}

	solver_options options;
	options.write_logs = has_arg(argc, argv, "--logs");
	beam_solver solver(options);
	solver.set_verbose(has_arg(argc, argv, "--verbose"));
	search_stream_writer stream(stdout);
	if (serve) solver.set_observer(&stream);
	const int idx = (int)(found - tasks.tasks.begin());
	if (!solver.load(*found, tasks.solutions_for(id), {idx, (int)tasks.tasks.size(), flags})) {
		fprintf(stderr, "%s: could not be read\n", id.c_str());
		return 1;
	}
	// A load puts both back to their defaults, so they are set after it.
	if (long long n = atoll(arg_after(argc, argv, "--max-attempts").c_str()); n > 0)
		solver.set_stop_after_attempts(n);
	if (int n = atoi(arg_after(argc, argv, "--start-examples").c_str()); n > 0)
		solver.set_start_example_count(n);

	command_inbox *inbox = new command_inbox;
	if (serve)
		thread([inbox] {
			string line;
			while (getline(cin, line)) {
				if (!line.empty() && line.back() == '\r') line.pop_back();
				inbox->push(line);
			}
			// Whoever was talking to it has gone, so it goes too.
			inbox->push("quit");
		}).detach();

	// A watched search waits to be told to run; a timed one runs at once.
	// The clock stops while it waits, so time searched is what it shows.
	search_progress &progress = solver.progress();
	bool running = !serve || timed;
	auto paused_since = chrono::steady_clock::now();
	auto elapsed = [&] {
		auto upto = running ? chrono::steady_clock::now() : paused_since;
		return chrono::duration<double>(upto - progress.start_time).count();
	};
	auto pause = [&] {
		if (!running) return;
		running = false;
		paused_since = chrono::steady_clock::now();
		stream.tick(solver, false, elapsed(), true);
	};

	for (bool quit = false; !quit;) {
		for (const string &command : inbox->take()) {
			istringstream in(command);
			string verb;
			in >> verb;
			if (verb == "quit") {
				quit = true;
			} else if (verb == "run" && !running) {
				progress.start_time += chrono::steady_clock::now() - paused_since;
				running = true;
			} else if (verb == "pause") {
				pause();
			} else if (verb == "infer") {
				// The pane is open: read the puzzle and try what the reading
				// spelled, so one that solves that way shows as solved unrun.
				solver.infer();
			} else if (verb == "verbose") {
				int on = 0;
				in >> on;
				solver.set_verbose(on != 0);
			} else if (verb == "import") {
				string program;
				getline(in >> ws, program);
				solver.apply_program(program);
			} else if (verb == "preview") {
				unsigned long long hash = 0;
				string program;
				in >> hash;
				getline(in >> ws, program);
				solver.read_facts();
				if (node *target = environment::parse(program)) {
					stream.previewed(hash, preview_of(solver.get_puzzle(), target));
					environment::remove(target);
				}
			}
		}
		if (quit) break;
		if (timed && (solver.get_puzzle().is_solved || elapsed() >= timeout)) break;

		if (running) {
			solver.step();
			// A SOLVED PUZZLE DOES NOT SEARCH ON ITS OWN, and neither does one
			// that just got shorter: a person unpauses it to hunt for shorter
			// still. A timed run stops on its solve instead.
			const search_news news = stream.take_news();
			if (serve && !timed && (news.train_solved || news.simplified)) pause();
			// One-shot: cleared as it fires, so unpausing by hand does not re-stop.
			const long long stop_at = solver.get_stop_after_attempts();
			if (stop_at > 0 && progress.attempts.load() >= stop_at) {
				solver.set_stop_after_attempts(0);
				pause();
				stream.stopped_at(progress.attempts.load());
			}
		} else {
			this_thread::sleep_for(chrono::milliseconds(5));
		}
		if (serve) stream.tick(solver, running, elapsed());
	}
	if (serve) stream.tick(solver, running, elapsed(), true);

	bool ok = true;
	if (!out_path.empty())
		ok = puzzle_result::of(solver.get_puzzle(), elapsed(), progress.attempts.load())
		         .write(out_path);
	// Said before the solver's buffers are freed on the way out, so the frees
	// that would reach a runtime already shutting down are skipped.
	cuda_runtime_shutting_down();
	return ok ? 0 : 1;
}

// THE KAGGLE RUN: this shard's puzzles, each in a child of this program.
static int run_shard(const filesystem::path &self) {
	const char *dataset = getenv("ARC_DATASET");
	const int flags = dataset ? dataset_from_name(dataset) : 0;
	if (flags == 0) {
		fprintf(stderr, "ARC_DATASET must be test, eval, train or custom. Refusing to guess.\n");
		return 1;
	}
	const double timeout = env_number("ARC_TIMEOUT", 120.0);
	const bool inference_only = env_number("ARC_INFERENCE_ONLY", 0) != 0;
	const int limit = (int)env_number("ARC_PUZZLE_LIMIT", 0);
	const int shard_index = (int)env_number("ARC_SHARD_INDEX", 0);
	const int shard_total = max(1, (int)env_number("ARC_SHARD_TOTAL", 1));

	const task_set tasks = load_task_set(flags);
	int total = (int)tasks.tasks.size();
	if (limit > 0) total = min(total, limit);
	const int chunk = (total + shard_total - 1) / shard_total;  // ceil
	const int first = min(total, shard_index * chunk);
	const int last = min(total, first + chunk);
	fprintf(stderr, "shard %d/%d: tasks [%d, %d) of %d\n",
	        shard_index, shard_total, first, last, total);
	if (inference_only) fprintf(stderr, "MODE: inference only, no search timeout\n");
	else fprintf(stderr, "MODE: search, %.0fs each\n", timeout);

	const filesystem::path scratch =
		filesystem::temp_directory_path() / ("arc_shard_" + to_string(shard_index));
	filesystem::create_directories(scratch);

	submission_writer submission;
	int run = 0, train_solved = 0, test_solved = 0;
	for (int i = first; i < last; ++i) {
		const challenge_task &task = tasks.tasks[i];
		const string result_path = (scratch / (task.id + ".txt")).string();
		const auto started = chrono::steady_clock::now();
		vector<string> command{self.string(), "--task", task.id, "--datasets",
		                       to_string(flags), "--out", result_path};
		if (inference_only) command.push_back("--infer-only");
		else { command.push_back("--timeout"); command.push_back(to_string(timeout)); }
		const int status = system(shell_line(command).c_str());
		if (status != 0) fprintf(stderr, "%s: its run exited with %d\n", task.id.c_str(), status);
		const double took = chrono::duration<double>(chrono::steady_clock::now() - started).count();

		// A child that died hands back nothing, but its task still needs its
		// line in the submission.
		puzzle_result result;
		if (auto handed_back = puzzle_result::read(result_path)) {
			result = *handed_back;
		} else {
			fprintf(stderr, "%s: its run handed back nothing, inputs echoed\n", task.id.c_str());
			result = puzzle_result::nothing_from(task, took);
		}
		error_code gone;
		filesystem::remove(result_path, gone);

		submission.add(result.entry);
		++run;
		train_solved += result.solved;
		test_solved += result.test_solved;
		fprintf(stderr, "%s\n", result.text.c_str());
	}
	fprintf(stderr, "SOLVE SUMMARY: %d puzzles, train solved %d, test solved %d\n",
	        run, train_solved, test_solved);

	filesystem::create_directories("kaggle/working");
	const string out_path = shard_total > 1
		? "kaggle/working/submission_" + to_string(shard_index) + ".json"
		: "kaggle/working/submission.json";
	const bool ok = submission.write(out_path);
	fprintf(stderr, "%s: %s (%zu tasks)\n", out_path.c_str(), ok ? "written" : "WRITE FAILED",
	        submission.count());
	return ok ? 0 : 1;
}

int main(int argc, char **argv) {
	const filesystem::path self = own_path(argv[0]);

	// Wherever this was started from, the data is found by walking up to the
	// folder holding kaggle/input: "/" on Kaggle, the repo on a desk.
	filesystem::path root = find_data_root(filesystem::current_path());
	if (root.empty()) {
		fprintf(stderr, "no kaggle/input above %s\n", filesystem::current_path().string().c_str());
		return 1;
	}
	filesystem::current_path(root);

	return has_arg(argc, argv, "--task") ? solve_one(argc, argv) : run_shard(self);
}
