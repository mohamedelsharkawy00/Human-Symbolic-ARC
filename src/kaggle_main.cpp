// THE SOLVER ON ITS OWN. No window, no thread, nothing from include/gui --
// this program is built with include/core alone, so it cannot reach anything
// else by accident.
//
// EVERY PUZZLE IN A FRESH PROCESS. A solver that has worked through one puzzle
// carries its memory into the next, and runs after the first went measurably
// worse than the same puzzle run first in a clean process. So a run over many
// puzzles is a parent that starts this same program once per puzzle, and
// reads none itself.
//
//   arc_headless
//       The Kaggle run: a shard of a dataset, one child per puzzle, merged
//       into a submission. Environment, set by kaggle/src/arc_build/build.py:
//         ARC_DATASET       test | eval | train | custom. Required, with no
//                           default ON PURPOSE: guessing wrong writes a
//                           submission built from the training set that looks
//                           exactly like a real one, and no one is watching.
//         ARC_PUZZLE_LIMIT  only the first N tasks overall (0 or unset = all).
//         ARC_SHARD_INDEX / ARC_SHARD_TOTAL
//                           which contiguous slice this process solves, one
//                           process per CPU thread. The last shard takes what
//                           is left.
//       Writes kaggle/working/submission.json under the data root, or
//       submission_<shard>.json when there is more than one shard.
//       *** CROSS-FILE CONTRACT: the "submission_<n>.json" name MUST match the
//           merge in kaggle/src/arc_build/build.py. Change one, change both,
//           or the merge silently finds nothing. ***
//
//   arc_headless --task ID --datasets MASK --out FILE
//       One puzzle, read: the first program its reading spells that draws
//       every training output, its result written to FILE (a puzzle_result).
//       What the run above starts per puzzle. MASK is the dataset_flag bits
//       to find ID in.
//
//   arc_headless --task ID --datasets MASK --serve [--out FILE]
//       One puzzle, watched: the reading streams on stdout and it takes
//       commands on stdin (22_search_stream.hpp). Every puzzle the GUI shows is
//       one of these. Without --out it waits for "infer" and goes until told
//       to quit or stdin closes; with it (a benchmark) it reads at once,
//       writes FILE and stops. It also takes --logs (write logs/<id>.txt and
//       the inference readout) and --verbose (debug lines in them too).
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
#include "16_solver.hpp"
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

// What --serve is told on stdin, read on a thread of its own so the reading
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

// ONE PUZZLE: load it, read it, hand back what the reading came to (or, served,
// read it when told to, until told to quit).
static int solve_one(int argc, char **argv) {
	const string id = arg_after(argc, argv, "--task");
	const int flags = atoi(arg_after(argc, argv, "--datasets").c_str());
	const string out_path = arg_after(argc, argv, "--out");
	const bool serve = has_arg(argc, argv, "--serve");
	if (id.empty() || flags == 0 || (!serve && out_path.empty())) {
		fprintf(stderr, "--task needs --datasets, and --out unless it is --serve\n");
		return 1;
	}

	const task_set tasks = load_task_set(flags);
	auto found = find_if(tasks.tasks.begin(), tasks.tasks.end(),
	                     [&](const challenge_task &t) { return t.id == id; });
	if (found == tasks.tasks.end()) {
		fprintf(stderr, "%s: no such task in datasets %d\n", id.c_str(), flags);
		return 1;
	}
	if (!serve) {
		// Same selection as arc_said: only programs emitted by inference, and
		// only the first that draws every training output exactly.
		const auto started = chrono::steady_clock::now();
		puzzle p;
		const auto* answers = tasks.solutions_for(id);
		if (!load_task_from_content(found->content, id, p, answers)) return 1;
		infer_into_puzzle(p);
		string accepted;
		for (const string& program : p.suggested_programs) {
			node* tree = environment::parse(program);
			if (!tree) continue;
			bool fits = !p.train.empty();
			for (const auto& pair : p.train)
				if (environment::draw(tree, pair, p) != pair.output) { fits = false; break; }
			if (fits) {
				p.best_tree = tree;
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
			const grid drawn = p.best_tree ? environment::draw(p.best_tree, ex, p) : ex.input;
			if (right && drawn != (*answers)[test_index]) right = false;
			++test_index;
			return drawn;
		});
		result.test_solved = right;
		result.elapsed = chrono::duration<double>(chrono::steady_clock::now() - started).count();
		const char* verdict = !p.is_solved ? "UNSOLVED" : !answers ? "FIT" : right ? "SOLVED" : "WRONG";
		result.text = id + "\t" + verdict + "\t" + accepted + "\n";
		const bool ok = result.write(out_path);
		environment::remove(p.best_tree);
		p.best_tree = nullptr;
		return ok ? 0 : 1;
	}

	solver_options options;
	options.write_logs = has_arg(argc, argv, "--logs");
	solver puzzle_solver(options);
	puzzle_solver.set_verbose(has_arg(argc, argv, "--verbose"));
	search_stream_writer stream(stdout);
	puzzle_solver.set_observer(&stream);
	const int idx = (int)(found - tasks.tasks.begin());
	if (!puzzle_solver.load(*found, tasks.solutions_for(id), {idx, (int)tasks.tasks.size(), flags})) {
		fprintf(stderr, "%s: could not be read\n", id.c_str());
		return 1;
	}

	// A benchmark reads at once, says what it came to, and stops.
	if (!out_path.empty()) {
		const auto started = chrono::steady_clock::now();
		puzzle_solver.infer();
		stream.tick(puzzle_solver, true);
		const double elapsed = chrono::duration<double>(chrono::steady_clock::now() - started).count();
		const bool ok = puzzle_result::of(puzzle_solver.get_puzzle(), elapsed).write(out_path);
		return ok ? 0 : 1;
	}

	command_inbox *inbox = new command_inbox;
	thread([inbox] {
		string line;
		while (getline(cin, line)) {
			if (!line.empty() && line.back() == '\r') line.pop_back();
			inbox->push(line);
		}
		// Whoever was talking to it has gone, so it goes too.
		inbox->push("quit");
	}).detach();

	for (bool quit = false; !quit;) {
		for (const string &command : inbox->take()) {
			istringstream in(command);
			string verb;
			in >> verb;
			if (verb == "quit") {
				quit = true;
			} else if (verb == "infer") {
				// SPACE, or the pane opening: read the puzzle and try what the
				// reading spelled.
				puzzle_solver.infer();
			} else if (verb == "verbose") {
				int on = 0;
				in >> on;
				puzzle_solver.set_verbose(on != 0);
			} else if (verb == "import") {
				string program;
				getline(in >> ws, program);
				puzzle_solver.apply_program(program);
			} else if (verb == "preview") {
				unsigned long long hash = 0;
				string program;
				in >> hash;
				getline(in >> ws, program);
				puzzle_solver.read_facts();
				stream.preview(puzzle_solver.get_puzzle(), program, hash);
			}
		}
		if (quit) break;
		stream.tick(puzzle_solver);
		this_thread::sleep_for(chrono::milliseconds(5));
	}
	stream.tick(puzzle_solver, true);
	return 0;
}

// THE KAGGLE RUN: this shard's puzzles, each in a child of this program.
static int run_shard(const filesystem::path &self) {
	const char *dataset = getenv("ARC_DATASET");
	const int flags = dataset ? dataset_from_name(dataset) : 0;
	if (flags == 0) {
		fprintf(stderr, "ARC_DATASET must be test, eval, train or custom. Refusing to guess.\n");
		return 1;
	}
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

	const filesystem::path scratch =
		filesystem::temp_directory_path() / ("arc_shard_" + to_string(shard_index));
	filesystem::create_directories(scratch);

	submission_writer submission;
	int run = 0, train_solved = 0, test_solved = 0;
	for (int i = first; i < last; ++i) {
		const challenge_task &task = tasks.tasks[i];
		const string result_path = (scratch / (task.id + ".txt")).string();
		const auto started = chrono::steady_clock::now();
		const vector<string> command{self.string(), "--task", task.id, "--datasets",
		                             to_string(flags), "--out", result_path};
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
