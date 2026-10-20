// EVERY PUZZLE READ, SCORED AND WRITTEN DOWN, in one run. For every training
// puzzle -- or, with ARC_DATASET=eval (as arc_headless names the sets), every
// evaluation puzzle -- or only the ids named -- infer, write the inference readout, take
// suggested_programs, run each on the train pairs, and where one fits them all,
// ask it for the held-out test answer. No search anywhere. Run it from the repo
// root, through scripts\said.bat, which builds it first:
//   arc_said [--as <label>] [--solved [label]] [id...]
// Everything it writes lands beside the program (out/x64-Release),
// or in out/<label>/ when a run is named to be compared with another:
//   said.txt                  a line per puzzle whose program fits the train
//                             pairs: id, SOLVED or WRONG, program
//   tally.txt                 the counts, also on stderr
//   timings.csv               elapsed milliseconds and work counts per puzzle
//   logs/inference/<id>.txt   the readout: background, readings, fired rules,
//                             sentences (log_inference writes it)
// scripts\readouts.py compares two labels.
//
// This is the guard on say_the_reading. A program it spells that draws the
// wrong thing shows up here as "WRONG" and nowhere else, so run it after
// touching the emitter.
//
// EVERY PUZZLE THE READING SPEAKS FOR, not only the ones with panes. The count
// was gated on pane_split while the pane emitter was the new thing, and that
// gate hid every other branch of say_the_reading from the only tool that runs
// what it says. The panes are still tallied apart.
//
// THE PUZZLES ARE READ AT ONCE, one per CPU thread (in_parallel.hpp). Each
// writes its own readout and runs its own programs. The lines come out in the
// dataset's order, as they did one puzzle at a time.
#include "16_solver.hpp"
#include "14_puzzle_loader.hpp"
#include "in_parallel.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace arc;
using namespace std;

// What one puzzle came to: whether the reading spoke, and the first program it
// spelled that fits every pair, with whether it drew the test answers too.
struct said_one {
    bool spoke = false, panes = false, fitted = false, right = false;
    string program;
    bool loaded = false;
    double total_ms = 0, infer_ms = 0, report_ms = 0, run_ms = 0;
    inference_work_counts work;
    matching_work_counts matching;
    inference_times times;
};

using run_clock = chrono::steady_clock;
static double milliseconds_since(run_clock::time_point start) {
    return chrono::duration<double, milli>(run_clock::now() - start).count();
}

// The first of the programs the reading spelled that draws every train pair,
// and whether it draws the test answers too.
static void run_what_it_said(const puzzle& task, const vector<grid>* answers, said_one& one) {
    for (const string& program : task.suggested_programs) {
        node* tree = environment::parse(program);
        if (!tree) continue;
        bool all_fit = true;
        for (const auto& pair : task.train)
            if (environment::draw(tree, pair, task) != pair.output) { all_fit = false; break; }
        if (all_fit) {
            bool every = answers && answers->size() == task.test.size();
            for (size_t i = 0; i < task.test.size() && every; ++i)
                every = environment::draw(tree, task.test[i], task) == (*answers)[i];
            one.fitted = true;
            one.right = every;
            one.program = program;
            environment::remove(tree);
            return;
        }
        environment::remove(tree);
    }
}

int main(int argc, char** argv) {
    // A name only when asked for: the run is kept apart to compare against.
    // --solved [label] reads only the IDs marked SOLVED in out/<label>/said.txt
    // (defaulting to out/x64-Release/said.txt).
    vector<string> args(argv + 1, argv + argc);
    string label;
    bool only_solved = false;
    string solved_from = "x64-Release";
    vector<string> only;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--as") {
            if (i + 1 >= args.size()) { fprintf(stderr, "arc_said [--as <label>] [--solved [label]] [id...]\n"); return 2; }
            label = args[++i];
        } else if (args[i] == "--solved") {
            only_solved = true;
            if (i + 1 < args.size() && !args[i + 1].starts_with("-")) {
                solved_from = args[++i];
            }
        } else {
            only.push_back(args[i]);
        }
    }
    if (only_solved) {
        filesystem::path path = filesystem::path("out") / solved_from / "said.txt";
        if (!filesystem::exists(path)) path = filesystem::absolute(argv[0]).parent_path() / "said.txt";
        FILE* f = fopen(path.string().c_str(), "r");
        if (!f) {
            fprintf(stderr, "cannot read %s to find solved puzzles\n", path.string().c_str());
            return 1;
        }
        char line[2048];
        while (fgets(line, sizeof(line), f)) {
            char* p = line;
            if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
                p += 3;
            char id[64] = {}, verdict[64] = {};
            if (sscanf(p, "%63s\t%63s", id, verdict) == 2 && strcmp(verdict, "SOLVED") == 0) {
                only.push_back(id);
            }
        }
        fclose(f);
        if (only.empty()) {
            fprintf(stderr, "no SOLVED puzzles found in %s\n", path.string().c_str());
            return 1;
        }
    }
    // Training unless ARC_DATASET names the evaluation set.
    const char* named = getenv("ARC_DATASET");
    const int dataset = named ? dataset_from_name(named) : DS_TRAIN;
    if (dataset != DS_TRAIN && dataset != DS_EVAL) { fprintf(stderr, "ARC_DATASET must be train or eval\n"); return 2; }
    task_set set = load_task_set(dataset);
    // Each puzzle asked, by its place in the dataset, which its readout names.
    vector<int> asked;
    for (int at = 0; at < (int)set.tasks.size(); ++at)
        if (only.empty() || ranges::find(only, set.tasks[(size_t)at].id) != only.end()) asked.push_back(at);
    bool found_every_id = true;
    for (const string& id : only)
        if (ranges::none_of(set.tasks, [&](const auto& entry) { return entry.id == id; })) {
            fprintf(stderr, "no such task: %s\n", id.c_str());
            found_every_id = false;
        }

    // EVERYTHING IT WRITES, IN ONE FOLDER: the program's own, or out/<label>,
    // the readouts in its logs/inference, where log_inference writes them.
    const filesystem::path out = label.empty()
        ? filesystem::absolute(argv[0]).parent_path()
        : filesystem::path("out") / label;
    filesystem::create_directories(out);
    filesystem::current_path(out);
    const bool audit_preparations = getenv("ARC_PREP_AUDIT") != nullptr;
    if (audit_preparations) filesystem::create_directories("preparations");
    FILE* timings = fopen("timings.csv", "w");
    if (!timings) { fprintf(stderr, "cannot write timings.csv\n"); return 1; }
    fputs("id,loaded,total_ms,infer_ms,report_ms,run_ms,preparations,readings,later_readings,lines,paint_replays,slide_replays,drop_replays,walks,lays,matching_calls,matching_rematches,matching_gathers,fixed_scans,whole_grid_scans,placement_cache_hits,fitting_edges", timings);
    for (const char* name : inference_times::names) fprintf(timings, ",%s_ms", name);
    fputs("\n", timings);
    fflush(timings);
    FILE* said_file = fopen("said.txt", "w");
    if (!said_file) { fprintf(stderr, "cannot write %s\n", (out / "said.txt").string().c_str()); return 1; }

    int spoke = 0, fitted = 0, right = 0, wrong = 0;
    int pane_spoke = 0, pane_fitted = 0, pane_right = 0, pane_wrong = 0;
    vector<said_one> said(asked.size());
    in_parallel((int)asked.size(), [&](int which) {
        const int at = asked[(size_t)which];
        const challenge_task& entry = set.tasks[(size_t)at];
        said_one& one = said[(size_t)which];
        const auto started = run_clock::now();
        puzzle task;
        one.loaded = load_task_from_content(entry.content, entry.id, task, set.solutions_for(entry.id));
        if (!one.loaded) { one.total_ms = milliseconds_since(started); return; }
        inference_work = {};
        matching_work = {};
        inference_time = {};
        struct preparation_record {
            string key;
            size_t first;
            double ms;
            unsigned long long calls, edges;
        };
        vector<preparation_record> preparations;
        if (audit_preparations) preparation_audit = [&](const vector<example>& pairs,
                const tiled_pieces& tiles, double ms, unsigned long long calls, unsigned long long edges) {
            string key = figures_key(pairs, tiles);
            size_t first = 0;
            while (first < preparations.size() && preparations[first].key != key) ++first;
            preparations.push_back({ std::move(key), first, ms, calls, edges });
        };
        const auto reading = run_clock::now();
        inference read = infer(task.train);
        one.infer_ms = milliseconds_since(reading);
        preparation_audit = {};
        if (audit_preparations) {
            const string path = "preparations/" + entry.id + ".csv";
            if (FILE* audit = fopen(path.c_str(), "w")) {
                fputs("preparation,first_identical,elapsed_ms,matching_calls,fitting_edges\n", audit);
                for (size_t i = 0; i < preparations.size(); ++i) {
                    const auto& row = preparations[i];
                    fprintf(audit, "%zu,%zu,%.3f,%llu,%llu\n", i, row.first, row.ms, row.calls, row.edges);
                }
                fclose(audit);
            } else fprintf(stderr, "cannot write preparation audit: %s\n", path.c_str());
        }
        one.work = inference_work;
        one.matching = matching_work;
        one.times = inference_time;
        const auto reporting = run_clock::now();
        take_the_reading(task, std::move(read), { at, (int)set.tasks.size(), dataset }, true);
        one.report_ms = milliseconds_since(reporting);
        one.spoke = !task.suggested_programs.empty();
        one.panes = task.inf.pane_split.holds();
        if (one.spoke) {
            const auto running = run_clock::now();
            run_what_it_said(task, set.solutions_for(entry.id), one);
            one.run_ms = milliseconds_since(running);
        }
        one.total_ms = milliseconds_since(started);
    }, [&](int which) {
        const said_one& one = said[(size_t)which];
        const auto& work = one.work;
        const auto& matching = one.matching;
        fprintf(timings, "%s,%d,%.3f,%.3f,%.3f,%.3f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
                set.tasks[(size_t)asked[(size_t)which]].id.c_str(), (int)one.loaded,
                one.total_ms, one.infer_ms, one.report_ms, one.run_ms,
                work.preparations, work.readings, work.later_readings, work.lines,
                work.paint_replays, work.slide_replays, work.drop_replays, work.walks, work.lays,
                matching.calls, matching.rematches, matching.gathers, matching.fixed_scans,
                matching.whole_grid_scans, matching.placement_cache_hits, matching.fitting_edges);
        for (const double ms : one.times.ms) fprintf(timings, ",%.3f", ms);
        fputs("\n", timings);
        fflush(timings);
        spoke += one.spoke;
        pane_spoke += one.spoke && one.panes;
        if (!one.fitted) return;
        ++fitted;
        pane_fitted += one.panes;
        (one.right ? right : wrong)++;
        if (one.panes) (one.right ? pane_right : pane_wrong)++;
        const string line = set.tasks[(size_t)asked[(size_t)which]].id + "\t" + (one.right ? "SOLVED" : "WRONG") +
                            "\t" + one.program + "\n";
        fputs(line.c_str(), stdout);
        fputs(line.c_str(), said_file);
    });
    fclose(timings);

    fclose(said_file);
    char tally[256];
    snprintf(tally, sizeof tally,
             "reading spoke: %d | fits train: %d | test right: %d | test wrong: %d\n"
             "  of those, the ones with panes: %d | fits train: %d | test right: %d | test wrong: %d\n",
             spoke, fitted, right, wrong, pane_spoke, pane_fitted, pane_right, pane_wrong);
    fputs(tally, stderr);
    if (FILE* tally_file = fopen("tally.txt", "w")) {
        fputs(tally, tally_file);
        fclose(tally_file);
    }
    fprintf(stderr, "written to %s\n", out.string().c_str());
    return found_every_id ? 0 : 2;
}
