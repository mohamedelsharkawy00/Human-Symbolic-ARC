// EVERY PUZZLE READ, SCORED AND WRITTEN DOWN, in one run. For every training
// puzzle -- or only the ids named -- infer, write the inference readout, take
// suggested_programs, run each on the train pairs, and where one fits them all,
// ask it for the held-out test answer. No search anywhere. Run it from the repo
// root, through scripts\said.bat, which builds it first:
//   arc_said <label> [id...]
// Everything it writes lands in out/<label>/:
//   said.txt                  a line per puzzle whose program fits the train
//                             pairs: id, SOLVED or WRONG, program
//   tally.txt                 the counts, also on stderr
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
// writes its own readout. Their programs run one puzzle at a time: every run on
// the GPU goes through its one scratch slot, and every program is parsed into
// one pool. The lines come out in the dataset's order, as they did one puzzle at
// a time.
//
// WITHOUT A GPU -- built with ARC_NO_GPU, as the Linux cloud builds it -- it
// runs nothing: it writes the readouts, and no score.
#include "16_solver_beam.hpp"
#include "14_puzzle_loader.hpp"
#ifndef ARC_NO_GPU
#include "20_cuda_solver.cuh"
#endif
#include "in_parallel.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

using namespace arc;
using namespace std;

// What one puzzle came to: whether the reading spoke, and the first program it
// spelled that fits every pair, with whether it drew the test answers too.
struct said_one {
    bool spoke = false, panes = false, fitted = false, right = false;
    string program;
};

#ifndef ARC_NO_GPU
// The first of the programs the reading spelled that draws every train pair,
// and whether it draws the test answers too.
static void run_what_it_said(const puzzle& task, const vector<grid>* answers, said_one& one) {
    for (const string& program : task.suggested_programs) {
        node* tree = environment::parse(program);
        if (!tree) continue;
        const int nodes = environment::count_nodes(tree);
        bool all_fit = true;
        for (const auto& pair : task.train) {
            grid drawn; float score = 0; bool solved = false;
            environment::execute_tree_via_flat(tree, pair, drawn, task, nodes, score, solved);
            if (drawn != pair.output) { all_fit = false; break; }
        }
        if (all_fit) {
            bool every = answers && answers->size() == task.test.size();
            for (size_t i = 0; i < task.test.size() && every; ++i) {
                grid drawn; float score = 0; bool solved = false;
                environment::execute_tree_via_flat(tree, task.test[i], drawn, task, nodes, score, solved);
                every = drawn == (*answers)[i];
            }
            one.fitted = true;
            one.right = every;
            one.program = program;
            environment::remove(tree);
            return;
        }
        environment::remove(tree);
    }
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "arc_said <label> [id...]\n"); return 2; }
    const vector<string> only(argv + 2, argv + argc);
    task_set set = load_task_set(DS_TRAIN);
#ifndef ARC_NO_GPU
    if (!cuda_init()) { fprintf(stderr, "no GPU\n"); return 1; }
#endif
    // Each puzzle asked, by its place in the dataset, which its readout names.
    vector<int> asked;
    for (int at = 0; at < (int)set.tasks.size(); ++at)
        if (only.empty() || ranges::find(only, set.tasks[(size_t)at].id) != only.end()) asked.push_back(at);
    bool found_every_id = true;
    for (const string& id : only)
        if (ranges::none_of(set.tasks, [&](const auto& entry) { return entry.id == id; })) {
            fprintf(stderr, "no such training task: %s\n", id.c_str());
            found_every_id = false;
        }

    // EVERYTHING IT WRITES, IN ONE FOLDER: out/<label>, the readouts in its
    // logs/inference, where log_inference writes them.
    const filesystem::path out = filesystem::path("out") / argv[1];
    filesystem::create_directories(out);
    filesystem::current_path(out);
#ifndef ARC_NO_GPU
    FILE* said_file = fopen("said.txt", "w");
    if (!said_file) { fprintf(stderr, "cannot write %s\n", (out / "said.txt").string().c_str()); return 1; }
#endif

    int spoke = 0, fitted = 0, right = 0, wrong = 0;
    int pane_spoke = 0, pane_fitted = 0, pane_right = 0, pane_wrong = 0;
    vector<said_one> said(asked.size());
#ifndef ARC_NO_GPU
    mutex gpu;
#endif
    in_parallel((int)asked.size(), [&](int which) {
        const int at = asked[(size_t)which];
        const challenge_task& entry = set.tasks[(size_t)at];
        said_one& one = said[(size_t)which];
        puzzle task;
        if (!load_task_from_content(entry.content, entry.id, task, set.solutions_for(entry.id))) return;
        infer_into_puzzle(task, { at, (int)set.tasks.size(), DS_TRAIN }, true);
        if (task.suggested_programs.empty()) return;
        one.spoke = true;
        one.panes = task.inf.pane_split.holds();
#ifndef ARC_NO_GPU
        const lock_guard on_the_gpu(gpu);
        run_what_it_said(task, set.solutions_for(entry.id), one);
#endif
    }, [&](int which) {
        const said_one& one = said[(size_t)which];
        spoke += one.spoke;
        pane_spoke += one.spoke && one.panes;
        if (!one.fitted) return;
        ++fitted;
        pane_fitted += one.panes;
        (one.right ? right : wrong)++;
        if (one.panes) (one.right ? pane_right : pane_wrong)++;
#ifndef ARC_NO_GPU
        const string line = set.tasks[(size_t)asked[(size_t)which]].id + "\t" + (one.right ? "SOLVED" : "WRONG") +
                            "\t" + one.program + "\n";
        fputs(line.c_str(), stdout);
        fputs(line.c_str(), said_file);
#endif
    });

#ifdef ARC_NO_GPU
    fprintf(stderr, "reading spoke: %d | no score: built without a GPU, readouts only\n", spoke);
#else
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
#endif
    fprintf(stderr, "written to %s\n", out.string().c_str());
    return found_every_id ? 0 : 2;
}
