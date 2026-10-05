// WHAT THE READING SOLVES ON ITS OWN, counted. For every training puzzle:
// infer, take suggested_programs, run each on the train pairs, and where one
// fits them all, ask it for the held-out test answer. No search anywhere.
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
// THE PUZZLES ARE READ AT ONCE, one per CPU thread (in_parallel.hpp). Their
// programs run one puzzle at a time: every run on the GPU goes through its one
// scratch slot, and every program is parsed into one pool. The lines come out
// in the dataset's order, as they did one puzzle at a time.
//
//   arc_said [task-id] one puzzle or every puzzle, then a tally on stderr
#include "16_solver_beam.hpp"
#include "14_puzzle_loader.hpp"
#include "20_cuda_solver.cuh"
#include "in_parallel.hpp"
#include <cstdio>
#include <mutex>
#include <string>

using namespace arc;
using namespace std;

// What one puzzle came to: whether the reading spoke, and the first program it
// spelled that fits every pair, with whether it drew the test answers too.
struct said_one {
    bool spoke = false, panes = false, fitted = false, right = false;
    string program;
};

int main(int argc, char** argv) {
    if (argc > 2) { fprintf(stderr, "arc_said [task-id]\n"); return 2; }
    const string requested_id = argc == 2 ? argv[1] : "";
    if (!cuda_init()) { fprintf(stderr, "no GPU\n"); return 1; }
    task_set set = load_task_set(DS_TRAIN);
    vector<const challenge_task*> asked;
    for (const auto& entry : set.tasks)
        if (requested_id.empty() || entry.id == requested_id) asked.push_back(&entry);
    const bool found_requested = requested_id.empty() || !asked.empty();
    int spoke = 0, fitted = 0, right = 0, wrong = 0;
    int pane_spoke = 0, pane_fitted = 0, pane_right = 0, pane_wrong = 0;
    vector<said_one> said(asked.size());
    mutex gpu;
    in_parallel((int)asked.size(), [&](int at) {
        const challenge_task& entry = *asked[(size_t)at];
        said_one& one = said[(size_t)at];
        puzzle task;
        if (!load_task_from_content(entry.content, entry.id, task,
                                    set.solutions_for(entry.id))) return;
        infer_into_puzzle(task);
        if (task.suggested_programs.empty()) return;
        one.spoke = true;
        one.panes = task.inf.pane_split.holds();
        const vector<grid>* answers = set.solutions_for(entry.id);
        const lock_guard on_the_gpu(gpu);
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
                break;
            }
            environment::remove(tree);
        }
    }, [&](int at) {
        const said_one& one = said[(size_t)at];
        spoke += one.spoke;
        pane_spoke += one.spoke && one.panes;
        if (!one.fitted) return;
        ++fitted;
        pane_fitted += one.panes;
        (one.right ? right : wrong)++;
        if (one.panes) (one.right ? pane_right : pane_wrong)++;
        printf("%s\t%s\t%s\n", asked[(size_t)at]->id.c_str(), one.right ? "SOLVED" : "WRONG",
               one.program.c_str());
    });
    fprintf(stderr, "reading spoke: %d | fits train: %d | test right: %d | test wrong: %d\n",
            spoke, fitted, right, wrong);
    fprintf(stderr, "  of those, the ones with panes: %d | fits train: %d | test right: %d | test wrong: %d\n",
            pane_spoke, pane_fitted, pane_right, pane_wrong);
    if (!found_requested) {
        fprintf(stderr, "no such training task: %s\n", requested_id.c_str());
        return 2;
    }
    return 0;
}
