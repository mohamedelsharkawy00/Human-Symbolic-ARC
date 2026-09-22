// WHAT THE READING SOLVES ON ITS OWN, counted. For every training puzzle:
// infer, take suggested_programs, run each on the train pairs, and where one
// fits them all, ask it for the held-out test answer. No search anywhere.
//
// This is the guard on say_the_reading. A program it spells that draws the
// wrong thing shows up here as "WRONG" and nowhere else, so run it after
// touching the emitter and compare the tail against tests/pane_program_truth.txt.
//
// EVERY PUZZLE THE READING SPEAKS FOR, not only the ones with panes. The count
// was gated on pane_split while the pane emitter was the new thing, and that
// gate hid every other branch of say_the_reading from the only tool that runs
// what it says. The panes are still tallied apart, so the pane truth file stays
// comparable to the run that wrote it.
//
//   arc_said            one line a puzzle it solved, then a tally on stderr
#include "15_solver_beam.hpp"
#include "13_puzzle_loader.hpp"
#include "19_cuda_solver.cuh"
#include <cstdio>
#include <string>

using namespace arc;
using namespace std;

int main() {
    if (!cuda_init()) { fprintf(stderr, "no GPU\n"); return 1; }
    task_set set = load_task_set(DS_TRAIN);
    int spoke = 0, fitted = 0, right = 0, wrong = 0;
    int pane_spoke = 0, pane_fitted = 0, pane_right = 0, pane_wrong = 0;
    for (const auto& entry : set.tasks) {
        puzzle task;
        if (!load_task_from_content(entry.content, entry.id, task,
                                    set.solutions_for(entry.id))) continue;
        infer_into_puzzle(task);
        if (task.suggested_programs.empty()) continue;
        const bool panes = task.inf.pane_split.holds();
        ++spoke;
        pane_spoke += panes;
        const vector<grid>* answers = set.solutions_for(entry.id);
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
                ++fitted;
                pane_fitted += panes;
                bool every = answers && answers->size() == task.test.size();
                for (size_t i = 0; i < task.test.size() && every; ++i) {
                    grid drawn; float score = 0; bool solved = false;
                    environment::execute_tree_via_flat(tree, task.test[i], drawn, task, nodes, score, solved);
                    every = drawn == (*answers)[i];
                }
                (every ? right : wrong)++;
                if (panes) (every ? pane_right : pane_wrong)++;
                printf("%s\t%s\t%s\n", task.id.c_str(), every ? "SOLVED" : "WRONG", program.c_str());
                environment::remove(tree);
                break;
            }
            environment::remove(tree);
        }
    }
    fprintf(stderr, "reading spoke: %d | fits train: %d | test right: %d | test wrong: %d\n",
            spoke, fitted, right, wrong);
    fprintf(stderr, "  of those, the ones with panes: %d | fits train: %d | test right: %d | test wrong: %d\n",
            pane_spoke, pane_fitted, pane_right, pane_wrong);
    return 0;
}
