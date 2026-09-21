// WHAT THE READING SOLVES ON ITS OWN, counted. For every training puzzle:
// infer, take suggested_programs, run each on the train pairs, and where one
// fits them all, ask it for the held-out test answer. No search anywhere.
//
// This is the guard on say_the_reading. A program it spells that draws the
// wrong thing shows up here as "WRONG" and nowhere else, so run it after
// touching the emitter and compare the tail against tests/pane_program_truth.txt.
//
//   arc_said            one line a puzzle it solved, then a tally on stderr
#include "15_solver_beam.hpp"
#include "13_puzzle_loader.hpp"
#include "19_cuda_solver.cuh"
#include <cstdio>
#include <string>

using namespace arc;
using namespace std;

static bool same(const grid& a, const grid& b) {
    if (a.width != b.width || a.height != b.height) return false;
    for (int r = 0; r < b.height; ++r)
        for (int c = 0; c < b.width; ++c)
            if (a.data[r][c] != b.data[r][c]) return false;
    return true;
}

int main() {
    if (!cuda_init()) { fprintf(stderr, "no GPU\n"); return 1; }
    task_set set = load_task_set(DS_TRAIN);
    int spoke = 0, fitted = 0, right = 0, wrong = 0;
    for (const auto& entry : set.tasks) {
        puzzle task;
        if (!load_task_from_content(entry.content, entry.id, task,
                                    set.solutions_for(entry.id))) continue;
        infer_into_puzzle(task);
        if (!task.inf.pane_split.holds() || task.suggested_programs.empty()) continue;
        ++spoke;
        const vector<grid>* answers = set.solutions_for(entry.id);
        for (const string& program : task.suggested_programs) {
            node* tree = environment::parse(program);
            if (!tree) continue;
            const int nodes = environment::count_nodes(tree);
            bool all_fit = true;
            for (const auto& pair : task.train) {
                grid drawn; float score = 0; bool solved = false;
                environment::execute_tree_via_flat(tree, pair, drawn, task, nodes, score, solved);
                if (!same(drawn, pair.output)) { all_fit = false; break; }
            }
            if (all_fit) {
                ++fitted;
                bool every = answers && answers->size() == task.test.size();
                for (size_t i = 0; i < task.test.size() && every; ++i) {
                    grid drawn; float score = 0; bool solved = false;
                    environment::execute_tree_via_flat(tree, task.test[i], drawn, task, nodes, score, solved);
                    every = same(drawn, (*answers)[i]);
                }
                (every ? right : wrong)++;
                printf("%s\t%s\t%s\n", task.id.c_str(), every ? "SOLVED" : "WRONG", program.c_str());
                environment::remove(tree);
                break;
            }
            environment::remove(tree);
        }
    }
    fprintf(stderr, "split holds & reading spoke: %d | fits train: %d | test right: %d | test wrong: %d\n",
            spoke, fitted, right, wrong);
    return 0;
}
