// A SPELLING BENCH. Hands one program string to one puzzle and shows what it
// paints beside what was wanted, pair by pair. Nothing is searched: this only
// answers "can the words we have already say this?".
//
//   spell <task-id> "<program>"          one program
//   spell <task-id> "<program>" --test   and what it draws on each test, fitting or not
//   spell <task-id> --file <path>        one program per line, first that solves wins
#include "16_solver.hpp"
#include "14_puzzle_loader.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace arc;
using namespace std;

static void show(const char* label, const grid& g) {
    printf("    %s", label);
    if (g.width <= 0 || g.height <= 0) { printf(" (nothing)\n"); return; }
    printf(" %dx%d\n", g.width, g.height);
    for (int r = 0; r < g.height; ++r) {
        printf("      ");
        for (int c = 0; c < g.width; ++c) printf("%d", (int)g.data[r][c]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "spell <task-id> \"<program>\" [--test] | --file <path>\n"); return 2; }
    const string id = argv[1];
    vector<string> programs;
    // Every test drawn, whether the program fits the pairs or not: what a
    // program not finished yet has drawn so far.
    bool every_test = false;
    if (string(argv[2]) == "--file") {
        ifstream in(argv[3]);
        for (string line; getline(in, line);)
            if (!line.empty() && line[0] != '#') programs.push_back(line);
    } else {
        programs.push_back(argv[2]);
        every_test = argc > 3 && string(argv[3]) == "--test";
    }

    task_set set = load_task_set(DS_TRAIN);
    puzzle task;
    bool found = false;
    for (const auto& entry : set.tasks)
        if (entry.id == id &&
            load_task_from_content(entry.content, entry.id, task, set.solutions_for(entry.id))) {
            found = true; break;
        }
    if (!found) { fprintf(stderr, "no such task: %s\n", id.c_str()); return 1; }
    infer_into_puzzle(task);

    for (const string& program : programs) {
        node* tree = environment::parse(program);
        if (!tree) { printf("UNPARSED   %s\n", program.c_str()); continue; }
        int fits = 0;
        vector<grid> drew(task.train.size());
        for (size_t i = 0; i < task.train.size(); ++i) {
            drew[i] = environment::draw(tree, task.train[i], task);
            fits += (drew[i] == task.train[i].output);
        }
        printf("%s  %zu/%zu train pairs   %s\n",
               fits == (int)task.train.size() ? "FITS      " : "no        ",
               (size_t)fits, task.train.size(), program.c_str());
        if (fits != (int)task.train.size() && programs.size() == 1)
            for (size_t i = 0; i < task.train.size(); ++i) {
                printf("  pair %zu\n", i);
                show("drew  ", drew[i]);
                show("wanted", task.train[i].output);
            }
        if (fits == (int)task.train.size() || every_test) {
            const vector<grid>* answers = set.solutions_for(id);
            for (size_t i = 0; i < task.test.size() && answers && i < answers->size(); ++i) {
                const grid drawn = environment::draw(tree, task.test[i], task);
                const bool right = drawn == (*answers)[i];
                printf("            test %zu: %s\n", i, right ? "RIGHT" : "wrong");
                // A program that fits every pair and still misses the test is the
                // mistake worth seeing, so it is shown the way a pair is.
                if ((!right && programs.size() == 1) || every_test) {
                    show("drew  ", drawn);
                    show("wanted", (*answers)[i]);
                }
            }
        }
        environment::remove(tree);
    }
    return 0;
}
