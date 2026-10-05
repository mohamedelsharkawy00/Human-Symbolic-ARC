// EVERY READOUT, WRITTEN DOWN. Infers every training puzzle -- or only the ids
// named -- and writes each one's inference readout to <out-dir>/logs/inference/
// <id>.txt, plus one line per puzzle in <out-dir>/summary.txt: its id, the
// shape-color table it learned, and the first program the reading said. Runs
// nothing on the GPU; arc_said is what runs the programs. Run it from the repo
// root, through scripts\probe.bat:
//   arc_probe <out-dir> [id...]
// scripts\readouts.py reads what it writes.
//
// The puzzles are read at once, one per CPU thread (in_parallel.hpp): each
// writes its own readout, and the summary keeps the dataset's order.
#include "16_solver_beam.hpp"
#include "14_puzzle_loader.hpp"
#include "in_parallel.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
using namespace arc;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "arc_probe <out-dir> [id...]\n"); return 2; }
    const vector<string> only(argv + 2, argv + argc);
    task_set set = load_task_set(DS_TRAIN);
    const filesystem::path out = argv[1];
    filesystem::create_directories(out);
    filesystem::current_path(out);
    FILE* summary = fopen("summary.txt", "w");
    if (!summary) { fprintf(stderr, "cannot write %s\n", (out / "summary.txt").string().c_str()); return 1; }
    // Each puzzle asked, by its place in the dataset, which its readout names.
    vector<int> asked;
    for (int at = 0; at < (int)set.tasks.size(); ++at)
        if (only.empty() || ranges::find(only, set.tasks[(size_t)at].id) != only.end()) asked.push_back(at);
    vector<string> lines(asked.size());
    in_parallel((int)asked.size(), [&](int which) {
        const int at = asked[(size_t)which];
        const auto& entry = set.tasks[(size_t)at];
        puzzle task;
        if (!load_task_from_content(entry.content, entry.id, task, set.solutions_for(entry.id))) return;
        infer_into_puzzle(task, { at, (int)set.tasks.size(), DS_TRAIN }, true);
        const auto& rule = task.inf.shape_color_map;
        char shape[64];
        snprintf(shape, sizeof shape, "\tshape %d/%d", (int)rule.claimed_by, rule.count);
        string& line = lines[(size_t)which];
        line = entry.id + shape;
        for (int i = 0; i < rule.count; ++i) {
            char key[32];
            snprintf(key, sizeof key, " %u>%d", rule.keys[i], (int)rule.colors[i]);
            line += key;
        }
        line += "\t" + (task.suggested_programs.empty() ? string("-") : task.suggested_programs.front()) + "\n";
    }, [&](int which) { fputs(lines[(size_t)which].c_str(), summary); });
    fclose(summary);
    return 0;
}
