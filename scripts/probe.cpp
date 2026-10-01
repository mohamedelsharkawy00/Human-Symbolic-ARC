// EVERY READOUT, WRITTEN DOWN. Infers every training puzzle -- or only the ids
// named -- and writes each one's inference readout to <out-dir>/logs/inference/
// <id>.txt, plus one line per puzzle in <out-dir>/summary.txt: its id, the
// shape-color table it learned, and the first program the reading said. Runs
// nothing on the GPU; arc_said is what runs the programs. Run it from the repo
// root, through scripts\probe.bat:
//   arc_probe <out-dir> [id...]
// scripts\readouts.py reads what it writes.
#include "16_solver_beam.hpp"
#include "14_puzzle_loader.hpp"
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
    int idx = 0;
    for (const auto& entry : set.tasks) {
        const int at = idx++;
        if (!only.empty() && ranges::find(only, entry.id) == only.end()) continue;
        puzzle task;
        if (!load_task_from_content(entry.content, entry.id, task, set.solutions_for(entry.id))) continue;
        infer_into_puzzle(task, { at, (int)set.tasks.size(), DS_TRAIN }, true);
        const auto& rule = task.inf.shape_color_map;
        fprintf(summary, "%s\tshape %d/%d", entry.id.c_str(), (int)rule.claimed_by, rule.count);
        for (int i = 0; i < rule.count; ++i) fprintf(summary, " %u>%d", rule.keys[i], rule.colors[i]);
        fprintf(summary, "\t%s\n", task.suggested_programs.empty() ? "-" : task.suggested_programs.front().c_str());
    }
    fclose(summary);
    return 0;
}
