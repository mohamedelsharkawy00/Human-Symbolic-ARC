// JSON layout must never change the training pairs or test-input order.
#include "14_puzzle_loader.hpp"
#include <cstdio>

using namespace arc;
static int checks = 0, failures = 0;
static void check(bool ok, const char* name) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", name); }
}

int main() {
    const string normal = R"({"train":[{"input":[[1]],"output":[[2]]},{"input":[[3,0]],"output":[[4,0]]}],"test":[{"input":[[5]]},{"input":[[6,0]]}]})";
    puzzle original;
    check(load_task_from_content(normal, "fixture", original), "canonical loads");
    check(original.train.size() == 2 && original.test.size() == 2, "pair counts");
    check(original.train[0].input.data[0][0] == 1 && original.train[0].output.data[0][0] == 2 &&
          original.train[1].input.width == 2 && original.train[1].input.data[0][0] == 3 &&
          original.train[1].output.data[0][0] == 4, "outputs belong to their own inputs");
    check(original.test[0].input.data[0][0] == 5 && original.test[1].input.data[0][0] == 6 &&
          original.test[0].output.width == 0 && original.test[1].output.width == 0,
          "test input order and absent answers");
    const string variants[] = {
        R"({"train":[{"output":[[2]],"input":[[1]]},{"output":[[4,0]],"input":[[3,0]]}],"test":[{"input":[[5]]},{"input":[[6,0]]}]})",
        R"( { "test" : [ {"input" : [[5]]}, {"input" : [[6,0]]} ], "train" : [ {"output" : [[2]], "input" : [[1]]}, {"input" : [[3,0]], "output" : [[4,0]]} ] } )",
        R"({"note":"ignore \"input\": [[9]] and } ]","train":[{"note":{"output":[[9]]},"input":[[1]],"output":[[2]]},{"input":[[3,0]],"output":[[4,0]]}],"test":[{"input":[[5]]},{"input":[[6,0]]}]})"
    };
    for (const auto& text : variants) {
        puzzle p;
        bool same = load_task_from_content(text, "fixture", p) &&
                    p.train.size() == original.train.size() && p.test.size() == original.test.size();
        if (same) {
            for (size_t i = 0; i < p.train.size(); ++i)
                same = same && p.train[i].input == original.train[i].input && p.train[i].output == original.train[i].output;
            for (size_t i = 0; i < p.test.size(); ++i)
                same = same && p.test[i].input == original.test[i].input && p.test[i].output == original.test[i].output;
        }
        check(same, "field order, whitespace and metadata do not change pairs");
    }
    vector<grid> answers{grid(1, 1), grid(2, 1)};
    answers[0].data[0][0] = 7; answers[1].data[0][0] = 8;
    puzzle keyed;
    check(load_task_from_content(normal, "fixture", keyed, &answers) &&
          keyed.test[0].output == answers[0] && keyed.test[1].output == answers[1], "separate test answers keep order");
    puzzle mixed;
    check(load_task_from_content(R"({"train":[{"output":[[2]],"input":[[1]]}],"test":[{"input":[[5]]},{"output":[[9]],"input":[[6]]}]})", "mixed", mixed, &answers) &&
          mixed.test.size() == 2 && mixed.test[0].input.data[0][0] == 5 &&
          mixed.test[0].output == answers[0] && mixed.test[1].input.data[0][0] == 6 && mixed.test[1].output.data[0][0] == 9,
          "inline answer cannot be borrowed by earlier test input");
    const string invalid[] = {
        R"({"train":[{"input":[[1]]},{"input":[[3]],"output":[[4]]}],"test":[{"input":[[5]]}]})",
        R"({"train":[{"output":[[2]]}],"test":[{"input":[[5]]}]})",
        R"({"train":[],"test":[{"input":[[5]]}]})",
        R"({"train":[{"input":[[1]],"output":[[2]]}],"test":[]})",
        R"({"train":[{"input":[],"output":[[2]]}],"test":[{"input":[[5]]}]})",
        R"({"train":[{"input":[[1]],"output":[[2]]},],"test":[{"input":[[5]]}]})",
        R"({"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[5]]})",
        R"({"train":[{"input":[[1]],"input":[[3]],"output":[[2]]}],"test":[{"input":[[5]]}]})"
    };
    for (const auto& text : invalid) {
        puzzle p;
        check(!load_task_from_content(text, "bad", p) && p.train.empty() && p.test.empty(), "bad task rejected without partial pairs");
    }
    std::printf("Puzzle loader: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
