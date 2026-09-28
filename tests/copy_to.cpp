// GPU contract checks for CopyTo and a size/color/marker-count transfer check.
// Build arc_copy_to_tests, then run it from the repository root.
#include "16_solver_beam.hpp"
#include "14_puzzle_loader.hpp"
#include "20_cuda_solver.cuh"
#include <cstdio>

using namespace arc;

static int checks = 0, failures = 0;

// Expected pictures use explicit top-left positions, independent of the
// implementation's frame-anchor calculation. Zero is a hole in these fixtures.
static void put(grid& canvas, const grid& pattern, int top, int left) {
    for (int r = 0; r < pattern.height; ++r)
        for (int c = 0; c < pattern.width; ++c)
            if (pattern.data[r][c] && canvas.contains(top + r, left + c))
                canvas.data[top + r][left + c] = pattern.data[r][c];
}

static grid picture(std::initializer_list<const char*> rows) {
    grid out((int)std::char_traits<char>::length(*rows.begin()), (int)rows.size());
    int r = 0;
    for (const char* row : rows) {
        for (int c = 0; c < out.width; ++c) out.data[r][c] = row[c] - '0';
        ++r;
    }
    return out;
}

static void check(const char* name, const example& pair, const std::string& program,
                  const puzzle* learned = nullptr) {
    puzzle local;
    if (!learned) {
        local.train = { pair };
        infer_into_puzzle(local);
        learned = &local;
    }
    node* tree = environment::parse(program);
    grid drawn;
    float score = 0;
    bool solved = false;
    if (tree) {
        environment::execute_tree_via_flat(tree, pair, drawn, *learned,
                                          environment::count_nodes(tree), score, solved);
        environment::remove(tree);
    }
    ++checks;
    if (drawn != pair.output) { ++failures; std::printf("FAIL %s: %s\n", name, program.c_str()); }
}

int main() {
    if (!cuda_init()) return 2;
    const std::string objects = "InputWithoutBackground Split[0] SelectAll Trim InvertSelection ";
    const std::string overlay = "Input SelectAll Union " + objects;
    const grid pattern = picture({ "234", "506", "789" });
    grid input(12, 12);
    put(input, pattern, 1, 1);
    input.data[6][7] = 1;
    input.data[10][10] = 1;

    // Every alignment, two destinations, colors and a transparent center.
    const int tops[] = {6, 5, 4}, lefts[] = {7, 6, 5};
    for (int anchor = 0; anchor < 9; ++anchor) {
        grid expected = input;
        put(expected, pattern, tops[anchor / 3], lefts[anchor % 3]);
        put(expected, pattern, tops[anchor / 3] + 4, lefts[anchor % 3] + 3);
        check("anchors / holes / clipping", {input, expected},
              overlay + "SelectLargestObject[0] CopyTo[" + std::to_string(anchor) + "] Union");
    }

    // Even frames: the earlier middle cell is the anchor, on both axes.
    const grid even = picture({ "2345", "6789" });
    grid rectangular(14, 12);
    put(rectangular, even, 1, 1);
    put(rectangular, picture({ "11", "11", "11" }), 6, 9);
    grid expected = rectangular;
    put(expected, even, 7, 8);
    check("unequal even frames", {rectangular, expected},
          overlay + "SelectLargestObject[0] CopyTo[4] Union");

    // Two independently selected templates, one destination; later copies
    // overlay earlier ones and holes reveal the earlier copy.
    grid multiple(14, 12);
    const grid one = picture({ "23", "45" }), two = picture({ "62", "07" });
    put(multiple, one, 1, 1);
    put(multiple, two, 1, 6);
    multiple.data[8][9] = 1;
    expected = grid(14, 12);
    put(expected, one, 8, 9);
    put(expected, two, 8, 9);
    check("multiple templates", {multiple, expected}, objects + "SelectColor[2] CopyTo[0] Union");

    expected = input;
    grid red = pattern;
    for (int r = 0; r < red.height; ++r)
        for (int c = 0; c < red.width; ++c) if (red.data[r][c]) red.data[r][c] = 2;
    put(expected, red, 5, 6);
    put(expected, red, 9, 9);
    check("copies retain selection for recolor", {input, expected},
          overlay + "SelectLargestObject[0] CopyTo[4] Recolor[2] Union");
    check("no selection is a no-op", {input, input}, objects + "CopyTo[4] SelectAll Union");
    check("no destinations", {input, grid(12, 12)}, objects + "SelectAll CopyTo[4] Union");

    // 6 sources x 6 destinations exceed the 32-object list: no partial set.
    grid crowded(18, 12);
    for (int i = 0; i < 6; ++i) {
        crowded.data[1][1 + i * 3] = 2;
        crowded.data[7][1 + i * 3] = 1;
    }
    check("capacity rejects a partial product", {crowded, grid(18, 12)},
          objects + "SelectColor[2] CopyTo[4] Union");

    // Underlay is chosen by the caller, independently of copying. On an opaque
    // input canvas the copies have nowhere to paint.
    check("composition stays separate", {input, input},
          overlay + "SelectLargestObject[0] CopyTo[4] Underlay");

    // The same learned sentence must read a differently sized, differently
    // colored template and a new marker count, without a pane or blue literal.
    auto centered = [](const grid& motif, std::initializer_list<std::pair<int, int>> markers) {
        example pair{grid(18, 16), grid(18, 16)};
        put(pair.input, motif, 1, 1);
        for (auto [r, c] : markers) pair.input.data[r][c] = 8;
        pair.output = pair.input;
        for (auto [r, c] : markers) put(pair.output, motif, r - motif.height / 2, c - motif.width / 2);
        return pair;
    };
    puzzle transfer;
    transfer.train = {centered(picture({ "232", "434", "222" }), {{6, 8}, {12, 14}}),
                      centered(picture({ "656", "757", "565" }), {{9, 10}})};
    infer_into_puzzle(transfer);
    const example held_out = centered(picture({ "29292", "92929", "29292", "92929", "29292" }),
                                     {{6, 9}, {12, 8}, {12, 14}});
    ++checks;
    if (!transfer.inf.copy_to.holds() || transfer.suggested_programs.empty()) {
        ++failures; std::printf("FAIL inference did not recognize copying\n");
    } else {
        for (const auto& pair : transfer.train)
            check("inferred training sentence", pair, transfer.suggested_programs.front(), &transfer);
        check("inferred transfer", held_out, transfer.suggested_programs.front(), &transfer);
    }
    std::printf("CopyTo: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
