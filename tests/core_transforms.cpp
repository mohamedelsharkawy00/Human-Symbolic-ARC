// CPU: cl /nologo /std:c++20 /EHsc /Iinclude/core tests/core_transforms.cpp /Fe:out/core_transforms.exe /Fo:out/core_transforms.obj
// GPU: add /MD /DARC_TEST_CUDA out/x64-Release/arc_gpu.lib to the CPU command,
// followed by /link /LIBPATH:"%CUDA_PATH%/lib/x64" cudart.lib.
#ifdef ARC_TEST_CUDA
#include "15_solver_beam.hpp"
#include "19_cuda_solver.cuh"
#else
#include "7_infer.hpp"
#endif
#include <cstdlib>
#include <iostream>

using namespace arc;

static void check(bool holds, const char* message) {
    if (holds) return;
    std::cerr << message << '\n';
    std::exit(1);
}

static grid picture(std::initializer_list<std::initializer_list<int>> rows) {
    grid result((int)rows.begin()->size(), (int)rows.size());
    int r = 0;
    for (const auto& row : rows) {
        int c = 0;
        for (int color : row) result.data[r][c++] = color;
        ++r;
    }
    return result;
}

static grid transpose(const grid& source) {
    grid result(source.height, source.width);
    for (int r = 0; r < source.height; ++r)
        for (int c = 0; c < source.width; ++c) result.data[c][r] = source.data[r][c];
    return result;
}

static grid with_background(grid cells) {
    for (int r = 0; r < cells.height; ++r)
        for (int c = 0; c < cells.width; ++c)
            if (cells.data[r][c] == TRANSPARENT) cells.data[r][c] = 9;
    return cells;
}

#ifdef ARC_TEST_CUDA
static void check_gpu(const grid& input, const grid& expected, const string& program) {
    puzzle task;
    task.rule.Chosen = output_dimension_rule::Identity;
    task.inf.size = task.rule;
    task.inf.background.across_inputs.push_back({.color = 9});
    task.inf.background.across_outputs.push_back({.color = 9});
    task.invariant_colors[9] = true;
    const example pair{with_background(input), with_background(expected)};
    node* tree = environment::parse(program);
    check(tree != nullptr, "GPU test program parses");
    grid drawn;
    float score = 0;
    bool solved = false;
    const bool ran = environment::execute_tree_via_flat(
        tree, pair, drawn, task, environment::count_nodes(tree), score, solved);
    environment::remove(tree);
    if (!ran || drawn != pair.output) {
        std::cerr << program << '\n';
        for (int r = 0; r < pair.output.height; ++r) {
            for (int c = 0; c < pair.output.width; ++c) std::cerr << pair.output.data[r][c] << ' ';
            std::cerr << " expected; drew ";
            for (int c = 0; c < drawn.width; ++c) std::cerr << drawn.data[r][c] << ' ';
            std::cerr << '\n';
        }
    }
    check(ran && drawn == pair.output, "GPU transformation preserves expected cells and selection");
}
#endif

static void check_move(const grid& input, const grid& expected, bool rows, int step, bool should_move) {
    object_cells packed;
    packed.clear();
    for (int r = 0; r < input.height; ++r)
        for (int c = 0; c < input.width; ++c) packed[r][c] = (int8_t)input.data[r][c];
    grid drawn(input.width, input.height);
    drawn.fill(TRANSPARENT);
    bool written[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    const bool moved = slam_cells(input.width, input.height, rows, step,
        [&](int r, int c) { return (int)packed[r][c]; },
        [&](int r, int c, int color) {
            check(drawn.contains(r, c), "slam stays inside its supplied dimensions");
            check(!written[r][c], "slam never overwrites another cell");
            written[r][c] = true;
            drawn.data[r][c] = color;
        });
    check(drawn == expected, "slam preserves cell order, colors, and gaps");
    check(moved == should_move, "slam reports whether anything moved");
#ifdef ARC_TEST_CUDA
    const string op = string(rows ? "SlamRow[" : "SlamColumn[") + to_string(step) + "] ";
    check_gpu(input, expected, "InputWithoutBackground SelectAll " + op + "Union");
    check_gpu(input, input, "InputWithoutBackground " + op + "SelectAll Union");
#endif
}

static void paint_box(grid& canvas, const cell_box& box, int color) {
    for (int row = box.row0; row <= box.row1; ++row)
        for (int col = box.col0; col <= box.col1; ++col) canvas.data[row][col] = color;
}

static void check_alignment() {
    cell_box empty;
    check(empty.empty() && empty.area() == 0, "a new box is empty");
    empty.include_cell(-3, 40);
    check(empty.row0 == -3 && empty.col0 == 40 && empty.area() == 1,
          "the first cell establishes all four edges, including outside-grid coordinates");
    empty.include_cell(2, 35);
    check(empty.row1 == 2 && empty.col0 == 35 && empty.area() == 36, "a box expands in either direction");

    cell_box colors[COLOR_COUNT];
    color_boxes(picture({{2, TRANSPARENT}, {TRANSPARENT, 2}}), colors);
    check(colors[2].area() == 4 && colors[1].empty(), "color boxes count holes within their bounds");
    color_boxes(picture({{1}}), colors);
    check(colors[2].empty() && colors[1].area() == 1, "reusing color boxes clears absent colors");

    const cell_box target{4, 4, 5, 5};
    struct landing { cell_box moving; int down, across; };
    const landing cases[] = {
        {{4, 0, 5, 1}, 0, 2},   // left of the target
        {{4, 9, 5, 10}, 0, -3}, // right
        {{0, 4, 1, 5}, 2, 0},   // above
        {{9, 4, 10, 5}, -3, 0}, // below
        {{4, 2, 5, 3}, 0, 0},   // already flush
        {{0, 0, 1, 1}, 0, 2},   // equal gaps choose horizontal movement
    };
    correspondence_election correspondence;
    correspondence.segmentation = segmentation_methods::all_cells_of_one_color;
    for (const auto& sample : cases) {
        check(shift_to_touch(sample.moving, target) == std::pair{sample.down, sample.across},
              "alignment chooses the expected axis, direction, and distance");
        grid input(12, 12);
        input.fill(TRANSPARENT);
        paint_box(input, target, 1);
        paint_box(input, sample.moving, 2);
        grid expected(12, 12);
        expected.fill(TRANSPARENT);
        paint_box(expected, target, 1);
        paint_box(expected, {sample.moving.row0 + sample.down, sample.moving.col0 + sample.across,
                             sample.moving.row1 + sample.down, sample.moving.col1 + sample.across}, 2);
        const auto inferred = slide_to_touch_rule::detect(
            {{with_background(input), with_background(expected)}}, correspondence, 9, 9);
        if (sample.down || sample.across)
            check(inferred.holds() && inferred.stays == 1 && inferred.slides == 2,
                  "inference recognizes the same alignment");
        else check(!inferred.holds(), "already touching is not evidence of movement");
#ifdef ARC_TEST_CUDA
        const string anchor = "InputWithoutBackground Colors SelectColor[1] Union ";
        check_gpu(input, expected, anchor +
            "InputWithoutBackground Colors SelectColor[2] Trim AlignToCanvas Union");
        check_gpu(input, input, anchor +
            "InputWithoutBackground Colors SelectAll Trim InvertSelection AlignToCanvas SelectAll Union");
#endif
    }
    check(shift_to_touch(target, target) == std::pair{0, -2}, "overlap and equal distances choose left");
    check(shift_to_touch({4, 2, 5, 7}, target) == std::pair{-2, 0},
          "overlap can choose vertical movement; equal distances choose up");
    check(shift_to_touch({-4, 0, -3, 1}, {0, 0, 1, 1}) == std::pair{2, 0},
          "alignment also handles boxes outside the canvas");
#ifdef ARC_TEST_CUDA
    // Overlapping boxes can still contain disjoint paint: target midpoints and
    // moving corners. Equal-distance ties must agree on the GPU as well.
    const cell_box frames[] = {{4, 4, 6, 6}, {4, 2, 6, 8}};
    const cell_box destinations[] = {{4, 1, 6, 3}, {1, 2, 3, 8}};
    for (int i = 0; i < 2; ++i) {
        grid input(12, 12);
        input.fill(TRANSPARENT);
        input.data[4][5] = input.data[5][4] = input.data[5][6] = input.data[6][5] = 1;
        grid expected = input;
        for (int row : {frames[i].row0, frames[i].row1})
            for (int col : {frames[i].col0, frames[i].col1}) input.data[row][col] = 2;
        for (int row : {destinations[i].row0, destinations[i].row1})
            for (int col : {destinations[i].col0, destinations[i].col1}) expected.data[row][col] = 2;
        check_gpu(input, expected,
            "InputWithoutBackground Colors SelectColor[1] Union "
            "InputWithoutBackground Colors SelectColor[2] Trim AlignToCanvas Union");
    }
    grid lone(12, 12);
    lone.fill(TRANSPARENT);
    paint_box(lone, {4, 0, 5, 1}, 2);
    check_gpu(lone, lone, "InputWithoutBackground Colors SelectAll Trim AlignToCanvas Union");
#endif
}

int main() {
#ifdef ARC_TEST_CUDA
    check(cuda_init(), "CUDA initialization");
#endif
    check_alignment();
    const int hole = TRANSPARENT;
    const grid input = picture({{0, hole, 2, 3, hole}, {hole, 1, hole, hole, 2}});
    const grid right = picture({{hole, 0, hole, 2, 3}, {hole, hole, 1, hole, 2}});
    const grid left = picture({{0, 2, 3, hole, hole}, {1, hole, hole, 2, hole}});
    for (bool rows : {true, false}) {
        const auto orient = [&](const grid& g) { return rows ? g : transpose(g); };
        check_move(orient(input), orient(right), rows, 1, true);
        check_move(orient(input), orient(left), rows, -1, true);
        check_move(orient(input), orient(input), rows, 0, false);
        const grid full = picture({{0, 1, 2, 3, 4}});
        const grid empty = picture({{hole, hole, hole}});
        for (int step : {-1, 1}) {
            check_move(orient(full), orient(full), rows, step, false);
            check_move(orient(empty), orient(empty), rows, step, false);
            const grid single = picture({{2}});
            check_move(single, single, rows, step, false);
        }
        grid edge(MAX_GRID_DIM, MAX_GRID_DIM);
        edge.fill(hole);
        edge.data[0][0] = 2;
        edge.data[MAX_GRID_DIM - 1][MAX_GRID_DIM - 1] = 3;
        grid shifted = edge;
        shifted.data[0][0] = hole;
        shifted.data[0][1] = 2;
        check_move(orient(edge), orient(shifted), rows, 1, true);
    }

    // Inference must recognize a real move, and reject a stationary full box.
    const grid shape = picture({{2, 0, 2, 0, 0}, {0, 0, 0, 0, 2}, {2, 2, 0, 0, 2}});
    const grid slid = picture({{0, 2, 0, 2, 0}, {0, 0, 0, 0, 2}, {0, 2, 2, 0, 2}});
    correspondence_election correspondence;
    correspondence.segmentation = segmentation_methods::all_cells_of_one_color;
    for (bool rows : {true, false}) {
        const example pair{rows ? shape : transpose(shape), rows ? slid : transpose(slid)};
        const auto rule = piece_slam_rule::detect({pair}, correspondence, 0, 0);
        check(rule.holds() && rule.rows == rows && rule.step == 1, "inference recognizes the shared movement");
    }
    const grid full = picture({{2, 2}, {2, 2}});
    check(!piece_slam_rule::detect({{full, full}}, correspondence, 0, 0).holds(),
          "stationary cells are not evidence of a slam");
    check(!piece_slam_rule::detect({}, correspondence, 0, 0).holds(), "empty training is not a rule");
#ifdef ARC_TEST_CUDA
    cuda_runtime_shutting_down();
#endif
    std::cout << "Transformation checks passed\n";
}
