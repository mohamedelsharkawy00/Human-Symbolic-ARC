#include "8_sentences.hpp"
#include <cassert>

// A rectangular, multicolor source with a nonzero background. This exercises
// placement independently of the benchmark's square, monochrome examples.
int main() {
    arc::example ex;
    ex.input = arc::grid(2, 3);
    const int cells[3][2] = {{1, 7}, {7, 2}, {1, 7}};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c) ex.input.data[r][c] = cells[r][c];
    ex.output = arc::grid(4, 9);
    for (int r = 0; r < 9; ++r)
        for (int c = 0; c < 4; ++c) ex.output.data[r][c] = 7;
    for (auto [top, left] : {std::pair{0, 0}, std::pair{3, 2}, std::pair{6, 0}})
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 2; ++c) ex.output.data[top+r][left+c] = cells[r][c];
    auto facts = arc::correspondence_facts::infer({ex}, 7);
    // Exercise the whole-input reading even if another segmentation ties it.
    facts.segmentation = arc::segmentation_methods::smallest_of_two_grids;
    facts.pairs = facts.reading_pairs[64];
    auto detect = [&](const arc::example& sample) {
        return arc::detect_input_copy_layout({sample}, facts, 7, 7).valid;
    };
    assert(detect(ex));
    arc::inference inf;
    inf.correspondence = facts;
    inf.size = arc::output_dimension_rule::infer_rule({ex});
    inf.input_copy_layout = arc::detect_input_copy_layout({ex}, facts, 7, 7);
    auto said = arc::say_the_reading(inf);
    assert(said.no_word_for.empty());
    assert(arc::programs_from(said) == std::vector<std::string>{
        "InputWithoutBackground SelectAll Fractal Union"});

    auto bad = ex;
    bad.output.data[0][1] = 1; // Paint a hole within an otherwise matching copy.
    assert(!detect(bad));
    bad = ex;
    bad.output.data[0][2] = 1; // Paint inside an unoccupied block.
    assert(!detect(bad));
    bad = ex;
    bad.output.data[0][0] = 2; // Same occupancy, wrong copy color.
    assert(!detect(bad));
    assert(!arc::detect_input_copy_layout({ex}, facts, 7, 0).valid);
    auto missing = facts;
    missing.pairs[0].edges.pop_back();
    assert(!arc::detect_input_copy_layout({ex}, missing, 7, 7).valid);
    auto turned = facts;
    turned.pairs[0].edges[0].pose = 3;
    assert(!arc::detect_input_copy_layout({ex}, turned, 7, 7).valid);

    // Without the placement relationship, do not invent Pattern or translate
    // copy offsets into movement instructions.
    inf.input_copy_layout.valid = false;
    said = arc::say_the_reading(inf);
    assert(said.sentences.empty() && !said.no_word_for.empty());
    inf.correspondence.segmentation = arc::segmentation_methods::connected_multicolor;
    said = arc::say_the_reading(inf);
    for (const auto& sentence : said.sentences) {
        assert(sentence.find("MoveRow") == std::string::npos);
        assert(sentence.find("MoveColumn") == std::string::npos);
    }
}
