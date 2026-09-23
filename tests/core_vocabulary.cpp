// From a Visual Studio developer shell at the repository root:
// cl /nologo /std:c++20 /EHsc /Iinclude/core tests/core_vocabulary.cpp /Fe:out/core_vocabulary.exe /Fo:out/core_vocabulary.obj
#include "7_vocabulary.hpp"
#include "8_sentences.hpp"
#include "11_dsl.cuh"
#include <cstdlib>
#include <iostream>

static void check(bool holds, const char* message) {
    if (holds) return;
    std::cerr << message << '\n';
    std::exit(1);
}

int main() {
    using namespace arc;
    static_assert(COLOR_PAIR_COUNT == 45);
    static_assert(GPU_GRID_CELLS == 900);
    static_assert(operation_named("MoveRow[-3]") == node_type::MoveRow);
    static_assert(operation_named("?") == node_type::type_count);
    static_assert(takes_value(node_type::ScaleUp, 3) && !takes_value(node_type::ScaleUp, 1),
                  "an argument is read off the op's own range");
    static_assert(takes_value(node_type::Skew, -1) && !takes_value(node_type::Skew, 2));

    check(mark_after("") == the_mark::none, "empty selection");
    check(mark_after("Input SelectColor[2] Recolor[3]") == the_mark::some,
          "acts preserve a partial selection");
    check(mark_after("SelectAll SelectColor[2]") == the_mark::all,
          "selectors add; they do not narrow all");
    check(mark_after("SelectAll InvertSelection") == the_mark::none, "invert all");
    check(mark_after("InvertSelection") == the_mark::all, "invert none");
    check(mark_after("SelectColor[2] InvertSelection") == the_mark::some, "invert some");
    for (const char* carve : {"Split[1]", "Colors", "Partition", "DeleteSelection"}) {
        check(mark_after(std::string("SelectAll ") + carve) == the_mark::none,
              "carving or deletion clears selection");
    }
    check(closed_with("Input SelectAll", "Union") == "Input SelectAll Union",
          "avoid redundant selection");
    check(closed_with("Input SelectColor[2]", "Union") ==
          "Input SelectColor[2] SelectAll Union", "whole-list painting selects all");
    check(closed_with("Input", "Withhold") == "Input Withhold", "withhold needs no selection");
    check(closed_on_the_pick("Input SelectColor[2]", "Union") ==
          "Input SelectColor[2] Union", "partial painting preserves the pick");

    const auto rows = sentences_of(
        "  Input   SelectAll MoveRow[-3] Withhold  CurrentObjects Union  Input");
    check(rows == std::vector<std::string>{"Input SelectAll MoveRow[-3] Withhold",
                                         "CurrentObjects Union"},
          "sentence boundaries, spacing, arguments, and incomplete tail");
    for (const char* source : {"Input", "InputWithoutBackground", "Output", "CurrentObjects", "Seeds"})
        check(!sentence_with_no_source(std::string(source) + " SelectAll Union"), "valid source");
    check(sentence_with_no_source("Input Union SelectAll Union"), "missing second source");
    check(sentence_with_no_source("? Union"), "unknown word is not a source");
    check(mark_after("SelectAll ?") == the_mark::all, "placeholder preserves selection");

    said_reading said;
    say_beats(said, "Input", {{"SelectAll", "ScaleUp[3]"}}, "Union");
    say_beats(said, "InputWithoutBackground", {{"SelectAll", "Fractal"}}, "Difference", true);
    say_beats(said, "Input", {{"SelectAll", "Trim"}}, "Union");
    check(said.sentences == std::vector<std::string>{
              "Input SelectAll ScaleUp[3] Union",
              "InputWithoutBackground SelectAll Fractal Difference",
              "CurrentObjects SelectAll Trim Union"},
          "a line said afresh reads its own source; the rest carry the hand");
    std::cout << "Core vocabulary checks passed\n";
}
