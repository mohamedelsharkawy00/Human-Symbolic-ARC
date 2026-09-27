// From an MSVC developer prompt at the repo root:
// cl /std:c++20 /EHsc /I include/core tests/hole_colors.cpp /Foout/hole_colors.obj /Feout/hole_colors.exe
// out/hole_colors.exe
#include "9_sentences.hpp"
#include <cassert>
#include <cstdio>
using namespace arc;
int main() {
    auto ring = [](int r, int c) { return r == 0 || c == 0 || r == 2 || c == 2; };
    assert(object_hole_count(3, 3, ring) == 1);
    assert(object_hole_count(3, 3, [&](int r, int c) { return ring(r,c) && !(r == 0 && c == 1); }) == 0);
    assert(object_hole_count(3, 3, [](int r, int c) { return (r == 1) != (c == 1); }) == 1);
    assert(object_hole_count(3, 5, [](int r, int c) { return r != 1 || c % 2 == 0; }) == 2);
    example pair;
    pair.input.width = pair.output.width = 3;
    pair.input.height = pair.output.height = 3;
    for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) {
        pair.input.data[r][c] = ring(r,c) ? 8 : 0;
        pair.output.data[r][c] = ring(r,c) ? 2 : 0;
    }
    auto rule = hole_color_rule::detect({pair}, 0);
    assert(rule.holds() && rule.colors[1] == 2);
    auto conflict = pair;
    for (int r=0; r<3; ++r) for (int c=0; c<3; ++c)
        if (ring(r,c)) conflict.output.data[r][c] = 3;
    assert(!hole_color_rule::detect({pair, conflict}, 0).holds());
    conflict = pair; conflict.output.data[1][1] = 2;
    assert(!hole_color_rule::detect({conflict}, 0).holds());
    conflict = pair; conflict.output.data[0][0] = 8;
    assert(!hole_color_rule::detect({conflict}, 0).holds());
    conflict = pair; conflict.output = conflict.input;
    assert(!hole_color_rule::detect({conflict}, 0).holds());
    inference inf; inf.hole_colors = rule;
    auto said = say_the_reading(inf);
    assert(said.no_word_for.empty());
    assert(programs_from(said).front().find("SelectObjectsWithHoles[1] Recolor[2]") != std::string::npos);
    puts("hole color checks passed");
}
