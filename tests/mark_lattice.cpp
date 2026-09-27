// From an MSVC developer prompt at the repo root:
// cl /std:c++20 /EHsc /utf-8 /I include/core tests/mark_lattice.cpp /Foout/mark_lattice.obj /Feout/mark_lattice.exe
// out/mark_lattice.exe
#include "9_sentences.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>
using namespace arc;

static grid from_rows(std::initializer_list<const char*> rows) {
    grid g;
    for (const char* row : rows) {
        g.width = 0;
        for (const char* cell = row; *cell; ++cell) g.data[g.height][g.width++] = *cell - '0';
        ++g.height;
    }
    return g;
}

int main() {
    // The arithmetic the op and the reading share.
    const int rows_apart[] = { 5, 7 };
    assert(mark_lattice_period(2, [&](int i) { return rows_apart[i]; }) == 4);
    assert(mark_lattice_period(1, [&](int i) { return rows_apart[i]; }) == 0);
    assert(mark_lattice_period(2, [](int) { return 3; }) == 0);
    auto laid = [](int top, int height, int period, int rows) {
        std::vector<std::pair<int, int>> rows_laid;
        each_lattice_row(top, height, period, rows,
                         [&](int row, int source) { rows_laid.push_back({ row, source }); });
        return rows_laid;
    };
    using laid_rows = std::vector<std::pair<int, int>>;
    assert(laid(5, 1, 4, 14) == laid_rows({ { 5, 0 }, { 9, 0 }, { 13, 0 } }));
    assert(laid(5, 2, 4, 11) == laid_rows({ { 5, 0 }, { 6, 1 }, { 9, 0 }, { 10, 1 } }));
    assert(laid(-2, 1, 4, 7) == laid_rows({ { 2, 0 }, { 6, 0 } }));
    assert(laid(5, 1, 0, 14).empty());
    assert(lies_on_its_side(9, 3) && !lies_on_its_side(3, 9) && !lies_on_its_side(4, 4));

    // Split hands pieces over as the walk meets them; Colors by colour.
    const grid met_high_first = from_rows({ "030", "000", "100" });
    assert(pieces_as_said(met_high_first, 0, segmentation_methods::connected_monochrome)
               .front().first_color() == 3);
    assert(pieces_as_said(met_high_first, 0, segmentation_methods::all_cells_of_one_color)
               .front().first_color() == 1);

    // Lines that come round on one row are painted in list order, later on top.
    grid drawn;
    assert(mark_lattice_rule::draw(from_rows({ "1", "2", "0", "0", "3", "0", "0" }), 0, 0,
                                   segmentation_methods::connected_monochrome, false, drawn));
    assert(drawn == from_rows({ "1", "2", "0", "1", "3", "0", "1" }));

    // Standing up: two marks, taking turns every two gaps.
    example standing;
    standing.input  = from_rows({ "000", "200", "000", "003", "000", "000", "000", "000", "000" });
    standing.output = from_rows({ "000", "222", "000", "333", "000", "222", "000", "333", "000" });
    // Lying on its side: the lines cross the short way.
    example lying;
    lying.input  = from_rows({ "010000000", "000000000", "000040000" });
    lying.output = from_rows({ "010040010", "010040010", "010040010" });

    const correspondence_election nobody_elected;
    auto upright = mark_lattice_rule::detect({ standing }, nobody_elected, 0, 0);
    assert(upright.holds() && !upright.portrait);
    assert(upright.segmentation == segmentation_methods::connected_monochrome);
    auto both = mark_lattice_rule::detect({ standing, lying }, nobody_elected, 0, 0);
    assert(both.holds() && both.portrait);

    // Nothing laid, a line out of place, or one mark alone: no lattice.
    example still = standing;
    still.output = still.input;
    assert(!mark_lattice_rule::detect({ still }, nobody_elected, 0, 0).holds());
    example stray = standing;
    stray.output.data[8][1] = 2;
    assert(!mark_lattice_rule::detect({ stray }, nobody_elected, 0, 0).holds());
    example alone = standing;
    alone.input.data[3][2] = 0;
    assert(!mark_lattice_rule::detect({ alone }, nobody_elected, 0, 0).holds());

    inference inf;
    inf.mark_lattice = both;
    auto said = say_the_reading(inf);
    assert(said.no_word_for.empty());
    assert(programs_from(said).front() ==
           "Input Split[1] SelectAll Trim NormalizeOrientation[0] TileX[1] Tile[3] RestoreOrientation Union");
    inf.mark_lattice = upright;
    assert(programs_from(say_the_reading(inf)).front() == "Input Split[1] SelectAll Trim TileX[1] Tile[3] Union");
    puts("mark lattice checks passed");
}
