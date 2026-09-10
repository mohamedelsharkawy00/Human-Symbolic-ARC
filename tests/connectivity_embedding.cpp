#include "3_background.hpp"
#include <cassert>
#include <iostream>

// Same geometry as connectivity_stress.py, exercised through production code.
arc::grid scene(bool sealed) {
    arc::grid g;
    g.width = g.height = 30;
    for (int top : {1, 16}) for (int left : {1, 16})
        for (int r = 0; r < 12; ++r) for (int c = 0; c < 12; ++c) {
            bool edge = r == 0 || r == 11 || c == 0 || c == 11;
            bool corner = (r == 0 || r == 11) && (c == 0 || c == 11);
            bool tooth = r >= 1 && r <= 8 && (c == 3 || c == 5 || c == 7 || c == 9);
            g.data[top+r][left+c] = (edge && (sealed || !corner)) || tooth ? 2 : 1;
            if (corner && !sealed) g.data[top+r][left+c] = 0;
        }
    return g;
}

int main() {
    for (bool sealed : {false, true}) {
        auto g = scene(sealed);
        auto objects = arc::label_objects(g);
        assert(std::count(objects.color.begin(), objects.color.end(), 2) == 4);
        int counts[arc::COLOR_COUNT] = {};
        arc::count_colors(g, counts);
        arc::background_evidence evidence;
        arc::measure_embedding(g, counts, evidence);
        assert(evidence.embeds[0] == (sealed ? 527 : 514));
        assert(evidence.embeds[1] == 0);
        assert(evidence.embeds[2] == (sealed ? 472 : 485));
        std::cout << (sealed ? "control" : "stress") << ": black embeds="
                  << evidence.embeds[0] << ", blue embeds=" << evidence.embeds[1] << '\n';
    }
}
