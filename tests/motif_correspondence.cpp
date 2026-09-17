#include "8_sentences.hpp"
#include <cassert>

// Synthetic repeating blocks, including an incomplete last output instance.
int main() {
    // Three columns contain one full two-column period and a matching fragment.
    // Inference must preserve the same unit that Pattern would extract.
    arc::grid narrow(3, 6), taller(3, 9);
    for (auto* g : {&narrow, &taller})
        for (int r = 0; r < g->height; ++r)
            for (int c = 0; c < g->width; ++c) g->data[r][c] = (r + c) % 2;
    const auto unit = arc::object_segmentation::block_of(narrow, taller, 0,
        arc::segmentation_methods::minimal_repeating_set, 0, 0);
    assert(unit.first == 2 && unit.second == 2);

    arc::example sample;
    auto pattern = [](int height, int color) {
        arc::grid g(2, height);
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < 2; ++c)
                g.data[r][c] = c == r % 3 % 2 ? color : 0;
        return g;
    };
    sample.input = pattern(6, 1);
    sample.output = pattern(10, 2);
    auto facts = arc::correspondence_facts::infer({sample}, 0);
    const int reading = int(arc::segmentation_methods::minimal_repeating_set) * 16 + 4;
    const auto& pair = facts.reading_pairs[reading][0];
    assert(pair.edges.size() == 1);
    assert(pair.edges[0].from == 0 && pair.edges[0].to == 0);
    assert(pair.left_over.empty());
    assert(pair.deleted == 0 && pair.invented == 0);
    for (size_t i = 1; i < pair.input_info.size(); ++i)
        assert(pair.input_info[i].instance_of == 0 && pair.input_info[i].edges_touching == 0);
    for (size_t i = 1; i < pair.output_info.size(); ++i)
        assert(pair.output_info[i].instance_of == 0 && pair.output_info[i].edges_touching == 0);
    arc::inference inf;
    inf.correspondence = facts;
    inf.size = arc::output_dimension_rule::infer_rule({sample});
    inf.motif_continuation = arc::detect_motif_continuation({sample}, facts);
    assert(inf.motif_continuation.valid);
    const auto programs = arc::programs_from(arc::say_the_reading(inf));
    assert(programs.size() == 1);
    assert(programs.front() == "Input SelectAll Pattern Union CurrentObjects SelectAll BidiColorMap[9] Union CurrentObjects SelectAll Tile[0] Union");
    // A changed background cell cannot be hidden by foreground-only coverage.
    auto broken = sample;
    broken.output.data[9][1] = 2;
    assert(!arc::detect_motif_continuation({broken}, facts).valid);
    auto missing_instance = facts;
    missing_instance.pairs[0].output_info.back().instance_of = -1;
    assert(!arc::detect_motif_continuation({sample}, missing_instance).valid);
    auto moved = facts;
    moved.pairs[0].edges[0].row = 1;
    moved.pairs[0].output_info[0].summary.y0 = 1;
    assert(!arc::detect_motif_continuation({sample}, moved).valid);
    // Recoloring is required; no instances may be explained without a motif link.
    const auto& unchanged = facts.reading_pairs[reading - 4][0];
    assert(unchanged.edges.empty() && !unchanged.left_over.empty());
    for (const auto& piece : unchanged.output_info) assert(piece.instance_of == -1);
}
