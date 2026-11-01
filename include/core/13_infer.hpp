// The reading in order: the ground plan, each cut and pass read to the end of
// its rules, the cut that says the most kept (says_the_most), and infer(),
// which asks them all.
#pragma once
#include "12_line.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arc {

// WHETHER THE CUT'S PIECES GO UNSAID WHOLE: matched, they do not redraw the
// outputs where they lie, and nothing on the line says what else happened.
inline bool pieces_go_unsaid(const inference& read) {
    const auto& facts = read.correspondence;
    return !facts.chosen_pairs().empty() && !read.pieces_redraw && facts.readings[facts.chosen].line.empty();
}

// HOW MUCH OF THE PAIRS A READING LEAVES UNSAID: what its chosen cut's line
// leaves unclaimed -- every cell, where its pieces go unsaid whole -- or its
// framed line where that says more, or the passes after it (read_the_rest);
// none where a replayed standalone rule already redraws every output.
inline int cells_left_unsaid(const inference& read) {
    const auto& readings = read.correspondence.readings;
    if (readings.empty()) return numeric_limits<int>::max();
    const auto& chosen = readings[read.correspondence.chosen];
    if (ranges::any_of(chosen.fired, [](const auto& rule) { return rule.stands_alone; })) return 0;
    int unsaid = pieces_go_unsaid(read) ? read.correspondence.cells : chosen.unclaimed();
    if (read.framed && !read.framed->correspondence.readings.empty()) {
        const auto& inside = read.framed->correspondence;
        const auto& line = inside.readings[inside.chosen];
        if (read.framed->pieces_redraw || !line.line.empty()) unsaid = min(unsaid, line.unclaimed());
        if (read.framed->rest_says_more) unsaid = min(unsaid, cells_left_unsaid(*read.framed->rest));
    }
    // What the passes after the line leave unsaid, read on the picture it drew.
    if (read.rest_says_more) unsaid = min(unsaid, cells_left_unsaid(*read.rest));
    return unsaid;
}

// HOW MANY PASSES A READING SAYS: its own, then each after it that says more
// (read_the_rest), the last one's number.
inline int passes_said(const inference& read) {
    return read.rest_says_more && read.rest ? passes_said(*read.rest) : read.pass;
}

// THE CANDIDATE WHOSE RULES SAY THE MOST: each one read in the order given, and
// the one whose reading leaves the fewest cells unsaid kept, a tie going to the
// one read first. Once one leaves none, no later one is read. The cuts compete
// here (read_figures), before the passes, so each is judged by its line alone.
// Whatever orders the candidates only breaks a tie.
// Hands back the one kept, where it stood, and the cells each one left unsaid,
// -1 for one never read.
struct said_most {
    inference reading;
    int kept = -1;
    vector<int> left;
};
template <class Read>
inline said_most says_the_most(int count, Read read) {
    said_most most;
    most.left.assign((size_t)count, -1);
    for (int i = 0; i < count; ++i) {
        inference one = read(i);
        most.left[i] = cells_left_unsaid(one);
        if (most.kept < 0 || most.left[i] < most.left[most.kept]) {
            most.reading = std::move(one);
            most.kept = i;
        }
        if (most.left[most.kept] == 0) break;
    }
    return most;
}

// Exact, self-delimiting picture data; tile boundaries affect segmentation too.
inline string figures_key(const vector<example>& train, const tiled_pieces& tiles) {
    string key;
    for (const auto& pair : train) {
        for (const grid* picture : { &pair.input, &pair.output }) {
            key.push_back((char)picture->width);
            key.push_back((char)picture->height);
            for (int row = 0; row < picture->height; ++row)
                for (int col = 0; col < picture->width; ++col)
                    key.push_back((char)picture->data[row][col]);
        }
        const auto tile = tiles.of(pair);
        key.push_back((char)tile.width);
        key.push_back((char)tile.height);
    }
    return key;
}

// One puzzle owns completed results, not in-progress readings or mutable claims.
// Eviction changes reuse only; readers still owning a result keep it alive.
struct figure_readings {
    map<string, shared_ptr<const inference>> completed;
    vector<string> oldest_first;
    void keep(string key, shared_ptr<const inference> read) {
        constexpr size_t capacity = 8;
        if (completed.contains(key)) return;
        if (oldest_first.size() == capacity) {
            completed.erase(oldest_first.front());
            oldest_first.erase(oldest_first.begin());
        }
        oldest_first.push_back(key);
        completed.emplace(std::move(key), std::move(read));
    }
};

inline void read_the_cut(inference& inf, const vector<example>& train, const vector<figure_pair>& pairs,
                         const tiled_pieces& tiles,
                         figure_readings& readings);
inline void read_the_rest(inference& inf, const vector<example>& train, figure_readings& readings);

// THE READING OF FIGURES: pairs with no ground in them -- a plan deleted it,
// or keeps every color as a figure -- read with no election and no ground in
// any reader, so every detector sees figures and empty cells, and an empty cell
// is nothing (where a plan deleted the ground, what it fills last). What reads
// no cut is asked once; what reads one is asked of each cut the rules weigh.
// `pass` is which pass this reading is, and once its cut is chosen the rest of
// the change is read on the picture its line draws (read_the_rest).
inline inference prepare_figures(const vector<example>& train, const vector<figure_pair>& pairs,
                                 const tiled_pieces& tiles, const turning_shape& shape) {
    const time_into clock{ inference_time.ms[inference_times::preparing] };
    const auto started = preparation_audit ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    const auto matching_before = matching_work;
    ++inference_work.preparations;
    inference inf;
    inf.size = output_dimension_rule::infer_rule(train);
    // A fixed or scaled size, read off the sizes alone, gives way to what is
    // drawn where a covered pattern's patch redraws every output so. Settled
    // before any other rule reads the size.
    inf.covered_pattern = covered_pattern_rule::detect(pairs, inf.size);
    if (!inf.covered_pattern.holds() && (inf.size.Chosen == output_dimension_rule::Fixed ||
                                         inf.size.Chosen == output_dimension_rule::Scale)) {
        const output_dimension_rule drawn{ .Chosen = output_dimension_rule::BoundingBox };
        if (const covered_pattern_rule patch = covered_pattern_rule::detect(pairs, drawn); patch.holds()) {
            inf.covered_pattern = patch;
            inf.size = drawn;
        }
    }
    inf.colors = color_facts::detect(train);  // rule-independent
    // No reader is handed a ground color: the ground is empty where a plan
    // deleted it, and a color like any other where a plan kept it.
    inf.colors.detect_output_colors_by_size(train);
    // The frame before the figures: whether this input is really several grids
    // side by side, walls and blank lines told apart by what is on them -- an
    // empty line is blank -- ahead of everything that would read the panes.
    inf.pane_split = pane_split_facts::detect(train);
    // What the panes come to, once there are panes. It reads the masks, not the
    // colors, so a combination the pairs never showed is still one it can answer.
    inf.pane_mask = pane_mask_rule::detect(train, pairs, inf.pane_split);
    // And when the answer is a pane rather than something made from the panes,
    // the only question left is which one. Asked after the mask, because making
    // a drawing is the stronger claim and should get first refusal.
    inf.pane_pick = pane_pick_rule::detect(train, pairs, inf.pane_split);
    // And when no one pane is the answer and nothing is made of them together,
    // every pane may be: its drawing kept, cropped, and laid where the pane was.
    inf.pane_gather = pane_gather_rule::detect(train, pairs, inf.pane_split, inf.size);
    // A noisy lattice cleaned by the vote. It reads the whole input, not the
    // pieces, and the ground the op reads off it (denoise_ground).
    inf.periodic_denoise = periodic_denoise_rule::detect(train, pairs);
    inf.copy_to = copy_to_rule::detect(train, pairs, inf.pane_split, tiles);
    inf.interior_fill = interior_fill_rule::detect(train, pairs, tiles);
    inf.interior_spread = interior_spread_rule::detect(train, pairs, tiles);
    inf.ranked_layout = ranked_layout_rule::detect(train, pairs, inf.size);
    inf.rarest_color = rarest_color_rule::detect(train, pairs, inf.size);
    inf.corner_crop = corner_crop_rule::detect(train, pairs, inf.size);

    // Dims is detector #0: its rule decides which questions even apply
    // (Backward Deduction step 1, applied to detectors instead of ops).
    // Identity = input and output overlay cell-for-cell, so the whole
    // pair-relation toolbox below is meaningful. Each detector re-asks the
    // same question itself (pairs_overlay), so this gate states the intent
    // and none of them depends on being called from here.
    if (inf.size.Chosen == output_dimension_rule::Identity) {
        inf.hole_colors = hole_color_rule::detect(train, tiles);
        inf.ray_extension = ray_extension_rule::detect(train);
        inf.unchanged   = unchanged_cells_facts::detect(train);
        inf.color_diff  = color_diff_facts::detect(train);
        inf.color_swaps = color_swap_candidates::detect(train, inf.color_diff);
    }
    // Scale / Fixed / BoundingBox: no rule-specific detectors yet.
    // Scale could later reuse unchanged_cells through the ratio (cell<->block).

    // Edge level of correspondence: the figures cut up every way that competes,
    // ranked by how well the pieces match.
    inf.correspondence = correspondence_election::infer(train, tiles, shape);
    if (preparation_audit)
        preparation_audit(train, tiles,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
            matching_work.calls - matching_before.calls,
            matching_work.fitting_edges - matching_before.fitting_edges);
    return inf;
}

// The facts and matching belong to the figures, before any rule changes them.
// Each reading takes its own copy; its claims, frames and later pictures stay its own.
// The pairs whole, with their figures, stay with the caller and remain available while rules read.
inline inference read_figures(const vector<example>& train, const vector<figure_pair>& pairs,
                              const tiled_pieces& tiles, int8_t pass,
                              bool partners, inference inf, figure_readings& readings) {
    ++inference_work.readings;
    inference_work.later_readings += pass > 1;
    inf.pass = pass;
    inf.partners = partners;
    // THE CUT IS CHOSEN HERE, just above the first rule that reads it: each cut
    // in the matching's order is read to the end of its line, and the one whose
    // rules say the most is said (says_the_most).
    const vector<correspondence_election::reading_index> ranked = inf.correspondence.ranked;
    if (ranked.empty()) {
        read_the_cut(inf, train, pairs, tiles, readings);
        read_the_rest(inf, train, readings);
        return inf;
    }
    said_most said = says_the_most((int)ranked.size(), [&](int rank) {
        inference read = inf;
        read.correspondence.choose((size_t)rank);
        read_the_cut(read, train, pairs, tiles, readings);
        return read;
    });
    inference best = std::move(said.reading);
    best.correspondence.left_by_rank = said.left;
    // A tied opening the rules ruled out -- read, and leaving more cells
    // unsaid -- is no longer one the reading could not rule out.
    erase_if(best.correspondence.tied_with_chosen, [&](correspondence_election::reading_index tie) {
        const auto at = ranges::find(ranked, tie);
        const int left = at == ranked.end() ? -1 : said.left[(size_t)(at - ranked.begin())];
        return left > said.left[(size_t)said.kept];
    });
    read_the_rest(best, train, readings);
    return best;
}

// THE GROUND OF EACH INPUT a plan that reads each input's own (per_grid) takes
// off: `elected`, where the grounds were elected already, else each input's own.
inline vector<int8_t> plan_grounds(const vector<example>& train, const ground_plan& plan,
                                   const vector<int8_t>& elected = {}) {
    vector<int8_t> grounds = elected;
    if (grounds.empty() && (plan.deleted.per_grid || plan.filled.per_grid))
        for (const example& pair : train)
            grounds.push_back(confident_background(elect_grid_background(pair.input)));
    return grounds;
}

// THE PLAN READ: the first the election allows (ground_plans) that leaves every
// output something to draw where the size is drawn. Under BoundingBox the
// canvas closes on what was painted (draw_on_canvas), so an output a plan
// empties whole is one no drawing can be: 1190e5a7's outputs are all ground,
// one cell a room.
inline ground_plan plan_to_read(const vector<example>& train, const background_election& election) {
    const vector<ground_plan> plans = ground_plans(election);
    if (output_dimension_rule::infer_rule(train).Chosen != output_dimension_rule::BoundingBox) return plans.front();
    for (const ground_plan& plan : plans)
        if (ranges::none_of(plan_figures(train, plan, plan_grounds(train, plan)),
                            [](const figure_pair& pair) { return pair.output_figures.empty(); }))
            return plan;
    return plans.front();
}

// A reading inside the puzzle's first one -- its framed reading, the passes
// after it -- cuts by the shape that one learned (`inherited_shape`), the one
// the program carries; the first reading, handed none, learns it off the pairs
// as given, before the plan deletes their ground (changed_shape), so a shape
// whose places turn into the ground is learned as well.
inline shared_ptr<const inference> read_with_ground(const vector<example>& train, const tiled_pieces& tiles,
        int8_t pass, bool partners, figure_readings& readings, const ground_plan& plan = {},
        const vector<int8_t>& elected = {}, const turning_shape* inherited_shape = nullptr) {
    string key = figures_key(train, tiles);
    for (const background_source side : {plan.deleted, plan.filled}) {
        key.push_back((char)side.color);
        key.push_back((char)side.per_grid);
    }
    key.push_back((char)plan.fill_is_ground_too);
    key.push_back((char)elected.size());
    for (const int8_t color : elected) key.push_back((char)color);
    key.push_back((char)pass);
    key.push_back((char)partners);
    key.push_back((char)(inherited_shape != nullptr));
    if (inherited_shape) {
        const object& cells = inherited_shape->cells;
        key.push_back((char)inherited_shape->from);
        key.push_back((char)inherited_shape->into);
        key.push_back((char)cells.width);
        key.push_back((char)cells.height);
        for (int row = 0; row < cells.height; ++row)
            for (int col = 0; col < cells.width; ++col) key.push_back((char)cells.data[row][col]);
    }
    if (const auto found = readings.completed.find(key); found != readings.completed.end())
        return found->second;
    const vector<int8_t> grounds = plan_grounds(train, plan, elected);
    const vector<figure_pair> pairs = plan_figures(train, plan, grounds);
    const vector<example> figures = figures_only(pairs);
    const tiled_pieces figure_tiles{figures.data(), tiles.originals};
    inference facts = prepare_figures(figures, pairs, figure_tiles,
                                      inherited_shape ? *inherited_shape : changed_shape(train));
    facts.plan = plan;
    facts.elected_backgrounds = grounds;
    auto read = make_shared<const inference>(read_figures(figures, pairs, figure_tiles, pass, partners,
        std::move(facts), readings));
    readings.keep(std::move(key), read);
    return read;
}

// THE PASSES A READING MAKES AT MOST: the first, and nine after it.
constexpr int8_t MOST_PASSES = 10;

// THE REST OF THE CHANGE, READ ON THE PICTURE SO FAR (the user: build on the
// output so far). Where a pass leaves cells unsaid and its line draws something
// new, the picture it draws is the input of a puzzle of its own, from each
// picture to its output, read by every rule a puzzle has: the next pass. It is
// said where it leaves fewer cells unsaid, and otherwise kept for the readout
// alone; a pass that draws nothing new ends the passes, as the tenth does.
// Every rule is read on the picture the program hands it, so a pass after the
// first says one change (finish_the_line).
inline void read_the_rest(inference& inf, const vector<example>& train, figure_readings& readings) {
    if (inf.pass >= MOST_PASSES || inf.picture.size() != train.size() || cells_left_unsaid(inf) == 0) return;
    vector<example> rest = train;
    bool drew = false;
    for (size_t which = 0; which < train.size(); ++which) {
        rest[which].input = inf.picture[which];
        drew = drew || !(inf.picture[which] == train[which].input);
        const int color = inf.plan.filled.per_grid ? inf.elected_backgrounds[which] : inf.plan.filled.color;
        if (is_color(color))
            for (grid* picture : {&rest[which].input, &rest[which].output})
                for (int row = 0; row < picture->height; ++row)
                    for (int col = 0; col < picture->width; ++col)
                        if (picture->data[row][col] == TRANSPARENT) picture->data[row][col] = (int8_t)color;
    }
    if (!drew) return;
    // A paint is still a figure on the next pass, even if it now fills most
    // of the picture. Keep the chosen ground rather than electing it again.
    inf.rest = read_with_ground(rest, {}, (int8_t)(inf.pass + 1), inf.partners, readings,
        ground_plan{inf.plan.filled, inf.plan.filled, true}, inf.elected_backgrounds,
        &inf.correspondence.shape);
    inf.rest_says_more = cells_left_unsaid(*inf.rest) < cells_left_unsaid(inf);
}

// THE RULES THAT READ THE CUT, asked of the one `inf` reads on: the rules read
// off its pieces, then -- once every rule is known -- the rivals, the rest of
// its line, what no rule on it says, and a size change read through its frame.
inline void read_the_cut(inference& inf, const vector<example>& train, const vector<figure_pair>& pairs,
                         const tiled_pieces& tiles,
                         figure_readings& readings) {
    inf.motif_continuation = motif_continuation_rule::detect(train, inf.correspondence);
    // The same continuation read along the diagonals. Asked after the motif
    // because a repeating box is the plainer claim and should get first refusal.
    inf.diagonal_continuation = diagonal_continuation_rule::detect(train, pairs, inf.correspondence);
    inf.piece_slam = piece_slam_rule::detect(train, pairs, inf.correspondence, tiles);
    inf.overlap_corner = overlap_corner_rule::detect(train, pairs, inf.correspondence, tiles);
    // Pieces that slid up against the ones that stayed. Asked after the chain
    // because a layout that puts every piece somewhere new is the larger claim
    // and should get first refusal; this one leaves most of the grid alone.
    inf.slide_to_touch = slide_to_touch_rule::detect(train, pairs, inf.correspondence, tiles);
    // Pieces that slide to the canvas edge, over whatever is in the way. It reads
    // its own cut when the elected one joins the slider to what it passes over.
    inf.edge_slide = edge_slide_rule::detect(train, pairs, inf.correspondence.segmentation, inf.size, tiles);
    inf.project_on_seeds = project_on_seeds_rule::detect(train, pairs, inf.correspondence, tiles);
    inf.snap_to_anchors = snap_to_anchors_rule::detect(train, pairs, inf.correspondence, inf.size, tiles);
    // Gaps between pieces of one colour, filled. It reads its own cut when the
    // elected one cannot say it, so it does not wait on the election's verdict.
    inf.bridge = bridge_rule::detect(train, pairs, inf.correspondence, tiles);
    // Marks laid as lines that take turns down the grid. Like the bridge, it
    // falls back on a cut of its own when the elected one cannot say it.
    inf.mark_lattice = mark_lattice_rule::detect(train, pairs, inf.correspondence, tiles);
    inf.input_blocks = input_blocks_rule::detect(train, pairs, inf.correspondence);
    // THE PUZZLE HAS ONE TABLE, the first pass's: the program carries that one
    // (environment::draw), so a pass after it reads none (e760a62e's third pass
    // named its pieces' colors by a symmetry table the program never had).
    if (inf.pass == 1 && inf.size.Chosen == output_dimension_rule::Identity) {
        inf.shape_color_map = shape_color_map_rule::detect_body_recolor(train, inf.correspondence, tiles);
        // Fallback: same shape->color table, but keyed on each object's own
        // shape and applied as an interior fill (ShapeToInteriorColorMap)
        // instead of a body recolor. Only run when the body-recolor
        // hypothesis didn't validate; the winner is recorded in claimed_by,
        // which is what tells the two ops apart downstream.
        if (!inf.shape_color_map.holds())
            inf.shape_color_map = shape_color_map_rule::detect_interior_fill(train, tiles);
    }

    // PIECES THAT DO NOT REDRAW THE OUTPUT where they lie: the cut's pieces,
    // matched, can leave nothing over and still not be the output -- 2013d3e2's
    // figure matched whole while the output is its corner, cce03e0d's copies
    // matched where half of them are gone -- so a line with no rules is held to
    // what it would draw. Asked of the cut alone, before the line is chosen.
    inf.pieces_redraw = redraws_every_output(train, pairs, [&](const example& pair, grid& drawn) {
        const auto pieces = pieces_as_said(pair.input, inf.correspondence.segmentation, tiles.of(pair),
                                           &inf.correspondence.shape);
        return !pieces.empty() && draw_on_canvas(pair.input, inf.size, drawn, [&](grid& canvas) {
            for (const object& piece : pieces)
                paint_piece(canvas, piece, piece.row, piece.col, true, /*clip=*/true);
            return true;
        });
    });

    if (inf.pass == 1) list_the_rivals(inf);
    finish_the_line(inf, train, pairs, tiles);
    inf.correspondence.list_what_is_missing();

    // A SIZE CHANGE, READ THROUGH ITS FRAME: once the frame cuts the output's
    // grid out of the input's, what is left is a same-size puzzle, read by every
    // rule a same-size puzzle has -- from each framed input to its output. A
    // pass after the first reads a picture its output's size.
    if (inf.pass > 1) return;
    inf.frame = choose_piece_frame(train, pairs, inf, tiles);
    if (inf.frame.holds()) {
        using kind = piece_frame_rule::kind_of;
        // The frames that take the pieces a pick takes off a cut of their own;
        // the rest take the whole input.
        const bool picks = inf.frame.kind == kind::cut_out || inf.frame.kind == kind::laid_out ||
                           inf.frame.kind == kind::filled;
        // The framed grid is what the frame leaves of the figures: nothing is
        // filled in it, and the reading inside it is a reading of figures too.
        const piece_pick& pick = inf.frame.pick;
        vector<example> framed(train.size());
        bool all = true;
        for (size_t which = 0; which < train.size() && all; ++which) {
            framed[which].output = train[which].output;
            const grid& input = train[which].input;
            grid& into = framed[which].input;
            const vector<object> pieces = picks ? inf.frame.pieces(input, tiles.of(train[which])) : vector<object>{};
            bool several = false;
            switch (inf.frame.kind) {
            case kind::laid_out:
                all = laid_out_input(pieces, pick, inf.frame.factor, input, inf.size, into);
                break;
            case kind::filled:
                all = filled_input(pieces, pick, inf.frame.keyed, inf.shape_color_map, input, inf.size,
                                   into);
                break;
            case kind::cut_out:
                all = framed_input(pieces, pick, input, inf.size, into, several);
                break;
            default: all = framed_whole(input, inf.frame, inf.size, into);
            }
        }
        if (!all) inf.frame = {};
        else {
            // Split before Tile: each copy keeps the original input's boundaries.
            const tiled_pieces copies = inf.frame.kind == kind::tiled
                ? tiled_pieces{ framed.data(), train.data() } : tiled_pieces{};
            inf.framed = read_with_ground(framed, copies, 1, inf.partners, readings, {}, {},
                                          &inf.correspondence.shape);
        }
    }
}

// Asks every question about the training pairs, in order, and returns the answers.
// THE GROUND IS CHOSEN ONCE: the election names the plans a puzzle allows
// (ground_plans) and the first that leaves every output something to draw is
// read (plan_to_read, read_figures). Each pass applies that plan before reading
// its figures.
inline inference infer(const vector<example>& train) {
    figure_readings readings;
    const background_election election = elect_background(train);
    inference read;
    {
        const time_into clock{ inference_time.ms[inference_times::plain] };
        read = *read_with_ground(train, {}, 1, true, readings, plan_to_read(train, election));
    }
    read.ground_election = election;
    return read;
}

}  // namespace arc
