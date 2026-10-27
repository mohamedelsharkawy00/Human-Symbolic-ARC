// The reading in order: each ground plan, cut and pass read to the end of its
// rules, the one that says the most kept (says_the_most), and infer(), which
// asks them all.
#pragma once
#include "12_line.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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
// the one whose reading leaves the fewest cells unsaid kept -- of those leaving
// as few, the one saying it in the fewest passes (passes_said), the plainer
// telling -- a tie going to the one read first. So once one leaves none in its
// own pass, no later one is read. Every choice the reading makes is made here
// and judged this one way, by its rules: the ground plan (infer), the cut
// (read_figures), each asked just above the first rule that reads it; the passes
// are read once a cut is chosen, so a cut is judged by its line alone, and only
// a plan by its passes too. Whatever orders the candidates only breaks a tie.
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
    int passes = 0;
    for (int i = 0; i < count; ++i) {
        inference one = read(i);
        most.left[i] = cells_left_unsaid(one);
        const int said_in = passes_said(one);
        if (most.kept < 0 || most.left[i] < most.left[most.kept] ||
            (most.left[i] == most.left[most.kept] && said_in < passes)) {
            most.reading = std::move(one);
            most.kept = i;
            passes = said_in;
        }
        if (most.left[most.kept] == 0 && passes == most.reading.pass) break;
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

inline void read_the_cut(inference& inf, const vector<example>& train, const tiled_pieces& tiles,
                         figure_readings& readings);
inline void read_the_rest(inference& inf, const vector<example>& train, figure_readings& readings);

// THE READING OF FIGURES: pairs with no ground in them -- a plan deleted it,
// or keeps every color as a figure -- read with no election and no ground in
// any reader, so every detector sees figures and empty cells, and an empty cell
// is nothing (where a plan deleted the ground, what it fills last). What reads
// no cut is asked once; what reads one is asked of each cut the rules weigh.
// `pass` is which pass this reading is, and once its cut is chosen the rest of
// the change is read on the picture its line draws (read_the_rest).
inline inference prepare_figures(const vector<example>& train, const tiled_pieces& tiles, const turning_shape& shape) {
    const time_into clock{ inference_time.ms[inference_times::preparing] };
    const auto started = preparation_audit ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    const auto matching_before = matching_work;
    ++inference_work.preparations;
    inference inf;
    inf.size = output_dimension_rule::infer_rule(train);
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
    inf.pane_mask = pane_mask_rule::detect(train, inf.pane_split);
    // And when the answer is a pane rather than something made from the panes,
    // the only question left is which one. Asked after the mask, because making
    // a drawing is the stronger claim and should get first refusal.
    inf.pane_pick = pane_pick_rule::detect(train, inf.pane_split);
    // And when no one pane is the answer and nothing is made of them together,
    // every pane may be: its drawing kept, cropped, and laid where the pane was.
    inf.pane_gather = pane_gather_rule::detect(train, inf.pane_split, inf.size);
    // A noisy lattice cleaned by the vote. It reads the whole input, not the
    // pieces, and the ground the op reads off it (denoise_ground).
    inf.periodic_denoise = periodic_denoise_rule::detect(train);
    inf.copy_to = copy_to_rule::detect(train, inf.pane_split, tiles);
    inf.interior_fill = interior_fill_rule::detect(train, tiles);
    inf.interior_spread = interior_spread_rule::detect(train, tiles);
    inf.ranked_layout = ranked_layout_rule::detect(train, inf.size);
    inf.rarest_color = rarest_color_rule::detect(train, inf.size);
    inf.corner_crop = corner_crop_rule::detect(train, inf.size);

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
inline inference read_figures(const vector<example>& train, const tiled_pieces& tiles, int8_t pass,
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
        read_the_cut(inf, train, tiles, readings);
        read_the_rest(inf, train, readings);
        return inf;
    }
    said_most said = says_the_most((int)ranked.size(), [&](int rank) {
        inference read = inf;
        read.correspondence.choose((size_t)rank);
        read_the_cut(read, train, tiles, readings);
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
        if (ranges::none_of(figures_only(train, plan, plan_grounds(train, plan)), [](const example& pair) {
                bool drawn = false;
                pair.output.for_each_colored_cell(TRANSPARENT, [&](int, int, int) { drawn = true; });
                return !drawn;
            }))
            return plan;
    return plans.front();
}

// THE WALLS OF A GRID OF FIGURES: its one-color pieces, joined at sides or
// corners as Split [1] joins them, that are walls (is_wall) -- each cell of
// each handed to `see(row, col)`.
template <class See>
inline void each_wall_cell(const grid& figures, See see) {
    object_walk_scratch scratch;
    each_object(figures, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& found, const int16_t* cells, int count) {
        const object piece = object::from_cells(figures, found, cells, count);
        if (is_wall(figures.width, figures.height, found.box(),
                    [&](int row, int col) { return piece.holds(row, col); }))
            piece.each_cell([&](int row, int col, int8_t) { see(row, col); });
    });
}

// THE FIGURES WITH THEIR WALLS SET APART (ground_plan::walls_apart): each
// pair's input loses its walls, and its output the same cells, which show them
// as they were, so the reading reads what lies between the walls and the
// program lays them back last (walls_laid_back). False, the figures left as
// they were, where some pair has no wall, or an output of another size, or one
// that changes a wall.
inline bool set_the_walls_apart(vector<example>& figures) {
    vector<vector<int16_t>> walls(figures.size());
    for (size_t which = 0; which < figures.size(); ++which) {
        const example& pair = figures[which];
        if (!pair.input.same_size(pair.output)) return false;
        each_wall_cell(pair.input, [&](int row, int col) { walls[which].push_back((int16_t)cell_index(row, col)); });
        if (walls[which].empty()) return false;
        for (const int16_t cell : walls[which])
            if (pair.output.data[cell_row(cell)][cell_col(cell)] != pair.input.data[cell_row(cell)][cell_col(cell)])
                return false;
    }
    for (size_t which = 0; which < figures.size(); ++which)
        for (const int16_t cell : walls[which])
            figures[which].input.data[cell_row(cell)][cell_col(cell)] =
                figures[which].output.data[cell_row(cell)][cell_col(cell)] = (int8_t)TRANSPARENT;
    return true;
}

// A reading inside the puzzle's first one -- its framed reading, the passes
// after it -- cuts by the shape that one learned (`inherited_shape`), the one
// the program carries; the first reading, handed none, learns it off the pairs
// as given, before the plan deletes their ground (changed_shape), so a shape
// whose places turn into the ground is learned as well.
inline shared_ptr<const inference> read_with_ground(const vector<example>& train, const tiled_pieces& tiles,
        int8_t pass, bool partners, figure_readings& readings,
        optional<inference>* prepared = nullptr, const ground_plan& plan = {},
        const vector<int8_t>& elected = {}, const turning_shape* inherited_shape = nullptr) {
    string key = figures_key(train, tiles);
    for (const background_source side : {plan.deleted, plan.filled}) {
        key.push_back((char)side.color);
        key.push_back((char)side.per_grid);
    }
    key.push_back((char)plan.fill_is_ground_too);
    key.push_back((char)plan.walls_apart);
    key.push_back((char)elected.size());
    for (const int8_t color : elected) key.push_back((char)color);
    key.push_back((char)pass);
    key.push_back((char)partners);
    key.push_back((char)(inherited_shape != nullptr));
    if (inherited_shape) {
        const object& cells = inherited_shape->cells;
        key.push_back((char)inherited_shape->from);
        key.push_back((char)inherited_shape->into);
        key.push_back((char)cells.summary.width);
        key.push_back((char)cells.summary.height);
        for (int row = 0; row < cells.summary.height; ++row)
            for (int col = 0; col < cells.summary.width; ++col) key.push_back((char)cells.data[row][col]);
    }
    if (const auto found = readings.completed.find(key); found != readings.completed.end())
        return found->second;
    optional<inference> local;
    auto& facts = prepared ? *prepared : local;
    const vector<int8_t> grounds = plan_grounds(train, plan, elected);
    vector<example> figures = figures_only(train, plan, grounds);
    if (plan.walls_apart) set_the_walls_apart(figures);
    const tiled_pieces figure_tiles{figures.data(), tiles.originals};
    if (!facts) {
        facts = prepare_figures(figures, figure_tiles, inherited_shape ? *inherited_shape : changed_shape(train));
        facts->plan = plan;
        facts->elected_backgrounds = grounds;
        for (const example& pair : train) facts->before_ground.push_back(pair.input);
    }
    auto read = make_shared<const inference>(read_figures(figures, figure_tiles, pass, partners,
        prepared ? *facts : std::move(*facts), readings));
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
        nullptr, ground_plan{inf.plan.filled, inf.plan.filled, true}, inf.elected_backgrounds,
        &inf.correspondence.shape);
    inf.rest_says_more = cells_left_unsaid(*inf.rest) < cells_left_unsaid(inf);
}

// THE RULES THAT READ THE CUT, asked of the one `inf` reads on: the rules read
// off its pieces, then -- once every rule is known -- the rivals, the rest of
// its line, what no rule on it says, and a size change read through its frame.
inline void read_the_cut(inference& inf, const vector<example>& train, const tiled_pieces& tiles,
                         figure_readings& readings) {
    inf.motif_continuation = motif_continuation_rule::detect(train, inf.correspondence);
    // The same continuation read along the diagonals. Asked after the motif
    // because a repeating box is the plainer claim and should get first refusal.
    inf.diagonal_continuation = diagonal_continuation_rule::detect(train, inf.correspondence);
    inf.piece_slam = piece_slam_rule::detect(train, inf.correspondence, tiles);
    inf.overlap_corner = overlap_corner_rule::detect(train, inf.correspondence, tiles);
    // Pieces that slid up against the ones that stayed. Asked after the chain
    // because a layout that puts every piece somewhere new is the larger claim
    // and should get first refusal; this one leaves most of the grid alone.
    inf.slide_to_touch = slide_to_touch_rule::detect(train, inf.correspondence, tiles);
    // Pieces that slide to the canvas edge, over whatever is in the way. It reads
    // its own cut when the elected one joins the slider to what it passes over.
    inf.edge_slide = edge_slide_rule::detect(train, inf.correspondence.segmentation, inf.size, tiles);
    inf.project_on_seeds = project_on_seeds_rule::detect(train, inf.correspondence, tiles);
    inf.snap_to_anchors = snap_to_anchors_rule::detect(train, inf.correspondence, inf.size, tiles);
    // Gaps between pieces of one colour, filled. It reads its own cut when the
    // elected one cannot say it, so it does not wait on the election's verdict.
    inf.bridge = bridge_rule::detect(train, inf.correspondence, tiles);
    // Marks laid as lines that take turns down the grid. Like the bridge, it
    // falls back on a cut of its own when the elected one cannot say it.
    inf.mark_lattice = mark_lattice_rule::detect(train, inf.correspondence, tiles);
    inf.input_blocks = input_blocks_rule::detect(train, inf.correspondence);
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
    inf.pieces_redraw = redraws_every_output(train, [&](const example& pair, grid& drawn) {
        const auto pieces = pieces_as_said(pair.input, inf.correspondence.segmentation, tiles.of(pair),
                                           &inf.correspondence.shape);
        return !pieces.empty() && draw_on_canvas(pair.input, inf.size, drawn, [&](grid& canvas) {
            for (const object& piece : pieces)
                paint_piece(canvas, piece, piece.summary.row, piece.summary.col, true, /*clip=*/true);
            return true;
        });
    });

    // Last, once every rule is known: the rivals, the rest of the cut's line,
    // and only then what no rule on it says. THE MATCHING'S ACTS, OR THE RULES
    // ALONE: the acts the matching read off the pieces open the line and the
    // rules read off the pieces go on from the hand they leave, or the acts are
    // left unsaid and the rules say every change themselves. A rule that says
    // what an act says and more is seen at work only on a hand the act has not
    // changed yet: 9565186b's "8 becomes 5" left its first pair nothing for
    // "every color but the most common becomes 5" to change. Whichever says
    // the most is said (says_the_most), a tie keeping the acts. EACH READ
    // AGAIN AHEAD, a paint weighed with what the picture lets be read after it
    // (finish_the_line), and kept only where it leaves fewer cells unsaid: read
    // ahead, 0d87d2a6's bars come before "the largest becomes 1", which left the
    // picture nothing to change in its first pair; but a line read ahead can
    // say less -- ae3edfdc's rays and the reading after them stopped its moves.
    // The picture reads recolors alone, so a line is read ahead only where read
    // plainly it left a recolor unsaid.
    // A PASS AFTER THE FIRST (read_the_rest) says one change in one sentence:
    // no rival, whose program is its own, and no reading ahead, which says a
    // second change after the first.
    if (inf.pass == 1) list_the_rivals(inf);
    const auto& election = inf.correspondence;
    const bool acts = !election.readings.empty() && !election.readings[election.chosen].line.empty();
    const int openings = acts ? 2 : 1;
    vector<inference> plain((size_t)openings);
    auto recolor_unsaid = [](const inference& one) {
        const auto& said = one.correspondence;
        return !said.readings.empty() && ranges::any_of(said.readings[said.chosen].pairs, [](const auto& pair) {
            return ranges::any_of(pair.left_over, [](const auto& cell) {
                return cell.claimed_by < 0 && cell.answers_to(correspondence_election::missing::a_recolor); });
        });
    };
    cut_paints paints;
    inf = says_the_most(inf.pass == 1 ? 2 * openings : openings, [&](int which) {
        const int opening = which % openings;
        const bool ahead = which >= openings;
        if (ahead && !recolor_unsaid(plain[(size_t)opening])) return plain[(size_t)opening];
        inference read = inf;
        if (opening == 1) leave_the_acts_unsaid(read);
        finish_the_line(read, train, tiles, ahead, paints);
        if (!ahead) plain[(size_t)opening] = read;
        return read;
    }).reading;
    inf.correspondence.list_what_is_missing();

    // A SIZE CHANGE, READ THROUGH ITS FRAME: once the frame cuts the output's
    // grid out of the input's, what is left is a same-size puzzle, read by every
    // rule a same-size puzzle has -- from each framed input to its output. A
    // pass after the first reads a picture its output's size.
    if (inf.pass > 1) return;
    inf.frame = choose_piece_frame(train, inf, tiles);
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
            inf.framed = read_with_ground(framed, copies, 1, inf.partners, readings, nullptr, {}, {},
                                          &inf.correspondence.shape);
        }
    }
}

// Asks every question about the training pairs, in order, and returns the answers.
// THE GROUND IS CHOSEN ONCE: the election names the plans a puzzle allows
// (ground_plans) and the first that leaves every output something to draw is
// read (plan_to_read, read_figures) -- then, where every pair has walls that
// stand as they were, the same plan with its walls set apart
// (set_the_walls_apart), and the one whose rules say the most, in the fewest
// passes, is said (says_the_most), a tie keeping the walls in. Each pass
// applies that plan before reading its figures.
inline inference infer(const vector<example>& train) {
    figure_readings readings;
    const background_election election = elect_background(train);
    const ground_plan elected = plan_to_read(train, election);
    vector<ground_plan> plans{ elected };
    if (vector<example> figures = figures_only(train, elected, plan_grounds(train, elected));
        !elected.keeps() && set_the_walls_apart(figures)) {
        plans.push_back(elected);
        plans.back().walls_apart = true;
    }
    const int count = (int)plans.size();
    vector<optional<inference>> prepared((size_t)count);
    auto read_plan = [&](int at, bool partners) {
        return *read_with_ground(train, {}, 1, partners, readings, &prepared[(size_t)at], plans[(size_t)at]);
    };
    vector<inference::plan_read> plans_read;
    // Each plan read, as the readout shows it: the rules its line said.
    auto note_read = [&](const ground_plan& plan, const inference& read) {
        string rules;
        if (!read.correspondence.readings.empty()) {
            const auto& line = read.correspondence.readings[read.correspondence.chosen];
            for (const int8_t said : line.line)
                rules += (rules.empty() ? "" : " | ") + string(line.fired[said].name) +
                         (line.fired[said].says.empty() ? "" : " " + line.fired[said].says);
        }
        plans_read.push_back({ plan, cells_left_unsaid(read), passes_said(read), rules, read.partners });
    };
    inference best = says_the_most(count, [&](int which) {
        const ground_plan& plan = plans[(size_t)which];
        inference read;
        {
            const time_into clock{ inference_time.ms[inference_times::plain] };
            read = read_plan(which, true);
        }
        read.plan = plan;
        note_read(plan, read);
        return read;
    }).reading;
    best.ground_election = election;
    best.plans_read = std::move(plans_read);
    return best;
}

}  // namespace arc
