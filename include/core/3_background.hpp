// This file tackles the concept of a background in ARC,
// which is frequent but not in all puzzles,
// but in the puzzles where it is found,
// it is important to be distinguished from the rest of the puzzle.
// Background is elected per grid role: inputs and outputs are judged separately.
#pragma once
#include "1_readings.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace arc {
using namespace std;
// Several colors may look like the background, so we keep a list of candidates,
// with the best candidate first. Inputs and outputs have separate lists:
// a puzzle may change the background color between input and output.
// Keeping the other candidates lets us try another background later.
// These values control how we compare candidates. They were chosen to match
// 43 hand-checked puzzles, once kept in tests/background_truth.txt (removed
// 2026-10-05; it is in git history).
// Shares use a scale of 0 to 1000: 500 means 50%, and 1000 means 100%.

// At least half of the measured amount.
inline constexpr int BG_HALF = 500;

// A clear lead means having at least twice as much as the other candidate.
inline constexpr int BG_DECISIVE = 2;

// Treat two amounts as nearly tied when the smaller is at least 4/5 (80%)
// of the larger. Compare small * 5 >= large * 4 to avoid division.
inline constexpr int BG_NEAR_TIE_NUM = 4;
inline constexpr int BG_NEAR_TIE_DEN = 5;

// The score must reach 700 for a candidate to qualify as a background.
inline constexpr int BG_FLOOR = 700;

// Each kind of evidence contributes to the score. A larger weight gives
// that evidence more influence.
inline constexpr double BG_W_AREA = 1.0;      // How much area the color covers.
inline constexpr double BG_W_STANDOUT = 0.3;  // How often it wins in individual grids.
inline constexpr double BG_W_PRIOR = 0.15;    // The preference for this color as a background.
inline constexpr double BG_W_EMBEDS = 0.25;   // How much figure area sits inside this color.
inline constexpr double BG_W_PUNCHED = 0.2;   // Evidence of a background showing through gaps.

// Extra points if the color also appears in the outputs.
inline constexpr int BG_PERSIST_BONUS = 50;

// Hypotheses: plural.
struct background_election {
    struct hypothesis {
        // -1 = "no background"
        int8_t color = -1;

        // Per mille of all foreground figure area that sits embedded in this
        // hypothesised background color, for the worst example grid
        int embeds = 0;

        // Per mille of all cells in the grid painted in this hypothesised
        // background color, for the worst example grid
        int dominance = 0;

        // Did this input color also appear in at least one output?
        // A transformation may delete or recolor the figures, so a figure's
        // color can vanish from the output. The background it sat on usually stays.
        // Worth a small bonus (BG_PERSIST_BONUS, 50 against a floor of 700), so
        // it only separates two candidates that area already left nearly tied.
        bool appears_in_any_output = true;

        // Per mille of all cells in the grid falling inside this color's
        // bounding box, for the worst example grid; everywhere = ~1000
        int span = 0;

        // Per mille of this color's own cells that touch a different color,
        // for the worst example grid; a thin web = ~1000, a solid mass = low
        int gaps = 0;

        // In EVERY grid this color is the quietest present and appears ONLY as
        // multiple enclosed windows — the ground showing through.
        bool underneath = false;

        // Weighted confidence that this color is the background. The election
        // takes the top score if it clears BG_FLOOR, else abstains.
        int score = 0;
    };

    // Background is an elected verdict for each direction:
    // the same election over different grid sets gives legitimately different
    // backgrounds, and the disagreement IS information. Best background first
    // in each.
    // One elected color per train pair, from that pair's INPUT alone
    vector<int8_t> across_input_output;
    vector<hypothesis> across_inputs;          // input grids only
    vector<hypothesis> across_outputs;         // output grids only
    vector<hypothesis> across_inputs_outputs;  // all grids pooled
    // Whether every input lies over its output cell for cell (pairs_overlay),
    // which decides whether a plan's fill is ground on the inputs too.
    bool overlays = false;
    static int8_t top(const vector<hypothesis>& ranked) {
        return ranked.empty() ? (int8_t)-1 : ranked[0].color;
    }
    // THE SPLIT: input background and output background are separate elected verdicts —
    // their relationship is part of the transformation rule (see
    // notes/4 - To Do/Background Per-Side Split.md). There is deliberately NO
    // one-answer merge; ask the side you mean.

    // INPUT-side verdict: the background the input figures sit on — what
    // segmentation excludes. Walks input-derived evidence STRONGEST -> WEAKEST
    // and takes the first real background; -1 = backgroundless or per-pair.
    int8_t input_elected() const {
        // A ground showing through beats the area winner by definition, since
        // the pattern covering it is what wins the area.
        for (const auto& candidate : across_inputs)
            if (candidate.underneath) return candidate.color;
        // The background is given in the input, undistorted by the
        // transformation — unless a confident pair contradicts it.
        if (int8_t color = top(across_inputs); color >= 0 && !input_vetoed) return color;
        // Agreeing pairs name a global color; differing real pairs mean
        // MODE=PER-PAIR (-1 here, across_input_output holds the colors).
        if (int8_t color = per_pair_majority(); color >= 0) return color;
        if (is_per_pair_raw()) return -1;
        // Pooled inputs+outputs, weaker but still needs input agreement
        if (int8_t color = top(across_inputs_outputs); color >= 0) return color;
        return -1;
    }

    // OUTPUT-side verdict: the background the output figures sit on. One that
    // PERSISTS from the input wins before any confidence gate; without that
    // support the winner must be dominance-CONFIDENT. NOT the embeds gate — embeds is a
    // worst-grid number and one grid zeroes it.
    int8_t output_elected() const {
        int8_t input_background = input_elected();
        if (input_background >= 0)
            for (const auto& candidate : across_outputs)
                if (candidate.color == input_background) return input_background;
        if (across_outputs.empty() || across_outputs[0].color < 0) return -1;
        return across_outputs[0].dominance >= BG_HALF
             ? across_outputs[0].color : (int8_t)-1;
    }

    // A color DECISIVELY beaten inside some pair cannot be the global background:
    // there it is a figure sitting ON another ground. Set by elect_background,
    // where each pair's dominances are on hand.
    bool input_vetoed = false;

    // A strict MAJORITY of confident pair grounds names the background — learn from
    // the clear pairs, extend to the flipped ones. Unanimity is the special
    // case; truly distinct pairs have no majority and stay per-pair.
    int8_t per_pair_majority() const {
        int wins[COLOR_COUNT] = {};
        for (int8_t color : across_input_output) if (color >= 0) ++wins[color];
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (wins[color] * 2 > (int)across_input_output.size()) return (int8_t)color;
        return -1;
    }

    // Per-pair test that does NOT consult input_elected() (avoids recursion
    // with the global tier). Same logic as is_per_pair minus that guard.
    bool is_per_pair_raw() const {
        if (across_input_output.size() < 2) return false;
        bool all_real = true, any_differ = false;
        for (int8_t color : across_input_output) all_real = all_real && (color >= 0);
        for (size_t i = 1; i < across_input_output.size(); ++i)
            any_differ = any_differ || (across_input_output[i] != across_input_output[0]);
        return all_real && any_differ;
    }

    // PER-PAIR background mode: the global election abstains, yet EVERY pair
    // has its own real background that DIFFERS across pairs. Constant WITHIN a
    // pair, varying ACROSS pairs. across_input_output holds the colors; this
    // just flags that the mode applies. Detect-only for now — nothing
    // segments per-pair.
    bool is_per_pair() const {
        // Matches input_elected()'s order, so the output-influenced directions
        // cannot suppress a strong per-pair input signal.
        if (int8_t color = top(across_inputs); color >= 0 && !input_vetoed) return false;
        if (per_pair_majority() >= 0) return false;  // a majority = global
        return is_per_pair_raw();
    }
};

// What the side's grids say about each color. Every share is folded WORST-GRID:
// a background has to hold up everywhere, so one bad grid is the honest number.
struct background_evidence {
    // Worst grid, out of 1000
    int embeds[COLOR_COUNT], dominance[COLOR_COUNT], span[COLOR_COUNT], gaps[COLOR_COUNT];
    // In EVERY grid: the quietest color present, showing only as enclosed windows
    bool only_windows[COLOR_COUNT];
    // Grids where this color was the standout background, and how many grids voted
    int standout_wins[COLOR_COUNT] = {};
    int grid_count = 0;
    // Total cells per color over the whole side
    long long global_cells[COLOR_COUNT] = {};

    ARC_HOST_DEVICE background_evidence() {
        for (int i = 0; i < COLOR_COUNT; ++i) {
            embeds[i] = dominance[i] = span[i] = gaps[i] = 1000;
            only_windows[i] = true;
        }
    }
};

// Per-color background-ness prior, most-background first. The LAST tiebreak
// only, when area/embeds/story can't separate two candidates — a prior, not a
// verdict, and structure always overrides it.
ARC_HOST_DEVICE inline int background_rank(int color) {
    constexpr int8_t order[COLOR_COUNT] = { 0, 9, 5, 6, 1, 2, 7, 3, 8, 4 };
    for (int rank = 0; rank < COLOR_COUNT; ++rank) if (order[rank] == color) return rank;
    return COLOR_COUNT;
}

// STAGE 1 — how much of this grid each color holds, how far it reaches, and
// which color reads as the background here.
template <class Grid>
ARC_HOST_DEVICE inline void measure_area_and_reach(const Grid& one_grid, const int cells_of_color[COLOR_COUNT],
                                   background_evidence& evidence) {
    const int area = one_grid.width * one_grid.height;
    for (int i = 0; i < COLOR_COUNT; ++i) evidence.global_cells[i] += cells_of_color[i];
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.dominance[i] = min(evidence.dominance[i], cells_of_color[i] * 1000 / area);

    // Punched-through probes (measured only): span = how much of the grid the
    // color's box covers; gaps = share of its cells touching another color.
    // A ground showing through is everywhere but always thin.
    // ADJACENCY, SO FOUR: a fraying measure asks what BORDERS this cell, and a
    // corner touch is not a border. Not the 8 that grouping uses — see the note
    // beside every_nth_way in 1_readings.hpp before making these agree.
    int touching[COLOR_COUNT] = {};
    cell_box box[COLOR_COUNT];
    color_boxes(one_grid, box);
    walk_colored_cells(one_grid, TRANSPARENT, [&](int row, int col, int color) {
        for (int way = 0; way < heading::count; way += every_nth_way<4>) {
            int next_row = row + step_of(way).down;
            int next_col = col + step_of(way).across;
            if (next_row < 0 || next_row >= one_grid.height || next_col < 0 || next_col >= one_grid.width) continue;
            if (one_grid.data[next_row][next_col] != color) { touching[color]++; return; }
        }
    });
    for (int i = 0; i < COLOR_COUNT; ++i) {
        if (!cells_of_color[i]) { evidence.span[i] = 0; evidence.gaps[i] = 0; continue; }
        int box_area = box[i].area();
        evidence.span[i] = min(evidence.span[i], box_area * 1000 / area);
        evidence.gaps[i] = min(evidence.gaps[i], touching[i] * 1000 / cells_of_color[i]);
    }

    // Standout in this grid: the background among the SUBSTANTIAL colors — those
    // co-dominant with the leader — where the background-ness prior picks the
    // quieter one. Not raw-most-cells: a separator grid inflates a foreground
    // color's area past the true background. Then the corpus VOTES, not unanimity.
    int most_cells = 0;
    for (int i = 0; i < COLOR_COUNT; ++i) most_cells = max(most_cells, cells_of_color[i]);
    int winner = -1;
    for (int i = 0; i < COLOR_COUNT; ++i) {
        if (cells_of_color[i] * BG_NEAR_TIE_DEN < most_cells * BG_NEAR_TIE_NUM) continue;
        if (winner < 0 || background_rank(i) < background_rank(winner)) winner = i;
    }
    if (winner >= 0 && most_cells > 0) evidence.standout_wins[winner]++;
    evidence.grid_count++;
}

// The objects of one grid — same-color cells that touch, corners counting —
// labelled once (the same walk gpu_flood_fill does). `cells` holds every
// object's cells, object by object, with first_cell naming where each starts.
struct object_map {
    int16_t id[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t color[MAX_GRID_DIM * MAX_GRID_DIM], size[MAX_GRID_DIM * MAX_GRID_DIM];
    int16_t first_cell[MAX_GRID_DIM * MAX_GRID_DIM], cells[MAX_GRID_DIM * MAX_GRID_DIM];
    bool on_border[MAX_GRID_DIM * MAX_GRID_DIM];
    int object_count = 0, cell_count = 0;
    ARC_HOST_DEVICE int count() const { return object_count; }
};

template <class Grid>
ARC_HOST_DEVICE inline void label_objects(const Grid& one_grid, object_map& objects) {
    for (int row = 0; row < one_grid.height; ++row)
        for (int col = 0; col < one_grid.width; ++col) objects.id[row][col] = -1;
    object_walk_scratch scratch;
    each_object(one_grid, TRANSPARENT, grouping::touching_same_color, scratch,
        [&](const object_summary& summary, const int16_t* cells, int count) {
            const int color = one_grid.data[cell_row(cells[0])][cell_col(cells[0])];
            if (!is_color(color)) return;
            const int16_t id = (int16_t)objects.count();
            objects.first_cell[id] = (int16_t)objects.cell_count;
            for (int i = 0; i < count; ++i) {
                const int row = cell_row(cells[i]), col = cell_col(cells[i]);
                objects.id[row][col] = id;
                objects.cells[objects.cell_count++] = cells[i];
            }
            objects.color[id] = (int16_t)color;
            objects.size[id] = (int16_t)count;
            objects.on_border[id] = summary.x0 == 0 || summary.y0 == 0 ||
                summary.x0 + summary.w == one_grid.width ||
                summary.y0 + summary.h == one_grid.height;
            ++objects.object_count;
        });
}

// STAGE 2 — each figure votes for the color it sits in, weighted by its own
// cells. FIGURE = an object whose outer contour is majority-owned by ONE other
// object; everything else is BACKDROP. The same pass feeds the UNDERNEATH check:
// a color is believable as ground showing through only if it is the quietest
// present and appears solely as enclosed windows.
template <class Grid>
ARC_HOST_DEVICE inline void measure_embedding(const Grid& one_grid, const int cells_of_color[COLOR_COUNT],
                              background_evidence& evidence) {
    object_map objects;
    label_objects(one_grid, objects);
    int figure_area = 0, embedded_area[COLOR_COUNT] = {};
    int objects_of_color[COLOR_COUNT] = {}, backdrop_objects_of_color[COLOR_COUNT] = {};

    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
    // One vote box per object, indexed by the object's own id — no lookup table to
    // run out of, so no contour can be too crowded to count. `voted` names the
    // boxes this object touched, so clearing costs what was used, not all of them.
    int16_t votes[MAX_GRID_DIM * MAX_GRID_DIM] = {};
    int16_t voted[MAX_GRID_DIM * MAX_GRID_DIM];
    int voted_count = 0;
    for (int this_object = 0; this_object < objects.count(); ++this_object) {
        objects_of_color[objects.color[this_object]]++;
        if (objects.on_border[this_object]) {
            backdrop_objects_of_color[objects.color[this_object]]++; continue;
        }
        // Flood "outside" = everything reachable from the border without
        // entering this object. Four-way space cannot leak through corners
        // joined by the object's eight-way connectivity.
        mark_outside(one_grid.height, one_grid.width,
            [&](int row, int col) { return objects.id[row][col] == this_object; }, outside);
        // Outer contour = the outside cells directly beside this object (edges
        // only, not corners). Every such cell
        // votes for the object it belongs to; a majority makes this a figure
        // sitting in that object.
        int contour_votes_total = 0;
        voted_count = 0;
        for (int nth = 0; nth < objects.size[this_object]; ++nth) {
            int row = cell_row(objects.cells[objects.first_cell[this_object] + nth]);
            int col = cell_col(objects.cells[objects.first_cell[this_object] + nth]);
            for (int way = 0; way < heading::count; way += every_nth_way<4>) {
                int next_row = row + step_of(way).down;
                int next_col = col + step_of(way).across;
                if (next_row < 0 || next_row >= one_grid.height || next_col < 0 || next_col >= one_grid.width) continue;
                if (!outside[next_row][next_col]) continue;
                int owner_object = objects.id[next_row][next_col];
                if (owner_object < 0) continue;
                if (votes[owner_object] == 0) voted[voted_count++] = (int16_t)owner_object;
                votes[owner_object]++;
                contour_votes_total++;
            }
        }
        int best_owner = -1, best_votes = 0;
        for (int i = 0; i < voted_count; ++i) {
            const int owner = voted[i];
            if (votes[owner] > best_votes) { best_votes = votes[owner]; best_owner = owner; }
        }
        for (int i = 0; i < voted_count; ++i) votes[voted[i]] = 0;
        bool is_figure = best_owner >= 0 && best_votes * 2 > contour_votes_total;
        if (!is_figure) { backdrop_objects_of_color[objects.color[this_object]]++; continue; }
        figure_area += objects.size[this_object];
        embedded_area[objects.color[best_owner]] += objects.size[this_object];
    }
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.embeds[i] = min(evidence.embeds[i],
                                 figure_area ? embedded_area[i] * 1000 / figure_area : 0);

    // UNDERNEATH for this grid: quietest color present, and only enclosed
    // windows (>= 2 objects, none touching the border). One enclosed object is
    // just an object; many windows are the ground showing through.
    int quietest = -1;
    for (int i = 0; i < COLOR_COUNT; ++i)
        if (cells_of_color[i] > 0 &&
            (quietest < 0 || background_rank(i) < background_rank(quietest)))
            quietest = i;
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.only_windows[i] = evidence.only_windows[i] && i == quietest &&
                                   objects_of_color[i] >= 2 && backdrop_objects_of_color[i] == 0;
}

// STAGE 3 — windows are believable only when cut into a PATTERNED coverer. The
// coverer is the global area leader; if it is solid, no color may claim
// underneath.
ARC_HOST_DEVICE inline void close_underneath(background_evidence& evidence) {
    int coverer = -1;
    for (int i = 0; i < COLOR_COUNT; ++i)
        if (evidence.global_cells[i] > 0 &&
            (coverer < 0 || evidence.global_cells[i] > evidence.global_cells[coverer]))
            coverer = i;
    if (coverer >= 0 && evidence.gaps[coverer] < BG_HALF)
        for (int i = 0; i < COLOR_COUNT; ++i) evidence.only_windows[i] = false;
}

// STAGE 4 — weighted signals sum to a confidence per color. No single rule is a
// hard gate. Area leads, standout corroborates, prior is a light tiebreak,
// punched-through and embeds are graded votes.
// KNOWN GAP: a background that is a MINORITY in MOST grids is not findable by area
// at all; that needs the object story.
template <class Candidates>
ARC_HOST_DEVICE inline void score_candidates(Candidates& ranked,
                             const background_evidence& evidence, const bool in_every[COLOR_COUNT]) {
    long long total_cells = 0;
    for (int i = 0; i < COLOR_COUNT; ++i) total_cells += evidence.global_cells[i];

    for (auto& candidate : ranked) {
        if (candidate.color < 0 || !in_every[candidate.color]) { candidate.score = 0; continue; }
        const int color = candidate.color;
        const int global_area =
            total_cells ? (int)(evidence.global_cells[color] * 1000 / total_cells) : 0;
        const int standout_share = evidence.grid_count
            ? evidence.standout_wins[color] * 1000 / evidence.grid_count : 0;
        // Most-background 1000 .. least 100
        const int prior = 1000 - background_rank(color) * 100;
        // Everywhere AND thin, believable as ground only for a quiet color
        const int punched = min(candidate.span, candidate.gaps);

        // Summed in this order on purpose: floating-point addition does not
        // associate, so reordering these can move the total across the floor.
        const double from_area     = global_area    * BG_W_AREA;
        const double from_standout = standout_share * BG_W_STANDOUT;
        const double from_prior    = prior          * BG_W_PRIOR;
        const double from_persists = candidate.appears_in_any_output ? BG_PERSIST_BONUS : 0;
        const double from_punched  = punched * BG_W_PUNCHED * prior / 1000;
        const double from_embeds   = candidate.embeds * BG_W_EMBEDS;

        candidate.score = (int)(from_area + from_standout + from_prior +
                                from_persists + from_punched + from_embeds);
    }
}

// STAGE 5 — best background first. The top score elects IF it clears BG_FLOOR,
// else the direction abstains (-1 sorts first) — the honest "ambiguous" answer.
template <class Candidates>
ARC_HOST_DEVICE inline void rank_candidates(Candidates& ranked) {
    const auto before = [](const auto& left, const auto& right) {
        const bool left_electable  = (left.color >= 0 && left.score >= BG_FLOOR);
        const bool right_electable = (right.color >= 0 && right.score >= BG_FLOOR);
        if (left_electable != right_electable) return left_electable;
        if (!left_electable) {                   // none electable: -1 first
            if ((left.color < 0) != (right.color < 0)) return left.color < 0;
            return background_rank(left.color) < background_rank(right.color);
        }
        return left.score > right.score;         // among electables, top score
    };
    // Stable insertion sort: at most eleven candidates, on either processor.
    for (int i = 1; i < (int)ranked.size(); ++i) {
        auto candidate = ranked[i];
        int j = i;
        while (j > 0 && before(candidate, ranked[j - 1])) {
            ranked[j] = ranked[j - 1];
            --j;
        }
        ranked[j] = candidate;
    }
}

// The same candidate construction and arithmetic for pooled and local elections.
template <class Candidates>
ARC_HOST_DEVICE inline void finish_background_election(Candidates& ranked,
        const background_evidence& evidence, const bool in_every[COLOR_COUNT],
        const bool* persists = nullptr) {
    ranked.push_back({ .color = -1, .appears_in_any_output = false });  // "none" always a candidate
    for (int candidate = 0; candidate < COLOR_COUNT; ++candidate) {
        if (!in_every[candidate]) continue;
        ranked.push_back({ .color = (int8_t)candidate,
                           .embeds = evidence.embeds[candidate],
                           .dominance = evidence.dominance[candidate],
                           .appears_in_any_output = persists ? persists[candidate] : true,
                           .span = evidence.span[candidate],
                           .gaps = evidence.gaps[candidate],
                           .underneath = evidence.only_windows[candidate] });
    }

    score_candidates(ranked, evidence, in_every);
    rank_candidates(ranked);
}

// A local election needs no allocation and no training pairs.
struct local_background_candidates {
    background_election::hypothesis values[COLOR_COUNT + 1];
    int used = 0;
    ARC_HOST_DEVICE int size() const { return used; }
    ARC_HOST_DEVICE auto* begin() { return values; }
    ARC_HOST_DEVICE auto* end() { return values + used; }
    ARC_HOST_DEVICE const auto* begin() const { return values; }
    ARC_HOST_DEVICE const auto* end() const { return values + used; }
    ARC_HOST_DEVICE auto& operator[](int i) { return values[i]; }
    ARC_HOST_DEVICE void push_back(background_election::hypothesis h) { values[used++] = h; }
};

template <class Grid>
ARC_HOST_DEVICE inline local_background_candidates elect_grid_background(const Grid& one_grid) {
    local_background_candidates ranked;
    if (one_grid.width <= 0 || one_grid.height <= 0) return ranked;
    int counts[COLOR_COUNT];
    count_colors(one_grid, counts);
    bool present[COLOR_COUNT];
    for (int c = 0; c < COLOR_COUNT; ++c) present[c] = counts[c] > 0;
    background_evidence evidence;
    measure_area_and_reach(one_grid, counts, evidence);
    measure_embedding(one_grid, counts, evidence);
    close_underneath(evidence);
    finish_background_election(ranked, evidence, present);
    return ranked;
}

// A candidate must appear in EVERY grid on the side.
inline void colors_in_every_grid(const vector<const grid*>& grids, bool in_every[COLOR_COUNT]) {
    for (int i = 0; i < COLOR_COUNT; ++i) in_every[i] = true;
    for (const grid* grid_ptr : grids) {
        bool seen[COLOR_COUNT];
        present_colors(*grid_ptr, seen);
        for (int i = 0; i < COLOR_COUNT; ++i) in_every[i] = in_every[i] && seen[i];
    }
}

// Elect the background of ONE side (a set of same-role grids) by ROLE alone.
// Candidates: any color present in EVERY grid on the side, plus -1 ("no
// background"). persists[c] = does color c survive into the outputs (nullptr =
// assume all persist, e.g. when electing outputs); it only breaks ties.
inline vector<background_election::hypothesis> elect_side_background(
        const vector<const grid*>& grids, const bool* persists = nullptr) {
    vector<background_election::hypothesis> ranked;
    if (grids.empty()) return ranked;

    bool in_every[COLOR_COUNT];
    colors_in_every_grid(grids, in_every);

    background_evidence evidence;
    for (const grid* grid_ptr : grids) {
        const grid& one_grid = *grid_ptr;
        if (one_grid.empty()) continue;
        int cells_of_color[COLOR_COUNT];
        count_colors(one_grid, cells_of_color);
        measure_area_and_reach(one_grid, cells_of_color, evidence);
        measure_embedding(one_grid, cells_of_color, evidence);
    }
    close_underneath(evidence);

    finish_background_election(ranked, evidence, in_every, persists);
    return ranked;
}

// A per-pair background must be CONFIDENT: on a single grid the standout share
// saturates, so area is the only honest signal here. Confident = dominates
// absolutely, OR beats the runner-up DECISIVELY, OR holds the figures while
// having a BODY rather than being a separator skeleton. This is what keeps a
// BACKGROUNDLESS puzzle from reading as per-pair. -1 when none is confident.
template <class Candidates>
ARC_HOST_DEVICE inline int8_t confident_background(const Candidates& local) {
    int best_dominance = 0, second_dominance = 0;
    for (const auto& candidate : local) {
        if (candidate.color < 0) continue;
        if (candidate.dominance > best_dominance) {
            second_dominance = best_dominance;
            best_dominance = candidate.dominance;
        } else if (candidate.dominance > second_dominance) {
            second_dominance = candidate.dominance;
        }
    }
    for (const auto& candidate : local) {
        if (candidate.color < 0 || candidate.dominance != best_dominance) continue;
        const bool dominates = best_dominance >= BG_HALF;
        const bool beats_runner_up = best_dominance >= BG_DECISIVE * second_dominance;
        const bool has_body = candidate.gaps * BG_NEAR_TIE_DEN < 1000 * BG_NEAR_TIE_NUM;
        const bool holds_figures = candidate.embeds >= BG_HALF && has_body;
        if (dominates || beats_runner_up || holds_figures) return candidate.color;
    }
    return -1;
}

// One side of a plan: a fixed color (including no ground), or election on the
// input in hand, which a program spells as SelectElectedBackground.
struct background_source {
    int8_t color = -1;
    bool per_grid = false;
    background_source(int value = -1) : color((int8_t)value) {}
    static background_source elected() { background_source s; s.per_grid = true; return s; }
    int8_t on(const grid& input) const {
        return per_grid ? confident_background(elect_grid_background(input)) : color;
    }
    bool removes_color() const { return per_grid || is_color(color); }
    bool operator==(const background_source&) const = default;
};

// THE VETO (see input_vetoed): the global input winner is DECISIVELY beaten in
// one pair when that pair's own ground has at least twice its area there --
// then the global winner is a figure there, not background. Unless it still
// HOLDS THE FIGURES there.
template <class Candidates>
inline bool vetoes(const Candidates& local,
                   int8_t global_background, int8_t pair_background) {
    if (global_background < 0 || pair_background < 0 || pair_background == global_background)
        return false;
    int global_dominance = 0, global_embeds = 0, local_dominance = 0;
    for (const auto& candidate : local) {
        if (candidate.color == global_background) {
            global_dominance = candidate.dominance;
            global_embeds = candidate.embeds;
        }
        if (candidate.color == pair_background) local_dominance = candidate.dominance;
    }
    return local_dominance >= BG_DECISIVE * global_dominance && global_embeds < BG_HALF;
}

// Elect the background in every DIRECTION: across inputs, across outputs, across
// all grids pooled, and per pair. Same election, different grid sets — they
// legitimately disagree, and the disagreement IS information.
// Priority among directions is decided by consumers, later.
inline background_election elect_background(const vector<example>& train) {
    background_election out;
    // Persistence: a color surviving into some output — the background stays,
    // content is consumed. Feeds the input election's tiebreak.
    bool persists[COLOR_COUNT] = {};
    for (const auto& ex : train)
        walk_colored_cells(ex.output, TRANSPARENT,
                           [&](int, int, int color) { persists[color] = true; });

    vector<const grid*> inputs, outputs, both;
    for (const auto& ex : train) {
        inputs.push_back(&ex.input);
        outputs.push_back(&ex.output);
        both.push_back(&ex.input);
        both.push_back(&ex.output);
    }
    out.across_inputs         = elect_side_background(inputs,  persists);
    out.across_outputs        = elect_side_background(outputs);
    out.across_inputs_outputs = elect_side_background(both,    persists);
    out.overlays              = pairs_overlay(train);

    // Per pair: elect over that pair's INPUT alone. NOT input+output pooled —
    // the transformation can flood the output with a new color, so pooling ties
    // and abstains. Background is given by the input and constant within the
    // pair, so the input is the honest source.
    out.across_input_output.clear();
    for (const auto& ex : train) {
        const auto local = elect_grid_background(ex.input);
        const int8_t pair_background = confident_background(local);
        out.across_input_output.push_back(pair_background);
        if (vetoes(local, background_election::top(out.across_inputs), pair_background))
            out.input_vetoed = true;
    }

    // A single pair's area flip cannot stand against CROSS-SIDE agreement: when
    // the output side independently elects the very color the veto targeted, the
    // flip was a color, not the background.
    if (out.input_vetoed &&
        background_election::top(out.across_inputs) ==
        background_election::top(out.across_outputs))
        out.input_vetoed = false;
    return out;
}

// WHAT A PROGRAM DOES WITH THE GROUND, decided once for the whole reading:
// keep it, a color like any other, or delete it before anything else and fill
// it after everything else -- one color for every grid, or each input's own.
// Nothing between those two ends knows there is a ground.
struct ground_plan {
    background_source deleted;  // what every sentence's input loses first
    background_source filled;   // what the last sentence lays under the rest
    // WHETHER THE INPUTS LOSE WHAT THE FILL LAYS UNDER, first, as well. Where
    // every input lies over its output, that color is ground on the input as
    // much as on the output: kept there, its cells that stay where they are
    // read as pieces removed (c35c1b4c's black). Where the output is a canvas
    // of its own, cells of that color can be figures the rule reads (1b2d62fb's
    // 0s, read through its panes).
    bool fill_is_ground_too = false;
    // What the fill takes off the inputs as well: none where they keep it.
    background_source filled_on_input() const { return fill_is_ground_too ? filled : background_source{}; }
    // Whether the inputs lose nothing.
    bool keeps() const { return !deleted.removes_color() && !filled_on_input().removes_color(); }
    bool operator==(const ground_plan&) const = default;
};

// The plans the election allows, in the order a tie goes. The two sides are
// separate verdicts and how they relate is part of the rule (Background
// Per-Side Split.md), so a plan pairs them: the input's ground deleted or kept
// as a color, the output's filled or not -- kept, recolored, deleted,
// introduced. One color where the inputs agree on it; keeping every color, the
// ground a subject like any other (0520fde7, 5614dbcf's vote); an output
// ground the inputs lack; and each input's own wherever a grid's own ground
// differs from the puzzle's, a grid with none deleting nothing (per GRID).
inline vector<ground_plan> ground_plans(const background_election& election) {
    const int8_t in = election.input_elected(), out = election.output_elected();
    vector<ground_plan> plans;
    auto offer = [&](background_source deleted, background_source filled) {
        const ground_plan plan{ deleted, filled, election.overlays };
        if (find(plans.begin(), plans.end(), plan) == plans.end()) plans.push_back(plan);
    };
    // The input side leads: where the outputs abstain, the ground the inputs
    // name comes back (kept, as most puzzles keep it). Deleting it and filling
    // nothing is never asked: filling it back does as well wherever no cell
    // shows it, and better wherever one does (0 of 87 puzzles chose it).
    if (is_color(in) && is_color(out)) offer(background_source(in), background_source(out));
    if (is_color(in)) offer(background_source(in), background_source(in));
    offer({}, {});
    if (!is_color(in)) offer({}, background_source(out));
    const bool own = election.is_per_pair() ||
        any_of(election.across_input_output.begin(), election.across_input_output.end(),
               [&](int8_t ground) { return ground != in; });
    if (own) {
        offer(background_source::elected(), background_source::elected());
        offer(background_source::elected(), background_source(out));
    }
    return plans;
}

// The puzzle a plan leaves to read: each input without the ground it deletes,
// and without what the fill lays under where that is ground on it too; each
// output without what its fill lays under. Those cells are empty, and an empty
// cell is nothing to every reader. Both colors come from the input as given,
// since that is all a program has.
inline vector<example> figures_only(const vector<example>& train, const ground_plan& plan) {
    vector<example> figures = train;
    auto empty = [](grid& one, int8_t color) {
        if (!is_color(color)) return;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.data[row][col] == color) one.data[row][col] = TRANSPARENT;
    };
    for (example& pair : figures) {
        const int8_t deleted = plan.deleted.on(pair.input), filled = plan.filled.on(pair.input);
        empty(pair.input, deleted);
        empty(pair.input, plan.filled_on_input().on(pair.input));
        empty(pair.output, filled);
    }
    return figures;
}

}  // namespace arc
