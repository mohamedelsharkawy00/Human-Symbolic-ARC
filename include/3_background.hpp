// This file tackles the concept of a background in ARC,
// which is frequent but not in all puzzles,
// but in the puzzles where it is found,
// it is important to be distinguished from the rest of the puzzle.
// Background is elected per grid role: inputs and outputs are judged separately.

#pragma once
#include "1_readings.hpp"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// Several colors may look like the background, so we keep a list of candidates,
// with the best candidate first. Inputs and outputs have separate lists:
// a puzzle may change the background color between input and output.
// Keeping the other candidates lets us try another background later.

// These values control how we compare candidates. They were chosen to match
// the examples in tests/background_truth.txt. Reuse them when adding rules.
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
struct background_hypotheses {
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
        // color can vanish from the output. The canvas it sat on usually stays.
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
    // Background is a per-DIRECTION fact, like every other fact in this file:
    // the same election over different grid sets gives legitimately different
    // canvases, and the disagreement IS information. Best canvas first in each.
    vector<hypothesis> across_inputs;          // input grids only
    vector<hypothesis> across_outputs;         // output grids only
    vector<hypothesis> across_inputs_outputs;  // all grids pooled
    // Per-pair (across_input_output): one elected color per train pair
    vector<int8_t> per_pair;
    static int8_t top(const vector<hypothesis>& ranked) {
        return ranked.empty() ? (int8_t)-1 : ranked[0].color;
    }
    // THE SPLIT: input background and output background are different facts —
    // their relationship is part of the transformation rule (see
    // notes/4 - To Do/Background Per-Side Split.md). There is deliberately NO
    // one-answer merge; ask the side you mean.

    // INPUT-side verdict: the canvas the input figures sit on — what
    // segmentation excludes. Walks input-derived evidence STRONGEST -> WEAKEST
    // and takes the first real canvas; -1 = backgroundless or per-pair.
    int8_t input_elected() const {
        // A ground showing through beats the area winner by definition, since
        // the pattern covering it is what wins the area.
        for (const auto& candidate : across_inputs)
            if (candidate.underneath) return candidate.color;
        // The background is given in the input, undistorted by the
        // transformation — unless a confident pair contradicts it.
        if (int8_t color = top(across_inputs); color >= 0 && !input_vetoed) return color;
        // Agreeing pairs name a global color; differing real pairs mean
        // MODE=PER-PAIR (-1 here, per_pair holds the colors).
        if (int8_t color = per_pair_majority(); color >= 0) return color;
        if (is_per_pair_raw()) return -1;
        // Pooled inputs+outputs, weaker but still needs input agreement
        if (int8_t color = top(across_inputs_outputs); color >= 0) return color;
        return -1;
    }

    // OUTPUT-side verdict: the canvas the output figures sit on. A canvas that
    // PERSISTS wins before any confidence gate; without input support the
    // winner must be dominance-CONFIDENT. NOT the embeds gate — embeds is a
    // worst-grid number and one grid zeroes it.
    int8_t output_elected() const {
        int8_t input_canvas = input_elected();
        if (input_canvas >= 0)
            for (const auto& candidate : across_outputs)
                if (candidate.color == input_canvas) return input_canvas;
        if (across_outputs.empty() || across_outputs[0].color < 0) return -1;
        return across_outputs[0].dominance >= BG_HALF
             ? across_outputs[0].color : (int8_t)-1;
    }

    // A color DECISIVELY beaten inside some pair cannot be the global canvas:
    // there it is a figure sitting ON another ground. Set by elect_background,
    // where each pair's dominances are on hand.
    bool input_vetoed = false;

    // A strict MAJORITY of confident pair grounds names the canvas — learn from
    // the clear pairs, extend to the flipped ones. Unanimity is the special
    // case; truly distinct pairs have no majority and stay per-pair.
    int8_t per_pair_majority() const {
        int wins[COLOR_COUNT] = {};
        for (int8_t color : per_pair) if (color >= 0) ++wins[color];
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (wins[color] * 2 > (int)per_pair.size()) return (int8_t)color;
        return -1;
    }

    // Per-pair test that does NOT consult input_elected() (avoids recursion
    // with the global tier). Same logic as is_per_pair minus that guard.
    bool is_per_pair_raw() const {
        if (per_pair.size() < 2) return false;
        bool all_real = true, any_differ = false;
        for (int8_t color : per_pair) all_real = all_real && (color >= 0);
        for (size_t i = 1; i < per_pair.size(); ++i)
            any_differ = any_differ || (per_pair[i] != per_pair[0]);
        return all_real && any_differ;
    }

    // PER-PAIR background mode: the global election abstains, yet EVERY pair
    // has its own real background that DIFFERS across pairs. Constant WITHIN a
    // pair, varying ACROSS pairs. per_pair holds the colors; this just flags
    // that the mode applies. Detect-only for now — nothing segments per-pair.
    bool is_per_pair() const {
        // Matches input_elected()'s order, so the output-influenced directions
        // cannot suppress a strong per-pair input signal.
        if (int8_t color = top(across_inputs); color >= 0 && !input_vetoed) return false;
        if (per_pair_majority() >= 0) return false;  // majority per_pair = global
        return is_per_pair_raw();
    }
};

// Per-color background-ness prior, most-background first. The LAST tiebreak
// only, when area/embeds/story can't separate two candidates — a prior, not a
// verdict, and structure always overrides it.
inline int background_rank(int color) {
    static constexpr int8_t order[COLOR_COUNT] = { 0, 9, 5, 6, 1, 2, 7, 3, 8, 4 };
    for (int rank = 0; rank < COLOR_COUNT; ++rank) if (order[rank] == color) return rank;
    return COLOR_COUNT;
}

// What the side's grids say about each color. Every share is folded WORST-GRID:
// a canvas has to hold up everywhere, so one bad grid is the honest number.
struct canvas_evidence {
    // Worst grid, out of 1000
    int embeds[COLOR_COUNT], dominance[COLOR_COUNT], span[COLOR_COUNT], gaps[COLOR_COUNT];
    // In EVERY grid: the quietest color present, showing only as enclosed windows
    bool only_windows[COLOR_COUNT];
    // Grids where this color was the standout canvas, and how many grids voted
    int standout_wins[COLOR_COUNT] = {};
    int grid_count = 0;
    // Total cells per color over the whole side
    long long global_cells[COLOR_COUNT] = {};

    canvas_evidence() {
        for (int i = 0; i < COLOR_COUNT; ++i) {
            embeds[i] = dominance[i] = span[i] = gaps[i] = 1000;
            only_windows[i] = true;
        }
    }
};

// A candidate must appear in EVERY grid on the side.
inline void colors_in_every_grid(const vector<const grid*>& grids, bool in_every[COLOR_COUNT]) {
    for (int i = 0; i < COLOR_COUNT; ++i) in_every[i] = true;
    for (const grid* grid_ptr : grids) {
        bool seen[COLOR_COUNT] = {};
        walk_colored_cells(*grid_ptr, TRANSPARENT,
                           [&](int, int, int color) { seen[color] = true; });
        for (int i = 0; i < COLOR_COUNT; ++i) in_every[i] = in_every[i] && seen[i];
    }
}

// STAGE 1 — how much of this grid each color holds, how far it reaches, and
// which color reads as the canvas here.
inline void measure_area_and_reach(const grid& one_grid, const int cells_of_color[COLOR_COUNT],
                                   canvas_evidence& evidence) {
    const int area = one_grid.width * one_grid.height;
    for (int i = 0; i < COLOR_COUNT; ++i) evidence.global_cells[i] += cells_of_color[i];
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.dominance[i] = min(evidence.dominance[i], cells_of_color[i] * 1000 / area);

    // Punched-through probes (measured only): span = how much of the grid the
    // color's box covers; gaps = share of its cells touching another color.
    // A ground showing through is everywhere but always thin.
    // ADJACENCY, SO FOUR: a fraying measure asks what BORDERS this cell, and a
    // corner touch is not a border. Not the 8 that grouping uses — see the note
    // beside neighbour_row_4 in 1_readings.hpp before making these agree.
    int touching[COLOR_COUNT] = {};
    color_box box[COLOR_COUNT];
    color_boxes(one_grid, box);
    walk_colored_cells(one_grid, TRANSPARENT, [&](int row, int col, int color) {
        for (int dir = 0; dir < 4; ++dir) {
            int next_row = row + neighbour_row_4(dir);
            int next_col = col + neighbour_col_4(dir);
            if (next_row < 0 || next_col < 0 ||
                next_row >= one_grid.height || next_col >= one_grid.width) continue;
            if (one_grid.data[next_row][next_col] != color) { touching[color]++; return; }
        }
    });
    for (int i = 0; i < COLOR_COUNT; ++i) {
        if (!cells_of_color[i]) { evidence.span[i] = 0; evidence.gaps[i] = 0; continue; }
        int box_area = box[i].area();
        evidence.span[i] = min(evidence.span[i], box_area * 1000 / area);
        evidence.gaps[i] = min(evidence.gaps[i], touching[i] * 1000 / cells_of_color[i]);
    }

    // Standout in this grid: the canvas among the SUBSTANTIAL colors — those
    // co-dominant with the leader — where the background-ness prior picks the
    // quieter one. Not raw-most-cells: a separator grid inflates a foreground
    // color's area past the true canvas. Then the corpus VOTES, not unanimity.
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

// The pieces of one grid — same-color cells that touch, corners counting —
// labelled once (the same walk gpu_flood_fill does). `cells` holds every
// piece's cells, piece by piece, with first_cell naming where each starts.
struct blob_map {
    int16_t id[MAX_GRID_DIM][MAX_GRID_DIM];
    vector<int> color, size, first_cell;
    vector<char> on_border;
    vector<pair<int8_t, int8_t>> cells;
    int count() const { return (int)color.size(); }
};

inline blob_map label_blobs(const grid& one_grid) {
    blob_map blobs;
    for (int row = 0; row < one_grid.height; ++row)
        for (int col = 0; col < one_grid.width; ++col) blobs.id[row][col] = -1;
    int16_t queue_row[MAX_GRID_DIM * MAX_GRID_DIM], queue_col[MAX_GRID_DIM * MAX_GRID_DIM];
    walk_colored_cells(one_grid, TRANSPARENT, [&](int start_row, int start_col, int color) {
        if (blobs.id[start_row][start_col] >= 0) return;
        int16_t id = (int16_t)blobs.color.size();
        queue_row[0] = (int16_t)start_row; queue_col[0] = (int16_t)start_col;
        blobs.id[start_row][start_col] = id;
        bool on_border = false;
        int first_cell = (int)blobs.cells.size();
        const int size = walk_out(one_grid.height, one_grid.width, queue_row, queue_col, 1,
            [&](int next_row, int next_col) {
                if (blobs.id[next_row][next_col] >= 0 ||
                    one_grid.data[next_row][next_col] != color) return false;
                blobs.id[next_row][next_col] = id; return true;
            },
            [&](int row, int col) {
                blobs.cells.push_back({ (int8_t)row, (int8_t)col });
                if (row == 0 || row == one_grid.height - 1 ||
                    col == 0 || col == one_grid.width - 1)
                    on_border = true;
            });
        blobs.color.push_back(color); blobs.size.push_back(size);
        blobs.first_cell.push_back(first_cell); blobs.on_border.push_back(on_border);
    });
    return blobs;
}

// STAGE 2 — each figure votes for the color it sits in, weighted by its own
// cells. FIGURE = a blob whose outer contour is majority-owned by ONE other
// blob; everything else is BACKDROP. The same pass feeds the UNDERNEATH check:
// a color is believable as ground showing through only if it is the quietest
// present and appears solely as enclosed windows.
inline void measure_embedding(const grid& one_grid, const int cells_of_color[COLOR_COUNT],
                              canvas_evidence& evidence) {
    const blob_map blobs = label_blobs(one_grid);
    int figure_area = 0, embedded_area[COLOR_COUNT] = {};
    int blobs_of_color[COLOR_COUNT] = {}, backdrop_blobs_of_color[COLOR_COUNT] = {};

    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_DIM * MAX_GRID_DIM], queue_col[MAX_GRID_DIM * MAX_GRID_DIM];
    // One vote box per blob, indexed by the blob's own id — no lookup table to
    // run out of, so no contour can be too crowded to count. `voted` names the
    // boxes this blob touched, so clearing costs what was used, not all of them.
    vector<int> votes(blobs.count(), 0);
    vector<int> voted;
    for (int blob = 0; blob < blobs.count(); ++blob) {
        blobs_of_color[blobs.color[blob]]++;
        if (blobs.on_border[blob]) { backdrop_blobs_of_color[blobs.color[blob]]++; continue; }
        // Flood "outside" = everything reachable from the border without
        // entering this blob
        for (int row = 0; row < one_grid.height; ++row)
            for (int col = 0; col < one_grid.width; ++col) outside[row][col] = false;
        int seeded = 0;
        for (int row = 0; row < one_grid.height; ++row)
            for (int col = 0; col < one_grid.width; ++col) {
                if (row != 0 && row != one_grid.height - 1 &&
                    col != 0 && col != one_grid.width - 1) continue;
                if (blobs.id[row][col] == blob || outside[row][col]) continue;
                outside[row][col] = true;
                queue_row[seeded] = (int16_t)row; queue_col[seeded] = (int16_t)col; seeded++;
            }
        walk_out(one_grid.height, one_grid.width, queue_row, queue_col, seeded,
            [&](int next_row, int next_col) {
                if (outside[next_row][next_col] ||
                    blobs.id[next_row][next_col] == blob) return false;
                outside[next_row][next_col] = true; return true;
            },
            [](int, int) {});
        // Outer contour = the outside cells directly beside this blob (edges
        // only, not corners). Every such cell
        // votes for the blob it belongs to; a majority makes this a figure
        // sitting in that blob.
        int contour_votes_total = 0;
        voted.clear();
        for (int nth = 0; nth < blobs.size[blob]; ++nth) {
            int row = blobs.cells[blobs.first_cell[blob] + nth].first;
            int col = blobs.cells[blobs.first_cell[blob] + nth].second;
            for (int dir = 0; dir < 4; ++dir) {
                int next_row = row + neighbour_row_4(dir);
                int next_col = col + neighbour_col_4(dir);
                if (next_row < 0 || next_row >= one_grid.height ||
                    next_col < 0 || next_col >= one_grid.width) continue;
                if (!outside[next_row][next_col]) continue;
                int owner_blob = blobs.id[next_row][next_col];
                if (owner_blob < 0) continue;
                if (votes[owner_blob] == 0) voted.push_back(owner_blob);
                votes[owner_blob]++;
                contour_votes_total++;
            }
        }
        int best_owner = -1, best_votes = 0;
        for (int owner : voted)
            if (votes[owner] > best_votes) { best_votes = votes[owner]; best_owner = owner; }
        for (int owner : voted) votes[owner] = 0;
        bool is_figure = best_owner >= 0 && best_votes * 2 > contour_votes_total;
        if (!is_figure) { backdrop_blobs_of_color[blobs.color[blob]]++; continue; }
        figure_area += blobs.size[blob];
        embedded_area[blobs.color[best_owner]] += blobs.size[blob];
    }
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.embeds[i] = min(evidence.embeds[i],
                                 figure_area ? embedded_area[i] * 1000 / figure_area : 0);

    // UNDERNEATH for this grid: quietest color present, and only enclosed
    // windows (>= 2 blobs, none touching the border). One enclosed blob is an
    // object; many windows are the ground showing through.
    int quietest = -1;
    for (int i = 0; i < COLOR_COUNT; ++i)
        if (cells_of_color[i] > 0 &&
            (quietest < 0 || background_rank(i) < background_rank(quietest)))
            quietest = i;
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.only_windows[i] = evidence.only_windows[i] && i == quietest &&
                                   blobs_of_color[i] >= 2 && backdrop_blobs_of_color[i] == 0;
}

// STAGE 3 — windows are believable only when cut into a PATTERNED coverer. The
// coverer is the global area leader; if it is solid, no color may claim
// underneath.
inline void close_underneath(canvas_evidence& evidence) {
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
// KNOWN GAP: a canvas that is a MINORITY in MOST grids is not findable by area
// at all; that needs the object story.
inline void score_candidates(vector<background_hypotheses::hypothesis>& ranked,
                             const canvas_evidence& evidence, const bool in_every[COLOR_COUNT]) {
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

// STAGE 5 — best canvas first. The top score elects IF it clears BG_FLOOR, else
// the direction abstains (-1 sorts first) — the honest "ambiguous" answer.
inline void rank_candidates(vector<background_hypotheses::hypothesis>& ranked) {
    stable_sort(ranked.begin(), ranked.end(), [&](const auto& left, const auto& right) {
        const bool left_electable  = (left.color >= 0 && left.score >= BG_FLOOR);
        const bool right_electable = (right.color >= 0 && right.score >= BG_FLOOR);
        if (left_electable != right_electable) return left_electable;
        if (!left_electable) {                   // none electable: -1 first
            if ((left.color < 0) != (right.color < 0)) return left.color < 0;
            return background_rank(left.color) < background_rank(right.color);
        }
        return left.score > right.score;         // among electables, top score
    });
}

// Elect the canvas of ONE side (a set of same-role grids) by ROLE alone.
// Candidates: any color present in EVERY grid on the side, plus -1 ("no
// background"). persists[c] = does color c survive into the outputs (nullptr =
// assume all persist, e.g. when electing outputs); it only breaks ties.
inline vector<background_hypotheses::hypothesis> elect_side_background(
        const vector<const grid*>& grids, const bool* persists = nullptr) {
    vector<background_hypotheses::hypothesis> ranked;
    if (grids.empty()) return ranked;

    bool in_every[COLOR_COUNT];
    colors_in_every_grid(grids, in_every);

    canvas_evidence evidence;
    for (const grid* grid_ptr : grids) {
        const grid& one_grid = *grid_ptr;
        if (one_grid.width <= 0 || one_grid.height <= 0) continue;
        int cells_of_color[COLOR_COUNT];
        count_colors(one_grid, cells_of_color);
        measure_area_and_reach(one_grid, cells_of_color, evidence);
        measure_embedding(one_grid, cells_of_color, evidence);
    }
    close_underneath(evidence);

    ranked.push_back({ (int8_t)-1, 0, 0, false, 0, 0, false, 0 });  // "none" always a candidate
    for (int candidate = 0; candidate < COLOR_COUNT; ++candidate) {
        if (!in_every[candidate]) continue;
        bool survives = persists ? persists[candidate] : true;
        ranked.push_back({ (int8_t)candidate, evidence.embeds[candidate],
                           evidence.dominance[candidate], survives,
                           evidence.span[candidate], evidence.gaps[candidate],
                           evidence.only_windows[candidate], 0 });
    }

    score_candidates(ranked, evidence, in_every);
    rank_candidates(ranked);
    return ranked;
}

// Elect the canvas in every DIRECTION: across inputs, across outputs, across
// all grids pooled, and per pair. Same election, different grid sets — they
// legitimately disagree, and the disagreement IS information.
// Priority among directions is decided by consumers, later.
inline background_hypotheses elect_background(const vector<example>& train) {
    background_hypotheses out;
    // Persistence: a color surviving into some output — the canvas stays,
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

    // Per pair: elect over that pair's INPUT alone. NOT input+output pooled —
    // the transformation can flood the output with a new color, so pooling ties
    // and abstains. Background is given by the input and constant within the
    // pair, so the input is the honest source.
    out.per_pair.clear();
    for (const auto& ex : train) {
        const vector<const grid*> just_this_input = { &ex.input };
        const vector<background_hypotheses::hypothesis> local =
            elect_side_background(just_this_input);
        // A per-pair background must be CONFIDENT: on a single grid the standout
        // share saturates, so area is the only honest signal here. Confident =
        // dominates absolutely, OR beats the runner-up DECISIVELY, OR holds the
        // figures while having a BODY rather than being a separator skeleton.
        // This is what keeps a BACKGROUNDLESS puzzle from reading as per-pair.
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
        int8_t pair_background = -1;
        for (const auto& candidate : local) {
            if (candidate.color < 0 || candidate.dominance != best_dominance) continue;
            const bool dominates = best_dominance >= BG_HALF;
            const bool beats_runner_up = best_dominance >= BG_DECISIVE * second_dominance;
            const bool has_body =
                candidate.gaps * BG_NEAR_TIE_DEN < 1000 * BG_NEAR_TIE_NUM;
            const bool holds_figures = candidate.embeds >= BG_HALF && has_body;
            if (dominates || beats_runner_up || holds_figures) {
                pair_background = candidate.color;
                break;
            }
        }
        out.per_pair.push_back(pair_background);
        // Veto check (see input_vetoed): the global input winner is DECISIVELY
        // beaten in this pair when the local ground has at least twice its local
        // area — then the global winner is a figure here, not canvas. Unless it
        // still HOLDS THE FIGURES locally.
        const int8_t global_background = background_hypotheses::top(out.across_inputs);
        if (global_background >= 0 && pair_background >= 0 &&
            pair_background != global_background) {
            int global_dominance = 0, global_embeds = 0, local_dominance = 0;
            for (const auto& candidate : local) {
                if (candidate.color == global_background) {
                    global_dominance = candidate.dominance;
                    global_embeds = candidate.embeds;
                }
                if (candidate.color == pair_background) local_dominance = candidate.dominance;
            }
            if (local_dominance >= BG_DECISIVE * global_dominance &&
                global_embeds < BG_HALF)
                out.input_vetoed = true;
        }
    }

    // A single pair's area flip cannot stand against CROSS-SIDE agreement: when
    // the output side independently elects the very color the veto targeted, the
    // flip was a color, not the background.
    if (out.input_vetoed &&
        background_hypotheses::top(out.across_inputs) ==
        background_hypotheses::top(out.across_outputs))
        out.input_vetoed = false;
    return out;
}

}  // namespace arc
