// This file tackles the concept of a background in ARC,
// which is frequent but not in all puzzles,
// but in the puzzles where it is found,
// it is important to be distinguished from the rest of the puzzle.
// Background is elected per grid role: inputs and outputs are judged separately.
#pragma once
#include "1_readings.hpp"
#include "2_size.hpp"
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

// The score must reach 550 for a candidate to qualify as a background: the bar
// black cleared when the floor was 700 and black's place in a color order
// added 150, now the same for every color.
inline constexpr int BG_FLOOR = 550;

// Each kind of evidence contributes to the score. A larger weight gives
// that evidence more influence. None of it is the color's name: a puzzle
// repainted, every color for another, elects the same ground repainted.
inline constexpr double BG_W_AREA = 1.0;      // How much area the color covers.
inline constexpr double BG_W_STANDOUT = 0.3;  // How often it wins in individual grids.
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
        // Worth a small bonus (BG_PERSIST_BONUS, 50 against a floor of 550), so
        // it only separates two candidates that area already left nearly tied.
        bool appears_in_any_output = true;

        // Per mille of all cells in the grid falling inside this color's
        // bounding box, for the worst example grid; everywhere = ~1000
        int span = 0;

        // Per mille of this color's own cells that touch a different color,
        // for the worst example grid; a thin web = ~1000, a solid mass = low
        int gaps = 0;

        // Weighted confidence that this color is the background. The election
        // takes the top score if it clears BG_FLOOR and no other ties it, else
        // abstains.
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
    // The color every output's crop cut against (cut_against), -1 for none.
    int8_t cut = -1;
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
        // What every output was cropped out of is the ground, before any score:
        // the crop took it off.
        if (is_color(cut)) return cut;
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
        if (is_color(cut)) return false;
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
    // Grids where this color was the standout background, and how many grids voted
    int standout_wins[COLOR_COUNT] = {};
    int grid_count = 0;
    // Total cells per color over the whole side
    long long global_cells[COLOR_COUNT] = {};

    background_evidence() {
        for (int i = 0; i < COLOR_COUNT; ++i) embeds[i] = dominance[i] = span[i] = gaps[i] = 1000;
    }
};

// The box each color's cells fill; a color no cell wears keeps an empty box.
inline void color_boxes(const grid& one_grid, cell_box boxes[COLOR_COUNT]) {
    for (int i = 0; i < COLOR_COUNT; ++i) boxes[i] = {};
    one_grid.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
        boxes[color].include_cell(row, col);
    });
}

// STAGE 1 — how much of this grid each color holds, how far it reaches, and
// which color reads as the background here.
inline void measure_area_and_reach(const grid& one_grid, const int cells_of_color[COLOR_COUNT],
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
    // beside every_nth_direction in 1_readings.hpp before making these agree.
    int touching[COLOR_COUNT] = {};
    cell_box box[COLOR_COUNT];
    color_boxes(one_grid, box);
    one_grid.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
        for (int neighbor_direction = 0; neighbor_direction < direction::count; neighbor_direction += every_nth_direction<4>) {
            int next_row = row + step_of(neighbor_direction).down;
            int next_col = col + step_of(neighbor_direction).across;
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

    // Standout in this grid: the one SUBSTANTIAL color, where no other is
    // co-dominant with it. Of two near equals the grid cannot say which is the
    // ground, and votes for neither. Then the corpus VOTES, not unanimity.
    int most_cells = 0;
    for (int i = 0; i < COLOR_COUNT; ++i) most_cells = max(most_cells, cells_of_color[i]);
    int winner = -1, substantial = 0;
    for (int i = 0; i < COLOR_COUNT; ++i) {
        if (cells_of_color[i] * BG_NEAR_TIE_DEN < most_cells * BG_NEAR_TIE_NUM) continue;
        winner = i;
        ++substantial;
    }
    if (substantial == 1 && most_cells > 0) evidence.standout_wins[winner]++;
    evidence.grid_count++;
}

// STAGE 2 — each figure votes for the color it sits in, weighted by its own
// cells. FIGURE = an object whose outer contour is majority-owned by ONE other
// object; everything else is BACKDROP.
inline void measure_embedding(const grid& one_grid, background_evidence& evidence) {
    // The objects: same-color cells that touch, corners counting, each the
    // color of its first cell.
    const grid_parts objects(one_grid, grouping::touching_same_color);
    auto color_of = [&](int object) {
        return one_grid.data[objects.rows[objects.first[object]]][objects.cols[objects.first[object]]];
    };
    int figure_area = 0, embedded_area[COLOR_COUNT] = {};

    // One vote box per object, indexed by the object's own id — no lookup table to
    // run out of, so no contour can be too crowded to count. `voted` names the
    // boxes this object touched, so clearing costs what was used, not all of them.
    int16_t votes[MAX_GRID_DIM * MAX_GRID_DIM] = {};
    int16_t voted[MAX_GRID_DIM * MAX_GRID_DIM];
    int voted_count = 0;
    for (int this_object = 0; this_object < objects.count; ++this_object) {
        if (objects.meets[this_object]) continue;
        // Outside = everything reachable from the border without entering this
        // object (grid_parts). Four-way space cannot leak through corners
        // joined by the object's eight-way connectivity.
        const grid_parts space(one_grid.width, one_grid.height,
            [&](int row, int col) { return objects.part[row][col] == this_object; });
        // Outer contour = the outside cells directly beside this object (edges
        // only, not corners). Every such cell
        // votes for the object it belongs to; a majority makes this a figure
        // sitting in that object.
        int contour_votes_total = 0;
        voted_count = 0;
        for (int at = objects.first[this_object]; at < objects.first[this_object + 1]; ++at) {
            const int row = objects.rows[at], col = objects.cols[at];
            for (int neighbor_direction = 0; neighbor_direction < direction::count; neighbor_direction += every_nth_direction<4>) {
                int next_row = row + step_of(neighbor_direction).down;
                int next_col = col + step_of(neighbor_direction).across;
                if (next_row < 0 || next_row >= one_grid.height || next_col < 0 || next_col >= one_grid.width) continue;
                if (!space.outside(next_row, next_col)) continue;
                int owner_object = objects.part[next_row][next_col];
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
        if (!is_figure) continue;
        figure_area += objects.size(this_object);
        embedded_area[color_of(best_owner)] += objects.size(this_object);
    }
    for (int i = 0; i < COLOR_COUNT; ++i)
        evidence.embeds[i] = min(evidence.embeds[i],
                                 figure_area ? embedded_area[i] * 1000 / figure_area : 0);
}

// STAGE 3 — weighted signals sum to a confidence per color. No single rule is a
// hard gate. Area leads, standout corroborates, punched-through and embeds are
// graded votes.
// KNOWN GAP: a background that is a MINORITY in MOST grids is not findable by area
// at all; that needs the object story.
inline void score_candidates(vector<background_election::hypothesis>& ranked, const background_evidence& evidence) {
    long long total_cells = 0;
    for (int i = 0; i < COLOR_COUNT; ++i) total_cells += evidence.global_cells[i];

    for (auto& candidate : ranked) {
        if (candidate.color < 0) { candidate.score = 0; continue; }
        const int color = candidate.color;
        const int global_area =
            total_cells ? (int)(evidence.global_cells[color] * 1000 / total_cells) : 0;
        const int standout_share = evidence.grid_count
            ? evidence.standout_wins[color] * 1000 / evidence.grid_count : 0;
        // Everywhere AND thin
        const int punched = min(candidate.span, candidate.gaps);

        // Summed in this order on purpose: floating-point addition does not
        // associate, so reordering these can move the total across the floor.
        const double from_area     = global_area    * BG_W_AREA;
        const double from_standout = standout_share * BG_W_STANDOUT;
        const double from_persists = candidate.appears_in_any_output ? BG_PERSIST_BONUS : 0;
        const double from_punched  = punched * BG_W_PUNCHED;
        const double from_embeds   = candidate.embeds * BG_W_EMBEDS;

        candidate.score = (int)(from_area + from_standout + from_persists + from_punched + from_embeds);
    }
}

// STAGE 4 — best background first. The top score elects IF it clears BG_FLOOR
// and no other candidate ties it, else the direction abstains (-1 sorts first)
// — the honest "ambiguous" answer: two colors the picture rates alike name no
// ground.
inline void rank_candidates(vector<background_election::hypothesis>& ranked) {
    const auto electable = [](const auto& candidate) { return candidate.color >= 0 && candidate.score >= BG_FLOOR; };
    const auto before = [&](const auto& left, const auto& right) {
        if (electable(left) != electable(right)) return electable(left);
        if (!electable(left) && (left.color < 0) != (right.color < 0)) return left.color < 0;
        return left.score > right.score;
    };
    // Stable insertion sort: at most eleven candidates.
    for (int i = 1; i < (int)ranked.size(); ++i) {
        auto candidate = ranked[i];
        int j = i;
        while (j > 0 && before(candidate, ranked[j - 1])) {
            ranked[j] = ranked[j - 1];
            --j;
        }
        ranked[j] = candidate;
    }
    if (ranked.size() >= 2 && electable(ranked[0]) && electable(ranked[1]) && ranked[0].score == ranked[1].score) {
        const auto none = ranges::find_if(ranked, [](const auto& candidate) { return candidate.color < 0; });
        rotate(ranked.begin(), none, none + 1);
    }
}

// A candidate must appear in EVERY grid on the side.
inline void colors_in_every_grid(const vector<const grid*>& grids, bool in_every[COLOR_COUNT]) {
    for (int i = 0; i < COLOR_COUNT; ++i) in_every[i] = true;
    for (const grid* grid_ptr : grids) {
        const color_tally counts(*grid_ptr);
        for (int i = 0; i < COLOR_COUNT; ++i) in_every[i] = in_every[i] && counts.has(i);
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
        const color_tally cells_of_color(one_grid);
        measure_area_and_reach(one_grid, cells_of_color.counts, evidence);
        measure_embedding(one_grid, evidence);
    }

    // "None" is always a candidate.
    ranked.push_back({ .color = -1, .appears_in_any_output = false });
    for (int candidate = 0; candidate < COLOR_COUNT; ++candidate) {
        if (!in_every[candidate]) continue;
        ranked.push_back({ .color = (int8_t)candidate,
                           .embeds = evidence.embeds[candidate],
                           .dominance = evidence.dominance[candidate],
                           .appears_in_any_output = persists ? persists[candidate] : true,
                           .span = evidence.span[candidate],
                           .gaps = evidence.gaps[candidate] });
    }
    score_candidates(ranked, evidence);
    rank_candidates(ranked);
    return ranked;
}

// The background of one grid on its own, with no training pairs: a side of one.
inline vector<background_election::hypothesis> elect_grid_background(const grid& one_grid) {
    return elect_side_background({ &one_grid });
}

// A per-pair background must be CONFIDENT: on a single grid the standout share
// saturates, so area is the only honest signal here. Confident = dominates
// absolutely, OR beats the runner-up DECISIVELY, OR holds the figures while
// having a BODY rather than being a separator skeleton. This is what keeps a
// BACKGROUNDLESS puzzle from reading as per-pair. -1 when none is confident, or
// two colors hold as much.
inline int8_t confident_background(const vector<background_election::hypothesis>& local) {
    int best_dominance = 0, second_dominance = 0;
    const background_election::hypothesis* best = nullptr;
    for (const auto& candidate : local) {
        if (candidate.color < 0) continue;
        if (candidate.dominance > best_dominance) {
            second_dominance = best_dominance;
            best_dominance = candidate.dominance;
            best = &candidate;
        } else if (candidate.dominance > second_dominance) {
            second_dominance = candidate.dominance;
        }
    }
    if (!best || second_dominance == best_dominance) return -1;
    const bool dominates = best_dominance >= BG_HALF;
    const bool beats_runner_up = best_dominance >= BG_DECISIVE * second_dominance;
    const bool has_body = best->gaps * BG_NEAR_TIE_DEN < 1000 * BG_NEAR_TIE_NUM;
    const bool holds_figures = best->embeds >= BG_HALF && has_body;
    return dominates || beats_runner_up || holds_figures ? best->color : (int8_t)-1;
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
inline bool vetoes(const vector<background_election::hypothesis>& local,
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

// THE GROUND A CROP TOOK OFF: where the output's size is what is drawn
// (BoundingBox) and every output is cut whole out of its input, found there
// once, the color most of the input's cells just outside the cut wear, the same
// in every pair -- what the crop cut against (e50d258f's black around the panel
// kept). -1 where a pair has no such cut, the cells around it tie, or the pairs
// disagree.
inline int8_t cut_against(const vector<example>& train) {
    if (train.empty() || output_dimension_rule::infer_rule(train).Chosen != output_dimension_rule::BoundingBox)
        return -1;
    int8_t agreed = -1;
    for (const example& pair : train) {
        const grid& in = pair.input;
        const grid& out = pair.output;
        int found = 0, top = 0, left = 0;
        for (int row = 0; row + out.height <= in.height; ++row)
            for (int col = 0; col + out.width <= in.width; ++col) {
                bool same = true;
                for (int down = 0; down < out.height && same; ++down)
                    for (int across = 0; across < out.width && same; ++across)
                        same = in.data[row + down][col + across] == out.data[down][across];
                if (same) {
                    ++found;
                    top = row;
                    left = col;
                }
            }
        if (found != 1) return -1;
        color_tally around;
        for (int row = top - 1; row <= top + out.height; ++row)
            for (int col = left - 1; col <= left + out.width; ++col)
                if (in.contains(row, col) &&
                    (row < top || row >= top + out.height || col < left || col >= left + out.width))
                    around.add(in.data[row][col]);
        int8_t most = -1;
        bool tied = false;
        for (int color = 0; color < COLOR_COUNT; ++color) {
            if (!around.has(color)) continue;
            if (most < 0 || around[color] > around[most]) {
                most = (int8_t)color;
                tied = false;
            } else if (around[color] == around[most]) {
                tied = true;
            }
        }
        if (most < 0 || tied || (agreed >= 0 && most != agreed)) return -1;
        agreed = most;
    }
    return agreed;
}

// Elect the background in every DIRECTION: across inputs, across outputs, across
// all grids pooled, and per pair. Same election, different grid sets — they
// legitimately disagree, and the disagreement IS information.
// Priority among directions is decided by consumers, later.
inline background_election elect_background(const vector<example>& train) {
    background_election out;
    // The outputs' cells, color by color. Persistence: a color surviving into
    // some output — the background stays, content is consumed. Feeds the input
    // election's tiebreak.
    color_tally output_area;
    for (const auto& ex : train)
        ex.output.for_each_colored_cell(TRANSPARENT, [&](int, int, int color) { output_area.add(color); });
    bool persists[COLOR_COUNT];
    for (int color = 0; color < COLOR_COUNT; ++color) persists[color] = output_area.has(color);

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
    out.cut                   = cut_against(train);

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
    // the outputs elect the very color the veto targeted, and it covers the most
    // of them with no other color near it, the flip was a color, not the
    // background (007bbfb7's outputs are mostly black). A lead near equals share
    // is a tie, and agrees on nothing (7ed72f31's 8 over 1, 288 cells to 283).
    const int8_t vetoed = background_election::top(out.across_inputs);
    if (out.input_vetoed && vetoed == background_election::top(out.across_outputs)) {
        bool covers_most = true;
        for (int color = 0; color < COLOR_COUNT; ++color)
            covers_most = covers_most && (color == vetoed ||
                output_area[color] * BG_NEAR_TIE_DEN < output_area[vetoed] * BG_NEAR_TIE_NUM);
        if (covers_most) out.input_vetoed = false;
    }
    return out;
}

// WHAT A PROGRAM DOES WITH THE GROUND, decided once for the whole reading:
// keep it as a color, or remove it after cuts that need it and fill it after
// the operators on the figures. Each pass repeats this order with the same
// chosen colors.
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
    // Where every input has a ground of its own and they differ, that is the plan.
    if (election.is_per_pair()) offer(background_source::elected(), background_source::elected());
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

// A PAIR AS A PLAN READS IT: whole, every cell in its own color, and which
// cells of each grid are figures. The rest, the ground the plan deletes or
// fills, is still there, in no figure.
struct figure_pair {
    example whole;
    cell_set input_figures, output_figures;
};

// WHETHER A CELL COMES OUT AS THE PAIR'S OUTPUT once the plan has laid its own
// cells, the ones no figure of the output holds (plan_figures): one showing a
// color, where the output shows that color; one left empty, where no figure of
// the output is, since the plan lays there what the output shows. So a cell
// painted in the color the plan lays there comes out as one left empty does
// (94133066's block, its body the color the plan fills with).
inline bool comes_out(const figure_pair& pair, int row, int col, int8_t shown) {
    return is_color(shown) ? pair.whole.output.data[row][col] == shown : !pair.output_figures.has(row, col);
}

// THE FIGURES A PLAN LEAVES TO READ: on each input every colored cell but the
// ground it deletes, and what the fill lays under where that is ground on the
// input too; on each output every colored cell but what its fill lays under.
// Both colors come from the input as given, since that is all a program has.
inline vector<figure_pair> plan_figures(const vector<example>& train, const ground_plan& plan,
                                        const vector<int8_t>& elected = {}) {
    vector<figure_pair> pairs;
    for (size_t which = 0; which < train.size(); ++which) {
        figure_pair& pair = pairs.emplace_back(figure_pair{ train[which] });
        auto color_of = [&](background_source source) {
            return source.per_grid && which < elected.size() ? elected[which] : source.on(pair.whole.input);
        };
        const int8_t deleted = color_of(plan.deleted), filled = color_of(plan.filled),
                     filled_here = color_of(plan.filled_on_input());
        pair.whole.input.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
            if (color != deleted && color != filled_here) pair.input_figures.add(row, col);
        });
        pair.whole.output.for_each_colored_cell(filled, [&](int row, int col, int) { pair.output_figures.add(row, col); });
    }
    return pairs;
}

// THE PAIRS WITH NOTHING BUT THEIR FIGURES, every other cell emptied: each
// input as the program holds it once the plan's opening sentence has run, each
// output as its sentences must draw it before the fill, so a rule replayed on
// them reads what its operator reads. Not every rule the language says can be
// read off them: a fill in a figure's own color empties that figure on the
// outputs (94133066's 1s), and the ground's own shapes are no pieces
// (868de0fa's hollows).
inline vector<example> figures_only(const vector<figure_pair>& pairs) {
    auto empty_the_rest = [](grid& one, const cell_set& figures) {
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (!figures.has(row, col)) one.data[row][col] = TRANSPARENT;
    };
    vector<example> only;
    for (const figure_pair& pair : pairs) {
        example& kept = only.emplace_back(pair.whole);
        empty_the_rest(kept.input, pair.input_figures);
        empty_the_rest(kept.output, pair.output_figures);
    }
    return only;
}

}  // namespace arc
