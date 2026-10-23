// The input read as panes, several grids laid side by side, and the rules that
// read an answer off them: the panes together, one pane, or every pane's
// drawing gathered where its pane was.
#pragma once
#include "7_facts.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>
#include <vector>

namespace arc {

// WHAT A CUT IS ALLOWED TO FOLLOW. Where the cut falls -- which axis, which
// color, how many panes -- is read off the grid; these are only the permissions,
// the way degrees_of_freedom holds them for matching.
struct split_freedom {
    // A full-span line of one color: a wall.
    bool wall = true;
    // A full-span line of empty cells, where the ground was: the same shape
    // of evidence, thinner.
    bool blank = true;
    // No line at all, k equal parts. It fits almost any grid, so on its own it
    // says nothing and something else has to license it.
    bool bare = true;
    // Cut both axes at once, leaving a lattice of panes instead of a row.
    bool lattice = true;
    // Panes that are not all one size: a picture beside a key, rather than a row
    // of things to compare.
    bool uneven = true;
};

// HOW A GRID WAS CUT INTO PANES, named by five numbers, so two pairs of
// different sizes can still be asked whether they were cut the same way.
struct pane_cut {
    // What the cut followed. The order is the precedence: plainest first.
    enum : int8_t { Wall, Blank, Bare };
    // Which way it ran. BothWays leaves a lattice rather than a row of panes.
    enum : int8_t { Across, Down, BothWays };

    int8_t cut = Wall;
    int8_t axis = Across;
    // The separator color, or -1 when the cut followed no line.
    int8_t color = -1;
    // How many panes the cut leaves. One of these is 1 unless it went both ways.
    int8_t down = 1, across = 1;

    bool operator==(const pane_cut& other) const = default;
};

// The input is not one grid but several laid side by side -- panes -- with a
// wall, a blank line, or nothing between them. input_blocks_rule below cuts the
// OUTPUT into input-sized blocks; this cuts the INPUT into panes.
//
// Nothing here is scored. A cut holds only when the SAME one holds on every
// training pair and no rival survives beside it: a split the pairs agree on
// without argument is a fact, and anything less is not a split at all. Ties
// break wall > blank > bare, fewest assumptions first.
struct pane_split_facts : pane_cut {
    // Where the panes sit in one training input, in reading order.
    struct pane {
        int8_t top = 0, left = 0, width = 0, height = 0;

        // Read local coordinates, optionally through a rotation or reflection.
        int color_at(const grid& source, int row, int col, int pose = 0) const {
            const auto [source_row, source_col] = pose_source(pose, width, height, row, col);
            return source.data[top + source_row][left + source_col];
        }
        // Its box on the grid.
        cell_box box() const { return { top, left, top + height - 1, left + width - 1 }; }
        // Whether the pane reads through this pose at its own size.
        bool keeps_size(int pose) const {
            return posed_size(pose, width, height) == pair<int, int>{ width, height };
        }
        // The pane cut out of its grid, as a grid of its own.
        grid drawn(const grid& source) const {
            grid out(width, height);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) out.data[row][col] = (int8_t)color_at(source, row, col);
            return out;
        }
    };
    vector<vector<pane>> pairs;

    bool held = false;
    bool holds() const { return held; }

    // The size every one of `panes` shares, none where they differ or there
    // are none: one pane's box, as Fit stretches over it (fit_to::pane).
    static tile_size shared_size(const vector<pane>& panes) {
        if (panes.empty()) return {};
        for (const pane& one : panes)
            if (one.width != panes[0].width || one.height != panes[0].height) return {};
        return { panes[0].width, panes[0].height };
    }

    // A pane this small is a cell, not a grid: without the floor any 3x3 can
    // claim to be nine panes. It is what keeps the lattice honest.
    static constexpr int SMALLEST_PANE = 2;

    // Rows and columns painted one color from end to end.
    static void full_lines(const grid& one, int color,
                           vector<int>& rows, vector<int>& cols) {
        rows.clear();
        cols.clear();
        for (int r = 0; r < one.height; ++r) {
            bool whole = true;
            for (int c = 0; c < one.width && whole; ++c) whole = one.data[r][c] == color;
            if (whole) rows.push_back(r);
        }
        for (int c = 0; c < one.width; ++c) {
            bool whole = true;
            for (int r = 0; r < one.height && whole; ++r) whole = one.data[r][c] == color;
            if (whole) cols.push_back(c);
        }
    }

    // The stretches left between the cut lines. Neighbouring lines leave nothing
    // between them, so a wall two cells thick costs no extra freedom.
    static vector<pair<int, int>> spans_between(int length, const vector<int>& lines) {
        vector<pair<int, int>> spans;
        int start = 0;
        for (int line : lines) {
            if (line > start) spans.push_back({ start, line });
            start = line + 1;
        }
        if (length > start) spans.push_back({ start, length });
        return spans;
    }

    // The same stretches when no line made them: k equal parts.
    static vector<pair<int, int>> even_spans(int length, int parts) {
        vector<pair<int, int>> spans;
        for (int i = 0; i < parts; ++i)
            spans.push_back({ i * (length / parts), (i + 1) * (length / parts) });
        return spans;
    }

    // Lays the panes out in reading order, and says no when they fail a guard.
    static bool lay_out(const vector<pair<int, int>>& rows,
                        const vector<pair<int, int>>& cols, bool uneven,
                        vector<pane>& panes) {
        panes.clear();
        for (auto [top, bottom] : rows)
            for (auto [left, right] : cols) {
                const int height = bottom - top, width = right - left;
                if (height < SMALLEST_PANE || width < SMALLEST_PANE) return false;
                panes.push_back({ (int8_t)top, (int8_t)left, (int8_t)width, (int8_t)height });
            }
        if (panes.size() < 2) return false;
        if (!uneven)
            for (const auto& one_pane : panes)
                if (one_pane.width != panes.front().width ||
                    one_pane.height != panes.front().height) return false;
        return true;
    }

    // Every way this one grid could be cut, under these permissions.
    static void readings_of(const grid& one, split_freedom allowed,
                            vector<pair<pane_cut, vector<pane>>>& found) {
        found.clear();
        vector<int> row_lines, col_lines;
        vector<pane> panes;
        // A color that is absent offers no full line, so it costs nothing to ask.
        // A line with nothing on it is blank, as a line of the ground is: after a
        // plan deleted the ground, that is what its lines are.
        for (int color = TRANSPARENT; color < COLOR_COUNT; ++color) {
            const int8_t cut = color == TRANSPARENT ? Blank : Wall;
            if (!(cut == Wall ? allowed.wall : allowed.blank)) continue;
            full_lines(one, color, row_lines, col_lines);
            const auto rows = spans_between(one.height, row_lines);
            const auto cols = spans_between(one.width, col_lines);
            if (allowed.lattice && rows.size() > 1 && cols.size() > 1) {
                if (lay_out(rows, cols, allowed.uneven, panes))
                    found.push_back({ { cut, BothWays, (int8_t)color,
                                        (int8_t)rows.size(), (int8_t)cols.size() }, panes });
                continue;
            }
            if (cols.size() > 1 && rows.size() == 1 && lay_out(rows, cols, allowed.uneven, panes))
                found.push_back({ { cut, Across, (int8_t)color, 1, (int8_t)cols.size() }, panes });
            if (rows.size() > 1 && cols.size() == 1 && lay_out(rows, cols, allowed.uneven, panes))
                found.push_back({ { cut, Down, (int8_t)color, (int8_t)rows.size(), 1 }, panes });
        }
        if (!allowed.bare) return;
        const vector<pair<int, int>> whole_height{ { 0, one.height } };
        const vector<pair<int, int>> whole_width{ { 0, one.width } };
        for (int parts = 2; parts <= 4; ++parts) {
            if (one.width % parts == 0 &&
                lay_out(whole_height, even_spans(one.width, parts), allowed.uneven, panes))
                found.push_back({ { Bare, Across, -1, 1, (int8_t)parts }, panes });
            if (one.height % parts == 0 &&
                lay_out(even_spans(one.height, parts), whole_width, allowed.uneven, panes))
                found.push_back({ { Bare, Down, -1, (int8_t)parts, 1 }, panes });
        }
        if (!allowed.lattice) return;
        for (int rows = 2; rows <= 3; ++rows)
            for (int cols = 2; cols <= 3; ++cols)
                if (one.height % rows == 0 && one.width % cols == 0 &&
                    lay_out(even_spans(one.height, rows), even_spans(one.width, cols),
                            allowed.uneven, panes))
                    found.push_back({ { Bare, BothWays, -1, (int8_t)rows, (int8_t)cols }, panes });
    }

    // Cuts a grid that was not among the pairs the same way the pairs were cut,
    // so whatever the panes come to can be asked of a test input too. The caller
    // checks the pane sizes it needs; this only reproduces the reading.
    bool panes_of(const grid& one, vector<pane>& found) const {
        if (!held) return false;
        vector<int> row_lines, col_lines;
        vector<pair<int, int>> rows, cols;
        if (cut == Bare) {
            if (one.height % down || one.width % across) return false;
            rows = even_spans(one.height, down);
            cols = even_spans(one.width, across);
        } else {
            const int line = cut == Wall ? wall_of(one) : color;
            if (!is_color(line)) return false;
            full_lines(one, line, row_lines, col_lines);
            rows = spans_between(one.height, row_lines);
            cols = spans_between(one.width, col_lines);
        }
        if ((int)rows.size() != down || (int)cols.size() != across) return false;
        return lay_out(rows, cols, true, found);
    }

    // The wall's color in this grid: the one the pairs agreed on, or when they
    // agreed only on the cut, the one color whose full lines
    // cut this grid the same way. -1 when none does, or several do.
    int wall_of(const grid& one) const {
        if (cut != Wall) return -1;
        if (is_color(color)) return color;
        int wall = -1;
        vector<int> row_lines, col_lines;
        vector<pane> panes;
        for (int line = 0; line < COLOR_COUNT; ++line) {
            full_lines(one, line, row_lines, col_lines);
            const auto rows = spans_between(one.height, row_lines);
            const auto cols = spans_between(one.width, col_lines);
            if ((int)rows.size() != down || (int)cols.size() != across ||
                !lay_out(rows, cols, true, panes)) continue;
            if (wall >= 0) return -1;
            wall = line;
        }
        return wall;
    }

    // Each grid's walls with their color left out, so the pairs can agree on a
    // cut whose color is each grid's own. A grid that two such walls cut the
    // same way has no one wall, so neither is kept.
    static void without_wall_colors(vector<pair<pane_cut, vector<pane>>>& found) {
        for (auto& [named, panes] : found)
            if (named.cut == Wall) named.color = -1;
        vector<pair<pane_cut, vector<pane>>> kept;
        for (const auto& one : found)
            if (ranges::count_if(found, [&](const auto& other) { return other.first == one.first; }) == 1)
                kept.push_back(one);
        found = std::move(kept);
    }

    // The same cut in every pair, wall color and all. Failing that, the same cut
    // with each grid naming its own wall: four areas split by one line each way
    // are the rule, whatever color the lines are drawn in.
    static pane_split_facts detect(const vector<example>& train,
                                   split_freedom allowed = {}) {
        pane_split_facts facts = agreed_on(train, allowed, false);
        if (!facts.holds()) facts = agreed_on(train, allowed, true);
        return facts;
    }

    // Reads every training input, keeps the readings they all agree on, and holds
    // only when exactly one of those survives the precedence.
    static pane_split_facts agreed_on(const vector<example>& train, split_freedom allowed,
                                      bool own_walls) {
        if (train.empty()) return {};
        vector<pair<pane_cut, vector<pane>>> found;
        vector<pane_cut> agreed;
        for (size_t i = 0; i < train.size(); ++i) {
            readings_of(train[i].input, allowed, found);
            if (own_walls) without_wall_colors(found);
            // A wall is a wall whether or not the answer is a pane wide: the
            // input alone shows it, so it is admitted on its own evidence. A cut
            // that follows no line shows nothing, and fits almost any grid, so it
            // is admitted only when the output is the size of one pane. The
            // asymmetry is the point -- it is the line between a fact and a guess.
            vector<pane_cut> here;
            for (const auto& [named, panes] : found) {
                if (named.cut != Bare) { here.push_back(named); continue; }
                for (const auto& one_pane : panes)
                    if (one_pane.width == train[i].output.width &&
                        one_pane.height == train[i].output.height) {
                        here.push_back(named);
                        break;
                    }
            }
            if (i == 0) { agreed = here; continue; }
            erase_if(agreed, [&](const pane_cut& named) {
                return ranges::find(here, named) == here.end(); });
            if (agreed.empty()) return {};
        }
        if (agreed.empty()) return {};
        // Fewest assumptions first: a wall outranks a blank line, which outranks
        // no line at all. Whatever is still tied after that is an argument, not a fact.
        const int8_t plainest = ranges::min(agreed, {}, &pane_cut::cut).cut;
        erase_if(agreed, [&](const pane_cut& named) { return named.cut != plainest; });
        if (agreed.size() != 1) return {};

        const pane_cut won = agreed.front();
        pane_split_facts facts{ won };
        for (const auto& train_pair : train) {
            readings_of(train_pair.input, allowed, found);
            if (own_walls) without_wall_colors(found);
            for (const auto& [named, panes] : found)
                if (named == won) { facts.pairs.push_back(panes); break; }
        }
        if (facts.pairs.size() != train.size()) return {};
        facts.held = true;
        return facts;
    }
};

// WHAT THE PANES SAY TOGETHER, once they have been cut apart.
//
// The plain way to read a pair of panes is to write down every combination of
// colors that turned up and what each produced. That is a table of 10^k rows
// which cannot answer a combination it never saw, and it carries no reason --
// sixteen rows where the puzzle has one idea.
//
// So the colors are dropped first. At each position only ONE thing is asked of
// each pane: is there something here, or is this ground? That leaves a mask of
// k bits, and for every mask the answer is either a color to paint or one named
// pane to copy. Both intersection and priority fall out of that shape without
// being named: an AND is the mask with every bit set painting one color, a
// priority order is each mask copying its leading pane. And a mask that never
// turned up in training is still answerable, because the colors that would have
// made it no longer matter.
struct pane_mask_rule {
    // What one mask says. A pane index copies that pane's own color at the cell;
    // otherwise paint is laid down; or the cell is left empty, where the output
    // has nothing because a plan fills it last. Exactly one of the three is set.
    struct answer {
        int8_t paint = -1;
        int8_t copy_pane = -1;
        bool leaves_empty = false;
        bool said() const { return paint >= 0 || copy_pane >= 0 || leaves_empty; }
    };
    // Indexed by the mask, so 1 << panes entries. A mask nothing ever said is
    // left unsaid rather than guessed.
    vector<answer> by_mask;
    // The pose every pane after the first is read through. Panes facing each
    // other across a wall are mirrored, not shifted, and without this their
    // masks line up wrong and every reading conflicts.
    int8_t posed = 0;
    // How many panes each mask counts, so a reader knows how wide the bits are.
    int8_t panes = 0;
    bool held = false;
    bool holds() const { return held; }

    // What pane p shows at (r, c). The first pane is read as it lies; the rest
    // are turned into its frame, which is what makes them comparable.
    static int shown(const vector<pane_split_facts::pane>& boxes, int p, const grid& input,
                     int r, int c, int pose) {
        return boxes[p].color_at(input, r, c, p ? pose : 0);
    }
    // Which panes have something at (r, c), one bit each.
    static int mask_at(const vector<pane_split_facts::pane>& boxes, const grid& input,
                       int r, int c, int pose) {
        int mask = 0;
        for (int p = 0; p < (int)boxes.size(); ++p)
            if (shown(boxes, p, input, r, c, pose) != TRANSPARENT) mask |= 1 << p;
        return mask;
    }

    // Reads every training pair under one pose and says what each mask means,
    // or says nothing when two cells with the same mask disagree beyond repair.
    static pane_mask_rule read_with(const vector<example>& train,
                                    const pane_split_facts& split, int pose) {
        pane_mask_rule rule;
        if (train.size() != split.pairs.size() || split.pairs.empty()) return {};
        const int panes = (int)split.pairs.front().size();
        if (panes < 2 || panes > 8) return {};
        rule.panes = (int8_t)panes;
        rule.posed = (int8_t)pose;
        rule.by_mask.assign((size_t)1 << panes, {});
        // A pane may only copy when every cell of that mask agreed it could, so
        // the candidates start open and are struck off as the pairs disagree.
        vector<uint8_t> may_copy((size_t)1 << panes, (uint8_t)((1u << panes) - 1));
        // What each mask laid first -- a color, or nothing -- and whether every
        // later cell of that mask laid the same.
        vector<uint8_t> painted((size_t)1 << panes, 0), agreed((size_t)1 << panes, 1);
        vector<int8_t> laid((size_t)1 << panes, TRANSPARENT);
        for (size_t which = 0; which < train.size(); ++which) {
            const auto& boxes = split.pairs[which];
            const grid& input = train[which].input;
            const grid& output = train[which].output;
            if ((int)boxes.size() != panes) return {};
            for (const auto& box : boxes)
                if (box.width != output.width || box.height != output.height || !box.keeps_size(pose))
                    return {};
            for (int r = 0; r < output.height; ++r)
                for (int c = 0; c < output.width; ++c) {
                    const int mask = mask_at(boxes, input, r, c, pose);
                    const int now = output.data[r][c];
                    for (int p = 0; p < panes; ++p)
                        if (shown(boxes, p, input, r, c, pose) != now)
                            may_copy[mask] &= (uint8_t)~(1u << p);
                    if (!painted[mask]) {
                        painted[mask] = 1;
                        laid[mask] = (int8_t)now;
                    } else if (laid[mask] != now) {
                        agreed[mask] = 0;
                    }
                }
        }
        // A mask that always laid the same color, or always left the cell empty,
        // says so; one that did not must name a pane it always copied, or it
        // says nothing and the rule falls.
        for (size_t mask = 0; mask < rule.by_mask.size(); ++mask) {
            if (!painted[mask]) continue;
            if (agreed[mask]) {
                if (laid[mask] == TRANSPARENT) rule.by_mask[mask].leaves_empty = true;
                else rule.by_mask[mask].paint = laid[mask];
                continue;
            }
            if (!may_copy[mask]) return {};
            for (int p = 0; p < panes; ++p)
                if (may_copy[mask] >> p & 1) { rule.by_mask[mask].copy_pane = (int8_t)p; break; }
        }
        rule.held = true;
        return rule;
    }

    // Draws what the panes of one input come to. Says false when the split will
    // not cut this grid, or when a mask turns up that training never explained --
    // declining beats guessing, since a guess here is a wrong answer that looks
    // like a right one.
    bool draw(const grid& input, const pane_split_facts& split, grid& drawn) const {
        vector<pane_split_facts::pane> boxes;
        if (!holds() || !split.panes_of(input, boxes)) return false;
        if ((int)boxes.size() != panes) return false;
        const auto& first = boxes.front();
        for (const auto& box : boxes)
            if (box.width != first.width || box.height != first.height || !box.keeps_size(posed))
                return false;
        drawn = grid(first.width, first.height);
        for (int r = 0; r < drawn.height; ++r)
            for (int c = 0; c < drawn.width; ++c) {
                const answer& says = by_mask[mask_at(boxes, input, r, c, posed)];
                if (!says.said()) return false;
                drawn.data[r][c] = (int8_t)(says.leaves_empty ? TRANSPARENT
                                     : says.paint >= 0   ? says.paint
                                                         : shown(boxes, says.copy_pane, input, r, c, posed));
            }
        return true;
    }

    // Tries every pose and keeps the plainest one that redraws every output. As
    // is comes first, so a pose is only ever worn when it is doing work.
    // With no ground color, "nothing here" is an empty cell: what is left
    // where a plan deleted the ground.
    static pane_mask_rule detect(const vector<example>& train,
                                 const pane_split_facts& split) {
        if (!split.holds()) return {};
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            const pane_mask_rule rule = read_with(train, split, pose);
            if (rule.holds() && redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    return rule.draw(pair.input, split, drawn);
                })) return rule;
        }
        return {};
    }
};

// WHICH PANE THE ANSWER USES, directly or enlarged behind the divider walls.
//
// pane_mask_rule reads the panes against each other and makes a new drawing.
// This one chooses a pane, either as the answer or as its miniature blueprint.
// The rule keeps the reason for choosing, and the ways to choose
// are ordered plainest first -- a property the pane HAS beats a place it SITS,
// because a place is a magic number and a property is a reason. The first way
// that names the right pane in every training pair wins; nothing is scored.
struct pane_pick_rule {
    enum : int8_t {
        // It is the one unlike its fellows, by drawing or by palette.
        OddOneOut, OddColor,
        // It is the one its fellows repeat.
        TheyRepeat,
        // It is the one that breaks a symmetry the others keep, or keeps one
        // they break. `which` is the pose the symmetry folds along.
        NotSymmetric, OnlySymmetric,
        // It is the one carrying most or least. Measured and never needed so
        // far, carried because they cost one line each and the panes are cut.
        Fullest, Emptiest, MostColored, LeastColored,
        // It is simply the one in that place. Last, because it gives no reason.
        AtIndex,
        WAYS
    };
    int8_t way = AtIndex;
    // The pose a symmetry folds along, or the place AtIndex means.
    int8_t which = 0;
    bool held = false;
    bool fitted = false;
    bool holds() const { return held; }

    // The four folds a square drawing can sit on, as poses: along the diagonal,
    // top to bottom, left to right, and half way round.
    static constexpr int8_t FOLDS[] = { 1, 2, 4, 6 };

    static bool same_drawing(const grid& one, const pane_split_facts::pane& left,
                             const pane_split_facts::pane& right) {
        return same_shape(left.width, left.height, [&](int row, int col) { return left.color_at(one, row, col); },
                          right.width, right.height, [&](int row, int col) { return right.color_at(one, row, col); },
                          DRAWING);
    }
    static bool folds_onto_itself(const grid& one, const pane_split_facts::pane& box, int pose) {
        if (!box.keeps_size(pose)) return false;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c)
                if (box.color_at(one, r, c) != box.color_at(one, r, c, pose)) return false;
        return true;
    }
    static uint16_t palette_of(const grid& one, const pane_split_facts::pane& box) {
        uint16_t seen = 0;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c) {
                const int was = box.color_at(one, r, c);
                if (is_color(was)) seen |= (uint16_t)(1u << was);
            }
        return seen;
    }
    static int carried_by(const grid& one, const pane_split_facts::pane& box) {
        int count = 0;
        for (int r = 0; r < box.height; ++r)
            for (int c = 0; c < box.width; ++c)
                count += box.color_at(one, r, c) != TRANSPARENT;
        return count;
    }

    // The one place where a count stands alone, or -1 when none does or several do.
    static int only_extreme(const vector<int>& counts, bool most) {
        int best = counts.front();
        for (int one : counts) best = most ? max(best, one) : min(best, one);
        int found = -1;
        for (size_t i = 0; i < counts.size(); ++i)
            if (counts[i] == best) { if (found >= 0) return -1; found = (int)i; }
        return found;
    }

    // Which pane this way names on this grid, or -1 when it names none.
    static int pick(const grid& one, const vector<pane_split_facts::pane>& boxes,
                    int8_t way, int8_t which) {
        const int count = (int)boxes.size();
        if (count < 2) return -1;
        switch (way) {
        case AtIndex:
            return which < count ? which : -1;
        case OddOneOut:
            return unique_piece((int)twin_by::drawing, count,
                [&](int i) { return boxes[i].width; },
                [&](int i) { return boxes[i].height; },
                [&](int i, int r, int c) { return boxes[i].color_at(one, r, c); });
        case TheyRepeat: {
            vector<int> alike(count, 0);
            for (int i = 0; i < count; ++i)
                for (int j = 0; j < count; ++j)
                    alike[i] += i != j && same_drawing(one, boxes[i], boxes[j]);
            // Alone among fellows that have each other, or one of the crowd.
            const int found = only_extreme(alike, way == TheyRepeat);
            if (found < 0 || alike[found] != (way == TheyRepeat ? count - 1 : 0)) return -1;
            return way == TheyRepeat || alike[found] == 0 ? found : -1;
        }
        case OddColor: {
            vector<int> alike(count, 0);
            for (int i = 0; i < count; ++i)
                for (int j = 0; j < count; ++j)
                    alike[i] += i != j && palette_of(one, boxes[i]) == palette_of(one, boxes[j]);
            const int found = only_extreme(alike, false);
            return found >= 0 && alike[found] == 0 ? found : -1;
        }
        case NotSymmetric: case OnlySymmetric: {
            int found = -1;
            for (int i = 0; i < count; ++i) {
                const bool folds = folds_onto_itself(one, boxes[i], which);
                if (folds != (way == OnlySymmetric)) continue;
                if (found >= 0) return -1;
                found = i;
            }
            return found;
        }
        case Fullest: case Emptiest: {
            vector<int> carried;
            for (const auto& box : boxes) carried.push_back(carried_by(one, box));
            return only_extreme(carried, way == Fullest);
        }
        case MostColored: case LeastColored: {
            vector<int> many;
            for (const auto& box : boxes) many.push_back(popcount(palette_of(one, box)));
            return only_extreme(many, way == MostColored);
        }
        default: return -1;
        }
    }

    // Hands back the named pane, or fits it behind the original wall color.
    bool draw(const grid& input, const pane_split_facts& split, grid& drawn) const {
        vector<pane_split_facts::pane> boxes;
        if (!holds() || !split.panes_of(input, boxes)) return false;
        const int found = pick(input, boxes, way, which);
        if (found < 0) return false;
        const auto& source = boxes[(size_t)found];
        if (!fitted) {
            drawn = source.drawn(input);
            return true;
        }
        // A cell of the chosen pane describes a pane of the full lattice.
        // Replay Fit's integer sampling, then restore the divider color, just
        // as the emitted sentences do. The existing pick owns ties and counts.
        const int wall = split.wall_of(input);
        if (!is_color(wall) ||
            source.width != split.across || source.height != split.down) return false;
        drawn = grid(input.width, input.height);
        for (int r = 0; r < input.height; ++r)
            for (int c = 0; c < input.width; ++c)
                drawn.data[r][c] = (int8_t)(input.data[r][c] == wall ? wall :
                    source.color_at(input, fit_source_cell(r, source.height, input.height),
                                           fit_source_cell(c, source.width, input.width)));
        return true;
    }

    // Walks the ways in order and keeps the first that names the right pane in
    // every pair. Order is the whole tiebreak: a reason before a place.
    static pane_pick_rule detect(const vector<example>& train,
                                 const pane_split_facts& split) {
        if (!split.holds() || train.empty()) return {};
        // Preserve direct picks before considering the enlarged reading.
        for (bool fitted : { false, true }) {
            auto names_it = [&](int8_t way, int8_t which) {
                const pane_pick_rule trying{ way, which, true, fitted };
                return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    return trying.draw(pair.input, split, drawn);
                });
            };
            // For a miniature blueprint, occupancy is the simplest reason:
            // its palette can also look odd merely because it has less ink.
            static constexpr int8_t fitted_ways[] = {
                Emptiest, Fullest, OddOneOut, OddColor, TheyRepeat,
                NotSymmetric, OnlySymmetric, MostColored, LeastColored, AtIndex
            };
            for (int8_t index = 0; index < WAYS; ++index) {
                const int8_t way = fitted ? fitted_ways[index] : index;
                if (way == NotSymmetric || way == OnlySymmetric) {
                    for (int8_t fold : FOLDS)
                        if (names_it(way, fold)) return { way, fold, true, fitted };
                } else if (way == AtIndex) {
                    for (int8_t place = 0; place < (int8_t)split.pairs.front().size(); ++place)
                        if (names_it(way, place)) return { way, place, true, fitted };
                } else if (names_it(way, 0)) {
                    return { way, 0, true, fitted };
                }
            }
        }
        return {};
    }
};

// EVERY PANE'S DRAWING, CUT DOWN TO WHAT IT HOLDS AND GATHERED WHERE ITS PANE WAS.
//
// pane_mask_rule makes one drawing out of the panes and pane_pick_rule keeps
// one of them; here each pane keeps its own drawing, cropped, and the drawings
// are laid together the way the panes lay, in any number of rows and columns.
// After explicit ground removal, `Panes Layout[0]` paints this: Panes hands each
// pane back in its own frame, Layout bands the pieces by their frames and draws
// what each holds, and Layout[0] makes every cell as big as the largest drawing.
// Nothing about the drawings' size is in the rule; each grid says its own.
struct pane_gather_rule {
    bool held = false;
    bool holds() const { return held; }

    // Draws what the sentence paints for one input, on the canvas the size rule
    // gives it -- or, under BoundingBox, on the largest canvas there is, closed on
    // what was painted, as the ops close it. Declines when no pane holds
    // anything, since then nothing is laid out at all.
    static bool draw(const grid& input, const pane_split_facts& split, const output_dimension_rule& size,
                     grid& drawn) {
        vector<pane_split_facts::pane> boxes;
        if (!split.panes_of(input, boxes) || boxes.empty() ||
            (int)boxes.size() > MAX_OBJECTS_IN_A_LIST) return false;
        const int count = (int)boxes.size();
        // What each pane holds, and the cell that holds the largest of them.
        vector<cell_box> drawings((size_t)count);
        int cell_w = 0, cell_h = 0;
        for (int p = 0; p < count; ++p) {
            const auto& box = boxes[(size_t)p];
            for (int r = 0; r < box.height; ++r)
                for (int c = 0; c < box.width; ++c)
                    if (box.color_at(input, r, c) != TRANSPARENT) drawings[(size_t)p].include_cell(r, c);
            cell_w = max(cell_w, drawings[(size_t)p].width());
            cell_h = max(cell_h, drawings[(size_t)p].height());
        }
        if (cell_w == 0 || cell_h == 0) return false;
        // The bands come off the frames, the same grouping Layout reads.
        int8_t row_of[MAX_OBJECTS_IN_A_LIST], col_of[MAX_OBJECTS_IN_A_LIST];
        auto frame = [&](int p) { return boxes[(size_t)p].box(); };
        const int rows = bands_of(count, true, frame, row_of), cols = bands_of(count, false, frame, col_of);
        const int laid_w = min(cols * cell_w, MAX_GRID_DIM), laid_h = min(rows * cell_h, MAX_GRID_DIM);
        return draw_on_canvas(input, size, drawn, [&](grid& canvas) {
            for (int p = 0; p < count; ++p) {
                const auto& box = boxes[(size_t)p];
                const cell_box& drawing = drawings[(size_t)p];
                for (int r = 0; r < min(cell_h, drawing.height()); ++r)
                    for (int c = 0; c < min(cell_w, drawing.width()); ++c) {
                        const int row = row_of[p] * cell_h + r, col = col_of[p] * cell_w + c;
                        const int color = box.color_at(input, drawing.row0 + r, drawing.col0 + c);
                        if (color == TRANSPARENT || row >= min(laid_h, canvas.height) ||
                            col >= min(laid_w, canvas.width)) continue;
                        canvas.data[row][col] = (int8_t)color;
                    }
            }
            return true;
        });
    }

    // With no ground color, what a pane holds is what is not empty.
    static pane_gather_rule detect(const vector<example>& train, const pane_split_facts& split,
                                   const output_dimension_rule& size) {
        if (!split.holds()) return {};
        pane_gather_rule rule;
        rule.held = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            return draw(pair.input, split, size, drawn);
        });
        return rule;
    }
};

}  // namespace arc
