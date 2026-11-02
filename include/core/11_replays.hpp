// How a rule is checked: replayed on every pair with the arithmetic its op runs,
// and judged pair by pair (brings_nearer). One replay per rule, and
// carry_the_hand, which carries the pieces through the rules already said.
#pragma once
#include "10_picks_and_frames.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace arc {

// THE CELLS OF EVERY PAIR, a list each, (row, col): what a replay sees.
using cells_by_pair = vector<vector<std::pair<int, int>>>;

// A cell laid down, in the color laid there; and every pair's, a list each.
struct cell_color { int8_t row, col, color; };
using colors_by_pair = vector<vector<cell_color>>;

// Whether `cells` holds the cell at (row, col) of pair `which`.
inline bool on_cells(const cells_by_pair& cells, size_t which, int row, int col) {
    return which < cells.size() &&
           ranges::find(cells[which], std::pair<int, int>{ row, col }) != cells[which].end();
}

// THE CELLS OF EVERY PAIR, MARKED where they lie: what on_cells answers, at a
// look rather than a search, for a claim that asks it of every cell left over.
// A cell past the edge of the largest grid -- a piece the rules moved off it --
// is no cell left over, and is not marked.
struct cell_marks {
    vector<cell_set> at;
    cell_marks() = default;
    explicit cell_marks(const cells_by_pair& cells) : at(cells.size()) {
        for (size_t which = 0; which < cells.size(); ++which)
            for (const auto& [row, col] : cells[which]) at[which].add(row, col);
    }
};
inline bool on_cells(const cell_marks& cells, size_t which, int row, int col) {
    return which < cells.at.size() && cells.at[which].has(row, col);
}

// Whether a paint laid `lay` lands on a cell where `down` shows: over it lands
// everywhere; under it only where nothing is, as Underlay paints; onto it only
// where something is, as MaskBy[10] keeps; onto one color only where that color
// is, as MaskBy of it keeps; and off one color everywhere else, as MaskOut of
// it keeps (mask_keeps, 15_op_math.hpp, what the ops keep).
inline bool lands(int8_t lay, int8_t down) {
    using rule = correspondence_election::fired_rule;
    if (lay == rule::laid_under) return down == (int8_t)TRANSPARENT;
    if (lay == rule::laid_onto) return mask_keeps(down, 0, /*any=*/true, /*keep_match=*/true);
    if (lay >= rule::laid_off_color) return mask_keeps(down, (int8_t)(lay - rule::laid_off_color), false, false);
    if (lay >= rule::laid_onto_color) return mask_keeps(down, (int8_t)(lay - rule::laid_onto_color), false, true);
    return true;
}

// HOW A PAINT IS LAID: the way `lay` says (lands) on what `canvas` holds down,
// pair by pair, the color each cell shows before the paint -- the hand the
// replay is handed, where it names no canvas (paint_holds) -- and in the
// colors it draws, or all in one color -- the one the output shows where it
// first lands, the line's Recolor after the paint, whatever the colors drawn --
// or in the colors it draws where the output agrees, and else in one.
struct laying {
    int8_t lay = correspondence_election::fired_rule::laid_over;
    const vector<cells>* canvas = nullptr;
    enum colors : int8_t { as_drawn, in_one, own_else_one } in = as_drawn;
};

// WHAT A REPLAY SAW, where its rule holds -- every replay below hands one back,
// or nothing where the rule does not hold. The cells the rule lays down right
// on each output, pair by pair: where its paint landed, where its pieces
// landed, what they came out in or kept. The cells its picked pieces stood on
// that it took them off or left as they were: where they slid from, what they
// lost, the pieces a copy was drawn from. The cells it lays wrong, in what it
// lays there (brings_nearer). How many pieces it changed; the one color it
// paints in, -1 for the colors it draws; and whether it picked every piece.
struct replayed {
    cells_by_pair drawn, left;
    colors_by_pair wrong;
    int pieces = 0;
    int8_t paint = -1;
    bool every_piece = true;
    int wrong_cells() const {
        int count = 0;
        for (const auto& cells : wrong) count += (int)cells.size();
        return count;
    }
    int right_cells() const {
        int count = 0;
        for (const auto& cells : drawn) count += (int)cells.size();
        return count;
    }
};

// THE ONE JUDGE OF A RULE, PAIR BY PAIR (the user: a rule thrown out for one
// wrong cell is thrown out too early). In a pair a rule makes cells right --
// `gained`: the cell comes out as the output with what it laid there, and did
// not before (comes_out) -- and lays cells wrong, where it does not. It
// holds in that pair where it gains more than it lays wrong, so it brings the
// pair nearer its output, one cell gained at least: at work (replay_facts).
// Its wrong cells are left for a rule said after it, or the next pass on the
// picture so far, to set right (bdad9b1f's two lines, each wrong only where
// they cross, the crossing a color of its own). Where the line does not draw
// its picture (replay_facts::tolerant), no pass could see a cell left wrong,
// and a rule holds only where it lays none. A move or a removal is at work
// only where it changed a piece too (`acted`). In a finished pair
// (replay_facts::finished) there is nothing to gain: a rule holds there where
// it lays nothing wrong.
struct pair_tally { int gained = 0, wrong = 0; bool acted = true; };
inline bool brings_nearer(const replay_facts& facts, size_t which, const pair_tally& tally) {
    if (facts.finished(which)) return tally.wrong == 0;
    return tally.acted && tally.gained > tally.wrong && (facts.tolerant || tally.wrong == 0);
}

// A PAINT, WALKED: on each pair, every cell its picked pieces paint, once each
// -- `draw(which, cut, put)` hands each one, in the color it paints it, to
// `put(row, col, color)`, which answers whether the paint may still hold, and
// answers false for a pair it cannot draw at all -- with the color laid there
// first and last, and whether two of its own disagree there (torn). Every way
// the paint is laid reads the walk (lay_paint), so a paint asked several ways
// is walked once, and laid on a new canvas without walking it again.
struct paint_walk {
    struct cell { int8_t row, col, first, last; bool torn; };
    vector<vector<cell>> pairs;
    bool every_piece = true;
};

// A LAY KEPT FOR SCORING: marks of its right cells and a count of its wrong
// ones. Its walk and laying can supply the full lists once this kind offers
// it. The walk belongs to the cut; the canvas is unchanged while offers are
// weighed, and a winning lay is materialized before that canvas changes.
struct marked_paint {
    cell_marks drawn;
    int wrong = 0;
    int8_t paint = -1;
    bool every_piece = true;
    const paint_walk* walk = nullptr;
    laying how;
    int wrong_cells() const { return wrong; }
};

// What a paint laid `how` reads where it lands on pair `which`: nothing, laid
// over what is down, where it lands everywhere; else its canvas's pair.
inline const cells* landing_on(const laying& how, size_t which) {
    using rule = correspondence_election::fired_rule;
    return how.lay == rule::laid_over ? nullptr : &(*how.canvas)[which];
}

// WHAT LANDS OF A WALK laid the way `how` says on its canvas (lands): over what
// is down, every cell, in the last color laid there; under it, where nothing
// is, in the first -- the first paint to reach a cell keeps it, this one's own
// earlier cells too, as Underlay paints; onto or off what is down, or one color
// of it, the cells the canvas lets through. Each handed to `see(cell, color,
// torn)`, in the order the walk first reached them.
template <class See>
inline void each_landing(const paint_walk& walk, size_t which, const laying& how, See see) {
    using rule = correspondence_election::fired_rule;
    const cells* down = landing_on(how, which);
    const bool first_wins = how.lay == rule::laid_under;
    for (const paint_walk::cell& cell : walk.pairs[which])
        if (!down || lands(how.lay, (*down)[cell.row][cell.col]))
            see(cell, first_wins ? cell.first : cell.last, !first_wins && cell.torn);
}

// A WALKED PAINT, LAID: what lands of it (each_landing) -- laid in one color,
// every cell in the color the output shows most where it lands, over every
// pair, the first to land of those on a tie: a paint is never no color -- and
// each pair judged (brings_nearer): a cell is gained where it comes out as the
// output with what landed and did not with what the canvas showed there before
// (comes_out), and a cell its own paint laid twice in two colors is wrong
// whatever the output shows. Sees that one color (-1 laid in the colors
// drawn), the cells that landed right and the ones that landed wrong, and
// whether it picked every piece.
template <class Seen = replayed>
inline optional<Seen> lay_paint(const replay_facts& facts, const paint_walk& walk, const laying& how) {
    ++inference_work.lays;
    Seen seen;
    seen.every_piece = walk.every_piece;
    const size_t pairs = walk.pairs.size();
    if (how.in == laying::in_one) {
        int shown[COLOR_COUNT] = {};
        for (size_t which = 0; which < pairs; ++which)
            each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t, bool) {
                if (const int color = facts.pairs[which].whole.output.data[cell.row][cell.col]; is_color(color))
                    ++shown[color];
            });
        const int most = *ranges::max_element(shown);
        for (size_t which = 0; which < pairs && seen.paint < 0 && most > 0; ++which)
            each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t, bool) {
                if (const int color = facts.pairs[which].whole.output.data[cell.row][cell.col];
                    seen.paint < 0 && is_color(color) && shown[color] == most)
                    seen.paint = (int8_t)color;
            });
        if (!is_color(seen.paint)) return nullopt;
    }
    if constexpr (std::is_same_v<Seen, marked_paint>) {
        seen.drawn.at.resize(pairs);
        seen.walk = &walk;
        seen.how = how;
    } else {
        seen.drawn.assign(pairs, {});
        seen.wrong.assign(pairs, {});
    }
    for (size_t which = 0; which < pairs; ++which) {
        const figure_pair& pair = facts.pairs[which];
        const cells& before = (*how.canvas)[which];
        pair_tally tally;
        each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t color, bool torn) {
            const int8_t laid = how.in == laying::in_one ? seen.paint : color;
            if (!torn && comes_out(pair, cell.row, cell.col, laid)) {
                if constexpr (std::is_same_v<Seen, marked_paint>)
                    seen.drawn.at[which].add(cell.row, cell.col);
                else seen.drawn[which].push_back({ cell.row, cell.col });
                tally.gained += !comes_out(pair, cell.row, cell.col, before[cell.row][cell.col]);
            } else {
                ++tally.wrong;
                if constexpr (std::is_same_v<Seen, marked_paint>) ++seen.wrong;
                else seen.wrong[which].push_back({ cell.row, cell.col, laid });
            }
        });
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    return seen;
}

// A WALK COUNTED BY WHAT IS DOWN WHERE ITS CELLS LIE, pair by pair: for each
// thing a cell can show on the canvas -- nothing, or each color -- how many of
// the walk's cells lie on it, and of those, laid in the last color laid there
// (over what is down, or onto it, and not torn) or the first (under it), how
// many come out as the output so and how many of those did not come out
// already with what is down (comes_out); and how many the output shows in each
// color, and of those how many came out already (laid in one color). Each way
// the walk is laid on that canvas takes a sum of these (may_hold), so a lay
// that cannot hold is known without going over its cells again.
struct walk_tally {
    // 0 where nothing is down, 1 + c where color c is.
    static constexpr int downs = COLOR_COUNT + 1;
    struct of_pair {
        int cells[downs] = {}, right_last[downs] = {}, gained_last[downs] = {}, right_first[downs] = {},
            gained_first[downs] = {};
        int shows[downs][COLOR_COUNT] = {}, shown_before[downs][COLOR_COUNT] = {};
    };
    vector<of_pair> pairs;
};
inline walk_tally tally_walk(const replay_facts& facts, const paint_walk& walk, const vector<cells>& canvas) {
    walk_tally tally;
    tally.pairs.resize(walk.pairs.size());
    for (size_t which = 0; which < walk.pairs.size(); ++which) {
        walk_tally::of_pair& counted = tally.pairs[which];
        const figure_pair& pair = facts.pairs[which];
        for (const paint_walk::cell& cell : walk.pairs[which]) {
            const int8_t down = which < canvas.size() ? canvas[which][cell.row][cell.col] : (int8_t)TRANSPARENT;
            const int at = down == (int8_t)TRANSPARENT ? 0 : 1 + down;
            const bool came_out = comes_out(pair, cell.row, cell.col, down);
            const int shows = pair.whole.output.data[cell.row][cell.col];
            ++counted.cells[at];
            if (!cell.torn && comes_out(pair, cell.row, cell.col, cell.last)) {
                ++counted.right_last[at];
                counted.gained_last[at] += !came_out;
            }
            if (comes_out(pair, cell.row, cell.col, cell.first)) {
                ++counted.right_first[at];
                counted.gained_first[at] += !came_out;
            }
            if (is_color(shows)) {
                ++counted.shows[at][shows];
                counted.shown_before[at][shows] += came_out;
            }
        }
    }
    return tally;
}

// WHETHER A WALK LAID `how` MAY HOLD, read off its tally on the canvas `how`
// names (walk_tally) as lay_paint judges it, cell by cell: false where it
// cannot; true where it may, and then it is laid cell by cell -- as it is
// where, laid in one color, two colors tie for the one the output shows most,
// which only the order the cells landed in breaks.
inline bool may_hold(const replay_facts& facts, const walk_tally& tally, const laying& how) {
    using rule = correspondence_election::fired_rule;
    constexpr int downs = walk_tally::downs;
    bool lands_on[downs];
    for (int at = 0; at < downs; ++at) lands_on[at] = lands(how.lay, at == 0 ? (int8_t)TRANSPARENT : (int8_t)(at - 1));
    const bool first_wins = how.lay == rule::laid_under;
    int paint = -1;
    if (how.in == laying::in_one) {
        long long shown[COLOR_COUNT] = {};
        for (const walk_tally::of_pair& counted : tally.pairs)
            for (int at = 0; at < downs; ++at)
                if (lands_on[at])
                    for (int color = 0; color < COLOR_COUNT; ++color) shown[color] += counted.shows[at][color];
        const long long most = *ranges::max_element(shown);
        if (most == 0) return false;
        if (ranges::count(shown, most) > 1) return true;
        paint = (int)(ranges::find(shown, most) - std::begin(shown));
    }
    for (size_t which = 0; which < tally.pairs.size(); ++which) {
        const walk_tally::of_pair& counted = tally.pairs[which];
        pair_tally said;
        int cells = 0, right = 0;
        for (int at = 0; at < downs; ++at) {
            if (!lands_on[at]) continue;
            cells += counted.cells[at];
            if (paint >= 0) {
                right += counted.shows[at][paint];
                said.gained += counted.shows[at][paint] - counted.shown_before[at][paint];
            } else if (first_wins) {
                right += counted.right_first[at];
                said.gained += counted.gained_first[at];
            } else {
                right += counted.right_last[at];
                said.gained += counted.gained_last[at];
            }
        }
        said.wrong = cells - right;
        if (!brings_nearer(facts, which, said)) return false;
    }
    return true;
}

// THE WALK `draw` MAKES (paint_walk), or nothing where a pair cannot be drawn
// -- or where, laid only the way `only` says, the paint cannot hold, done with
// at once: allowed no wrong cell, at the first -- laid in the colors drawn, a
// cell that does not come out so (comes_out); laid in one, a cell the output
// shows in another color than where the first landed -- and else once as many
// cells are sure to be wrong as the pair could gain. THE MOST A PAIR COULD
// GAIN is every cell that does not come out already with what is down; a cell
// first laid in the colors drawn that does not come out so stays wrong -- laid
// again in another, it is torn. Laid in one color, no cell is sure to be wrong
// before the color is known.
template <class Draw>
inline optional<paint_walk> walk_paint(const replay_facts& facts, const vector<partner_pieces>& handed, Draw draw,
                                       const laying* only = nullptr) {
    ++inference_work.walks;
    using rule = correspondence_election::fired_rule;
    if (!facts.overlays || facts.train.empty()) return nullopt;
    paint_walk walk;
    walk.pairs.resize(facts.train.size());
    const bool exact = only && !facts.tolerant;
    int8_t first_shown = (int8_t)TRANSPARENT;
    bool first = true;
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const figure_pair& pair = facts.pairs[which];
        const grid& output = pair.whole.output;
        const partner_pieces& cut = handed[which];
        walk.every_piece = walk.every_piece && cut.every_piece_picked();
        // Each cell's place in the walk, -1 for none yet: every cell, since a
        // piece the rules moved may lay one past the grid's edge.
        int16_t index[MAX_GRID_DIM][MAX_GRID_DIM];
        std::memset(index, 0xFF, sizeof index);
        auto& cells = walk.pairs[which];
        // Laid only one way, what is down where it lands (landing_on); laid
        // under, its own cells go down as they land, on a copy of its own.
        optional<arc::cells> under;
        const arc::cells* down = only ? landing_on(*only, which) : nullptr;
        if (only && only->lay == rule::laid_under) {
            under.emplace(*down);
            down = &*under;
        }
        bool holds = true;
        int could_gain = -1, sure_wrong = 0;
        const bool drew = draw(which, cut, [&](int row, int col, int8_t color) {
            if (!holds) return false;
            int16_t& at = index[row][col];
            const bool fresh = at < 0;
            if (fresh) {
                at = (int16_t)cells.size();
                cells.push_back({ (int8_t)row, (int8_t)col, color, color, false });
            } else {
                paint_walk::cell& cell = cells[(size_t)at];
                cell.torn = cell.torn || cell.last != color;
                cell.last = color;
            }
            if (!only || (down && !lands(only->lay, (*down)[row][col]))) return holds;
            // Under what is down, its own cell is down now: something, whatever its color.
            if (under) (*under)[row][col] = is_color(color) ? color : (int8_t)0;
            if (exact) {
                if (only->in == laying::as_drawn) holds = comes_out(pair, row, col, color);
                else {
                    const int8_t shows = (int8_t)output.data[row][col];
                    if (first) first_shown = shows;
                    first = false;
                    holds = is_color(first_shown) && shows == first_shown;
                }
            } else if (fresh && only->in == laying::as_drawn && !comes_out(pair, row, col, color)) {
                if (could_gain < 0) {
                    could_gain = 0;
                    const arc::cells& before = (*only->canvas)[which];
                    for (int r = 0; r < output.height; ++r)
                        for (int c = 0; c < output.width; ++c)
                            could_gain += !comes_out(pair, r, c, before[r][c]);
                }
                holds = ++sure_wrong < could_gain;
            }
            return holds;
        });
        if (!drew || !holds) return nullopt;
    }
    return walk;
}

// A PAINT LAID ON WHAT IS DOWN, replayed: what every replay of a rule that
// draws asks, once -- its walk (walk_paint), laid the one way `how` says
// (lay_paint), done with as soon as it cannot hold so. Laid in its own colors
// or else in one, it is walked once and laid both ways: in its own colors where
// they lay no cell wrong, else whichever way says more, right cells less wrong
// ones, its own colors on a tie. Handed the pieces its pick takes
// (handed_pieces), made ready once for every way the rule is asked. Laid on
// the canvas `how` names, or else on the hand `facts` holds, where it lies
// (pieces_down): the acts replayed as paints -- recolors, mirrored copies, a
// color traded with empty -- change that hand itself.
template <class Draw>
inline optional<replayed> paint_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      laying how, Draw draw) {
    ++inference_work.paint_replays;
    if (!how.canvas) how.canvas = &facts.pieces_down();
    if (how.in == laying::own_else_one) {
        const auto walk = walk_paint(facts, handed, draw);
        if (!walk) return nullopt;
        auto own = lay_paint(facts, *walk, { how.lay, how.canvas });
        if (own && own->wrong_cells() == 0) return own;
        auto one = lay_paint(facts, *walk, { how.lay, how.canvas, laying::in_one });
        if (one && (!own || one->right_cells() - one->wrong_cells() > own->right_cells() - own->wrong_cells()))
            return one;
        return own;
    }
    const auto walk = walk_paint(facts, handed, draw, &how);
    return walk ? lay_paint(facts, *walk, how) : nullopt;
}

// The color a recolor's replay names for a piece the op leaves as it is.
constexpr int8_t LEFT_AS_IT_IS = -2;

// PICKED PIECES REPAINTED WHERE THEY LIE, replayed: what every recolor's replay
// asks, once. On each pair, `colors_of(which, cut, i, to)` fills `to`, the color
// each color picked piece i wears comes out in -- LEFT_AS_IT_IS where the rule
// leaves it as it is (ColorFrom's piece with no partner; ShapeColorMap's key the
// table does not name; a color no key pairs) -- or answers false where the rule
// cannot say; the piece is then a paint laid over itself in those colors
// (paint_holds), so a piece the rule leaves as it is has to come out as it
// was: the rule picked it, and does not get to leave it to another (4f537728's
// blocks in the marker's column, left by "takes the color of a piece in its
// rows", came out recolored). Sees the cells of the pieces it repainted and
// how many pieces changed color.
// Cell by cell: `cells_of(which, cut, i, repaint)` hands each cell of picked
// piece i to repaint(row, col, was, now), or answers false where the rule
// cannot say.
template <class CellsOf>
inline optional<replayed> cells_repainted_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                               CellsOf cells_of) {
    int pieces = 0;
    auto seen = paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool changed = false;
            if (!cells_of(which, cut, i, [&](int row, int col, int8_t was, int8_t now) {
                    put(row, col, now);
                    changed = changed || was != now;
                }))
                return false;
            pieces += changed;
        }
        return true;
    });
    if (seen) seen->pieces = pieces;
    return seen;
}
template <class ColorsOf>
inline optional<replayed> repaints_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                        ColorsOf colors_of) {
    return cells_repainted_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, auto&& repaint) {
        int8_t to[COLOR_COUNT];
        ranges::fill(to, LEFT_AS_IT_IS);
        if (!colors_of(which, cut, i, to)) return false;
        cut.pieces[i].each_cell([&](int row, int col, int8_t was) {
            repaint(row, col, was, to[was] == LEFT_AS_IT_IS ? was : to[was]);
        });
        return true;
    });
}

// A RECOLOR OF WHOLE PIECES, replayed: `color_of(which, cut, i, color)` says the
// one color picked piece i comes out in, every cell of it, or LEFT_AS_IT_IS
// (repaints_hold).
template <class ColorOf>
inline optional<replayed> recolors_hold(const replay_facts& facts, const vector<partner_pieces>& handed,
                                        ColorOf color_of) {
    return repaints_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, int8_t* to) {
        int8_t color = -1;
        if (!color_of(which, cut, i, color)) return false;
        std::fill(to, to + COLOR_COUNT, color);
        return true;
    });
}

// COLORS THAT TRADE PLACES, replayed as BidiColorMap trades them (swaps_of): a
// fixed pair, or past those the pairs a key drawn in the grid names -- every
// piece not picked, read along its rows or its columns (key_swaps) -- every cell
// of a picked piece in a color the pair or the key names coming out in its
// partner's (repaints_hold). A key that names no swap leaves them as they were,
// so that pair shows the rule at no work.
inline optional<replayed> key_swap_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                         int value) {
    vector<array<int8_t, COLOR_COUNT>> swaps(handed.size());
    for (size_t which = 0; which < handed.size(); ++which) swaps_of(value, handed[which], swaps[which].data());
    return repaints_hold(facts, handed, [&](size_t which, const partner_pieces&, int, int8_t* to) {
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (swaps[which][color] >= 0) to[color] = swaps[which][color];
        return true;
    });
}

// A COLOR TRADING PLACES WITH EMPTY, replayed: BidiColorMap past the key's two
// over every cell of each picked piece's box (traded_with_empty, the op's own
// rule) -- that color's cells left empty, the box's empty cells that color --
// the box laid over what is down as it comes out (paint_holds).
inline optional<replayed> empty_swap_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                           int color) {
    return paint_holds(facts, handed, {}, [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            for (int row = 0; row < piece.height; ++row)
                for (int col = 0; col < piece.width; ++col)
                    put(piece.row + row, piece.col + col,
                        traded_with_empty((int8_t)piece.data[row][col], color));
        }
        return true;
    });
}

// A PIECE THAT TAKES ITS COLOR FROM ANOTHER, replayed. ColorFrom paints every
// picked piece in the color of its partner among the rest, and leaves one with
// no partner as it is (recolors_hold: b7256dcd's shape with no dot at its side
// stays 6). A piece whose partners disagree the op leaves too, but the rule
// cannot say it: which color the puzzle means there, no pair shows (d23f8c26's
// middle column, left by "takes the color of a piece in its columns", read as
// a rule that kept it, and was wrong on its test).
inline optional<replayed> color_from_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                           partner_by how) {
    return recolors_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, int8_t& color) {
        const int worn = cut.color_from(how, i);
        color = worn >= 0 ? (int8_t)worn : LEFT_AS_IT_IS;
        return worn >= 0 || !cut.partnered(how, i);
    });
}
static_assert(row_of(node_type::ColorFrom).high == COLORS_FROM_ITS_SHAPE + SHAPE_LIKENESSES - 1,
              "ColorFrom takes one value for each way partner_by finds a partner, then the key by place, "
              "then each likeness of shape cell by cell");

// A PIECE COLORED CELL BY CELL BY ANOTHER OF ITS SHAPE, replayed: ColorFrom
// past the key by place paints each cell of every picked piece in the color of
// the cell of its one partner of its shape it lies on (colors_from_its_shape,
// the walk the op takes); a piece with no partner of its shape is left as it
// is, one with several the rule cannot say.
inline optional<replayed> colors_from_its_shape_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                                      int like) {
    const partner_by way = (partner_by)((int)partner_by::same_shape + like);
    return cells_repainted_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, auto&& repaint) {
        const bool laid = colors_from_its_shape(like, i, cut, [&](int row, int col, int color) {
            repaint(row, col, (int8_t)cut.color_at(i, row, col), is_color(color) ? (int8_t)color
                                                                                 : (int8_t)cut.color_at(i, row, col));
        });
        if (laid) return true;
        if (cut.partnered(way, i)) return false;
        cut.pieces[i].each_cell([&](int row, int col, int8_t was) { repaint(row, col, was, was); });
        return true;
    });
}

// A PIECE THAT TAKES ITS COLOR OFF A KEY BY ITS PLACE, replayed: ColorFrom past
// its partner ways reads the key on each grid -- every piece not picked -- and
// paints the n-th picked piece in reading order in the key's n-th entry
// (key_place_colors: 33b52de3's shapes, its key cell by cell). Where the key
// does not say, every piece is left as it is, so that pair shows the rule at no
// work (recolors_hold).
inline optional<replayed> key_place_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    vector<vector<int8_t>> places;
    for (const partner_pieces& cut : handed) places.push_back(cut.key_places());
    return recolors_hold(facts, handed, [&](size_t which, const partner_pieces&, int i, int8_t& color) {
        const int8_t worn = places[which][(size_t)i];
        color = worn >= 0 ? worn : LEFT_AS_IT_IS;
        return true;
    });
}

// A PIECE THAT TAKES THE KEY'S COLOR, replayed: ColorFrom [the key] paints
// every picked piece in the one color the key wears -- every piece not picked
// (key_color) -- and, where the key wears several, leaves every one as it is,
// so that pair shows the rule at no work (recolors_hold).
inline optional<replayed> key_color_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return recolors_hold(facts, handed, [&](size_t, const partner_pieces& cut, int, int8_t& color) {
        const int worn = key_color(cut);
        color = worn >= 0 ? (int8_t)worn : LEFT_AS_IT_IS;
        return true;
    });
}

// A RECOLOR A PICK DECIDES, replayed: every piece picked comes out wholly in
// `color` (recolors_hold).
inline optional<replayed> picked_recolor_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                               int8_t color) {
    return recolors_hold(facts, handed, [&](size_t, const partner_pieces&, int, int8_t& to) {
        to = color;
        return true;
    });
}

// ONE PAIR'S HAND AS SizeRankColorMap'S SENTENCE RANKS IT (beats_of): where the
// cut takes every cell of one color as one piece, each split into its connected
// pieces first (Split [0]), then biggest first, ties keeping that order
// (SortBySize [0], size_order). The n-th piece takes the palette's n-th color.
inline vector<object> ranked_by_size(const vector<object>& hand, const grid& input, segmentation_methods cut) {
    vector<object> ranked;
    if (cut != segmentation_methods::all_cells_of_one_color) ranked = hand;
    else
        for (const object& piece : hand) {
            grid layer(input.width, input.height);
            paint_piece(layer, piece, piece.row, piece.col, true, /*clip=*/true);
            object_walk_scratch scratch;
            each_object(layer, grouping::touching_any_color, scratch,
                        [&](const object_summary& box, const int16_t* cells, int count) {
                ranked.push_back(object::from_cells(layer, box, cells, count));
            });
        }
    vector<int> order(ranked.size());
    size_order((int)ranked.size(), true, [&](int i) { return (int)ranked[(size_t)i].cell_count(); }, order.data());
    vector<object> sorted;
    for (const int i : order) sorted.push_back(ranked[(size_t)i]);
    return sorted;
}

// SIZE RANK, replayed. SizeRankColorMap takes the pieces as the line hands them
// over, ranked (ranked_by_size), and paints the first the output's most common
// color, the next the next, and the rest the ground: a recolor of every piece
// (recolors_hold).
inline optional<replayed> size_rank_holds(const replay_facts& facts, const color_facts& palette,
                                          segmentation_methods cut) {
    if (palette.output_colors_by_size_count <= 0) return nullopt;
    replay_facts ranked = facts;
    for (size_t which = 0; which < ranked.cuts().size(); ++which) {
        vector<object> pieces = ranked_by_size(ranked.cuts()[which], ranked.train[which].input, cut);
        ranked.change(which).hand = std::move(pieces);
    }
    return recolors_hold(ranked, handed_pieces(ranked, piece_pick::all()),
                         [&](size_t, const partner_pieces&, int i, int8_t& color) {
        color = i < palette.output_colors_by_size_count ? palette.output_colors_by_size[i] : (int8_t)TRANSPARENT;
        return true;
    });
}

// PICKED PIECES SLID, replayed: each by its own `down` and `across`, and the
// hand then drawn as the line lays it (draw_the_hand), every piece landing on
// the output. A moved piece's cell lands right where it shows -- in the
// drawing, not covered by a piece the line lays over it, and coming out as the
// output (comes_out) -- and leaves its cell right where nothing else now
// covers it and its color does not come out there -- a cell still showing it
// was a copy left behind, not a move; what does show there is another rule's
// to say (18286ef8's box closing behind its 9). Each cell otherwise is wrong.
// A right cell is gained where the drawing changed there and now comes out
// (brings_nearer, which its caller asks, with one piece at least moved). Adds
// the cells each one left and the cells it landed on, right, and the ones
// wrong to the pair's lists, and counts the pieces that moved.
inline bool slides_show(const figure_pair& pair, const partner_pieces& cut, bool under,
                        const vector<int>& down, const vector<int>& across, pair_tally& tally,
                        vector<std::pair<int, int>>& left, vector<std::pair<int, int>>& landed,
                        vector<cell_color>& wrong, int& pieces_moved) {
    grid drawn(pair.whole.output.width, pair.whole.output.height), was = drawn;
    if (!draw_the_hand(drawn, cut.pieces, under, /*clip=*/false,
                       [&](size_t i) { return direction_step{ down[i], across[i] }; }))
        return false;
    draw_the_hand(was, cut.pieces, under, /*clip=*/true, [](size_t) { return direction_step{ 0, 0 }; });
    auto gains = [&](int row, int col) {
        return drawn.data[row][col] != was.data[row][col] && comes_out(pair, row, col, drawn.data[row][col]);
    };
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i] || (down[i] == 0 && across[i] == 0)) continue;
        cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
            const int to_row = row + down[i], to_col = col + across[i];
            if (drawn.data[to_row][to_col] == color && comes_out(pair, to_row, to_col, color)) {
                landed.push_back({ to_row, to_col });
                tally.gained += gains(to_row, to_col);
            } else {
                ++tally.wrong;
                wrong.push_back({ (int8_t)to_row, (int8_t)to_col, (int8_t)drawn.data[to_row][to_col] });
            }
            if (drawn.data[row][col] != TRANSPARENT || !comes_out(pair, row, col, color)) {
                left.push_back({ row, col });
                tally.gained += gains(row, col);
            } else {
                ++tally.wrong;
                wrong.push_back({ (int8_t)row, (int8_t)col, (int8_t)TRANSPARENT });
            }
        });
        ++pieces_moved;
    }
    return true;
}

// PICKED PIECES SLID, EACH ITS OWN WAY, replayed: what every move rule's replay
// asks, once. On each pair, `where(which, cut, i, by)` says how far picked piece
// i goes, down and across, or false where the rule cannot say; the pieces then
// slide as slides_show reads them, and the rule holds only where one piece at
// least moved and it brings the pair nearer (brings_nearer). Sees the cells each
// one left and the cells it landed on, right and wrong, and how many pieces
// moved. Handed the pieces its pick takes (handed_pieces).
template <class Where>
inline optional<replayed> slides_hold(const replay_facts& facts, const vector<partner_pieces>& handed, Where where) {
    ++inference_work.slide_replays;
    if (!facts.overlays) return nullopt;
    replayed seen;
    seen.left.assign(facts.train.size(), {});
    seen.drawn.assign(facts.train.size(), {});
    seen.wrong.assign(facts.train.size(), {});
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const int before = seen.pieces;
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        vector<int> downs(cut.pieces.size(), 0), acrosses(cut.pieces.size(), 0);
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            direction_step by{ 0, 0 };
            if (!where(which, cut, i, by)) return nullopt;
            downs[i] = by.down;
            acrosses[i] = by.across;
        }
        pair_tally tally;
        if (!slides_show(facts.pairs[which], cut, facts.laid_under, downs, acrosses, tally, seen.left[which],
                         seen.drawn[which], seen.wrong[which], seen.pieces))
            return nullopt;
        tally.acted = seen.pieces > before;
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    if (seen.pieces == 0) return nullopt;
    return seen;
}

// WHERE A MOVE TAKES PICKED PIECE i of pair `which`, the move `kind` said with
// its op's own `value`, as the op slides it: down or across alone until its box
// starts on the row its one partner's box starts on -- or, for a value past the
// partner ways, the column (AlignTo, goes_level); toward its one partner of the
// kind `value` % PARTNER_KINDS names -- each way partner_by finds one, then each
// color -- one step, its own size, or until it touches it (MoveToward,
// goes_toward, move_toward in 15_op_math.hpp); that way until its box meets
// the grid's edge (MoveToEdge, steps_to_edge); or its own size that way
// (Shift, shift_by). How far down and across in `by`, or false for a piece with
// no one partner, which its op leaves where it is.
inline bool goes_by(const replay_facts& facts, size_t which, const partner_pieces& cut, int i, move_by kind,
                    int value, direction_step& by) {
    constexpr int ways = (int)partner_by::count;
    const grid& output = facts.train[which].output;
    switch (kind) {
    case move_by::align_to: return cut.goes_level(i, value >= ways, (partner_by)(value % ways), by);
    case move_by::toward: return cut.goes_toward(i, value, output.width, output.height, by);
    case move_by::to_edge: {
        const direction_step step = step_of(value);
        const int steps = steps_to_edge(cut.boxes[i], output.width, output.height, value);
        by = { steps * step.down, steps * step.across };
        return true;
    }
    case move_by::shift: by = shift_by(value, cut.boxes[i].width(), cut.boxes[i].height()); return true;
    case move_by::count: break;
    }
    return false;
}
static_assert(row_of(node_type::AlignTo).high + 1 == 2 * (int)partner_by::count,
              "AlignTo takes rows then columns, each by every way partner_by finds a partner");
static_assert(row_of(node_type::MoveToward).high + 1 == (int)toward_by::count * PARTNER_KINDS,
              "MoveToward takes each way of going toward each kind of partner");

// THE VALUES A MOVE IS ASKED WITH, each handed to `visit(value)`, in the order a
// tie goes: AlignTo's rows, then its columns, of each partner way the pairs ask
// (asks_way); MoveToward's each way of going toward each kind of partner the
// pairs ask (asks_kind); MoveToEdge's and Shift's eight directions, the sides
// before the corners (direction_asked).
template <class Visit>
inline void each_move_value(const replay_facts& facts, move_by kind, Visit visit) {
    constexpr int ways = (int)partner_by::count;
    switch (kind) {
    case move_by::align_to:
        for (int value = 0; value < 2 * ways; ++value)
            if (facts.asks_way((partner_by)(value % ways))) visit(value);
        break;
    case move_by::toward:
        for (int value = 0; value <= row_of(node_type::MoveToward).high; ++value)
            if (facts.asks_kind(value % PARTNER_KINDS)) visit(value);
        break;
    case move_by::to_edge:
    case move_by::shift:
        for (int at = 0; at < direction::count; ++at) visit(direction_asked(at));
        break;
    case move_by::count: break;
    }
}

// A PIECE THAT MOVES, replayed: every picked piece slid where the move `kind`
// with `value` takes it (goes_by).
inline optional<replayed> moves_hold(const replay_facts& facts, const vector<partner_pieces>& handed, move_by kind,
                                     int value) {
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, direction_step& by) {
        return goes_by(facts, which, cut, i, kind, value, by);
    });
}

// A PICKED PIECE SLID THE SAME WAY IN EVERY PAIR, replayed: `down` and `across`,
// the MoveColumn and MoveRow a pick by holes or by size is said with.
inline optional<replayed> move_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int down,
                                     int across) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces&, int, direction_step& by) {
        by = { down, across };
        return true;
    });
}

// Where the first piece a pick takes could have gone in the first pair: every
// offset at which the output shows it whole, it not standing still. Moves are
// read off the output, not guessed, and every one is then replayed on every pair.
inline vector<std::pair<int, int>> offsets_to_ask(const replay_facts& facts, const vector<partner_pieces>& handed) {
    vector<std::pair<int, int>> offsets;
    if (facts.train.empty() || !facts.overlays) return offsets;
    const example& pair = facts.train[0];
    const partner_pieces& cut = handed[0];
    int first = -1;
    for (int i = 0; i < cut.count() && first < 0; ++i)
        if (cut.picked[i]) first = i;
    if (first < 0) return offsets;
    for (int down = -pair.output.height + 1; down < pair.output.height; ++down)
        for (int across = -pair.output.width + 1; across < pair.output.width; ++across) {
            if (down == 0 && across == 0) continue;
            bool shows = true;
            cut.pieces[first].each_cell([&](int row, int col, int8_t color) {
                shows = shows && pair.output.contains(row + down, col + across) &&
                        pair.output.data[row + down][col + across] == color;
            });
            if (shows) offsets.push_back({ down, across });
        }
    return offsets;
}

inline optional<int8_t> paint_at(const colors_by_pair& painted, size_t which, int row, int col) {
    if (which < painted.size())
        for (auto cell = painted[which].rbegin(); cell != painted[which].rend(); ++cell)
            if (cell->row == row && cell->col == col) return cell->color;
    return nullopt;
}

template <class Keeps>
inline grid hand_after_drops(const replay_facts& facts, size_t which, Keeps keeps) {
    const auto size = facts.canvas(which);
    grid drawn(size.width, size.height);
    const auto& pieces = facts.cuts()[which];
    for (size_t index = 0; index < pieces.size(); ++index) {
        object piece = pieces[index];
        for (int row = 0; row < piece.height; ++row)
            for (int col = 0; col < piece.width; ++col)
                if (!keeps(index, row + piece.row, col + piece.col))
                    piece.data[row][col] = (int8_t)TRANSPARENT;
        paint_piece(drawn, piece, piece.row, piece.col, !facts.laid_under, true);
    }
    return drawn;
}

// Keep pieces contributing an answer cell. The rest can go only where those
// contributors, followed by the separate paint, leave the required picture.
inline vector<vector<bool>> pieces_to_remove(const replay_facts& facts, const colors_by_pair& painted) {
    if (!facts.hand_known || !facts.overlays) return {};
    vector<vector<bool>> removed;
    for (size_t which = 0; which < facts.cuts().size(); ++which) {
        const auto& pieces = facts.cuts()[which];
        const figure_pair& pair = facts.pairs[which];
        const grid& output = pair.whole.output;
        auto& one = removed.emplace_back(pieces.size(), true);
        for (size_t index = 0; index < pieces.size(); ++index)
            pieces[index].each_cell([&](int row, int col, int8_t color) {
                if (!output.contains(row, col) ||
                    (!paint_at(painted, which, row, col) && comes_out(pair, row, col, color))) one[index] = false;
            });
        const grid kept = hand_after_drops(facts, which, [&](size_t index, int, int) { return !one[index]; });
        for (size_t index = 0; index < pieces.size(); ++index) {
            if (!one[index]) continue;
            pieces[index].each_cell([&](int row, int col, int8_t) {
                const auto above = paint_at(painted, which, row, col);
                if (!output.contains(row, col) || !comes_out(pair, row, col, above ? *above : kept.data[row][col]))
                    one[index] = false;
            });
        }
    }
    return removed;
}


// THE PIECES THE OUTPUT SHOWS IN OTHER COLORS WHERE THEY LIE, pair by pair: every
// cell of the piece on its output and painted there, one at least in another
// color than the piece's. None where the pairs do not lie over their outputs.
inline vector<vector<bool>> pieces_recolored(const replay_facts& facts) {
    vector<vector<bool>> recolored;
    if (!facts.overlays) return recolored;
    for (size_t which = 0; which < facts.cuts().size(); ++which) {
        const auto& pieces = facts.cuts()[which];
        const grid& output = facts.train[which].output;
        auto& one = recolored.emplace_back(pieces.size(), false);
        for (size_t index = 0; index < pieces.size(); ++index) {
            bool shown = true, other = false;
            pieces[index].each_cell([&](int row, int col, int8_t color) {
                const bool painted = output.contains(row, col) && is_color(output.data[row][col]);
                shown = shown && painted;
                other = other || (painted && output.data[row][col] != color);
            });
            one[index] = shown && other;
        }
    }
    return recolored;
}

// THE RECOLORED PIECES A FIND COLORS AS THE OUTPUT SHOWS THEM, pair by pair:
// ColorFrom's `how` with the recolored pieces picked -- its partner's color
// among the pieces the output does not recolor, the key's color at its place,
// the key's one color, or its colors cell by cell from the one piece of its
// shape -- every cell of the piece as the output shows it. What the op does,
// read off the pairs rather than replayed for every pick.
inline vector<vector<bool>> colored_right(const replay_facts& facts, const vector<vector<bool>>& recolored, int how) {
    vector<vector<bool>> right;
    for (size_t which = 0; which < recolored.size(); ++which) {
        const grid& output = facts.train[which].output;
        const partner_pieces cut(facts.cuts()[which], recolored[which], facts.answers_of(which));
        const vector<int8_t> places = how == KEY_BY_PLACE ? cut.key_places() : vector<int8_t>{};
        const int keyed = how == KEY_COLOR ? key_color(cut) : -1;
        auto shows = [&](int row, int col, int color) { return output.contains(row, col) && output.data[row][col] == color; };
        auto& one = right.emplace_back(cut.pieces.size(), false);
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool all = true;
            if (how >= COLORS_FROM_ITS_SHAPE) {
                const bool laid = colors_from_its_shape(how - COLORS_FROM_ITS_SHAPE, i, cut, [&](int row, int col, int color) {
                    all = all && shows(row, col, is_color(color) ? color : cut.color_at(i, row, col));
                });
                one[(size_t)i] = laid && all;
                continue;
            }
            const int color = how == KEY_BY_PLACE ? places[(size_t)i] : how == KEY_COLOR ? keyed
                            : cut.color_from((partner_by)how, i);
            if (color < 0) continue;
            cut.pieces[i].each_cell([&](int row, int col, int8_t) { all = all && shows(row, col, color); });
            one[(size_t)i] = all;
        }
    }
    return right;
}

// PICKED PIECES THAT LOSE CELLS WHERE THEY LIE, replayed: what the removal's
// and the outline's replays ask, once. Each picked piece keeps the cells
// `keeps(cut, i, row, col)` names and loses the rest. A kept cell is right
// where it comes out in the piece's own color (comes_out), and a lost one
// where what is left there comes out: nothing, where the plan lays what the
// output shows, or a piece still there that covers it -- one not picked, or a
// picked one that keeps that cell. Each cell otherwise is wrong. A lost cell
// right is gained where the hand as it was did not come out there, and in each
// pair one piece at least lost such a cell, and the rule brings the pair
// nearer (brings_nearer). Sees the cells the picked pieces keep (drawn) and
// lose (left), right, the ones wrong, and how many pieces lost a cell that
// shows it.
template <class Keeps>
inline optional<replayed> drops_hold(const replay_facts& facts, const vector<partner_pieces>& handed, Keeps keeps,
                                      const colors_by_pair& painted = {}) {
    ++inference_work.drop_replays;
    if (!facts.overlays || facts.train.empty()) return nullopt;
    replayed seen;
    seen.drawn.assign(facts.train.size(), {});
    seen.left.assign(facts.train.size(), {});
    seen.wrong.assign(facts.train.size(), {});
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const figure_pair& pair = facts.pairs[which];
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        const grid before_picture = hand_after_drops(facts, which, [](size_t, int, int) { return true; });
        const grid after_picture = hand_after_drops(facts, which, [&](size_t index, int row, int col) {
            return !cut.picked[index] || keeps(cut, (int)index, row, col);
        });
        const int before = seen.pieces;
        pair_tally tally;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool changed = false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                if (!pair.whole.output.contains(row, col)) { ++tally.wrong; return; }
                if (const auto above = paint_at(painted, which, row, col)) {
                    if (!keeps(cut, i, row, col) && comes_out(pair, row, col, *above))
                        seen.left[which].push_back({ row, col });
                    return;
                }
                if (keeps(cut, i, row, col)) {
                    if (comes_out(pair, row, col, color)) seen.drawn[which].push_back({ row, col });
                    else {
                        ++tally.wrong;
                        seen.wrong[which].push_back({ (int8_t)row, (int8_t)col, color });
                    }
                    return;
                }
                if (comes_out(pair, row, col, after_picture.data[row][col])) {
                    seen.left[which].push_back({ row, col });
                    const bool gained = !comes_out(pair, row, col, before_picture.data[row][col]);
                    tally.gained += gained;
                    changed = changed || gained;
                } else {
                    ++tally.wrong;
                    seen.wrong[which].push_back({ (int8_t)row, (int8_t)col, (int8_t)TRANSPARENT });
                }
            });
            seen.pieces += changed;
        }
        tally.acted = seen.pieces > before;
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    if (seen.pieces == 0) return nullopt;
    return seen;
}

// PICKED PIECES REMOVED, replayed. DeleteSelection takes the pieces a pick takes
// out of the hand, so nothing paints them: they keep no cell (drops_hold), and
// the cells they sat on are left.
inline optional<replayed> removal_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       const colors_by_pair& painted = {}) {
    return drops_hold(facts, handed, [](const partner_pieces&, int, int, int) { return false; }, painted);
}

// A PIECE HOLLOWED TO ITS EDGE, as Outline leaves it: the cells that have, among
// their eight neighbours, a cell that is not the piece's (on_object_edge, the
// test the op takes), the rest dropped.
inline object outlined(const object& piece) {
    object rim = piece;
    auto occupied = [&](int row, int col) {
        return row >= 0 && col >= 0 && row < piece.height && col < piece.width &&
               piece.data[row][col] != (int8_t)TRANSPARENT;
    };
    for (int row = 0; row < piece.height; ++row)
        for (int col = 0; col < piece.width; ++col) {
            if (!occupied(row, col)) continue;
            if (!on_object_edge(row, col, occupied)) rim.data[row][col] = (int8_t)TRANSPARENT;
        }
    return rim;
}

// A PIECE HOLLOWED TO ITS EDGE, replayed. Outline keeps the cells of each picked
// piece that have, among their eight neighbours, a cell that is not the piece's
// (on_object_edge, the test the op takes), and drops the rest (drops_hold).
inline optional<replayed> outline_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return drops_hold(facts, handed, [](const partner_pieces& cut, int i, int row, int col) {
        const object& piece = cut.pieces[i];
        return on_object_edge(row - piece.row, col - piece.col, [&](int down, int across) {
            return down >= 0 && across >= 0 && down < piece.height && across < piece.width &&
                   piece.data[down][across] != (int8_t)TRANSPARENT;
        });
    });
}

// THE HAND, POURED, as FitInto [poured] pours its objects: every pour each_pour
// finds on `hand` with `picked` marked -- that color's cells gone from the
// picked pieces, a piece left with none gone with them, and each gap one piece
// in that color at the end. False, the hand as it was, where nothing pours.
inline bool pour_the_hand(vector<object>& hand, const vector<bool>& picked, hand_answers& answers) {
    vector<object> gaps;
    uint16_t poured = 0;
    each_pour(partner_pieces(hand, picked, answers), [&](int color, const int16_t* rows, const int16_t* cols, int count) {
        cell_box box;
        for (int k = 0; k < count; ++k) box.include_cell(rows[k], cols[k]);
        object& gap = gaps.emplace_back(object{ (int16_t)box.col0, (int16_t)box.row0, (int16_t)box.width(), (int16_t)box.height() });
        for (int k = 0; k < count; ++k) gap.data[rows[k] - box.row0][cols[k] - box.col0] = (int8_t)color;
        poured = (uint16_t)(poured | 1u << color);
    });
    if (gaps.empty()) return false;
    vector<object> kept;
    for (size_t i = 0; i < hand.size(); ++i) {
        object piece = hand[i];
        int lost = 0;
        if (picked[i])
            for (int row = 0; row < piece.height; ++row)
                for (int col = 0; col < piece.width; ++col)
                    if (int8_t& cell = piece.data[row][col]; is_color(cell) && (poured >> cell & 1)) {
                        cell = (int8_t)TRANSPARENT;
                        ++lost;
                    }
        if (!lost || piece.cell_count() > 0) kept.push_back(piece);
    }
    for (const object& gap : gaps)
        if (kept.size() < (size_t)MAX_OBJECTS_IN_A_LIST) kept.push_back(gap);
    hand = std::move(kept);
    return true;
}

// THE HAND, ITS PIECES MOVED INTO THEIR SOCKETS, as FitInto [moved] moves its
// objects (each_fit_move): each piece that goes, its painted cells turned as
// the place names and laid with their corner there. False, the hand as it
// was, where none goes.
inline bool move_the_hand_in(vector<object>& hand, const vector<bool>& picked, hand_answers& answers) {
    vector<std::array<int, 4>> moves;
    each_fit_move(partner_pieces(hand, picked, answers),
                  [&](int i, int pose, int row, int col) { moves.push_back({ i, pose, row, col }); });
    for (const auto& [i, pose, row, col] : moves) {
        object& piece = hand[(size_t)i];
        const cell_box paint = piece.painted_box();
        object tight{ (int16_t)paint.col0, (int16_t)paint.row0, (int16_t)paint.width(), (int16_t)paint.height() };
        piece.each_cell([&](int at_row, int at_col, int8_t color) {
            tight.data[at_row - paint.row0][at_col - paint.col0] = color;
        });
        piece = posed(tight, pose);
        piece.row = (int16_t)row;
        piece.col = (int16_t)col;
    }
    return !moves.empty();
}

// THE HAND PUT INTO ITS SOCKETS as FitInto puts its objects, the way `way`
// names: poured (pour_the_hand) or moved (move_the_hand_in).
inline bool into_sockets(vector<object>& hand, const vector<bool>& picked, hand_answers& answers, int way) {
    return way == (int)fit_into_way::moved ? move_the_hand_in(hand, picked, answers)
                                           : pour_the_hand(hand, picked, answers);
}

// THE HAND, EACH PICKED PIECE BECOME ITS PARTNER MIRRORED, as ShapeFrom makes
// it (each_shape_taken, the partners found the way `how` names): its cells the
// ones handed for it, every partner read off the hand as it was. False, the
// hand as it was, where no piece takes a shape.
inline bool take_partner_shapes(vector<object>& hand, const vector<bool>& picked, hand_answers& answers, int how) {
    vector<object> taken = hand;
    grid drawn(MAX_GRID_DIM, MAX_GRID_DIM);
    vector<int16_t> at;
    int taking = -1;
    cell_box box;
    auto set_drawing = [&] {
        if (taking >= 0)
            taken[(size_t)taking] = object::from_cells(drawn, summary_of(box, (int)at.size()), at.data(), (int)at.size());
    };
    each_shape_taken(how, partner_pieces(hand, picked, answers), [&](int i, int row, int col, int color) {
        if (i != taking) {
            set_drawing();
            taking = i;
            at.clear();
            box = {};
        }
        drawn.data[row][col] = (int8_t)color;
        at.push_back((int16_t)cell_index(row, col));
        box.include_cell(row, col);
    });
    set_drawing();
    if (taking < 0) return false;
    hand = std::move(taken);
    return true;
}

// A CHANGE OF THE HAND, replayed: each pair's hand changed as the op changes
// it -- act(hand, picked, answers), answering whether it acted -- drawn before
// and after (draw_the_hand), and judged by the cells that changes there
// (brings_nearer, something done in every pair): a cell right where it now
// comes out (comes_out) -- a cell filled, a cell its piece left -- gained
// where it did not before, and wrong elsewhere. Sees the cells left and
// filled, and how many pairs it acted in.
template <class Act>
inline optional<replayed> hand_act_holds(const replay_facts& facts, const vector<partner_pieces>& handed, Act act) {
    if (!facts.overlays) return nullopt;
    replayed seen;
    seen.left.assign(facts.train.size(), {});
    seen.drawn.assign(facts.train.size(), {});
    seen.wrong.assign(facts.train.size(), {});
    auto still = [](size_t) { return direction_step{ 0, 0 }; };
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const partner_pieces& cut = handed[which];
        seen.every_piece = seen.every_piece && cut.every_piece_picked();
        vector<object> hand = facts.cuts()[which];
        pair_tally tally;
        tally.acted = act(hand, cut.picked, facts.answers_of(which));
        const figure_pair& pair = facts.pairs[which];
        const grid& output = pair.whole.output;
        grid was(output.width, output.height), now = was;
        if (!draw_the_hand(was, facts.cuts()[which], facts.laid_under, /*clip=*/true, still) ||
            !draw_the_hand(now, hand, facts.laid_under, /*clip=*/true, still))
            return nullopt;
        for (int row = 0; row < output.height; ++row)
            for (int col = 0; col < output.width; ++col) {
                const int8_t shows = now.data[row][col];
                if (shows == was.data[row][col]) continue;
                if (comes_out(pair, row, col, shows)) {
                    (shows == (int8_t)TRANSPARENT ? seen.left : seen.drawn)[which].push_back({ row, col });
                    tally.gained += !comes_out(pair, row, col, was.data[row][col]);
                } else {
                    ++tally.wrong;
                    seen.wrong[which].push_back({ (int8_t)row, (int8_t)col, shows });
                }
            }
        seen.pieces += tally.acted;
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    return seen;
}

// PIECES PUT INTO THEIR SOCKETS, replayed: each pair's hand put in as FitInto
// puts it, the way `way` names (into_sockets), judged as every change of the
// hand is (hand_act_holds).
inline optional<replayed> fit_into_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int way) {
    return hand_act_holds(facts, handed, [&](vector<object>& hand, const vector<bool>& picked, hand_answers& answers) {
        return into_sockets(hand, picked, answers, way);
    });
}

// PIECES BECOME THEIR PARTNERS MIRRORED, replayed: each pair's hand as ShapeFrom
// leaves it, its partners found the way `how` names (take_partner_shapes),
// judged as every change of the hand is (hand_act_holds).
inline optional<replayed> shape_from_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int how) {
    return hand_act_holds(facts, handed, [&](vector<object>& hand, const vector<bool>& picked, hand_answers& answers) {
        return take_partner_shapes(hand, picked, answers, how);
    });
}

// A PIECE DRAWN AGAIN MIRRORED AROUND ANOTHER'S MIDDLE, replayed. MirrorAround
// draws every picked piece again mirrored around the middle of its one partner
// among the rest, the grid, or the picked pattern (each_mirror), laid over
// what is down (paint_holds). A pattern may be clipped by the grid; other
// copies must stay inside it. A row mirrored end to end onto
// itself is no work. Sees the cells the picked pieces sit on (left) and the
// cells their copies landed on, and how many copies it drew.
inline optional<replayed> mirror_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       mirror_by mirror, int how) {
    cells_by_pair kept(facts.train.size());
    int copies = 0;
    auto seen = paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& input = facts.train[which].input;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool on_grid = true;
            const bool drawn = each_mirror(mirror, how, i, cut, input.width, input.height,
                                           [&](int pose, int row2, int col2) {
                mirrored(cut.pieces[i], pose, row2, col2).each_cell([&](int row, int col, int8_t color) {
                    if (input.contains(row, col)) put(row, col, color);
                    else if (how != MIRROR_PATTERN) on_grid = false;
                });
                ++copies;
            });
            if (!drawn || !on_grid) return false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t) { kept[which].push_back({ row, col }); });
        }
        return true;
    });
    if (seen) {
        seen->left = std::move(kept);
        seen->pieces = copies;
    }
    return seen;
}
static_assert(row_of(node_type::MirrorAround).high + 1 == (int)mirror_by::count * MIRROR_PARTNERS,
              "MirrorAround takes each way of mirroring around each partner, the grid and the pattern");

// The cut's pieces grown in step along `track` on a canvas `width` by
// `height`, a tie taken by the piece lying furthest `side` (takes_the_tie):
// `owner` comes back holding each cell's piece and `turn` its turn
// (grow_in_step), and `met` hears each meeting.
template <class Met>
inline void grow_the_cut(const partner_pieces& cut, grown_along track, int side, int width, int height,
                         int16_t* owner, int16_t* turn, Met met) {
    // Each cell's piece, the first in the cut that holds it, as the op reads its list.
    int16_t queue[MAX_GRID_CELLS];
    uint8_t cell_directions[MAX_GRID_CELLS];
    std::fill_n(owner, width * height, (int16_t)-1);
    for (int i = 0; i < cut.count(); ++i)
        cut.pieces[i].each_cell([&](int row, int col, int8_t) {
            if (row >= 0 && col >= 0 && row < height && col < width && owner[row * width + col] < 0)
                owner[row * width + col] = (int16_t)i;
        });
    grow_in_step(width, height, track, [&](int piece) { return cut.is_picked(piece); },
                 [&](int one, int other) { return takes_the_tie(side, one, cut.boxes[one], other, cut.boxes[other]); },
                 owner, turn, queue, cell_directions, met);
}

// WHICH SIDE TAKES A TIE, read off the pairs, never tried: the cut's picked
// pieces grown in step along `track` as the list orders a tie, every cell two
// reached in one turn -- where the output gives it to one of the two, that one
// takes it -- and of the list's order and the four sides, the one every such
// cell agrees with: the list's first, else the one side alone; the list's
// where none or several do, so the replay judges.
inline int tie_side_read(const replay_facts& facts, const vector<partner_pieces>& handed, grown_along track) {
    struct tie { int taker, other; cell_box taker_box, other_box; };
    vector<tie> ties;
    for (size_t which = 0; which < handed.size(); ++which) {
        const partner_pieces& cut = handed[which];
        const grid& output = facts.train[which].output;
        int16_t owner[MAX_GRID_CELLS], turn[MAX_GRID_CELLS];
        vector<std::pair<int, int>> reached;
        grow_the_cut(cut, track, -1, output.width, output.height, owner, turn, [&](int at, int, int piece, bool in_one_turn) {
            if (in_one_turn) reached.push_back({ at, piece });
        });
        for (const auto& [at, lost] : reached) {
            const int kept = owner[at];
            if (kept < 0 || kept == lost) continue;
            const int shown = output.data[at / output.width][at % output.width];
            if (shown == cut.color(kept)) ties.push_back({ kept, lost, cut.boxes[kept], cut.boxes[lost] });
            else if (shown == cut.color(lost)) ties.push_back({ lost, kept, cut.boxes[lost], cut.boxes[kept] });
        }
    }
    auto agrees = [&](int side) {
        return ranges::all_of(ties, [&](const tie& one) {
            return takes_the_tie(side, one.taker, one.taker_box, one.other, one.other_box);
        });
    };
    if (agrees(-1)) return -1;
    int found = -1, sides = 0;
    for (const int side : { (int)direction::up, (int)direction::right, (int)direction::down, (int)direction::left })
        if (agrees(side)) {
            found = side;
            ++sides;
        }
    return sides == 1 ? found : -1;
}

// PIECES GROWN IN STEP UNTIL THEY MEET, walked: ExtendTo past its eight ways,
// every picked piece growing along its track at once (grow_in_step, the growth
// the op runs), a tie taken by the piece lying furthest `side`, each cell laid
// in the color of the piece that took it, and the bodies laid down again where
// they lie, as the op hands them back. Refused where a ray stops at a meeting
// but the output carries it on past (23581191).
inline optional<paint_walk> grown_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       grown_along track, int side) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int width = output.width, height = output.height;
        int16_t owner[MAX_GRID_CELLS], turn[MAX_GRID_CELLS];
        vector<array<int16_t, 3>> meetings;
        grow_the_cut(cut, track, side, width, height, owner, turn, [&](int at, int grow_direction, int piece, bool) {
            if (!grown_spreads(track)) meetings.push_back({ (int16_t)at, (int16_t)grow_direction, (int16_t)piece });
        });
        for (const auto& [at, grow_direction, piece] : meetings) {
            const direction_step step = step_of(grow_direction);
            const int row = at / width + step.down, col = at % width + step.across;
            if (row >= 0 && col >= 0 && row < height && col < width && turn[row * width + col] < 0 &&
                output.data[row][col] == cut.color(piece))
                return false;
        }
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            bool laid = true;
            cut.pieces[i].each_cell([&](int row, int col, int8_t body) {
                if (laid && output.contains(row, col)) laid = put(row, col, body);
            });
            if (!laid) return false;
        }
        for (int at = 0; at < width * height; ++at)
            if (turn[at] > 0 && !put(at / width, at % width, (int8_t)cut.color(owner[at]))) return false;
        return true;
    });
}

// A TRAIL GROWN OFF A PIECE, walked. ExtendTo marches from every cell of each
// picked piece that way (march_trail) and fills what it passes in the piece's
// color (walk_paint), laid every way the line asks (lay_paint). The op hands
// back the body with its trail, so the paint lays the body down again where it
// lies: the output must show it there too (66e6c45b's line removed a piece
// whose trail then drew it back). Past the eight ways, the pieces grow in step
// until they meet (grown_walk), a tie taken as the value's side says.
inline optional<paint_walk> trail_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int grow_direction) {
    if (grow_direction >= direction::count) {
        const auto [growth, side] = growth_of(grow_direction);
        return grown_walk(facts, handed, (grown_along)growth, side);
    }
    const direction_step step = step_of(grow_direction);
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        bool occupied[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        for (const object& piece : cut.pieces)
            piece.each_cell([&](int row, int col, int8_t) { occupied[row][col] = true; });
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const int8_t color = cut.pieces[i].first_color();
            cut.pieces[i].each_cell([&](int row, int col, int8_t body) {
                if (output.contains(row, col)) put(row, col, body);
                march_trail(row, col, step, output.width, output.height,
                            [&](int at_row, int at_col) { return cut.painted(i, at_row, at_col); },
                            [&](int at_row, int at_col) { return occupied[at_row][at_col]; },
                            [&](int at_row, int at_col) { put(at_row, at_col, color); });
            });
        }
        return true;
    });
}

// THE PICKED PIECES COPIED ONTO THE OTHERS, walked. CopyTo draws each picked
// piece again onto every piece not picked, lined up the way `anchor` numbers
// CopyTo's nine -- top, middle or bottom by left, middle or right,
// aligned_copy_start (15_op_math.hpp), the step the op takes -- each copied
// cell that lands on the grid (walk_paint), laid every way the line asks
// (lay_paint). Picking none, or every piece, or more copies than a list holds,
// it draws nothing.
inline optional<paint_walk> copy_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int anchor) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        int picked = 0;
        for (const bool one : cut.picked) picked += one;
        if (picked == 0 || picked == cut.count() || picked * (cut.count() - picked) > MAX_OBJECTS_IN_A_LIST)
            return false;
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& pattern = cut.pieces[i];
            for (int j = 0; j < cut.count(); ++j) {
                if (cut.picked[j]) continue;
                const object& target = cut.pieces[j];
                const int row0 = aligned_copy_start(pattern.height, target.row, target.height, anchor / 3);
                const int col0 = aligned_copy_start(pattern.width, target.col, target.width, anchor % 3);
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    const int at_row = row0 + row - pattern.row, at_col = col0 + col - pattern.col;
                    if (output.contains(at_row, at_col)) put(at_row, at_col, color);
                });
            }
        }
        return true;
    });
}

// THE PICKED PIECES COPIED WHERE A MOVE WOULD SLIDE THEM, walked: each drawn
// again where the move `kind` with `value` takes it (goes_by), in its own
// colors, each cell that lands on the grid (walk_paint) -- its op's sentence
// laying the moved pieces over what is down, the pieces themselves down
// already. Nothing where a piece has no one partner.
inline optional<paint_walk> moved_copy_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                            move_by kind, int value) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        for (int i = 0; i < cut.count(); ++i) {
            direction_step by{ 0, 0 };
            if (!cut.picked[i]) continue;
            if (!goes_by(facts, which, cut, i, kind, value, by)) return false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                if (output.contains(row + by.down, col + by.across)) put(row + by.down, col + by.across, color);
            });
        }
        return true;
    });
}

// How many colors the picked pieces of a cut hold together (color_count).
inline int picked_colors(const partner_pieces& cut) {
    return color_count(cut.count(), [&](int i) { return (bool)cut.picked[(size_t)i]; },
                       [&](int i) { return (uint32_t)palette(cut.pieces[(size_t)i]); });
}

// A PIECE REPEATED, walked: each picked piece laid again where its Tile lays it
// (lay_tiled), each step read off the pieces the pair's cut hands over --
// walked (walk_paint) and laid every way the line asks (lay_paint). A piece no
// step moves is drawn where it lies, as the op hands it on. THE OP LAYS EVERY
// COPY INTO ITS ONE LIST and drops those past its room, so a pair whose pieces
// and copies would not fit in it cannot be drawn (8a371977's 3s, each repeated
// along its row: the later rows' copies were dropped).
inline optional<paint_walk> tile_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      const tile_value& tile) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const int colors = picked_colors(cut);
        const cell_box key = key_box(cut);
        // The list's slots taken: every piece not picked rides through in one.
        int slots = (int)std::count(cut.picked.begin(), cut.picked.end(), false);
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            if (!lay_tiled(tile, piece.height, piece.width, [&](int row, int col) { return (int8_t)piece.data[row][col]; },
                           piece.row, piece.col, colors, key, facts.train[which].output,
                           [&] { return ++slots <= MAX_OBJECTS_IN_A_LIST; }, put))
                return false;
        }
        return true;
    });
}

// THE PICKED PIECES, AS ONE FIGURE, REPEATED FROM PAST THEMSELVES, walked:
// UnionList folds them into one figure and Trim fits its box to them, Shift
// moves it its own size the way `shift` names (shift_by, the step the op
// takes), and the Tile lays it again from there (lay_tiled), each step read off
// the figure and the key, the pieces not picked (12422b43's rows beside the
// 5s, stacked under the figure every key height). Picking nothing, it draws
// nothing.
inline optional<paint_walk> past_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int shift,
                                      const tile_value& tile) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        cell_box figure;
        cells drawn;
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) {
                figure.include(cut.boxes[i]);
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) { drawn[row][col] = color; });
            }
        if (figure.empty()) return false;
        const direction_step by = shift_by(shift, figure.width(), figure.height());
        return lay_tiled(tile, figure.height(), figure.width(),
                         [&](int row, int col) { return (int8_t)drawn[figure.row0 + row][figure.col0 + col]; },
                         figure.row0 + by.down, figure.col0 + by.across, picked_colors(cut), key_box(cut),
                         facts.train[which].output, [] { return true; }, put);
    });
}

// EACH CELL THE RAYS OF A PAIR'S PICKED PIECES REACH on its output, `radius`
// cells long, from the middle of each piece's box -- up, down and to the
// sides, or along the diagonals (ray_seed_of and each_extended_ray,
// 15_op_math.hpp, the walk the op takes) -- handed to `visit(i, row, col,
// along)`, i the piece whose ray it is, `along` steps out along its ray, in the
// order the op draws them.
template <class Visit>
inline void each_ray_cell(const replay_facts& facts, size_t which, const partner_pieces& cut, bool diagonal,
                          int radius, Visit visit) {
    const grid& output = facts.train[which].output;
    const vector<ray_seed>& seeds = cut.ray_seeds();
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i]) continue;
        const object& box = cut.pieces[i];
        const ray_seed seed = seeds[(size_t)i];
        each_extended_ray(seed, radius, diagonal, [&](int down, int across, int along) {
            const int row = box.row + seed.row + down, col = box.col + seed.col + across;
            if (output.contains(row, col)) visit(i, row, col, along);
        });
    }
}

// RAYS GROWN OFF A PIECE, walked. Extend grows rays from the middle of each
// picked piece's box, `radius` cells long, in the color of its cell nearest
// that middle (each_ray_cell; walk_paint), and the line may paint them over in one color of
// its own: laid every way the line asks, in one color (lay_paint), which it
// sees too.
inline optional<paint_walk> rays_walk(const replay_facts& facts, const vector<partner_pieces>& handed, bool diagonal,
                                      int radius) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        each_ray_cell(facts, which, cut, diagonal, radius, [&](int, int row, int col, int) { put(row, col, (int8_t)TRANSPARENT); });
        return true;
    });
}

// RAYS ACROSS OR ALONG A PARTNER, walked: ExtendByPartner's rays from each
// picked piece's seed (each_partner_ray) to the edge of each pair's output,
// laid as rays_walk lays them.
inline optional<paint_walk> partner_rays_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int value) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const vector<ray_seed>& seeds = cut.ray_seeds();
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& box = cut.pieces[i];
            const ray_seed seed = seeds[(size_t)i];
            each_partner_ray(value, i, cut, seed, [&](int down, int across) {
                const int row = box.row + seed.row + down, col = box.col + seed.col + across;
                if (output.contains(row, col)) put(row, col, (int8_t)TRANSPARENT);
            });
        }
        return true;
    });
}

// THE REACHES A CROSSING IS ASKED WITH: rays to the grid's edge, up, down and
// to the sides [0] or along the diagonals [1]; then each track grown in step,
// overrun [CROSSING_RAYS + the track].
constexpr int CROSSING_RAYS = 2, CROSSING_REACHES = CROSSING_RAYS + (int)grown_along::count;

// WHERE THE PICKED PIECES MEET, walked: each picked piece's reach as its op
// hands it back -- its rays to the grid's edge (each_ray_cell), or its body and
// its growth in step with the others, overrun: the cells it took and the cells
// it reached in the turn another did (grow_in_step) -- and IntersectionList
// keeps the cells two or more of them reach (intersected_cell, 15_op_math.hpp,
// what the op keeps): 11e1fe23's three corners, grown along their diagonals to
// the square's middle, and 29c11459's dots, two to a row, grown along their
// rows to its middle. Laid in one color, the line's Recolor after it
// (lay_paint).
inline optional<paint_walk> crossing_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int reach) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int width = output.width, height = output.height;
        // What each piece the op hands back holds, cell by cell.
        vector<cells> held((size_t)cut.count());
        vector<int> grown;
        if (reach < CROSSING_RAYS) {
            const vector<ray_seed>& seeds = cut.ray_seeds();
            for (int i = 0; i < cut.count(); ++i)
                if (cut.picked[i] && seeds[(size_t)i].row >= 0) grown.push_back(i);
            each_ray_cell(facts, which, cut, reach == 1, row_of(node_type::ExtendOrthogonal).high,
                          [&](int i, int row, int col, int) { held[(size_t)i][row][col] = seeds[(size_t)i].color; });
        } else {
            int16_t owner[MAX_GRID_CELLS], turn[MAX_GRID_CELLS], queue[MAX_GRID_CELLS];
            uint8_t cell_directions[MAX_GRID_CELLS];
            std::fill_n(owner, width * height, (int16_t)-1);
            for (int i = 0; i < cut.count(); ++i)
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    if (!output.contains(row, col)) return;
                    if (owner[row * width + col] < 0) owner[row * width + col] = (int16_t)i;
                    if (cut.picked[i]) held[(size_t)i][row][col] = color;
                });
            grow_in_step(width, height, (grown_along)(reach - CROSSING_RAYS), [&](int piece) { return cut.is_picked(piece); },
                         [](int one, int other) { return one < other; }, owner, turn, queue, cell_directions,
                         [&](int at, int, int piece, bool in_one_turn) {
                             if (in_one_turn) held[(size_t)piece][at / width][at % width] = (int8_t)cut.color(piece);
                         });
            for (int at = 0; at < width * height; ++at)
                if (turn[at] > 0) held[(size_t)owner[at]][at / width][at % width] = (int8_t)cut.color(owner[at]);
            for (int i = 0; i < cut.count(); ++i)
                if (cut.picked[i]) grown.push_back(i);
        }
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                if (const int8_t kept = intersected_cell((int)grown.size(),
                                                         [&](int k) { return held[(size_t)grown[(size_t)k]][row][col]; });
                    is_color(kept))
                    put(row, col, kept);
        return true;
    });
}

// THE RAYS OF EVERY LENGTH, WALKED ONCE. A ray `radius` long reaches the cells
// a longer one reaches within `radius` of its seed, so the longest rays, each
// cell kept with how far along the nearest one reaches it, hold the cells of
// every length. And laid in one color, none torn (rays_walk), a length's cells
// judge the same whatever order they landed in -- but for two colors tied for
// the one the output shows most, which only that order breaks (lay_paint).
struct ray_fan {
    struct cell { int8_t row, col; };
    // Pair by pair, the cells the rays reach, nearest first: those first
    // reached `along` cells out are cells[starts[along]] up to cells[starts[along + 1]].
    struct of_pair { vector<cell> cells; vector<int> starts; };
    vector<of_pair> pairs;
    bool every_piece = true;
};
// The rays `reach` cells long fanned out; nothing where rays_walk walks
// nothing (walk_paint).
inline optional<ray_fan> rays_fan(const replay_facts& facts, const vector<partner_pieces>& handed, bool diagonal,
                                  int reach) {
    if (!facts.overlays || facts.train.empty()) return nullopt;
    ray_fan fan;
    fan.pairs.resize(facts.train.size());
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const partner_pieces& cut = handed[which];
        fan.every_piece = fan.every_piece && cut.every_piece_picked();
        // How far along the nearest ray reaches each cell, 0 where none does.
        int8_t nearest[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        each_ray_cell(facts, which, cut, diagonal, reach, [&](int, int row, int col, int along) {
            int8_t& at = nearest[row][col];
            if (at == 0 || along < at) at = (int8_t)along;
        });
        ray_fan::of_pair& fanned = fan.pairs[which];
        fanned.starts.assign((size_t)reach + 2, 0);
        for (const auto& row : nearest)
            for (const int8_t along : row)
                if (along) ++fanned.starts[(size_t)along + 1];
        for (size_t along = 1; along < fanned.starts.size(); ++along) fanned.starts[along] += fanned.starts[along - 1];
        fanned.cells.resize((size_t)fanned.starts.back());
        vector<int> next(fanned.starts.begin(), fanned.starts.end() - 1);
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < MAX_GRID_DIM; ++col)
                if (const int8_t along = nearest[row][col])
                    fanned.cells[(size_t)next[(size_t)along]++] = { (int8_t)row, (int8_t)col };
    }
    return fan;
}

// EVERY LENGTH OF THE RAYS LAID AT ONCE, `how` says, in one color, as lay_paint
// lays each length's own walk: adding the cells one step further out, length
// by length, it counts what lands (landing_on, lands), what the output shows
// there and whether it came out already with what is down (comes_out), takes
// the color shown most and judges each pair (brings_nearer) -- the cells laid
// right, as marks, and how many are laid wrong. A length
// whose cells tie for the color is laid off its own walk (lay_paint), and a
// length that holds keeps its walk for its lists (heard::materialize):
// `walk_of(length)` hands it over, walked once. Lengths 0 to `reach`.
template <class WalkOf>
inline vector<optional<marked_paint>> lay_every_length(const replay_facts& facts, const ray_fan& fan,
                                                       const laying& how, int reach, WalkOf walk_of) {
    // What has landed so far on a pair: how many cells, of those how many the
    // output shows in each color, how many came out so already, and where.
    struct landed_so_far {
        int cells = 0, shows[COLOR_COUNT] = {}, shown_before[COLOR_COUNT] = {};
        cell_set right[COLOR_COUNT];
    };
    const size_t pairs = fan.pairs.size();
    vector<landed_so_far> landed(pairs);
    vector<optional<marked_paint>> laid((size_t)reach + 1);
    for (int length = 1; length <= reach; ++length) {
        for (size_t which = 0; which < pairs; ++which) {
            const ray_fan::of_pair& fanned = fan.pairs[which];
            if ((size_t)length + 1 >= fanned.starts.size()) continue;
            const figure_pair& pair = facts.pairs[which];
            const cells* down = landing_on(how, which);
            const cells& before = (*how.canvas)[which];
            landed_so_far& so_far = landed[which];
            for (int at = fanned.starts[(size_t)length]; at < fanned.starts[(size_t)length + 1]; ++at) {
                const ray_fan::cell& cell = fanned.cells[(size_t)at];
                if (down && !lands(how.lay, (*down)[cell.row][cell.col])) continue;
                ++so_far.cells;
                const int shows = pair.whole.output.data[cell.row][cell.col];
                if (!is_color(shows)) continue;
                ++so_far.shows[shows];
                so_far.shown_before[shows] += comes_out(pair, cell.row, cell.col, before[cell.row][cell.col]);
                so_far.right[shows].add(cell.row, cell.col);
            }
        }
        long long shown[COLOR_COUNT] = {};
        for (const landed_so_far& so_far : landed)
            for (int color = 0; color < COLOR_COUNT; ++color) shown[color] += so_far.shows[color];
        const long long most = *ranges::max_element(shown);
        if (most == 0) continue;
        if (ranges::count(shown, most) > 1) {
            if (const paint_walk* walk = walk_of(length)) laid[(size_t)length] = lay_paint<marked_paint>(facts, *walk, how);
            continue;
        }
        const int8_t paint = (int8_t)(ranges::find(shown, most) - std::begin(shown));
        int wrong = 0;
        bool holds = true;
        for (size_t which = 0; which < pairs; ++which) {
            const landed_so_far& so_far = landed[which];
            const pair_tally tally{ so_far.shows[paint] - so_far.shown_before[paint], so_far.cells - so_far.shows[paint] };
            holds = holds && brings_nearer(facts, which, tally);
            wrong += tally.wrong;
        }
        const paint_walk* walk = holds ? walk_of(length) : nullptr;
        if (!walk) continue;
        marked_paint& seen = laid[(size_t)length].emplace();
        seen.drawn.at.resize(pairs);
        for (size_t which = 0; which < pairs; ++which) seen.drawn.at[which] = landed[which].right[paint];
        seen.wrong = wrong;
        seen.paint = paint;
        seen.every_piece = fan.every_piece;
        seen.walk = walk;
        seen.how = how;
    }
    return laid;
}

// THE PICKED PIECES, AS ONE FIGURE, DRAWN AGAIN TURNED in their own box,
// replayed: UnionList folds them into one, Trim fits its box to them, and
// TurnRight, twice or TurnLeft -- `pose` 3, 6 or 5, read through pose_source as
// the ops read it, posed_place here -- turns the figure in that box, its
// top-left corner where it was. Laid over what is down, the canvas (paint_holds),
// and the figure itself is down where it lies, so the output must show it there
// too wherever the turned copy does not cover it: the copy and the figure are
// drawn, both, since the matching read the figure turned (11852cab's design,
// the same from all four sides). Picking nothing, it draws nothing.
inline optional<replayed> turned_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int pose,
                                       const vector<cells>& canvas) {
    using rule = correspondence_election::fired_rule;
    return paint_holds(facts, handed, { rule::laid_over, &canvas },
                       [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        cell_box figure;
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) figure.include(cut.boxes[i]);
        if (figure.empty()) return false;
        cells copy;
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i])
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    const auto [down, across] = posed_place(pose, figure.width(), figure.height(),
                                                            row - figure.row0, col - figure.col0);
                    if (output.contains(figure.row0 + down, figure.col0 + across))
                        copy[figure.row0 + down][figure.col0 + across] = color;
                });
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i])
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
                    if (copy[row][col] == (int8_t)TRANSPARENT) put(row, col, color);
                });
        for (int row = 0; row < output.height; ++row)
            for (int col = 0; col < output.width; ++col)
                if (copy[row][col] != (int8_t)TRANSPARENT) put(row, col, copy[row][col]);
        return true;
    });
}

// THE SYMMETRIES A PICTURE IS MADE TO KEEP, the narrowest first: the poses
// (1_readings.hpp's three bits) laid over the picture in turn, each taken of
// what is drawn by then, so the last pose laid makes the whole symmetry -- one
// mirror or the half turn; then both mirrors, or every quarter turn; then all
// eight ways round.
struct symmetry { const char* says; int8_t poses[3]; };
constexpr symmetry symmetries[] = {
    { "mirrored left to right", { 4, -1, -1 } },      { "mirrored top to bottom", { 2, -1, -1 } },
    { "mirrored across its diagonal", { 1, -1, -1 } }, { "mirrored across its other diagonal", { 7, -1, -1 } },
    { "turned a half turn", { 6, -1, -1 } },          { "mirrored both ways", { 4, 2, -1 } },
    { "turned every quarter turn", { 3, 6, -1 } },    { "kept all eight ways round", { 4, 2, 3 } } };

// THE PICTURE MADE SYMMETRIC, replayed: the picture so far -- the one piece
// `handed` holds in each pair, the grid's own size -- laid over itself posed
// each way the symmetry names, in turn, every pose taken of what is drawn by
// then, read through pose_source as the ops read it, as `Output SelectAll`
// and the pose's operators lay it around the grid's middle. A pose that trades
// rows for columns fits only a square grid.
inline optional<replayed> symmetric_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                          const symmetry& kept) {
    return paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        if (cut.count() != 1) return false;
        grid drawn(output.width, output.height);
        cut.pieces[0].each_cell([&](int row, int col, int8_t color) {
            if (drawn.contains(row, col)) drawn.data[row][col] = color;
        });
        for (const int8_t pose : kept.poses) {
            if (pose < 0) break;
            const auto [width, height] = posed_size(pose, output.width, output.height);
            if (width != output.width || height != output.height) return false;
            grid layer(width, height);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const auto [from_row, from_col] = pose_source(pose, output.width, output.height, row, col);
                    layer.data[row][col] = drawn.data[from_row][from_col];
                }
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    if (layer.data[row][col] != (int8_t)TRANSPARENT) {
                        drawn.data[row][col] = layer.data[row][col];
                        put(row, col, (int8_t)layer.data[row][col]);
                    }
        }
        return true;
    });
}

// A PIECE'S BOX FILLED, replayed. AddColor paints the cells each picked piece
// leaves empty in its box in one color, and the box is laid under what is down
// (paint_holds) as the sentence lays it: the fill, and the piece itself in its
// own colors -- where the line moved the piece, its old place shows it again
// (834ec97d). The fill alone chooses the color.
inline optional<replayed> box_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                    const vector<cells>& canvas) {
    using rule = correspondence_election::fired_rule;
    auto boxes = [](bool with_the_piece, int8_t fill) {
        return [with_the_piece, fill](size_t, const partner_pieces& cut, auto&& put) {
            for (int i = 0; i < cut.count(); ++i) {
                if (!cut.picked[i]) continue;
                const cell_box& box = cut.boxes[i];
                for (int row = box.row0; row <= box.row1; ++row)
                    for (int col = box.col0; col <= box.col1; ++col)
                        if (const int8_t own = (int8_t)cut.color_at(i, row, col); !is_color(own)) put(row, col, fill);
                        else if (with_the_piece) put(row, col, own);
            }
            return true;
        };
    };
    const auto filled = paint_holds(facts, handed, { rule::laid_under, &canvas, laying::in_one },
                                    boxes(false, (int8_t)TRANSPARENT));
    if (!filled) return nullopt;
    auto seen = paint_holds(facts, handed, { rule::laid_under, &canvas }, boxes(true, filled->paint));
    if (seen) seen->paint = filled->paint;
    return seen;
}

// WHICH ROOM A ROOM PAINT FILLS (filled_room's value), in the order a tie goes:
// every room (d511f180's ground, all of it 8); by size, ranked among the rooms
// alone, as SelectLargestObject and SelectSmallestObject rank what is left once
// every other piece is deleted -- the largest, the smallest, the 2nd largest,
// the 2nd smallest... -- then the room of the part the picked pieces fence
// off, numbered as SelectPartition numbers the parts.
// Past those, every room at once, each in the color its own key names -- its
// shape or its symmetry (shape_key) -- as a table read off the pairs says
// (rooms_table). Then the ground's places shaped like the shape the pairs'
// changes take, every one (shaped_rooms). Last, the table again over only the
// rooms the outputs repaint, where a pick read off the pairs names them
// (rooms_a_pick_names).
constexpr int ROOM_EVERY = 0, ROOM_SIZES = 1, ROOM_PARTS = ROOM_SIZES + 2 * SIZE_RANKS,
              ROOM_CHOICES = ROOM_PARTS + row_of(node_type::SelectPartition).high + 1;
constexpr int ROOM_TABLE = ROOM_CHOICES, ROOM_VALUES = ROOM_TABLE + (int)shape_key::count, ROOM_SHAPE = ROOM_VALUES,
              ROOM_NAMED = ROOM_SHAPE + 1;
constexpr bool room_by_size(int chosen) { return chosen >= ROOM_SIZES && chosen < ROOM_PARTS; }
constexpr bool room_largest(int chosen) { return (chosen - ROOM_SIZES) % 2 == 0; }
constexpr int room_rank(int chosen) { return (chosen - ROOM_SIZES) / 2; }
constexpr int room_part(int chosen) { return chosen - ROOM_PARTS; }

// A PART'S ROOM: the part it lies in, its cells, and whether another piece
// shares its part.
struct fenced_room {
    int part = -1;
    vector<int16_t> cells;
    bool shared = false;
};

// THE ROOMS THE PICKED PIECES FENCE OFF on each pair: each part's cells the
// pieces leave, the input's cells that are no figure, which Partition cuts
// into rooms, one a color, before the ground is deleted. In the order
// Partition hands them back (each_fenced_part, the op's own walk), on the
// canvas the pieces lie on, as Partition reads it.
inline vector<vector<fenced_room>> rooms_fenced(const replay_facts& facts, const vector<partner_pieces>& handed) {
    vector<vector<fenced_room>> rooms(handed.size());
    for (size_t which = 0; which < handed.size() && which < facts.train.size(); ++which) {
        const tile_size canvas = facts.canvas(which);
        const figure_pair& pair = facts.pairs[which];
        const partner_pieces& pieces = handed[which];
        each_fenced_part(pieces, canvas.width, canvas.height,
            [&](int part, const cell_box& box, auto in_part) {
                bool clipped = false;
                for (int index = 0; index < pieces.count() && !clipped; ++index) {
                    if (pieces.is_picked(index)) continue;
                    const cell_box own = pieces.box(index);
                    for (int row = max(box.row0, own.row0); row <= min(box.row1, own.row1) && !clipped; ++row)
                        for (int col = max(box.col0, own.col0); col <= min(box.col1, own.col1) && !clipped; ++col)
                            clipped = in_part(row, col) && pieces.painted(index, row, col);
                }
                for (int color = 0; color < COLOR_COUNT; ++color) {
                    fenced_room room{ part, {}, clipped };
                    for (int row = box.row0; row <= box.row1; ++row)
                        for (int col = box.col0; col <= box.col1; ++col)
                            if (pair.whole.input.contains(row, col) && pair.whole.input.data[row][col] == color &&
                                !pair.input_figures.has(row, col) && in_part(row, col))
                                room.cells.push_back((int16_t)cell_index(row, col));
                    if (!room.cells.empty()) rooms[which].push_back(std::move(room));
                }
            });
    }
    return rooms;
}

// THE GROUND'S PLACES SHAPED LIKE THE SHAPE THE CHANGES TAKE, on each pair:
// the places the shape cut takes (each_shape_place) among the input's cells
// that are no figure -- the places Split [4] cuts off the ground's layer, as
// Partition cuts rooms off it, overlapping where they do.
inline vector<vector<fenced_room>> shaped_rooms(const turning_shape& shape, const vector<figure_pair>& pairs) {
    vector<vector<fenced_room>> rooms(pairs.size());
    for (size_t which = 0; which < pairs.size() && shape.holds(); ++which) {
        const figure_pair& pair = pairs[which];
        each_shape_place(pair.whole.input, shape, {}, [&](int row, int col) {
            fenced_room& room = rooms[which].emplace_back();
            shape.cells.each_cell([&](int down, int across, int8_t) {
                room.cells.push_back((int16_t)cell_index(row + down, col + across));
            });
        }, [&](int row, int col) { return !pair.input_figures.has(row, col); });
    }
    return rooms;
}

// THE ROOM `chosen` NAMES among one pair's rooms (ROOM_PARTS), by size or by
// part, or none: the part's own room, where no other piece lies in that part;
// or, by size, the room at that rank, where no other room has its size, so the
// rank names it whatever order the op ranks ties in.
inline const fenced_room* room_chosen(const vector<fenced_room>& rooms, int chosen) {
    if (!room_by_size(chosen)) {
        const auto found = ranges::find(rooms, room_part(chosen), &fenced_room::part);
        return found == rooms.end() || found->shared ? nullptr : &*found;
    }
    const size_t rank = (size_t)room_rank(chosen);
    if (rank >= rooms.size()) return nullptr;
    vector<int> order(rooms.size());
    size_order((int)rooms.size(), room_largest(chosen), [&](int i) { return (int)rooms[(size_t)i].cells.size(); },
               order.data());
    auto cells_at = [&](size_t at) { return rooms[(size_t)order[at]].cells.size(); };
    const bool tied = (rank > 0 && cells_at(rank - 1) == cells_at(rank)) ||
                      (rank + 1 < order.size() && cells_at(rank + 1) == cells_at(rank));
    return tied ? nullptr : &rooms[(size_t)order[rank]];
}

// A ROOM FILLED, replayed. Partition hands back the ground of each part the
// picked pieces fence off as its room (rooms_fenced), and Recolor paints the
// room `chosen` names -- or every room -- in one color, laid the way `lay`
// says on what is down (paint_holds): over it, or under it, filling only what
// nothing has painted.
inline optional<replayed> room_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                     const vector<vector<fenced_room>>& rooms, const vector<cells>& canvas,
                                     int chosen, int8_t lay) {
    return paint_holds(facts, handed, { lay, &canvas, laying::in_one },
                       [&](size_t which, const partner_pieces&, auto&& put) {
        if (which >= rooms.size()) return false;
        auto fill = [&](const fenced_room& room) {
            for (const int16_t cell : room.cells) put(cell_row(cell), cell_col(cell), (int8_t)TRANSPARENT);
        };
        if (chosen == ROOM_EVERY) {
            ranges::for_each(rooms[which], fill);
            return !rooms[which].empty();
        }
        const fenced_room* room = room_chosen(rooms[which], chosen);
        if (!room) return false;
        if (room_by_size(chosen)) fill(*room);
        else
            for (const fenced_room& part : rooms[which])
                if (part.part == room->part) fill(part);
        return true;
    });
}

// A ROOM'S KEY of the kind `kind` names (shape_key_of), read as ShapeColorMap
// reads a room Partition hands back: its cells, in the box they fill, all one
// color, the ground's. The reading never knows the ground (the plan owns it),
// and a drawing in one color keys alike in any, so 0 draws it.
inline uint32_t room_key(const fenced_room& room, int kind) {
    cell_box box;
    bool in_room[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (const int16_t cell : room.cells) {
        box.include_cell(cell_row(cell), cell_col(cell));
        in_room[cell_row(cell)][cell_col(cell)] = true;
    }
    return shape_key_of(kind, box.width(), box.height(), [&](int row, int col) {
        return in_room[box.row0 + row][box.col0 + col] ? (int8_t)0 : (int8_t)TRANSPARENT;
    });
}

// THE ROOMS' TABLE, read off the pairs (read_off) as the shape rule reads its
// pieces': each room's key of the kind `kind` names, and the one color the
// output shows over it. A room the output leaves empty stays as it is and
// says nothing. None where a room comes out in two colors, or none holds.
inline shape_color_map_rule rooms_table(const replay_facts& facts, const vector<vector<fenced_room>>& rooms,
                                        int kind) {
    vector<shape_color_map_rule::seen_key> seen;
    for (size_t which = 0; which < rooms.size() && which < facts.train.size(); ++which) {
        const grid& output = facts.train[which].output;
        const uint16_t shown = shape_color_map_rule::colors_shown(facts.train[which].input);
        for (const fenced_room& room : rooms[which]) {
            // -2 until a cell is read, -3 once two cells disagree.
            int worn = -2;
            for (const int16_t cell : room.cells) {
                const int there = output.data[cell_row(cell)][cell_col(cell)];
                worn = worn == -2 || worn == there ? there : -3;
            }
            if (worn == TRANSPARENT) continue;
            if (!is_color(worn) ||
                !shape_color_map_rule::see(seen, room_key(room, kind), (int8_t)worn, which, shown))
                return {};
        }
    }
    return shape_color_map_rule::read_off(seen);
}

// THE ROOMS THE OUTPUTS REPAINT, AND THE PICK THAT NAMES THEM: the rooms each
// output shows in one color of its own, read off the pairs, named by the pick
// that takes exactly them in every pair (pick_for_group) among the rooms as
// Partition hands them back once the walls are deleted -- a piece each, in the
// ground's colors, trimmed -- with the rooms it takes. None where no pick names
// them, or where every room is repainted, which every room says (868de0fa's
// hollows, each taking the color its shape names, the room around the walls
// left as it is).
struct named_rooms {
    piece_pick pick;
    vector<vector<fenced_room>> rooms;
};
inline optional<named_rooms> rooms_a_pick_names(const replay_facts& facts, const vector<vector<fenced_room>>& rooms) {
    vector<vector<object>> hands(rooms.size());
    vector<vector<bool>> repainted(rooms.size());
    for (size_t which = 0; which < rooms.size() && which < facts.train.size(); ++which)
        for (const fenced_room& room : rooms[which]) {
            const grid& output = facts.train[which].output;
            cell_box box;
            // -2 until a cell is read, -3 once two cells disagree.
            int worn = -2;
            for (const int16_t cell : room.cells) {
                box.include_cell(cell_row(cell), cell_col(cell));
                const int there = output.data[cell_row(cell)][cell_col(cell)];
                worn = worn == -2 || worn == there ? there : -3;
            }
            hands[which].push_back(object::from_cells(facts.pairs[which].whole.input,
                                                      summary_of(box, (int)room.cells.size()), room.cells.data(),
                                                      (int)room.cells.size()));
            repainted[which].push_back(is_color(worn));
        }
    const optional<piece_pick> pick = pick_for_group(replay_facts(facts.train, facts.pairs, std::move(hands)), repainted);
    if (!pick || pick->every()) return nullopt;
    named_rooms named{ *pick, vector<vector<fenced_room>>(rooms.size()) };
    for (size_t which = 0; which < rooms.size(); ++which)
        for (size_t at = 0; at < rooms[which].size(); ++at)
            if (repainted[which][at]) named.rooms[which].push_back(rooms[which][at]);
    return named;
}

// THE ROOMS, EACH IN THE COLOR ITS KEY NAMES, replayed: ShapeColorMap over
// every room Partition hands back (rooms_fenced), in the color `table` names
// for its key -- a room it names none for left as it is -- laid the way `lay`
// says on what is down (paint_holds).
inline optional<replayed> room_table_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                           const vector<vector<fenced_room>>& rooms, const vector<cells>& canvas,
                                           const shape_color_map_rule& table, int kind, int8_t lay) {
    return paint_holds(facts, handed, { lay, &canvas }, [&](size_t which, const partner_pieces&, auto&& put) {
        if (which >= rooms.size()) return false;
        for (const fenced_room& room : rooms[which]) {
            const int color = table.color_of(room_key(room, kind));
            if (color < 0) continue;
            for (const int16_t cell : room.cells) put(cell_row(cell), cell_col(cell), (int8_t)color);
        }
        return true;
    });
}

// BARS BETWEEN PIECES OF ONE COLOR, replayed. Bridge reaches from each picked
// piece to the picked pieces of its own color whose middles share its row or its
// column, the `way` it is told (bridge_bars, the walk the op takes), and fills
// the box between two -- laid under what is down (paint_holds), so a bar only
// fills what nothing has painted -- in the bar's own color, or all of them in one.
// Past those ways, the L from each picked piece to its partner (bridge_bars),
// SAID ONLY WHERE THE PAIRS SHOW IT TURN, in one pair at least: an L that never
// turns is a bar, and which way it turns would be a guess on a test, as a bar is
// said no wider than the pairs show it (833966f4's column of cells, where the
// top one ran straight down to the 6 in every pair, a coincidence).
// Each selected color must contribute a visible bar in each unfinished pair.
// A color that does nothing supplies no evidence for joining it on a test.
inline optional<replayed> bridge_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       const vector<cells>& canvas, int way) {
    using rule = correspondence_election::fired_rule;
    bool turned = false;
    auto seen = paint_holds(facts, handed, { rule::laid_under, &canvas, laying::own_else_one },
                            [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        uint16_t selected_colors = 0, bridging_colors = 0;
        // The bars take one place more in the pieces' list, all of them one drawing.
        if (cut.count() >= MAX_OBJECTS_IN_A_LIST) return false;
        for (int piece = 0; piece < cut.count(); ++piece)
            if (cut.is_picked(piece) && is_color(cut.color(piece)))
                selected_colors = (uint16_t)(selected_colors | 1u << cut.color(piece));
        turned = bridge_bars(way, output.width, output.height, cut,
                             [&](const cell_box& bar, int8_t color) {
            for (int row = bar.row0; row <= bar.row1; ++row)
                for (int col = bar.col0; col <= bar.col1; ++col)
                    if (output.contains(row, col)) {
                        if (is_color(color) && lands(rule::laid_under, canvas[which][row][col]) &&
                            !comes_out(facts.pairs[which], row, col, canvas[which][row][col]))
                            bridging_colors = (uint16_t)(bridging_colors | 1u << color);
                        put(row, col, color);
                    }
            return true;
        }) || turned;
        return way >= BRIDGE_TO_PARTNER || facts.finished(which) || !(selected_colors & ~bridging_colors);
    });
    if (way >= BRIDGE_TO_PARTNER && !turned) return nullopt;
    return seen;
}

// THE CELLS A PIECE ENCLOSES: those its own paint walls off from the edge of its
// box (grid_parts, the hollows the ops fill), each handed to `see(row, col)`
// where it lies.
template <class See>
inline void each_hollow(const object& piece, See see) {
    grid_parts(piece.width, piece.height, [&](int row, int col) {
        return piece.data[row][col] != (int8_t)TRANSPARENT;
    }).each_hollow([&](int row, int col) { see(piece.row + row, piece.col + col); });
}

// THE HOLLOWS OF A PIECE, FILLED, replayed. InteriorColor fills each picked
// piece's hollows (each_hollow), in each piece's own color or all of them in
// one, and its sentence lays the whole piece under what is down (paint_holds):
// its own cells as they are -- a piece the line removed comes back with it
// (d5d6de2d's ring) -- and a marker sitting in a hollow keeps its own. In its
// own colors where they lay no cell wrong, else whichever says more, right
// cells less wrong ones, its own colors on a tie; in one, the color a paint in
// one color reads off the output where the hollows land (lay_paint).
inline optional<replayed> fill_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                     const vector<cells>& canvas) {
    using rule = correspondence_election::fired_rule;
    // Each picked piece, its own cells as they are and its hollows in `fill`,
    // or in its own color where `fill` is no color.
    auto filled = [&](int8_t fill) {
        return paint_holds(facts, handed, { rule::laid_under, &canvas },
                           [&](size_t, const partner_pieces& cut, auto&& put) {
            for (int i = 0; i < cut.count(); ++i)
                if (cut.picked[i]) {
                    const object& piece = cut.pieces[i];
                    piece.each_cell([&](int row, int col, int8_t color) { put(row, col, color); });
                    const int8_t worn = is_color(fill) ? fill : piece.first_color();
                    each_hollow(piece, [&](int row, int col) { put(row, col, worn); });
                }
            return true;
        });
    };
    auto own = filled((int8_t)TRANSPARENT);
    if (own && own->wrong_cells() == 0) return own;
    const auto hollows = paint_holds(facts, handed, { rule::laid_under, &canvas, laying::in_one },
                                     [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) each_hollow(cut.pieces[i], [&](int row, int col) { put(row, col, (int8_t)TRANSPARENT); });
        return true;
    });
    if (!hollows) return own;
    auto one = filled(hollows->paint);
    if (one) one->paint = hollows->paint;
    if (one && (!own || one->right_cells() - one->wrong_cells() > own->right_cells() - own->wrong_cells()))
        return one;
    return own;
}

// THE INSIDE OF THE PICKED PIECES, IN ONE COLOR, replayed: Outline hollows each
// picked piece to its edge (outlined), then InteriorColor fills what the edge
// encloses (each_hollow) -- the piece's core and any hollow it had -- in one
// color, the same in every pair, read off the output where the first lands.
// Laid over what is down, the canvas, the edge as it was: bb43febb's blocks,
// red inside a gray rim; 694f12f3's, red inside the larger and blue inside the
// smaller.
inline optional<replayed> inside_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       const vector<cells>& canvas) {
    using rule = correspondence_election::fired_rule;
    int8_t inside = (int8_t)TRANSPARENT;
    for (size_t which = 0; which < handed.size() && inside == (int8_t)TRANSPARENT; ++which)
        for (int i = 0; i < handed[which].count() && inside == (int8_t)TRANSPARENT; ++i)
            if (handed[which].picked[i])
                each_hollow(outlined(handed[which].pieces[i]), [&](int row, int col) {
                    const grid& output = facts.pairs[which].whole.output;
                    if (inside == (int8_t)TRANSPARENT && output.contains(row, col)) inside = (int8_t)output.data[row][col];
                });
    if (!is_color(inside)) return nullopt;
    auto seen = paint_holds(facts, handed, { rule::laid_over, &canvas },
                            [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object rim = outlined(cut.pieces[i]);
            rim.each_cell([&](int row, int col, int8_t color) { put(row, col, color); });
            each_hollow(rim, [&](int row, int col) { put(row, col, inside); });
        }
        return true;
    });
    if (seen) seen->paint = inside;
    return seen;
}

// THE HOLLOWS, FILLED BY SHAPE, replayed. ShapeToInteriorColorMap takes the
// pieces as the line hands them over, looks each one's shape up in the learned
// table, and fills its hollows (each_hollow) with that color -- laid under the
// hand, so a marker already sitting in a hollow keeps its own (paint_holds).
// Sees the cells it fills and how many pieces it filled.
inline optional<replayed> interior_fills_hold(const replay_facts& facts, const shape_color_map_rule& table) {
    using rule = correspondence_election::fired_rule;
    int pieces_filled = 0;
    auto seen = paint_holds(facts, handed_pieces(facts, piece_pick::all()), { rule::laid_under },
                       [&](size_t, const partner_pieces& cut, auto&& put) {
        for (const object& piece : cut.pieces) {
            const int8_t fill = (int8_t)table.color_of(shape_key::shape, piece);
            if (fill < 0) continue;
            bool painted = false;
            each_hollow(piece, [&](int row, int col) { put(row, col, fill); painted = true; });
            pieces_filled += painted;
        }
        return true;
    });
    if (seen) seen->pieces = pieces_filled;
    return seen;
}

// THE PIECES AS THE LINE LEAVES THEM. Each rule on the line acts on the hand the
// rules before it left, as the program runs them, so a rule asked after them is
// replayed on that hand and not on the cut: the pieces it picks, where they
// stand and what they wear. Carried here pair by pair, one rule at a time, with
// the arithmetic its op runs -- the pick read off the hand as it is, each
// partner found before any piece is touched, copies joining the end of the list.
// A rule that reads a table off the pairs is one stage more: the puzzle's shape
// table and its palette by size are the reading's (inference). False for
// anything else, after which the rules below are asked the way they were
// before the hand was carried.
inline bool carry_the_hand(const correspondence_election::fired_rule& rule, const inference& inf,
                           replay_facts& facts) {
    using said = correspondence_election::fired_rule;
    const shape_color_map_rule& table = inf.shape_color_map;
    // A KEY'S SHAPE NAMING THE FIGURE'S COLOR, carried as beats_of says it, in
    // three rules: the key, the smallest, takes the color its own shape names;
    // the figure, the largest, its nearest piece's, the key's; and the key goes.
    if (rule.as == said::learned_table && table.claimed_by == shape_color_map_rule::body_recolor) {
        said key = listed("", "", said::shape_recolor, (int8_t)shape_key::shape);
        key.pick = { 0, piece_pick::smallest, 0 };
        said figure = listed("", "", said::color_from, (int8_t)partner_by::nearest);
        figure.pick = { 0, piece_pick::largest, 0 };
        said gone = listed("", "", said::removed);
        gone.pick = key.pick;
        return carry_the_hand(key, inf, facts) && carry_the_hand(figure, inf, facts) &&
               carry_the_hand(gone, inf, facts);
    }
    const piece_pick& pick = rule.pick;
    auto recolored = [](object& piece, int color) { piece.repaint([&](int8_t) { return color; }); };
    auto moved = [](object& piece, int down, int across) {
        piece.row = (int16_t)(piece.row + down);
        piece.col = (int16_t)(piece.col + across);
    };
    for (size_t which = 0; which < facts.cuts().size(); ++which) {
        const auto changing = facts.change(which);
        vector<object>& hand = changing.hand;
        const example& pair = facts.train[which];
        const vector<bool> picked = facts.picked(which, pick);
        switch (rule.as) {
        case said::turn:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i] = posed(hand[i], rule.value);
            break;
        case said::move_down:
        case said::move_across:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) moved(hand[i], rule.as == said::move_down ? rule.value : 0,
                                     rule.as == said::move_across ? rule.value : 0);
            break;
        case said::recolor:
        case said::recolor_picked:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) recolored(hand[i], rule.value);
            break;
        case said::swap_colors:
        case said::key_swap: {
            // What trades places, read off the hand as it is before any piece
            // changes: a fixed pair, or the pairs the key the pick leaves names.
            int8_t to[COLOR_COUNT];
            swaps_of(rule.value, facts.handed(which, pick), to);
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i].repaint([&](int8_t now) { return to[now] >= 0 ? to[now] : now; });
            break;
        }
        case said::empty_swap:
            // Every cell of each picked piece's box comes out as the op's own
            // rule says (traded_with_empty), the box kept.
            for (size_t i = 0; i < hand.size(); ++i) {
                if (!picked[i]) continue;
                object& piece = hand[i];
                for (int row = 0; row < piece.height; ++row)
                    for (int col = 0; col < piece.width; ++col)
                        piece.data[row][col] = traded_with_empty(piece.data[row][col], rule.value - EMPTY_PAIRS);
            }
            break;
        case said::color_from:
        case said::align_to:
        case said::move_toward: {
            // Every partner is found on the hand as it was, before any piece
            // changes; a piece with no one partner stays as it is.
            const partner_pieces cut = facts.handed(which, pick);
            if (rule.as == said::color_from && rule.value >= COLORS_FROM_ITS_SHAPE) {
                // Cell by cell from a piece of its shape, every cell read off the
                // hand as it was and laid on a copy of it.
                vector<object> laid = hand;
                for (int i = 0; i < cut.count(); ++i)
                    if (cut.picked[i])
                        colors_from_its_shape(rule.value - COLORS_FROM_ITS_SHAPE, i, cut, [&](int row, int col, int color) {
                            object& piece = laid[(size_t)i];
                            if (is_color(color)) piece.data[row - piece.row][col - piece.col] = (int8_t)color;
                        });
                hand = std::move(laid);
                break;
            }
            constexpr int ways = (int)partner_by::count;
            vector<int> found(hand.size(), -1);
            vector<direction_step> by(hand.size(), direction_step{ 0, 0 });
            const bool by_place = rule.as == said::color_from && rule.value == KEY_BY_PLACE;
            const vector<int8_t> places = by_place ? cut.key_places() : vector<int8_t>{};
            const int keyed = rule.as == said::color_from && rule.value == KEY_COLOR ? key_color(cut) : -1;
            for (int i = 0; i < cut.count(); ++i) {
                if (!cut.picked[i]) continue;
                if (by_place) found[i] = places[(size_t)i];
                else if (rule.as == said::color_from && rule.value == KEY_COLOR) found[i] = keyed;
                else if (rule.as == said::color_from) found[i] = cut.color_from((partner_by)rule.value, i);
                else if (rule.as == said::align_to)
                    cut.goes_level(i, rule.value >= ways, (partner_by)(rule.value % ways), by[i]);
                else cut.goes_toward(i, rule.value, pair.output.width, pair.output.height, by[i]);
            }
            for (size_t i = 0; i < hand.size(); ++i) {
                if (found[i] >= 0) recolored(hand[i], found[i]);
                moved(hand[i], by[i].down, by[i].across);
            }
            break;
        }
        case said::to_edge:
        case said::shifted: {
            const direction_step step = step_of(rule.value);
            for (size_t i = 0; i < hand.size(); ++i) {
                if (!picked[i]) continue;
                const cell_box box = hand[i].painted_box();
                if (rule.as == said::to_edge) {
                    const int steps = steps_to_edge(box, pair.output.width, pair.output.height, rule.value);
                    moved(hand[i], steps * step.down, steps * step.across);
                } else {
                    const direction_step by = shift_by(rule.value, box.width(), box.height());
                    moved(hand[i], by.down, by.across);
                }
            }
            break;
        }
        case said::removed: {
            vector<object> kept;
            for (size_t i = 0; i < hand.size(); ++i)
                if (!picked[i]) kept.push_back(hand[i]);
            hand.swap(kept);
            break;
        }
        case said::fit_into:
            into_sockets(hand, picked, facts.answers_of(which), rule.value);
            break;
        case said::shape_from:
            take_partner_shapes(hand, picked, facts.answers_of(which), rule.value);
            break;
        case said::outlined:
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i]) hand[i] = outlined(hand[i]);
            break;
        case said::shape_recolor:
            // Each picked piece wholly in the color its key names, a key the
            // table names no color for leaving it as it was (ShapeColorMap).
            for (size_t i = 0; i < hand.size(); ++i)
                if (picked[i])
                    if (const int color = table.color_of((shape_key)rule.value, hand[i]); color >= 0)
                        recolored(hand[i], color);
            break;
        case said::learned_table:
            // THE HOLLOWS FILLED BY SHAPE (ShapeToInteriorColorMap, every piece
            // picked): each piece's hollows (each_hollow) in the color its shape
            // names, a shape the table names no color for left as it was.
            if (table.claimed_by != shape_color_map_rule::interior_fill) return false;
            for (object& piece : hand) {
                const int fill = table.color_of(shape_key::shape, piece);
                if (fill < 0) continue;
                vector<std::pair<int, int>> hollows;
                each_hollow(piece, [&](int row, int col) { hollows.push_back({ row, col }); });
                for (const auto& [row, col] : hollows)
                    piece.data[row - piece.row][col - piece.col] = (int8_t)fill;
            }
            break;
        case said::size_rank: {
            // Ranked (ranked_by_size), the n-th in the palette's n-th color, and
            // every piece past the palette gone (SizeRankColorMap).
            const color_facts& palette = inf.colors;
            vector<object> ranked = ranked_by_size(hand, pair.input, inf.correspondence.segmentation);
            ranked.resize(min(ranked.size(), (size_t)max(palette.output_colors_by_size_count, 0)));
            for (size_t i = 0; i < ranked.size(); ++i) recolored(ranked[i], palette.output_colors_by_size[i]);
            hand = std::move(ranked);
            break;
        }
        case said::mirror_around: {
            const partner_pieces cut = facts.handed(which, pick);
            const auto mirror = (mirror_by)(rule.value / MIRROR_PARTNERS);
            const int how = rule.value % MIRROR_PARTNERS;
            vector<object> copies;
            for (int i = 0; i < cut.count(); ++i)
                if (cut.picked[i])
                    each_mirror(mirror, how, i, cut, pair.input.width, pair.input.height, [&](int pose, int row2, int col2) {
                        copies.push_back(mirrored(hand[i], pose, row2, col2));
                    });
            for (const object& copy : copies)
                if (hand.size() < (size_t)MAX_OBJECTS_IN_A_LIST) hand.push_back(copy);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

// WHAT A CHANGE OF THE HAND COMES TO, pair by pair (pair_tally): the hand
// drawn before and after it as the line lays it (draw_the_hand), a cell gained
// where the drawing changed to one that comes out (comes_out) and did not
// before, and wrong where it changed to one that does not. None where a pair
// does not lie over its output, or a hand cannot be drawn.
inline optional<vector<pair_tally>> hand_change(const replay_facts& before, const replay_facts& after) {
    if (!before.overlays) return nullopt;
    auto still = [](size_t) { return direction_step{ 0, 0 }; };
    vector<pair_tally> tallies(before.cuts().size());
    for (size_t which = 0; which < tallies.size(); ++which) {
        const figure_pair& pair = before.pairs[which];
        const grid& output = pair.whole.output;
        grid was(output.width, output.height), now(output.width, output.height);
        if (!draw_the_hand(was, before.cuts()[which], before.laid_under, /*clip=*/true, still) ||
            !draw_the_hand(now, after.cuts()[which], after.laid_under, /*clip=*/true, still))
            return nullopt;
        for (int row = 0; row < output.height; ++row)
            for (int col = 0; col < output.width; ++col) {
                if (now.data[row][col] == was.data[row][col]) continue;
                if (!comes_out(pair, row, col, now.data[row][col])) ++tallies[which].wrong;
                else tallies[which].gained += !comes_out(pair, row, col, was.data[row][col]);
            }
    }
    return tallies;
}

// RULES CARRIED ON A COPY OF THE HAND, JUDGED AS ONE (brings_nearer) by what
// they come to (hand_change): steps that read wrong alone -- a key recolored
// and then removed, a piece moved before its recolor -- are judged together.
inline bool hand_change_holds(const replay_facts& before, const replay_facts& after) {
    const auto tallies = hand_change(before, after);
    if (!tallies) return false;
    for (size_t which = 0; which < tallies->size(); ++which)
        if (!brings_nearer(before, which, (*tallies)[which])) return false;
    return true;
}

// Whether a change of the hand changed something in every pair not finished
// (replay_facts::finished): each rule of a chain judged as one is still seen
// at work everywhere.
inline bool hand_changes_everywhere(const replay_facts& before, const replay_facts& after) {
    const auto tallies = hand_change(before, after);
    if (!tallies) return false;
    for (size_t which = 0; which < tallies->size(); ++which)
        if (!before.finished(which) && (*tallies)[which].gained + (*tallies)[which].wrong == 0) return false;
    return true;
}

}  // namespace arc
