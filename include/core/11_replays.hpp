// How a word is checked: replayed on every pair with the arithmetic its op runs,
// and judged pair by pair (brings_nearer). One replay per word, and
// carry_the_hand, which carries the pieces through the words already said.
#pragma once
#include "10_picks_and_frames.hpp"
#include <algorithm>
#include <array>
#include <bitset>
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
// A cell past the edge of the largest grid -- a piece the words moved off it --
// is no cell left over, and is not marked.
struct cell_marks {
    vector<std::bitset<MAX_GRID_CELLS>> at;
    cell_marks() = default;
    explicit cell_marks(const cells_by_pair& cells) : at(cells.size()) {
        for (size_t which = 0; which < cells.size(); ++which)
            for (const auto& [row, col] : cells[which])
                if (row >= 0 && col >= 0 && row < MAX_GRID_DIM && col < MAX_GRID_DIM)
                    at[which].set((size_t)cell_index(row, col));
    }
};
inline bool on_cells(const cell_marks& cells, size_t which, int row, int col) {
    return which < cells.at.size() && cells.at[which].test((size_t)cell_index(row, col));
}

// WHAT IS DOWN, pair by pair, where a paint lands: the color each cell shows
// already, TRANSPARENT where nothing is -- here the pieces `facts` hands over,
// where they lie.
inline vector<cells> pieces_down(const replay_facts& facts) {
    vector<cells> down(facts.cuts().size());
    for (size_t which = 0; which < facts.cuts().size(); ++which)
        for (const object& piece : facts.cuts()[which])
            piece.each_cell([&](int row, int col, int8_t color) {
                if (facts.train[which].input.contains(row, col)) down[which][row][col] = color;
            });
    return down;
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
// pair by pair -- nothing, where there is no canvas -- and in the colors it
// draws, or all in one color -- the one the output shows where it first lands,
// the line's Recolor after the word, whatever the colors drawn -- or in the
// colors it draws where the output agrees, and else in one.
struct laying {
    int8_t lay = correspondence_election::fired_rule::laid_over;
    const vector<cells>* canvas = nullptr;
    enum colors : int8_t { as_drawn, in_one, own_else_one } in = as_drawn;
};

// WHAT A REPLAY SAW, where its word holds -- every replay below hands one back,
// or nothing where the word does not hold. The cells the word lays down right
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

// THE ONE JUDGE OF A WORD, PAIR BY PAIR (the user: a word thrown out for one
// wrong cell is thrown out too early). In a pair a word makes cells right --
// `gained`: the output shows what it laid there, and the cell did not show it
// before -- and lays cells wrong, where the output shows something else. It
// holds in that pair where it gains more than it lays wrong, so it brings the
// pair nearer its output, one cell gained at least: at work (replay_facts).
// Its wrong cells are left for a word said after it, or the next pass on the
// picture so far, to set right (bdad9b1f's two lines, each wrong only where
// they cross, the crossing a color of its own). Where the line does not draw
// its picture (replay_facts::tolerant), no pass could see a cell left wrong,
// and a word holds only where it lays none. A move or a removal is at work
// only where it changed a piece too (`acted`). In a finished pair
// (replay_facts::finished) there is nothing to gain: a word holds there where
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
// first and last, whether two of its own disagree there (torn), and what the
// picked pieces showed there before. Every way the paint is laid reads the
// walk (lay_paint), so a paint asked several ways is walked once, and laid on
// a new canvas without walking it again.
struct paint_walk {
    struct cell { int8_t row, col, first, last, was; bool torn; };
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
// over what is down, where it lands everywhere; else its canvas's pair, and
// nothing down where there is no canvas.
inline const cells* landing_on(const laying& how, size_t which) {
    using rule = correspondence_election::fired_rule;
    static const cells nothing_down;
    return how.lay == rule::laid_over ? nullptr : how.canvas ? &(*how.canvas)[which] : &nothing_down;
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
// each pair judged (brings_nearer): a cell is gained where the output shows
// what landed and the picked pieces did not show it there, and a cell its own
// paint laid twice in two colors is wrong whatever the output shows. Sees that
// one color (-1 laid in the colors drawn), the cells that landed right and the
// ones that landed wrong, and whether it picked every piece.
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
                if (const int color = facts.train[which].output.data[cell.row][cell.col]; is_color(color)) ++shown[color];
            });
        const int most = *ranges::max_element(shown);
        for (size_t which = 0; which < pairs && seen.paint < 0 && most > 0; ++which)
            each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t, bool) {
                if (const int color = facts.train[which].output.data[cell.row][cell.col];
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
        const grid& output = facts.train[which].output;
        pair_tally tally;
        each_landing(walk, which, how, [&](const paint_walk::cell& cell, int8_t color, bool torn) {
            const int8_t laid = how.in == laying::in_one ? seen.paint : color;
            if (!torn && output.data[cell.row][cell.col] == laid) {
                if constexpr (std::is_same_v<Seen, marked_paint>)
                    seen.drawn.at[which].set((size_t)cell_index(cell.row, cell.col));
                else seen.drawn[which].push_back({ cell.row, cell.col });
                tally.gained += cell.was != laid;
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
// many the output shows so and how many of those the picked pieces did not
// show already; and how many the output shows in each color, and of those how
// many the picked pieces showed so already (laid in one color). Each way the
// walk is laid on that canvas takes a sum of these (may_hold), so a lay that
// cannot hold is known without going over its cells again.
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
        const grid& output = facts.train[which].output;
        for (const paint_walk::cell& cell : walk.pairs[which]) {
            const int8_t down = which < canvas.size() ? canvas[which][cell.row][cell.col] : (int8_t)TRANSPARENT;
            const int at = down == (int8_t)TRANSPARENT ? 0 : 1 + down;
            const int shows = output.data[cell.row][cell.col];
            ++counted.cells[at];
            if (!cell.torn && shows == cell.last) {
                ++counted.right_last[at];
                counted.gained_last[at] += cell.was != cell.last;
            }
            if (shows == cell.first) {
                ++counted.right_first[at];
                counted.gained_first[at] += cell.was != cell.first;
            }
            if (is_color(shows)) {
                ++counted.shows[at][shows];
                counted.shown_before[at][shows] += cell.was == shows;
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
// cell the output shows in another; laid in one, a cell the output shows in
// another than where the first landed -- and else once as many cells are sure
// to be wrong as the pair could gain. THE MOST A PAIR COULD GAIN is every cell
// where the output shows anything but what the picked pieces show there; a
// cell first laid in a color the output does not show stays wrong -- laid
// again in another, it is torn -- and laid in one color, a cell the output
// shows bare is wrong whatever the color.
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
        const example& pair = facts.train[which];
        const partner_pieces& cut = handed[which];
        walk.every_piece = walk.every_piece && cut.every_piece_picked();
        // What the picked pieces show, where they lie: drawn again as it is, a
        // cell of them is no work.
        const auto& before = cut.shown(pair.input.width, pair.input.height);
        // Each cell's place in the walk, -1 for none yet: every cell, since a
        // piece the words moved may lay one past the grid's edge.
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
                cells.push_back({ (int8_t)row, (int8_t)col, color, color, before[row][col], false });
            } else {
                paint_walk::cell& cell = cells[(size_t)at];
                cell.torn = cell.torn || cell.last != color;
                cell.last = color;
            }
            if (!only || (down && !lands(only->lay, (*down)[row][col]))) return holds;
            // Under what is down, its own cell is down now: something, whatever its color.
            if (under) (*under)[row][col] = is_color(color) ? color : (int8_t)0;
            const int8_t shows = (int8_t)pair.output.data[row][col];
            if (exact) {
                if (only->in == laying::as_drawn) holds = shows == color;
                else {
                    if (first) first_shown = shows;
                    first = false;
                    holds = is_color(first_shown) && shows == first_shown;
                }
            } else if (fresh && (only->in == laying::in_one ? shows == (int8_t)TRANSPARENT : shows != color)) {
                if (could_gain < 0) {
                    could_gain = 0;
                    for (int r = 0; r < pair.output.height; ++r)
                        for (int c = 0; c < pair.output.width; ++c)
                            could_gain += pair.output.data[r][c] != before[r][c];
                }
                holds = ++sure_wrong < could_gain;
            }
            return holds;
        });
        if (!drew || !holds) return nullopt;
    }
    return walk;
}

// A PAINT LAID ON WHAT IS DOWN, replayed: what every replay of a word that
// draws asks, once -- its walk (walk_paint), laid the one way `how` says
// (lay_paint), done with as soon as it cannot hold so. Laid in its own colors
// or else in one, it is walked once and laid both ways: in its own colors where
// they lay no cell wrong, else whichever way says more, right cells less wrong
// ones, its own colors on a tie. Handed the pieces its pick takes
// (handed_pieces), made ready once for every way the word is asked.
template <class Draw>
inline optional<replayed> paint_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      const laying& how, Draw draw) {
    ++inference_work.paint_replays;
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
// each color picked piece i wears comes out in -- LEFT_AS_IT_IS where the word
// leaves it as it is (ColorFrom's piece with no partner; ShapeColorMap's key the
// table does not name; a color no key pairs) -- or answers false where the word
// cannot say; the piece is then a paint laid over itself in those colors
// (paint_holds), so a piece the word leaves as it is has to come out as it
// was: the word picked it, and does not get to leave it to another (4f537728's
// blocks in the marker's column, left by "takes the color of a piece in its
// rows", came out recolored). Sees the cells of the pieces it repainted and
// how many pieces changed color.
// Cell by cell: `cells_of(which, cut, i, repaint)` hands each cell of picked
// piece i to repaint(row, col, was, now), or answers false where the word
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

// COLORS THAT TRADE PLACES AS A KEY PAIRS THEM, replayed: BidiColorMap past its
// fixed pairs reads the key on each grid -- every piece not picked, along its
// rows or its columns (key_swaps) -- and every cell of a picked piece in a color
// the key pairs comes out in its partner's (repaints_hold). A key that names no
// swap leaves them as they were, so that pair shows the word at no work.
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
            for (int row = 0; row < piece.summary.height; ++row)
                for (int col = 0; col < piece.summary.width; ++col)
                    put(piece.summary.row + row, piece.summary.col + col,
                        traded_with_empty((int8_t)piece.data[row][col], color));
        }
        return true;
    });
}

// A PIECE THAT TAKES ITS COLOR FROM ANOTHER, replayed. ColorFrom paints every
// picked piece in the color of its partner among the rest, and leaves one with
// no partner as it is (recolors_hold: b7256dcd's shape with no dot at its side
// stays 6). A piece whose partners disagree the op leaves too, but the word
// cannot say it: which color the rule means there, no pair shows (d23f8c26's
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
// is, one with several the word cannot say.
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
// does not say, every piece is left as it is, so that pair shows the word at no
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
// so that pair shows the word at no work (recolors_hold).
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
// (SortBySize [0]). The n-th piece takes the palette's n-th color.
inline vector<object> ranked_by_size(const vector<object>& hand, const grid& input, segmentation_methods cut) {
    vector<object> ranked;
    if (cut != segmentation_methods::all_cells_of_one_color) ranked = hand;
    else
        for (const object& piece : hand) {
            grid layer(input.width, input.height);
            paint_piece(layer, piece, piece.summary.row, piece.summary.col, true, /*clip=*/true);
            object_walk_scratch scratch;
            each_object(layer, -1, grouping::touching_any_color, scratch,
                        [&](const object_summary& box, const int16_t* cells, int count) {
                ranked.push_back(object::from_cells(layer, box, cells, count));
            });
        }
    stable_sort(ranked.begin(), ranked.end(), [](const object& one, const object& other) {
        return one.summary.count > other.summary.count; });
    return ranked;
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
// drawing, not covered by a piece the line lays over it, and in the output --
// and leaves its cell right where nothing else now covers it and the output
// shows anything but the piece itself -- a cell still showing it was a copy
// left behind, not a move; what does show there is another word's to say
// (18286ef8's box closing behind its 9). Each cell otherwise is wrong. A right
// cell is gained where the drawing changed there and now shows what the output
// shows (brings_nearer, which its caller asks, with one piece at least moved).
// Adds the cells each one left and the cells it landed on, right, and the ones
// wrong to the pair's lists, and counts the pieces that moved.
inline bool slides_show(const example& pair, const partner_pieces& cut, bool under,
                        const vector<int>& down, const vector<int>& across, pair_tally& tally,
                        vector<std::pair<int, int>>& left, vector<std::pair<int, int>>& landed,
                        vector<cell_color>& wrong, int& pieces_moved) {
    grid drawn(pair.output.width, pair.output.height), was = drawn;
    if (!draw_the_hand(drawn, cut.pieces, under, /*clip=*/false,
                       [&](size_t i) { return direction_step{ down[i], across[i] }; }))
        return false;
    draw_the_hand(was, cut.pieces, under, /*clip=*/true, [](size_t) { return direction_step{ 0, 0 }; });
    auto gains = [&](int row, int col) {
        return drawn.data[row][col] != was.data[row][col] && drawn.data[row][col] == pair.output.data[row][col];
    };
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i] || (down[i] == 0 && across[i] == 0)) continue;
        cut.pieces[i].each_cell([&](int row, int col, int8_t color) {
            const int to_row = row + down[i], to_col = col + across[i];
            if (drawn.data[to_row][to_col] == color && pair.output.data[to_row][to_col] == color) {
                landed.push_back({ to_row, to_col });
                tally.gained += gains(to_row, to_col);
            } else {
                ++tally.wrong;
                wrong.push_back({ (int8_t)to_row, (int8_t)to_col, (int8_t)drawn.data[to_row][to_col] });
            }
            if (drawn.data[row][col] != TRANSPARENT || pair.output.data[row][col] != color) {
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

// PICKED PIECES SLID, EACH ITS OWN WAY, replayed: what every move word's replay
// asks, once. On each pair, `where(which, cut, i, by)` says how far picked piece
// i goes, down and across, or false where the word cannot say; the pieces then
// slide as slides_show reads them, and the word holds only where one piece at
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
        if (!slides_show(facts.train[which], cut, facts.laid_under, downs, acrosses, tally, seen.left[which],
                         seen.drawn[which], seen.wrong[which], seen.pieces))
            return nullopt;
        tally.acted = seen.pieces > before;
        if (!brings_nearer(facts, which, tally)) return nullopt;
    }
    if (seen.pieces == 0) return nullopt;
    return seen;
}

// A PIECE THAT MOVES LEVEL WITH ANOTHER, replayed. AlignTo slides every picked
// piece, down or across alone, until its box starts on the row -- or with
// `columns`, the column -- its one partner's box starts on (goes_level).
inline optional<replayed> align_holds(const replay_facts& facts, const vector<partner_pieces>& handed, bool columns,
                                      partner_by how) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, direction_step& by) {
        return cut.goes_level(i, columns, how, by);
    });
}
static_assert(row_of(node_type::AlignTo).high + 1 == 2 * (int)partner_by::count,
              "AlignTo takes rows then columns, each by every way partner_by finds a partner");


// A PIECE THAT MOVES TOWARD ANOTHER, replayed. MoveToward moves every picked
// piece toward its one partner of the kind `value` % PARTNER_KINDS names -- each
// way partner_by finds one, then each color -- by what `value` / PARTNER_KINDS
// names: one step, its own size, or until it touches it (goes_toward,
// move_toward in 15_op_math.hpp, the move the op makes).
inline optional<replayed> toward_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int value) {
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, direction_step& by) {
        const grid& output = facts.train[which].output;
        return cut.goes_toward(i, value, output.width, output.height, by);
    });
}
static_assert(row_of(node_type::MoveToward).high + 1 == (int)toward_by::count * PARTNER_KINDS,
              "MoveToward takes each way of going toward each kind of partner");

// A PICKED PIECE SLID THE SAME WAY IN EVERY PAIR, replayed: `down` and `across`,
// the MoveColumn and MoveRow a pick by holes or by size is said with.
inline optional<replayed> move_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int down,
                                     int across) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces&, int, direction_step& by) {
        by = { down, across };
        return true;
    });
}

// A PICKED PIECE SLID TO THE EDGE OF THE GRID, replayed: that way, until its box
// meets the edge (steps_to_edge, 15_op_math.hpp, the walk MoveToEdge takes).
inline optional<replayed> to_edge_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int move_direction) {
    const direction_step step = step_of(move_direction);
    return slides_hold(facts, handed, [&](size_t which, const partner_pieces& cut, int i, direction_step& by) {
        const grid& output = facts.train[which].output;
        const int steps = steps_to_edge(cut.boxes[i], output.width, output.height, move_direction);
        by = { steps * step.down, steps * step.across };
        return true;
    });
}

// A PICKED PIECE MOVED ITS OWN SIZE, replayed: the direction `move_direction` names
// (shift_by, 15_op_math.hpp, the step the op takes).
inline optional<replayed> shift_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int move_direction) {
    return slides_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i, direction_step& by) {
        by = shift_by(move_direction, cut.boxes[i].width(), cut.boxes[i].height());
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
        for (int row = 0; row < piece.summary.height; ++row)
            for (int col = 0; col < piece.summary.width; ++col)
                if (!keeps(index, row + piece.summary.row, col + piece.summary.col))
                    piece.data[row][col] = (int8_t)TRANSPARENT;
        paint_piece(drawn, piece, piece.summary.row, piece.summary.col, !facts.laid_under, true);
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
        const auto& output = facts.train[which].output;
        auto& one = removed.emplace_back(pieces.size(), true);
        for (size_t index = 0; index < pieces.size(); ++index)
            pieces[index].each_cell([&](int row, int col, int8_t color) {
                if (!output.contains(row, col) ||
                    (!paint_at(painted, which, row, col) && output.data[row][col] == color)) one[index] = false;
            });
        const grid kept = hand_after_drops(facts, which, [&](size_t index, int, int) { return !one[index]; });
        for (size_t index = 0; index < pieces.size(); ++index) {
            if (!one[index]) continue;
            pieces[index].each_cell([&](int row, int col, int8_t) {
                const auto above = paint_at(painted, which, row, col);
                if (!output.contains(row, col) || (above ? *above : kept.data[row][col]) != output.data[row][col])
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
// where the output shows it in the piece's own color, and a lost one where the
// output shows it bare -- the ground -- or a piece still there covers it: one
// not picked, or a picked one that keeps that cell. Each cell otherwise is
// wrong. A lost cell right is gained where the output shows something else
// there now, and in each pair one piece at least lost such a cell, and the
// word brings the pair nearer (brings_nearer). Sees the cells the picked pieces
// keep (drawn) and lose (left), right, the ones wrong, and how many pieces lost
// a cell that shows it.
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
        const example& pair = facts.train[which];
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
                if (!pair.output.contains(row, col)) { ++tally.wrong; return; }
                const int8_t there = (int8_t)pair.output.data[row][col];
                if (const auto above = paint_at(painted, which, row, col)) {
                    if (!keeps(cut, i, row, col) && *above == there)
                        seen.left[which].push_back({ row, col });
                    return;
                }
                if (keeps(cut, i, row, col)) {
                    if (there == color) seen.drawn[which].push_back({ row, col });
                    else {
                        ++tally.wrong;
                        seen.wrong[which].push_back({ (int8_t)row, (int8_t)col, color });
                    }
                    return;
                }
                if (there == after_picture.data[row][col]) {
                    seen.left[which].push_back({ row, col });
                    tally.gained += there != before_picture.data[row][col];
                    changed = changed || there != before_picture.data[row][col];
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
        return row >= 0 && col >= 0 && row < piece.summary.height && col < piece.summary.width &&
               piece.data[row][col] != (int8_t)TRANSPARENT;
    };
    rim.summary.count = 0;
    for (int row = 0; row < piece.summary.height; ++row)
        for (int col = 0; col < piece.summary.width; ++col) {
            if (!occupied(row, col)) continue;
            if (on_object_edge(row, col, occupied)) ++rim.summary.count;
            else rim.data[row][col] = (int8_t)TRANSPARENT;
        }
    return rim;
}

// A PIECE HOLLOWED TO ITS EDGE, replayed. Outline keeps the cells of each picked
// piece that have, among their eight neighbours, a cell that is not the piece's
// (on_object_edge, the test the op takes), and drops the rest (drops_hold).
inline optional<replayed> outline_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    return drops_hold(facts, handed, [](const partner_pieces& cut, int i, int row, int col) {
        const object& piece = cut.pieces[i];
        return on_object_edge(row - piece.summary.row, col - piece.summary.col, [&](int down, int across) {
            return down >= 0 && across >= 0 && down < piece.summary.height && across < piece.summary.width &&
                   piece.data[down][across] != (int8_t)TRANSPARENT;
        });
    });
}

// The object turned or flipped to `pose` in its own box, its top-left
// corner where it was: each cell read through pose_source, as the ops read it.
inline object posed(const object& piece, int pose) {
    object turned = piece;
    const auto [width, height] = posed_size(pose, piece.summary.width, piece.summary.height);
    turned.summary.width = (int8_t)width;
    turned.summary.height = (int8_t)height;
    turned.data.clear();
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const auto [from_row, from_col] = pose_source(pose, piece.summary.width, piece.summary.height, row, col);
            turned.data[row][col] = piece.data[from_row][from_col];
        }
    return turned;
}

// One copy of a piece, mirrored so.
inline object mirrored(const object& piece, int row2, int col2, bool down, bool across) {
    object copy = posed(piece, mirror_pose(down, across));
    const cell_box to = mirrored_box(piece.summary.box(), row2, col2, down, across);
    copy.summary.row = (int8_t)to.row0;
    copy.summary.col = (int8_t)to.col0;
    return copy;
}

// A PIECE DRAWN AGAIN MIRRORED AROUND ANOTHER'S MIDDLE, replayed. MirrorAround
// draws every picked piece again mirrored around the middle of its one partner
// among the rest -- or, for the last of MIRROR_PARTNERS, around the middle of the
// grid itself (mirror_middle) -- laid over what is down (paint_holds); a copy
// that falls off the grid cannot be drawn, and a row mirrored end to end onto
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
            int row2 = 0, col2 = 0;
            if (!mirror_middle(how, i, cut, input.width, input.height, row2, col2)) return false;
            cut.pieces[i].each_cell([&](int row, int col, int8_t) { kept[which].push_back({ row, col }); });
            bool on_grid = true;
            each_mirror(mirror, [&](bool down, bool across) {
                mirrored(cut.pieces[i], row2, col2, down, across).each_cell([&](int row, int col, int8_t color) {
                    if (!input.contains(row, col)) on_grid = false;
                    else put(row, col, color);
                });
            });
            if (!on_grid) return false;
            copies += mirror == mirror_by::four_ways ? 3 : 1;
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
              "MirrorAround takes each way of mirroring around each partner, then the grid");

// PIECES GROWN IN STEP UNTIL THEY MEET, walked: ExtendTo past its eight ways,
// every picked piece growing along its track at once (grow_in_step, the growth
// the op runs), each cell laid in the color of the piece that took it, and the
// bodies laid down again where they lie, as the op hands them back. Refused
// where a ray stops at a meeting but the output carries it on past (23581191).
inline optional<paint_walk> grown_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       grown_along track) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int width = output.width, height = output.height;
        // Each cell's piece, the first in the cut that holds it, as the op reads its list.
        int16_t owner[MAX_GRID_CELLS], turn[MAX_GRID_CELLS], queue[MAX_GRID_CELLS];
        uint8_t cell_directions[MAX_GRID_CELLS];
        std::fill_n(owner, width * height, (int16_t)-1);
        for (int i = 0; i < cut.count(); ++i)
            cut.pieces[i].each_cell([&](int row, int col, int8_t) {
                if (output.contains(row, col) && owner[row * width + col] < 0) owner[row * width + col] = (int16_t)i;
            });
        vector<array<int16_t, 3>> meetings;
        grow_in_step(width, height, track, [&](int piece) { return cut.is_picked(piece); }, owner, turn, queue, cell_directions,
                     [&](int at, int grow_direction, int piece) { meetings.push_back({ (int16_t)at, (int16_t)grow_direction, (int16_t)piece }); });
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
// until they meet (grown_walk).
inline optional<paint_walk> trail_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int grow_direction) {
    if (grow_direction >= direction::count) return grown_walk(facts, handed, (grown_along)(grow_direction - direction::count));
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
            const auto& pattern = cut.pieces[i].summary;
            for (int j = 0; j < cut.count(); ++j) {
                if (cut.picked[j]) continue;
                const auto& target = cut.pieces[j].summary;
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

// A PIECE REPEATED, walked: each picked piece laid again where its Tile lays it
// (each_tile_offset, 15_op_math.hpp, the arithmetic the op runs), each step read
// off the pieces the pair's cut hands over, a copy one step of the piece where
// a step is shorter (copy_size) -- walked (walk_paint) and laid every way the
// line asks (lay_paint). A piece no step moves is drawn where it lies, as the
// op hands it on. THE OP LAYS EVERY COPY INTO ITS ONE LIST and drops those past
// its room, so a pair whose pieces and copies would not fit in it cannot be
// drawn (8a371977's 3s, each repeated along its row: the later rows' copies
// were dropped).
inline optional<paint_walk> tile_walk(const replay_facts& facts, const vector<partner_pieces>& handed,
                                      const tile_value& tile) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        const int colors = color_count(cut.count(), [&](int i) { return (bool)cut.picked[(size_t)i]; },
                                       [&](int i) { return (uint32_t)palette(cut.pieces[(size_t)i]); });
        const cell_box key = key_box(cut);
        // The list's slots taken: every piece not picked rides through in one.
        int slots = (int)std::count(cut.picked.begin(), cut.picked.end(), false);
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const object& piece = cut.pieces[i];
            const auto& box = piece.summary;
            const tile_moves moves = moves_of(tile, measures_of(tile, box.height, box.width, [&](int row, int col) {
                return piece.data[row][col] != (int8_t)TRANSPARENT; }, colors, key));
            const auto [rows, columns] = copy_size(moves, box.height, box.width);
            bool laid = true;
            auto lay_copy = [&](int down, int across) {
                if (++slots > MAX_OBJECTS_IN_A_LIST) return laid = false;
                for (int row = 0; row < rows && laid; ++row)
                    for (int col = 0; col < columns && laid; ++col) {
                        const int8_t color = piece.data[row][col];
                        const int at_row = box.row + down + row, at_col = box.col + across + col;
                        if (color != (int8_t)TRANSPARENT && output.contains(at_row, at_col))
                            laid = put(at_row, at_col, color);
                    }
                return laid;
            };
            if (moves.count == 0) lay_copy(0, 0);
            else each_tile_offset(tile, moves, box.row, box.col, rows, columns, output.width, output.height, lay_copy);
            if (!laid) return false;
        }
        return true;
    });
}

// THE PICKED PIECES, AS ONE FIGURE, REPEATED FROM PAST THEMSELVES, walked:
// UnionList folds them into one figure and Trim fits its box to them, Shift
// moves it its own size the way `shift` names (shift_by, the step the op
// takes), and the Tile lays it again from there, as tile_walk lays a piece,
// each step read off the figure and the key, the pieces not picked (12422b43's
// rows beside the 5s, stacked under the figure every key height). Picking
// nothing, it draws nothing.
inline optional<paint_walk> past_walk(const replay_facts& facts, const vector<partner_pieces>& handed, int shift,
                                      const tile_value& tile) {
    return walk_paint(facts, handed, [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        cell_box figure;
        cells drawn;
        for (int i = 0; i < cut.count(); ++i)
            if (cut.picked[i]) {
                figure.include(cut.boxes[i]);
                cut.pieces[i].each_cell([&](int row, int col, int8_t color) { drawn[row][col] = color; });
            }
        if (figure.empty()) return false;
        const int colors = color_count(cut.count(), [&](int i) { return (bool)cut.picked[(size_t)i]; },
                                       [&](int i) { return (uint32_t)palette(cut.pieces[(size_t)i]); });
        const tile_moves moves = moves_of(tile, measures_of(tile, figure.height(), figure.width(), [&](int row, int col) {
            return drawn[figure.row0 + row][figure.col0 + col] != (int8_t)TRANSPARENT; }, colors, key_box(cut)));
        const auto [rows, columns] = copy_size(moves, figure.height(), figure.width());
        const direction_step by = shift_by(shift, figure.width(), figure.height());
        const int top = figure.row0 + by.down, left = figure.col0 + by.across;
        bool laid = true;
        auto lay_copy = [&](int down, int across) {
            for (int row = 0; row < rows && laid; ++row)
                for (int col = 0; col < columns && laid; ++col) {
                    const int8_t color = drawn[figure.row0 + row][figure.col0 + col];
                    const int at_row = top + down + row, at_col = left + across + col;
                    if (color != (int8_t)TRANSPARENT && output.contains(at_row, at_col)) laid = put(at_row, at_col, color);
                }
            return laid;
        };
        if (moves.count == 0) lay_copy(0, 0);
        else each_tile_offset(tile, moves, top, left, rows, columns, output.width, output.height, lay_copy);
        return laid;
    });
}

// EACH CELL THE RAYS OF A PAIR'S PICKED PIECES REACH on its output, `radius`
// cells long, from the middle of each piece's box -- up, down and to the
// sides, or along the diagonals (ray_seed_of and each_extended_ray,
// 15_op_math.hpp, the walk the op takes) -- handed to `visit(row, col, along)`,
// `along` steps out along its ray, in the order the op draws them.
template <class Visit>
inline void each_ray_cell(const replay_facts& facts, size_t which, const partner_pieces& cut, bool diagonal,
                          int radius, Visit visit) {
    const grid& output = facts.train[which].output;
    const vector<ray_seed>& seeds = cut.ray_seeds();
    for (int i = 0; i < cut.count(); ++i) {
        if (!cut.picked[i]) continue;
        const auto& box = cut.pieces[i].summary;
        const ray_seed seed = seeds[(size_t)i];
        each_extended_ray(seed, radius, diagonal, [&](int down, int across, int along) {
            const int row = box.row + seed.row + down, col = box.col + seed.col + across;
            if (output.contains(row, col)) visit(row, col, along);
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
        each_ray_cell(facts, which, cut, diagonal, radius, [&](int row, int col, int) { put(row, col, (int8_t)TRANSPARENT); });
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
    struct cell { int8_t row, col, was; };
    // Pair by pair, the cells the rays reach, nearest first: those first
    // reached `along` cells out are cells[starts[along]] up to cells[starts[along + 1]].
    struct of_pair { vector<cell> cells; vector<int> starts; };
    vector<of_pair> pairs;
    bool every_piece = true;
};
// The rays `reach` cells long fanned out, with what the picked pieces showed
// at each cell before; nothing where rays_walk walks nothing (walk_paint).
inline optional<ray_fan> rays_fan(const replay_facts& facts, const vector<partner_pieces>& handed, bool diagonal,
                                  int reach) {
    if (!facts.overlays || facts.train.empty()) return nullopt;
    ray_fan fan;
    fan.pairs.resize(facts.train.size());
    for (size_t which = 0; which < facts.train.size(); ++which) {
        const example& pair = facts.train[which];
        const partner_pieces& cut = handed[which];
        fan.every_piece = fan.every_piece && cut.every_piece_picked();
        const auto& before = cut.shown(pair.input.width, pair.input.height);
        // How far along the nearest ray reaches each cell, 0 where none does.
        int8_t nearest[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        each_ray_cell(facts, which, cut, diagonal, reach, [&](int row, int col, int along) {
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
                    fanned.cells[(size_t)next[(size_t)along]++] = { (int8_t)row, (int8_t)col, before[row][col] };
    }
    return fan;
}

// EVERY LENGTH OF THE RAYS LAID AT ONCE, `how` says, in one color, as lay_paint
// lays each length's own walk: adding the cells one step further out, length
// by length, it counts what lands (landing_on, lands) and what the output
// shows there, takes the color shown most and judges each pair (brings_nearer)
// -- the cells laid right, as marks, and how many are laid wrong. A length
// whose cells tie for the color is laid off its own walk (lay_paint), and a
// length that holds keeps its walk for its lists (heard::materialize):
// `walk_of(length)` hands it over, walked once. Lengths 0 to `reach`.
template <class WalkOf>
inline vector<optional<marked_paint>> lay_every_length(const replay_facts& facts, const ray_fan& fan,
                                                       const laying& how, int reach, WalkOf walk_of) {
    // What has landed so far on a pair: how many cells, of those how many the
    // output shows in each color, how many the picked pieces showed so already,
    // and where.
    struct landed_so_far {
        int cells = 0, shows[COLOR_COUNT] = {}, shown_before[COLOR_COUNT] = {};
        std::bitset<MAX_GRID_CELLS> right[COLOR_COUNT];
    };
    const size_t pairs = fan.pairs.size();
    vector<landed_so_far> landed(pairs);
    vector<optional<marked_paint>> laid((size_t)reach + 1);
    for (int length = 1; length <= reach; ++length) {
        for (size_t which = 0; which < pairs; ++which) {
            const ray_fan::of_pair& fanned = fan.pairs[which];
            if ((size_t)length + 1 >= fanned.starts.size()) continue;
            const grid& output = facts.train[which].output;
            const cells* down = landing_on(how, which);
            landed_so_far& so_far = landed[which];
            for (int at = fanned.starts[(size_t)length]; at < fanned.starts[(size_t)length + 1]; ++at) {
                const ray_fan::cell& cell = fanned.cells[(size_t)at];
                if (down && !lands(how.lay, (*down)[cell.row][cell.col])) continue;
                ++so_far.cells;
                const int shows = output.data[cell.row][cell.col];
                if (!is_color(shows)) continue;
                ++so_far.shows[shows];
                so_far.shown_before[shows] += cell.was == shows;
                so_far.right[shows].set((size_t)cell_index(cell.row, cell.col));
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
// top-left corner where it was. Laid over what is down (paint_holds), and the
// figure itself is down where it lies, so the output must show it there too
// wherever the turned copy does not cover it: the copy and the figure are drawn,
// both, since the matching read the figure turned (11852cab's design, the same
// from all four sides). Picking nothing, it draws nothing.
inline optional<replayed> turned_holds(const replay_facts& facts, const vector<partner_pieces>& handed, int pose) {
    return paint_holds(facts, handed, {}, [&](size_t which, const partner_pieces& cut, auto&& put) {
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
// and the pose's words lay it around the grid's middle. A pose that trades
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

// A PIECE'S BOX FILLED, replayed. AddColor paints every cell of each picked
// piece's box in its one color -- laid under what is down (paint_holds), so it
// fills only what the piece and every paint before it leave empty.
inline optional<replayed> box_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                    const vector<cells>& canvas) {
    using rule = correspondence_election::fired_rule;
    return paint_holds(facts, handed, { rule::laid_under, &canvas, laying::in_one },
                       [&](size_t, const partner_pieces& cut, auto&& put) {
        for (int i = 0; i < cut.count(); ++i) {
            if (!cut.picked[i]) continue;
            const cell_box& box = cut.boxes[i];
            for (int row = box.row0; row <= box.row1; ++row)
                for (int col = box.col0; col <= box.col1; ++col) put(row, col, (int8_t)TRANSPARENT);
        }
        return true;
    });
}

// WHICH ROOM A ROOM PAINT FILLS (filled_room's value), in the order a tie goes:
// every room (d511f180's ground, all of it 8); by size, ranked among the rooms
// alone, as SelectLargestObject and SelectSmallestObject rank what is left once
// every other piece is deleted -- the largest, the smallest, the 2nd largest,
// the 2nd smallest... -- then the room of the part the picked pieces fence
// off, numbered as SelectPartition numbers the parts.
// Past those, every room at once, each in the color its own key names -- its
// shape or its symmetry (shape_key) -- as a table read off the pairs says
// (rooms_table). Last, the ground's places shaped like the shape the pairs'
// changes take, every one (shaped_rooms).
constexpr int ROOM_EVERY = 0, ROOM_SIZES = 1, ROOM_PARTS = ROOM_SIZES + 2 * SIZE_RANKS,
              ROOM_CHOICES = ROOM_PARTS + row_of(node_type::SelectPartition).high + 1;
constexpr int ROOM_TABLE = ROOM_CHOICES, ROOM_VALUES = ROOM_TABLE + (int)shape_key::count, ROOM_SHAPE = ROOM_VALUES;
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
// pieces leave, the ground the plan deleted, which Partition cuts into rooms
// before it is deleted. In the order Partition hands them back
// (each_fenced_part, the op's own walk), on the canvas the pieces lie on,
// as Partition reads it.
inline vector<vector<fenced_room>> rooms_fenced(const replay_facts& facts, const vector<partner_pieces>& handed,
                                               const vector<grid>& before_ground) {
    vector<vector<fenced_room>> rooms(handed.size());
    for (size_t which = 0; which < handed.size() && which < facts.train.size(); ++which) {
        const tile_size canvas = facts.canvas(which);
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
                            if (which < before_ground.size() && before_ground[which].contains(row, col) &&
                                before_ground[which].data[row][col] == color &&
                                facts.train[which].input.data[row][col] == TRANSPARENT && in_part(row, col))
                                room.cells.push_back((int16_t)cell_index(row, col));
                    if (!room.cells.empty()) rooms[which].push_back(std::move(room));
                }
            });
    }
    return rooms;
}

// THE GROUND'S PLACES SHAPED LIKE THE SHAPE THE CHANGES TAKE, on each pair:
// every place the shape fits (each_place_it_fits) on the color it turns from
// or into, on the input before its ground is deleted, its every cell ground
// there -- the places Split [4] cuts off the ground's layer, as Partition cuts
// rooms off it, overlapping where they do.
inline vector<vector<fenced_room>> shaped_rooms(const replay_facts& facts, const turning_shape& shape,
                                                const vector<grid>& before_ground) {
    vector<vector<fenced_room>> rooms(facts.train.size());
    for (size_t which = 0; which < facts.train.size() && which < before_ground.size() && shape.holds(); ++which) {
        const grid& figures = facts.train[which].input;
        each_place_it_fits(before_ground[which], shape.cells, {}, [&](int row, int col, int color) {
            if (color != shape.from && color != shape.into) return;
            fenced_room room;
            bool ground = true;
            shape.cells.each_cell([&](int down, int across, int8_t) {
                ground = ground && figures.data[row + down][col + across] == TRANSPARENT;
                room.cells.push_back((int16_t)cell_index(row + down, col + across));
            });
            if (ground) rooms[which].push_back(std::move(room));
        });
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
    const bool largest = room_largest(chosen);
    const size_t rank = (size_t)room_rank(chosen);
    if (rank >= rooms.size()) return nullptr;
    vector<const fenced_room*> ranked;
    for (const fenced_room& room : rooms) ranked.push_back(&room);
    ranges::stable_sort(ranked, [&](const fenced_room* one, const fenced_room* other) {
        return largest ? one->cells.size() > other->cells.size() : one->cells.size() < other->cells.size();
    });
    const size_t size = ranked[rank]->cells.size();
    const bool tied = (rank > 0 && ranked[rank - 1]->cells.size() == size) ||
                      (rank + 1 < ranked.size() && ranked[rank + 1]->cells.size() == size);
    return tied ? nullptr : ranked[rank];
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

// THE ROOMS' TABLE, read off the pairs (read_off) as the shape word reads its
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
inline optional<replayed> bridge_holds(const replay_facts& facts, const vector<partner_pieces>& handed,
                                       const vector<cells>& canvas, int way) {
    using rule = correspondence_election::fired_rule;
    bool turned = false;
    auto seen = paint_holds(facts, handed, { rule::laid_under, &canvas, laying::own_else_one },
                            [&](size_t which, const partner_pieces& cut, auto&& put) {
        const grid& output = facts.train[which].output;
        // The bars take one place more in the pieces' list, all of them one drawing.
        if (cut.count() >= MAX_OBJECTS_IN_A_LIST) return false;
        turned = bridge_bars(way, output.width, output.height, cut,
                             [&](const cell_box& bar, int8_t color) {
            for (int row = bar.row0; row <= bar.row1; ++row)
                for (int col = bar.col0; col <= bar.col1; ++col)
                    if (output.contains(row, col)) put(row, col, color);
            return true;
        }) || turned;
        return true;
    });
    if (way >= BRIDGE_TO_PARTNER && !turned) return nullopt;
    return seen;
}

// THE CELLS A PIECE ENCLOSES: those its own paint walls off from the edge of its
// box (mark_outside, the four-way flood the ops flood), each handed to
// `see(row, col)` where it lies.
template <class See>
inline void each_hollow(const object& piece, See see) {
    const int height = piece.summary.height, width = piece.summary.width;
    auto occupied = [&](int row, int col) { return piece.data[row][col] != (int8_t)TRANSPARENT; };
    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
    mark_outside(height, width, occupied, outside);
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            if (!occupied(row, col) && !outside[row][col]) see(piece.summary.row + row, piece.summary.col + col);
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
// Laid over what is down, the edge as it was: bb43febb's blocks, red inside a
// gray rim; 694f12f3's, red inside the larger and blue inside the smaller.
inline optional<replayed> inside_holds(const replay_facts& facts, const vector<partner_pieces>& handed) {
    int8_t inside = (int8_t)TRANSPARENT;
    for (size_t which = 0; which < handed.size() && inside == (int8_t)TRANSPARENT; ++which)
        for (int i = 0; i < handed[which].count() && inside == (int8_t)TRANSPARENT; ++i)
            if (handed[which].picked[i])
                each_hollow(outlined(handed[which].pieces[i]), [&](int row, int col) {
                    if (inside == (int8_t)TRANSPARENT && facts.train[which].output.contains(row, col))
                        inside = (int8_t)facts.train[which].output.data[row][col];
                });
    if (!is_color(inside)) return nullopt;
    auto seen = paint_holds(facts, handed, {}, [&](size_t, const partner_pieces& cut, auto&& put) {
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
    const vector<cells> hand = pieces_down(facts);
    int pieces_filled = 0;
    auto seen = paint_holds(facts, handed_pieces(facts, piece_pick::all()), { rule::laid_under, &hand },
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

// THE PIECES AS THE LINE LEAVES THEM. Each word on the line acts on the hand the
// words before it left, as the program runs them, so a word asked after them is
// replayed on that hand and not on the cut: the pieces it picks, where they
// stand and what they wear. Carried here pair by pair, one word at a time, with
// the arithmetic its op runs -- the pick read off the hand as it is, each
// partner found before any piece is touched, copies joining the end of the list.
// A word that reads a table off the pairs is one stage more: the puzzle's shape
// table and its palette by size are the reading's (inference). False for
// anything else, after which the words below are asked the way they were
// before the hand was carried.
inline bool carry_the_hand(const correspondence_election::fired_rule& rule, const inference& inf,
                           replay_facts& facts) {
    using said = correspondence_election::fired_rule;
    const shape_color_map_rule& table = inf.shape_color_map;
    // A KEY'S SHAPE NAMING THE FIGURE'S COLOR, carried as beats_of says it, in
    // three words: the key, the smallest, takes the color its own shape names;
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
        piece.summary.row = (int8_t)(piece.summary.row + down);
        piece.summary.col = (int8_t)(piece.summary.col + across);
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
                int count = 0;
                for (int row = 0; row < piece.summary.height; ++row)
                    for (int col = 0; col < piece.summary.width; ++col) {
                        int8_t& cell = piece.data[row][col];
                        cell = traded_with_empty(cell, rule.value - EMPTY_PAIRS);
                        count += cell != (int8_t)TRANSPARENT;
                    }
                piece.summary.count = (int16_t)count;
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
                            if (is_color(color)) piece.data[row - piece.summary.row][col - piece.summary.col] = (int8_t)color;
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
                    piece.data[row - piece.summary.row][col - piece.summary.col] = (int8_t)fill;
                piece.summary.count = (int16_t)(piece.summary.count + (int)hollows.size());
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
            for (int i = 0; i < cut.count(); ++i) {
                if (!cut.picked[i]) continue;
                int row2 = 0, col2 = 0;
                if (!mirror_middle(how, i, cut, pair.input.width, pair.input.height, row2, col2)) continue;
                each_mirror(mirror, [&](bool down, bool across) {
                    copies.push_back(mirrored(hand[i], row2, col2, down, across));
                });
            }
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

}  // namespace arc
