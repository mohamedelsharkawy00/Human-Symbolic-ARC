// What each op does to the cells, said once. The op calls it (src/ops.cpp),
// and the reading replays the op by calling the same (7_infer.hpp), so the two
// cannot draw differently. A word's arithmetic goes here before the word is said.
// Grouped as 8_vocabulary.hpp groups the ops: reduce, select, translate, map,
// generate, after what several of them share.
#pragma once
#include "1_readings.hpp"
#include "3_background.hpp"  // mark_outside
#include "8_vocabulary.hpp"  // color_pair, BidiColorMap's key values
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace arc {
using namespace std;

// SHARED · partners (ColorFrom, AlignTo, MoveToward,
// MirrorAround, WouldFit, SelectLikeKey) and bands (Layout, SelectRow,
// SelectColumn)

// WHO A PIECE TAKES ITS COLOR FROM: another piece of the same grid, found by how
// the two sit. The rule that reads it off the pairs and the op that does it
// both ask here, so the two cannot disagree about which piece is the partner.
// NUMBERED IN THE ORDER THE WAYS ARE ASKED IN, so the order a tie between two
// that say as much is broken in: the surer a way names one partner, the
// sooner. The piece around it or inside it first -- one box holds another --
// then the piece lined up with it, in its rows or its columns, which a second
// piece in the same band leaves without one (18419cfa's test shape, under two
// frames), then the nearest, which ties as soon as two stand as far (ddf7fa4f's
// test block, between two dots), then the piece it touches at a side, the
// four-connected touch, before the one it touches at a side or a corner,
// four-connected before eight until eight is needed (b7256dcd's pair meets the
// 3 at a side and the 4 only at a corner; 4938f0c2's shape meets its block only
// at a corner), then one of its shape (same_shape, 1_readings.hpp): as it
// lies or turned any of the eight ways round, then the same at any size, then
// stretched, each as it lies before turned.
enum class partner_by : int8_t { around_it, inside_it, same_row, same_column, nearest, touching_sides,
                                 touching, same_shape, same_shape_turned, same_shape_scaled,
                                 same_shape_scaled_turned, same_shape_stretched, same_shape_stretched_turned,
                                 count };

// THE LIKENESSES OF SHAPE, the partner ways from same_shape on, in their
// order: as it lies, then turned, at its own size, at any size, stretched --
// what each lets a shape change (shape_freedom), read off how far past
// same_shape it is.
constexpr int SHAPE_LIKENESSES = (int)partner_by::count - (int)partner_by::same_shape;
constexpr shape_freedom likeness_freedom(int like) {
    return { (like & 1) != 0, like / 2 == 1, like / 2 == 2 };
}
// The likeness a partner way is, -1 for a way that is none.
constexpr int likeness_of(partner_by how) {
    return how >= partner_by::same_shape && how < partner_by::count ? (int)how - (int)partner_by::same_shape : -1;
}
// Whether a likeness lets size change: at any size, or stretched.
constexpr bool sized_likeness(int like) { return like >= 2; }
static_assert(SHAPE_LIKENESSES == 6 && (int)partner_by::same_shape_stretched_turned == (int)partner_by::count - 1,
              "the likenesses are the last partner ways: as it lies or turned, at its size, any size, stretched");

// The gap between two boxes, the empty rows plus the empty columns between
// them; 0 when they touch or overlap.
inline int box_gap(const cell_box& one, const cell_box& other) {
    const int rows = one.row0 > other.row1 ? one.row0 - other.row1 - 1
                   : other.row0 > one.row1 ? other.row0 - one.row1 - 1 : 0;
    const int cols = one.col0 > other.col1 ? one.col0 - other.col1 - 1
                   : other.col0 > one.col1 ? other.col0 - one.col1 - 1 : 0;
    return rows + cols;
}

// WHETHER THE BOX `to` LIES WHOLLY THAT WAY OF THE BOX `from`: past its last
// row below it, or above it, past its last column to its right, or its left --
// a corner way, both of its two sides. SelectPartner's side of a partner: a
// partner below a piece stays below it however far either slides across
// (4364c1c4's stacks, each upper piece a step left, each lower one right).
inline bool lies_toward(const cell_box& from, const cell_box& to, int partner_direction) {
    const direction_step step = step_of(partner_direction);
    return (step.down == 0 || (step.down > 0 ? to.row0 > from.row1 : to.row1 < from.row0)) &&
           (step.across == 0 || (step.across > 0 ? to.col0 > from.col1 : to.col1 < from.col0));
}

// WHETHER THE PIECE `mine`, MOVED `by`, TOUCHES THE PIECE `other`: a cell of it
// on a cell of the other, or beside one at a side -- or, unless `sides_only`,
// at a corner. The pieces are asked as each_partner asks them, below.
template <class Pieces>
inline bool touches(const Pieces& pieces, int mine, int other, direction_step by, bool sides_only) {
    const cell_box own = pieces.box(mine);
    for (int row = own.row0; row <= own.row1; ++row)
        for (int col = own.col0; col <= own.col1; ++col) {
            if (!pieces.painted(mine, row, col)) continue;
            const int at_row = row + by.down, at_col = col + by.across;
            if (pieces.painted(other, at_row, at_col)) return true;
            for (int neighbor_direction = 0; neighbor_direction < direction::count; neighbor_direction += sides_only ? every_nth_direction<4> : every_nth_direction<8>)
                if (pieces.painted(other, at_row + step_of(neighbor_direction).down, at_col + step_of(neighbor_direction).across)) return true;
        }
    return false;
}

// Whether pieces mine and other are one shape, as the likeness `like` lets them differ.
template <class Pieces>
inline bool shapes_alike(const Pieces& pieces, int mine, int other, int like) {
    const cell_box own = pieces.box(mine), box = pieces.box(other);
    const bool sized = sized_likeness(like);
    return same_shape_in_blocks(
        own.width(), own.height(),
        [&](int row, int col) { return pieces.color_at(mine, own.row0 + row, own.col0 + col); },
        sized ? pieces.blocks(mine) : shape_blocks{}, box.width(), box.height(),
        [&](int row, int col) { return pieces.color_at(other, box.row0 + row, box.col0 + col); },
        sized ? pieces.blocks(other) : shape_blocks{}, likeness_freedom(like));
}

// Every partner the piece `mine` has by `how` among the pieces not picked --
// or, `among_all`, among every piece but itself (SelectPartner, whose pick of
// every piece leaves none unpicked) -- handed to `visit(other)`; for the
// nearest, only the ones no other is nearer than. THE PIECES ARE ASKED, so one
// walk serves a grid's pieces and the op's alike: `count()`; `box(i)`, the box
// piece i's paint fills on the grid; `painted(i, row, col)`, whether it paints
// that cell of the grid, false off it; `is_picked(i)`; `touching(i, j, sides_only)`
// and `alike(i, j, like)` (touches, shapes_alike), which a reading may keep; and,
// for the color questions below, `color(i)`, the color an op reads off its first
// cell, and `holds(i, color)`, whether it holds a cell of that color.
template <class Pieces, class Visit>
inline void each_partner(partner_by how, int mine, const Pieces& pieces, Visit visit,
                         bool among_all = false) {
    const cell_box own = pieces.box(mine);
    if (own.empty()) return;
    auto may_partner = [&](int other) { return other != mine && (among_all || !pieces.is_picked(other)); };
    int nearest = 1 << 30;
    if (how == partner_by::nearest)
        for (int other = 0; other < pieces.count(); ++other) {
            if (!may_partner(other)) continue;
            const cell_box box = pieces.box(other);
            if (box.empty()) continue;
            const int gap = box_gap(own, box);
            if (gap < nearest) nearest = gap;
        }
    for (int other = 0; other < pieces.count(); ++other) {
        if (!may_partner(other)) continue;
        const cell_box box = pieces.box(other);
        if (box.empty()) continue;
        bool partner = false;
        switch (how) {
            case partner_by::nearest:     partner = box_gap(own, box) == nearest; break;
            case partner_by::same_row:    partner = own.row0 <= box.row1 && box.row0 <= own.row1; break;
            case partner_by::same_column: partner = own.col0 <= box.col1 && box.col0 <= own.col1; break;
            case partner_by::around_it:
                partner = box.row0 <= own.row0 && box.col0 <= own.col0 && own.row1 <= box.row1 && own.col1 <= box.col1;
                break;
            case partner_by::inside_it:
                partner = own.row0 <= box.row0 && own.col0 <= box.col0 && box.row1 <= own.row1 && box.col1 <= own.col1;
                break;
            case partner_by::touching:
            case partner_by::touching_sides:
                partner = pieces.touching(mine, other, how == partner_by::touching_sides);
                break;
            case partner_by::same_shape:
            case partner_by::same_shape_turned:
            case partner_by::same_shape_scaled:
            case partner_by::same_shape_scaled_turned:
            case partner_by::same_shape_stretched:
            case partner_by::same_shape_stretched_turned:
                partner = pieces.alike(mine, other, (int)how - (int)partner_by::same_shape);
                break;
            default: break;
        }
        if (partner) visit(other);
    }
}

// The color the piece `mine` takes from its partners: the one color every
// partner wears, or -1 when it has none or they disagree.
template <class Pieces>
inline int partner_color(partner_by how, int mine, const Pieces& pieces) {
    int color = -1;
    bool clash = false;
    each_partner(how, mine, pieces, [&](int other) {
        const int worn = pieces.color(other);
        if (color < 0) color = worn;
        else if (worn != color) clash = true;
    });
    return clash ? -1 : color;
}

// The piece `mine`'s one partner, or -1 when it has none or several: a place
// is read off one piece, never agreed among many.
template <class Pieces>
inline int partner_of(partner_by how, int mine, const Pieces& pieces) {
    int found = -1, partners = 0;
    each_partner(how, mine, pieces, [&](int other) {
        found = other;
        ++partners;
    });
    return partners == 1 ? found : -1;
}

// Whether the piece `mine` has a partner by `how` at all, one or several, among
// the pieces not picked: what a piece like another asks (like_the_key), where
// partner_of asks for the one.
template <class Pieces>
inline bool any_partner(partner_by how, int mine, const Pieces& pieces) {
    bool any = false;
    each_partner(how, mine, pieces, [&](int) { any = true; });
    return any;
}

// A PARTNER OF ANY KIND: the ways partner_by finds one by how two pieces sit,
// then, past those, by color -- the one piece holding a cell of color `kind -
// partner_by::count` (18286ef8's 9 steps toward the 6, wherever the 6 is).
// Asked among the pieces not picked, as partner_of asks. -1 when there is
// none, or several.
constexpr int PARTNER_KINDS = (int)partner_by::count + COLOR_COUNT;
template <class Pieces>
inline int partner_of_kind(int kind, int mine, const Pieces& pieces) {
    if (kind < (int)partner_by::count) return partner_of((partner_by)kind, mine, pieces);
    int found = -1;
    for (int other = 0; other < pieces.count(); ++other) {
        if (other == mine || pieces.is_picked(other) || !pieces.holds(other, kind - (int)partner_by::count)) continue;
        if (found >= 0) return -1;
        found = other;
    }
    return found;
}

// Share the executor's span grouping between inference and Layout/SelectRow/SelectColumn.
inline int overlapping_bands(int count, const int16_t* lows,
                                               const int16_t* highs, int8_t* band_of,
                                               int* order) {
    for (int i = 0; i < count; ++i) order[i] = i;
    for (int i = 1; i < count; ++i) {
        int j = i;
        while (j > 0) {
            const int a = order[j], b = order[j - 1];
            if (lows[a] >= lows[b]) break;
            order[j] = b;
            order[j - 1] = a;
            --j;
        }
    }
    int bands = 0, band_hi = -1;
    for (int m = 0; m < count; ++m) {
        const int i = order[m];
        const int need = (highs[i] - lows[i] + 2) / 2;
        if (bands == 0 || band_hi - lows[i] + 1 < need) {
            ++bands;
            band_hi = highs[i];
        } else if (highs[i] > band_hi) {
            band_hi = highs[i];
        }
        band_of[i] = (int8_t)(bands - 1);
    }
    return bands;
}

// REDUCE · Outline, Fit

// The boundary used by Outline: any missing eight-way neighbour makes an edge.
// The caller supplies occupancy, including false outside its own frame.
template <class Occupied>
inline bool on_object_edge(int row, int col, Occupied occupied) {
    for (int neighbor_direction = 0; neighbor_direction < direction::count; ++neighbor_direction)
        if (!occupied(row + step_of(neighbor_direction).down, col + step_of(neighbor_direction).across)) return true;
    return false;
}

// WHAT Fit STRETCHES TO, its value: the input's frame, the canvas the output is
// drawn on, the box of what is painted on it -- none where nothing is -- or one
// pane's box at the canvas's corner, `pane` the size every pane of the input
// shares -- none where they differ or there are none. Reading and execution
// ask the same helper.
enum class fit_to : int8_t { input, canvas, painted, pane, count };
inline cell_box fit_frame(int to, int input_width, int input_height, const grid& canvas, tile_size pane) {
    if (to == (int)fit_to::input) return {0, 0, input_height - 1, input_width - 1};
    if (to == (int)fit_to::canvas) return {0, 0, canvas.height - 1, canvas.width - 1};
    if (to == (int)fit_to::pane)
        return pane.width > 0 && pane.height > 0 ? cell_box{0, 0, pane.height - 1, pane.width - 1} : cell_box{};
    cell_box box;
    if (to == (int)fit_to::painted)
        for (int r = 0; r < canvas.height; ++r)
            for (int c = 0; c < canvas.width; ++c)
                if (is_color(canvas.data[r][c])) box.include_cell(r, c);
    return box;
}
inline int fit_source_cell(int target_cell, int source_size, int target_size) {
    return target_cell * source_size / target_size;
}

// SELECT · SelectUnique, SelectAtLeastCells, SelectObjectsWithHoles,
// SelectRowFromMiddle, WouldFit, SelectPartition, SelectLikeKey, SelectPartner

// WHETHER SelectAtLeastCells[at_least] PICKS A PIECE: it paints at least that
// many cells of its `height` by `width` box. Counted here for the op and for
// the reading that replays it, so the two cannot pick different pieces.
template <class Occupied>
inline bool paints_at_least(int height, int width, Occupied occupied, int at_least) {
    int cells = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) cells += occupied(row, col) ? 1 : 0;
    return cells >= at_least;
}

// Count four-connected enclosed spaces in an occupancy mask: the runs of empty
// cells no four-way walk in from the edge of the box reaches (mark_outside).
// Shared by inference and SelectObjectsWithHoles so learned keys mean the same
// at runtime.
template <class Occupied>
inline int object_hole_count(int height, int width, Occupied occupied) {
    if (width <= 0 || height <= 0) return 0;
    // Walked: outside, or in a hole already counted.
    bool walked[MAX_GRID_DIM][MAX_GRID_DIM];
    mark_outside(height, width, occupied, walked);
    auto claim = [&](int row, int col) {
        if (walked[row][col] || occupied(row, col)) return false;
        walked[row][col] = true;
        return true;
    };
    int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
    int holes = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            if (!claim(row, col)) continue;
            ++holes;
            queue_row[0] = (int16_t)row;
            queue_col[0] = (int16_t)col;
            walk_out<4>(height, width, queue_row, queue_col, 1, claim, [](int, int) {});
        }
    return holes;
}

// WHAT A PIECE COUNTS, the measure a pick of the most, the fewest or the odd
// one out reads (the user, 2026-10-13: the reading could not count): its
// cells, its colors, its holes (object_hole_count), or its cells of one
// color, of_color + the color. color_at(row, col) reads the piece in its box,
// TRANSPARENT where it paints none. Said once for the ops and the reading.
enum class count_kind : int8_t { cells, colors, holes, of_color, count = of_color + COLOR_COUNT };
template <class ColorAt>
inline int count_of(int kind, int height, int width, ColorAt color_at) {
    if (kind == (int)count_kind::holes)
        return object_hole_count(height, width, [&](int row, int col) { return color_at(row, col) != TRANSPARENT; });
    uint32_t colors = 0;
    int cells = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int color = color_at(row, col);
            if (!is_color(color)) continue;
            colors |= 1u << color;
            cells += kind == (int)count_kind::cells || color == kind - (int)count_kind::of_color;
        }
    if (kind != (int)count_kind::colors) return cells;
    int many = 0;
    for (; colors; colors >>= 1) many += (int)(colors & 1);
    return many;
}

// THE RANKS BY SIZE SelectLargestObject and SelectSmallestObject take, the
// n-th piece by its cells; past them, every piece with the most -- or the
// fewest -- of what count_kind names, value - SIZE_RANKS, ties and all.
constexpr int SIZE_RANKS = 10;

// EVERY ONE OF `count` WITH THE MOST -- or the fewest -- of what value(i)
// counts, handed to pick(i), ties and all; none where every one counts the
// same, which tells none apart. Said once for the op and for the reading.
template <class Value, class Pick>
inline void each_extreme(int count, bool most, Value value, Pick pick) {
    if (count <= 0) return;
    int best = value(0), other = best;
    for (int i = 1; i < count; ++i) {
        const int one = value(i);
        if (most ? one > best : one < best) best = one;
        if (most ? one < other : one > other) other = one;
    }
    if (best == other) return;
    for (int i = 0; i < count; ++i)
        if (value(i) == best) pick(i);
}

// WHAT MAKES TWO PIECES TWINS, SelectUnique's value: the same drawing,
// colors and all [0]; the same shape as it lies [1] or any way round [2],
// colors aside (same_shape); the same colors, wherever they lie [3]; or past
// those, the same count of what count_kind names, by_count + the kind.
enum class twin_by : int8_t { drawing, shape, shape_turned, colors, by_count,
                              count = by_count + (int)count_kind::count };

// The number twins by colors or by a count share: the colors a piece wears,
// a bit each, or what it counts (count_of); -1 for the twins by shape.
template <class ColorAt>
inline int twin_key(int how, int height, int width, ColorAt color_at) {
    if (how >= (int)twin_by::by_count) return count_of(how - (int)twin_by::by_count, height, width, color_at);
    if (how != (int)twin_by::colors) return -1;
    int colors = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            if (is_color(color_at(row, col))) colors |= 1 << color_at(row, col);
    return colors;
}

// THE ONE OF `count` WITH NO TWIN, twins(i, j) saying which are: -1 where none
// or several are without one, or where there are fewer than three -- two
// unlike each other are each the other's odd one.
template <class Twins>
inline int lone_one(int count, Twins twins) {
    if (count < 3) return -1;
    int found = -1;
    for (int i = 0; i < count; ++i) {
        bool repeated = false;
        for (int j = 0; j < count && !repeated; ++j) repeated = i != j && twins(i, j);
        if (!repeated) {
            if (found >= 0) return -1;
            found = i;
        }
    }
    return found;
}

// THE ONE PIECE WITH NO TWIN in the list, as `how` reads twins (twin_by), in
// local coordinates -- placement never matters: piece i is `width(i)` by
// `height(i)`, read by color_at(i, row, col), TRANSPARENT where it paints
// none (lone_one). Said once for SelectUnique and for the reading that picks
// by it, which reads a key once a piece (twin_key) where twins share one.
template<class Width, class Height, class ColorAt>
inline int unique_piece(int how, int count, Width width, Height height, ColorAt color_at) {
    if (how < 0 || how >= (int)twin_by::count) return -1;
    return lone_one(count, [&](int i, int j) {
        auto cell_of = [&](int piece) { return [&, piece](int r, int c) { return (int)color_at(piece, r, c); }; };
        if (how >= (int)twin_by::colors)
            return twin_key(how, height(i), width(i), cell_of(i)) == twin_key(how, height(j), width(j), cell_of(j));
        shape_freedom free;
        free.turn = how == (int)twin_by::shape_turned;
        free.recolor = how != (int)twin_by::drawing;
        return same_shape(width(i), height(i), cell_of(i), width(j), height(j), cell_of(j), free);
    });
}

// THE LINE `from` STEPS PAST THE MIDDLE OF A SPAN of lines lo..hi, the middle
// being the one line as many lines from either end: -1 where the span has two
// middle lines, or the line falls outside it. SelectRowFromMiddle and
// SelectColumnFromMiddle read the span every piece's paint covers, and the
// reading replays them here (d23f8c26 keeps the middle column).
inline int line_from_middle(int lo, int hi, int from) {
    if (lo > hi || (hi - lo) % 2 != 0) return -1;
    const int line = (lo + hi) / 2 + from;
    return line >= lo && line <= hi ? line : -1;
}

// WHETHER THE PIECE `mine` WOULD FIT INTO `partner`: turned or mirrored any of
// the eight ways and set down anywhere, it fills one of the partner's gaps
// whole, paints no cell the partner paints, and whatever it has over hangs out
// past the partner's box on the side the gap opens to. A GAP, for now: a run of
// the cells in the partner's box the partner leaves empty, joined side to side,
// that meets the edge of that box on one side alone -- a side the piece lies
// past. Nothing moves: this is the question WouldFit asks, and moving a piece
// into its gap would be the same question answered with the place. The pieces
// are asked as each_partner asks them.
template <class Pieces>
inline bool would_fit(int mine, int partner, const Pieces& pieces) {
    const cell_box own = pieces.box(mine), box = pieces.box(partner);
    if (own.empty() || box.empty()) return false;
    auto painted = [&](int i, int row, int col) { return pieces.painted(i, row, col); };
    constexpr uint8_t top = 1, bottom = 2, left = 4, right = 8;
    const uint8_t past = (uint8_t)((own.row1 < box.row0 ? top : 0) | (own.row0 > box.row1 ? bottom : 0) |
                                   (own.col1 < box.col0 ? left : 0) | (own.col0 > box.col1 ? right : 0));
    if (!past) return false;
    int cells = 0;
    for (int row = own.row0; row <= own.row1; ++row)
        for (int col = own.col0; col <= own.col1; ++col) cells += painted(mine, row, col) ? 1 : 0;
    const int tall = box.height(), wide = box.width();
    // Each empty cell of the box: 0 not walked yet, 1 walked, 2 in the run walked now.
    uint8_t seen[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    int16_t run_row[MAX_GRID_CELLS], run_col[MAX_GRID_CELLS];
    // Whether the piece, some way round, fills whole the run of `count` cells
    // starting at (first_row, first_col), hanging out past `side` with whatever
    // it has over.
    auto fills = [&](int first_row, int first_col, int count, uint8_t side) {
        const int anchor_row = box.row0 + first_row, anchor_col = box.col0 + first_col;
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            const auto [broad, high] = posed_size(pose, own.width(), own.height());
            auto at = [&](int row, int col) {
                const auto [from_row, from_col] = pose_source(pose, own.width(), own.height(), row, col);
                return (bool)painted(mine, own.row0 + from_row, own.col0 + from_col);
            };
            // The run's first cell is filled by one of the piece's cells: each in turn.
            for (int lands_row = 0; lands_row < high; ++lands_row)
                for (int lands_col = 0; lands_col < broad; ++lands_col) {
                    if (!at(lands_row, lands_col)) continue;
                    const int down = anchor_row - lands_row, across = anchor_col - lands_col;
                    int covered = 0;
                    bool sits = true;
                    for (int row = 0; row < high && sits; ++row)
                        for (int col = 0; col < broad && sits; ++col) {
                            if (!at(row, col)) continue;
                            const int on_row = row + down, on_col = col + across;
                            if (painted(partner, on_row, on_col)) { sits = false; break; }
                            if (on_row >= box.row0 && on_row <= box.row1 && on_col >= box.col0 && on_col <= box.col1) {
                                if (seen[on_row - box.row0][on_col - box.col0] == 2) ++covered;
                                else sits = false;
                            } else {
                                sits = side == top ? on_row < box.row0 : side == bottom ? on_row > box.row1
                                     : side == left ? on_col < box.col0 : on_col > box.col1;
                            }
                        }
                    if (sits && covered == count) return true;
                }
        }
        return false;
    };
    for (int start_row = 0; start_row < tall; ++start_row)
        for (int start_col = 0; start_col < wide; ++start_col) {
            if (seen[start_row][start_col] || painted(partner, box.row0 + start_row, box.col0 + start_col)) continue;
            // Walk the run from its first cell in reading order, noting the box's sides it meets.
            uint8_t meets = 0;
            seen[start_row][start_col] = 2;
            run_row[0] = (int16_t)start_row;
            run_col[0] = (int16_t)start_col;
            const int count = walk_out<4>(tall, wide, run_row, run_col, 1,
                [&](int row, int col) {
                    if (seen[row][col] || painted(partner, box.row0 + row, box.col0 + col)) return false;
                    seen[row][col] = 2;
                    return true;
                },
                [&](int row, int col) {
                    if (row == 0) meets |= top;
                    if (row == tall - 1) meets |= bottom;
                    if (col == 0) meets |= left;
                    if (col == wide - 1) meets |= right;
                });
            const bool one_side = meets == top || meets == bottom || meets == left || meets == right;
            if (one_side && (meets & past) && count <= cells && fills(start_row, start_col, count, meets)) return true;
            for (int i = 0; i < count; ++i) seen[run_row[i]][run_col[i]] = 1;
        }
    return false;
}

// THE PARTS WALLS FENCE OFF: every cell of a canvas `width` by `height` that is
// no wall, joined to the cells beside it at a side -- four ways, as water runs
// (walk_out), so walls that meet only at a corner still fence -- each cell's
// part into `part`, -1 on a wall and off the canvas, and how many parts there
// are. Numbered in reading order by the first cell of each, row by row, left to
// right, as rows wrap. Walls across the whole canvas fence the tiles of a
// lattice, numbered row by row; walls that cross nothing fence what they close
// off against the edges (63613498's grey corner, its key alone in part 0).
// `wall(row, col)` says whether a cell is a wall. Partition cuts on these parts
// and SelectPartition picks by them, so a part is one thing wherever it is asked.
template <class Wall>
inline int fence_parts(int width, int height, Wall wall, int16_t (&part)[MAX_GRID_DIM][MAX_GRID_DIM]) {
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) part[row][col] = -1;
    if (width > MAX_GRID_DIM) width = MAX_GRID_DIM;
    if (height > MAX_GRID_DIM) height = MAX_GRID_DIM;
    int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
    int parts = 0;
    auto claim = [&](int row, int col) {
        if (part[row][col] >= 0 || wall(row, col)) return false;
        part[row][col] = (int16_t)parts;
        return true;
    };
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            if (!claim(row, col)) continue;
            queue_row[0] = (int16_t)row;
            queue_col[0] = (int16_t)col;
            walk_out<4>(height, width, queue_row, queue_col, 1, claim, [](int, int) {});
            ++parts;
        }
    return parts;
}

// THE PARTS A PICK FENCES OFF (fence_parts): its walls are every cell a picked
// piece paints, as Partition and SelectPartition read them. The pieces are
// asked as each_partner asks them.
template <class Pieces>
inline int parts_the_pick_fences(const Pieces& pieces, int width, int height,
                                                int16_t (&part)[MAX_GRID_DIM][MAX_GRID_DIM]) {
    bool wall[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (int i = 0; i < pieces.count(); ++i) {
        if (!pieces.is_picked(i)) continue;
        const cell_box box = pieces.box(i);
        if (box.empty()) continue;
        for (int row = box.row0 < 0 ? 0 : box.row0; row <= box.row1 && row < MAX_GRID_DIM; ++row)
            for (int col = box.col0 < 0 ? 0 : box.col0; col <= box.col1 && col < MAX_GRID_DIM; ++col)
                wall[row][col] = wall[row][col] || pieces.painted(i, row, col);
    }
    return fence_parts(width, height, [&](int row, int col) { return wall[row][col]; }, part);
}

// WHAT Partition HANDS BACK past the walls, part by part in reading order
// (parts_the_pick_fences): every piece not picked, clipped to the part,
// `clip(i, box, in_part)`, then the part's cells no piece holds as one room,
// `room(part, box, empty)`. `box` is the part's; in_part(row, col) and
// empty(row, col) say which of its cells. The pieces are asked as each_partner
// asks them.
template <class Pieces, class Clip, class Room>
inline void each_partition_piece(const Pieces& pieces, int width, int height, Clip clip, Room room) {
    int16_t part[MAX_GRID_DIM][MAX_GRID_DIM];
    const int parts = parts_the_pick_fences(pieces, width, height, part);
    bool held[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (int i = 0; i < pieces.count(); ++i) {
        const cell_box box = pieces.box(i);
        if (box.empty()) continue;
        for (int row = box.row0 < 0 ? 0 : box.row0; row <= box.row1 && row < MAX_GRID_DIM; ++row)
            for (int col = box.col0 < 0 ? 0 : box.col0; col <= box.col1 && col < MAX_GRID_DIM; ++col)
                held[row][col] = held[row][col] || pieces.painted(i, row, col);
    }
    for (int which = 0; which < parts; ++which) {
        cell_box box;
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < MAX_GRID_DIM; ++col)
                if (part[row][col] == which) box.include_cell(row, col);
        if (box.empty()) continue;
        auto in_part = [&](int row, int col) { return part[row][col] == which; };
        for (int i = 0; i < pieces.count(); ++i)
            if (!pieces.is_picked(i)) clip(i, box, in_part);
        room(which, box, [&](int row, int col) { return in_part(row, col) && !held[row][col]; });
    }
}

// WHETHER THE PIECE `mine` LIES IN PART `which` of `part` (parts_the_pick_fences):
// it paints a cell, and every cell it paints is in that part. A piece over a
// wall, or across two parts, lies in none. SelectPartition[which] picks these.
template <class Pieces>
inline bool lies_in_part(const Pieces& pieces, int mine, int which,
                         const int16_t (&part)[MAX_GRID_DIM][MAX_GRID_DIM]) {
    const cell_box box = pieces.box(mine);
    if (box.empty()) return false;
    for (int row = box.row0; row <= box.row1; ++row)
        for (int col = box.col0; col <= box.col1; ++col)
            if (pieces.painted(mine, row, col) &&
                (row < 0 || col < 0 || row >= MAX_GRID_DIM || col >= MAX_GRID_DIM || part[row][col] != which))
                return false;
    return true;
}

// WHETHER SelectLikeKey[how] KEEPS THE PIECE `mine`: some piece of the key --
// every piece not picked, as the key words read it (each_key_entry) -- is of
// its shape, by the likeness `how` names (SHAPE_LIKENESSES: as it lies or
// turned any of the eight ways, at its size, any size or stretched), as
// each_partner finds a piece of the same shape (63613498: the piece shaped as
// the key in the grey corner lies turns grey, not the ones turned). The pieces
// are asked as each_partner asks them.
constexpr int KEY_LIKENESSES = SHAPE_LIKENESSES;
template <class Pieces>
inline bool like_the_key(int how, int mine, const Pieces& pieces) {
    return how >= 0 && how < KEY_LIKENESSES &&
           any_partner((partner_by)((int)partner_by::same_shape + how), mine, pieces);
}

// WHERE SelectPartner[value] MOVES THE PICK: onto every partner each picked
// piece has by the way `value` % partner_by::count names, among every piece,
// picked or not (each_partner), lying on the side of it `value` /
// partner_by::count names -- any side [0], or past it wholly that way of it,
// each of the eight directions from up clockwise (lies_toward). Each one handed
// to `mark(other)`: the pick becomes those, and a piece no picked piece has for
// a partner comes out unpicked (4364c1c4: of every piece, the partners above
// each, the upper piece of each stack). The pieces are asked as each_partner
// asks them.
constexpr int PARTNER_DIRECTIONS = 1 + direction::count;
constexpr int SELECT_PARTNER_VALUES = PARTNER_DIRECTIONS * (int)partner_by::count;
constexpr partner_by partner_way_of(int value) { return (partner_by)(value % (int)partner_by::count); }
// The direction a SelectPartner value names: -1 for any direction.
constexpr int partner_direction_of(int value) { return value / (int)partner_by::count - 1; }
template <class Pieces, class Mark>
inline void each_partner_picked(int value, const Pieces& pieces, Mark mark) {
    if (value < 0 || value >= SELECT_PARTNER_VALUES) return;
    const partner_by how = partner_way_of(value);
    const int partner_direction = partner_direction_of(value);
    for (int mine = 0; mine < pieces.count(); ++mine) {
        if (!pieces.is_picked(mine)) continue;
        each_partner(how, mine, pieces, [&](int other) {
            if (partner_direction < 0 || lies_toward(pieces.box(mine), pieces.box(other), partner_direction)) mark(other);
        }, true);
    }
}

// TRANSLATE · NormalizeOrientation, ScaleDown, Shift, MoveToEdge, MoveToward,
// SlamRow and SlamColumn, AlignToCanvas

// A GRID LYING ON ITS SIDE is wider than it is tall. NormalizeOrientation[0]
// reads one transposed, so its long way runs down, and a square one as it
// stands. Asked here by the op and by any reading that replays it, so the two
// cannot turn different grids.
constexpr bool lies_on_its_side(int width, int height) { return width > height; }

// The color most of a `factor` by `factor` block wears, as ScaleDown votes.
inline int8_t majority_vote(const cells& mask, int row, int col,
                            int factor, int height, int width) {
    color_tally tally;
    for (int dr = 0; dr < factor; ++dr)
        for (int dc = 0; dc < factor; ++dc) {
            const int source_row = row * factor + dr;
            const int source_col = col * factor + dc;
            if (source_row < height && source_col < width) tally.add(mask[source_row][source_col]);
        }
    return tally.most();
}

// WHERE SHIFT MOVES A BOX, its own size away: one step that way (step_of), its
// height a step down or up and its width a step across. The op and the
// reading that replays it take the move from here; a direction out of range stays.
inline direction_step shift_by(int move_direction, int width, int height) {
    if (move_direction < 0 || move_direction >= direction::count) return { 0, 0 };
    const direction_step step = step_of(move_direction);
    return { step.down * height, step.across * width };
}

// How far a box goes that way before its far side meets the edge of a canvas
// this size, as MoveToEdge takes it: negative when that side already lies past
// the edge, and on a diagonal whichever edge comes first.
inline int steps_to_edge(const cell_box& box, int width, int height, int move_direction) {
    const direction_step step = step_of(move_direction);
    const int room_down = step.down > 0 ? height - 1 - box.row1 : box.row0;
    const int room_across = step.across > 0 ? width - 1 - box.col1 : box.col0;
    if (step.across == 0) return room_down;
    if (step.down == 0) return room_across;
    return room_down < room_across ? room_down : room_across;
}

// THE DIRECTION FROM ONE BOX TOWARD ANOTHER: the one of the eight directions whose
// compass sector the line between their middles points into, each sector 45
// degrees wide, its edges 22.5 degrees off the axes. A line steeper than that
// off the rows goes up or down, and one flatter than that off the columns goes
// across; tan 22.5 is sqrt 2 - 1, irrational, so with whole cells no line lies
// on an edge, and the test needs no fractions. -1 when the middles coincide.
inline int direction_toward(const cell_box& from, const cell_box& to) {
    // Twice the middles' distance, so a middle between two cells is whole.
    const int down = (to.row0 + to.row1) - (from.row0 + from.row1);
    const int across = (to.col0 + to.col1) - (from.col0 + from.col1);
    const int tall = down < 0 ? -down : down, wide = across < 0 ? -across : across;
    const int reach = (tall + wide) * (tall + wide);
    const int vertical = reach > 2 * wide * wide ? (down > 0 ? 1 : -1) : 0;
    const int horizontal = reach > 2 * tall * tall ? (across > 0 ? 1 : -1) : 0;
    if (vertical == 0 && horizontal == 0) return -1;
    for (int partner_direction = 0; partner_direction < direction::count; ++partner_direction)
        if (step_of(partner_direction).down == vertical && step_of(partner_direction).across == horizontal) return partner_direction;
    return -1;
}

// HOW FAR MoveToward GOES toward its partner: one step, its own size, or until
// it touches the partner.
enum class toward_by : int8_t { a_step, its_size, touching, count };

// WHERE MoveToward MOVES A PIECE: that way toward its partner (direction_toward,
// between their boxes), by `far` -- one step; its own height or width each way
// it goes; or step by step until a cell of it touches a cell of the partner, at
// a side or a corner, as partner_by::touching asks, on a canvas `width` by
// `height`. A piece that touches already, or that would leave the canvas first,
// stays. Hands back how far down and across; the op and the reading's replay
// both take the move from here. The pieces are asked as each_partner asks them.
template <class Pieces>
inline direction_step move_toward(toward_by far, int mine, int partner, const Pieces& pieces,
                                                int width, int height) {
    const cell_box own = pieces.box(mine);
    const int partner_direction = direction_toward(own, pieces.box(partner));
    if (partner_direction < 0 || own.empty()) return { 0, 0 };
    const direction_step step = step_of(partner_direction);
    if (far == toward_by::a_step) return step;
    if (far == toward_by::its_size) return { step.down * own.height(), step.across * own.width() };
    for (int steps = 0;; ++steps) {
        const int down = steps * step.down, across = steps * step.across;
        if (own.row0 + down < 0 || own.row1 + down >= height || own.col0 + across < 0 || own.col1 + across >= width)
            return { 0, 0 };
        if (touches(pieces, mine, partner, { down, across }, false)) return { down, across };
    }
}

// Slide cells within each row or column, preserving their order and colors.
// Each cell moves by step, stopping early at the edge or the cell ahead.
// Read from the original cells and write into separate, already-cleared storage.
// Returns whether any cell moved. Selection and dimension limits belong to callers.
template <class At, class Put>
inline bool slam_cells(int width, int height, bool rows, int step,
                       At at, Put put) {
    const int lines = rows ? height : width;
    const int length = rows ? width : height;
    const int step_sign = step > 0 ? 1 : -1;
    bool moved = false;
    for (int line = 0; line < lines; ++line) {
        int wall = step > 0 ? length - 1 : 0;
        for (int i = 0; i < length; ++i) {
            const int source = step > 0 ? length - 1 - i : i;
            const int color = rows ? at(line, source) : at(source, line);
            if (color == TRANSPARENT) continue;
            int target = source + step;
            if (step > 0 ? target > wall : target < wall) target = wall;
            put(rows ? line : target, rows ? target : line, color);
            wall = target - step_sign;
            moved = moved || target != source;
        }
    }
    return moved;
}

// Shift a non-empty box flush against another, without changing the other axis.
// The larger gap chooses the axis (ties go across); the nearer side chooses
// the shift (ties go left or up). Overlapping boxes have negative gaps.
// The result is {down, across}, in grid coordinates.
inline std::pair<int, int> shift_to_touch(const cell_box& moving,
                                                         const cell_box& target) {
    const int left = target.col0 - moving.col1 - 1;
    const int right = target.col1 - moving.col0 + 1;
    const int above = target.row0 - moving.row1 - 1;
    const int below = target.row1 - moving.row0 + 1;
    int across = abs(left) <= abs(right) ? left : right;
    int down = abs(above) <= abs(below) ? above : below;
    const int column_gap = max(target.col0 - moving.col1, moving.col0 - target.col1);
    const int row_gap = max(target.row0 - moving.row1, moving.row0 - target.row1);
    if (column_gap >= row_gap) down = 0;
    else across = 0;
    return {down, across};
}

// MAP · Recolor and AddColor, ShapeColorMap, BidiColorMap, VoteDenoise,
// SnapToAnchors, ProjectOnSeeds

// Recolor and AddColor's relative color sources, shared by reading and execution.
// Ties choose the lowest color, and an empty object stays transparent.
template <class ColorAt>
inline int8_t object_paint_color(int height, int width, int8_t value, ColorAt color_at) {
    if (value < COLOR_COUNT) return value;
    color_tally tally;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) tally.add(color_at(row, col));
    // COLOR_COUNT names the color the fewest cells wear, past it the most.
    return value == COLOR_COUNT ? tally.least() : tally.most();
}

// WHICH OF THE EIGHT WAYS ROUND LEAVE A DRAWING AS IT WAS, colors and empty
// cells and all, a bit each (pose_source, the turns' and flips' own reading):
// 1 for a drawing only the same as it lies, 255 for one the same every way.
template <class ColorAt>
inline uint32_t symmetry_of(int width, int height, ColorAt color_at) {
    uint32_t poses = 0;
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, width, height);
        bool same = posed_width == width && posed_height == height;
        for (int row = 0; row < height && same; ++row)
            for (int col = 0; col < width && same; ++col) {
                const auto [from_row, from_col] = pose_source(pose, width, height, row, col);
                same = color_at(from_row, from_col) == color_at(row, col);
            }
        if (same) poses |= 1u << pose;
    }
    return poses;
}

// WHAT A LEARNED TABLE KNOWS A PIECE BY, ShapeColorMap's value: its shape,
// whichever way round it lies (key_of_shape), or its symmetry, which
// ways round leave its drawing as it was (symmetry_of: 44f52bb0's picture, 7
// where no turn or flip leaves it as it was, else 1). color_at(row, col) reads
// the piece's drawing in its own box. The reading keys its table here and the
// op looks it up here, so the two cannot know a piece by different keys.
enum class shape_key : int8_t { shape, symmetry, count };
template <class ColorAt>
inline uint32_t shape_key_of(int kind, int width, int height, ColorAt color_at) {
    if (kind == (int)shape_key::symmetry) return symmetry_of(width, height, color_at);
    return key_of_shape(width, height, color_at, ANY_WAY_ROUND);
}

// A LEARNED TABLE OF KEYS (shape_key_of): the color each key it knows names,
// and the color every other key takes -- -1 where a piece it does not know
// stays as it was (25d8a9c8: the full row 5, any other shape 0). The reading
// learns it and the ops look it up, both through color_of, so the two cannot
// name a key's color differently.
struct shape_table {
    static constexpr int MAX_KEYS = 16;
    uint32_t keys[MAX_KEYS] = {};
    int8_t colors[MAX_KEYS] = {};
    int count = 0;
    int8_t otherwise = -1;
    bool holds() const { return count > 0; }
    int color_of(uint32_t key) const {
        for (int i = 0; i < count && i < MAX_KEYS; ++i)
            if (keys[i] == key) return colors[i];
        return otherwise;
    }
    // Remembers which color goes with this key. Returns false if the key gets
    // conflicting colors.
    bool learn(uint32_t key, int8_t color) {
        for (int i = 0; i < count; ++i)
            if (keys[i] == key) return colors[i] == color;
        if (count >= MAX_KEYS) return false;
        keys[count] = key;
        colors[count] = color;
        count++;
        return true;
    }
};

// A KEY DRAWN IN THE GRID: a table the grid holds itself, read off each grid
// rather than learned off the pairs, so every grid brings its own (0becf7df's
// corner pairs 1 with 3 and 2 with 8 in one pair, 4 with 2 and 3 with 7 in the
// next). The key is every piece not picked, however many and wherever they
// lie, read together as one drawing, a later piece over an earlier one as
// Union lays them; each entry in it names what a property of a piece maps to,
// and each way of reading it below is one property. The pieces are asked as
// each_partner asks them, `color_at(i, row, col)`, the color piece i paints at
// that cell of the grid, and `key_piece(row, col)` (key_piece_at, which a
// reading may keep).
// The piece that shows a cell of the key's drawing, the last not picked to
// paint it, or -1 where none does.
template <class Pieces>
inline int key_piece_at(const Pieces& pieces, int row, int col) {
    int found = -1;
    for (int i = 0; i < pieces.count(); ++i)
        if (!pieces.is_picked(i) && pieces.painted(i, row, col)) found = i;
    return found;
}
// The color the key's drawing shows at a cell, TRANSPARENT where it shows none.
template <class Pieces>
inline int key_color_at(const Pieces& pieces, int row, int col) {
    const int i = pieces.key_piece(row, col);
    return i < 0 ? (int)TRANSPARENT : (int)pieces.color_at(i, row, col);
}

// READ BY COLOR: two cells of two colors side by side, alone together along a
// row -- or, `along_rows` false, a column -- are an entry, handed to
// `visit(first, second)`, the first cell's color and the second's.
template <class Pieces, class Visit>
inline void each_key_entry(const Pieces& pieces, bool along_rows, Visit visit) {
    auto key_at = [&](int row, int col) { return key_color_at(pieces, row, col); };
    const int down = along_rows ? 0 : 1, across = along_rows ? 1 : 0;
    for (int i = 0; i < pieces.count(); ++i) {
        if (pieces.is_picked(i)) continue;
        const cell_box box = pieces.box(i);
        if (box.empty()) continue;
        for (int row = box.row0; row <= box.row1; ++row)
            for (int col = box.col0; col <= box.col1; ++col) {
                if (pieces.key_piece(row, col) != i) continue;
                const int first = key_at(row, col), second = key_at(row + down, col + across);
                if (!is_color(key_at(row - down, col - across)) && is_color(second) && second != first &&
                    !is_color(key_at(row + 2 * down, col + 2 * across)))
                    visit(first, second);
            }
    }
}

// THE SWAPS A KEY NAMES, read by color (each_key_entry): the two colors of each
// entry trade places, `to[color]` the color each comes out in, -1 for a color no
// entry names. False, and no color traded, where the key names no entry, or
// names a color with two partners: a table that is no swap.
template <class Pieces>
inline bool key_swaps(const Pieces& pieces, bool along_rows, int8_t* to) {
    for (int color = 0; color < COLOR_COUNT; ++color) to[color] = -1;
    bool any = false, swaps = true;
    each_key_entry(pieces, along_rows, [&](int first, int second) {
        swaps = swaps && (to[first] < 0 || to[first] == second) && (to[second] < 0 || to[second] == first);
        to[first] = (int8_t)second;
        to[second] = (int8_t)first;
        any = true;
    });
    if (any && swaps) return true;
    for (int color = 0; color < COLOR_COUNT; ++color) to[color] = -1;
    return false;
}

// WHAT A CELL OF A PICKED PIECE'S FRAME COMES OUT AS where BidiColorMap trades
// `color` with empty (EMPTY_PAIRS): that color's cells empty, the empty ones --
// empty, or a room's -- that color, the rest as they were (d511f180's 8s and
// the ground the plan deleted, 5s, trading places).
constexpr int8_t traded_with_empty(int8_t cell, int color) {
    return cell == color ? (int8_t)TRANSPARENT : !is_color(cell) ? (int8_t)color : cell;
}

// WHAT BidiColorMap[value] TRADES, as `to[color]`, -1 for a color it leaves as
// it was: the fixed pair `value` numbers (color_pair), or past those, the pairs
// a key drawn in the grid names (key_swaps), read along its rows or its
// columns. False where it trades nothing, or trades with empty.
template <class Pieces>
inline bool swaps_of(int value, const Pieces& pieces, int8_t* to) {
    if (value == KEY_PAIRS_ALONG_ROWS || value == KEY_PAIRS_ALONG_COLUMNS)
        return key_swaps(pieces, value == KEY_PAIRS_ALONG_ROWS, to);
    for (int color = 0; color < COLOR_COUNT; ++color) to[color] = -1;
    if (value >= EMPTY_PAIRS) return false;
    const color_pair pair = color_pair::from_index(value);
    if (!is_color(pair.first)) return false;
    to[pair.first] = (int8_t)pair.second;
    to[pair.second] = (int8_t)pair.first;
    return true;
}

// READ BY PLACE: the key's entries are its cells in reading order, row by row,
// left to right, each handed to `visit(color)` -- every one, or, `grounded`,
// every one but those of its ground, the color most of its cells wear
// (color_tally), as a grid's ground is: the panel the entries are drawn on.
template <class Pieces, class Visit>
inline void each_key_cell(const Pieces& pieces, bool grounded, Visit visit) {
    cell_box box;
    for (int i = 0; i < pieces.count(); ++i) {
        if (pieces.is_picked(i)) continue;
        const cell_box own = pieces.box(i);
        if (own.empty()) continue;
        box.include_cell(own.row0, own.col0);
        box.include_cell(own.row1, own.col1);
    }
    if (box.empty()) return;
    int ground = TRANSPARENT;
    if (grounded) {
        color_tally tally;
        for (int row = box.row0; row <= box.row1; ++row)
            for (int col = box.col0; col <= box.col1; ++col) tally.add(key_color_at(pieces, row, col));
        ground = tally.most();
    }
    for (int row = box.row0; row <= box.row1; ++row)
        for (int col = box.col0; col <= box.col1; ++col) {
            const int color = key_color_at(pieces, row, col);
            if (is_color(color) && color != ground) visit(color);
        }
}

// ColorFrom PAST ITS PARTNER WAYS: each picked piece takes its color off a key
// by its place (key_place_colors); past that, its colors cell by cell from the
// one piece of its shape each likeness finds (colors_from_its_shape).
constexpr int KEY_BY_PLACE = (int)partner_by::count;
constexpr int COLORS_FROM_ITS_SHAPE = KEY_BY_PLACE + 1;

// A PIECE COLORED CELL BY CELL BY ANOTHER OF ITS SHAPE, said once for ColorFrom
// past the key by place and for the reading that replays it: the one partner
// the likeness `like` finds among the pieces not picked (partner_of, the same
// shape as it lies, turned, at any size or stretched), the two shapes laid one
// on the other as same_shape lays them (shape_fit), and each cell of the piece
// handed to paint(row, col, color) in the color of the partner's cell it lies
// on (fit_place: 103eff5b's azure shape takes the key's colors, the key turned
// and grown to it). False where it has no one partner. Colors never decide
// the fit: they are what it carries over.
template <class Pieces, class Paint>
inline bool colors_from_its_shape(int like, int mine, const Pieces& pieces, Paint paint) {
    if (like < 0 || like >= SHAPE_LIKENESSES) return false;
    const int partner = partner_of((partner_by)((int)partner_by::same_shape + like), mine, pieces);
    if (partner < 0) return false;
    const cell_box own = pieces.box(mine), theirs = pieces.box(partner);
    auto own_cell = [&](int row, int col) { return pieces.color_at(mine, own.row0 + row, own.col0 + col); };
    auto their_cell = [&](int row, int col) { return pieces.color_at(partner, theirs.row0 + row, theirs.col0 + col); };
    shape_fit fit;
    const bool sized = sized_likeness(like);
    if (!same_shape_in_blocks(own.width(), own.height(), own_cell, sized ? pieces.blocks(mine) : shape_blocks{},
                              theirs.width(), theirs.height(), their_cell,
                              sized ? pieces.blocks(partner) : shape_blocks{}, likeness_freedom(like), &fit))
        return false;
    for (int row = 0; row < own.height(); ++row)
        for (int col = 0; col < own.width(); ++col) {
            if (own_cell(row, col) == TRANSPARENT) continue;
            const auto [their_row, their_col] = fit_place(fit, row, col);
            paint(own.row0 + row, own.col0 + col, their_cell(their_row, their_col));
        }
    return true;
}

// THE COLOR A KEY GIVES EACH PICKED PIECE BY ITS PLACE: the picked pieces in
// reading order -- by the bands of rows they sit in, as SelectRow reads a row
// (overlapping_bands), then left to right, then top to bottom -- and the key's
// entries in reading order (each_key_cell), the first piece taking the first
// entry's color, the next the next. The entries are every cell of the key where
// it holds one for each picked piece (33b52de3's sixteen cells, sixteen
// shapes), else every cell but its ground's (1da012fc's gray panel: its four
// dots, four shapes, one dot on the panel's very edge). `color[i]` gets piece
// i's, -1 for a piece not picked. False, and no color given, where the picked
// pieces and the entries are still not as many. `lows`, `highs`, `band` and
// `order` are scratch, as many as the pieces.
template <class Pieces>
inline bool key_place_colors(const Pieces& pieces, int8_t* color, int16_t* lows, int16_t* highs,
                             int8_t* band, int* order) {
    // Past every row, a piece not picked bands after every picked one, apart.
    constexpr int16_t past = 2 * MAX_GRID_DIM;
    const int count = pieces.count();
    int picked = 0;
    for (int i = 0; i < count; ++i) {
        color[i] = -1;
        const cell_box box = pieces.box(i);
        const bool mine = pieces.is_picked(i) && !box.empty();
        lows[i] = mine ? (int16_t)box.row0 : past;
        highs[i] = mine ? (int16_t)box.row1 : past;
        picked += mine;
    }
    if (picked == 0) return false;
    overlapping_bands(count, lows, highs, band, order);
    // Each picked piece's place: how many picked pieces come before it, by
    // band, then left column, then top row, then the order they came in.
    for (int i = 0; i < count; ++i) highs[i] = lows[i] == past ? past : (int16_t)pieces.box(i).col0;
    for (int i = 0; i < count; ++i) {
        order[i] = -1;
        if (lows[i] == past) continue;
        int before = 0;
        for (int j = 0; j < count; ++j) {
            if (j == i || lows[j] == past) continue;
            const bool earlier = band[j] != band[i] ? band[j] < band[i]
                               : highs[j] != highs[i] ? highs[j] < highs[i]
                               : lows[j] != lows[i] ? lows[j] < lows[i] : j < i;
            before += earlier;
        }
        order[i] = before;
    }
    int cells = 0;
    each_key_cell(pieces, false, [&](int) { ++cells; });
    int entries = 0;
    each_key_cell(pieces, cells != picked, [&](int worn) {
        for (int i = 0; i < count; ++i)
            if (order[i] == entries) color[i] = (int8_t)worn;
        ++entries;
    });
    if (entries == picked) return true;
    for (int i = 0; i < count; ++i) color[i] = -1;
    return false;
}

// VoteDenoise's arithmetic, in one place for the op and for inference.
// Period per axis: a spacing qualifies when the structure it aligns agrees
// >=70% (pairs with both cells non-background), and the winner is the
// qualifier with the best agreements-minus-disagreements margin, where
// aligning structure against background counts as disagreement -- a true
// period only loses margin to noise, a false one bleeds it everywhere.
// Rows/columns whose cells match their residue-class majority <70% are
// outside the pattern and become background. Inside, each cell becomes the
// most common value among in-pattern cells whole periods away on both axes;
// ties keep the cell. `ground_at(r, c)` reads a cell, a hole read as the ground,
// which is what makes the ground the evidence it is (notes/4 - To Do/VoteDenoise
// Needs The Background.md); a ground deleted upstream is empty, and empty is a
// value that votes like a color (vote_slot). `put(r, c, v)`
// writes the result somewhere else. False, and nothing written, when neither
// axis repeats.
// THE GROUND THE VOTE READS BY, chosen from the piece in hand and nothing
// else: empty, where the ground was deleted upstream and left the piece with
// empty cells; otherwise the color most of it wears, the lowest on a tie. No
// puzzle-level background is consulted, by the op or by the reading.
template <typename At>
inline int8_t denoise_ground(int w, int h, At at) {
    color_tally tally;
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            const int8_t v = at(r, c);
            if (!is_color(v)) return (int8_t)TRANSPARENT;
            tally.add(v);
        }
    return tally.most();
}

// An empty value votes in a slot of its own, past the ten colors, and comes
// back out empty: a ground deleted upstream is evidence like any other.
inline int vote_slot(int8_t v) { return is_color(v) ? (int)v : COLOR_COUNT; }
inline int8_t slot_value(int slot) {
    return slot == COLOR_COUNT ? (int8_t)TRANSPARENT : (int8_t)slot;
}

template <typename At, typename Put>
inline bool vote_denoise_cells(int w, int h, int8_t bg, At ground_at, Put put) {
    // Period per axis (0 = no repeat found).
    int px = 0, py = 0;
    for (int axis = 0; axis < 2; ++axis) {
        const int dim = axis ? h : w;
        int best_margin = 0; bool found = false;
        for (int p = 2; p <= dim / 2; ++p) {
            int agree = 0, dis = 0, both = 0;
            for (int r = 0; r < (axis ? h - p : h); ++r)
                for (int c = 0; c < (axis ? w : w - p); ++c) {
                    int8_t a = ground_at(r, c);
                    int8_t b = axis ? ground_at(r + p, c) : ground_at(r, c + p);
                    if (a == bg && b == bg) continue;
                    if (a != bg && b != bg) {
                        ++both;
                        if (a == b) ++agree; else ++dis;
                    } else ++dis;
                }
            if (agree < 8 || agree * 10 < both * 7) continue;
            int margin = agree - dis;
            if (!found || margin > best_margin) {
                found = true; best_margin = margin;
                (axis ? py : px) = p;
            }
        }
    }
    if (px == 0 && py == 0) return false;

    // In-pattern lines: a line is inside iff >=70% of its cells equal the
    // per-cell majority of its residue class along that axis alone -- the
    // voted template is noise-free, so a noisy copy only pays for its own
    // noise, while an off-pattern line disagrees almost everywhere.
    bool col_in[MAX_GRID_DIM], row_in[MAX_GRID_DIM];
    for (int c = 0; c < w; ++c) col_in[c] = true;
    for (int r = 0; r < h; ++r) row_in[r] = true;
    // Read either axis as lines of cells. Keeping the scan order also keeps
    // the existing tie rule: the first color to reach the winning count wins.
    auto mark_pattern_lines = [&](int lines, int length, int period, bool across, bool* inside) {
        if (period <= 0) return;
        auto at = [&](int line, int cell) {
            return across ? ground_at(cell, line) : ground_at(line, cell);
        };
        for (int line = 0; line < lines; ++line) {
            int agree = 0;
            for (int cell = 0; cell < length; ++cell) {
                int counts[COLOR_COUNT + 1] = {};
                int8_t most_common = bg;
                int most = 0;
                for (int source = line % period; source < lines; source += period) {
                    const int8_t color = at(source, cell);
                    if (++counts[vote_slot(color)] > most) { most = counts[vote_slot(color)]; most_common = color; }
                }
                if (at(line, cell) == most_common) ++agree;
            }
            inside[line] = agree * 10 >= length * 7;
        }
    };
    mark_pattern_lines(w, h, px, true, col_in);
    mark_pattern_lines(h, w, py, false, row_in);

    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            if (!row_in[r] || !col_in[c]) { put(r, c, bg); continue; }
            int counts[COLOR_COUNT + 1] = {};
            for (int sr = (py > 0 ? r % py : r); sr < h; sr += (py > 0 ? py : h))
                for (int sc = (px > 0 ? c % px : c); sc < w; sc += (px > 0 ? px : w)) {
                    if (!row_in[sr] || !col_in[sc]) continue;
                    ++counts[vote_slot(ground_at(sr, sc))];
                }
            int8_t own = ground_at(r, c);
            int8_t best_v = own;
            int best_n = counts[vote_slot(own)];
            for (int slot = 0; slot <= COLOR_COUNT; ++slot)
                if (counts[slot] > best_n) { best_n = counts[slot]; best_v = slot_value(slot); }
            put(r, c, best_v);
        }
    return true;
}

// A BODY TURNED ONE OF THE EIGHT WAYS: the offset (row, col) from the cell it
// turns about, `way` & 3 quarter turns clockwise, then from 4 on flipped top to
// bottom -- the turns SnapToAnchors tries.
inline void snap_turn(int way, int row, int col, int& to_row, int& to_col) {
    for (int turn = 0; turn < (way & 3); ++turn) {
        const int turned = col;
        col = -row;
        row = turned;
    }
    if (way >= 4) row = -row;
    to_row = row;
    to_col = col;
}

// THE BODIES SNAPPED ONTO THEIR ANCHORS, placed the way SnapToAnchors places
// them -- one home for the placing, so the op and the rule that replays it
// cannot set a body down two ways. The cells `at` hands over, TRANSPARENT where
// empty, are read as they lie:
//   - the body color is the color most cells wear, the lowest on a tie;
//   - a body is an 8-connected run of painted cells, of any colors, that holds
//     the body color; its keys are cells of other colors with external anchors,
//     the first SNAP_MOST_KEYS of them in reading order;
//   - an anchor is a cell of a run that does not hold it, SNAP_MOST_ANCHORS of
//     each color at most.
// Each body, in the order the reading meets its run, is turned one of the eight
// ways (snap_turn) and set down where the most of its keys land on anchors of
// their own colors -- the first placing tried wins a tie, and one key at least
// has to land. `label` hands back each cell's run, -1 for none. `place` is told
// the body's run, the turn, the key the turn is taken about and the anchor that
// key lands on, and says whether it set anything down; only then are the
// anchors its keys cover used up, so no later body can take them.
constexpr int SNAP_MOST_RUNS = 64, SNAP_MOST_KEYS = 12, SNAP_MOST_ANCHORS = 16;
template <class At, class Place>
inline void snap_to_anchors(int width, int height, At at,
                            int8_t (&label)[MAX_GRID_DIM][MAX_GRID_DIM], Place place) {
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) label[row][col] = -1;
    color_tally worn;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) worn.add(at(row, col));
    const int body = worn.most();
    if (body < 0) return;

    // The runs, each walked eight ways out from the first cell the reading meets.
    bool holds_body[SNAP_MOST_RUNS] = {};
    int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
    int runs = 0;
    for (int row = 0; row < height && runs < SNAP_MOST_RUNS; ++row)
        for (int col = 0; col < width && runs < SNAP_MOST_RUNS; ++col) {
            if (at(row, col) == TRANSPARENT || label[row][col] != -1) continue;
            const int run = runs++;
            label[row][col] = (int8_t)run;
            queue_row[0] = (int16_t)row;
            queue_col[0] = (int16_t)col;
            walk_out<8>(height, width, queue_row, queue_col, 1,
                [&](int next_row, int next_col) {
                    if (at(next_row, next_col) == TRANSPARENT || label[next_row][next_col] != -1) return false;
                    label[next_row][next_col] = (int8_t)run;
                    return true;
                },
                [&](int seen_row, int seen_col) { if (at(seen_row, seen_col) == body) holds_body[run] = true; });
        }

    // The anchors: where each lies, and a list of them by color.
    int8_t loose[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t anchor_row[COLOR_COUNT][SNAP_MOST_ANCHORS], anchor_col[COLOR_COUNT][SNAP_MOST_ANCHORS];
    int anchors[COLOR_COUNT] = {};
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) loose[row][col] = (int8_t)TRANSPARENT;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int color = at(row, col);
            const int run = label[row][col];
            if (color == TRANSPARENT || run < 0 || holds_body[run]) continue;
            loose[row][col] = (int8_t)color;
            if (is_color(color) && anchors[color] < SNAP_MOST_ANCHORS) {
                anchor_row[color][anchors[color]] = (int16_t)row;
                anchor_col[color][anchors[color]] = (int16_t)col;
                ++anchors[color];
            }
        }

    // Each body, set down where the most of its keys meet anchors of their colors.
    // Only colors with an external anchor are keys. Body details with no partner
    // must not consume the bounded key list before the matching cells are read.
    for (int run = 0; run < runs; ++run) {
        if (!holds_body[run]) continue;
        int key_color[SNAP_MOST_KEYS], key_row[SNAP_MOST_KEYS], key_col[SNAP_MOST_KEYS], keys = 0;
        for (int row = 0; row < height && keys < SNAP_MOST_KEYS; ++row)
            for (int col = 0; col < width && keys < SNAP_MOST_KEYS; ++col) {
                const int color = at(row, col);
                if (label[row][col] != run || color == body || !is_color(color) || anchors[color] == 0) continue;
                key_color[keys] = color;
                key_row[keys] = row;
                key_col[keys] = col;
                ++keys;
            }
        if (keys == 0) continue;
        int best = 0, best_key = -1, best_row = 0, best_col = 0, best_way = 0;
        for (int key = 0; key < keys; ++key) {
            const int color = key_color[key];
            // A key is a color; said in full where it indexes the lists by color.
            if (color < 0 || color >= COLOR_COUNT) continue;
            for (int anchor = 0; anchor < anchors[color]; ++anchor)
                for (int way = 0; way < 8; ++way) {
                    int met = 0;
                    for (int other = 0; other < keys; ++other) {
                        int down, across;
                        snap_turn(way, key_row[other] - key_row[key], key_col[other] - key_col[key], down, across);
                        const int row = anchor_row[color][anchor] + down, col = anchor_col[color][anchor] + across;
                        met += row >= 0 && row < height && col >= 0 && col < width && loose[row][col] == key_color[other];
                    }
                    if (met > best) {
                        best = met;
                        best_key = key;
                        best_row = anchor_row[color][anchor];
                        best_col = anchor_col[color][anchor];
                        best_way = way;
                    }
                }
        }
        if (best_key < 0 || !place(run, best_way, key_row[best_key], key_col[best_key], best_row, best_col)) continue;
        for (int other = 0; other < keys; ++other) {
            int down, across;
            snap_turn(best_way, key_row[other] - key_row[best_key], key_col[other] - key_col[best_key], down, across);
            const int row = best_row + down, col = best_col + across, color = key_color[other];
            if (row < 0 || row >= height || col < 0 || col >= width || loose[row][col] != color) continue;
            loose[row][col] = (int8_t)TRANSPARENT;
            for (int anchor = 0; anchor < anchors[color]; ++anchor)
                if (anchor_row[color][anchor] == row && anchor_col[color][anchor] == col) {
                    anchor_row[color][anchor] = anchor_row[color][anchors[color] - 1];
                    anchor_col[color][anchor] = anchor_col[color][anchors[color] - 1];
                    --anchors[color];
                    break;
                }
        }
    }
}

// Calls land(row, col) for each stamp of tpl along marker until it leaves the
// canvas or land returns false.
template <class Land>
inline void project_along(const cell_box& tpl, const cell_box& marker,
                          int width, int height, Land land) {
    const int across = (marker.col0 + marker.col1) / 2 - (tpl.col0 + tpl.col1) / 2;
    const int down   = (marker.row0 + marker.row1) / 2 - (tpl.row0 + tpl.row1) / 2;
    int col = tpl.col0, row = tpl.row0;
    if (across > 0)      col = marker.col0;
    else if (across < 0) col = marker.col1 - (tpl.width() - 1);
    if (down > 0)        row = marker.row0;
    else if (down < 0)   row = marker.row1 - (tpl.height() - 1);
    const int step_col = col - tpl.col0, step_row = row - tpl.row0;
    if (step_col == 0 && step_row == 0) return;
    for (int repeat = 1; repeat <= MAX_GRID_DIM; ++repeat) {
        const int at_col = tpl.col0 + repeat * step_col;
        const int at_row = tpl.row0 + repeat * step_row;
        if (at_col + tpl.width() <= 0 || at_col >= width ||
            at_row + tpl.height() <= 0 || at_row >= height) return;
        if (!land(at_row, at_col)) return;
    }
}

// GENERATE · CopyTo, Bridge, ExtendTo, ExtendDiagonal and ExtendOrthogonal,
// Tile, MirrorAround

// Align like anchors of two spans: 0 = first cell, 1 = middle cell,
// 2 = last cell. Even spans use the earlier of the two middle cells.
// Copy placement shares this arithmetic between the reading and the op.
constexpr int aligned_copy_start(int source_size, int target_start,
                                 int target_size, int anchor) {
    return target_start + (target_size - 1) * anchor / 2 - (source_size - 1) * anchor / 2;
}

// WHAT BRIDGE DRAWS, said once for the op and for the reading that replays it.
// Each picked piece reaches for the picked pieces of its own colour whose
// centres share its row or its column, and the box between two is filled in
// that colour; a piece not picked neither reaches nor is reached (0d87d2a6's
// blue marks are bridged across the red blocks, which are not). HOW IT
// REACHES is Bridge's value, `way`: along rows and columns both, along rows
// alone, or along columns alone (way % 3), and to the nearest piece that way,
// to every one, or only to the one facing it across the grid (way / 3) --
// Bridge[0] reaching both ways for the nearest. Two pieces face each other
// across the `width` by `height` grid when one touches its left edge and the
// other its right, along a row, or its top and its bottom, along a column
// (0d87d2a6's test: two marks on the left edge share a column and are not
// joined; each is joined to the mark facing it on the right).
// PAST THOSE, Bridge[BRIDGE_TO_PARTNER + kind] joins each picked piece to its
// one partner of `kind` (partner_of_kind, among the pieces not picked: each
// way partner_by finds one, then the piece of each color) with an L: along the
// picked piece's middle row to the partner's middle column, then along that
// column to the partner. The start is the pick and the end its partner, so
// which way the L turns is said by which of the two is picked (0e671a1a: the
// 2 runs to the 4 along its own row, and the 4 to the 3 along its own). Like a
// bar, it is the path between the two: neither one's own box is its to fill,
// so two pieces side by side leave it nothing to draw (833966f4's column of
// cells, each beside the next).
// THE PIECES ARE ASKED as partner_of_kind asks them, in the op's own order;
// `color(i)` is the colour the op reads off piece i's first cell, -1 for none.
// land(bar, colour) sees each bar in the order the op lays them into its one
// drawing of every bar, and may say false to stop. The op calls it, so
// a reading and the op cannot draw different bars. Answers whether an L it
// drew turned: both its legs had cells to fill.
constexpr int BRIDGE_WAYS = 9, BRIDGE_TO_PARTNER = BRIDGE_WAYS,
              BRIDGE_VALUES = BRIDGE_TO_PARTNER + PARTNER_KINDS;
template <class Pieces, class Land>
inline bool bridge_bars(int way, int width, int height, const Pieces& pieces, Land land) {
    if (way < 0 || way >= BRIDGE_VALUES) return false;
    const int count = pieces.count();
    auto box_of_piece = [&](int i) { return pieces.box(i); };
    auto color_of = [&](int i) { return (int8_t)pieces.color(i); };
    auto picked = [&](int i) { return pieces.is_picked(i); };
    constexpr int most_pairs = 64;
    int16_t lows[most_pairs], highs[most_pairs];
    int pairs = 0;
    auto across = [&](int i) { const cell_box box = box_of_piece(i); return (box.col0 + box.col1) / 2; };
    auto down = [&](int i) { const cell_box box = box_of_piece(i); return (box.row0 + box.row1) / 2; };
    if (way >= BRIDGE_TO_PARTNER) {
        // A leg, one cell wide, cut back off a box that holds one of its ends.
        auto off = [](cell_box leg, const cell_box& box) {
            const bool along_row = leg.row0 == leg.row1;
            if (along_row ? leg.row0 < box.row0 || leg.row0 > box.row1 : leg.col0 < box.col0 || leg.col0 > box.col1)
                return leg;
            int& low = along_row ? leg.col0 : leg.row0;
            int& high = along_row ? leg.col1 : leg.row1;
            const int box_low = along_row ? box.col0 : box.row0, box_high = along_row ? box.col1 : box.row1;
            if (box_low <= low) low = max(low, box_high + 1);
            else if (box_high >= high) high = min(high, box_low - 1);
            return leg;
        };
        bool going = true, turned = false;
        for (int i = 0; i < count && going; ++i) {
            if (!picked(i) || box_of_piece(i).empty()) continue;
            const int other = partner_of_kind(way - BRIDGE_TO_PARTNER, i, pieces);
            if (other < 0) continue;
            const int row = down(i), col = across(i), turn_col = across(other), end_row = down(other);
            cell_box along_row, along_column;
            along_row.row0 = along_row.row1 = row;
            along_row.col0 = min(col, turn_col); along_row.col1 = max(col, turn_col);
            along_column.col0 = along_column.col1 = turn_col;
            along_column.row0 = min(row, end_row); along_column.row1 = max(row, end_row);
            const cell_box start = box_of_piece(i), end = box_of_piece(other);
            along_row = off(off(along_row, start), end);
            along_column = off(off(along_column, start), end);
            const int8_t color = color_of(i), paint = color < 0 ? (int8_t)0 : color;
            turned = turned || (!along_row.empty() && !along_column.empty());
            going = (along_row.empty() || land(along_row, paint)) && (along_column.empty() || land(along_column, paint));
        }
        return turned;
    }
    const bool rows = way % 3 != 2, columns = way % 3 != 1, every = way / 3 == 1, facing = way / 3 == 2;
    auto pair_up = [&](int i, int partner) {
        const int low = min(i, partner), high = max(i, partner);
        bool said = false;
        for (int p = 0; p < pairs && !said; ++p) said = lows[p] == low && highs[p] == high;
        if (said || pairs >= most_pairs) return;
        lows[pairs] = (int16_t)low;
        highs[pairs] = (int16_t)high;
        ++pairs;
    };
    // Whether pieces i and j face each other across the grid, along a row or
    // along a column.
    auto face = [&](int i, int j, bool along_rows) {
        const cell_box one = box_of_piece(i), other = box_of_piece(j);
        return along_rows ? (one.col0 <= 0 && other.col1 >= width - 1) || (other.col0 <= 0 && one.col1 >= width - 1)
                          : (one.row0 <= 0 && other.row1 >= height - 1) || (other.row0 <= 0 && one.row1 >= height - 1);
    };
    for (int i = 0; i < count; ++i) {
        if (!picked(i)) continue;
        const int8_t color = color_of(i);
        if (color < 0) continue;
        const int my_across = across(i), my_down = down(i);
        // Ties keep the first found, as the op's list has them.
        int partner = -1, nearest = 0x7FFF;
        for (int j = 0; j < count; ++j) {
            if (j == i || !picked(j) || color_of(j) != color) continue;
            const int their_across = across(j), their_down = down(j);
            const bool on_row = rows && their_down == my_down, on_column = columns && their_across == my_across;
            if (!on_row && !on_column) continue;
            if (facing) {
                if ((on_row && face(i, j, true)) || (on_column && face(i, j, false))) pair_up(i, j);
                continue;
            }
            if (every) { pair_up(i, j); continue; }
            const int distance = abs(their_across - my_across) + abs(their_down - my_down);
            if (distance < nearest) { nearest = distance; partner = j; }
        }
        if (partner >= 0) pair_up(i, partner);
    }
    for (int p = 0; p < pairs; ++p) {
        const cell_box one = box_of_piece(lows[p]), other = box_of_piece(highs[p]);
        const int apart_across = abs(across(highs[p]) - across(lows[p]));
        const int apart_down = abs(down(highs[p]) - down(lows[p]));
        if (apart_across == 0 && apart_down == 0) continue;
        // The gap along the way they lie apart, as deep as the two together.
        cell_box bar;
        if (apart_across >= apart_down) {
            bar.col0 = min(one.col1, other.col1) + 1; bar.col1 = max(one.col0, other.col0) - 1;
            bar.row0 = min(one.row0, other.row0);     bar.row1 = max(one.row1, other.row1);
        } else {
            bar.row0 = min(one.row1, other.row1) + 1; bar.row1 = max(one.row0, other.row0) - 1;
            bar.col0 = min(one.col0, other.col0);     bar.col1 = max(one.col1, other.col1);
        }
        // Pieces that touch or overlap along that way leave no gap to fill.
        if (bar.empty()) continue;
        const int8_t color = color_of(lows[p]);
        if (!land(bar, color < 0 ? (int8_t)0 : color)) return false;
    }
    return false;
}

// WHAT MaskBy AND MaskOut KEEP of a cell, said once for the ops and for the
// reading that replays a paint laid onto or off what is down: MaskBy keeps a
// cell where the canvas shows `color` -- any color at all, with `any`, as
// MaskBy[10] -- and MaskOut every other.
inline bool mask_keeps(int8_t canvas, int8_t color, bool any, bool keep_match) {
    const bool match = any ? canvas != TRANSPARENT : canvas == color;
    return match == keep_match;
}

// THE TRAIL ExtendTo GROWS, said once for the op and for the reading that
// replays it: from the cell (row, col) of a piece, step by step that way to the
// edge of a canvas `width` by `height` or to a cell another piece holds,
// through the piece's own cells -- `own(row, col)` and `taken(row, col)` say
// which -- and each cell passed that is not its own handed to `lay(row, col)`.
template <class Own, class Taken, class Lay>
inline void march_trail(int row, int col, direction_step step, int width, int height,
                        Own own, Taken taken, Lay lay) {
    for (int at_row = row + step.down, at_col = col + step.across;
         at_row >= 0 && at_row < height && at_col >= 0 && at_col < width;
         at_row += step.down, at_col += step.across) {
        if (own(at_row, at_col)) continue;
        if (taken(at_row, at_col)) break;
        lay(at_row, at_col);
    }
}

// THE TRACKS ExtendTo GROWS ALONG past its eight ways, every picked piece in
// step with the others. Along lines: its rows, its columns, its diagonals down
// to the right, its diagonals up to the right, both diagonals, its rows and
// columns, or every way, straight and diagonal. Or through the open cells: to
// every side, to every side along the canvas's edge, or to every side and
// corner. The order is ExtendTo's, so a value said once keeps its meaning.
enum class grown_along : int8_t {
    rows, columns, sides, canvas_edge,
    diagonals_down_right, diagonals_up_right, diagonals, rows_and_columns, every_direction, sides_and_corners,
    count
};

// The directions a track grows, one bit for each direction (1_readings.hpp): a line
// both ways along it, or the sides, the corners, or both, of every cell.
constexpr uint8_t grown_directions(grown_along track) {
    auto both = [](int first_direction, int second_direction) { return (uint8_t)(1u << first_direction | 1u << second_direction); };
    const uint8_t sides = (uint8_t)(both(direction::up, direction::down) | both(direction::right, direction::left));
    const uint8_t corners =
        (uint8_t)(both(direction::down_right, direction::up_left) | both(direction::up_right, direction::down_left));
    switch (track) {
    case grown_along::rows:                 return both(direction::right, direction::left);
    case grown_along::columns:              return both(direction::up, direction::down);
    case grown_along::diagonals_down_right: return both(direction::down_right, direction::up_left);
    case grown_along::diagonals_up_right:   return both(direction::up_right, direction::down_left);
    case grown_along::diagonals:            return corners;
    case grown_along::every_direction:
    case grown_along::sides_and_corners:    return (uint8_t)(sides | corners);
    default:                                return sides;
    }
}

// Whether a track spreads: every cell taken grows every way of the track, so
// the piece floods the open cells. Otherwise the track is lines, rays that
// keep their heading.
constexpr bool grown_spreads(grown_along track) {
    return track == grown_along::sides || track == grown_along::canvas_edge || track == grown_along::sides_and_corners;
}

// GROWN IN STEP UNTIL THEY MEET, said once for ExtendTo past its eight ways and
// for the reading that replays it. `owner` comes in holding each cell's piece,
// -1 where none is, and `grows(piece)` says which pieces were picked. Every
// cell of a picked piece then grows at once, a step a turn, onto the empty
// cells its track reaches -- a cell any piece holds is none to grow onto, so a
// piece not picked stands in the way -- each cell taken by the first piece to
// reach it and, reached by two in one turn, by the earlier piece in the list.
// Along the canvas's edge, a piece grows from its cells on the edge alone.
// `owner` comes back holding the piece that took each cell and `turn` the turn
// it was taken in: 0 for a piece's own cell, -1 for one no piece took. `turn`,
// `queue` and `cell_directions` are scratch, width * height cells each.
//
// THROUGH THE OPEN CELLS a piece floods: every cell it takes grows every way
// of the track. ALONG LINES every cell of a picked piece sends a ray each way
// of the track, and two rays meet where they reach one cell in one turn -- the
// earlier piece's goes on, the other stops -- or where one runs into a cell a
// ray along its own line took before it, head-on or behind; a ray of another
// line taken before it is crossed, as a line drawn across another goes on
// (21f83797's lines cross; 1190bc91's rays meet in one turn and stop). `met(cell,
// grow_direction, piece)` hears of each ray stopped by meeting another.
template <class Grows, class Met>
inline void grow_in_step(int width, int height, grown_along track, Grows grows,
                         int16_t* owner, int16_t* turn, int16_t* queue, uint8_t* cell_directions, Met met) {
    auto on_track = [&](int row, int col) {
        return track != grown_along::canvas_edge || row == 0 || col == 0 || row == height - 1 || col == width - 1;
    };
    const uint8_t track_directions = grown_directions(track);
    // The cells grown from: every picked cell on the track, and as the piece
    // floods, each cell it takes. Along lines, a picked cell's `cell_directions` are its
    // rays still going, and a taken cell's the directions of the rays that took it.
    int tail = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int at = row * width + col;
            turn[at] = (int16_t)(owner[at] >= 0 ? 0 : -1);
            cell_directions[at] = 0;
            if (owner[at] < 0) continue;
            if (!grows(owner[at])) owner[at] = -1;
            else if (on_track(row, col)) { queue[tail++] = (int16_t)at; cell_directions[at] = track_directions; }
        }
    if (grown_spreads(track)) {
        // Taken turn by turn, in the order reached: a cell is grown from only
        // once every cell of the turn before has reached it, so its piece is
        // settled.
        for (int head = 0; head < tail; ++head) {
            const int at = queue[head], row = at / width, col = at % width;
            for (int grow_direction = direction::up; grow_direction < direction::count; ++grow_direction) {
                if (!(track_directions >> grow_direction & 1u)) continue;
                const direction_step step = step_of(grow_direction);
                const int next_row = row + step.down, next_col = col + step.across;
                if (next_row < 0 || next_col < 0 || next_row >= height || next_col >= width ||
                    !on_track(next_row, next_col))
                    continue;
                const int next = next_row * width + next_col;
                if (turn[next] < 0) {
                    turn[next] = (int16_t)(turn[at] + 1);
                    owner[next] = owner[at];
                    queue[tail++] = (int16_t)next;
                } else if (turn[next] == turn[at] + 1 && owner[at] < owner[next]) {
                    owner[next] = owner[at];
                }
            }
        }
        return;
    }
    for (int16_t step_number = 1; ; ++step_number) {
        bool going = false;
        for (int i = 0; i < tail; ++i) {
            const int from = queue[i], from_row = from / width, from_col = from % width;
            for (int grow_direction = direction::up; grow_direction < direction::count; ++grow_direction) {
                const uint8_t ray = (uint8_t)(1u << grow_direction);
                if (!(cell_directions[from] & ray)) continue;
                const direction_step step = step_of(grow_direction);
                const int row = from_row + step.down * step_number, col = from_col + step.across * step_number;
                if (row < 0 || col < 0 || row >= height || col >= width) {
                    cell_directions[from] = (uint8_t)(cell_directions[from] & ~ray);
                    continue;
                }
                const int at = row * width + col;
                const uint8_t own_line = (uint8_t)(ray | 1u << (grow_direction + direction::count / 2) % direction::count);
                bool stops = false;
                if (turn[at] < 0) {
                    turn[at] = step_number;
                    owner[at] = owner[from];
                    cell_directions[at] = ray;
                } else if (turn[at] == step_number && owner[from] < owner[at]) {
                    // Met in this turn by an earlier piece: the rays that took it stop.
                    for (int arrival_direction = direction::up; arrival_direction < direction::count; ++arrival_direction) {
                        if (!(cell_directions[at] >> arrival_direction & 1u)) continue;
                        const direction_step back = step_of(arrival_direction);
                        const int start = (row - back.down * step_number) * width + col - back.across * step_number;
                        cell_directions[start] = (uint8_t)(cell_directions[start] & ~(1u << arrival_direction));
                        met(at, arrival_direction, owner[at]);
                    }
                    owner[at] = owner[from];
                    cell_directions[at] = ray;
                } else if (turn[at] == step_number && owner[from] == owner[at]) {
                    cell_directions[at] = (uint8_t)(cell_directions[at] | ray);
                } else {
                    // A body, a later piece's ray met in this turn, or a ray
                    // along its own line before it; another line's is crossed.
                    stops = turn[at] == 0 || turn[at] == step_number || (cell_directions[at] & own_line) != 0;
                    if (stops && turn[at] > 0) met(at, grow_direction, owner[from]);
                }
                if (stops) cell_directions[from] = (uint8_t)(cell_directions[from] & ~ray);
                else going = true;
            }
        }
        if (!going) return;
    }
}

// Extend's seed and rays are shared by inference and execution. Ties go to
// the first cell in row-major order; the seed itself is not part of the rays.
// Each ray cell is handed over as its offset from the seed, down and across,
// so whoever paints it clips it to what it paints on: a long ray runs on to the
// edge of the grid, not to the end of some frame of its own.
struct ray_seed { int row = -1, col = -1; };

template <class Cell>
inline ray_seed nearest_ray_seed(int height, int width,
                                 int center_row, int center_col, Cell cell) {
    ray_seed seed;
    int best = MAX_GRID_CELLS;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            if (cell(row, col) == TRANSPARENT) continue;
            const int dr = row - center_row, dc = col - center_col;
            const int distance = (dr < 0 ? -dr : dr) + (dc < 0 ? -dc : dc);
            if (distance < best) { best = distance; seed = { row, col }; }
        }
    return seed;
}

// The rays run the four sides of the compass, its even directions, or the four
// diagonals, its odd ones (direction, 1_readings.hpp).
template <class Paint>
inline void each_extended_ray(ray_seed seed, int radius, bool diagonal,
                                               Paint paint) {
    if (seed.row < 0) return;
    for (int ray_direction = diagonal ? 1 : 0; ray_direction < direction::count; ray_direction += 2) {
        const direction_step step = step_of(ray_direction);
        for (int along = 1; along <= radius; ++along) paint(step.down * along, step.across * along);
    }
}

// TILE, SAID ONCE FOR THE OP AND FOR THE READINGS THAT REPLAY IT: what its
// steps measure, how much of a piece one copy is, and where the copies lie.

// WHAT A Tile'S STEPS MEASURE, read for one piece: its own width and height,
// the fewest empty rows and columns lying between two of its parts
// (inner_gap), and how many colors the picked pieces hold together
// (color_count).
struct tile_measures {
    int own_width = 0, own_height = 0, row_gap = 0, column_gap = 0, color_count = 0;
};

// The fewest empty rows -- or, `rows` false, columns -- lying between two
// painted ones of a piece `height` by `width`, painted(row, col) saying which
// cells it paints; 0 where none lie between.
template <class Painted>
inline int inner_gap(int height, int width, bool rows, Painted painted) {
    const int lines = rows ? height : width, along = rows ? width : height;
    int fewest = 0, last = -1;
    for (int line = 0; line < lines; ++line) {
        bool any = false;
        for (int at = 0; at < along && !any; ++at) any = rows ? painted(line, at) : painted(at, line);
        if (!any) continue;
        const int empty = line - last - 1;
        if (last >= 0 && empty > 0 && (fewest == 0 || empty < fewest)) fewest = empty;
        last = line;
    }
    return fewest;
}

// How many colors the picked ones of `pieces` hold together, palette(i) piece
// i's colors, a bit each.
template <class Picked, class Palette>
inline int color_count(int pieces, Picked picked, Palette palette) {
    uint32_t colors = 0;
    for (int i = 0; i < pieces; ++i)
        if (picked(i)) colors |= palette(i);
    return popcount(colors);
}

// Whether a Tile names `measure` in any step.
inline bool names(const tile_value& tile, tile_measure measure) {
    for (const tile_value::step& step : tile.steps)
        for (const tile_number* number : { &step.down, &step.across })
            for (const tile_number::term& term : number->terms)
                if (term.measure == measure) return true;
    return false;
}

// What a Tile measures of one piece `height` by `width`, painted(row, col)
// saying which cells it paints, in a hand holding `colors` colors; its gaps
// are read only where the Tile names them.
template <class Painted>
inline tile_measures measures_of(const tile_value& tile, int height, int width, Painted painted, int colors) {
    return { width, height,
             names(tile, tile_measure::row_gap) ? inner_gap(height, width, true, painted) : 0,
             names(tile, tile_measure::column_gap) ? inner_gap(height, width, false, painted) : 0,
             colors };
}

// A Tile's steps as moves in cells, for one piece as measured. A step that
// moves nothing is dropped: every copy of it lies on the piece.
struct tile_moves {
    int down[tile_value::STEPS] = {}, across[tile_value::STEPS] = {};
    int count = 0;
};
inline tile_moves moves_of(const tile_value& tile, const tile_measures& measures) {
    auto measured = [&](tile_measure measure) {
        switch (measure) {
        case tile_measure::own_width:   return measures.own_width;
        case tile_measure::own_height:  return measures.own_height;
        case tile_measure::row_gap:     return measures.row_gap;
        case tile_measure::column_gap:  return measures.column_gap;
        case tile_measure::color_count: return measures.color_count;
        default:                        return 0;
        }
    };
    auto cells = [&](const tile_number& number) {
        int sum = number.count;
        for (const tile_number::term& term : number.terms) sum += (term.minus ? -1 : 1) * measured(term.measure);
        return sum;
    };
    tile_moves moves;
    for (int which = 0; which < tile_value::STEPS; ++which) {
        const int down = cells(tile.steps[which].down), across = cells(tile.steps[which].across);
        if (down == 0 && across == 0) continue;
        moves.down[moves.count] = down;
        moves.across[moves.count] = across;
        ++moves.count;
    }
    return moves;
}

// HOW MUCH OF A PIECE `height` BY `width` ONE COPY IS: its first rows and
// columns, as many as the shortest move a step makes that way, where that is
// shorter than the piece, so no copy lies over the next (5117e062's 3s carried
// down its columns in its first row's colors). {rows, columns}.
inline std::pair<int, int> copy_size(const tile_moves& moves, int height, int width) {
    for (int which = 0; which < moves.count; ++which) {
        if (moves.down[which] != 0) height = min(height, abs(moves.down[which]));
        if (moves.across[which] != 0) width = min(width, abs(moves.across[which]));
    }
    return { height, width };
}

// Whole division rounded down, and up, whatever the signs.
constexpr int floor_div(int a, int b) { return a / b - (a % b != 0 && (a < 0) != (b < 0) ? 1 : 0); }
constexpr int ceil_div(int a, int b) { return -floor_div(-a, b); }

// WHERE A Tile LAYS ITS COPIES of one piece -- `rows` by `columns` of it, its
// top-left at (top, left) -- each offset handed once to visit(down, across),
// which says false to stop: every copy from `start` steps to `end`, as far as
// the grid goes where either says the edge, that still shows on a canvas
// `width` by `height`. Counted from where the piece stands, so a piece off the
// canvas's own count is laid where it lies (4f537728's marker at column 11,
// three apart). In lines each step is taken from the piece alone, step by step,
// each count of it in order; in a grid every copy takes every step again, the
// first step's count outermost.
template <class Visit>
inline void each_tile_offset(const tile_value& tile, const tile_moves& moves, int top, int left, int rows,
                             int columns, int width, int height, Visit visit) {
    // A copy shows where its offset lies strictly between these.
    const int down_low = -top - rows, down_high = height - top;
    const int across_low = -left - columns, across_high = width - left;
    constexpr int edge = 1 << 20;
    const int start = tile.start.edge ? -edge : tile.start.steps, end = tile.end.edge ? edge : tile.end.steps;
    bool seen[2 * MAX_GRID_DIM][2 * MAX_GRID_DIM] = {};
    auto lay = [&](int down, int across) {
        if (down <= down_low || down >= down_high || across <= across_low || across >= across_high) return true;
        const int row = top + down + MAX_GRID_DIM, col = left + across + MAX_GRID_DIM;
        if (row >= 0 && col >= 0 && row < 2 * MAX_GRID_DIM && col < 2 * MAX_GRID_DIM) {
            if (seen[row][col]) return true;
            seen[row][col] = true;
        }
        return visit(down, across);
    };
    // Two steps along one line make no grid -- a single step says what they
    // lay -- so they are taken one at a time, as lines are.
    const int det = moves.count < 2 ? 0 : moves.down[0] * moves.across[1] - moves.down[1] * moves.across[0];
    if (tile.spread == tile_spread::lines || det == 0) {
        for (int which = 0; which < moves.count; ++which) {
            // The counts of this step whose copy can show, within start and end.
            int least = start, most = end;
            auto within = [&](int low, int high, int move) {
                if (move == 0) return;
                if (move < 0) {
                    move = -move;
                    low = -low;
                    high = -high;
                    std::swap(low, high);
                }
                least = max(least, floor_div(low, move) + 1);
                most = min(most, ceil_div(high, move) - 1);
            };
            within(down_low, down_high, moves.down[which]);
            within(across_low, across_high, moves.across[which]);
            for (int steps = least; steps <= most; ++steps)
                if (!lay(steps * moves.down[which], steps * moves.across[which])) return;
        }
        return;
    }
    // A GRID: the counts of the two steps that reach the canvas's corners bound
    // every copy that shows.
    int least[2] = { edge, edge }, most[2] = { -edge, -edge };
    for (const int down : { down_low, down_high })
        for (const int across : { across_low, across_high }) {
            const int first = down * moves.across[1] - across * moves.down[1];
            const int second = across * moves.down[0] - down * moves.across[0];
            least[0] = min(least[0], floor_div(first, det));
            most[0] = max(most[0], ceil_div(first, det));
            least[1] = min(least[1], floor_div(second, det));
            most[1] = max(most[1], ceil_div(second, det));
        }
    for (int first = max(least[0], start); first <= min(most[0], end); ++first)
        for (int second = max(least[1], start); second <= min(most[1], end); ++second)
            if (!lay(first * moves.down[0] + second * moves.down[1], first * moves.across[0] + second * moves.across[1]))
                return;
}

// HOW A COPY MIRRORS ITS PIECE around a middle: left to right, top to bottom,
// or all four ways at once. The middle is a partner's, one of the partner_by
// ways, or past those the grid's own -- so MIRROR_PARTNERS counts one more.
enum class mirror_by : int8_t { left_right, top_bottom, four_ways, count };
constexpr int MIRROR_PARTNERS = (int)partner_by::count + 1;

// WHERE MirrorAround DRAWS, said once for the op and for the reading that
// replays it. The middle a picked piece is mirrored around, twice its row and
// column so a middle between two cells is whole: its one partner's by `how`
// among the pieces not picked (partner_of), or past those the grid's own --
// false for a piece with no one partner, which gets no copy.
template <class Pieces>
inline bool mirror_middle(int how, int mine, const Pieces& pieces, int width, int height,
                          int& row2, int& col2) {
    row2 = height - 1;
    col2 = width - 1;
    if (how >= (int)partner_by::count) return true;
    const int partner = partner_of((partner_by)how, mine, pieces);
    if (partner < 0) return false;
    const cell_box there = pieces.box(partner);
    row2 = there.row0 + there.row1;
    col2 = there.col0 + there.col1;
    return true;
}
// Each copy the mirror draws, handed to `copy(down, across)`, the ways it flips:
// across for left to right, down for top to bottom, and all four ways each
// alone and then both.
template <class Copy>
inline void each_mirror(mirror_by mirror, Copy copy) {
    for (int way = 0; way < 3; ++way)
        if (mirror == mirror_by::four_ways || way == (int)mirror) copy(way != 0, way != 1);
}
// Where a copy lies, its far side where the piece's near side was, and the pose
// its cells are read through (pose_source): flipped top to bottom where it is
// mirrored down, left to right where across.
inline cell_box mirrored_box(const cell_box& box, int row2, int col2, bool down, bool across) {
    return { down ? row2 - box.row1 : box.row0, across ? col2 - box.col1 : box.col0,
             down ? row2 - box.row0 : box.row1, across ? col2 - box.col0 : box.col1 };
}
constexpr int mirror_pose(bool down, bool across) { return (down ? 2 : 0) | (across ? 4 : 0); }

}  // namespace arc
