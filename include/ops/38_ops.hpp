// THE OPS: what each op does to the list a program holds and to the canvas it
// paints, run one after another on one example (environment::draw). What an op
// shares with the reading's replay is 15_op_math.hpp's arithmetic, so the two
// cannot draw differently. Only arc_headless has include/ops on its include
// path, so only it can run a program: the GUI cannot.
#pragma once
#include "20_environment.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace arc {
namespace {

// The lists one run works in, made once a thread: lists as long as a grid has
// cells are far past what a thread's own storage should hold.
struct run_lists {
    object_list input_pieces, seeds, pool[2];
};

// ONE PROGRAM ON ONE EXAMPLE: what its ops read -- the input, and the facts the
// reading learned -- and what they write: the canvas, and the list in hand.
struct program_run {
    const grid& input;
    grid canvas;
    run_lists& lists;
    // WHERE THIS EXAMPLE'S PANES LIE. Geometry, not cells: Panes clips whatever
    // the chain holds to these boxes, so what counts as figure stays the
    // caller's operators (Colors, SelectColor, DeleteSelection).
    vector<pane_split_facts::pane> panes;
    // The training outputs' colors, most cells first (SizeRankColorMap).
    const int8_t* colors_by_size;
    int colors_by_size_count;
    // The learned shape -> color table (ShapeColorMap and kin).
    const shape_table& shape_colors;
    // The shape the pairs' changes take (changed_shape), the one Split [4] cuts by.
    const turning_shape& cut_shape;
    // How NormalizeOrientation posed this example, for RestoreOrientation to
    // undo exactly that: 1 when it laid a landscape input upright, 8 | the pose
    // while a canonical pose is active.
    int8_t orientation_flipped = 0;
    // The list the last op handed on; none before the first.
    object_list* current = nullptr;

    // A list for the next op to write: the one not in hand, emptied, nothing
    // picked. Two are enough: the list an op reads and the list it writes.
    // Fresh lists have NOTHING selected (Excel model), and every list flows
    // through here, so this is where that holds.
    object_list& fresh() {
        object_list& list = current == &lists.pool[0] ? lists.pool[1] : lists.pool[0];
        list.count = 0;
        for (held_object& one : list.objects) one.selected = false;
        return list;
    }
};

// Color-argument sentinel meaning "any non-transparent color" (value 10, one past
// the 0..9 real colors). Used by MaskBy[10]/MaskOut[10] to match on the canvas
// being painted at all, regardless of which color.
constexpr int8_t COLOR_ANY = COLOR_COUNT;

// --- Shared building blocks ---

// Every painted cell of an object, in its own frame: see(row, col, color).
template <class See>
void each_painted(const held_object& one, See see) {
    for (int row = 0; row < one.height; ++row)
        for (int col = 0; col < one.width; ++col)
            if (const int8_t color = one.data[row][col]; color != TRANSPARENT) see(row, col, color);
}

// The box the paint fills in the object's own frame, which is not always its
// frame -- Colors hands back layers carrying the whole source box.
cell_box paint_in_frame(const held_object& one) {
    cell_box box;
    each_painted(one, [&](int row, int col, int8_t) { box.include_cell(row, col); });
    return box;
}

// `source` cut to `box`, a box of its own frame: the same object, so it keeps
// its mark.
held_object cut_to(const held_object& source, const cell_box& box) {
    held_object one;
    one.selected = source.selected;
    one.col = (int16_t)(source.col + box.col0); one.row = (int16_t)(source.row + box.row0);
    one.width = (int16_t)box.width(); one.height = (int16_t)box.height();
    for (int row = 0; row < one.height; ++row)
        for (int col = 0; col < one.width; ++col) one.data[row][col] = source.data[box.row0 + row][box.col0 + col];
    return one;
}

// Every cell an object paints, in its own frame, painted `color`.
void recolor(held_object& one, int8_t color) {
    for (int row = 0; row < one.height; ++row)
        for (int col = 0; col < one.width; ++col)
            if (one.data[row][col] != TRANSPARENT) one.data[row][col] = color;
}

// Shared color-source rule for painting ops: resolve an annotation value to a
// per-object color. 0..9 = that literal; 10 = the object's rarest color;
// 11 = its most common. Frequency ties break toward the lower color number.
// TRANSPARENT if it paints no cell.
int8_t paint_color(const held_object& one, int8_t value) {
    return object_paint_color(one.height, one.width, value,
        [&](int row, int col) { return (int8_t)one.data[row][col]; });
}

// The first `count` objects of `from`, as they are, into `to`.
void copy_objects(object_list& to, const object_list& from, int count) {
    for (int i = 0; i < count; ++i) to.objects[i] = from.objects[i];
    to.count = count;
}

// A list an op hands on as it came.
void copy_list(object_list& to, const object_list& from) {
    copy_objects(to, from, clamp(from.count, 0, MAX_OBJECTS_IN_A_LIST));
}

// The objects drawn onto a width x height grid, a later one over an earlier.
void paint_objects(const object_list& from, int width, int height, grid& out) {
    out.width = width;
    out.height = height;
    out.clear_cells();
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        each_painted(one, [&](int row, int col, int8_t color) {
            const int at_row = one.row + row, at_col = one.col + col;
            if (at_row >= 0 && at_row < MAX_GRID_DIM && at_col >= 0 && at_col < MAX_GRID_DIM)
                out.data[at_row][at_col] = color;
        });
    }
}

// The pieces of `g`, each in the frame of what they were cut from (col, row,
// width, height, which holds every painted cell of g). What counts as one
// object is arc::each_object's to say, or cut by a shape each_shape_piece's,
// as the reading's is, so the two cannot come to different answers;
// segmentation never tightens a frame (Trim is the only shrinker). Cut by a
// shape, its places come back picked and the rest not, as Bridge hands back
// its bars, so an operator after it acts on the places.
void segment_objects(const grid& g, object_list& out, grouping how, int x0 = 0, int y0 = 0, int width = -1,
                     int height = -1, const turning_shape* shape = nullptr) {
    // No frame given: the grid's own.
    if (width < 0) width = g.width;
    if (height < 0) height = g.height;
    out.count = 0;
    if (g.width <= 0 || g.height <= 0 || width <= 0 || height <= 0) return;
    object_walk_scratch scratch;
    auto take = [&](const object_summary&, const int16_t* taken, int taken_count) {
        if (out.count >= MAX_OBJECTS_IN_A_LIST) return;
        held_object& one = out.objects[out.count++];
        one.col = (int16_t)x0; one.row = (int16_t)y0;
        one.width = (int16_t)width; one.height = (int16_t)height;
        one.data.clear();
        for (int i = 0; i < taken_count; ++i) {
            const int row = cell_row(taken[i]), col = cell_col(taken[i]);
            one.data[row - y0][col - x0] = g.data[row][col];
        }
    };
    if (!shape) {
        each_object(g, how, scratch, take);
        return;
    }
    const int places = each_shape_piece(g, *shape, scratch, take);
    for (int i = 0; i < places && i < out.count; ++i) out.objects[i].selected = true;
}

// Calls cell(row, col, color) for each painted, on-canvas cell of the picked
// objects among from's first `count`, one object at a time so later objects win
// shared cells.
template <class Cell>
void each_picked_cell(const program_run& run, const object_list& from, int count, Cell cell) {
    for (int i = 0; i < count; ++i) {
        const held_object& one = from.objects[i];
        if (!one.selected) continue;
        each_painted(one, [&](int row, int col, int8_t color) {
            const int at_row = one.row + row, at_col = one.col + col;
            if (at_row >= 0 && at_row < run.canvas.height && at_col >= 0 && at_col < run.canvas.width)
                cell(at_row, at_col, color);
        });
    }
}

// Each picked object of `from` rebuilt into `res` -- size(source, width,
// height) says its new size, pixel(source, row, col) each of its cells -- and
// the rest handed on as they are.
template <class Size, class Pixel>
void remap_picked(const object_list& from, object_list& res, Size size, Pixel pixel) {
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < count; ++i) {
        const held_object& source = from.objects[i];
        held_object& one = res.objects[i] = source;
        if (!source.selected) continue;
        int width = 0, height = 0;
        size(source, width, height);
        for (int at = 0; at < MAX_GRID_CELLS; ++at) {
            const int row = cell_row(at), col = cell_col(at);
            one.data.values[at] = row < height && col < width ? pixel(source, row, col) : (int8_t)TRANSPARENT;
        }
        one.width = (int16_t)width;
        one.height = (int16_t)height;
    }
    res.count = count;
}

// Which objects of a list are picked.
vector<bool> picks_of(const object_list& list) {
    vector<bool> picked((size_t)list.count);
    for (int i = 0; i < list.count; ++i) picked[(size_t)i] = list.objects[i].selected;
    return picked;
}

// THE LIST AS THE PARTNER HELPERS ASK IT (partner_pieces, 15_op_math.hpp): a
// copy of its objects and which are picked, on the input's canvas, asked as a
// rule's replay asks the pieces it replays.
struct listed_objects {
    vector<object> objects;
    hand_answers answered;
};
struct listed_pieces : private listed_objects, partner_pieces {
    listed_pieces(const object_list& list, const grid& input)
        : listed_objects{ vector<object>(list.objects, list.objects + list.count),
                          hand_answers({ input.width, input.height }) },
          partner_pieces(objects, picks_of(list), answered) {}
};

// --- Sources ---

// Input: the input, read whole. Every cell of an input is colored, so its frame
// is its content's too.
object_list& op_input(program_run& run) {
    object_list& res = run.fresh();
    segment_objects(run.input, res, grouping::whole_grid);
    return res;
}

// Output is a snapshot of the WHOLE canvas, including its empty frame, read
// back as it is with its transparent cells kept, so what follows sees what has
// been painted so far. Trim is the only operator that reduces this frame to its
// painted bounds.
object_list& op_output(program_run& run) {
    object_list& res = run.fresh();
    segment_objects(run.canvas, res, grouping::whole_grid);
    return res;
}

// --- Reduce ---

// Split[depth]: segmentation depth, coarse to fine. [0] = connected object,
// colors mixed. [1] = connected object of one color. [2] = single pixels.
// [3], finer than [1] and coarser than [2], came last: one color's cells
// joined by sides alone, a corner joining nothing. [4] takes every place the
// shape the pairs' changes take (run.cut_shape) fits on the colors it turns
// from and into as a piece, picked, then cuts the rest as [1] does.
object_list& op_split(program_run& run, const object_list& from, int8_t depth) {
    object_list& res = run.fresh();
    grid drawn;
    // The whole 30x30 ceiling, not the input's size: what is held may have
    // grown past the input's box (Tile, Shift, Extend...), and an input-sized
    // canvas would crop those cells before they were cut.
    paint_objects(from, MAX_GRID_DIM, MAX_GRID_DIM, drawn);
    if (depth == 2) {
        // One 1x1 object per painted cell; a list holds one object per cell, so all fit.
        for (int row = 0; row < MAX_GRID_DIM && res.count < MAX_OBJECTS_IN_A_LIST; ++row)
            for (int col = 0; col < MAX_GRID_DIM && res.count < MAX_OBJECTS_IN_A_LIST; ++col) {
                const int8_t color = drawn.data[row][col];
                if (color == TRANSPARENT) continue;
                held_object& one = res.objects[res.count++];
                one.col = (int16_t)col;
                one.row = (int16_t)row;
                one.width = one.height = 1;
                one.data.clear();
                one.data[0][0] = color;
            }
        return res;
    }
    // EACH PIECE KEEPS THE FRAME OF WHAT IT WAS CUT FROM: the frame the objects
    // it was handed span, as UnionList's one object spans it -- the whole input
    // where the input was cut -- the way Colors' layers keep theirs. So Tile
    // after it lays each piece's copies the input's size apart, a copy apiece
    // (10fcaaa3: cut, then tiled, the copies never join).
    int x0 = MAX_GRID_DIM, y0 = MAX_GRID_DIM, x1 = -1, y1 = -1;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        x0 = min(x0, (int)one.col); y0 = min(y0, (int)one.row);
        x1 = max(x1, one.col + one.width - 1); y1 = max(y1, one.row + one.height - 1);
    }
    x0 = max(x0, 0); y0 = max(y0, 0);
    x1 = min(x1, MAX_GRID_DIM - 1); y1 = min(y1, MAX_GRID_DIM - 1);
    segment_objects(drawn, res,
                    depth == 0 ? grouping::touching_any_color
                    : depth == 3 ? grouping::touching_sides_same_color
                                 : grouping::touching_same_color,
                    x0, y0, x1 - x0 + 1, y1 - y0 + 1, depth == 4 ? &run.cut_shape : nullptr);
    return res;
}

// Colors splits each source object into ALL its color layers, unselected.
// Every layer keeps the source frame. Removal and trimming are separate
// operators.
object_list& op_colors(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count; ++i) {
        const held_object& source = from.objects[i];
        const int width = source.width, height = source.height;
        for (int color = 0; color < COLOR_COUNT; ++color) {
            bool found = false;
            for (int row = 0; row < height && !found; ++row)
                for (int col = 0; col < width; ++col)
                    if (source.data[row][col] == color) { found = true; break; }
            if (!found) continue;
            if (res.count >= MAX_OBJECTS_IN_A_LIST) return res;
            held_object& layer = res.objects[res.count];
            layer.col = source.col; layer.row = source.row;
            layer.width = source.width; layer.height = source.height;
            // A carve CLEARS, and that is what makes a carve useful: the pick
            // that follows it is how a sentence narrows to one layer. Handing
            // the mark down instead leaves every layer marked, and since
            // selectors only ever ADD, the next pick can no longer say "just
            // this one" -- measured on 05a7bcf2, where it recoloured all three.
            layer.selected = false;
            layer.data.clear();
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    if (source.data[row][col] == color) layer.data[row][col] = (int8_t)color;
            res.count++;
        }
    }
    return res;
}

// --- Select ---

// SelectMostCommonColor: mark the objects holding the most frequent color
// across the whole list (the count the reading replays, color_tally). After a
// carve (Colors/Split) each object is single-color, so this is exactly "the
// top-color object(s)".
object_list& op_select_most_common_color(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    copy_list(res, from);
    color_tally freq;
    for (int i = 0; i < res.count; ++i) {
        const held_object& one = res.objects[i];
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col) freq.add((int8_t)one.data[row][col]);
    }
    const int top = freq.most();
    if (top < 0) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        for (int row = 0; row < one.height && !one.selected; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.data[row][col] == top) { one.selected = true; break; }
    }
    return res;
}

// SortBySize: reorder the PICKED objects by pixel count (size_order), among the
// slots they already occupy. Everything unpicked stays exactly where it is, so
// the act touches only what was chosen -- the same law the other acts follow,
// said for an ordering: a sort of some is a sort of their own places. Biggest
// first when `biggest_first`. Nothing picked means nothing is reordered.
object_list& op_sort_by_size(program_run& run, const object_list& from, bool biggest_first) {
    object_list& res = run.fresh();
    if (from.count <= 0) return res;
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    copy_objects(res, from, count);
    int slots[MAX_OBJECTS_IN_A_LIST], order[MAX_OBJECTS_IN_A_LIST];
    int picked = 0;
    for (int i = 0; i < count; ++i)
        if (from.objects[i].selected) slots[picked++] = i;
    size_order(picked, biggest_first, [&](int k) { return from.objects[slots[k]].cell_count(); }, order);
    // The k-th picked slot receives the k-th picked object in size order.
    for (int k = 0; k < picked; ++k) res.objects[slots[k]] = from.objects[slots[order[k]]];
    return res;
}

// SizeRankColorMap: recolor each object by the puzzle's by-size palette at its
// rank. An empty palette hands back nothing; objects past the palette drop.
object_list& op_size_rank_color_map(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int ranked = run.colors_by_size_count;
    if (ranked <= 0) return res;
    const int count = min(min(from.count, ranked), MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < count; ++i) recolor(res.objects[res.count++] = from.objects[i], run.colors_by_size[i]);
    return res;
}

// Add a mark to the one object with no twin, as `how` reads twins
// (unique_piece); keep every object and prior mark.
object_list& op_select_unique(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const int found = unique_piece(how, from.count,
        [&](int i) { return from.objects[i].width; },
        [&](int i) { return from.objects[i].height; },
        [&](int i, int row, int col) { return (int8_t)from.objects[i].data[row][col]; });
    if (found >= 0) res.objects[found].selected = true;
    return res;
}

// SelectLargestObject[rank] / SelectSmallestObject[rank]: mark the rank-th
// object by pixel count, biggest (or smallest) first (size_order). Past the
// ranks, every object with the most -- or the fewest -- of a count (count_of).
object_list& op_select_object_by_size(program_run& run, const object_list& from, int rank, bool largest) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (rank >= SIZE_RANKS) {
        const int kind = rank - SIZE_RANKS;
        each_extreme(res.count, largest, [&](int i) {
            const held_object& one = res.objects[i];
            return count_of(kind, min((int)one.height, MAX_GRID_DIM), min((int)one.width, MAX_GRID_DIM),
                            [&](int row, int col) { return (int)one.data[row][col]; });
        }, [&](int i) { res.objects[i].selected = true; });
        return res;
    }
    if (rank < 0 || rank >= res.count) return res;
    int order[MAX_OBJECTS_IN_A_LIST];
    size_order(res.count, largest, [&](int i) { return res.objects[i].cell_count(); }, order);
    res.objects[order[rank]].selected = true;
    return res;
}

// `source` clipped to rows [row0, row1) and columns [col0, col1), and there to
// the cells `keep(row, col)` holds, in that frame; false when nothing is left.
template <class Keep>
bool clip_to(const held_object& source, int row0, int col0, int row1, int col1, Keep keep, held_object& one) {
    one.col = (int16_t)col0; one.row = (int16_t)row0;
    one.width = (int16_t)(col1 - col0); one.height = (int16_t)(row1 - row0);
    one.selected = false;
    one.data.clear();
    bool any = false;
    each_painted(source, [&](int row, int col, int8_t color) {
        const int at_row = source.row + row, at_col = source.col + col;
        if (at_row < row0 || at_row >= row1 || at_col < col0 || at_col >= col1 || !keep(at_row, at_col)) return;
        one.data[at_row - row0][at_col - col0] = color;
        any = true;
    });
    return any;
}

// Panes: Partition's act with the cut already decided. Partition reads the
// walls off a selection; the elected split read them off every training pair at
// once, so here the boxes simply come up with the example. What is cut is
// whatever the chain holds -- explicitly remove the ground first and the panes come
// back as figures, hand it Colors and they come back a layer at a time.
object_list& op_panes(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (const pane_split_facts::pane& pane : run.panes) {
        if (pane.width <= 0 || pane.height <= 0) continue;
        for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
            held_object one;
            clip_to(from.objects[i], pane.top, pane.left, pane.top + pane.height, pane.left + pane.width,
                    [](int, int) { return true; }, one);
            // EVERY PANE COMES BACK, even one the source never reached -- empty,
            // but there. Partition drops such a tile because its tiles are
            // whatever the walls happen to leave; panes are a fixed set that gets
            // stacked, so a missing one would silently shift every later pane
            // into the wrong place in the list.
            res.objects[res.count++] = one;
        }
    }
    return res;
}

// --- Combine ---

// Intersection: canvas := canvas ∩ picked -- clear every canvas cell no picked
// object paints -- then hand the list on. Union (canvas ∪ picked) and
// Difference (canvas − picked) are its kin: each writes the canvas only.
object_list& op_intersection(program_run& run, const object_list& from) {
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    bool covered[MAX_GRID_CELLS] = {};
    each_picked_cell(run, from, count, [&](int row, int col, int8_t) { covered[cell_index(row, col)] = true; });
    for (int row = 0; row < run.canvas.height; ++row)
        for (int col = 0; col < run.canvas.width; ++col)
            if (!covered[cell_index(row, col)]) run.canvas.data[row][col] = TRANSPARENT;
    object_list& res = run.fresh();
    copy_objects(res, from, count);
    return res;
}

// Union: lay the picked objects' cells on the canvas, a later object over an
// earlier one, and hand the list on unchanged. Union must NOT collapse the list
// into one object (that's UnionList); keeping it lets a later per-object op
// still see each object.
object_list& op_union(program_run& run, const object_list& from) {
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    each_picked_cell(run, from, count, [&](int row, int col, int8_t color) { run.canvas.data[row][col] = color; });
    object_list& res = run.fresh();
    copy_objects(res, from, count);
    return res;
}

// Withhold: closes the sentence and paints nothing. The hand rides through so
// the next sentence can act on it, and the canvas is left exactly as it was --
// which is the whole point: a sentence that only prepares should read as having
// done nothing, not as having damaged the picture.
object_list& op_withhold(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    copy_objects(res, from, min(from.count, MAX_OBJECTS_IN_A_LIST));
    return res;
}

// Underlay: like Union but paints each cell ONLY into transparent canvas cells
// (paint-behind). One-op, non-destructive z-order -- "put this under what's
// already there" instead of reordering the chain.
object_list& op_underlay(program_run& run, const object_list& from) {
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    each_picked_cell(run, from, count, [&](int row, int col, int8_t color) {
        lay_under(run.canvas, row, col, color);
    });
    object_list& res = run.fresh();
    copy_objects(res, from, count);
    return res;
}

// Difference: erase from the canvas every cell a picked object paints. Only
// what was picked lands: a paint that ignored the mark would undo the pick the
// sentence just made.
object_list& op_difference(program_run& run, const object_list& from) {
    each_picked_cell(run, from, from.count, [&](int row, int col, int8_t) { run.canvas.data[row][col] = TRANSPARENT; });
    object_list& res = run.fresh();
    copy_list(res, from);
    return res;
}

// --- Map ---

// The object with every empty cell its own paint walls off from its frame's
// edge (grid_parts' hollows) painted `fill`.
void paint_interior(const held_object& from, held_object& out, int8_t fill) {
    out = from;
    grid_parts(from.width, from.height, [&](int row, int col) { return from.data[row][col] != TRANSPARENT; })
        .each_hollow([&](int row, int col) { out.data[row][col] = fill; });
}

// Fills the interior of each picked, non-empty object with fill(object); a
// negative fill leaves the object as it is. Empty objects drop out.
template <class Fill>
object_list& fill_interiors(program_run& run, const object_list& from, Fill fill) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    int kept = 0;
    for (int i = 0; i < count; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        const int color = one.selected ? fill(one) : -1;
        if (color < 0) res.objects[kept] = one;
        else paint_interior(one, res.objects[kept], (int8_t)color);
        ++kept;
    }
    res.count = kept;
    return res;
}

object_list& op_interior_color(program_run& run, const object_list& from, int8_t color) {
    return fill_interiors(run, from, [&](const held_object& one) {
        return color < 0 ? (int)one.first_color() : (int)color;
    });
}

// Recolor: every painted cell of a picked object takes `color`; an unpicked
// object rides through untouched. No selection in the chain means Recolor
// changes nothing — acts NEVER fall back to "everything".
object_list& op_recolor(program_run& run, const object_list& from, int8_t color) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < count; ++i) {
        const held_object& source = from.objects[i];
        held_object& one = res.objects[i] = source;
        // Past the ten colors, the color is read per object: 10 its rarest, 11
        // its most common.
        if (source.selected) recolor(one, color >= COLOR_COUNT ? paint_color(source, color) : color);
    }
    res.count = count;
    return res;
}

// Bridge[way]: the pieces handed back as they came, unmarked, then every bar as
// one drawing, marked, so what follows works on the bars. Which pieces pair up
// (the picked ones alone), the way they reach, and where each bar lies is
// 15_op_math.hpp's bridge_bars, which the reading replays too; this only lays
// the bars down, each cell to the first bar that crosses it. One drawing, so a
// list that holds the pieces holds every bar too, however many gaps there are --
// one per gap ran out of room (dbc1a6ce's test, 19 dots and 17 bars). It sits
// in the whole grid's frame, as Split's pieces do.
object_list& op_bridge(program_run& run, const object_list& from, int8_t way) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        res.objects[res.count] = from.objects[i];
        res.objects[res.count++].selected = false;
    }
    if (res.count >= MAX_OBJECTS_IN_A_LIST) return res;
    // A list slot keeps whatever it last held, so every field is said.
    held_object& bars = res.objects[res.count];
    bars.col = 0; bars.row = 0;
    bars.width = bars.height = (int16_t)MAX_GRID_DIM;
    bars.selected = true;
    bars.data.clear();
    bool drew = false;
    bridge_bars(way, run.canvas.width, run.canvas.height, listed_pieces(from, run.input),
        [&](const cell_box& bar, int8_t color) {
            for (int row = max(bar.row0, 0); row <= min(bar.row1, MAX_GRID_DIM - 1); ++row)
                for (int col = max(bar.col0, 0); col <= min(bar.col1, MAX_GRID_DIM - 1); ++col)
                    if (bars.data[row][col] == TRANSPARENT) {
                        bars.data[row][col] = color;
                        drew = true;
                    }
            return true;
        });
    if (drew) ++res.count;
    return res;
}

// BidiColorMap[k]: the picked objects trade colors, each pixel of one color of
// a pair becoming the other. k in [0,44] is one fixed pair {a,b}, 0<=a<b<=9,
// numbered row-major (k=0 -> {0,1}, k=1 -> {0,2}, ... k=44 -> {8,9}); past
// those, the pairs a key drawn in the grid names, the objects not picked read
// along their rows [45] or their columns [46] -- 0becf7df's corner, a pair a
// row, other colors in every grid. What it trades is swaps_of's to say
// (15_op_math.hpp), the table inference replays, so the two cannot disagree.
// Self-inverse (a swap), so it expresses involutive recolor tasks where
// Recolor/MapTo cannot: one node does a->b and b->a with identical program
// behavior, surviving examples that need the swap in either direction (e.g.
// 0d3d703e: chain Swap{1,5},{2,6},{3,4},{8,9}). Colors not in a pair are
// untouched, so disjoint swaps compose order-free.
object_list& op_bidi_color_map(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    // One color with empty, over every cell of each picked object's frame.
    if (value >= EMPTY_PAIRS) {
        for (int i = 0; i < res.count; ++i) {
            held_object& one = res.objects[i];
            if (!one.selected) continue;
            for (int row = 0; row < one.height; ++row)
                for (int col = 0; col < one.width; ++col)
                    one.data[row][col] = traded_with_empty(one.data[row][col], value - EMPTY_PAIRS);
        }
        return res;
    }
    int8_t to[COLOR_COUNT];
    if (!swaps_of(value, listed_pieces(from, run.input), to)) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        // Only what was picked swaps colors; the rest ride through as they are.
        if (!one.selected) continue;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col) {
                const int8_t color = one.data[row][col];
                if (is_color(color) && to[color] >= 0) one.data[row][col] = to[color];
            }
    }
    return res;
}

// ColorFrom [the key at its place], past the partner ways: each picked object
// takes the color a key drawn in the grid -- the objects not picked -- shows at
// its place, the n-th picked in reading order the key's n-th entry
// (key_place_colors, 15_op_math.hpp, the walk inference replays).
void paint_key_places(const object_list& from, object_list& res, const grid& input) {
    int8_t color[MAX_OBJECTS_IN_A_LIST], band[MAX_OBJECTS_IN_A_LIST];
    int16_t lows[MAX_OBJECTS_IN_A_LIST], highs[MAX_OBJECTS_IN_A_LIST];
    int order[MAX_OBJECTS_IN_A_LIST];
    if (!key_place_colors(listed_pieces(from, input), color, lows, highs, band, order)) return;
    for (int i = 0; i < res.count; ++i)
        if (color[i] >= 0) recolor(res.objects[i], color[i]);
}

// ColorFrom[how]: each picked object takes the color of its partner among the
// objects not picked, each way partner_by finds one -- or the key's one color,
// the objects not picked as one drawing (key_color) -- and paints every cell it
// paints in it. The partner is found by partner_color (15_op_math.hpp), the
// walk inference replays, so the two cannot disagree. An object with no
// partner, or partners that disagree, keeps its colors, as every one does
// where the key wears several. Nothing moves and nothing is dropped.
object_list& op_color_from(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (how == KEY_BY_PLACE) {
        paint_key_places(from, res, run.input);
        return res;
    }
    const listed_pieces view(from, run.input);
    if (how == KEY_COLOR) {
        const int color = key_color(view);
        for (int i = 0; i < res.count && color >= 0; ++i)
            if (res.objects[i].selected) recolor(res.objects[i], (int8_t)color);
        return res;
    }
    // Cell by cell, from the one piece of its shape (colors_from_its_shape).
    if (how >= COLORS_FROM_ITS_SHAPE) {
        for (int i = 0; i < from.count; ++i) {
            if (!from.objects[i].selected) continue;
            held_object& one = res.objects[i];
            colors_from_its_shape(how - COLORS_FROM_ITS_SHAPE, i, view, [&](int row, int col, int color) {
                const int down = row - one.row, across = col - one.col;
                if (down >= 0 && across >= 0 && down < one.height && across < one.width && is_color(color))
                    one.data[down][across] = (int8_t)color;
            });
        }
        return res;
    }
    if (how < 0 || how >= (int8_t)partner_by::count) return res;
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        if (const int color = view.color_from((partner_by)how, i); color >= 0) recolor(res.objects[i], (int8_t)color);
    }
    return res;
}

// ShapeFrom[how]: each picked object becomes its one partner among the objects
// not picked, the way partner_by `how` finds it, mirrored across the partner's
// side that faces it, in the color most of the object wears -- its old cells
// gone, its frame the new drawing's box (each_shape_taken, 15_op_math.hpp, the
// walk inference replays). An object with no one partner, or one not wholly
// past a side of it, stays as it is.
object_list& op_shape_from(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    // Each new drawing gathered whole on a grid of its own, then laid as its
    // object's cells; every partner is read off the list as it came.
    int taking = -1;
    cells drawn;
    cell_box box;
    auto set_drawing = [&] {
        if (taking < 0) return;
        held_object& one = res.objects[taking];
        one.data.clear();
        for (int row = box.row0; row <= box.row1; ++row)
            for (int col = box.col0; col <= box.col1; ++col) one.data[row - box.row0][col - box.col0] = drawn[row][col];
        one.col = (int16_t)box.col0; one.row = (int16_t)box.row0;
        one.width = (int16_t)box.width(); one.height = (int16_t)box.height();
    };
    each_shape_taken(how, listed_pieces(from, run.input), [&](int i, int row, int col, int color) {
        if (i != taking) {
            set_drawing();
            taking = i;
            drawn.clear();
            box = {};
        }
        drawn[row][col] = (int8_t)color;
        box.include_cell(row, col);
    });
    set_drawing();
    return res;
}

// WouldFit[how]: of the picked objects, only those that would fit into a gap of
// their one partner among the objects not picked stay picked -- the partner by
// `how`, as partner_of (15_op_math.hpp) finds it, and the fit as would_fit asks
// it, turned or mirrored any way: the walk inference replays, so the two cannot
// disagree. Every partner is read off the marks as they came in. Nothing moves
// and nothing is painted; an object with no partner, or several, is not picked.
object_list& op_would_fit(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (how < 0 || how >= (int8_t)partner_by::count) return res;
    const listed_pieces view(from, run.input);
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int partner = view.partner((partner_by)how, i);
        res.objects[i].selected = partner >= 0 && would_fit(i, partner, view);
    }
    return res;
}

// SelectLikeKey[how]: of the picked objects, only those like a piece of the key
// -- every object not picked -- stay picked (like_the_key, 15_op_math.hpp), the
// question inference replays. Every likeness is read off the marks as they came in.
object_list& op_select_like_key(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const listed_pieces view(from, run.input);
    for (int i = 0; i < from.count; ++i)
        if (from.objects[i].selected) res.objects[i].selected = like_the_key(how, i, view);
    return res;
}

// SelectPartner[value]: the pick moves onto every partner each picked object
// has, among every object, by the way and on the side `value` names
// (each_partner_picked, 15_op_math.hpp, which inference replays). Every
// partner is read off the marks as they came in; an object no picked one has
// for a partner comes out unpicked.
object_list& op_select_partner(program_run& run, const object_list& from, int value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = false;
    each_partner_picked(value, listed_pieces(from, run.input), [&](int other) { res.objects[other].selected = true; });
    return res;
}

// SelectPartition[which]: the picked objects are walls, as Partition reads them,
// and the pick becomes every object lying in the part `which` they fence off
// (parts_the_pick_fences and lies_in_part, 15_op_math.hpp, which inference
// replays), the parts numbered in reading order. The walls lie in no part, so
// they come out unpicked. The canvas is the input's, as Partition's is.
object_list& op_select_partition(program_run& run, const object_list& from, int which) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const listed_pieces view(from, run.input);
    const grid_parts parts = parts_the_pick_fences(view, run.input.width, run.input.height);
    for (int i = 0; i < from.count; ++i) res.objects[i].selected = lies_in_part(view, i, which, parts);
    return res;
}

// Partition: the selected objects are the walls. Strict, like every act: no
// selection means no walls, so the whole thing stays one part -- Partition
// NEVER falls back to "all painted are walls". The walls come back first,
// unmarked, so SelectPartition can still read them; then the parts they fence
// off, whatever their shape, part after part in reading order
// (each_fenced_part, 15_op_math.hpp, the parts SelectPartition picks by):
// every other object clipped to the part, in the part's box -- severed where it
// crosses into another, which is what partitioning means. Cut before the ground
// is deleted, the ground comes back a part at a time: each part's room. A
// part's cells no object holds stay empty. Walls across the whole grid fence
// the tiles of a lattice, row by row. No merge: distinct source layers stay
// distinct; want each part as one object? place a UnionList yourself.
object_list& op_partition(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int grid_width = run.input.width, grid_height = run.input.height;
    if (grid_width <= 0 || grid_height <= 0) return res;
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i)
        if (from.objects[i].selected) {
            res.objects[res.count] = from.objects[i];
            res.objects[res.count++].selected = false;
        }
    each_fenced_part(listed_pieces(from, run.input), grid_width, grid_height,
        [&](int, const cell_box& box, auto in_part) {
            for (int index = 0; index < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++index) {
                if (from.objects[index].selected) continue;
                held_object one;
                if (clip_to(from.objects[index], box.row0, box.col0, box.row1 + 1, box.col1 + 1, in_part, one))
                    res.objects[res.count++] = one;
            }
        });
    return res;
}

// --- Translate ---

// AlignTo[value]: each picked object slides, down or across alone, until its
// painted box starts on the row its one partner's does -- or, for values past
// the first partner_by::count, the column. The partner is found among the
// objects not picked by partner_of (15_op_math.hpp), the walk inference
// replays: value % count says how. Partners are never picked, so none of them
// moves while the others read them. An object with no partner, or several,
// stays where it is.
object_list& op_align_to(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    constexpr int ways = (int)partner_by::count;
    if (value < 0 || value >= 2 * ways) return res;
    const bool columns = value >= ways;
    const auto how = (partner_by)(value % ways);
    const listed_pieces view(from, run.input);
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int partner = view.partner(how, i);
        if (partner < 0) continue;
        const cell_box own = view.box(i), there = view.box(partner);
        held_object& one = res.objects[i];
        if (columns) {
            const int16_t by = (int16_t)(there.col0 - own.col0);
            one.col += by;
        } else {
            const int16_t by = (int16_t)(there.row0 - own.row0);
            one.row += by;
        }
    }
    return res;
}

// MoveToward[value]: each picked object moves toward its one partner among the
// objects not picked -- of the kind value % PARTNER_KINDS names, each way
// partner_by finds one and then each color (partner_of_kind) -- by what value /
// PARTNER_KINDS names: one step, its own size, or until it touches the partner
// (move_toward, 15_op_math.hpp), the move inference replays. Partners are never
// picked, so none of them moves while the others read them. An object with no
// partner, or several, stays where it is.
object_list& op_move_toward(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (value < 0 || value >= (int)toward_by::count * PARTNER_KINDS) return res;
    const auto far = (toward_by)(value / PARTNER_KINDS);
    const int kind = value % PARTNER_KINDS;
    const listed_pieces view(from, run.input);
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int partner = view.partner_of_kind(kind, i);
        if (partner < 0) continue;
        const direction_step by = move_toward(far, i, partner, view, run.canvas.width, run.canvas.height);
        held_object& one = res.objects[i];
        one.col += (int16_t)by.across;
        one.row += (int16_t)by.down;
    }
    return res;
}

// --- Generate ---

// MirrorAround[value]: each picked object is drawn again mirrored around the
// middle of its one partner among the objects not picked -- left to right, top
// to bottom, all four ways at once, or across it the one way it runs, value /
// MIRROR_PARTNERS -- around a partner, the grid or the picked pattern.
// Across needs a partner with a direction. partner_of (15_op_math.hpp) finds
// the partner, the walk inference
// replays. The picked object stays; its copies join the list after every
// object there was, picked as it is, as CopyTo's are -- a sentence that picked
// every piece paints what it holds without picking again. An object with no
// partner, or several, gets no copy.
object_list& op_mirror_around(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    constexpr int partners = MIRROR_PARTNERS;
    if (value < 0 || value >= (int)mirror_by::count * partners) return res;
    const auto mirror = (mirror_by)(value / partners);
    const int how = value % partners;
    const listed_pieces view(from, run.input);
    for (int i = 0; i < from.count; ++i) {
        const held_object& piece = from.objects[i];
        if (!piece.selected || piece.width <= 0 || piece.height <= 0) continue;
        // The copies and where each lies are the reading's arithmetic
        // (each_mirror, mirrored).
        bool full = false;
        each_mirror(mirror, how, i, view, run.input.width, run.input.height, [&](int pose, int row2, int col2) {
            if (full || res.count >= MAX_OBJECTS_IN_A_LIST) { full = true; return; }
            res.objects[res.count++] = { mirrored(piece, pose, row2, col2), piece.selected };
        });
        if (full) return res;
    }
    return res;
}

// --- Select, by what a piece is ---

// SelectObjectsWithHoles[h]: mark the objects with exactly h enclosed holes;
// everything else rides through unmarked. A selector answers "which" and
// nothing else — the acting is a separate op (Recolor). Nothing is dropped, so
// several select/Recolor pairs chain order-free.
object_list& op_select_objects_with_holes(program_run& run, const object_list& from, int8_t holes) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        const int found = grid_parts(one.width, one.height,
            [&](int row, int col) { return one.data[row][col] != TRANSPARENT; }).hollows();
        if (found == (int)holes) one.selected = true;
    }
    return res;
}

// SelectAtLeast[n of a measure]: mark the objects with at least n of it
// (counts_at_least, which the reading replays); the rest ride through
// unmarked. InvertSelection after it says "fewer than n".
object_list& op_select_at_least(program_run& run, const object_list& from, int value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        if (counts_at_least(value, min((int)one.height, MAX_GRID_DIM), min((int)one.width, MAX_GRID_DIM),
                            [&](int row, int col) { return (int)one.data[row][col]; }))
            one.selected = true;
    }
    return res;
}

// SelectRunningAlong[way]: mark the objects that run that way (way_it_runs,
// which the reading replays), or past the ways the way most cells run
// (the_way_most_cells_run); the rest ride through unmarked.
object_list& op_select_running_along(program_run& run, const object_list& from, int8_t way) {
    object_list& res = run.fresh();
    copy_list(res, from);
    auto runs = [&](int i) {
        const held_object& one = res.objects[i];
        return way_it_runs(min((int)one.height, MAX_GRID_DIM), min((int)one.width, MAX_GRID_DIM),
                           [&](int row, int col) { return one.data[row][col] != TRANSPARENT; });
    };
    runs_along wanted = (runs_along)way;
    if (way == WAY_MOST_CELLS_RUN) {
        wanted = the_way_most_cells_run(res.count, runs, [&](int i) {
            const held_object& one = res.objects[i];
            int cells = 0;
            for (int row = 0; row < min((int)one.height, MAX_GRID_DIM); ++row)
                for (int col = 0; col < min((int)one.width, MAX_GRID_DIM); ++col) cells += one.data[row][col] != TRANSPARENT;
            return cells;
        });
        if (wanted == runs_along::no_way) return res;
    }
    for (int i = 0; i < res.count; ++i)
        if (runs(i) == wanted) res.objects[i].selected = true;
    return res;
}

// FitInto[moved]: each picked object moved into the one object it fits into
// (each_fit_move, which the reading replays), its painted cells turned as the
// place names and laid with their corner there, the rest where they were.
void fit_moved(object_list& res, const object_list& from, const grid& input) {
    each_fit_move(listed_pieces(from, input), [&](int i, int pose, int row, int col) {
        const held_object& source = from.objects[i];
        const cell_box paint = paint_in_frame(source);
        const auto [broad, high] = posed_size(pose, paint.width(), paint.height());
        held_object& one = res.objects[i];
        one.data.clear();
        for (int down = 0; down < high; ++down)
            for (int across = 0; across < broad; ++across) {
                const auto [from_row, from_col] = pose_source(pose, paint.width(), paint.height(), down, across);
                one.data[down][across] = source.data[paint.row0 + from_row][paint.col0 + from_col];
            }
        one.col = (int16_t)col; one.row = (int16_t)row;
        one.width = (int16_t)broad; one.height = (int16_t)high;
    });
}

// FitInto[poured]: each color the picked objects wear, every picked cell of it
// together, poured into the one gap of an object wearing none of it that holds
// as many cells (each_pour, which the reading replays): its cells gone from the
// picked objects, an object left with none gone with them, and the gap's cells
// one new object in that color, picked. FitInto[moved]: fit_moved.
object_list& op_fit_into(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (how == (int8_t)fit_into_way::moved) {
        fit_moved(res, from, run.input);
        return res;
    }
    vector<held_object> gaps;
    vector<int8_t> poured;
    each_pour(listed_pieces(from, run.input), [&](int color, const int16_t* rows, const int16_t* cols, int count) {
        cell_box box;
        for (int k = 0; k < count; ++k) box.include_cell(rows[k], cols[k]);
        held_object& one = gaps.emplace_back();
        one.col = (int16_t)box.col0; one.row = (int16_t)box.row0;
        one.width = (int16_t)box.width(); one.height = (int16_t)box.height();
        one.data.clear();
        for (int k = 0; k < count; ++k) one.data[rows[k] - box.row0][cols[k] - box.col0] = (int8_t)color;
        one.selected = true;
        poured.push_back((int8_t)color);
    });
    if (gaps.empty()) return res;
    int kept = 0;
    for (int i = 0; i < res.count; ++i) {
        held_object one = res.objects[i];
        bool lost = false, left = false;
        for (int row = 0; row < min((int)one.height, MAX_GRID_DIM); ++row)
            for (int col = 0; col < min((int)one.width, MAX_GRID_DIM); ++col) {
                int8_t& cell = one.data[row][col];
                if (one.selected && ranges::find(poured, cell) != poured.end()) {
                    cell = (int8_t)TRANSPARENT;
                    lost = true;
                }
                left = left || cell != (int8_t)TRANSPARENT;
            }
        if (left || !lost) res.objects[kept++] = one;
    }
    res.count = kept;
    for (const held_object& gap : gaps)
        if (res.count < MAX_OBJECTS_IN_A_LIST) res.objects[res.count++] = gap;
    return res;
}

// SelectWalls: mark the objects that are walls of the input's canvas (is_wall,
// which the reading replays); the rest ride through unmarked.
object_list& op_select_walls(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const listed_pieces view(from, run.input);
    for (int i = 0; i < res.count; ++i)
        if (view.wall(i)) res.objects[i].selected = true;
    return res;
}

// SelectColor[c]: mark every object that contains color c; the rest ride
// through unmarked.
object_list& op_select_color(program_run& run, const object_list& from, int8_t color) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (!is_color(color)) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        for (int row = 0; row < one.height && !one.selected; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.data[row][col] == color) { one.selected = true; break; }
    }
    return res;
}

// Elect from the current layout [0], keeping its frames, or the original
// input [1]. Like SelectColor, mark whole objects without clearing earlier marks.
object_list& op_select_elected_background(program_run& run, const object_list& from, int source) {
    if (source == 1)
        return op_select_color(run, from, confident_background(elect_grid_background(run.input)));
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool has_frame = false;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        if (!has_frame) {
            x0 = x1 = one.col; y0 = y1 = one.row;
            has_frame = true;
        }
        x0 = min(x0, (int)one.col); y0 = min(y0, (int)one.row);
        x1 = max(x1, one.col + one.width); y1 = max(y1, one.row + one.height);
    }
    grid laid;
    laid.width = max(0, min(MAX_GRID_DIM, x1 - x0));
    laid.height = max(0, min(MAX_GRID_DIM, y1 - y0));
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        each_painted(one, [&](int row, int col, int8_t color) {
            const int at_row = one.row + row - y0, at_col = one.col + col - x0;
            if (at_row >= 0 && at_row < laid.height && at_col >= 0 && at_col < laid.width)
                laid.data[at_row][at_col] = color;
        });
    }
    const int8_t color = confident_background(elect_grid_background(laid));
    // No winner adds no marks, including on wholly transparent objects.
    if (color < 0) {
        object_list& res = run.fresh();
        copy_list(res, from);
        return res;
    }
    return op_select_color(run, from, color);
}

// SelectAll: mark every object — the explicit "act on it all".
object_list& op_select_all(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = true;
    return res;
}

// InvertSelection: flip every object's mark. With additive selectors (OR) this
// is NOT, and OR + NOT is the whole boolean algebra — "big AND red" is
// invert(invert(big) ∪ invert(red)) by De Morgan, no extra machinery.
object_list& op_invert_selection(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = !res.objects[i].selected;
    return res;
}

// The color the puzzle-level table names for a piece's key of `kind` (its
// key, read in its current box as the reading reads it), as the reading looks it up
// (shape_table::color_of): -1 where it names none. The one lookup both shape
// ops make.
int shape_color(const program_run& run, const held_object& one, int kind) {
    return run.shape_colors.color_of(shape_color_map_rule::key_of((shape_key)kind, one));
}

// ShapeColorMap[kind]: every picked piece takes the color its own key names in
// the puzzle-level table the reading learned -- its SHAPE, the same turned either
// way or mirrored either way, or its SYMMETRY, the ways round that leave its
// drawing as it was (shape_key_of, the key the reading learned) -- or the color
// the table gives every other key. A key the table names no color for leaves
// its piece as it was; nothing else changes. A piece colored by ANOTHER's shape
// is said with other operators (009d5c81's key naming the figure's color): the
// key takes its shape's color, the figure the key's (ColorFrom), and the key
// goes (DeleteSelection).
object_list& op_shape_color_map(program_run& run, const object_list& from, int kind) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (!run.shape_colors.holds()) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        if (!one.selected) continue;
        if (const int color = shape_color(run, one, kind); color >= 0) recolor(one, (int8_t)color);
    }
    return res;
}

// ShapeToInteriorColorMap: each picked piece's enclosed interior filled with
// the color its OWN shape names in the puzzle-level table (the lookup
// ShapeColorMap[0] makes); a miss passes the piece through unchanged. A
// painted cell inside the hole (a marker) is a wall to the flood, so it keeps
// its color.
object_list& op_shape_to_interior_color_map(program_run& run, const object_list& from) {
    return fill_interiors(run, from, [&](const held_object& one) {
        return shape_color(run, one, (int)shape_key::shape);
    });
}

// --- Map, by where pieces sit ---

// OverlapCorner [pose]: lay the objects out as a diagonal corner-chain, as the
// canvas seen in that pose lays it plainly (each_corner_link): there the
// objects are ordered left-to-right (by column, ties by row); the first is
// placed with its top-left at the canvas origin (0,0); each later object's
// top-left cell is set to the previous object's bottom-right cell -- a 1-pixel
// corner overlap. Pose 0 is that chain itself, from the top left by columns.
// Only positions change (cells and colors untouched); the trailing Union
// resolves the shared corner -- the later (further-down-chain) object wins, as
// Union lays a later object over an earlier.
//
// This is the sequential / corner / overlap generalization of AlignToCanvas,
// which is the batch / edge-flush / gap-0 corner of the same space:
//     placement: batch (fixed canvas)   ->  sequential (running canvas)
//     anchor:    edge-flush             ->  corner-to-corner
//     offset:    gap = 0                ->  stride = 1 (overlap)
// Not built: the stride, the per-step diagonal offset (overlap 1 today; 0 =
// corner-touch, >0 = gap). Inseparable from this op -- it's the cascade's step,
// not a downstream op, because it compounds with chain index (object i shifts i*).
object_list& op_overlap_corner(program_run& run, const object_list& from, int8_t pose) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    if (count <= 0) { copy_list(res, from); return res; }
    // The chain as each_corner_link lays it, the reading's replay the same.
    each_corner_link(count, pose, run.canvas.width, run.canvas.height, [&](int i) {
        const held_object& one = from.objects[i];
        return cell_box{ one.row, one.col, one.row + one.height - 1, one.col + one.width - 1 };
    }, [&](int i, int row, int col) {
        const held_object& source = from.objects[i];
        held_object& one = res.objects[res.count++] = source;
        one.col = (int16_t)col;                       one.row = (int16_t)row;
        return true;
    });
    return res;
}

// AlignToCanvas: snap each picked object flush against the box of what is
// painted on the canvas: on the axis with the larger gap, the nearer of the two
// edge snaps (shift_to_touch); the other axis is left as it is. An empty canvas
// moves nothing.
object_list& op_align_to_canvas(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    // Read the painted canvas once; moving objects do not enlarge this target.
    cell_box target;
    for (int row = 0; row < run.canvas.height; ++row)
        for (int col = 0; col < run.canvas.width; ++col)
            if (run.canvas.data[row][col] != TRANSPARENT) target.include_cell(row, col);
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!one.selected || target.empty()) continue;
        const auto [down, across] = shift_to_touch(one.box(), target);
        one.col = (int16_t)(one.col + across);
        one.row = (int16_t)(one.row + down);
    }
    return res;
}

// Origin: each picked object moves to the canvas's top-left corner.
object_list& op_origin(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        // Only what was picked moves; the rest ride through where they are.
        if (!one.selected) continue;
        one.col = 0; one.row = 0;
    }
    return res;
}

// --- Reduce, folding a list ---

// Intersect the objects whose `selected` equals `want` into ONE object, where
// they lie: each cell of the grid they span as intersected_cell keeps it, as
// union_group lays them. Appends it to res carrying the shared flag if anything
// survives.
void intersect_group(const object_list& from, bool want, object_list& res) {
    int members[MAX_OBJECTS_IN_A_LIST], count = 0;
    cell_box span;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.selected != want) continue;
        span.include(one.box());
        members[count++] = i;
    }
    span = { max(span.row0, 0), max(span.col0, 0), min(span.row1, MAX_GRID_DIM - 1), min(span.col1, MAX_GRID_DIM - 1) };
    if (count == 0 || span.empty() || res.count >= MAX_OBJECTS_IN_A_LIST) return;
    held_object one;
    one.row = (int16_t)span.row0; one.col = (int16_t)span.col0;
    one.width = (int16_t)span.width(); one.height = (int16_t)span.height();
    one.selected = want;
    int kept = 0;
    for (int row = span.row0; row <= span.row1; ++row)
        for (int col = span.col0; col <= span.col1; ++col) {
            const int8_t color = intersected_cell(count, [&](int k) { return from.objects[members[k]].color_at(row, col); });
            if (color == TRANSPARENT) continue;
            one.data[row - span.row0][col - span.col0] = color;
            ++kept;
        }
    if (kept == 0) return;
    res.objects[res.count++] = one;
}

// IntersectionList: intersect to one object PER selection group, where the
// objects lie — the cells two or more selected objects hold into one selected
// object, those of the unselected into one unselected object. The mark
// survives the fold as object identity, never dropped. No selection => one
// unselected object.
object_list& op_intersection_list(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    if (from.count <= 0) return res;
    intersect_group(from, false, res);
    intersect_group(from, true, res);
    return res;
}

// Merge the objects whose `selected` equals `want` into ONE object in grid
// coordinates. Appends it to res, carrying the shared flag, if any such object
// has a painted cell.
void union_group(const object_list& from, bool want, object_list& res) {
    int x0 = MAX_GRID_DIM, y0 = MAX_GRID_DIM, x1 = -1, y1 = -1;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.selected != want) continue;
        x0 = min(x0, (int)one.col); y0 = min(y0, (int)one.row);
        x1 = max(x1, one.col + one.width - 1); y1 = max(y1, one.row + one.height - 1);
    }
    if (x1 < 0 || y1 < 0 || res.count >= MAX_OBJECTS_IN_A_LIST) return;
    held_object& merged = res.objects[res.count];
    merged.col = (int16_t)x0; merged.row = (int16_t)y0;
    merged.width = (int16_t)(x1 - x0 + 1); merged.height = (int16_t)(y1 - y0 + 1);
    merged.selected = want;
    merged.data.clear();
    bool any = false;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.selected != want) continue;
        each_painted(one, [&](int row, int col, int8_t color) {
            merged.data[one.row + row - y0][one.col + col - x0] = color;
            any = true;
        });
    }
    if (any) res.count++;
}

// UnionList: merge to one object PER selection group — all selected pixels fuse
// into one selected object, all unselected into one unselected object. The
// wall/content split (which lives in `selected`) survives the merge as object
// identity, so `UnionList Partition` cuts whole tiles with the walls still marked.
// No selection => a single unselected object (plain "merge all").
object_list& op_union_list(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    if (from.count <= 0) return res;
    union_group(from, false, res);  // content first, then walls
    union_group(from, true, res);
    return res;
}

// --- Bands ---

// WHICH BAND A PIECE SITS IN, not what the piece is. Every other selector names
// something the piece carries -- its color, its holes, its size -- so none of
// them can say "the middle row of them". This one names where a piece sits among
// the others, which is a question about the arrangement and not about the piece.
// Bands come from bands_of, the same grouping Layout reads, so a row
// means one thing wherever it is asked about. Selectors ADD, so a band past the
// last one marks nothing rather than clearing what is already picked.
object_list& op_select_band(program_run& run, const object_list& from, int8_t which, bool rows) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const int count = min(res.count, MAX_OBJECTS_IN_A_LIST);
    if (count <= 0) return res;
    int8_t band_of[MAX_OBJECTS_IN_A_LIST];
    bands_of(count, rows, [&](int i) { return res.objects[i].box(); }, band_of);
    for (int i = 0; i < count; ++i)
        if (band_of[i] == which) res.objects[i].selected = true;
    return res;
}

// SelectRowFromMiddle[from] / SelectColumnFromMiddle[from]: mark every object
// with a painted cell on the row (column) `lines` lines past the middle of the
// span every object's paint covers together -- the line as many lines from
// either end, none where the span has two (line_from_middle, the reading's own
// arithmetic). Keeps every object and prior mark.
object_list& op_select_from_middle(program_run& run, const object_list& from, int8_t lines, bool rows) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const int count = min(res.count, MAX_OBJECTS_IN_A_LIST);
    int low = MAX_GRID_DIM * 2, high = -MAX_GRID_DIM * 2;
    for (int i = 0; i < count; ++i) {
        const held_object& one = res.objects[i];
        each_painted(one, [&](int row, int col, int8_t) {
            const int line = rows ? one.row + row : one.col + col;
            low = min(low, line);
            high = max(high, line);
        });
    }
    const int line = line_from_middle(low, high, lines);
    if (line < 0) return res;
    for (int i = 0; i < count; ++i) {
        held_object& one = res.objects[i];
        bool on = false;
        for (int row = 0; row < one.height && !on; ++row)
            for (int col = 0; col < one.width && !on; ++col)
                on = one.data[row][col] != TRANSPARENT && (rows ? one.row + row : one.col + col) == line;
        if (on) one.selected = true;
    }
    return res;
}

// Layout[k]: arrange the objects on k-sized graph paper. Overlapping vertical
// spans share a row, overlapping horizontal spans share a column; rows sort
// top-to-bottom, columns left-to-right. Each object's drawing is copied
// UNRESIZED into its own k x k cell (cell origin = (col*k, row*k)), clipped to
// the cell. Result is one (cols*k) x (rows*k) object at the origin — grid-
// aligned by construction, so ScaleDown[k] reads exactly one object per cell.
// WHERE A PIECE SITS IS ITS FRAME; WHAT IT DRAWS IS WHAT IT HOLDS. The bands
// read the frames and the cells get the painted cells, from the top-left of
// what is painted rather than of the frame. For a trimmed piece the two are
// the same box. A pane is not trimmed -- its frame is the pane -- so after
// Panes the bands are the panes' own rows and columns while each cell gets the
// figure the pane holds, wherever it sat inside it.
// Layout[0] reads the cell off the pieces instead of taking it as a number: as
// wide as the widest drawing and as tall as the tallest. Where each drawing
// lands is lay_out's (15_op_math.hpp), which the readings replay.
object_list& op_layout(program_run& run, const object_list& from, int cell) {
    object_list& res = run.fresh();
    held_object& laid = res.objects[0];
    laid.data.clear();
    const layout_paper paper = lay_out(min(from.count, MAX_OBJECTS_IN_A_LIST), cell,
        [&](int i) { return from.objects[i].box(); },
        [&](int i) { return paint_in_frame(from.objects[i]); },
        // Shift/Union can grow the box beyond the cells it can hold.
        [&](int i, int row, int col) {
            return row < MAX_GRID_DIM && col < MAX_GRID_DIM ? (int)from.objects[i].data[row][col] : TRANSPARENT;
        },
        [&](int row, int col, int color) { laid.data[row][col] = (int8_t)color; });
    if (paper.width <= 0) return res;
    laid.col = 0; laid.row = 0;
    laid.width = (int16_t)paper.width; laid.height = (int16_t)paper.height;
    res.count = 1;
    return res;
}

// DeleteSelection: drop the selected OBJECTS from the list, pass the rest
// through unchanged. Selection is per object, so this is a straight
// compaction — no per-cell erase, no box refit. Erasing part of an object is
// not a selection job: carve the piece out (Split/Colors) and delete that
// whole object instead. Survivors carry no marks (they were the unselected ones).
object_list& op_delete_selection(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < count; ++i) {
        const held_object& one = from.objects[i];
        if (one.selected) continue;
        if (res.count >= MAX_OBJECTS_IN_A_LIST) break;
        res.objects[res.count++] = one;
    }
    return res;
}

// --- Translate, by itself ---

// Shift[move_direction]: each picked object moves its own size away, the direction `move_direction`
// names (shift_by, 15_op_math.hpp). No selection means Shift moves nothing.
object_list& op_shift(program_run& run, const object_list& from, int8_t move_direction) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        if (!one.selected) continue;
        const direction_step by = shift_by(move_direction, one.width, one.height);
        one.col += (int16_t)by.across; one.row += (int16_t)by.down;
    }
    return res;
}

// TurnRight, TurnLeft, FlipHorizontal, FlipVertical: each picked object read
// through its pose (pose_of), the one the reading's replays turn a piece through.
object_list& op_turn(program_run& run, const object_list& from, node_type type) {
    object_list& res = run.fresh();
    const bool is_turn = type == node_type::TurnRight || type == node_type::TurnLeft;
    remap_picked(from, res,
        [&](const held_object& source, int& width, int& height) {
            width = min((int)(is_turn ? source.height : source.width), MAX_GRID_DIM);
            height = min((int)(is_turn ? source.width : source.height), MAX_GRID_DIM);
        },
        [&](const held_object& source, int row, int col) {
            const auto [from_row, from_col] = pose_source(pose_of(type), source.width, source.height, row, col);
            return (int8_t)source.data[from_row][from_col];
        });
    return res;
}

// MoveRow[delta] / MoveColumn[delta]: each picked object moves `delta` cells
// across (MoveRow) or down (MoveColumn).
object_list& op_move(program_run& run, const object_list& from, int16_t delta, bool across) {
    object_list& res = run.fresh();
    copy_objects(res, from, min(from.count, MAX_OBJECTS_IN_A_LIST));
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        if (!one.selected) continue;
        if (across) one.col += delta;
        else one.row += delta;
    }
    return res;
}

// SlamRow / SlamColumn: each picked object's cells slide a step along its rows
// or columns and pile at the wall (slam_cells, 15_op_math.hpp).
object_list& op_slam(program_run& run, const object_list& from, bool rows, int16_t step) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        if (!one.selected || step == 0) continue;
        // Shift/Union can grow the box beyond the cells it can hold.
        const int width = min((int)one.width, MAX_GRID_DIM);
        const int height = min((int)one.height, MAX_GRID_DIM);
        cells drawn;
        slam_cells(width, height, rows, step,
            [&](int row, int col) { return (int)one.data[row][col]; },
            [&](int row, int col, int color) { drawn[row][col] = (int8_t)color; });
        one.data = drawn;
    }
    return res;
}

// ScaleUp[factor]: each picked object drawn `factor` times larger, each cell a
// factor x factor block.
object_list& op_scale_up(program_run& run, const object_list& from, int8_t factor) {
    object_list& res = run.fresh();
    if (factor <= 1) {
        copy_objects(res, from, min(from.count, MAX_OBJECTS_IN_A_LIST));
        return res;
    }
    remap_picked(from, res,
        [&](const held_object& source, int& width, int& height) {
            width = min(source.width * factor, MAX_GRID_DIM);
            height = min(source.height * factor, MAX_GRID_DIM);
        },
        [&](const held_object& source, int row, int col) {
            const int from_row = row / factor, from_col = col / factor;
            return from_row < source.height && from_col < source.width ? (int8_t)source.data[from_row][from_col]
                                                                       : (int8_t)TRANSPARENT;
        });
    return res;
}

// ScaleDown[factor]: each picked object drawn `factor` times smaller, each cell
// the majority vote of its factor x factor block (majority_vote).
object_list& op_scale_down(program_run& run, const object_list& from, int8_t factor) {
    object_list& res = run.fresh();
    if (factor <= 1) {
        copy_objects(res, from, min(from.count, MAX_OBJECTS_IN_A_LIST));
        return res;
    }
    remap_picked(from, res,
        [&](const held_object& source, int& width, int& height) {
            width = max((source.width + factor - 1) / factor, 1);
            height = max((source.height + factor - 1) / factor, 1);
        },
        [&](const held_object& source, int row, int col) {
            return majority_vote(source.data, row, col, factor, source.height, source.width);
        });
    return res;
}

// --- Generate, on a piece ---

// AddColor[value]: every empty cell in a picked object's frame takes the color
// `value` names, resolved once off the object as it came (the same literal /
// least-common / most-common rule as Recolor).
object_list& op_add_color(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        if (!one.selected) continue;
        const int8_t fill = paint_color(from.objects[i], value);
        if (!is_color(fill)) continue;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.data[row][col] == TRANSPARENT) one.data[row][col] = fill;
    }
    return res;
}

// Outline: keep each picked object's edge cells, drop the core to transparent.
// A cell is edge when any of its 8 neighbours is not part of the object
// (on_object_edge; outside the frame counts as outside). Reads the source
// cells, writes the copy's, so the erosion never sees its own output.
object_list& op_outline(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        held_object& one = res.objects[res.count++] = source;
        if (!one.selected) continue;
        for (int row = 0; row < source.height; ++row)
            for (int col = 0; col < source.width; ++col) {
                if (source.data[row][col] == TRANSPARENT) continue;
                const bool edge = on_object_edge(row, col, [&](int at_row, int at_col) {
                    return at_row >= 0 && at_col >= 0 && at_row < source.height && at_col < source.width &&
                           source.data[at_row][at_col] != TRANSPARENT;
                });
                if (!edge) one.data[row][col] = TRANSPARENT;
            }
    }
    return res;
}

// Fit[to]: each picked object stretched, nearest cell, over the frame `to`
// names (fit_frame): the input's, the canvas's, what is painted on it, or one
// of this example's panes.
object_list& op_fit(program_run& run, const object_list& from, int to) {
    const cell_box target = fit_frame(to, run.input.width, run.input.height, run.canvas,
                                      pane_split_facts::shared_size(run.panes));
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (source.width <= 0 || source.height <= 0) continue;
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        if (target.empty()) continue;
        const int width = target.width(), height = target.height();
        held_object one;
        // Same object, same mark -- a fresh one would start unmarked.
        one.selected = source.selected;
        one.col = (int16_t)target.col0; one.row = (int16_t)target.row0;
        one.width = (int16_t)width; one.height = (int16_t)height;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                one.data[row][col] = source.data[fit_source_cell(row, source.height, height)]
                                                [fit_source_cell(col, source.width, width)];
        res.objects[res.count++] = one;
    }
    return res;
}

// Skew slope table: variant -> {num, den}; the line slope is num/den (Δcol per
// Δrow), applied as a Bresenham shift floor((row+r)*num/den). Stern-Brocot /
// complexity order (simplest slopes first). Counts: reduced |num|,|den| <= 6
// gives 23 distinct magnitudes; with both signs + the identity (0/1) that is
// 23*2 + 1 = 47 entries. Index 0 = identity and index 1 = +45 deg.
constexpr int8_t skew_slopes[47][2] = {
    {0,1},
    {1,1},{-1,1},   {1,2},{-1,2},   {2,1},{-2,1},   {1,3},{-1,3},   {3,1},{-3,1},
    {1,4},{-1,4},   {2,3},{-2,3},   {3,2},{-3,2},   {4,1},{-4,1},   {1,5},{-1,5},
    {5,1},{-5,1},   {1,6},{-1,6},   {2,5},{-2,5},   {3,4},{-3,4},   {4,3},{-4,3},
    {5,2},{-5,2},   {6,1},{-6,1},   {3,5},{-3,5},   {5,3},{-5,3},   {4,5},{-4,5},
    {5,4},{-5,4},   {5,6},{-5,6},   {6,5},{-6,5},
};

// Floor division toward -inf (C integer division truncates toward zero). den is
// always > 0 in the slope table, so this keeps +/- slopes mirror-symmetric.
int skew_shift(int row, int numerator, int denominator) {
    const int product = row * numerator;
    const int whole = product / denominator, rest = product % denominator;
    return rest < 0 ? whole - 1 : whole;
}

// Skew[variant]: each picked object's rows lean along a slope, each row shifted
// across by its own floor of the slope (skew_slopes).
object_list& op_skew(program_run& run, const object_list& from, int8_t variant) {
    object_list& res = run.fresh();
    const int slope = variant >= 0 && variant < 47 ? variant : 0;
    const int numerator = skew_slopes[slope][0], denominator = skew_slopes[slope][1];
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count] = from.objects[i];
        // Only what was picked moves; the rest ride through where they are.
        if (!one.selected) { res.count++; continue; }
        const int top = one.row;
        int least_col = 1000, most_col = -1000, least_row = 1000, most_row = -1000;
        each_painted(one, [&](int row, int col, int8_t) {
            const int to_col = col + skew_shift(top + row, numerator, denominator);
            least_col = min(least_col, to_col); most_col = max(most_col, to_col);
            least_row = min(least_row, row); most_row = max(most_row, row);
        });
        if (most_col < least_col) { res.count++; continue; }
        const int width = min(most_col - least_col + 1, MAX_GRID_DIM);
        const int height = min(most_row - least_row + 1, MAX_GRID_DIM);
        cells leaned;
        each_painted(one, [&](int row, int col, int8_t color) {
            const int to_row = row - least_row, to_col = col + skew_shift(top + row, numerator, denominator) - least_col;
            if (to_row >= 0 && to_row < height && to_col >= 0 && to_col < width) leaned[to_row][to_col] = color;
        });
        one.data = leaned;
        one.width = (int16_t)width; one.height = (int16_t)height;
        one.col = (int16_t)(one.col + least_col); one.row = (int16_t)(top + least_row);
        res.count++;
    }
    return res;
}

// Pattern: each picked object reduced to its smallest repeating unit
// (repeat_periods). Visible units combine the evidence of incomplete copies.
object_list& op_pattern(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (source.width <= 0 || source.height <= 0) continue;
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        if (how == 1) {
            const grid unit = visible_pattern(source.drawn());
            held_object one = source;
            one.width = (int16_t)unit.width; one.height = (int16_t)unit.height; one.data = unit.data;
            res.objects[res.count++] = one;
            continue;
        }
        int unit_width = 0, unit_height = 0;
        repeat_periods(grid(source.width, source.height, source.data), unit_width, unit_height);
        // Every copy agrees with the top-left one, so the unit is that copy.
        res.objects[res.count++] = cut_to(source, { 0, 0, unit_height - 1, unit_width - 1 });
    }
    return res;
}

// VoteDenoise: noise-tolerant majority vote across repeating structure. The
// arithmetic is vote_denoise_cells (15_op_math.hpp), shared with inference.
// No period on either axis -> object passes unchanged.
object_list& op_vote_denoise(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (source.width <= 0 || source.height <= 0) continue;
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        held_object one = source;
        const int width = source.width, height = source.height;
        // The ground is read off this piece alone (denoise_ground, shared with
        // the reading): empty where a deletion upstream left it empty, else the
        // color most of the piece wears. No puzzle-level background is read.
        const int8_t ground = denoise_ground(width, height, [&](int row, int col) { return (int8_t)source.data[row][col]; });
        // A cell emptied by a background removal upstream is still the ground:
        // read it as the ground, which is itself empty when the ground was deleted.
        const auto ground_at = [&](int row, int col) -> int8_t {
            const int8_t color = source.data[row][col];
            return color == TRANSPARENT ? ground : color;
        };
        vote_denoise_cells(width, height, ground, ground_at,
                           [&](int row, int col, int8_t color) { one.data[row][col] = color; });
        res.objects[res.count++] = one;
    }
    return res;
}

// Trim: each picked object's frame fitted to its paint -- a deliberate act
// (the frame law), so it happens only to what was picked. An object with no
// paint drops out.
object_list& op_trim(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        const cell_box paint = paint_in_frame(source);
        if (paint.empty()) continue;
        res.objects[res.count++] = cut_to(source, paint);
    }
    return res;
}

// Tile [steps, start, end, spread]: each picked piece laid again where its
// Tile says (each_tile_offset, 15_op_math.hpp, the arithmetic the readings
// replay), each step measured on this hand: the piece's own size and gaps,
// how many colors the picked pieces hold together, and the size of the key,
// the pieces not picked (measures_of, key_box). A COPY
// IS A PIECE, keeping the piece's mark, so a row of copies can be picked and
// turned; want one sheet? Say UnionList after. A copy is the part of the
// piece one step covers where a step is shorter, from the side the steps move
// away from (copy_part). A piece no step moves, and every
// piece not picked, rides through as it came; copies past the list's room are
// dropped, as Split[2] drops cells past it.
object_list& op_tile(program_run& run, const object_list& from, const tile_value& tile) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    const int colors = color_count(count, [&](int i) { return from.objects[i].selected; }, [&](int i) {
        uint32_t held = 0;
        each_painted(from.objects[i], [&](int, int, int8_t color) { if (is_color(color)) held |= 1u << color; });
        return held;
    });
    const cell_box key = key_box(listed_pieces(from, run.input));
    for (int i = 0; i < count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        const tile_moves moves = !source.selected ? tile_moves{}
            : moves_of(tile, measures_of(tile, source.height, source.width,
                                         [&](int row, int col) { return source.data[row][col] != TRANSPARENT; }, colors,
                                         key));
        if (moves.count == 0 || source.width <= 0 || source.height <= 0) {
            res.objects[res.count++] = source;
            continue;
        }
        const held_object part = cut_to(source, copy_part(moves, source.height, source.width));
        each_tile_offset(tile, moves, part.row, part.col, part.height, part.width, run.canvas.width, run.canvas.height,
            [&](int down, int across) {
                if (res.count >= MAX_OBJECTS_IN_A_LIST) return false;
                held_object& copy = res.objects[res.count++] = part;
                copy.row = (int16_t)(part.row + down);
                copy.col = (int16_t)(part.col + across);
                return true;
            });
    }
    return res;
}

// Fractal: each picked object stamped into itself over the canvas: where its
// own scaled-up drawing is painted, its cells repeat.
object_list& op_fractal(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int width = run.canvas.width, height = run.canvas.height;
    if (width <= 0 || height <= 0) return res;
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (source.width <= 0 || source.height <= 0) continue;
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        held_object& stamped = res.objects[res.count];
        // Same object grown, so it keeps its mark.
        stamped.selected = source.selected;
        stamped.col = 0; stamped.row = 0;
        stamped.width = (int16_t)width; stamped.height = (int16_t)height;
        stamped.data.clear();
        bool any = false;
        for (int row = 0; row < height; ++row) {
            const int big_row = row * source.height / height, own_row = row % source.height;
            for (int col = 0; col < width; ++col) {
                const int big_col = col * source.width / width, own_col = col % source.width;
                if (source.data[big_row][big_col] == TRANSPARENT) continue;
                const int8_t color = source.data[own_row][own_col];
                if (color != TRANSPARENT) { stamped.data[row][col] = color; any = true; }
            }
        }
        if (any) ++res.count;
    }
    return res;
}

// For each picked object: rays for `radius` steps in the four directions from
// `diagonal`'s set, grown from the middle of its box (ray_seed_of,
// each_extended_ray), in the color of its cell nearest that middle. The rays
// only, one object per input.
object_list& op_extend_rays(program_run& run, const object_list& from, int8_t radius, bool diagonal) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        if (!one.selected) { res.objects[res.count++] = one; continue; }
        const auto seed = ray_seed_of(one.height, one.width, [&](int row, int col) { return one.data[row][col]; });
        if (seed.row < 0) continue;
        // The object's box grown by radius on every side, clipped to the canvas:
        // a ray that runs past the edge would never be painted there anyway, and
        // a frame no wider than the canvas holds every ray that shows -- one
        // anchored off the object's corner cut the long ones short on its far
        // sides.
        const int x0 = max(one.col - radius, 0), y0 = max(one.row - radius, 0);
        const int x1 = min(one.col + one.width - 1 + radius, MAX_GRID_DIM - 1),
                  y1 = min(one.row + one.height - 1 + radius, MAX_GRID_DIM - 1);
        if (x1 < x0 || y1 < y0) continue;
        held_object& rays = res.objects[res.count++];
        // The rays belong to the object that grew them, so they carry its mark.
        rays.selected = one.selected;
        rays.col = (int16_t)x0; rays.row = (int16_t)y0;
        rays.width = (int16_t)(x1 - x0 + 1); rays.height = (int16_t)(y1 - y0 + 1);
        rays.data.clear();
        // RAYS ONLY: the piece's own cells are not copied, so a caller that
        // wants them re-stamps them (a Union). A one-cell piece gives only its
        // arms, and radius 0 gives nothing.
        const int seed_row = one.row + seed.row - y0, seed_col = one.col + seed.col - x0;
        each_extended_ray(seed, radius, diagonal, [&](int down, int across, int) {
            const int row = seed_row + down, col = seed_col + across;
            if (row >= 0 && row < rays.height && col >= 0 && col < rays.width) rays.data[row][col] = seed.color;
        });
    }
    return res;
}

// Whether the object paints the grid cell (x, y).
bool paints_at(const held_object& one, int x, int y) {
    const int across = x - one.col, down = y - one.row;
    return across >= 0 && across < one.width && down >= 0 && down < one.height && one.data[down][across] != TRANSPARENT;
}

// MoveToEdge[move_direction]: each picked object slides along move_direction until its painted cells
// meet the canvas edge that way, ignoring paint. Shape unchanged. (ExtendTo
// grows a trail; Slam squashes inside the box; this one TRAVELS.)
// PAST THE EDGE ALREADY, IT SLIDES BACK until its far side lies on the edge,
// rather than refusing to move. A grid larger than the canvas pushed right
// shows its right side, so a crop at a corner is the size rule's box and a
// push, with no count of cells in it. Inside the canvas nothing changes: the
// slide stops at whichever edge it meets first.
object_list& op_move_to_edge(program_run& run, const object_list& from, int8_t move_direction) {
    object_list& res = run.fresh();
    const direction_step step = step_of(move_direction);
    copy_objects(res, from, min(from.count, MAX_OBJECTS_IN_A_LIST));
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        if (!one.selected) continue;
        cell_box painted;
        each_painted(one, [&](int row, int col, int8_t) { painted.include_cell(one.row + row, one.col + col); });
        if (painted.empty()) continue;
        const int steps = steps_to_edge(painted, run.canvas.width, run.canvas.height, move_direction);
        one.col = (int16_t)(one.col + steps * step.across);
        one.row = (int16_t)(one.row + steps * step.down);
    }
    return res;
}

// ExtendTo PAST ITS EIGHT WAYS: every picked object grows in step with the
// others along its track, lines or the open cells, until they meet --
// grow_in_step, 15_op_math.hpp, the growth the reading replays -- and hands
// back its body and every cell it took, in its first color, as a trail hands
// back its body and its trail. OVERRUN, each keeps too the cells it reached in
// the turn another did, so IntersectionList keeps where they meet.
object_list& op_grow_in_step(program_run& run, const object_list& from, grown_along track, bool overrun) {
    object_list& res = run.fresh();
    int width = run.canvas.width > 0 ? run.canvas.width : run.input.width;
    int height = run.canvas.height > 0 ? run.canvas.height : run.input.height;
    if (width <= 0 || height <= 0) return res;
    width = min(width, MAX_GRID_DIM);
    height = min(height, MAX_GRID_DIM);
    // Each cell's object, the first in the list that holds it; the turn it was
    // taken in; the cells still to grow from; the directions each cell was reached.
    int16_t owner[MAX_GRID_CELLS] = {}, turn[MAX_GRID_CELLS] = {}, queue[MAX_GRID_CELLS] = {};
    uint8_t cell_directions[MAX_GRID_CELLS] = {};
    for (int at = 0; at < width * height; ++at) owner[at] = -1;
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < count; ++i) {
        const held_object& one = from.objects[i];
        each_painted(one, [&](int row, int col, int8_t) {
            const int x = one.col + col, y = one.row + row;
            if (x >= 0 && x < width && y >= 0 && y < height && owner[y * width + x] < 0)
                owner[y * width + x] = (int16_t)i;
        });
    }
    // The cells each object reached in the turn another took them, (cell, object).
    vector<std::pair<int, int>> reached_together;
    grow_in_step(width, height, track, [&](int piece) { return from.objects[piece].selected; },
                 owner, turn, queue, cell_directions, [&](int at, int, int piece, bool in_one_turn) {
                     if (overrun && in_one_turn) reached_together.push_back({ at, piece });
                 });
    for (int i = 0; i < count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        if (!one.selected) { res.objects[res.count++] = one; continue; }
        cell_box reach;
        each_painted(one, [&](int row, int col, int8_t) { reach.include_cell(one.row + row, one.col + col); });
        for (int at = 0; at < width * height; ++at)
            if (turn[at] > 0 && owner[at] == i) reach.include_cell(at / width, at % width);
        for (const auto& [at, piece] : reached_together)
            if (piece == i) reach.include_cell(at / width, at % width);
        if (reach.empty()) continue;
        const int grown_width = min(reach.width(), MAX_GRID_DIM);
        const int grown_height = min(reach.height(), MAX_GRID_DIM);
        held_object& grown = res.objects[res.count];
        // The grown object is the SAME object, so it keeps the mark it was
        // picked by, as a trail's does.
        grown.selected = one.selected;
        grown.col = (int16_t)reach.col0; grown.row = (int16_t)reach.row0;
        grown.width = (int16_t)grown_width; grown.height = (int16_t)grown_height;
        grown.data.clear();
        each_painted(one, [&](int row, int col, int8_t color) {
            const int across = one.col + col - reach.col0, down = one.row + row - reach.row0;
            if (across >= 0 && across < grown_width && down >= 0 && down < grown_height) grown.data[down][across] = color;
        });
        const int8_t color = one.first_color();
        for (int at = 0; at < width * height; ++at) {
            if (turn[at] <= 0 || owner[at] != i) continue;
            const int across = at % width - reach.col0, down = at / width - reach.row0;
            if (across >= 0 && across < grown_width && down >= 0 && down < grown_height) grown.data[down][across] = color;
        }
        for (const auto& [at, piece] : reached_together)
            if (piece == i) grown.data[at / width - reach.row0][at % width - reach.col0] = color;
        ++res.count;
    }
    return res;
}

// ExtendTo[grow_direction]: every picked object extrudes its cells along one of 8
// directions, painting the swept path in the object's color. A ray stops just
// before the first FOREIGN occupied cell (a wall / another object) or the grid
// edge, and passes through the object's own body (march_trail, the walk the
// reading replays). Hands back each object's body and trail, which compose via
// Union. Past the eight ways, the objects grow in step (op_grow_in_step), and
// past those, the same growths overrun.
object_list& op_extend_to(program_run& run, const object_list& from, int8_t grow_direction) {
    if (grow_direction >= direction::count)
        return op_grow_in_step(run, from, (grown_along)((grow_direction - direction::count) % (int)grown_along::count),
                               grow_direction >= OVERRUN_GROWTHS);
    object_list& res = run.fresh();
    const direction_step step = step_of(grow_direction);
    int width = run.canvas.width > 0 ? run.canvas.width : run.input.width;
    int height = run.canvas.height > 0 ? run.canvas.height : run.input.height;
    if (width <= 0 || height <= 0) return res;
    width = min(width, MAX_GRID_DIM);
    height = min(height, MAX_GRID_DIM);
    // Every cell some held object paints: the walls a trail stops at.
    bool occupied[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        each_painted(one, [&](int row, int col, int8_t) {
            const int x = one.col + col, y = one.row + row;
            if (x >= 0 && x < MAX_GRID_DIM && y >= 0 && y < MAX_GRID_DIM) occupied[y][x] = true;
        });
    }
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        // Only what was picked is touched; the rest ride through as they are.
        if (!one.selected) { res.objects[res.count++] = one; continue; }
        const int8_t color = one.first_color();
        const auto own = [&](int y, int x) { return paints_at(one, x, y); };
        const auto taken = [&](int y, int x) { return occupied[y][x]; };
        cell_box reach;
        const auto reached = [&](int y, int x) { reach.include_cell(y, x); };
        each_painted(one, [&](int row, int col, int8_t) {
            reached(one.row + row, one.col + col);
            march_trail(one.row + row, one.col + col, step, width, height, own, taken, reached);
        });
        if (reach.empty()) continue;
        const int grown_width = min(reach.width(), MAX_GRID_DIM);
        const int grown_height = min(reach.height(), MAX_GRID_DIM);
        held_object& grown = res.objects[res.count];
        // The grown object is the SAME object, longer, so the mark it was picked
        // by rides along with it.
        grown.selected = one.selected;
        grown.col = (int16_t)reach.col0; grown.row = (int16_t)reach.row0;
        grown.width = (int16_t)grown_width; grown.height = (int16_t)grown_height;
        grown.data.clear();
        each_painted(one, [&](int row, int col, int8_t body) {
            const int at_row = one.row + row, at_col = one.col + col;
            const int down = at_row - reach.row0, across = at_col - reach.col0;
            // The body keeps its colors.
            if (across >= 0 && across < grown_width && down >= 0 && down < grown_height) grown.data[down][across] = body;
            march_trail(at_row, at_col, step, width, height, own, taken, [&](int y, int x) {
                const int trail_across = x - reach.col0, trail_down = y - reach.row0;
                if (trail_across >= 0 && trail_across < grown_width && trail_down >= 0 && trail_down < grown_height)
                    grown.data[trail_down][trail_across] = color;
            });
        });
        ++res.count;
    }
    return res;
}

// --- Map, by what the canvas shows ---

// Repaints each painted cell of the picked objects as paint(color, canvas color
// under it); an object left with no paint drops out, the unpicked ride through.
template <class Paint>
object_list& repaint_over_canvas(program_run& run, const object_list& from, Paint paint) {
    object_list& res = run.fresh();
    const int width = run.canvas.width, height = run.canvas.height;
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        held_object one = source;
        bool any = false;
        for (int row = 0; row < source.height; ++row)
            for (int col = 0; col < source.width; ++col) {
                const int8_t color = one.data[row][col];
                if (color == TRANSPARENT) continue;
                const int x = source.col + col, y = source.row + row;
                const int8_t under = x >= 0 && x < width && y >= 0 && y < height ? (int8_t)run.canvas.data[y][x]
                                                                                 : (int8_t)TRANSPARENT;
                const int8_t painted = paint(color, under);
                one.data[row][col] = painted;
                any = any || painted != TRANSPARENT;
            }
        if (any) res.objects[res.count++] = one;
    }
    return res;
}

// ReadUnder: "what's on the canvas under this?" — each painted cell of a picked
// object takes the canvas color at its place (none off the canvas). The canvas
// is untouched.
object_list& op_read_under(program_run& run, const object_list& from) {
    return repaint_over_canvas(run, from, [](int8_t, int8_t under) { return under; });
}

// MaskBy[c] / MaskOut[c]: keep a picked object's cells only where the canvas
// under them is (MaskBy) or is not (MaskOut) color c -- any painted color at
// c == 10 (mask_keeps), past it the solid blocks of color c - 11 (mask_cells).
// Off the canvas reads transparent, so MaskOut[10] keeps exactly the
// empty-canvas cells. Non-destructive: it filters the layer, never touches the
// canvas.
object_list& op_mask(program_run& run, const object_list& from, int8_t color, bool keep_match) {
    object_list& res = run.fresh();
    const cell_set marked = mask_cells(run.canvas, color);
    for (int index = 0; index < from.count; ++index) {
        const held_object& source = from.objects[index];
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        held_object one{ masked(source, marked, keep_match), true };
        if (one.cell_count() > 0) res.objects[res.count++] = one;
    }
    return res;
}

// --- Translate, the whole picture ---

// ---- Canonical pose (NormalizeOrientation[1]) ----
// What the eight poses mean is 1_readings.hpp's to say (pose_source and its
// inverse posed_place). Every example picks the pose whose row-major cell
// stream is lexicographically smallest — the whole grid votes, so rotated or
// mirrored pairs land identically.
int best_pose(const grid& g) {
    int best = 0;
    const int cell_count = g.width * g.height;
    for (int pose = 1; pose < 8; ++pose) {
        const int pose_width = (pose & 1) ? g.height : g.width;
        const int best_width = (best & 1) ? g.height : g.width;
        for (int i = 0; i < cell_count; ++i) {
            const auto [row, col] = pose_source(pose, g.width, g.height, i / pose_width, i % pose_width);
            const auto [best_row, best_col] = pose_source(best, g.width, g.height, i / best_width, i % best_width);
            const int8_t posed = g.data[row][col], kept = g.data[best_row][best_col];
            if (posed != kept) { if (posed < kept) best = pose; break; }
        }
    }
    return best;
}

// The canvas set in the pose, or `undo`ne out of it.
void pose_canvas(program_run& run, int pose, bool undo) {
    const int width = run.canvas.width, height = run.canvas.height;
    const int frame_width = undo ? ((pose & 1) ? height : width) : width;
    const int frame_height = undo ? ((pose & 1) ? width : height) : height;
    const int posed_width = (pose & 1) ? frame_height : frame_width;
    const int posed_height = (pose & 1) ? frame_width : frame_height;
    const cells before = run.canvas.data;
    for (int row = 0; row < posed_height; ++row)
        for (int col = 0; col < posed_width; ++col) {
            const auto [from_row, from_col] = pose_source(pose, frame_width, frame_height, row, col);
            if (undo) run.canvas.data[from_row][from_col] = before[row][col];
            else      run.canvas.data[row][col] = before[from_row][from_col];
        }
    run.canvas.width = undo ? frame_width : posed_width;
    run.canvas.height = undo ? frame_height : posed_height;
}

// Normalize (pick and apply the best pose) or restore (undo the remembered
// pose). The objects are always posed; the canvas once a program, flag-gated,
// only while its size lives in the input's frame.
object_list& op_canonical(program_run& run, const object_list& from, bool normalize) {
    // The pose is kept in the flag while active, worked out only on the first Normalize.
    const int pose = normalize
        ? ((run.orientation_flipped & 8) ? (run.orientation_flipped & 7) : best_pose(run.input))
        : (run.orientation_flipped & 7);
    const int width = run.input.width, height = run.input.height;
    const auto placed = [&](int row, int col) {
        return normalize ? posed_place(pose, width, height, row, col) : pose_source(pose, width, height, row, col);
    };
    object_list& res = run.fresh();
    res.count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < res.count; ++i) {
        const held_object& source = from.objects[i];
        held_object& posed = res.objects[i] = source;
        const auto [first_row, first_col] = placed(source.row, source.col);
        const auto [last_row, last_col] = placed(source.row + source.height - 1, source.col + source.width - 1);
        posed.row = (int16_t)min(first_row, last_row);
        posed.col = (int16_t)min(first_col, last_col);
        posed.width = (pose & 1) ? source.height : source.width;
        posed.height = (pose & 1) ? source.width : source.height;
        const int posed_height = min((int)posed.height, MAX_GRID_DIM), posed_width = min((int)posed.width, MAX_GRID_DIM);
        for (int row = 0; row < posed_height; ++row)
            for (int col = 0; col < posed_width; ++col) posed.data[row][col] = TRANSPARENT;
        const int source_height = min((int)source.height, MAX_GRID_DIM), source_width = min((int)source.width, MAX_GRID_DIM);
        for (int row = 0; row < source_height; ++row)
            for (int col = 0; col < source_width; ++col) {
                const int8_t color = source.data[row][col];
                if (color == TRANSPARENT) continue;
                auto [to_row, to_col] = placed(source.row + row, source.col + col);
                to_row -= posed.row;
                to_col -= posed.col;
                if (to_row >= 0 && to_row < MAX_GRID_DIM && to_col >= 0 && to_col < MAX_GRID_DIM)
                    posed.data[to_row][to_col] = color;
            }
    }
    if (normalize) {
        if (!(run.orientation_flipped & 8)) pose_canvas(run, pose, false);
        run.orientation_flipped = (int8_t)(8 | pose);
    } else if (run.orientation_flipped & 8) {
        pose_canvas(run, pose, true);
        run.orientation_flipped = 0;
    }
    return res;
}

// NormalizeOrientation[0] lays a landscape example upright: the objects turn
// on their side iff the INPUT lies on its side, and the canvas is brought
// upright (only when it lies on its side, so asking twice changes nothing).
// RestoreOrientation undoes exactly what Normalize did this program -- the flag
// says, where a square grid's size could not. Objects and canvas keep one
// orientation after each.
object_list& op_rotate(program_run& run, const object_list& from, bool normalize) {
    bool turn_objects, turn_canvas;
    if (normalize) {
        const bool flip = lies_on_its_side(run.input.width, run.input.height);
        turn_objects = flip;
        turn_canvas = lies_on_its_side(run.canvas.width, run.canvas.height);
        run.orientation_flipped = flip ? 1 : 0;
    } else {
        const bool flip = run.orientation_flipped != 0;
        turn_objects = flip;
        turn_canvas = flip;
        run.orientation_flipped = 0;
    }
    object_list& res = run.fresh();
    if (!turn_objects) {
        copy_objects(res, from, min(from.count, MAX_OBJECTS_IN_A_LIST));
    } else {
        for (int i = 0; i < from.count && i < MAX_OBJECTS_IN_A_LIST; ++i) {
            const held_object& source = from.objects[i];
            // Turned on its side: the new width is the old height.
            const int width = min((int)source.height, MAX_GRID_DIM), height = min((int)source.width, MAX_GRID_DIM);
            held_object& turned = res.objects[i] = source;
            for (int at = 0; at < MAX_GRID_CELLS; ++at) {
                const int row = cell_row(at), col = cell_col(at);
                turned.data.values[at] = row < height && col < width ? (int8_t)source.data[col][row] : (int8_t)TRANSPARENT;
            }
            turned.col = source.row; turned.row = source.col;
            turned.width = (int16_t)width; turned.height = (int16_t)height;
            res.count = i + 1;
        }
    }
    if (turn_canvas) {
        cells& drawn = run.canvas.data;
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = row + 1; col < MAX_GRID_DIM; ++col) swap(drawn[row][col], drawn[col][row]);
        swap(run.canvas.width, run.canvas.height);
    }
    return res;
}

// --- Generate, onto other pieces ---

// CopyTo[anchor]: copy each selected object to each unselected destination,
// aligning like anchors of their frames. 0..8 name the 3x3 anchors in row-major
// order; 4 aligns centers. The result is copies only, all selected, in source
// then destination order. Colors, holes and sizes survive; destinations
// contribute only positions and frames. Painting/recoloring remain separate
// operations.
object_list& op_copy_to(program_run& run, const object_list& from, int anchor) {
    object_list& res = run.fresh();
    int picked = 0;
    for (int i = 0; i < from.count; ++i)
        if (from.objects[i].selected) ++picked;
    if (!picked) { copy_list(res, from); return res; }
    // Do not silently hand on only a prefix of every pattern at every place.
    if (picked * (from.count - picked) > MAX_OBJECTS_IN_A_LIST) return res;
    for (int i = 0; i < from.count; ++i) {
        const held_object& pattern = from.objects[i];
        if (!pattern.selected || pattern.width <= 0 || pattern.height <= 0) continue;
        for (int j = 0; j < from.count; ++j) {
            const held_object& target = from.objects[j];
            if (target.selected || target.width <= 0 || target.height <= 0) continue;
            held_object& copy = res.objects[res.count++] = pattern;
            copy.col = (int16_t)aligned_copy_start(pattern.width, target.col, target.width, anchor % 3);
            copy.row = (int16_t)aligned_copy_start(pattern.height, target.row, target.height, anchor / 3);
        }
    }
    return res;
}

// ProjectOnSeeds: project the template (the first object held) onto each seed
// -- every piece of the input but the largest. For every seed the copies march
// outward from it away from the template until off the grid (project_along),
// each stamp in the seed's color. The input itself comes first, so the stamps
// paint over it.
object_list& op_project_on_seeds(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const object_list& seeds = run.lists.seeds;
    if (from.count <= 0 || seeds.count <= 0) return res;
    const held_object& stencil = from.objects[0];
    if (stencil.width <= 0 || stencil.height <= 0) return res;
    segment_objects(run.input, res, grouping::whole_grid);
    const cell_box stencil_box = stencil.box();
    for (int s = 0; s < seeds.count; ++s) {
        const held_object& seed = seeds.objects[s];
        const int8_t seed_color = seed.first_color();
        const int8_t paint = seed_color >= 0 ? seed_color : stencil.first_color();
        project_along(stencil_box, seed.box(),
                      run.canvas.width, run.canvas.height, [&](int row, int col) {
            if (res.count >= MAX_OBJECTS_IN_A_LIST) return false;
            held_object& stamp = res.objects[res.count++];
            stamp.col = (int16_t)col; stamp.row = (int16_t)row;
            stamp.width = stencil.width; stamp.height = stencil.height;
            stamp.data.clear();
            each_painted(stencil, [&](int at_row, int at_col, int8_t) { stamp.data[at_row][at_col] = paint; });
            return true;
        });
    }
    return res;
}

// SnapToAnchors: rigidly stamp each "body" onto its color-coded anchor cluster.
// Where each body goes is arc::snap_to_anchors' to say -- the body color, the
// runs, the keys and anchors, the eight turns and the tie-break -- and the rule
// that replays this op asks it the same question, so the two cannot set a body
// down differently. What stays here is only the shape the answer is written
// into: one object per body set down, drawn turned about its key onto the
// anchor that key lands on. Output = the stamped bodies only; originals and
// loose anchors are consumed.
object_list& op_snap_to_anchors(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    grid drawn;
    // The whole ceiling, as Split's is: a grown chain is not cropped to the input.
    paint_objects(from, MAX_GRID_DIM, MAX_GRID_DIM, drawn);
    const int width = drawn.width, height = drawn.height;
    if (width <= 0 || height <= 0) return res;
    int8_t label[MAX_GRID_DIM][MAX_GRID_DIM];
    snap_to_anchors(width, height, [&](int row, int col) { return (int)drawn.data[row][col]; }, label,
        [&](int body, int way, int key_row, int key_col, int to_row, int to_col) {
            if (res.count >= MAX_OBJECTS_IN_A_LIST) return false;
            // Where each of the body's cells lands, turned about its key.
            const auto landing = [&](int row, int col, int& at_row, int& at_col) {
                int down = 0, across = 0;
                snap_turn(way, row - key_row, col - key_col, down, across);
                at_row = to_row + down;
                at_col = to_col + across;
                return at_row >= 0 && at_row < MAX_GRID_DIM && at_col >= 0 && at_col < MAX_GRID_DIM;
            };
            cell_box landed;
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    int at_row = 0, at_col = 0;
                    if (label[row][col] == body && landing(row, col, at_row, at_col)) landed.include_cell(at_row, at_col);
                }
            if (landed.empty()) return false;
            held_object& stamped = res.objects[res.count];
            stamped.col = (int16_t)landed.col0; stamped.row = (int16_t)landed.row0;
            stamped.width = (int16_t)landed.width(); stamped.height = (int16_t)landed.height();
            stamped.data.clear();
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    int at_row = 0, at_col = 0;
                    if (label[row][col] == body && landing(row, col, at_row, at_col))
                        stamped.data[at_row - landed.row0][at_col - landed.col0] = drawn.data[row][col];
                }
            res.count++;
            return true;
        });
    return res;
}

// --- The program ---

// The op with its value, or its Tile, on the list in hand, `from`: the list it
// hands on. With nothing in hand only a source makes anything; every other op
// hands on an empty list.
object_list& one_op(program_run& run, const flat_program_op& op, const object_list* from) {
    const node_type type = op.type;
    const int8_t value = (int8_t)op.value;
    if (type == node_type::Input) return op_input(run);
    if (type == node_type::Output) return op_output(run);
    if (!from) return run.fresh();
    const object_list& in = *from;
    switch (type) {
    case node_type::TurnRight:
    case node_type::TurnLeft:
    case node_type::FlipHorizontal:
    case node_type::FlipVertical:            return op_turn(run, in, type);
    case node_type::ScaleUp:                 return op_scale_up(run, in, value);
    case node_type::ScaleDown:               return op_scale_down(run, in, value);
    // [0] lays a landscape example upright; [1] sets every example in its
    // canonical pose, the whole grid voting, which works on square grids too.
    case node_type::NormalizeOrientation:    return value == 1 ? op_canonical(run, in, true) : op_rotate(run, in, true);
    case node_type::RestoreOrientation:      return (run.orientation_flipped & 8) ? op_canonical(run, in, false)
                                                                                  : op_rotate(run, in, false);
    case node_type::DeleteSelection:         return op_delete_selection(run, in);
    case node_type::Union:                   return op_union(run, in);
    case node_type::Underlay:                return op_underlay(run, in);
    case node_type::Withhold:                return op_withhold(run, in);
    case node_type::Intersection:            return op_intersection(run, in);
    case node_type::Difference:              return op_difference(run, in);
    case node_type::InteriorColor:           return op_interior_color(run, in, value);
    case node_type::ShapeToInteriorColorMap: return op_shape_to_interior_color_map(run, in);
    case node_type::Recolor:                 return op_recolor(run, in, value);
    case node_type::MoveRow:                 return op_move(run, in, value, true);
    case node_type::MoveColumn:              return op_move(run, in, value, false);
    case node_type::Split:                   return op_split(run, in, value);
    case node_type::Colors:                  return op_colors(run, in);
    case node_type::SelectMostCommonColor:   return op_select_most_common_color(run, in);
    case node_type::SelectUnique:            return op_select_unique(run, in, value);
    case node_type::SelectLargestObject:     return op_select_object_by_size(run, in, value, true);
    case node_type::SelectSmallestObject:    return op_select_object_by_size(run, in, value, false);
    case node_type::SortBySize:              return op_sort_by_size(run, in, value == 0);
    case node_type::SizeRankColorMap:        return op_size_rank_color_map(run, in);
    case node_type::Bridge:                  return op_bridge(run, in, value);
    case node_type::Partition:               return op_partition(run, in);
    case node_type::Panes:                   return op_panes(run, in);
    case node_type::BidiColorMap:            return op_bidi_color_map(run, in, value);
    case node_type::ColorFrom:               return op_color_from(run, in, value);
    case node_type::WouldFit:                return op_would_fit(run, in, value);
    case node_type::AlignTo:                 return op_align_to(run, in, value);
    case node_type::MoveToward:              return op_move_toward(run, in, value);
    case node_type::MirrorAround:            return op_mirror_around(run, in, value);
    case node_type::SelectObjectsWithHoles:  return op_select_objects_with_holes(run, in, value);
    case node_type::SelectAtLeast:           return op_select_at_least(run, in, op.value);
    case node_type::SelectRunningAlong:      return op_select_running_along(run, in, value);
    case node_type::SelectWalls:             return op_select_walls(run, in);
    case node_type::FitInto:                 return op_fit_into(run, in, value);
    case node_type::ShapeFrom:               return op_shape_from(run, in, value);
    case node_type::SelectColor:             return op_select_color(run, in, value);
    case node_type::SelectElectedBackground: return op_select_elected_background(run, in, value);
    case node_type::SelectAll:               return op_select_all(run, in);
    case node_type::InvertSelection:         return op_invert_selection(run, in);
    case node_type::Shift:                   return op_shift(run, in, value);
    case node_type::SlamRow:                 return op_slam(run, in, true, value);
    case node_type::SlamColumn:              return op_slam(run, in, false, value);
    case node_type::AddColor:                return op_add_color(run, in, value);
    case node_type::Fit:                     return op_fit(run, in, value);
    case node_type::Skew:                    return op_skew(run, in, value);
    case node_type::Trim:                    return op_trim(run, in);
    case node_type::Fractal:                 return op_fractal(run, in);
    case node_type::Tile:                    return op_tile(run, in, op.tile);
    case node_type::Pattern:                 return op_pattern(run, in, value);
    case node_type::ProjectOnSeeds:          return op_project_on_seeds(run, in);
    case node_type::CopyTo:                  return op_copy_to(run, in, value);
    case node_type::IntersectionList:        return op_intersection_list(run, in);
    case node_type::UnionList:               return op_union_list(run, in);
    case node_type::Layout:                  return op_layout(run, in, value);
    case node_type::Origin:                  return op_origin(run, in);
    case node_type::AlignToCanvas:           return op_align_to_canvas(run, in);
    case node_type::OverlapCorner:           return op_overlap_corner(run, in, value);
    case node_type::SnapToAnchors:           return op_snap_to_anchors(run, in);
    case node_type::ShapeColorMap:           return op_shape_color_map(run, in, value);
    case node_type::ExtendOrthogonal:        return op_extend_rays(run, in, value, false);
    case node_type::ExtendDiagonal:          return op_extend_rays(run, in, value, true);
    case node_type::ExtendTo:                return op_extend_to(run, in, value);
    case node_type::MaskBy:                  return op_mask(run, in, value, true);
    case node_type::MaskOut:                 return op_mask(run, in, value, false);
    case node_type::ReadUnder:               return op_read_under(run, in);
    case node_type::MoveToEdge:              return op_move_to_edge(run, in, value);
    case node_type::VoteDenoise:             return op_vote_denoise(run, in);
    case node_type::Outline:                 return op_outline(run, in);
    case node_type::SelectRow:               return op_select_band(run, in, value, true);
    case node_type::SelectColumn:            return op_select_band(run, in, value, false);
    case node_type::SelectRowFromMiddle:     return op_select_from_middle(run, in, value, true);
    case node_type::SelectColumnFromMiddle:  return op_select_from_middle(run, in, value, false);
    case node_type::SelectPartition:         return op_select_partition(run, in, value);
    case node_type::SelectLikeKey:           return op_select_like_key(run, in, value);
    case node_type::SelectPartner:           return op_select_partner(run, in, op.value);
    default:                                 return run.fresh();
    }
}

// ONE HOME FOR THE MARK. Many ops build their objects fresh, so a mark handed
// in would die unless every op remembered to carry it -- and forgetting means
// painting nothing, a silent wrong answer rather than a loud one. Carried once,
// here, so a new op cannot forget: what came in picked comes out picked,
// object for object when the op handed back as many objects as it took, every
// one otherwise, since a carve's pieces have no one parent to ask. Only when
// the op marked NOTHING itself -- an op that has an opinion about which of its
// objects are picked keeps it. Selectors are excluded by role (marking IS
// their work), the ops that clear do so for reasons of their own, and Fit
// carries marks itself, even when an empty target drops every picked object --
// as UnionList and IntersectionList do, the mark their folds' identity, even
// when the picked fold keeps no cell and only the unpicked one is handed on.
void carry_the_marks(node_type type, const object_list* last, object_list& made) {
    if (!last || !is_act_op(type) || type == node_type::Fit || type == node_type::UnionList ||
        type == node_type::IntersectionList || clears_the_marks(type))
        return;
    bool came_in_picked = false, went_out_picked = false;
    for (int i = 0; i < last->count; ++i) came_in_picked |= last->objects[i].selected;
    for (int i = 0; i < made.count; ++i) went_out_picked |= made.objects[i].selected;
    if (!came_in_picked || went_out_picked) return;
    const bool one_for_one = made.count == last->count;
    for (int i = 0; i < made.count; ++i) made.objects[i].selected = one_for_one ? last->objects[i].selected : true;
}

// The program's ops one after another, from the first.
void run_ops(program_run& run, const flat_program& program) {
    for (size_t i = 0; i < program.size(); ++i) {
        const node_type type = program[i].type;
        // The chain starts from the input: the FIRST op sees the input rather
        // than an empty list, so no program needs a leading Input.
        if (i == 0 && !run.current && !is_get_op(type)) run.current = &op_input(run);
        // While a canonical pose is active, anything the previous op fetched from
        // the input's frame joins the canonical frame by itself — one Normalize
        // per chain instead of one per fetch.
        if ((run.orientation_flipped & 8) && i > 0 && run.current &&
            (program[i - 1].type == node_type::Input || program[i - 1].type == node_type::Seeds))
            run.current = &op_canonical(run, *run.current, true);
        // CurrentObjects: the objects in hand are already the ones in hand.
        if (type == node_type::CurrentObjects) continue;
        if (type == node_type::Seeds) { run.current = &run.lists.seeds; continue; }
        object_list* const last = run.current;
        object_list& made = one_op(run, program[i], last);
        carry_the_marks(type, last, made);
        run.current = &made;
    }
    if (!run.current) run.current = &run.fresh();
}

// THE SEEDS: every piece of the input but the largest, smallest box first,
// leaving out the elected ground's pieces (read_seeds, 4_objects.hpp, which the
// reading reads too, so what an op is handed and what a reading says it will
// be handed cannot drift apart).
void read_the_seeds(const grid& input, int ground, run_lists& lists) {
    object_list& pieces = lists.input_pieces;
    pieces.count = 0;
    object_walk_scratch scratch;
    each_object(input, grouping::touching_same_color, scratch,
        [&](const object_summary& box, const int16_t* taken, int taken_count) {
            if (pieces.count >= MAX_OBJECTS_IN_A_LIST) return;
            held_object& one = pieces.objects[pieces.count++];
            one.col = box.col; one.row = box.row;
            one.width = box.width; one.height = box.height;
            one.selected = false;
            one.data.clear();
            for (int i = 0; i < taken_count; ++i) {
                const int row = cell_row(taken[i]), col = cell_col(taken[i]);
                one.data[row - one.row][col - one.col] = input.data[row][col];
            }
        });
    if (ground >= 0) {
        int kept = 0;
        for (int i = 0; i < pieces.count; ++i)
            if (pieces.objects[i].first_color() != ground) pieces.objects[kept++] = pieces.objects[i];
        pieces.count = kept;
    }
    object_list& seeds = lists.seeds;
    seeds.count = 0;
    vector<object_summary> boxes;
    for (int i = 0; i < pieces.count; ++i) {
        object_summary box;
        box.width = (int8_t)pieces.objects[i].width;
        box.height = (int8_t)pieces.objects[i].height;
        boxes.push_back(box);
    }
    for (int which : read_seeds(boxes).seeds) seeds.objects[seeds.count++] = pieces.objects[which];
}

// The canvas cut down to the box of what is painted on it, when anything is.
void crop_to_paint(grid& canvas) {
    cell_box painted;
    for (int row = 0; row < canvas.height; ++row)
        for (int col = 0; col < canvas.width; ++col)
            if (canvas.data[row][col] != TRANSPARENT) painted.include_cell(row, col);
    if (painted.empty()) return;
    const int width = painted.width(), height = painted.height();
    if (painted.row0 > 0 || painted.col0 > 0) {
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                canvas.data[row][col] = canvas.data[painted.row0 + row][painted.col0 + col];
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < MAX_GRID_DIM; ++col)
                if (row >= height || col >= width) canvas.data[row][col] = TRANSPARENT;
    }
    canvas.width = width;
    canvas.height = height;
}

}  // namespace

inline grid environment::draw(const node* tree, const example& ex, const puzzle& p, object_list* held) {
    if (held) held->count = 0;
    const flat_program program = flatten_unary_program(tree);
    if (program.empty()) return {};
    static thread_local auto lists = make_unique<run_lists>();
    // ONLY WHEN THE RULE STANDS: a detector that gave up part way through
    // leaves the keys it had already learned behind it, and holds() is the
    // whole of what says whether they were ever meant to be used.
    static const shape_table no_table;
    const inference& read = p.inf;
    program_run run{
        .input = ex.input,
        .lists = *lists,
        .colors_by_size = read.colors.output_colors_by_size,
        .colors_by_size_count = read.colors.output_colors_by_size_count,
        .shape_colors = read.shape_color_map.holds() ? static_cast<const shape_table&>(read.shape_color_map) : no_table,
        .cut_shape = read.correspondence.shape,
    };
    // The canvas the size rule gives this input, the reading's own sizing.
    int width = 0, height = 0;
    read.size.canvas_size_for(ex.input.width, ex.input.height, width, height);
    run.canvas.width = width;
    run.canvas.height = height;
    read_the_seeds(ex.input, read.ground_election.input_elected(), *lists);
    // The cut is a fact the puzzle holds, and a test input is cut by the same
    // reading the pairs were: the fact reads the grid again rather than
    // remembering the pairs, so what the op cuts on and what inference said
    // cannot drift apart.
    if (!read.pane_split.holds() || !read.pane_split.panes_of(ex.input, run.panes)) run.panes.clear();
    run_ops(run, program);
    // Under BoundingBox the canvas was the whole ceiling, and the drawing is
    // the box of what was painted on it.
    if (read.size.Chosen == output_dimension_rule::BoundingBox) crop_to_paint(run.canvas);
    if (held) copy_list(*held, *run.current);
    return run.canvas;
}

}  // namespace arc
