// THE OPS: what each op does to the list a program holds and to the canvas it
// paints, run one after another on one example (environment::draw). What an op
// shares with the reading's replay is 8b_op_math.hpp's arithmetic, so the two
// cannot draw differently.
#include "13_environment.hpp"
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
    // caller's words (Colors, SelectColor, DeleteSelection).
    vector<pane_split_facts::pane> panes;
    // The training outputs' colors, most cells first (SizeRankColorMap).
    const int8_t* colors_by_size;
    int colors_by_size_count;
    // The learned shape -> color table (ShapeColorMap and kin).
    const shape_table& shape_colors;
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

// Every cell an object holds, in its own frame: see(row, col, color), a room's
// empty cells (BLANK) too.
template <class See>
void each_painted(const held_object& one, See see) {
    for (int row = 0; row < one.height; ++row)
        for (int col = 0; col < one.width; ++col)
            if (const int8_t color = one.mask[row][col]; color != TRANSPARENT) see(row, col, color);
}

// Every cell of an object a grid shows, in its own frame: its colored ones, a
// room's empty cells showing nothing.
template <class See>
void each_shown(const held_object& one, See see) {
    each_painted(one, [&](int row, int col, int8_t color) { if (is_color(color)) see(row, col, color); });
}

int painted_count(const held_object& one) {
    int count = 0;
    each_painted(one, [&](int, int, int8_t) { ++count; });
    return count;
}

// The box the paint actually fills, in the object's own frame, which is not
// always the box the object declares -- Colors hands back layers carrying the
// whole source box.
cell_box painted_box(const held_object& one) {
    cell_box box;
    each_painted(one, [&](int row, int col, int8_t) { box.include_cell(row, col); });
    return box;
}

// An object's first painted color in reading order, TRANSPARENT when it has none.
int8_t first_color(const held_object& one) {
    for (int row = 0; row < one.height; ++row)
        for (int col = 0; col < one.width; ++col)
            if (one.mask[row][col] != TRANSPARENT) return one.mask[row][col];
    return (int8_t)TRANSPARENT;
}

// Shared color-source rule for painting ops: resolve an annotation value to a
// per-object color. 0..9 = that literal; 10 = the object's rarest color;
// 11 = its most common. Frequency ties break toward the lower color number.
// TRANSPARENT if the mask is empty.
int8_t paint_color(const held_object& one, int8_t value) {
    return object_paint_color(one.height, one.width, value,
        [&](int row, int col) { return (int8_t)one.mask[row][col]; });
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
        each_shown(one, [&](int row, int col, int8_t color) {
            const int at_row = one.y0 + row, at_col = one.x0 + col;
            if (at_row >= 0 && at_row < MAX_GRID_DIM && at_col >= 0 && at_col < MAX_GRID_DIM)
                out.data[at_row][at_col] = color;
        });
    }
}

// The pieces of `g`, each in the frame of what they were cut from (x0, y0,
// width, height, which holds every painted cell of g). What counts as one
// object is arc::each_object's to say, as the reading's is, so the two cannot
// come to different answers; segmentation never tightens a frame (Trim is the
// only shrinker).
void segment_objects(const grid& g, object_list& out, grouping how, int x0, int y0, int width, int height) {
    out.count = 0;
    if (g.width <= 0 || g.height <= 0 || width <= 0 || height <= 0) return;
    object_walk_scratch scratch;
    each_object(g, TRANSPARENT, how, scratch, [&](const object_summary&, const int16_t* taken, int taken_count) {
        if (out.count >= MAX_OBJECTS_IN_A_LIST) return;
        held_object& one = out.objects[out.count++];
        one.x0 = (int16_t)x0; one.y0 = (int16_t)y0;
        one.x1 = (int16_t)(x0 + width - 1); one.y1 = (int16_t)(y0 + height - 1);
        one.width = (int16_t)width; one.height = (int16_t)height;
        one.mask.clear();
        for (int i = 0; i < taken_count; ++i) {
            const int row = cell_row(taken[i]), col = cell_col(taken[i]);
            one.mask[row - y0][col - x0] = g.data[row][col];
        }
    });
}

// Calls cell(row, col, color) for each shown, on-canvas cell of the picked
// objects among from's first `count`, one object at a time so later objects win
// shared cells.
template <class Cell>
void each_picked_cell(const program_run& run, const object_list& from, int count, Cell cell) {
    for (int i = 0; i < count; ++i) {
        const held_object& one = from.objects[i];
        if (!one.selected) continue;
        each_shown(one, [&](int row, int col, int8_t color) {
            const int at_row = one.y0 + row, at_col = one.x0 + col;
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
            one.mask.values[at] = row < height && col < width ? pixel(source, row, col) : (int8_t)TRANSPARENT;
        }
        one.width = (int16_t)width;
        one.height = (int16_t)height;
        one.x1 = (int16_t)(one.x0 + width - 1);
        one.y1 = (int16_t)(one.y0 + height - 1);
    }
    res.count = count;
}

// THE OBJECTS A PARTNER WORD READS, as the op holds them: the box each one's
// paint fills on the canvas, whether it paints a cell there, whether it is
// picked, its color and the colors it holds -- the questions the partner
// helpers ask (each_partner and kin, 8b_op_math.hpp), asked as inference's
// partner_pieces asks them of the pieces it replays.
struct partner_view {
    const object_list& list;
    int count() const { return list.count; }
    bool is_picked(int i) const { return list.objects[i].selected; }
    int color(int i) const { return first_color(list.objects[i]); }
    cell_box box(int i) const {
        const held_object& one = list.objects[i];
        const cell_box paint = painted_box(one);
        cell_box found;
        if (!paint.empty()) {
            found.include_cell(one.y0 + paint.row0, one.x0 + paint.col0);
            found.include_cell(one.y0 + paint.row1, one.x0 + paint.col1);
        }
        return found;
    }
    bool painted(int i, int row, int col) const { return color_at(i, row, col) != TRANSPARENT; }
    // The blocks object i's shape is drawn in (blocks_of), its colors aside.
    shape_blocks blocks(int i) const {
        const cell_box own = box(i);
        return blocks_of(own.width(), own.height(),
                         [&](int row, int col) { return color_at(i, own.row0 + row, own.col0 + col); }, AS_IT_LIES);
    }
    // The color object i paints at that cell of the canvas, transparent off it.
    int color_at(int i, int row, int col) const {
        const held_object& one = list.objects[i];
        const int down = row - one.y0, across = col - one.x0;
        return down >= 0 && across >= 0 && down < one.height && across < one.width ? (int)one.mask[down][across]
                                                                                    : TRANSPARENT;
    }
    // Whether object i holds a cell of `color`.
    bool holds(int i, int color) const {
        const held_object& one = list.objects[i];
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] != TRANSPARENT && one.mask[row][col] == color) return true;
        return false;
    }
    // What each_partner asks (touches, shapes_alike).
    bool touching(int mine, int other, bool sides_only) const { return touches(*this, mine, other, { 0, 0 }, sides_only); }
    bool alike(int mine, int other, int like) const { return shapes_alike(*this, mine, other, like); }
    // The object that shows a cell of the key's drawing (key_piece_at).
    int key_piece(int row, int col) const { return key_piece_at(*this, row, col); }
    // Object i's one partner by `how` among the objects not picked, or -1.
    int partner(partner_by how, int i) const { return partner_of(how, i, *this); }
    // Object i's one partner of `kind` (partner_of_kind) among the objects not picked, or -1.
    int partner_of_kind(int kind, int i) const { return arc::partner_of_kind(kind, i, *this); }
    // The color object i takes from its partners by `how`, or -1.
    int color_from(partner_by how, int i) const { return partner_color(how, i, *this); }
};

// Read the key in the piece's current box, as inference does. Only Trim
// removes empty margins before the key is read.
uint32_t shape_key_of_piece(const held_object& one, int kind) {
    return shape_key_of(kind, one.width, one.height, [&](int row, int col) {
        return one.mask[row][col];
    });
}

// --- Sources ---

// A source has no frame to inherit, so it takes the one its own content makes:
// the box around every cell that is not transparent. This is not the tightening
// the frame law reserves for Trim — there is no earlier box being discarded,
// only a box being chosen for the first time. A grid with nothing on it makes no
// object at all, because nothing painted is nothing to hand on.
void frame_to_content(object_list& res, const grid& g) {
    cell_box content;
    for (int row = 0; row < g.height; ++row)
        for (int col = 0; col < g.width; ++col)
            if (g.data[row][col] != TRANSPARENT) content.include_cell(row, col);
    if (content.empty()) { res.count = 0; return; }
    held_object& one = res.objects[0];
    one.x0 = (int16_t)content.col0; one.y0 = (int16_t)content.row0;
    one.x1 = (int16_t)content.col1; one.y1 = (int16_t)content.row1;
    one.width = (int16_t)content.width();
    one.height = (int16_t)content.height();
    one.mask.clear();
    for (int row = content.row0; row <= content.row1; ++row)
        for (int col = content.col0; col <= content.col1; ++col)
            one.mask[row - content.row0][col - content.col0] = g.data[row][col];
    res.count = 1;
}

object_list& op_input(program_run& run) {
    object_list& res = run.fresh();
    frame_to_content(res, run.input);
    return res;
}

// Output is a snapshot of the WHOLE canvas, including its empty frame, read
// back as it is with its transparent cells kept, so what follows sees what has
// been painted so far. Trim is the only word that reduces this frame to its
// painted bounds.
object_list& op_output(program_run& run) {
    object_list& res = run.fresh();
    if (run.canvas.width <= 0 || run.canvas.height <= 0) return res;
    held_object& one = res.objects[0];
    one.x0 = one.y0 = 0;
    one.width = (int16_t)run.canvas.width;
    one.height = (int16_t)run.canvas.height;
    one.x1 = (int16_t)(one.width - 1);
    one.y1 = (int16_t)(one.height - 1);
    one.mask = run.canvas.data;
    res.count = 1;
    return res;
}

// --- Reduce ---

// Split[depth]: segmentation depth, coarse to fine. [0] = connected object,
// colors mixed. [1] = connected object of one color. [2] = single pixels.
// [3], finer than [1] and coarser than [2], came last: one color's cells
// joined by sides alone, a corner joining nothing.
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
                one.x0 = one.x1 = (int16_t)col;
                one.y0 = one.y1 = (int16_t)row;
                one.width = one.height = 1;
                one.mask.clear();
                one.mask[0][0] = color;
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
        x0 = min(x0, (int)one.x0); y0 = min(y0, (int)one.y0);
        x1 = max(x1, (int)one.x1); y1 = max(y1, (int)one.y1);
    }
    x0 = max(x0, 0); y0 = max(y0, 0);
    x1 = min(x1, MAX_GRID_DIM - 1); y1 = min(y1, MAX_GRID_DIM - 1);
    segment_objects(drawn, res,
                    depth == 0 ? grouping::touching_any_color
                    : depth == 3 ? grouping::touching_sides_same_color
                                 : grouping::touching_same_color,
                    x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    return res;
}

// Colors splits each source object into ALL its color layers, unselected.
// Every layer keeps the source frame. Removal and trimming are separate words.
object_list& op_colors(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count; ++i) {
        const held_object& source = from.objects[i];
        const int width = source.width, height = source.height;
        for (int color = 0; color < COLOR_COUNT; ++color) {
            bool found = false;
            for (int row = 0; row < height && !found; ++row)
                for (int col = 0; col < width; ++col)
                    if (source.mask[row][col] == color) { found = true; break; }
            if (!found) continue;
            if (res.count >= MAX_OBJECTS_IN_A_LIST) return res;
            held_object& layer = res.objects[res.count];
            layer.x0 = source.x0; layer.y0 = source.y0; layer.x1 = source.x1; layer.y1 = source.y1;
            layer.width = source.width; layer.height = source.height;
            // A carve CLEARS, and that is what makes a carve useful: the pick
            // that follows it is how a sentence narrows to one layer. Handing
            // the mark down instead leaves every layer marked, and since
            // selectors only ever ADD, the next pick can no longer say "just
            // this one" -- measured on 05a7bcf2, where it recoloured all three.
            layer.selected = false;
            layer.mask.clear();
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    if (source.mask[row][col] == color) layer.mask[row][col] = (int8_t)color;
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
            for (int col = 0; col < one.width; ++col) freq.add((int8_t)one.mask[row][col]);
    }
    const int top = freq.most();
    if (top < 0) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        for (int row = 0; row < one.height && !one.selected; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] == top) { one.selected = true; break; }
    }
    return res;
}

// SortBySize: reorder the PICKED objects by pixel count, among the slots they
// already occupy. Everything unpicked stays exactly where it is, so the act
// touches only what was chosen -- the same law the other acts follow, said for
// an ordering: a sort of some is a sort of their own places. Biggest first
// when `biggest_first`. Nothing picked means nothing is reordered.
object_list& op_sort_by_size(program_run& run, const object_list& from, bool biggest_first) {
    object_list& res = run.fresh();
    if (from.count <= 0) return res;
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    copy_objects(res, from, count);
    int slots[MAX_OBJECTS_IN_A_LIST], sizes[MAX_OBJECTS_IN_A_LIST], order[MAX_OBJECTS_IN_A_LIST];
    int picked = 0;
    for (int i = 0; i < count; ++i)
        if (from.objects[i].selected) {
            slots[picked] = i;
            order[picked] = i;
            sizes[picked] = painted_count(from.objects[i]);
            ++picked;
        }
    if (picked == 0) return res;
    for (int i = 1; i < picked; ++i)
        for (int j = i; j > 0 && (biggest_first ? sizes[j] > sizes[j - 1] : sizes[j] < sizes[j - 1]); --j) {
            swap(sizes[j], sizes[j - 1]);
            swap(order[j], order[j - 1]);
        }
    // The k-th picked slot receives the k-th object in sorted order.
    for (int k = 0; k < picked; ++k) res.objects[slots[k]] = from.objects[order[k]];
    return res;
}

// SizeRankColorMap: recolor each object by the puzzle's by-size palette at its
// rank. An empty palette hands back nothing; objects past the palette drop.
object_list& op_size_rank_color_map(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int ranked = run.colors_by_size_count;
    if (ranked <= 0) return res;
    const int count = min(min(from.count, ranked), MAX_OBJECTS_IN_A_LIST);
    for (int i = 0; i < count; ++i) {
        held_object& one = res.objects[res.count] = from.objects[i];
        const int8_t color = run.colors_by_size[i];
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] != TRANSPARENT) one.mask[row][col] = color;
        res.count++;
    }
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
        [&](int i, int row, int col) { return (int8_t)from.objects[i].mask[row][col]; });
    if (found >= 0) res.objects[found].selected = true;
    return res;
}

// SelectLargestObject[rank] / SelectSmallestObject[rank]: mark the rank-th
// object by pixel count, biggest (or smallest) first. Past the ranks, every
// object with the most -- or the fewest -- of a count (count_of).
object_list& op_select_object_by_size(program_run& run, const object_list& from, int rank, bool largest) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (rank >= SIZE_RANKS) {
        const int kind = rank - SIZE_RANKS;
        each_extreme(res.count, largest, [&](int i) {
            const held_object& one = res.objects[i];
            return count_of(kind, min((int)one.height, MAX_GRID_DIM), min((int)one.width, MAX_GRID_DIM),
                            [&](int row, int col) { return (int)one.mask[row][col]; });
        }, [&](int i) { res.objects[i].selected = true; });
        return res;
    }
    if (rank < 0 || rank >= res.count) return res;
    int sizes[MAX_OBJECTS_IN_A_LIST], order[MAX_OBJECTS_IN_A_LIST];
    for (int i = 0; i < res.count; ++i) {
        sizes[i] = painted_count(res.objects[i]);
        order[i] = i;
    }
    for (int i = 1; i < res.count; ++i)
        for (int j = i; j > 0 && (largest ? sizes[j] > sizes[j - 1] : sizes[j] < sizes[j - 1]); --j) {
            swap(sizes[j], sizes[j - 1]);
            swap(order[j], order[j - 1]);
        }
    res.objects[order[rank]].selected = true;
    return res;
}

// `source` clipped to rows [row0, row1) and columns [col0, col1), and there to
// the cells `keep(row, col)` holds, in that frame; false when nothing is left.
template <class Keep>
bool clip_to(const held_object& source, int row0, int col0, int row1, int col1, Keep keep, held_object& one) {
    one.x0 = (int16_t)col0; one.y0 = (int16_t)row0;
    one.x1 = (int16_t)(col1 - 1); one.y1 = (int16_t)(row1 - 1);
    one.width = (int16_t)(col1 - col0); one.height = (int16_t)(row1 - row0);
    one.selected = false;
    one.mask.clear();
    bool any = false;
    each_painted(source, [&](int row, int col, int8_t color) {
        const int at_row = source.y0 + row, at_col = source.x0 + col;
        if (at_row < row0 || at_row >= row1 || at_col < col0 || at_col >= col1 || !keep(at_row, at_col)) return;
        one.mask[at_row - row0][at_col - col0] = color;
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
        if (run.canvas.data[row][col] == TRANSPARENT) run.canvas.data[row][col] = color;
    });
    object_list& res = run.fresh();
    copy_objects(res, from, count);
    return res;
}

// Difference: erase from the canvas every cell a picked object shows. Only
// what was picked lands: a paint that ignored the mark would undo the pick the
// sentence just made.
object_list& op_difference(program_run& run, const object_list& from) {
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (!one.selected) continue;
        each_shown(one, [&](int row, int col, int8_t) {
            const int at_row = one.y0 + row, at_col = one.x0 + col;
            if (at_row >= 0 && at_row < run.canvas.height && at_col >= 0 && at_col < run.canvas.width)
                run.canvas.data[at_row][at_col] = TRANSPARENT;
        });
    }
    object_list& res = run.fresh();
    copy_list(res, from);
    return res;
}

// --- Map ---

// The object with every empty cell its own paint walls off from its frame's
// edge -- the outside is what reaches the edge by sides (mark_outside) --
// painted `fill`.
void paint_interior(const held_object& from, held_object& out, int8_t fill) {
    out = from;
    const int width = min((int)from.width, MAX_GRID_DIM), height = min((int)from.height, MAX_GRID_DIM);
    if (width <= 0 || height <= 0) return;
    const auto wall = [&](int row, int col) { return from.mask[row][col] != TRANSPARENT; };
    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
    mark_outside(height, width, wall, outside);
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            if (!wall(row, col) && !outside[row][col]) out.mask[row][col] = fill;
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
        return color < 0 ? (int)first_color(one) : (int)color;
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
        if (!source.selected) continue;
        // Past the ten colors, the color is read per object: 10 its rarest, 11
        // its most common.
        const int8_t paint = color >= COLOR_COUNT ? paint_color(source, color) : color;
        for (int8_t& cell : one.mask.values)
            if (cell != TRANSPARENT) cell = paint;
    }
    res.count = count;
    return res;
}

// Bridge[way]: the pieces handed back as they came, unmarked, then every bar as
// one drawing, marked, so what follows works on the bars. Which pieces pair up
// (the picked ones alone), the way they reach, and where each bar lies is
// 8b_op_math.hpp's bridge_bars, which the reading replays too; this only lays
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
    bars.x0 = 0; bars.y0 = 0;
    bars.x1 = (int16_t)(MAX_GRID_DIM - 1); bars.y1 = (int16_t)(MAX_GRID_DIM - 1);
    bars.width = bars.height = (int16_t)MAX_GRID_DIM;
    bars.selected = true;
    bars.mask.clear();
    bool drew = false;
    bridge_bars(way, run.canvas.width, run.canvas.height, partner_view{ from },
        [&](const cell_box& bar, int8_t color) {
            for (int row = max(bar.row0, 0); row <= min(bar.row1, MAX_GRID_DIM - 1); ++row)
                for (int col = max(bar.col0, 0); col <= min(bar.col1, MAX_GRID_DIM - 1); ++col)
                    if (bars.mask[row][col] == TRANSPARENT) {
                        bars.mask[row][col] = color;
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
// (8b_op_math.hpp), the table inference replays, so the two cannot disagree.
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
                    one.mask[row][col] = traded_with_empty(one.mask[row][col], value - EMPTY_PAIRS);
        }
        return res;
    }
    int8_t to[COLOR_COUNT];
    if (!swaps_of(value, partner_view{ from }, to)) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        // Only what was picked swaps colors; the rest ride through as they are.
        if (!one.selected) continue;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col) {
                const int8_t color = one.mask[row][col];
                if (is_color(color) && to[color] >= 0) one.mask[row][col] = to[color];
            }
    }
    return res;
}

// ColorFrom[9], past the partner ways: each picked object takes the color a key
// drawn in the grid -- the objects not picked -- shows at its place, the n-th
// picked in reading order the key's n-th entry (key_place_colors, 8b_op_math.hpp,
// the walk inference replays).
void paint_key_places(const object_list& from, object_list& res) {
    int8_t color[MAX_OBJECTS_IN_A_LIST], band[MAX_OBJECTS_IN_A_LIST];
    int16_t lows[MAX_OBJECTS_IN_A_LIST], highs[MAX_OBJECTS_IN_A_LIST];
    int order[MAX_OBJECTS_IN_A_LIST];
    if (!key_place_colors(partner_view{ from }, color, lows, highs, band, order)) return;
    for (int i = 0; i < res.count; ++i) {
        if (color[i] < 0) continue;
        held_object& one = res.objects[i];
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] != TRANSPARENT) one.mask[row][col] = color[i];
    }
}

// ColorFrom[how]: each picked object takes the color of its partner among the
// objects not picked -- the nearest, one in its rows or columns, the one around
// it or inside it, one it touches, one of its shape as it lies or turned -- and
// paints every cell it paints in it. The partner is found by partner_color
// (8b_op_math.hpp), the walk inference replays, so the two cannot disagree. An
// object with no partner, or partners that disagree, keeps its colors. Nothing
// moves and nothing is dropped.
object_list& op_color_from(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (how == KEY_BY_PLACE) {
        paint_key_places(from, res);
        return res;
    }
    const partner_view view{ from };
    // Cell by cell, from the one piece of its shape (colors_from_its_shape).
    if (how >= COLORS_FROM_ITS_SHAPE) {
        for (int i = 0; i < from.count; ++i) {
            if (!from.objects[i].selected) continue;
            held_object& one = res.objects[i];
            colors_from_its_shape(how - COLORS_FROM_ITS_SHAPE, i, view, [&](int row, int col, int color) {
                const int down = row - one.y0, across = col - one.x0;
                if (down >= 0 && across >= 0 && down < one.height && across < one.width && is_color(color))
                    one.mask[down][across] = (int8_t)color;
            });
        }
        return res;
    }
    if (how < 0 || how >= (int8_t)partner_by::count) return res;
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int color = view.color_from((partner_by)how, i);
        if (color < 0) continue;
        held_object& one = res.objects[i];
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] != TRANSPARENT) one.mask[row][col] = (int8_t)color;
    }
    return res;
}

// WouldFit[how]: of the picked objects, only those that would fit into a gap of
// their one partner among the objects not picked stay picked -- the partner by
// `how`, as partner_of (8b_op_math.hpp) finds it, and the fit as would_fit asks
// it, turned or mirrored any way: the walk inference replays, so the two cannot
// disagree. Every partner is read off the marks as they came in. Nothing moves
// and nothing is painted; an object with no partner, or several, is not picked.
object_list& op_would_fit(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (how < 0 || how >= (int8_t)partner_by::count) return res;
    const partner_view view{ from };
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int partner = view.partner((partner_by)how, i);
        res.objects[i].selected = partner >= 0 && would_fit(i, partner, view);
    }
    return res;
}

// SelectLikeKey[how]: of the picked objects, only those like a piece of the key
// -- every object not picked -- stay picked (like_the_key, 8b_op_math.hpp), the
// question inference replays. Every likeness is read off the marks as they came in.
object_list& op_select_like_key(program_run& run, const object_list& from, int8_t how) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const partner_view view{ from };
    for (int i = 0; i < from.count; ++i)
        if (from.objects[i].selected) res.objects[i].selected = like_the_key(how, i, view);
    return res;
}

// SelectPartner[value]: the pick moves onto every partner each picked object
// has, among every object, by the way and on the side `value` names
// (each_partner_picked, 8b_op_math.hpp, which inference replays). Every
// partner is read off the marks as they came in; an object no picked one has
// for a partner comes out unpicked.
object_list& op_select_partner(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = false;
    each_partner_picked(value, partner_view{ from }, [&](int other) { res.objects[other].selected = true; });
    return res;
}

// SelectPartition[which]: the picked objects are walls, as Partition reads them,
// and the pick becomes every object lying in the part `which` they fence off
// (parts_the_pick_fences and lies_in_part, 8b_op_math.hpp, which inference
// replays), the parts numbered in reading order. The walls lie in no part, so
// they come out unpicked. The canvas is the input's, as Partition's is.
object_list& op_select_partition(program_run& run, const object_list& from, int which) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const partner_view view{ from };
    int16_t part[MAX_GRID_DIM][MAX_GRID_DIM];
    parts_the_pick_fences(view, run.input.width, run.input.height, part);
    for (int i = 0; i < from.count; ++i) res.objects[i].selected = lies_in_part(view, i, which, part);
    return res;
}

// Partition: the selected objects are the walls. Strict, like every act: no
// selection means no walls, so the whole thing stays one part -- Partition
// NEVER falls back to "all painted are walls". The walls come back first,
// unmarked, so SelectPartition can still read them; then the parts they fence
// off, whatever their shape, part after part in reading order
// (each_partition_piece, 8b_op_math.hpp, the parts SelectPartition picks by):
// every other object clipped to the part, in the part's box -- severed where it
// crosses into another, which is what partitioning means -- and the part's
// empty cells as one more object, a room, BLANK until a recolor paints it.
// Walls across the whole grid fence the tiles of a lattice, row by row. No
// merge: distinct source layers stay distinct; want parts as single masks?
// place a UnionList yourself.
object_list& op_partition(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int grid_width = run.input.width, grid_height = run.input.height;
    if (grid_width <= 0 || grid_height <= 0) return res;
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i)
        if (from.objects[i].selected) {
            res.objects[res.count] = from.objects[i];
            res.objects[res.count++].selected = false;
        }
    each_partition_piece(partner_view{ from }, grid_width, grid_height,
        [&](int i, const cell_box& box, auto in_part) {
            held_object one;
            if (res.count < MAX_OBJECTS_IN_A_LIST &&
                clip_to(from.objects[i], box.row0, box.col0, box.row1 + 1, box.col1 + 1, in_part, one))
                res.objects[res.count++] = one;
        },
        [&](int, const cell_box& box, auto empty) {
            if (res.count >= MAX_OBJECTS_IN_A_LIST) return;
            held_object& room = res.objects[res.count];
            room.x0 = (int16_t)box.col0; room.y0 = (int16_t)box.row0;
            room.x1 = (int16_t)box.col1; room.y1 = (int16_t)box.row1;
            room.width = (int16_t)box.width(); room.height = (int16_t)box.height();
            room.selected = false;
            room.mask.clear();
            bool any = false;
            for (int row = box.row0; row <= box.row1; ++row)
                for (int col = box.col0; col <= box.col1; ++col)
                    if (empty(row, col)) {
                        room.mask[row - box.row0][col - box.col0] = (int8_t)BLANK;
                        any = true;
                    }
            if (any) ++res.count;
        });
    return res;
}

// --- Translate ---

// AlignTo[value]: each picked object slides, down or across alone, until its
// painted box starts on the row its one partner's does -- or, for values past
// the first partner_by::count, the column. The partner is found among the
// objects not picked by partner_of (8b_op_math.hpp), the walk inference
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
    const partner_view view{ from };
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int partner = view.partner(how, i);
        if (partner < 0) continue;
        const cell_box own = view.box(i), there = view.box(partner);
        held_object& one = res.objects[i];
        if (columns) {
            const int16_t by = (int16_t)(there.col0 - own.col0);
            one.x0 += by; one.x1 += by;
        } else {
            const int16_t by = (int16_t)(there.row0 - own.row0);
            one.y0 += by; one.y1 += by;
        }
    }
    return res;
}

// MoveToward[value]: each picked object moves toward its one partner among the
// objects not picked -- of the kind value % PARTNER_KINDS names, each way
// partner_by finds one and then each color (partner_of_kind) -- by what value /
// PARTNER_KINDS names: one step, its own size, or until it touches the partner
// (move_toward, 8b_op_math.hpp), the move inference replays. Partners are never
// picked, so none of them moves while the others read them. An object with no
// partner, or several, stays where it is.
object_list& op_move_toward(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (value < 0 || value >= (int)toward_by::count * PARTNER_KINDS) return res;
    const auto far = (toward_by)(value / PARTNER_KINDS);
    const int kind = value % PARTNER_KINDS;
    const partner_view view{ from };
    for (int i = 0; i < from.count; ++i) {
        if (!from.objects[i].selected) continue;
        const int partner = view.partner_of_kind(kind, i);
        if (partner < 0) continue;
        const direction_step by = move_toward(far, i, partner, view, run.canvas.width, run.canvas.height);
        held_object& one = res.objects[i];
        one.x0 += (int16_t)by.across; one.x1 += (int16_t)by.across;
        one.y0 += (int16_t)by.down;   one.y1 += (int16_t)by.down;
    }
    return res;
}

// --- Generate ---

// MirrorAround[value]: each picked object is drawn again mirrored around the
// middle of its one partner among the objects not picked -- left to right, top
// to bottom, or all four ways at once, value / MIRROR_PARTNERS -- or, for the
// last partner, around the middle of the grid itself. partner_of (8b_op_math.hpp)
// finds the partner, the walk inference replays. The picked object stays; its
// copies join the list after every object there was, picked as it is, as
// CopyTo's are -- a sentence that picked every piece paints what it holds
// without picking again. An object with no partner, or several, gets no copy.
object_list& op_mirror_around(program_run& run, const object_list& from, int8_t value) {
    object_list& res = run.fresh();
    copy_list(res, from);
    constexpr int partners = MIRROR_PARTNERS;
    if (value < 0 || value >= (int)mirror_by::count * partners) return res;
    const auto mirror = (mirror_by)(value / partners);
    const int how = value % partners;
    const partner_view view{ from };
    for (int i = 0; i < from.count; ++i) {
        const held_object& piece = from.objects[i];
        if (!piece.selected || piece.width <= 0 || piece.height <= 0) continue;
        // The middle, the copies and where each lies are the reading's
        // arithmetic (mirror_middle, each_mirror, mirrored_box).
        int middle_row = 0, middle_col = 0;
        if (!mirror_middle(how, i, view, run.input.width, run.input.height, middle_row, middle_col)) continue;
        const cell_box box{ piece.y0, piece.x0, piece.y0 + piece.height - 1, piece.x0 + piece.width - 1 };
        bool full = false;
        each_mirror(mirror, [&](bool down, bool across) {
            if (full || res.count >= MAX_OBJECTS_IN_A_LIST) { full = true; return; }
            held_object& copy = res.objects[res.count++];
            copy = piece;
            const int pose = mirror_pose(down, across);
            for (int row = 0; row < piece.height; ++row)
                for (int col = 0; col < piece.width; ++col) {
                    const auto [from_row, from_col] = pose_source(pose, piece.width, piece.height, row, col);
                    copy.mask[row][col] = (int8_t)piece.mask[from_row][from_col];
                }
            const cell_box to = mirrored_box(box, middle_row, middle_col, down, across);
            copy.y0 = (int16_t)to.row0; copy.x0 = (int16_t)to.col0;
            copy.y1 = (int16_t)to.row1; copy.x1 = (int16_t)to.col1;
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
        const int found = object_hole_count(min((int)one.height, MAX_GRID_DIM), min((int)one.width, MAX_GRID_DIM),
            [&](int row, int col) { return one.mask[row][col] != TRANSPARENT; });
        if (found == (int)holes) one.selected = true;
    }
    return res;
}

// SelectAtLeastCells[n]: mark the objects that paint at least n cells
// (paints_at_least, which the reading replays); the rest ride through
// unmarked. InvertSelection after it says "fewer than n".
object_list& op_select_at_least_cells(program_run& run, const object_list& from, int8_t cells_wanted) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        if (paints_at_least(min((int)one.height, MAX_GRID_DIM), min((int)one.width, MAX_GRID_DIM),
                            [&](int row, int col) { return one.mask[row][col] != TRANSPARENT; }, cells_wanted))
            one.selected = true;
    }
    return res;
}

// SelectColor[c]: mark every object that contains color c; the rest ride
// through unmarked.
object_list& op_select_color(program_run& run, const object_list& from, int8_t color) {
    object_list& res = run.fresh();
    copy_list(res, from);
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        for (int row = 0; row < one.height && !one.selected; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] == color) { one.selected = true; break; }
    }
    return res;
}

// Elect from the current layout, keeping the objects' frames (not their paint
// bounds). Like SelectColor, selection is additive and marks whole objects.
object_list& op_select_elected_background(program_run& run, const object_list& from) {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool has_frame = false;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        if (!has_frame) {
            x0 = x1 = one.x0; y0 = y1 = one.y0;
            has_frame = true;
        }
        x0 = min(x0, (int)one.x0); y0 = min(y0, (int)one.y0);
        x1 = max(x1, one.x0 + one.width); y1 = max(y1, one.y0 + one.height);
    }
    grid laid;
    laid.width = max(0, min(MAX_GRID_DIM, x1 - x0));
    laid.height = max(0, min(MAX_GRID_DIM, y1 - y0));
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        each_shown(one, [&](int row, int col, int8_t color) {
            const int at_row = one.y0 + row - y0, at_col = one.x0 + col - x0;
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
// key, shape_key_of_piece), as the reading looks it up
// (shape_table::color_of): -1 where it names none. The one lookup both shape
// ops make.
int shape_color(const program_run& run, const held_object& one, int kind) {
    return run.shape_colors.color_of(shape_key_of_piece(one, kind));
}

// ShapeColorMap[kind]: every picked piece takes the color its own key names in
// the puzzle-level table the reading learned -- its SHAPE, the same turned either
// way or mirrored either way, or its SYMMETRY, the ways round that leave its
// drawing as it was (shape_key_of, the key the reading learned) -- or the color
// the table gives every other key. A key the table names no color for leaves
// its piece as it was; nothing else changes. A piece colored by ANOTHER's shape
// is said in words (009d5c81's key naming the figure's color): the key takes
// its shape's color, the figure the key's (ColorFrom), and the key goes
// (DeleteSelection).
object_list& op_shape_color_map(program_run& run, const object_list& from, int kind) {
    object_list& res = run.fresh();
    copy_list(res, from);
    if (!run.shape_colors.holds()) return res;
    for (int i = 0; i < res.count; ++i) {
        held_object& one = res.objects[i];
        if (!one.selected) continue;
        const int color = shape_color(run, one, kind);
        if (color < 0) continue;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                if (one.mask[row][col] != TRANSPARENT) one.mask[row][col] = (int8_t)color;
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

// OverlapCorner: lay the objects out as a diagonal corner-chain. Objects are
// ordered left-to-right (by box x0, ties by y0); the first is placed with its
// top-left at the canvas origin (0,0); each later object's top-left cell is
// set to the previous object's bottom-right cell -- a 1-pixel corner overlap.
// Only positions change (masks/colors untouched); the trailing Union resolves
// the shared corner -- the later (further-down-chain) object wins, as Union
// lays a later object over an earlier.
//
// GENERAL FORM (not built; promote when a puzzle needs it). This is the
// sequential / corner / overlap generalization of AlignToCanvas, which is the
// batch / edge-flush / gap-0 corner of the same space:
//     placement: batch (fixed canvas)   ->  sequential (running canvas)
//     anchor:    edge-flush             ->  corner-to-corner
//     offset:    gap = 0                ->  stride = 1 (overlap)
// Two future annotations, both output-determined:
//   * direction: the 4 diagonals {DR,DL,UR,UL}. Fixed DR (down-right) here --
//     the only verified case. Add as [dir]; do NOT try to reach the other three
//     via Flip*/Turn* composition, which needs an un-flip pass to undo the mirror.
//   * stride: the per-step diagonal offset (overlap 1 today; 0 = corner-touch,
//     >0 = gap). Inseparable from this op -- it's the cascade's step, not a
//     downstream op, because it compounds with chain index (object i shifts i*).
object_list& op_overlap_corner(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    if (count <= 0) { copy_list(res, from); return res; }
    // Chain order = box (x0, y0) ascending. Selection sort.
    int order[MAX_OBJECTS_IN_A_LIST];
    for (int i = 0; i < count; ++i) order[i] = i;
    for (int i = 0; i < count; ++i) {
        int best = i;
        for (int j = i + 1; j < count; ++j) {
            const held_object& one = from.objects[order[j]];
            const held_object& other = from.objects[order[best]];
            if (one.x0 < other.x0 || (one.x0 == other.x0 && one.y0 < other.y0)) best = j;
        }
        swap(order[i], order[best]);
    }
    // The first top-left at (0,0); each next top-left = the previous bottom-right.
    int at_x = 0, at_y = 0;
    for (int k = 0; k < count; ++k) {
        const held_object& source = from.objects[order[k]];
        held_object& one = res.objects[res.count++] = source;
        one.x0 = (int16_t)at_x;                       one.y0 = (int16_t)at_y;
        one.x1 = (int16_t)(at_x + source.width - 1);  one.y1 = (int16_t)(at_y + source.height - 1);
        at_x = one.x1; at_y = one.y1;
    }
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
        const auto [down, across] = shift_to_touch({ one.y0, one.x0, one.y1, one.x1 }, target);
        one.x0 = (int16_t)(one.x0 + across);
        one.y0 = (int16_t)(one.y0 + down);
        one.x1 = (int16_t)(one.x1 + across);
        one.y1 = (int16_t)(one.y1 + down);
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
        one.x1 = (int16_t)(one.width - 1);
        one.y1 = (int16_t)(one.height - 1);
        one.x0 = 0; one.y0 = 0;
    }
    return res;
}

// --- Reduce, folding a list ---

// Intersect the objects whose `selected` equals `want` into ONE object (a cell
// survives only where every such object is painted, on the shared smallest box
// at the origin). Appends it to res carrying the shared flag if anything
// survives. The first of the group gives the color.
void intersect_group(const object_list& from, bool want, object_list& res) {
    int members = 0, first = -1, least_width = 0, least_height = 0;
    for (int i = 0; i < from.count; ++i) {
        if (from.objects[i].selected != want) continue;
        if (first < 0) {
            first = i;
            least_width = from.objects[i].width;
            least_height = from.objects[i].height;
        } else {
            least_width = min(least_width, (int)from.objects[i].width);
            least_height = min(least_height, (int)from.objects[i].height);
        }
        ++members;
    }
    if (members == 0 || least_width <= 0 || least_height <= 0 || res.count >= MAX_OBJECTS_IN_A_LIST) return;
    held_object one;
    one.width = (int16_t)least_width; one.height = (int16_t)least_height;
    one.x1 = (int16_t)(least_width - 1); one.y1 = (int16_t)(least_height - 1);
    one.selected = want;
    int kept = 0;
    for (int row = 0; row < least_height; ++row)
        for (int col = 0; col < least_width; ++col) {
            bool every = true;
            for (int i = 0; i < from.count && every; ++i)
                every = from.objects[i].selected != want || from.objects[i].mask[row][col] != TRANSPARENT;
            if (every) { one.mask[row][col] = from.objects[first].mask[row][col]; ++kept; }
        }
    if (kept == 0) return;
    res.objects[res.count++] = one;
}

// IntersectionList: intersect to one object PER selection group — the selected
// objects AND into one selected object, the unselected into one unselected
// object. The mark survives the fold as object identity, never dropped. No
// selection => one unselected object (plain "intersect all").
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
        x0 = min(x0, (int)one.x0); y0 = min(y0, (int)one.y0);
        x1 = max(x1, (int)one.x1); y1 = max(y1, (int)one.y1);
    }
    if (x1 < 0 || y1 < 0 || res.count >= MAX_OBJECTS_IN_A_LIST) return;
    held_object& merged = res.objects[res.count];
    merged.x0 = (int16_t)x0; merged.y0 = (int16_t)y0;
    merged.x1 = (int16_t)x1; merged.y1 = (int16_t)y1;
    merged.width = (int16_t)(x1 - x0 + 1); merged.height = (int16_t)(y1 - y0 + 1);
    merged.selected = want;
    merged.mask.clear();
    bool any = false;
    for (int i = 0; i < from.count; ++i) {
        const held_object& one = from.objects[i];
        if (one.selected != want) continue;
        each_painted(one, [&](int row, int col, int8_t color) {
            merged.mask[one.y0 + row - y0][one.x0 + col - x0] = color;
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

// Bands for Layout: objects whose spans overlap meaningfully share a band
// (overlapping_bands). `rows` reads vertical spans (row bands), else
// horizontal (column bands). band_of[i] = the object's band rank.
int layout_bands(const object_list& from, int count, bool rows, int8_t* band_of) {
    int16_t lows[MAX_OBJECTS_IN_A_LIST], highs[MAX_OBJECTS_IN_A_LIST];
    int order[MAX_OBJECTS_IN_A_LIST];
    for (int i = 0; i < count; ++i) {
        lows[i] = rows ? from.objects[i].y0 : from.objects[i].x0;
        highs[i] = rows ? from.objects[i].y1 : from.objects[i].x1;
    }
    return overlapping_bands(count, lows, highs, band_of, order);
}

// WHICH BAND A PIECE SITS IN, not what the piece is. Every other selector names
// something the piece carries -- its color, its holes, its size -- so none of
// them can say "the middle row of them". This one names where a piece sits among
// the others, which is a question about the arrangement and not about the piece.
// Bands come from layout_bands, the same grouping Layout reads, so a row
// means one thing wherever it is asked about. Selectors ADD, so a band past the
// last one marks nothing rather than clearing what is already picked.
object_list& op_select_band(program_run& run, const object_list& from, int8_t which, bool rows) {
    object_list& res = run.fresh();
    copy_list(res, from);
    const int count = min(res.count, MAX_OBJECTS_IN_A_LIST);
    if (count <= 0) return res;
    int8_t band_of[MAX_OBJECTS_IN_A_LIST];
    layout_bands(res, count, rows, band_of);
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
            const int line = rows ? one.y0 + row : one.x0 + col;
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
                on = one.mask[row][col] != TRANSPARENT && (rows ? one.y0 + row : one.x0 + col) == line;
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
// wide as the widest drawing and as tall as the tallest.
object_list& op_layout(program_run& run, const object_list& from, int cell) {
    object_list& res = run.fresh();
    const int count = min(from.count, MAX_OBJECTS_IN_A_LIST);
    if (count <= 0 || cell < 0) return res;
    // Where each drawing starts inside its frame, and the cell that holds them.
    int8_t top_of[MAX_OBJECTS_IN_A_LIST], left_of[MAX_OBJECTS_IN_A_LIST];
    int cell_width = cell, cell_height = cell;
    for (int i = 0; i < count; ++i) {
        const cell_box drawing = painted_box(from.objects[i]);
        top_of[i] = (int8_t)(drawing.empty() ? 0 : drawing.row0);
        left_of[i] = (int8_t)(drawing.empty() ? 0 : drawing.col0);
        if (cell == 0) {
            cell_width = max(cell_width, drawing.width());
            cell_height = max(cell_height, drawing.height());
        }
    }
    if (cell_width <= 0 || cell_height <= 0) return res;
    int8_t row_band[MAX_OBJECTS_IN_A_LIST], column_band[MAX_OBJECTS_IN_A_LIST];
    const int rows = layout_bands(from, count, true, row_band);
    const int columns = layout_bands(from, count, false, column_band);
    const int width = min(columns * cell_width, MAX_GRID_DIM);
    const int height = min(rows * cell_height, MAX_GRID_DIM);
    held_object& laid = res.objects[0];
    laid.x0 = 0; laid.y0 = 0;
    laid.x1 = (int16_t)(width - 1); laid.y1 = (int16_t)(height - 1);
    laid.width = (int16_t)width; laid.height = (int16_t)height;
    laid.mask.clear();
    for (int i = 0; i < count; ++i) {
        const held_object& source = from.objects[i];
        const int at_col = column_band[i] * cell_width, at_row = row_band[i] * cell_height;
        const int top = top_of[i], left = left_of[i];
        // Shift/Union can grow the box beyond what the mask holds.
        const int source_height = min((int)source.height, MAX_GRID_DIM), source_width = min((int)source.width, MAX_GRID_DIM);
        for (int row = 0; row < cell_height && top + row < source_height; ++row)
            for (int col = 0; col < cell_width && left + col < source_width; ++col) {
                const int to_row = at_row + row, to_col = at_col + col;
                const int8_t color = source.mask[top + row][left + col];
                if (to_row < height && to_col < width && color != TRANSPARENT) laid.mask[to_row][to_col] = color;
            }
    }
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
// names (shift_by, 8b_op_math.hpp). No selection means Shift moves nothing.
object_list& op_shift(program_run& run, const object_list& from, int8_t move_direction) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        if (!one.selected) continue;
        const direction_step by = shift_by(move_direction, one.width, one.height);
        one.x0 += (int16_t)by.across; one.y0 += (int16_t)by.down;
        one.x1 += (int16_t)by.across; one.y1 += (int16_t)by.down;
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
            return (int8_t)source.mask[from_row][from_col];
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
        if (across) { one.x0 += delta; one.x1 += delta; }
        else { one.y0 += delta; one.y1 += delta; }
    }
    return res;
}

// SlamRow / SlamColumn: each picked object's cells slide a step along its rows
// or columns and pile at the wall (slam_cells, 8b_op_math.hpp).
object_list& op_slam(program_run& run, const object_list& from, bool rows, int16_t step) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        held_object& one = res.objects[res.count++] = from.objects[i];
        if (!one.selected || step == 0) continue;
        // Shift/Union can grow the box beyond what the mask holds.
        const int width = min((int)one.width, MAX_GRID_DIM);
        const int height = min((int)one.height, MAX_GRID_DIM);
        cells drawn;
        slam_cells(width, height, rows, step,
            [&](int row, int col) { return (int)one.mask[row][col]; },
            [&](int row, int col, int color) { drawn[row][col] = (int8_t)color; });
        one.mask = drawn;
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
            return from_row < source.height && from_col < source.width ? (int8_t)source.mask[from_row][from_col]
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
            return majority_vote(source.mask, row, col, factor, source.height, source.width);
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
                if (one.mask[row][col] == TRANSPARENT) one.mask[row][col] = fill;
    }
    return res;
}

// Outline: keep each picked object's edge cells, drop the core to transparent.
// A cell is edge when any of its 8 neighbours is not part of the object
// (on_object_edge; outside the frame counts as outside). Reads the source
// mask, writes the copy, so the erosion never sees its own output.
object_list& op_outline(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        held_object& one = res.objects[res.count++] = source;
        if (!one.selected) continue;
        for (int row = 0; row < source.height; ++row)
            for (int col = 0; col < source.width; ++col) {
                if (source.mask[row][col] == TRANSPARENT) continue;
                const bool edge = on_object_edge(row, col, [&](int at_row, int at_col) {
                    return at_row >= 0 && at_col >= 0 && at_row < source.height && at_col < source.width &&
                           source.mask[at_row][at_col] != TRANSPARENT;
                });
                if (!edge) one.mask[row][col] = TRANSPARENT;
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
        one.x0 = (int16_t)target.col0; one.y0 = (int16_t)target.row0;
        one.width = (int16_t)width; one.height = (int16_t)height;
        one.x1 = (int16_t)target.col1; one.y1 = (int16_t)target.row1;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                one.mask[row][col] = source.mask[fit_source_cell(row, source.height, height)]
                                                [fit_source_cell(col, source.width, width)];
        res.objects[res.count++] = one;
    }
    return res;
}

// Skew slope table: variant -> {num, den}; the line slope is num/den (Δcol per
// Δrow), applied as a Bresenham shift floor((y0+r)*num/den). Stern-Brocot /
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
        const int top = one.y0;
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
        one.mask = leaned;
        one.width = (int16_t)width; one.height = (int16_t)height;
        one.x0 = (int16_t)(one.x0 + least_col); one.y0 = (int16_t)(top + least_row);
        one.x1 = (int16_t)(one.x0 + width - 1); one.y1 = (int16_t)(one.y0 + height - 1);
        res.count++;
    }
    return res;
}

// Lets repeat_periods read an object: it wants width, height and data[r][c],
// and a mask already indexes that way. No copy, no second period rule.
struct object_view {
    const cells& data;
    int width, height;
};

// Pattern: each picked object reduced to its smallest repeating unit
// (repeat_periods, 4_objects.hpp, one finder for everyone).
object_list& op_pattern(program_run& run, const object_list& from) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        if (source.width <= 0 || source.height <= 0) continue;
        if (!source.selected) { res.objects[res.count++] = source; continue; }
        int unit_width = 0, unit_height = 0;
        repeat_periods(object_view{ source.mask, source.width, source.height }, unit_width, unit_height);
        held_object unit;
        // The unit is the same object, so it keeps its mark.
        unit.selected = source.selected;
        unit.x0 = source.x0; unit.y0 = source.y0;
        unit.width = (int16_t)unit_width; unit.height = (int16_t)unit_height;
        unit.x1 = (int16_t)(unit.x0 + unit_width - 1); unit.y1 = (int16_t)(unit.y0 + unit_height - 1);
        // Every copy agrees with the top-left one, so the unit is that copy.
        for (int row = 0; row < unit_height; ++row)
            for (int col = 0; col < unit_width; ++col) unit.mask[row][col] = source.mask[row][col];
        res.objects[res.count++] = unit;
    }
    return res;
}

// VoteDenoise: noise-tolerant majority vote across repeating structure. The
// arithmetic is vote_denoise_cells (8b_op_math.hpp), shared with inference.
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
        const int8_t ground = denoise_ground(width, height, [&](int row, int col) { return (int8_t)source.mask[row][col]; });
        // A cell emptied by a background removal upstream is still the ground:
        // read it as the ground, which is itself empty when the ground was deleted.
        const auto ground_at = [&](int row, int col) -> int8_t {
            const int8_t color = source.mask[row][col];
            return color == TRANSPARENT ? ground : color;
        };
        vote_denoise_cells(width, height, ground, ground_at,
                           [&](int row, int col, int8_t color) { one.mask[row][col] = color; });
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
        const cell_box paint = painted_box(source);
        if (paint.empty()) continue;
        held_object one;
        // A rebuilt object is the SAME object, so it keeps its mark.
        one.selected = source.selected;
        one.x0 = (int16_t)(source.x0 + paint.col0); one.y0 = (int16_t)(source.y0 + paint.row0);
        one.width = (int16_t)paint.width(); one.height = (int16_t)paint.height();
        one.x1 = (int16_t)(one.x0 + one.width - 1); one.y1 = (int16_t)(one.y0 + one.height - 1);
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col)
                one.mask[row][col] = source.mask[paint.row0 + row][paint.col0 + col];
        res.objects[res.count++] = one;
    }
    return res;
}

// Tile [steps, start, end, spread]: each picked piece laid again where its
// Tile says (each_tile_offset, 8b_op_math.hpp, the arithmetic the readings
// replay), each step measured on this hand: the piece's own size and gaps,
// and how many colors the picked pieces hold together (measures_of). A COPY
// IS A PIECE, keeping the piece's mark, so a row of copies can be picked and
// turned; want one sheet? Say UnionList after. A copy is one step of the
// piece where a step is shorter (copy_size). A piece no step moves, and every
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
    for (int i = 0; i < count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& source = from.objects[i];
        const tile_moves moves = !source.selected ? tile_moves{}
            : moves_of(tile, measures_of(tile, source.height, source.width,
                                         [&](int row, int col) { return source.mask[row][col] != TRANSPARENT; }, colors));
        if (moves.count == 0 || source.width <= 0 || source.height <= 0) {
            res.objects[res.count++] = source;
            continue;
        }
        const auto [rows, columns] = copy_size(moves, source.height, source.width);
        each_tile_offset(tile, moves, source.y0, source.x0, rows, columns, run.canvas.width, run.canvas.height,
            [&](int down, int across) {
                if (res.count >= MAX_OBJECTS_IN_A_LIST) return false;
                held_object& copy = res.objects[res.count++] = source;
                copy.y0 = (int16_t)(source.y0 + down);
                copy.x0 = (int16_t)(source.x0 + across);
                copy.height = (int16_t)rows;
                copy.width = (int16_t)columns;
                copy.y1 = (int16_t)(copy.y0 + rows - 1);
                copy.x1 = (int16_t)(copy.x0 + columns - 1);
                for (int at = 0; at < MAX_GRID_CELLS; ++at)
                    if (cell_row(at) >= rows || cell_col(at) >= columns) copy.mask.values[at] = TRANSPARENT;
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
        stamped.x0 = 0; stamped.y0 = 0;
        stamped.x1 = (int16_t)(width - 1); stamped.y1 = (int16_t)(height - 1);
        stamped.width = (int16_t)width; stamped.height = (int16_t)height;
        stamped.mask.clear();
        bool any = false;
        for (int row = 0; row < height; ++row) {
            const int big_row = row * source.height / height, own_row = row % source.height;
            for (int col = 0; col < width; ++col) {
                const int big_col = col * source.width / width, own_col = col % source.width;
                if (source.mask[big_row][big_col] == TRANSPARENT) continue;
                const int8_t color = source.mask[own_row][own_col];
                if (color != TRANSPARENT) { stamped.mask[row][col] = color; any = true; }
            }
        }
        if (any) ++res.count;
    }
    return res;
}

// For each picked object: find the cell nearest its middle (nearest_ray_seed),
// draw rays for `radius` steps in the four directions from `diagonal`'s set
// (each_extended_ray) in that cell's color. The rays only, one object per input.
object_list& op_extend_rays(program_run& run, const object_list& from, int8_t radius, bool diagonal) {
    object_list& res = run.fresh();
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        if (!one.selected) { res.objects[res.count++] = one; continue; }
        const auto seed = nearest_ray_seed(one.height, one.width,
            (one.y0 + one.y1) / 2 - one.y0, (one.x0 + one.x1) / 2 - one.x0,
            [&](int row, int col) { return one.mask[row][col]; });
        if (seed.row < 0) continue;
        const int8_t seed_color = one.mask[seed.row][seed.col];
        // The object's box grown by radius on every side, clipped to the canvas:
        // a ray that runs past the edge would never be painted there anyway, and
        // a frame no wider than the canvas holds every ray that shows -- one
        // anchored off the object's corner cut the long ones short on its far
        // sides.
        const int x0 = max(one.x0 - radius, 0), y0 = max(one.y0 - radius, 0);
        const int x1 = min(one.x1 + radius, MAX_GRID_DIM - 1), y1 = min(one.y1 + radius, MAX_GRID_DIM - 1);
        if (x1 < x0 || y1 < y0) continue;
        held_object& rays = res.objects[res.count++];
        // The rays belong to the object that grew them, so they carry its mark.
        rays.selected = one.selected;
        rays.x0 = (int16_t)x0; rays.y0 = (int16_t)y0;
        rays.x1 = (int16_t)x1; rays.y1 = (int16_t)y1;
        rays.width = (int16_t)(x1 - x0 + 1); rays.height = (int16_t)(y1 - y0 + 1);
        rays.mask.clear();
        // RAYS ONLY: the seed's own cells are not copied, so a caller that
        // wants them re-stamps them (a Union). A one-cell seed gives only its
        // arms, and radius 0 gives nothing.
        const int seed_row = one.y0 + seed.row - y0, seed_col = one.x0 + seed.col - x0;
        each_extended_ray(seed, radius, diagonal, [&](int down, int across) {
            const int row = seed_row + down, col = seed_col + across;
            if (row >= 0 && row < rays.height && col >= 0 && col < rays.width) rays.mask[row][col] = seed_color;
        });
    }
    return res;
}

// Whether the object paints the grid cell (x, y).
bool paints_at(const held_object& one, int x, int y) {
    const int across = x - one.x0, down = y - one.y0;
    return across >= 0 && across < one.width && down >= 0 && down < one.height && one.mask[down][across] != TRANSPARENT;
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
        each_painted(one, [&](int row, int col, int8_t) { painted.include_cell(one.y0 + row, one.x0 + col); });
        if (painted.empty()) continue;
        const int steps = steps_to_edge(painted, run.canvas.width, run.canvas.height, move_direction);
        one.x0 = (int16_t)(one.x0 + steps * step.across);
        one.x1 = (int16_t)(one.x1 + steps * step.across);
        one.y0 = (int16_t)(one.y0 + steps * step.down);
        one.y1 = (int16_t)(one.y1 + steps * step.down);
    }
    return res;
}

// ExtendTo PAST ITS EIGHT WAYS: every picked object grows in step with the
// others along its track, lines or the open cells, until they meet --
// grow_in_step, 8b_op_math.hpp, the growth the reading replays -- and hands
// back its body and every cell it took, in its first color, as a trail hands
// back its body and its trail.
object_list& op_grow_in_step(program_run& run, const object_list& from, grown_along track) {
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
            const int x = one.x0 + col, y = one.y0 + row;
            if (x >= 0 && x < width && y >= 0 && y < height && owner[y * width + x] < 0)
                owner[y * width + x] = (int16_t)i;
        });
    }
    grow_in_step(width, height, track, [&](int piece) { return from.objects[piece].selected; },
                 owner, turn, queue, cell_directions, [](int, int, int) {});
    for (int i = 0; i < count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        if (!one.selected) { res.objects[res.count++] = one; continue; }
        cell_box reach;
        each_painted(one, [&](int row, int col, int8_t) { reach.include_cell(one.y0 + row, one.x0 + col); });
        for (int at = 0; at < width * height; ++at)
            if (turn[at] > 0 && owner[at] == i) reach.include_cell(at / width, at % width);
        if (reach.empty()) continue;
        const int grown_width = min(reach.width(), MAX_GRID_DIM);
        const int grown_height = min(reach.height(), MAX_GRID_DIM);
        held_object& grown = res.objects[res.count];
        // The grown object is the SAME object, so it keeps the mark it was
        // picked by, as a trail's does.
        grown.selected = one.selected;
        grown.x0 = (int16_t)reach.col0; grown.y0 = (int16_t)reach.row0;
        grown.x1 = (int16_t)(reach.col0 + grown_width - 1); grown.y1 = (int16_t)(reach.row0 + grown_height - 1);
        grown.width = (int16_t)grown_width; grown.height = (int16_t)grown_height;
        grown.mask.clear();
        each_painted(one, [&](int row, int col, int8_t color) {
            const int across = one.x0 + col - reach.col0, down = one.y0 + row - reach.row0;
            if (across >= 0 && across < grown_width && down >= 0 && down < grown_height) grown.mask[down][across] = color;
        });
        const int8_t color = first_color(one);
        for (int at = 0; at < width * height; ++at) {
            if (turn[at] <= 0 || owner[at] != i) continue;
            const int across = at % width - reach.col0, down = at / width - reach.row0;
            if (across >= 0 && across < grown_width && down >= 0 && down < grown_height) grown.mask[down][across] = color;
        }
        ++res.count;
    }
    return res;
}

// ExtendTo[grow_direction]: every picked object extrudes its cells along one of 8
// directions, painting the swept path in the object's color. A ray stops just
// before the first FOREIGN occupied cell (a wall / another object) or the grid
// edge, and passes through the object's own body (march_trail, the walk the
// reading replays). Hands back each object's body and trail, which compose via
// Union. Past the eight ways, the objects grow in step (op_grow_in_step).
object_list& op_extend_to(program_run& run, const object_list& from, int8_t grow_direction) {
    if (grow_direction >= direction::count) return op_grow_in_step(run, from, (grown_along)(grow_direction - direction::count));
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
            const int x = one.x0 + col, y = one.y0 + row;
            if (x >= 0 && x < MAX_GRID_DIM && y >= 0 && y < MAX_GRID_DIM) occupied[y][x] = true;
        });
    }
    for (int i = 0; i < from.count && res.count < MAX_OBJECTS_IN_A_LIST; ++i) {
        const held_object& one = from.objects[i];
        if (one.width <= 0 || one.height <= 0) continue;
        // Only what was picked is touched; the rest ride through as they are.
        if (!one.selected) { res.objects[res.count++] = one; continue; }
        const int8_t color = first_color(one);
        const auto own = [&](int y, int x) { return paints_at(one, x, y); };
        const auto taken = [&](int y, int x) { return occupied[y][x]; };
        cell_box reach;
        const auto reached = [&](int y, int x) { reach.include_cell(y, x); };
        each_painted(one, [&](int row, int col, int8_t) {
            reached(one.y0 + row, one.x0 + col);
            march_trail(one.y0 + row, one.x0 + col, step, width, height, own, taken, reached);
        });
        if (reach.empty()) continue;
        const int grown_width = min(reach.width(), MAX_GRID_DIM);
        const int grown_height = min(reach.height(), MAX_GRID_DIM);
        held_object& grown = res.objects[res.count];
        // The grown object is the SAME object, longer, so the mark it was picked
        // by rides along with it.
        grown.selected = one.selected;
        grown.x0 = (int16_t)reach.col0; grown.y0 = (int16_t)reach.row0;
        grown.x1 = (int16_t)(reach.col0 + grown_width - 1); grown.y1 = (int16_t)(reach.row0 + grown_height - 1);
        grown.width = (int16_t)grown_width; grown.height = (int16_t)grown_height;
        grown.mask.clear();
        each_painted(one, [&](int row, int col, int8_t body) {
            const int at_row = one.y0 + row, at_col = one.x0 + col;
            const int down = at_row - reach.row0, across = at_col - reach.col0;
            // The body keeps its colors.
            if (across >= 0 && across < grown_width && down >= 0 && down < grown_height) grown.mask[down][across] = body;
            march_trail(at_row, at_col, step, width, height, own, taken, [&](int y, int x) {
                const int trail_across = x - reach.col0, trail_down = y - reach.row0;
                if (trail_across >= 0 && trail_across < grown_width && trail_down >= 0 && trail_down < grown_height)
                    grown.mask[trail_down][trail_across] = color;
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
                const int8_t color = one.mask[row][col];
                if (color == TRANSPARENT) continue;
                const int x = source.x0 + col, y = source.y0 + row;
                const int8_t under = x >= 0 && x < width && y >= 0 && y < height ? (int8_t)run.canvas.data[y][x]
                                                                                 : (int8_t)TRANSPARENT;
                const int8_t painted = paint(color, under);
                one.mask[row][col] = painted;
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
// c == 10 (mask_keeps). Off the canvas reads transparent, so MaskOut[10] keeps
// exactly the empty-canvas cells. Non-destructive: it filters the layer, never
// touches the canvas.
object_list& op_mask(program_run& run, const object_list& from, int8_t color, bool keep_match) {
    const bool any_color = color == COLOR_ANY;
    return repaint_over_canvas(run, from, [&](int8_t painted, int8_t under) {
        return mask_keeps(under, color, any_color, keep_match) ? painted : (int8_t)TRANSPARENT;
    });
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
        const auto [first_row, first_col] = placed(source.y0, source.x0);
        const auto [last_row, last_col] = placed(source.y0 + source.height - 1, source.x0 + source.width - 1);
        posed.y0 = (int16_t)min(first_row, last_row);
        posed.x0 = (int16_t)min(first_col, last_col);
        posed.width = (pose & 1) ? source.height : source.width;
        posed.height = (pose & 1) ? source.width : source.height;
        const int posed_height = min((int)posed.height, MAX_GRID_DIM), posed_width = min((int)posed.width, MAX_GRID_DIM);
        for (int row = 0; row < posed_height; ++row)
            for (int col = 0; col < posed_width; ++col) posed.mask[row][col] = TRANSPARENT;
        const int source_height = min((int)source.height, MAX_GRID_DIM), source_width = min((int)source.width, MAX_GRID_DIM);
        for (int row = 0; row < source_height; ++row)
            for (int col = 0; col < source_width; ++col) {
                const int8_t color = source.mask[row][col];
                if (color == TRANSPARENT) continue;
                auto [to_row, to_col] = placed(source.y0 + row, source.x0 + col);
                to_row -= posed.y0;
                to_col -= posed.x0;
                if (to_row >= 0 && to_row < MAX_GRID_DIM && to_col >= 0 && to_col < MAX_GRID_DIM)
                    posed.mask[to_row][to_col] = color;
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
                turned.mask.values[at] = row < height && col < width ? (int8_t)source.mask[col][row] : (int8_t)TRANSPARENT;
            }
            turned.x0 = source.y0; turned.y0 = source.x0;
            turned.width = (int16_t)width; turned.height = (int16_t)height;
            turned.x1 = (int16_t)(source.y0 + width - 1);
            turned.y1 = (int16_t)(source.x0 + height - 1);
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
            copy.x0 = (int16_t)aligned_copy_start(pattern.width, target.x0, target.width, anchor % 3);
            copy.y0 = (int16_t)aligned_copy_start(pattern.height, target.y0, target.height, anchor / 3);
            copy.x1 = (int16_t)(copy.x0 + copy.width - 1);
            copy.y1 = (int16_t)(copy.y0 + copy.height - 1);
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
    held_object& base = res.objects[res.count++];
    base.x0 = 0; base.y0 = 0;
    base.x1 = (int16_t)(run.input.width - 1);
    base.y1 = (int16_t)(run.input.height - 1);
    base.width = (int16_t)run.input.width;
    base.height = (int16_t)run.input.height;
    base.mask = run.input.data;
    const cell_box stencil_box{ stencil.y0, stencil.x0, stencil.y1, stencil.x1 };
    for (int s = 0; s < seeds.count; ++s) {
        const held_object& seed = seeds.objects[s];
        const int8_t seed_color = first_color(seed);
        const int8_t paint = seed_color >= 0 ? seed_color : first_color(stencil);
        project_along(stencil_box, { seed.y0, seed.x0, seed.y1, seed.x1 },
                      run.canvas.width, run.canvas.height, [&](int row, int col) {
            if (res.count >= MAX_OBJECTS_IN_A_LIST) return false;
            held_object& stamp = res.objects[res.count++];
            stamp.x0 = (int16_t)col; stamp.y0 = (int16_t)row;
            stamp.x1 = (int16_t)(col + stencil.width - 1);
            stamp.y1 = (int16_t)(row + stencil.height - 1);
            stamp.width = stencil.width; stamp.height = stencil.height;
            stamp.mask.clear();
            each_painted(stencil, [&](int at_row, int at_col, int8_t) { stamp.mask[at_row][at_col] = paint; });
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
            stamped.x0 = (int16_t)landed.col0; stamped.y0 = (int16_t)landed.row0;
            stamped.x1 = (int16_t)landed.col1; stamped.y1 = (int16_t)landed.row1;
            stamped.width = (int16_t)landed.width(); stamped.height = (int16_t)landed.height();
            stamped.mask.clear();
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    int at_row = 0, at_col = 0;
                    if (label[row][col] == body && landing(row, col, at_row, at_col))
                        stamped.mask[at_row - landed.row0][at_col - landed.col0] = drawn.data[row][col];
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
    case node_type::SelectAtLeastCells:      return op_select_at_least_cells(run, in, value);
    case node_type::SelectColor:             return op_select_color(run, in, value);
    case node_type::SelectElectedBackground: return op_select_elected_background(run, in);
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
    case node_type::Pattern:                 return op_pattern(run, in);
    case node_type::ProjectOnSeeds:          return op_project_on_seeds(run, in);
    case node_type::CopyTo:                  return op_copy_to(run, in, value);
    case node_type::IntersectionList:        return op_intersection_list(run, in);
    case node_type::UnionList:               return op_union_list(run, in);
    case node_type::Layout:                  return op_layout(run, in, value);
    case node_type::Origin:                  return op_origin(run, in);
    case node_type::AlignToCanvas:           return op_align_to_canvas(run, in);
    case node_type::OverlapCorner:           return op_overlap_corner(run, in);
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
    case node_type::SelectPartner:           return op_select_partner(run, in, value);
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
// carries marks itself, even when an empty target drops every picked object.
void carry_the_marks(node_type type, const object_list* last, object_list& made) {
    if (!last || !is_act_op(type) || type == node_type::Fit || clears_the_marks(type)) return;
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
    each_object(input, TRANSPARENT, grouping::touching_same_color, scratch,
        [&](const object_summary& box, const int16_t* taken, int taken_count) {
            if (pieces.count >= MAX_OBJECTS_IN_A_LIST) return;
            held_object& one = pieces.objects[pieces.count++];
            one.x0 = box.col; one.y0 = box.row;
            one.width = box.width; one.height = box.height;
            one.x1 = (int16_t)(one.x0 + one.width - 1);
            one.y1 = (int16_t)(one.y0 + one.height - 1);
            one.selected = false;
            one.mask.clear();
            for (int i = 0; i < taken_count; ++i) {
                const int row = cell_row(taken[i]), col = cell_col(taken[i]);
                one.mask[row - one.y0][col - one.x0] = input.data[row][col];
            }
        });
    if (ground >= 0) {
        int kept = 0;
        for (int i = 0; i < pieces.count; ++i)
            if (first_color(pieces.objects[i]) != ground) pieces.objects[kept++] = pieces.objects[i];
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

grid environment::draw(const node* tree, const example& ex, const puzzle& p, object_list* held) {
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
