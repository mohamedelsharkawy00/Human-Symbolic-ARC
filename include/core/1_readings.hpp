// The facts a person reads off a grid, used to infer the puzzle's rule from
// its examples, to be applied later to the test cases.
#pragma once
#include "0_core.hpp"
#include <cstdint>
#include <utility>

namespace arc {
using namespace std;

// COLORS · how many cells each color has

// How many cells wear each color, counted a cell at a time -- what is no color
// is not counted -- and the color the most of them wear, or the fewest, the
// lowest on a tie: TRANSPARENT where nothing was counted. The one tie rule for
// every op and every reading that asks which color leads.
struct color_tally {
    int counts[COLOR_COUNT] = {};

    constexpr color_tally() = default;

    // Read off a grid, leaving out `skip` (the background, wherever a caller
    // wants it ignored; TRANSPARENT counts all ten).
    explicit color_tally(const grid& one_grid, int skip = TRANSPARENT) {
        one_grid.for_each_colored_cell(skip, [&](int, int, int color) { add(color); });
    }

    int operator[](int color) const { return counts[color]; }

    void add(int color) { if (is_color(color)) ++counts[color]; }
    bool has(int color) const { return is_color(color) && counts[color] > 0; }
    int8_t most() const { return leading(true); }
    int8_t least() const { return leading(false); }

private:
    int8_t leading(bool most) const {
        int8_t best = TRANSPARENT;
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (counts[color] && (best < 0 || (most ? counts[color] > counts[best] : counts[color] < counts[best])))
                best = (int8_t)color;
        return best;
    }
};


// BOXES · a frame around a group of cells

// A box in grid coordinates, with inclusive bottom and right edges.
// An empty box has its last row or column before its first.
struct cell_box {
    int row0 = 0, col0 = 0, row1 = -1, col1 = -1;

    // Its own size, so no caller has to spell the +1 out and none can spell it
    // differently. A box nobody wrote to has no cells, and says so itself
    // rather than leaving the reader to check the tally beside it.
    bool empty() const { return row1 < row0 || col1 < col0; }
    int height() const { return empty() ? 0 : row1 - row0 + 1; }
    int width()  const { return empty() ? 0 : col1 - col0 + 1; }
    int area()   const { return width() * height(); }

    void include_cell(int row, int col) {
        if (empty()) {
            row0 = row1 = row;
            col0 = col1 = col;
            return;
        }
        if (row < row0) row0 = row;
        if (row > row1) row1 = row;
        if (col < col0) col0 = col;
        if (col > col1) col1 = col;
    }
    // Grown to hold another box as well; an empty one adds nothing.
    void include(const cell_box& other) {
        if (other.empty()) return;
        include_cell(other.row0, other.col0);
        include_cell(other.row1, other.col1);
    }
};

// OBJECTS · a group of cells read as one thing, and the walk that finds them

// An object is a related group of cells.
// Humans mostly decompose an ARC grid into objects when reading the grid,
// so the code will match that level of abstraction
// vs carrying each cell as its own entity.

// An object's box and cell count, usable without storing its cells.
// Packed small; box() reads it as a cell_box.
struct object_summary {
    // Declared column first, so the braces that build one ({ col, row, width,
    // height, count }) read left, top, then size.
    int8_t  col = 0, row = 0;        // top-left in grid coordinates
    int8_t  width = 0, height = 0;   // box width and height
    int16_t count = 0;               // number of cells in the object
    cell_box box() const { return { row, col, row + height - 1, col + width - 1 }; }
};

// A box as an object's summary has it, with its cell count (object_summary::box
// reads it back).
inline object_summary summary_of(const cell_box& box, int count) {
    return { (int8_t)box.col0, (int8_t)box.row0, (int8_t)box.width(), (int8_t)box.height(),
             (int16_t)count };
}

struct object {
    object_summary summary;
    // Stored relative to the box's top-left.
    cells data;

    // Build from the grid-cell indices returned by each_object.
    // Cells outside the group stay transparent, including holes in its box.
    static object from_cells(const grid& source, const object_summary& box,
                             const int16_t* indices, int count) {
        object made;
        made.summary = box;
        for (int i = 0; i < count; ++i) {
            const int row = cell_row(indices[i]), col = cell_col(indices[i]);
            made.data[row - box.row][col - box.col] = (int8_t)source.data[row][col];
        }
        return made;
    }

    // Object cell accessors here take grid coordinates, not box-local coordinates; they do the shift.
    // Gaps or holes within the box read as TRANSPARENT, and so does anything outside it.
    int8_t color_at(int row, int col) const {
        const int local_row = row - summary.row;
        const int local_col = col - summary.col;
        if (local_row < 0 || local_row >= summary.height ||
            local_col < 0 || local_col >= summary.width) return (int8_t)TRANSPARENT;
        return data[local_row][local_col];
    }

    bool holds(int row, int col) const {
        return color_at(row, col) != (int8_t)TRANSPARENT;
    }

    // The color of its first cell in reading order, which is the cell the walk
    // started from and the one an op reads a piece's color off. TRANSPARENT
    // when it has no cells.
    int8_t first_color() const {
        for (int row = 0; row < summary.height; ++row)
            for (int col = 0; col < summary.width; ++col)
                if (data[row][col] != (int8_t)TRANSPARENT) return data[row][col];
        return (int8_t)TRANSPARENT;
    }

    // Every cell the object fills, in reading order: see(row, col, color), in
    // grid coordinates.
    template <class See>
    void each_cell(See see) const {
        for (int row = 0; row < summary.height; ++row)
            for (int col = 0; col < summary.width; ++col) {
                const int8_t color = data[row][col];
                if (color != TRANSPARENT) see(summary.row + row, summary.col + col, color);
            }
    }

    // The box its cells fill -- which the summary's box need not be: a block
    // keeps its whole box, empty cells and all.
    cell_box painted_box() const {
        cell_box box;
        each_cell([&](int row, int col, int8_t) { box.include_cell(row, col); });
        return box;
    }

    // Every cell it fills repainted in the color `color_of(was)` names.
    template <class ColorOf>
    void repaint(ColorOf color_of) {
        for (int row = 0; row < summary.height; ++row)
            for (int col = 0; col < summary.width; ++col)
                if (data[row][col] != (int8_t)TRANSPARENT) data[row][col] = (int8_t)color_of((int8_t)data[row][col]);
    }

    // The object on its own, as a grid just big enough to hold it: its own
    // colors where it reaches, TRANSPARENT in the gaps and nowhere else's cells.
    grid drawn() const {
        grid out(summary.width, summary.height);
        for (int row = 0; row < summary.height; ++row)
            for (int col = 0; col < summary.width; ++col)
                out.data[row][col] = data[row][col];
        return out;
    }
};

// THE EIGHT DIRECTIONS, the one numbering of directions in the project: the cells
// around a cell, and the directions a piece can go, numbered as a compass is,
// clockwise from up, as MoveToEdge, Shift and ExtendTo's first eight take
// them. The opposite direction is four on, a quarter turn two on, and the sides are the
// even directions.
struct direction {
    enum : int8_t { up, up_right, right, down_right, down, down_left, left, up_left, count };
};

// One step in the given direction; a direction out of range goes up.
struct direction_step { int down, across; };

inline direction_step step_of(int step_direction) {
    switch (step_direction) {
    case direction::up_right:   return { -1,  1 };
    case direction::right:      return {  0,  1 };
    case direction::down_right: return {  1,  1 };
    case direction::down:       return {  1,  0 };
    case direction::down_left:  return {  1, -1 };
    case direction::left:       return {  0, -1 };
    case direction::up_left:    return { -1, -1 };
    default:                  return { -1,  0 };
    }
}

// THE CELLS THAT JOIN A CELL: all eight directions, or only the four sides,
// the even directions -- the step between the directions asked, for 8 or for 4.
//
// Grouping cells into an object uses eight directions, "is this one object?" -- save
// the one finer cut that asks it of sides alone (touching_sides_same_color).
// Adjacency uses four directions, "what shares an edge with this cell?": a diagonal
// cell touches only at a corner.
//
// Why four there and eight for grouping. Picture a ring of blocks meeting only
// at their corners. Ask "is this one ring" and the answer is yes, corners
// count, which is the eight. Ask "can the water inside get out" and corners
// must not count, or the water escapes through the very tip just called a
// wall and the ring has no inside at all. So anything measuring a border --
// what frays against what, what surrounds what, whether background leaks to
// the grid edge -- takes the four. Whatever the shapes use, the space around
// them uses the other one. Let both use eight and no shape has an inside any
// more, so holes stop existing.
template <int direction_count>
constexpr int every_nth_direction = direction_count == 4 ? 2 : 1;

// Walks connected cells: eight directions for objects, four for surrounding space.
// `claim` decides membership into that group and must mark what it accepts, or the walk never ends.
template <int direction_count = 8, class Claim, class See>
inline int walk_out(int rows, int cols, int16_t queue_row[], int16_t queue_col[], int seeded,
                    Claim claim, See see) {
    static_assert(direction_count == 4 || direction_count == 8);
    int head = 0, tail = seeded;
    while (head < tail) {
        const int row = queue_row[head], col = queue_col[head];
        ++head;
        see(row, col);
        for (int neighbor_direction = 0; neighbor_direction < direction::count; neighbor_direction += every_nth_direction<direction_count>) {
            const int next_row = row + step_of(neighbor_direction).down, next_col = col + step_of(neighbor_direction).across;
            if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) continue;
            if (!claim(next_row, next_col)) continue;
            queue_row[tail] = (int16_t)next_row; queue_col[tail] = (int16_t)next_col;
            ++tail;
        }
    }
    return tail;
}

// What counts as one object. The walk answers all four the same way, so no
// two readings can disagree about which cells belong together.
enum class grouping : int8_t {
    // Cells that touch, all of them the same color. The usual object.
    touching_same_color,
    // Cells that touch, whatever their colors. One object may be many colors.
    touching_any_color,
    // Every cell of a color at once, touching or not. One object per color.
    all_of_one_color,
    // Cells that share a side, all of them the same color: the usual object cut
    // finer, a corner joining nothing (aedd82e4's lone cells).
    touching_sides_same_color,
};

// Caller-owned so a caller can lend memory it already holds.
struct object_walk_scratch {
    bool taken[MAX_GRID_DIM][MAX_GRID_DIM];
    int16_t queue_row[MAX_GRID_CELLS];
    int16_t queue_col[MAX_GRID_CELLS];
    int16_t cells[MAX_GRID_CELLS];
};

// Boundaries for reading copies separately; zero means one whole picture.
struct tile_size { int width = 0, height = 0; };

// Find every touching group and hand it to `take`: summary, cells as pointer
// and length. The reading and the ops both find objects here, so they share
// one answer about what one object is.
//
// The cells point into `scratch`, which the next group overwrites -- `take`
// must COPY anything it keeps. That is what lets this run without allocating.
//
// `includes` chooses among colored cells without changing the source grid.
template <class Take, class Includes>
inline void each_object(const grid& one_grid, grouping how, object_walk_scratch& scratch,
                        Take take, tile_size tiles, Includes includes) {
    const bool any_color = (how == grouping::touching_any_color);
    // One object per color takes every cell of the color at once, wherever it
    // is; the touching kinds walk out from the seed cell only.
    const bool whole_color = (how == grouping::all_of_one_color);
    if (one_grid.width <= 0 || one_grid.height <= 0) return;
    // A group never leaves its seed's copy when boundaries are supplied.
    bool (&taken)[MAX_GRID_DIM][MAX_GRID_DIM] = scratch.taken;
    for (int row = 0; row < one_grid.height; ++row)
        for (int col = 0; col < one_grid.width; ++col)
            taken[row][col] = !is_color(one_grid.data[row][col]) || !includes(row, col);
    int16_t* queue_row = scratch.queue_row;
    int16_t* queue_col = scratch.queue_col;
    int16_t* found_cells = scratch.cells;
    for (int row = 0; row < one_grid.height; ++row)
        for (int col = 0; col < one_grid.width; ++col) {
            if (taken[row][col]) continue;
            const int color = one_grid.data[row][col];
            taken[row][col] = true;
            queue_row[0] = (int16_t)row; queue_col[0] = (int16_t)col;
            int seen = 0;
            int seeded = 1;
            cell_box box;
            auto in_this_copy = [&](int other_row, int other_col) {
                return (tiles.height <= 0 || other_row / tiles.height == row / tiles.height) &&
                       (tiles.width <= 0 || other_col / tiles.width == col / tiles.width);
            };
            // Every other cell of this color joins before the walk starts, and
            // the walk below then claims nothing, so the group is the color.
            if (whole_color)
                for (int other_row = 0; other_row < one_grid.height; ++other_row)
                    for (int other_col = 0; other_col < one_grid.width; ++other_col) {
                        if (taken[other_row][other_col] || !in_this_copy(other_row, other_col) ||
                            one_grid.data[other_row][other_col] != color) continue;
                        taken[other_row][other_col] = true;
                        queue_row[seeded] = (int16_t)other_row;
                        queue_col[seeded] = (int16_t)other_col;
                        ++seeded;
                    }
            // One flood fill for the whole project, so no two readings can
            // disagree about which cells belong to one group: eight directions,
            // or four where only a side joins.
            auto claim = [&](int next_row, int next_col) {
                if (whole_color || taken[next_row][next_col] || !in_this_copy(next_row, next_col)) return false;
                const int there = one_grid.data[next_row][next_col];
                if (!any_color && there != color) return false;
                taken[next_row][next_col] = true;
                return true;
            };
            auto see = [&](int seen_row, int seen_col) {
                found_cells[seen++] = (int16_t)cell_index(seen_row, seen_col);
                box.include_cell(seen_row, seen_col);
            };
            const int count = how == grouping::touching_sides_same_color
                ? walk_out<4>(one_grid.height, one_grid.width, queue_row, queue_col, seeded, claim, see)
                : walk_out<8>(one_grid.height, one_grid.width, queue_row, queue_col, seeded, claim, see);
            take(summary_of(box, count), found_cells, seen);
        }
}

template <class Take>
inline void each_object(const grid& one_grid, grouping how, object_walk_scratch& scratch,
                        Take take, tile_size tiles = {}) {
    each_object(one_grid, how, scratch, take, tiles, [](int, int) { return true; });
}

// SHAPES · a shape as one number, and the ways round that leave it be

// Turn a shape into one number, so "is this the same shape?" becomes "are these
// two numbers equal?". Shape here means which cells of the box around it are
// filled, plus w and h. Not the color, not where it sits on the grid.
// One side stores these numbers and the other looks them up, so if the two
// ever hashed differently every lookup would miss. Hence one function, called
// by both.
//
// occ(i) says whether cell i is filled, counting across each row in turn. Every
// caller must count the same way or the two sides hash the same shape apart.
template <typename OccupancyAt>
inline uint32_t shape_occupancy_hash(int w, int h, OccupancyAt occ) {
    uint32_t hsh = 2166136261u;
    auto mix = [&](uint8_t b) { hsh = (hsh ^ b) * 16777619u; };
    mix((uint8_t)w); mix((uint8_t)h);
    for (int i = 0; i < w * h; ++i) mix(occ(i) ? (uint8_t)1 : (uint8_t)0);
    return hsh;
}

// THE EIGHT POSES A GRID CAN TAKE: four turns by two mirrors, packed in three
// bits, pose 0 being the grid as given.
//   1  rows and columns trade places, transpose
//   2  flipped top to bottom
//   4  flipped left to right
// Single home of what the bits mean. Every reader of a pose goes through the
// ones below, so none of them can drift on what "the same shape, turned" is.
constexpr int POSE_COUNT = 8;

// How big a grid reads once posed. Only the trade changes the size.
inline pair<int, int> posed_size(int pose, int width, int height) {
    return (pose & 1) ? pair<int, int>{ height, width } : pair<int, int>{ width, height };
}

// Which cell of the ORIGINAL a posed cell reads from. Width and height are the
// original's own, which is what the mirrors are measured against.
inline pair<int, int> pose_source(int pose, int width, int height,
                                  int posed_row, int posed_col) {
    int row = (pose & 1) ? posed_col : posed_row;
    int col = (pose & 1) ? posed_row : posed_col;
    if (pose & 2) row = height - 1 - row;
    if (pose & 4) col = width  - 1 - col;
    return { row, col };
}

// Where a cell of the ORIGINAL lands once posed, so a place can be carried into
// the frame a pair is compared in. The exact inverse of pose_source.
inline pair<int, int> posed_place(int pose, int width, int height,
                                  int row, int col) {
    const int down   = (pose & 2) ? height - 1 - row : row;
    const int across = (pose & 4) ? width  - 1 - col : col;
    return (pose & 1) ? pair<int, int>{ across, down } : pair<int, int>{ down, across };
}

// THE EIGHT POSES IN THE ORDER A SHAPE ASKS THEM, turns before flips and
// rights before lefts: as it lies, a quarter turn right, a half turn, a
// quarter turn left, then flipped left to right, top to bottom, and across
// either diagonal. Where two ways lay a shape on another, the first is how it
// lies there (103eff5b's key, which a flip lays on the azure shape too, in the
// wrong colors).
constexpr int pose_asked(int asked) {
    switch (asked) {
    case 1: return 3;
    case 2: return 6;
    case 3: return 5;
    case 4: return 4;
    case 5: return 2;
    case 6: return 1;
    case 7: return 7;
    default: return 0;
    }
}

// What a shape may change and still be the same shape: colors, turns, flips, size.
struct shape_freedom { bool turn = false, scale = false, stretch = false, recolor = true; };
// A shape as it lies, a shape any way round, and a drawing: nothing may change.
constexpr shape_freedom AS_IT_LIES{}, ANY_WAY_ROUND{ true }, DRAWING{ false, false, false, false };

// Whether two cells of shapes are alike, as `free` lets their colors differ.
constexpr bool cells_alike(int one, int other, shape_freedom free) {
    return free.recolor ? (one != TRANSPARENT) == (other != TRANSPARENT) : one == other;
}

// THE BLOCKS A SHAPE IS DRAWN IN: the most rows, and the most columns, its
// cells repeat in alike, lined up with its box -- 1 where nothing repeats.
// Shrunk by them, a shape is its smallest drawing, the one every bigger
// drawing of it shrinks to: every factor it repeats in divides the largest,
// so the largest shrinks it all the way.
struct shape_blocks { int down = 1, across = 1; };
template <class Cell>
inline shape_blocks blocks_of(int width, int height, Cell cell, shape_freedom free) {
    auto repeats = [&](int factor, bool rows) {
        if ((rows ? height : width) % factor != 0) return false;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                if (!cells_alike(cell(row, col), cell(rows ? row - row % factor : row, rows ? col : col - col % factor),
                                 free))
                    return false;
        return true;
    };
    shape_blocks blocks;
    for (int factor = height; factor > 1; --factor)
        if (repeats(factor, true)) { blocks.down = factor; break; }
    for (int factor = width; factor > 1; --factor)
        if (repeats(factor, false)) { blocks.across = factor; break; }
    return blocks;
}

// A SHAPE WITH NO SIZE BUT ITS OWN: one whose smallest drawing is one cell,
// a dot or a filled box, is every filled box at some size, so a likeness at
// another size says nothing of it. It keeps its own size: at any size it is
// the same only as itself. Its blocks, as a shape may shrink by them.
inline shape_blocks sized_blocks(shape_blocks blocks, int width, int height) {
    return blocks.down == height && blocks.across == width ? shape_blocks{} : blocks;
}

// The blocks a shape is shrunk by, as `free` lets its size change: none
// where it may not, nor where it has no size but its own (sized_blocks).
template <class Cell>
inline shape_blocks shrunk_by(int width, int height, Cell cell, shape_freedom free) {
    return free.scale || free.stretch ? sized_blocks(blocks_of(width, height, cell, free), width, height)
                                      : shape_blocks{};
}

// THE KEY OF A SHAPE, as `free` lets it change: one number every drawing of
// the same shape shares -- read off its smallest drawing where its size may
// change (scaled, with its blocks' proportions), the smallest over the eight
// ways round where it may turn, done by index arithmetic, no pose grids
// built. Only which cells it paints enters a key, never their colors. Equal
// shapes have equal keys; collisions are possible. THE KEY OF THE LEARNED
// SHAPE TABLE: the reading learns it and the ops that read the table look it
// up, both here, so a shape the pairs showed one way round is the same shape
// the test shows another (e8593010's L, turned).
template <class Cell>
inline uint32_t key_of_shape(int width, int height, Cell cell, shape_freedom free) {
    const shape_blocks blocks = shrunk_by(width, height, cell, free);
    const int small_width = width / blocks.across, small_height = height / blocks.down;
    uint32_t best = 0xFFFFFFFFu;
    for (int pose = 0; pose < (free.turn ? POSE_COUNT : 1); ++pose) {
        const auto [posed_width, posed_height] = posed_size(pose, small_width, small_height);
        uint32_t hash = shape_occupancy_hash(posed_width, posed_height, [&](int i) {
            const auto [row, col] = pose_source(pose, small_width, small_height, i / posed_width,
                                                i % posed_width);
            return cell(row * blocks.down, col * blocks.across) != TRANSPARENT;
        });
        if (free.scale && !free.stretch) {
            int down = (pose & 1) ? blocks.across : blocks.down, across = (pose & 1) ? blocks.down : blocks.across;
            for (int common = down; common > 1; --common)
                if (down % common == 0 && across % common == 0) { down /= common; across /= common; break; }
            hash = (hash ^ (uint32_t)(down * MAX_GRID_DIM + across)) * 16777619u;
        }
        if (hash < best) best = hash;
    }
    return best;
}

// HOW ONE SHAPE LIES ON ANOTHER: the way round the first is turned (pose),
// the blocks each is shrunk by, and the first's smallest drawing.
struct shape_fit {
    int pose = 0;
    shape_blocks mine, theirs;
    int small_width = 0, small_height = 0;
};

// The cell of the second shape that the first's cell (row, col) lies on,
// read from the top-left of its block, both from their boxes' top-left.
inline pair<int, int> fit_place(const shape_fit& fit, int row, int col) {
    const auto [down, across] = posed_place(fit.pose, fit.small_width, fit.small_height,
                                            row / fit.mine.down, col / fit.mine.across);
    return { down * fit.theirs.down, across * fit.theirs.across };
}

// WHETHER TWO SHAPES ARE THE SAME, as `free` lets them differ: the first
// `width` by `height`, read by cell(row, col), the second `other_width` by
// `other_height`, read by other(row, col). Where size may change, each is
// shrunk to its smallest drawing by its blocks (`mine`, `theirs`: blocks_of,
// read by the asker where it keeps them -- each piece's once), and a scaled
// one's blocks keep the other's proportions; then the first, turned one of the
// ways round it may take, in the order they are asked (pose_asked,
// pose_source), lies on the second cell for cell alike. How it lies there is
// handed to `fit`.
template <class Cell, class Other>
inline bool same_shape_in_blocks(int width, int height, Cell cell, shape_blocks mine,
                                 int other_width, int other_height, Other other, shape_blocks theirs,
                                 shape_freedom free, shape_fit* fit = nullptr) {
    if (!free.scale && !free.stretch) mine = theirs = shape_blocks{};
    mine = sized_blocks(mine, width, height);
    theirs = sized_blocks(theirs, other_width, other_height);
    const int small_width = width / mine.across, small_height = height / mine.down;
    for (int asked = 0; asked < (free.turn ? POSE_COUNT : 1); ++asked) {
        const int pose = pose_asked(asked);
        const auto [posed_width, posed_height] = posed_size(pose, small_width, small_height);
        if (posed_width * theirs.across != other_width || posed_height * theirs.down != other_height) continue;
        const int down = (pose & 1) ? mine.across : mine.down, across = (pose & 1) ? mine.down : mine.across;
        if (!free.stretch && down * theirs.across != across * theirs.down) continue;
        bool same = true;
        for (int row = 0; row < posed_height && same; ++row)
            for (int col = 0; col < posed_width && same; ++col) {
                const auto [from_row, from_col] = pose_source(pose, small_width, small_height, row, col);
                same = cells_alike(cell(from_row * mine.down, from_col * mine.across),
                                   other(row * theirs.down, col * theirs.across), free);
            }
        if (!same) continue;
        if (fit) *fit = { pose, mine, theirs, small_width, small_height };
        return true;
    }
    return false;
}
// The same, each shape's blocks read here.
template <class Cell, class Other>
inline bool same_shape(int width, int height, Cell cell, int other_width, int other_height,
                       Other other, shape_freedom free) {
    return same_shape_in_blocks(width, height, cell, shrunk_by(width, height, cell, free),
                                other_width, other_height, other, shrunk_by(other_width, other_height, other, free),
                                free);
}

}  // namespace arc
