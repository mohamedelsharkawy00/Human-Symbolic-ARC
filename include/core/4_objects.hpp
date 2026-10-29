// Which cells belong together as an object?
// A grid might contain separate shapes, groups of one color, or repeating blocks.
// We try these different groupings so the next chapters can ask what happened
// to the objects. Dividing a grid this way is called segmentation.
#pragma once
#include "1_readings.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// The grid can be divided into objects in several ways.
// Dividing a grid into objects is called segmentation.
// Connected objects use 8-way connectivity: edges or corners -- save one cut,
// read after every other (6_correspondence.hpp), that joins one color's cells
// by edges alone.
// Surrounding space and holes use 4-way connectivity: edges only.
// And one cut, read after all of them, takes every place the shape the pairs'
// changes take fits on the colors it turns from and into (changed_shape) as a
// piece, pieces free to share cells, then joins the rest as the connected
// one-color cut does.
// Segmentation methods are compared by how well they explain the training pairs.
enum class segmentation_methods : int8_t {
    connected_monochrome, connected_multicolor, all_cells_of_one_color,
    minimal_repeating_set, smallest_of_two_grids, four_connected_monochrome, shape_fits,
    // How many real methods there are, so a loop can run over them.
    count,
    // Nothing has been chosen yet. It shares count's value so that loops
    // over the real methods skip it, and every test for a real method
    // (boxed_by_its_cells and the like) says no to it for free.
    unset = count
};

// Label used for GUI
inline constexpr const char* segmentation_names_lead = "An Object Is";

// GUI and logs
inline constexpr const char* const segmentation_names[] = {
    "8-Connected Monochrome Cells",
    "8-Connected Multicolor Cells",
    "All Cells Of One Color",
    "Minimal Repeating Set Of Cells",
    "Smallest Of The Two Grids",
    "4-Connected Monochrome Cells",
    "Where The Changed Shape Fits, The Rest 8-Connected",
    "Not Chosen Yet"  // segmentation_methods::unset
};

// Whether a segmentation measures each object in the box its own cells make. The
// connected and one-color groupings and the shape cut do; a repeating block
// keeps its whole block, blank cells and all.
inline constexpr bool boxed_by_its_cells(segmentation_methods segmentation) {
    return segmentation == segmentation_methods::connected_monochrome ||
           segmentation == segmentation_methods::connected_multicolor ||
           segmentation == segmentation_methods::all_cells_of_one_color ||
           segmentation == segmentation_methods::four_connected_monochrome ||
           segmentation == segmentation_methods::shape_fits;
}

// An election nobody filled in must not pass for a real reading: unset is
// what such an object holds, and this is what keeps it from sneaking
// through a boxed_by_its_cells gate.
static_assert(!boxed_by_its_cells(segmentation_methods::unset),
              "unset must not look like a real segmentation");

// What counts as joined under a segmentation that follows the cells. Whether
// an object has to be one color is a question of its OWN, asked apart from
// what counts as joined: the one walk in 1_readings.hpp answers both.
inline constexpr grouping grouping_of(segmentation_methods segmentation) {
    return segmentation == segmentation_methods::connected_multicolor      ? grouping::touching_any_color :
           segmentation == segmentation_methods::all_cells_of_one_color    ? grouping::all_of_one_color :
           segmentation == segmentation_methods::four_connected_monochrome ? grouping::touching_sides_same_color :
                                                                             grouping::touching_same_color;
}

// Whether every piece a segmentation cuts wears one color, so a pick by a color
// takes pieces of that color alone.
inline constexpr bool one_color_a_piece(segmentation_methods segmentation) {
    return boxed_by_its_cells(segmentation) && grouping_of(segmentation) != grouping::touching_any_color;
}

// The unit Pattern cuts: the smallest repeating width and height, or the grid's
// own. Every cell counts, an empty one too; one copy and a matching fragment do.
inline void repeat_periods(const grid& g, int& width, int& height) {
    // Does the grid repeat every `period` cells along one axis? `across` asks
    // about columns, otherwise rows. One test for both, so they cannot drift.
    auto repeats_every = [&](int period, bool across) {
        for (int r = across ? 0 : period; r < g.height; ++r)
            for (int c = across ? period : 0; c < g.width; ++c) {
                const int folded_row = across ? r : r % period;
                const int folded_col = across ? c % period : c;
                if (g.data[r][c] != g.data[folded_row][folded_col]) return false;
            }
        return true;
    };
    // Find the shortest valid period, or use the full size if none repeats.
    auto smallest_period = [&](int size, bool across) {
        for (int period = 1; period < size; ++period)
            if (repeats_every(period, across)) return period;
        return size;
    };
    width = smallest_period(g.width, true);
    height = smallest_period(g.height, false);
}

// EVERY PLACE A SHAPE FITS ON ONE COLOR of `one_grid`, among the colored cells
// `includes(row, col)` takes: each top-left where every cell of `shape`, an
// object at its own box's corner, lands on a cell it takes, all of one color,
// within one copy where `tiles` bounds the copies -- handed to
// `see(row, col, color)` in reading order.
template <class See, class Includes>
inline void each_place_it_fits(const grid& one_grid, const object& shape, tile_size tiles, See see,
                               Includes includes) {
    for (int row = 0; row + shape.height <= one_grid.height; ++row)
        for (int col = 0; col + shape.width <= one_grid.width; ++col) {
            int color = TRANSPARENT;
            bool fits = true;
            shape.each_cell([&](int down, int across, int8_t) {
                const int at_row = row + down, at_col = col + across;
                const int there = one_grid.data[at_row][at_col];
                fits = fits && is_color(there) && includes(at_row, at_col) &&
                       (color == TRANSPARENT || there == color) &&
                       (tiles.height <= 0 || at_row / tiles.height == row / tiles.height) &&
                       (tiles.width <= 0 || at_col / tiles.width == col / tiles.width);
                color = there;
            });
            if (fits && is_color(color)) see(row, col, color);
        }
}

// A SHAPE WHOSE PLACES TURN ONE COLOR INTO ANOTHER (changed_shape): its cells,
// an object at its own box's corner, the color its places turn `from` and the
// one they turn `into`, the two the shape cut finds it on.
struct turning_shape {
    object cells;
    int8_t from = TRANSPARENT, into = TRANSPARENT;
    bool holds() const { return cells.cell_count() > 0; }
};

// THE PLACES THE SHAPE CUT TAKES: every place the shape fits on the color it
// turns from or into, among the cells `includes(row, col)` takes -- each
// top-left handed to `see(row, col)` in reading order.
template <class See, class Includes>
inline void each_shape_place(const grid& one_grid, const turning_shape& shape, tile_size tiles, See see,
                             Includes includes) {
    each_place_it_fits(one_grid, shape.cells, tiles, [&](int row, int col, int color) {
        if (color == shape.from || color == shape.into) see(row, col);
    }, includes);
}

// THE PIECES THE SHAPE CUT CUTS (shape_fits): its places (each_shape_place), a
// piece each, in reading order, places sharing the cells where they overlap;
// then the cells no place took, joined as the connected one-color cut joins
// them. As many as a list holds, one for each cell of the largest grid, the
// rest dropped as Tile drops copies past the list's room. Handed to
// `take(box, cells, count)` as each_object hands its groups; hands back how
// many places came first.
template <class Take>
inline int each_shape_piece(const grid& one_grid, const turning_shape& shape, object_walk_scratch& scratch,
                            Take take, tile_size tiles = {}) {
    bool in_shape[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    int pieces = 0;
    int16_t cells[MAX_GRID_CELLS];
    each_shape_place(one_grid, shape, tiles, [&](int row, int col) {
        if (pieces >= MAX_GRID_CELLS) return;
        int count = 0;
        cell_box box;
        shape.cells.each_cell([&](int down, int across, int8_t) {
            cells[count++] = (int16_t)cell_index(row + down, col + across);
            box.include_cell(row + down, col + across);
            in_shape[row + down][col + across] = true;
        });
        take(summary_of(box, count), cells, count);
        ++pieces;
    }, [](int, int) { return true; });
    const int places = pieces;
    each_object(one_grid, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* found, int count) {
        if (pieces++ < MAX_GRID_CELLS) take(box, found, count);
    }, tiles, [&](int row, int col) { return !in_shape[row][col]; });
    return places;
}

// THE SHAPE THE CHANGES TAKE: where cells of one color turn another in every
// pair, the fewest of them that touch, two at least, joined by sides or
// corners, as they lie -- held only where in every pair the places it fits on
// the first color are the cells that turn, no more and no fewer. Read off the
// pairs as given, their ground a color like any. The shape the shape cut cuts
// by; none, with no cells, where none holds.
inline turning_shape changed_shape(const vector<example>& train) {
    if (train.empty()) return {};
    for (const example& pair : train)
        if (!pair.input.same_size(pair.output)) return {};
    object_walk_scratch scratch;
    for (int was = 0; was < COLOR_COUNT; ++was)
        for (int now = 0; now < COLOR_COUNT; ++now) {
            if (was == now) continue;
            // Each pair's cells that turn, laid on an empty grid.
            vector<grid> turned;
            bool every = true;
            for (const example& pair : train) {
                grid turning(pair.input.width, pair.input.height);
                bool any = false;
                for (int row = 0; row < turning.height; ++row)
                    for (int col = 0; col < turning.width; ++col)
                        if (pair.input.data[row][col] == was && pair.output.data[row][col] == now) {
                            turning.data[row][col] = (int8_t)now;
                            any = true;
                        }
                every = every && any;
                turned.push_back(turning);
            }
            if (!every) continue;
            // The fewest that touch, the first found on a tie.
            object shape;
            for (const grid& turning : turned)
                each_object(turning, grouping::touching_same_color, scratch,
                            [&](const object_summary& box, const int16_t* found, int count) {
                    if (count >= 2 && (shape.cell_count() == 0 || count < shape.cell_count()))
                        shape = object::from_cells(turning, box, found, count);
                });
            if (shape.cell_count() == 0) continue;
            shape.row = shape.col = 0;
            bool held = true;
            for (size_t which = 0; which < train.size() && held; ++which) {
                grid covered(train[which].input.width, train[which].input.height);
                each_place_it_fits(train[which].input, shape, {}, [&](int row, int col, int color) {
                    if (color != was) return;
                    shape.each_cell([&](int down, int across, int8_t) {
                        covered.data[row + down][col + across] = (int8_t)now;
                    });
                }, [](int, int) { return true; });
                held = covered == turned[which];
            }
            if (held) return { shape, (int8_t)was, (int8_t)now };
        }
    return {};
}

// A patch of background: its box and cell count, and whether the ground closes
// it in (a hole) rather than running off the edge of the grid.
struct blank_patch {
    object_summary box;
    bool enclosed = false;
    // A patch with no box was never found.
    bool found() const { return box.width > 0; }
};

// The object records and the operations that build and group them.
struct object_segmentation {
    struct object_info {
        // The one color every cell wears; -1 when it wears several, as a block does.
        int8_t color = -1;
        // Bounding box and cell count, defined in 1_readings.hpp.
        object_summary summary;
        // Canonical shape hash (orientationless, positionless)
        uint32_t shape = 0;
        // Its shape at any size, any way round (key_of_shape): the key a piece
        // found bigger or smaller on the other grid shares with it -- none
        // but its own size's for a dot or a filled box (sized_blocks).
        uint32_t scaled_shape = 0;
        // Number of matches touching this object, counted after the matches are final.
        int edges_touching = 0;
        // Whether a match identifies a partner on the other grid.
        bool has_partner = false;
        // Explained by a matched representative, without an individual edge.
        int16_t instance_of = -1;
        // Copy group on this grid; -1 means no group.
        int16_t motif = -1;
        // True for the group member with the fewest cell differences from the others.
        bool medoid = false;
        // Background region on the other grid touched by an unmatched object.
        blank_patch blank;
    };
    // Operations used by the later chapters.
    // THE SMALLER OF ONE PAIR'S TWO GRIDS, as the block the larger one is cut
    // into: its size when the larger is a whole number of it both ways, equal
    // sizes included -- one block each, the whole grid, which is how an output
    // that is its input turned reads. {0, 0} when neither divides the other.
    // Asked pair by pair: an output made of copies of its input is made of THIS
    // input, whatever size the other pairs' inputs were.
    static pair<int, int> smaller_grid_step(const example& pair) {
        const bool output_grows = pair.output.width >= pair.input.width &&
                                  pair.output.height >= pair.input.height;
        const grid& block = output_grows ? pair.input : pair.output;
        const grid& tiled = output_grows ? pair.output : pair.input;
        if (block.empty()) return { 0, 0 };
        if (tiled.width % block.width || tiled.height % block.height) return { 0, 0 };
        return { block.width, block.height };
    }

    // Divides a grid into objects, then sorts them by size (largest first) and position.
    // The shape cut cuts by `shape`, and nothing without one.
    static void segment_for_correspondence(const grid& one_grid, segmentation_methods segmentation,
                                    vector<object_info>& info,
                                    vector<object>* objects,
                                    int step_width = 0, int step_height = 0, tile_size tiles = {},
                                    const turning_shape* shape = nullptr) {
        segment_into_objects(one_grid, segmentation, info, objects, step_width, step_height, tiles, shape);
        vector<int> reading_order(info.size());
        iota(reading_order.begin(), reading_order.end(), 0);
        stable_sort(reading_order.begin(), reading_order.end(), [&](int left, int right) {
            if (info[left].summary.count != info[right].summary.count)
                return info[left].summary.count > info[right].summary.count;
            if (info[left].summary.row != info[right].summary.row)
                return info[left].summary.row < info[right].summary.row;
            return info[left].summary.col < info[right].summary.col;
        });
        vector<object_info> sorted;
        sorted.reserve(info.size());
        for (int i : reading_order) sorted.push_back(info[i]);
        info.swap(sorted);
        if (objects) {
            vector<object> moved;
            moved.reserve(objects->size());
            for (int i : reading_order) moved.push_back((*objects)[i]);
            objects->swap(moved);
        }
    }

    // Helps choose which segmentation to use: prefer better compression.
    // Compares storing shapes and their positions with storing individual colored cells.
    // 0.5 means compression to half the size; 2 means expansion to twice the size.
    static double describe_cost(const grid& one_grid, const vector<object_info>& objects) {
        const double cell_bits = log2((double)COLOR_COUNT);
        const double place_bits =
            log2((double)max(1, one_grid.width * one_grid.height));
        int colored = 0;
        one_grid.for_each_colored_cell(TRANSPARENT, [&](int, int, int) { ++colored; });
        // Nothing on the grid, so there is nothing to save and nothing to lose.
        const double plain = colored * cell_bits;
        if (plain <= 0.0) return 1.0;
        // Charge an empty segmentation the full grid cost plus a position cost.
        if (objects.empty()) return (plain + place_bits) / plain;
        // A position per object, plus the cells of each shape the first time it is
        // met. Same shape turned round is the same shape — one key (key_of_shape).
        double cost = objects.size() * place_bits;
        for (size_t i = 0; i < objects.size(); ++i) {
            bool said_before = false;
            for (size_t j = 0; j < i && !said_before; ++j)
                said_before = (objects[j].shape == objects[i].shape);
            if (!said_before) cost += objects[i].summary.count * cell_bits;
        }
        return cost / plain;
    }

    // Choose a shared block size using whichever grid compresses better.
    // Uses both training grids, so this must not supply a block to an input-only operation.
    static pair<int, int> block_of(const grid& input, const grid& output,
                                   segmentation_methods segmentation, int sizes_width, int sizes_height) {
        if (segmentation != segmentation_methods::minimal_repeating_set)
            return { sizes_width, sizes_height };
        const auto from_input = pattern_block_of(input);
        const auto from_output = pattern_block_of(output);
        // Cut each grid on its own candidate and ask what that costs to say; the
        // cheaper block is the one that found real repetition rather than noise.
        auto cost_of = [&](const grid& one_grid, pair<int, int> block) {
            vector<object_info> objects;
            segment_for_correspondence(one_grid, segmentation, objects, nullptr, block.first, block.second);
            return describe_cost(one_grid, objects);
        };
        return cost_of(output, from_output) < cost_of(input, from_input) ? from_output
                                                                        : from_input;
    }

    // The block Pattern cuts a grid into (repeat_periods).
    static pair<int, int> pattern_block_of(const grid& one_grid) {
        int width, height;
        repeat_periods(one_grid, width, height);
        return { width, height };
    }

    // Groups copies of an object and chooses the most typical copy in each group (the medoid).
    static void find_copies(vector<object_info>& info,
                            const vector<object>& objects, bool one_set_for_the_grid) {
        vector<uint32_t> sets;
        for (size_t i = 0; i < info.size(); ++i) {
            const uint32_t key = one_set_for_the_grid ? 0u : info[i].shape;
            const bool has_company = one_set_for_the_grid ||
                ranges::any_of(info, [&](const auto& other) {
                    return &other != &info[i] && other.shape == info[i].shape &&
                           other.summary.count == info[i].summary.count; });
            if (!has_company) continue;
            auto at = ranges::find(sets, key);
            if (at == sets.end()) { sets.push_back(key); at = sets.end() - 1; }
            info[i].motif = (int16_t)(at - sets.begin());
        }
        for (int16_t set = 0; set < (int16_t)sets.size(); ++set) {
            int best = -1, least_apart = 0;
            for (size_t i = 0; i < info.size(); ++i) {
                if (info[i].motif != set) continue;
                int apart = 0;
                for (size_t j = 0; j < info.size(); ++j)
                    if (j != i && info[j].motif == set)
                        apart += count_differing_cells(objects[i], objects[j]);
                if (best < 0 || apart < least_apart) { best = (int)i; least_apart = apart; }
            }
            if (best >= 0) info[best].medoid = true;
        }
    }

private:
    // Divides a grid of figures into objects. It has no ground: the plan
    // deleted it (3_background.hpp), and an empty cell is in no object.
    static void segment_into_objects(const grid& one_grid, segmentation_methods segmentation,
                                    vector<object_info>& info,
                                    vector<object>* objects,
                                    int step_width = 0, int step_height = 0, tile_size tiles = {},
                                    const turning_shape* shape = nullptr) {
        // Allow up to one object per grid cell.
        static constexpr int MAX_OBJECTS = MAX_GRID_CELLS;
        // One object found: its record always, its cells when the caller keeps them.
        auto keep = [&](const object& found, int8_t color) {
            info.push_back({ .color = color, .summary = summary_of(found.box(), found.cell_count()), .shape = shape_of(found),
                             .scaled_shape = scaled_shape_of(found) });
            if (objects) objects->push_back(found);
        };

        // Cut on a fixed step: equal blocks, one object each, cells of any color kept
        // together. Blocks with no colored cells at all are not objects.
        if (segmentation == segmentation_methods::smallest_of_two_grids ||
            segmentation == segmentation_methods::minimal_repeating_set) {
            if (step_width <= 0 || step_height <= 0) return;
            for (int block_top = 0; block_top < one_grid.height; block_top += step_height)
            for (int block_left = 0; block_left < one_grid.width; block_left += step_width) {
                if ((int)info.size() >= MAX_OBJECTS) return;
                // A step that does not divide the grid leaves a short block at the
                // edge; it is still a block, and dropping it would lose real cells.
                const int block_width  = min(step_width,  one_grid.width  - block_left);
                const int block_height = min(step_height, one_grid.height - block_top);
                object piece{ (int16_t)block_left, (int16_t)block_top, (int16_t)block_width, (int16_t)block_height };
                int lit = 0;
                for (int row = 0; row < block_height; ++row)
                    for (int col = 0; col < block_width; ++col) {
                        const int color = one_grid.data[block_top + row][block_left + col];
                        const bool is_colored = is_color(color);
                        if (is_colored) piece.data[row][col] = (int8_t)color;
                        lit += is_colored;
                    }
                if (!lit) continue;
                keep(piece, -1);
            }
            // Repeating-block segmentation needs at least two nonempty blocks.
            if (segmentation != segmentation_methods::smallest_of_two_grids && info.size() < 2) {
                info.clear();
                if (objects) objects->clear();
            }
            return;
        }
        object_walk_scratch scratch;
        auto take = [&](const object_summary& found, const int16_t* found_cells, int how_many) {
            if ((int)info.size() >= MAX_OBJECTS) return;
            // An object wears one color only if every cell of it does; -1 says several.
            int8_t color = (int8_t)one_grid.data[cell_row(found_cells[0])][cell_col(found_cells[0])];
            for (int i = 1; i < how_many && color >= 0; ++i)
                if (one_grid.data[cell_row(found_cells[i])][cell_col(found_cells[i])] != color) color = -1;
            keep(object::from_cells(one_grid, found, found_cells, how_many), color);
        };
        if (segmentation != segmentation_methods::shape_fits)
            each_object(one_grid, grouping_of(segmentation), scratch, take, tiles);
        else if (shape)
            each_shape_piece(one_grid, *shape, scratch, take, tiles);
    }

    // Canonical shape of an object: which cells of its box it fills, the same
    // number whichever way round it sits.
    // Its shape at any size, any way round.
    static uint32_t scaled_shape_of(const object& one) {
        return key_of_shape(one.width, one.height,
                            [&](int row, int col) { return (int)one.data[row][col]; }, { true, true });
    }
    static uint32_t shape_of(const object& one) {
        return key_of_shape(one.width, one.height,
                            [&](int row, int col) { return (int)one.data[row][col]; }, ANY_WAY_ROUND);
    }

    // Counts differing cells when two objects are aligned at their top-left corners.
    static int count_differing_cells(const object& left, const object& right) {
        const int width = max(left.width, right.width);
        const int height = max(left.height, right.height);
        int apart = 0;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) {
                const int here = left.color_at(left.row + row, left.col + col);
                const int there = right.color_at(right.row + row, right.col + col);
                apart += (here != there);
            }
        return apart;
    }

};

// WHICH PIECE THE SEEDS POINT AT. An op that repeats one piece along the others
// has to settle this before it draws -- the largest box is the piece being
// repeated, and everything else is a seed -- and a reading that says what that
// op will draw has to settle it the same way, or it describes a drawing that
// never happens. So the rule is written here once and both sides read it: the
// DSL when it fills its seed list (17_fixed_dsl.hpp), inference when it replays
// an op to check a claim (9_rules.hpp).
//
// THE TWO MEASURES ARE NOT THE SAME ONE. A piece is dropped from the seeds by
// the BOX it fills, while the operator that hands the repeated piece over
// first, SortBySize[0], ranks by CELLS. They usually agree and are not the
// same question, so they are kept apart here rather than quietly merged.
struct seeded_pieces {
    // Indices into the boxes handed in. -1 when nothing was.
    int repeated = -1;
    // The rest, smallest box first, which is the order an op walks them in.
    vector<int> seeds;
};

inline seeded_pieces read_seeds(const vector<object_summary>& boxes) {
    seeded_pieces reading;
    if (boxes.empty()) return reading;
    auto area_of = [&](int which) { return (int)boxes[which].width * (int)boxes[which].height; };
    // Ties go to the first found, which is the way both sides broke them before
    // they shared this.
    int largest = 0;
    for (int which = 1; which < (int)boxes.size(); ++which)
        if (area_of(which) > area_of(largest)) largest = which;
    reading.repeated = largest;
    for (int which = 0; which < (int)boxes.size(); ++which)
        if (which != largest) reading.seeds.push_back(which);
    stable_sort(reading.seeds.begin(), reading.seeds.end(),
                [&](int one, int other) { return area_of(one) < area_of(other); });
    return reading;
}

} // namespace arc
