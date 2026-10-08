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
#include <string>
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
// Segmentation methods are compared by how well they explain the training pairs.
enum class segmentation_methods : int8_t {
    connected_monochrome, connected_multicolor, all_cells_of_one_color,
    minimal_repeating_set, smallest_of_two_grids, four_connected_monochrome,
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
    "Not Chosen Yet"  // segmentation_methods::unset
};

// Whether a segmentation measures each object in the box its own cells make. The
// connected and one-color groupings do; a repeating block keeps its whole block,
// blank cells and all.
inline constexpr bool boxed_by_its_cells(segmentation_methods segmentation) {
    return segmentation == segmentation_methods::connected_monochrome ||
           segmentation == segmentation_methods::connected_multicolor ||
           segmentation == segmentation_methods::all_cells_of_one_color ||
           segmentation == segmentation_methods::four_connected_monochrome;
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

// Find the smallest repeating width and height; use the grid size if none repeats.
// Ignore background cells when matching; pass -2 to compare every cell.
// `must_fit_twice` requires two full copies along each repeating direction.
template <typename Grid>
ARC_HOST_DEVICE inline void repeat_periods(const Grid& g, int background,
                                           int& width, int& height,
                                           bool must_fit_twice = true) {
    // Does the grid repeat every `period` cells along one axis? `across` asks
    // about columns, otherwise rows. One test for both, so they cannot drift.
    auto repeats_every = [&](int period, bool across) {
        for (int r = across ? 0 : period; r < g.height; ++r)
            for (int c = across ? period : 0; c < g.width; ++c) {
                const int here = g.data[r][c];
                if (here == background) continue;
                const int folded_row = across ? r : r % period;
                const int folded_col = across ? c % period : c;
                if (here != g.data[folded_row][folded_col]) return false;
            }
        return true;
    };
    // Find the shortest valid period, or use the full size if none repeats.
    auto smallest_period = [&](int size, bool across) {
        const int longest = must_fit_twice ? size / 2 : size - 1;
        for (int period = 1; period <= longest; ++period)
            if (repeats_every(period, across)) return period;
        return size;
    };
    width = smallest_period(g.width, true);
    height = smallest_period(g.height, false);
}

// Extend's seed and rays are shared by inference and execution. Ties go to
// the first cell in row-major order; the seed itself is not part of the rays.
// Each ray cell is handed over as its offset from the seed, down and across,
// so whoever paints it clips it to what it paints on: a long ray runs on to the
// edge of the grid, not to the end of some frame of its own.
struct ray_seed { int row = -1, col = -1; };

template <class Cell>
ARC_HOST_DEVICE inline ray_seed nearest_ray_seed(int height, int width,
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

// The rays run the four sides of the compass, its even ways, or the four
// diagonals, its odd ones (heading, 1_readings.hpp).
template <class Paint>
ARC_HOST_DEVICE inline void each_extended_ray(ray_seed seed, int radius, bool diagonal,
                                               Paint paint) {
    if (seed.row < 0) return;
    for (int way = diagonal ? 1 : 0; way < heading::count; way += 2) {
        const heading_step step = step_of(way);
        for (int along = 1; along <= radius; ++along) paint(step.down * along, step.across * along);
    }
}

// TileX's and TileY's step, shared by inference and execution. A tiled piece is
// laid again and again `period` cells apart, counted from where it stands, its
// first column (TileX) or row (TileY) at `start`: it stays put, and its copies
// run both ways from it to the edges. So the cell `at` along that axis shows
// the piece's own cell `(at - start) mod period` along it -- or none, -1, where
// that falls past the piece's `size` cells. Counted from the canvas's edge
// instead, a piece off that count -- 4f537728's marker at column 11, three
// apart -- is laid in the gaps beside its own place.
ARC_HOST_DEVICE inline int tiled_source(int at, int start, int period, int size) {
    if (period <= 0) return -1;
    const int source = ((at - start) % period + period) % period;
    return source < size ? source : -1;
}

// VoteDenoise's arithmetic, in one place for the kernel and for inference.
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
ARC_HOST_DEVICE inline int8_t denoise_ground(int w, int h, At at) {
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
ARC_HOST_DEVICE inline int vote_slot(int8_t v) { return is_color(v) ? (int)v : COLOR_COUNT; }
ARC_HOST_DEVICE inline int8_t slot_value(int slot) {
    return slot == COLOR_COUNT ? (int8_t)TRANSPARENT : (int8_t)slot;
}

template <typename At, typename Put>
ARC_HOST_DEVICE inline bool vote_denoise_cells(int w, int h, int8_t bg, At ground_at, Put put) {
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

// A patch of background: its box and cell count, and whether the ground closes
// it in (a hole) rather than running off the edge of the grid.
struct blank_patch {
    object_summary box;
    bool enclosed = false;
    // A patch with no box was never found.
    bool found() const { return box.w > 0; }
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
    static void segment_for_correspondence(const grid& one_grid, segmentation_methods segmentation,
                                    vector<object_info>& info,
                                    vector<object>* objects,
                                    int step_width = 0, int step_height = 0) {
        segment_into_objects(one_grid, segmentation, info, objects, step_width, step_height);
        vector<int> reading_order(info.size());
        iota(reading_order.begin(), reading_order.end(), 0);
        stable_sort(reading_order.begin(), reading_order.end(), [&](int left, int right) {
            if (info[left].summary.count != info[right].summary.count)
                return info[left].summary.count > info[right].summary.count;
            if (info[left].summary.y0 != info[right].summary.y0)
                return info[left].summary.y0 < info[right].summary.y0;
            return info[left].summary.x0 < info[right].summary.x0;
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
        walk_colored_cells(one_grid, TRANSPARENT, [&](int, int, int) { ++colored; });
        // Nothing on the grid, so there is nothing to save and nothing to lose.
        const double plain = colored * cell_bits;
        if (plain <= 0.0) return 1.0;
        // Charge an empty segmentation the full grid cost plus a position cost.
        if (objects.empty()) return (plain + place_bits) / plain;
        // A position per object, plus the cells of each shape the first time it is
        // met. Same shape turned round is the same shape — the hash is canonical.
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

    // The block Pattern cuts a grid into. Background cells are evidence too, and
    // a complete copy plus a matching fragment can establish a period.
    static pair<int, int> pattern_block_of(const grid& one_grid) {
        int width, height;
        repeat_periods(one_grid, -2, width, height, /*must_fit_twice=*/false);
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
                                    int step_width = 0, int step_height = 0) {
        // Allow up to one object per grid cell.
        static constexpr int MAX_OBJECTS = MAX_GRID_CELLS;
        // One object found: its record always, its cells when the caller keeps them.
        auto keep = [&](const object& found, int8_t color) {
            info.push_back({ .color = color, .summary = found.summary, .shape = shape_of(found) });
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
                object piece;
                piece.cells.clear();  // every cell TRANSPARENT until the block claims it
                int lit = 0;
                for (int row = 0; row < block_height; ++row)
                    for (int col = 0; col < block_width; ++col) {
                        const int color = one_grid.data[block_top + row][block_left + col];
                        const bool is_colored = is_color(color);
                        if (is_colored) piece.cells[row][col] = (int8_t)color;
                        lit += is_colored;
                    }
                if (!lit) continue;
                piece.summary = { (int8_t)block_left, (int8_t)block_top,
                                  (int8_t)block_width, (int8_t)block_height, (int16_t)lit };
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
        each_object(one_grid, TRANSPARENT, grouping_of(segmentation), scratch,
                    [&](const object_summary& found, const int16_t* found_cells, int how_many) {
            if ((int)info.size() >= MAX_OBJECTS) return;
            // An object wears one color only if every cell of it does; -1 says several.
            int8_t color = (int8_t)one_grid.data[cell_row(found_cells[0])][cell_col(found_cells[0])];
            for (int i = 1; i < how_many && color >= 0; ++i)
                if (one_grid.data[cell_row(found_cells[i])][cell_col(found_cells[i])] != color) color = -1;
            keep(object::from_cells(one_grid, found, found_cells, how_many), color);
        });
    }

    // Canonical shape of an object: which cells of its box it fills, the same
    // number whichever way round it sits.
    static uint32_t shape_of(const object& one) {
        return canonical_shape_hash(one.summary.w, one.summary.h,
                                    [&](int row, int col) { return one.cells[row][col] != TRANSPARENT; });
    }

    // Counts differing cells when two objects are aligned at their top-left corners.
    static int count_differing_cells(const object& left, const object& right) {
        const int width = max(left.summary.w, right.summary.w);
        const int height = max(left.summary.h, right.summary.h);
        int apart = 0;
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) {
                const int here = left.color_at(left.summary.y0 + row, left.summary.x0 + col);
                const int there = right.color_at(right.summary.y0 + row, right.summary.x0 + col);
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
// an op to check a claim (7_infer.hpp).
//
// THE TWO MEASURES ARE NOT THE SAME ONE. A piece is dropped from the seeds by
// the BOX it fills, while the word that hands the repeated piece over first,
// SortBySize[0], ranks by CELLS. They usually agree and are not the same
// question, so they are kept apart here rather than quietly merged.
struct seeded_pieces {
    // Indices into the boxes handed in. -1 when nothing was.
    int repeated = -1;
    // The rest, smallest box first, which is the order an op walks them in.
    vector<int> seeds;
};

inline seeded_pieces read_seeds(const vector<object_summary>& boxes) {
    seeded_pieces reading;
    if (boxes.empty()) return reading;
    auto area_of = [&](int which) { return (int)boxes[which].w * (int)boxes[which].h; };
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
