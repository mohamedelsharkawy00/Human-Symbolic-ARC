// The rules that explain every pair by themselves, each saying a whole program
// of its own; the frame a size change is read through; and inference, which
// holds every answer about one puzzle.
#pragma once
#include "8_panes.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arc {

// A stationary object's number of enclosed spaces determines its body color.
// Include unchanged objects and ground in the checks: a changed subset alone
// could endorse a key that also selects objects which must keep another color.
struct hole_color_rule {
    static constexpr int key_count = 5; // SelectObjectsWithHoles[0..4]
    array<int8_t, key_count> colors = { -1, -1, -1, -1, -1 };
    bool held = false;
    bool holds() const { return held; }

    static hole_color_rule detect(const vector<example>& train, const tiled_pieces& tiles) {
        hole_color_rule rule;
        if (!pairs_overlay(train)) return {};
        object_walk_scratch scratch;
        bool changed = false;
        for (const auto& ex : train) {
            bool consistent = true;
            for (int r = 0; r < ex.input.height; ++r)
                for (int c = 0; c < ex.input.width; ++c)
                    if (ex.input.data[r][c] == TRANSPARENT && ex.output.data[r][c] != TRANSPARENT)
                        return {};
            each_object(ex.input, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    if (!consistent) return;
                    bool occupied[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                    int color = -1;
                    for (int i = 0; i < count; ++i) {
                        const int r = cell_row(cells[i]), c = cell_col(cells[i]);
                        occupied[r - box.row][c - box.col] = true;
                        const int target = ex.output.data[r][c];
                        if (!is_color(target) || (color >= 0 && color != target)) {
                            consistent = false;
                            return;
                        }
                        color = target;
                        changed = changed || target != ex.input.data[r][c];
                    }
                    const int key = object_hole_count(box.height, box.width,
                        [&](int r, int c) { return occupied[r][c]; });
                    if (key >= key_count || (rule.colors[key] >= 0 && rule.colors[key] != color)) {
                        consistent = false;
                        return;
                    }
                    rule.colors[key] = (int8_t)color;
                }, tiles.of(ex));
            if (!consistent) return {};
        }
        rule.held = changed;
        return rule;
    }
};

// A repeating pattern carried across the whole output, with its colors remapped.
struct motif_continuation_rule {
    bool held = false;
    recoloring color_map;
    bool holds() const { return held; }

    // Checks that the input is one tiling pattern and reads the recolor off the
    // output. Read off the repeating block's own best reading, elected or not:
    // that the grid repeats is this rule's claim, not a way of seeing objects.
    static motif_continuation_rule detect(const vector<example>& train,
                                          const correspondence_election& correspondence) {
        motif_continuation_rule rule;
        const auto* motif = correspondence.best_reading_of(segmentation_methods::minimal_repeating_set);
        if (train.empty() || !motif || motif->pairs.size() != train.size()) return rule;
        for (size_t i = 0; i < train.size(); ++i) {
            const auto& pair = motif->pairs[i];
            if (pair.edges.size() != 1 || !pair.left_over.empty()) return {};
            const auto& edge = pair.edges.front();
            if (!edge.from_input || !edge.same_shape || edge.pose || edge.scale != 1 || edge.hidden || edge.outside ||
                edge.from < 0 || edge.to < 0 || edge.from >= (int)pair.input_info.size() ||
                edge.to >= (int)pair.output_info.size()) return {};
            const auto& from = pair.input_info[edge.from].summary;
            const auto& into = pair.output_info[edge.to].summary;
            if (from.col || from.row || into.col || into.row ||
                from.width != into.width || from.height != into.height) return {};
            auto instances_only = [](const auto& objects, int representative) {
                for (int j = 0; j < (int)objects.size(); ++j)
                    if (j != representative && objects[j].instance_of != representative) return false;
                return true;
            };
            if (pair.output_info.size() < 2 ||
                !instances_only(pair.input_info, edge.from) ||
                !instances_only(pair.output_info, edge.to)) return {};
            const auto& input = train[i].input;
            const auto& output = train[i].output;
            const auto [width, height] = object_segmentation::pattern_block_of(input);
            // Pattern is input-only. A block discovered only on the output cannot
            // justify spelling Pattern with a different unit on the input.
            if (width != from.width || height != from.height) return {};
            for (int r = 0; r < input.height; ++r)
                for (int c = 0; c < input.width; ++c)
                    if (input.data[r][c] != input.data[r % height][c % width]) return {};
            for (int r = 0; r < output.height; ++r)
                for (int c = 0; c < output.width; ++c) {
                    const int was = input.data[r % height][c % width];
                    const int now = output.data[r][c];
                    // An empty cell of the motif stays empty.
                    if (is_color(was) != is_color(now)) return {};
                    if (!is_color(was)) continue;
                    if (rule.color_map[was] >= 0 && rule.color_map[was] != now) return {};
                    rule.color_map[was] = (int8_t)now;
                }
        }
        rule.held = true;
        return rule;
    }
};

// THE INPUT TILED BY ITS OWN SIZE, from where it stands onward, every copy
// taking both steps again: what a tiled frame, the input's blocks and a motif
// carried over the grid say.
inline constexpr tile_value tiled_by_its_own_size{
    .steps = { { .down = measured(tile_measure::own_height), .across = { 0 } },
               { .down = { 0 }, .across = measured(tile_measure::own_width) } },
    .start = {}, .end = { .edge = true }, .spread = tile_spread::grid };

// THE SAME CONTINUATION ALONG THE DIAGONALS. The colors cycle as the diagonals
// are crossed, the input paints part of that cycle, and the output is the cycle
// carried over every cell -- the motif above with diagonals in place of a box.
//
// THE PERIOD IS HOW MANY COLORS THERE ARE, a measure the Tile reads on each
// grid, not a choice made here: every cell steps along its diagonal, and across
// by as many cells as there are colors. Diagonals that repeat on any other
// period are a reading this has no way to say, so it does not claim them.
//
// ONE COLOR IS NO CYCLE. With a single color every diagonal is the same
// diagonal, the claim comes out true of any grid painted all one color, and it
// would say "diagonals" about a puzzle that never mentioned them.
//
// AND IT HAS TO CARRY SOMETHING. An output the input already was is the cycle
// redrawn where it already sat, which says nothing about diagonals either.
struct diagonal_continuation_rule {
    bool held = false;
    // Which diagonals: 0 those of constant row + column, stepped down and to
    // the left; 1 those of constant row - column, down and to the right.
    int8_t lean = 0;
    bool holds() const { return held; }

    // The Tile that says it, on the input's own cells, one piece each (Split[2]).
    static constexpr tile_value tile_of(int8_t lean) {
        return { .steps = { { .down = { 1 }, .across = { (int8_t)(lean ? 1 : -1) } },
                            { .down = { 0 }, .across = measured(tile_measure::color_count) } },
                 .start = { .edge = true }, .end = { .edge = true }, .spread = tile_spread::grid };
    }

    // Reads the cycle off each input and keeps the lean that redraws every
    // output exactly. Nothing about the colors is kept: which color sits on
    // which diagonal is read off the grid in hand, differently in every pair,
    // and that is what makes this a rule rather than a table.
    static diagonal_continuation_rule detect(const vector<example>& train,
                                             const correspondence_election& correspondence) {
        // The cycle is read off the cells where they sit, so the pieces have to
        // be the input's own cells in the input's own frame.
        if (train.empty() || !boxed_by_its_cells(correspondence.segmentation)) return {};
        auto redraws_every_output = [&](int8_t lean) {
            const tile_value tile = tile_of(lean);
            bool carried_something = false;
            for (const auto& pair : train) {
                // The input's own cells in reading order, each laid again where
                // the Tile lays it, a later cell's copies over an earlier one's.
                struct cell_at { int row, col, color; };
                vector<cell_at> cells;
                pair.input.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
                    cells.push_back({ row, col, color });
                });
                const int colors = color_count((int)cells.size(), [](int) { return true; },
                                               [&](int i) { return 1u << cells[(size_t)i].color; });
                if (colors < 2) return false;
                const tile_moves moves = moves_of(tile, { 1, 1, 0, 0, colors });
                grid drawn(pair.output.width, pair.output.height);
                for (const cell_at& cell : cells)
                    each_tile_offset(tile, moves, cell.row, cell.col, 1, 1, drawn.width, drawn.height, [&](int down, int across) {
                        drawn.data[cell.row + down][cell.col + across] = (int8_t)cell.color;
                        return true;
                    });
                if (drawn != pair.output) return false;
                for (int row = 0; row < pair.output.height; ++row)
                    for (int col = 0; col < pair.output.width; ++col)
                        carried_something = carried_something ||
                            row >= pair.input.height || col >= pair.input.width ||
                            pair.input.data[row][col] != pair.output.data[row][col];
            }
            return carried_something;
        };
        for (const int8_t lean : {(int8_t)0, (int8_t)1})
            if (redraws_every_output(lean)) return {true, lean};
        return {};
    }
};

// A REPEATING PICTURE WITH NOISE ON IT, AND THE OUTPUT IS THE PICTURE CLEAN.
// The input is a lattice of copies that noise has scribbled over, so no cut of
// it finds the copies -- 0607ce86 cuts into rubble every way it is read. What
// finds them is the vote: each cell takes the majority of the cells whole
// periods away. Asked by running that vote, VoteDenoise's own arithmetic, on
// every input under the ground the op reads off it (denoise_ground: empty where
// a plan deleted it, else its most common color), holding only where it draws
// every output exactly. The ground is kept in, because it is the evidence the
// period search is built on (notes/4 - To Do/VoteDenoise Needs The Background.md).
//
// AND IT HAS TO CLEAN SOMETHING. An output the input already was is a grid
// that happens to repeat, which says nothing about noise.
struct periodic_denoise_rule {
    bool held = false;
    bool holds() const { return held; }

    static periodic_denoise_rule detect(const vector<example>& train) {
        if (train.empty()) return {};
        bool cleaned_something = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const grid& input = pair.input;
            if (!input.same_size(pair.output)) return false;
            drawn = input;
            // The ground the op reads off the input, as the op reads it.
            const int8_t background = denoise_ground(input.width, input.height,
                [&](int r, int c) { return (int8_t)input.data[r][c]; });
            const bool voted = vote_denoise_cells(input.width, input.height, background,
                [&](int r, int c) -> int8_t {
                    const int8_t v = (int8_t)input.data[r][c];
                    return v == TRANSPARENT ? background : v;
                },
                [&](int r, int c, int8_t v) { drawn.data[r][c] = v; });
            cleaned_something = cleaned_something || drawn != input;
            return voted;
        });
        return { redraws && cleaned_something };
    }
};

// Every foreground color layer grows rays from its connected pieces. Colors
// are selected by cell-count rank, so the same reading survives recoloring.
// Branch order is inferred by peeling matching layers off the output.
struct ray_extension_rule {
    enum : int8_t { Orthogonal, Diagonal };
    struct branch {
        int8_t rank = 0; // ascending cell count
        int8_t ray_kind = Orthogonal;
        int8_t radius = 2;
        bool fill_interior = false;
        bool stamp_body = false;
    };
    using ranked_colors = vector<int8_t>;
    struct candidate {
        branch how;
        vector<grid> ink;
    };
    vector<branch> branches;
    bool holds() const { return !branches.empty(); }

    static bool rank_colors(const grid& input, ranked_colors& ranked) {
        const color_tally counts(input);
        ranked.clear();
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (counts[color]) ranked.push_back((int8_t)color);
        sort(ranked.begin(), ranked.end(), [&](int8_t x, int8_t y) { return counts[x] < counts[y]; });
        // Equal sizes do not establish a color-independent rank.
        for (size_t i = 1; i < ranked.size(); ++i)
            if (counts[ranked[i - 1]] == counts[ranked[i]]) return false;
        return !ranked.empty();
    }

    static bool draw_layer(const grid& input, int8_t color, const branch& how, grid& ink) {
        grid layer(input.width, input.height);
        ink = grid(input.width, input.height);
        cell_box bounds;
        for (int row = 0; row < input.height; ++row)
            for (int col = 0; col < input.width; ++col) {
                layer.data[row][col] = TRANSPARENT;
                ink.data[row][col] = TRANSPARENT;
                if (input.data[row][col] == color) {
                    layer.data[row][col] = color;
                    bounds.include_cell(row, col);
                }
            }
        if (bounds.empty()) return false;

        if (how.fill_interior) {
            bool outside[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            mark_outside(bounds.height(), bounds.width(),
                         [&](int row, int col) {
                             return layer.data[bounds.row0 + row][bounds.col0 + col] == color;
                         }, outside);
            for (int row = 0; row < bounds.height(); ++row)
                for (int col = 0; col < bounds.width(); ++col)
                    if (!outside[row][col])
                        layer.data[bounds.row0 + row][bounds.col0 + col] = color;
        }

        object_walk_scratch scratch{};
        bool drew = false;
        each_object(layer, TRANSPARENT, grouping::touching_same_color, scratch,
                    [&](const object_summary& box, const int16_t* cells, int count) {
            grid object(box.width, box.height);
            for (int i = 0; i < count; ++i) {
                const int row = cell_row(cells[i]), col = cell_col(cells[i]);
                object.data[row - box.row][col - box.col] = color;
                if (how.stamp_body) { ink.data[row][col] = color; drew = true; }
            }
            const auto seed = ray_seed_of(box.height, box.width, [&](int row, int col) { return object.data[row][col]; });
            each_extended_ray(seed, how.radius, how.ray_kind == Diagonal,
                [&](int down, int across, int) {
                    const int row = box.row + seed.row + down, col = box.col + seed.col + across;
                    if (ink.contains(row, col)) { ink.data[row][col] = color; drew = true; }
                });
        });
        return drew;
    }

    static vector<candidate> options_for(const vector<example>& train,
                                         const vector<ranked_colors>& ranked, int8_t rank) {
        vector<candidate> options;
        const auto orthogonal = row_of(node_type::ExtendOrthogonal);
        const auto diagonal = row_of(node_type::ExtendDiagonal);
        for (int radius = min(orthogonal.low, diagonal.low);
             radius <= max(orthogonal.high, diagonal.high); ++radius) {
            for (int ray_kind = Orthogonal; ray_kind <= Diagonal; ++ray_kind) {
                if (!takes_value(ray_kind == Orthogonal ? node_type::ExtendOrthogonal
                                                        : node_type::ExtendDiagonal, radius)) continue;
                for (int fill = 0; fill <= 1; ++fill)
                    for (int stamp = 0; stamp <= 1; ++stamp) {
                        candidate trying{ { rank, (int8_t)ray_kind, (int8_t)radius,
                                            fill != 0, stamp != 0 }, {} };
                        for (size_t i = 0; i < train.size(); ++i) {
                            grid ink;
                            if (!draw_layer(train[i].input, ranked[i][rank], trying.how, ink)) break;
                            trying.ink.push_back(std::move(ink));
                        }
                        if (trying.ink.size() != train.size()) continue;
                        const auto same = find_if(options.begin(), options.end(), [&](const candidate& other) {
                            return other.ink == trying.ink;
                        });
                        if (same == options.end()) options.push_back(std::move(trying));
                        // The same drawing with a longer ray: the shorter ran into the
                        // edge, and a larger test grid is where it stops too soon
                        // (ea786f4a), so the longer one takes its place, as on the line.
                        else if (same->how.ray_kind == trying.how.ray_kind &&
                                 same->how.fill_interior == trying.how.fill_interior &&
                                 same->how.stamp_body == trying.how.stamp_body)
                            same->how.radius = trying.how.radius;
                    }
            }
        }
        return options;
    }

    // A remaining color's topmost layer must explain every visible cell of
    // that color, and may paint elsewhere only beneath an already peeled layer.
    // This one check handles every position in the compositing order.
    static bool peel(const candidate& layer, const vector<example>& train,
                     const vector<ranked_colors>& ranked, const vector<grid>& covered,
                     vector<grid>& next) {
        next = covered;
        for (size_t i = 0; i < train.size(); ++i) {
            const int color = ranked[i][layer.how.rank];
            for (int row = 0; row < train[i].output.height; ++row)
                for (int col = 0; col < train[i].output.width; ++col) {
                    if (covered[i].data[row][col]) continue;
                    const bool painted = layer.ink[i].data[row][col] != TRANSPARENT;
                    const int output = train[i].output.data[row][col];
                    if (painted && output != color) return false;
                    // A cell of the color the layer did not paint is one it leaves unsaid.
                    if (output == color && !painted) return false;
                    if (painted) next[i].data[row][col] = 1;
                }
        }
        return true;
    }

    static ray_extension_rule detect(const vector<example>& train) {
        if (train.empty() || !pairs_overlay(train)) return {};
        vector<ranked_colors> ranked(train.size());
        vector<grid> covered;
        for (size_t i = 0; i < train.size(); ++i) {
            const auto& pair = train[i];
            if (!rank_colors(pair.input, ranked[i]) ||
                ranked[i].size() != ranked[0].size()) return {};
            for (int row = 0; row < pair.output.height; ++row)
                for (int col = 0; col < pair.output.width; ++col) {
                    const int color = pair.output.data[row][col];
                    if (color != TRANSPARENT && find(ranked[i].begin(), ranked[i].end(), color) == ranked[i].end())
                        return {};
                }
            covered.emplace_back(pair.output.width, pair.output.height);
            covered.back().fill(0);
        }
        vector<vector<candidate>> options;
        for (int rank = 0; rank < (int)ranked[0].size(); ++rank) {
            options.push_back(options_for(train, ranked, (int8_t)rank));
            if (options.back().empty()) return {};
        }
        ray_extension_rule found;
        auto choose = [&](auto&& self, unsigned used, const vector<grid>& mask) -> bool {
            if (found.branches.size() == options.size()) return true;
            for (size_t rank = 0; rank < options.size(); ++rank) {
                if (used & (1u << rank)) continue;
                for (const auto& option : options[rank]) {
                    vector<grid> next;
                    if (!peel(option, train, ranked, mask, next)) continue;
                    found.branches.push_back(option.how);
                    if (self(self, used | (1u << rank), next)) return true;
                    found.branches.pop_back();
                }
            }
            return false;
        };
        if (!choose(choose, 0, covered)) return {};
        reverse(found.branches.begin(), found.branches.end());
        return found;
    }
};

// The one color every cell off the ground wears, or TRANSPARENT when there is no
// figure or it wears several. What a person means by "the shape's color": read
// off the grid in hand, so a rule that asks for it names a color no example
// showed as readily as one they all did.
inline int8_t the_figure_color(const grid& one_grid, int ground) {
    const color_tally counts(one_grid, ground);
    int8_t figure = TRANSPARENT;
    for (int color = 0; color < COLOR_COUNT; ++color) {
        if (!counts[color]) continue;
        if (figure != TRANSPARENT) return TRANSPARENT;
        figure = (int8_t)color;
    }
    return figure;
}

// The output is a grid of input-sized blocks, each one a turned copy of the
// input or plain background. Every pose that draws a block is kept, not just one.
// Which blocks are copies is the correspondence's to say; this is what it adds.
struct input_blocks_rule {
    bool held = false;
    // One cell-level map shared by every copy; -1 means the color keeps itself.
    recoloring color_map;
    // OR EVERY COPY IS ITS INPUT'S NEGATIVE: figure and ground trade places.
    // The figure's color comes off each pair's own input and is kept nowhere,
    // so this is a derivation and not a table -- 0692e18c paints its copies 6,
    // 7 and 4, then asks about 3, which no table read off those pairs holds.
    bool negative = false;
    // The poses whose own figure is where the copies are: the filled blocks are
    // exactly the figure cells of the input posed that way, in every pair.
    uint8_t figure_poses = 0;
    struct layout {
        int8_t rows = 0, cols = 0;
        // Bit p is set when pose p draws the block; 0 is a block of background.
        uint8_t poses[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    };
    vector<layout> pairs;
    bool holds() const { return held; }

    // Reads the copies off the correspondence, where every block was already
    // asked whether it is the input, turned and recolored: the block cut's own
    // best reading, elected or not. What is asked here is only what places the
    // copies and what they wear -- never, a second time, whether they are copies.
    static input_blocks_rule detect(const vector<example>& train,
                                    const correspondence_election& correspondence) {
        if (train.empty()) return {};
        for (const auto& pair : train)
            if (pair.input.empty() ||
                pair.output.width % pair.input.width || pair.output.height % pair.input.height) return {};
        const auto* blocks = correspondence.best_reading_of(segmentation_methods::smallest_of_two_grids);
        if (!blocks || blocks->pairs.size() != train.size()) return {};
        // Background blocks are empty blocks: where a plan deleted the ground
        // the inputs have empty cells, and empty is what the blocks are made of.
        // A puzzle read whole, with no empty cell, has no background block.
        const int ground = TRANSPARENT;
        const bool has_ground = ranges::any_of(train, [](const example& pair) {
            for (int r = 0; r < pair.input.height; ++r)
                for (int c = 0; c < pair.input.width; ++c)
                    if (pair.input.data[r][c] == TRANSPARENT) return true;
            return false; });

        input_blocks_rule rule;
        rule.figure_poses = has_ground ? 0xFF : 0;
        // What each pair's copies wear, color by color of its input; -1 for a
        // color the input does not have, and everywhere in a pair with no copy.
        vector<recoloring> worn(train.size());
        bool copied_any = false;
        for (size_t which = 0; which < train.size(); ++which) {
            const example& pair = train[which];
            const auto& read = blocks->pairs[which];
            const int width = pair.input.width, height = pair.input.height;
            // The input is the one block the output was cut into copies of.
            if (read.input_info.size() != 1) return {};
            layout at;
            at.rows = (int8_t)(pair.output.height / height);
            at.cols = (int8_t)(pair.output.width / width);
            vector<bool> copied(read.output_info.size(), false);
            bool has_copy = false;
            for (const auto& edge : read.edges) {
                if (!edge.from_input || edge.scale != 1 || edge.to < 0 || edge.to >= (int)read.output_info.size())
                    continue;
                const auto& landing = read.output_info[edge.to].summary;
                at.poses[landing.row / height][landing.col / width] |= edge.poses;
                recoloring wears;
                pair.input.for_each_colored_cell(TRANSPARENT, [&](int, int, int was) {
                    wears[was] = edge.color_map.apply(was);
                });
                if (has_ground) wears[TRANSPARENT] = edge.color_map.apply(TRANSPARENT);
                // Copies of one rule wear one set of colors, within a pair at least.
                if (has_copy && wears != worn[which]) return {};
                worn[which] = wears;
                copied[edge.to] = true;
                has_copy = true;
            }
            copied_any = copied_any || has_copy;
            // A block with anything in it that is no copy is something else drawn
            // there; a block with nothing in it is ground, where there is one.
            if (!ranges::all_of(copied, [](bool one) { return one; })) return {};
            for (int r = 0; r < at.rows; ++r)
                for (int c = 0; c < at.cols; ++c)
                    if (!at.poses[r][c] && !has_ground) return {};
            for (int pose = 0; pose < POSE_COUNT; ++pose) {
                const auto [posed_width, posed_height] = posed_size(pose, width, height);
                bool figure = posed_width == at.cols && posed_height == at.rows;
                for (int r = 0; r < at.rows && figure; ++r)
                    for (int c = 0; c < at.cols && figure; ++c) {
                        const auto [row, col] = pose_source(pose, width, height, r, c);
                        figure = pair.input.data[row][col] != ground
                                 ? (at.poses[r][c] >> pose & 1) != 0 : at.poses[r][c] == 0;
                    }
                if (!figure) rule.figure_poses &= (uint8_t)~(1 << pose);
            }
            rule.pairs.push_back(at);
        }
        if (!copied_any) return {};

        // WHAT THE COPIES WEAR. One table every pair agrees on is what they
        // literally show, so it is asked first. Colors that change from pair to
        // pair are the evidence that each copy takes its ink from its own input,
        // and the one such reading claimed is the negative: figure and ground
        // trading places. A figure of several colors has no one negative.
        recoloring table;
        bool one_table = true;
        for (const recoloring& wears : worn)
            for (int was = TRANSPARENT; was < COLOR_COUNT; ++was) {
                if (!wears.named(was)) continue;
                one_table = one_table && (!table.named(was) || table[was] == wears[was]);
                table[was] = wears[was];
            }
        // A copy that trades its figure with empty is the negative, whatever
        // one table says: no color word says a figure and empty trading places.
        if (one_table && !table.changes(TRANSPARENT)) {
            for (int was = 0; was < COLOR_COUNT; ++was)
                rule.color_map[was] = table.changes(was) ? table[was] : recoloring::UNSET;
        } else {
            if (!has_ground) return {};
            for (size_t which = 0; which < train.size(); ++which) {
                const recoloring& wears = worn[which];
                if (wears == recoloring{}) continue;
                const int8_t figure = the_figure_color(train[which].input, ground);
                if (!is_color(figure) ||
                    (wears.named(ground) && wears[ground] != figure) ||
                    wears[figure] != ground) return {};
            }
            rule.negative = true;
        }
        rule.held = true;
        return rule;
    }
};

// Every piece slammed its own cells against one side of its own box.
// Only held when sliding every piece that way redraws every output exactly.
struct piece_slam_rule {
    // True when the rows slide sideways, false when the columns slide up or down.
    bool rows = true;
    // +1 toward the right or the bottom, -1 toward the left or the top.
    int8_t step = 0;
    bool holds() const { return step != 0; }

    // Tries all four slide directions and keeps the one that redraws every output.
    static piece_slam_rule detect(const vector<example>& train,
                                  const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // The slide happens in each piece's own box, and the cells it leaves are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !pairs_overlay(train)) return {};
        // Draws each output from its input's pieces slid that way, noting whether any
        // cell moved at all.
        auto draws_every_output = [&](bool rows, int step, bool& moved) {
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                const auto pieces = pieces_of(pair.input, correspondence.segmentation, tiles.of(pair));
                drawn = grid(pair.output.width, pair.output.height);
                bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                for (const object& piece : pieces) {
                    bool collision = false;
                    moved |= slam_cells(piece.summary.width, piece.summary.height, rows, step,
                        [&](int row, int col) { return (int)piece.data[row][col]; },
                        [&](int row, int col, int color) {
                            row += piece.summary.row;
                            col += piece.summary.col;
                            // Two pieces landing on one cell would leave paint order to decide.
                            collision = collision || painted[row][col];
                            painted[row][col] = true;
                            drawn.data[row][col] = (int8_t)color;
                        });
                    if (collision) return false;
                }
                return true;
            });
        };
        for (const bool rows : {true, false})
            for (const int step : {1, -1}) {
                bool moved = false;
                if (draws_every_output(rows, step, moved) && moved)
                    return {rows, (int8_t)step};
            }
        return {};
    }
};

// BODIES SNAPPED ONTO THEIR ANCHORS: each output is the input's bodies, turned
// and set down on the anchors their keys match, over the ground and nothing
// else -- snap_to_anchors run on the pieces the cut hands over, painted into the
// one grid the op reads them as. Held only where that draws every output
// exactly, and something other than the input in one pair at least: like the
// slam and the chain, it redraws each output whole and picks nothing, so a pair
// it leaves as it was is one it explains, not one it stands idle in.
struct snap_to_anchors_rule {
    bool held = false;
    bool holds() const { return held; }

    static snap_to_anchors_rule detect(const vector<example>& train, const correspondence_election& correspondence,
                                       const output_dimension_rule& size, const tiled_pieces& tiles) {
        if (train.empty()) return {};
        bool at_work = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            // The pieces laid into one grid as large as a grid gets, a later one over an earlier.
            int8_t cells[MAX_GRID_DIM][MAX_GRID_DIM];
            for (auto& row : cells)
                for (int8_t& cell : row) cell = (int8_t)TRANSPARENT;
            for (const object& piece : pieces_as_said(pair.input, correspondence.segmentation, tiles.of(pair)))
                piece.each_cell([&](int row, int col, int8_t color) { cells[row][col] = color; });
            const bool painted = draw_on_canvas(pair.input, size, drawn, [&](grid& canvas) {
                int8_t label[MAX_GRID_DIM][MAX_GRID_DIM];
                bool set_down = false;
                snap_to_anchors(MAX_GRID_DIM, MAX_GRID_DIM, [&](int row, int col) { return (int)cells[row][col]; },
                                label, [&](int run, int way, int key_row, int key_col, int to_row, int to_col) {
                    // The body drawn turned about its key, over what the ones before it drew.
                    bool any = false;
                    for (int row = 0; row < MAX_GRID_DIM; ++row)
                        for (int col = 0; col < MAX_GRID_DIM; ++col) {
                            if (label[row][col] != run) continue;
                            int down, across;
                            snap_turn(way, row - key_row, col - key_col, down, across);
                            const int at_row = to_row + down, at_col = to_col + across;
                            if (at_row < 0 || at_row >= MAX_GRID_DIM || at_col < 0 || at_col >= MAX_GRID_DIM) continue;
                            any = true;
                            if (canvas.contains(at_row, at_col)) canvas.data[at_row][at_col] = cells[row][col];
                        }
                    set_down = set_down || any;
                    return any;
                });
                return set_down;
            });
            if (!painted) return false;
            at_work = at_work || drawn != pair.input;
            return true;
        });
        return { redraws && at_work };
    }
};

// THE PIECES LAID OUT AS ONE DIAGONAL CHAIN, each hung off the bottom-right
// corner of the one before it, overlapping it by the single cell they share.
// Only held when replaying that chain redraws every output exactly, which is
// what makes the claim checkable rather than a guess at what the layout means:
// a chain that does not come out cell for cell is not what happened.
struct overlap_corner_rule {
    bool held = false;
    bool holds() const { return held; }

    // Lays every input's pieces into the chain and keeps the claim only if that
    // drew each output on the nose.
    static overlap_corner_rule detect(const vector<example>& train,
                                      const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // Each piece is hung off the last one's box, so the pieces have to be
        // measured in their own, and the cells no piece reaches are ground.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            train.empty()) return {};
        // OverlapCorner lays out as many objects as one list holds; a chain this
        // detector verified past that point is one the op would not draw.
        constexpr size_t most_the_op_lays_out = MAX_OBJECTS_IN_A_LIST;
        bool laid_any_piece_somewhere_new = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            const auto pieces = pieces_of(pair.input, correspondence.segmentation, tiles.of(pair));
            // One piece put in the corner says nothing about chaining.
            if (pieces.size() < 2 || pieces.size() > most_the_op_lays_out) return false;
            // The chain runs left to right, ties top to bottom -- the order the
            // op itself walks the objects in, so the reading and the word agree
            // about which piece is first.
            vector<const object*> chain;
            for (const object& piece : pieces) chain.push_back(&piece);
            stable_sort(chain.begin(), chain.end(), [](const object* one, const object* other) {
                return one->summary.col != other->summary.col
                     ? one->summary.col < other->summary.col
                     : one->summary.row < other->summary.row;
            });
            // Tied corners leave the order to the op's list, which we can't see.
            for (size_t i = 1; i < chain.size(); ++i)
                if (same_corner(chain[i]->summary, chain[i - 1]->summary)) return false;
            drawn = grid(pair.output.width, pair.output.height);
            // The head of the chain sits in the corner; each piece after it puts
            // its top-left on the last one's bottom-right and paints over it,
            // which is the paint order the closing Union gives them for free.
            int x = 0, y = 0;
            for (const object* piece : chain) {
                if (!paint_piece(drawn, *piece, y, x)) return false;
                laid_any_piece_somewhere_new = laid_any_piece_somewhere_new ||
                    x != piece->summary.col || y != piece->summary.row;
                x += piece->summary.width - 1;
                y += piece->summary.height - 1;
            }
            return true;
        });
        // Pieces already sitting in the chain were never laid into one.
        return { redraws && laid_any_piece_somewhere_new };
    }
};

// PIECES SLID UP AGAINST THE ONES THAT STAY. One colour is already where it
// belongs and the other closes the gap until the two touch. That is what
// AlignToCanvas does, and this reads the same movement off the pairs: the still
// pieces are painted, the box their cells make is the whole of what the op gets
// to see of the canvas, and every other piece lands flush against that box,
// along whichever axis it stood further apart on. Only held when replaying it
// redraws every output exactly, which is what makes the claim checkable rather
// than a guess at where the pieces were headed.
//
// TWO COLOURS, BECAUSE A PICK NAMES A PIECE BY ITS COLOUR. The sentence has to
// say which pieces stay and which slide, and SelectColor is the only word that
// names a piece by what it is -- so a grid where three colours share the work
// is a movement with no sentence, and is not claimed here.
//
// shift_to_touch is shared with the op, so both choose the same landing.
struct slide_to_touch_rule {
    int8_t stays  = -1;   // the colour that is already where it belongs
    int8_t slides = -1;   // the colour that closes the gap until they touch
    bool holds() const { return is_color(stays) && is_color(slides); }

    static slide_to_touch_rule detect(const vector<example>& train,
                                      const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // The pieces slide in grid coordinates and the cells they leave behind
        // are ground, so the two grids have to lie over one another; and the
        // boxes measured here are the boxes Trim hands the op.
        if (!boxed_by_its_cells(correspondence.segmentation) ||
            !pairs_overlay(train)) return {};

        // The colours the pieces wear: two of them, the same two in every pair,
        // or there is no pair of picks that can say this.
        int8_t colors[2] = { -1, -1 };
        for (const auto& pair : train) {
            vector<object_segmentation::object_info> info;
            object_segmentation::segment_for_correspondence(pair.input,
                                                            correspondence.segmentation, info, nullptr,
                                                                0, 0, tiles.of(pair));
            // One list is all the op is ever handed, so a grid cut into more
            // pieces than that is one the word could not draw anyway.
            if (info.empty() || info.size() > MAX_OBJECTS_IN_A_LIST) return {};
            for (const auto& piece : info) {
                // A piece of several colours answers to both picks and to neither.
                if (!is_color(piece.color)) return {};
                if (piece.color == colors[0] || piece.color == colors[1]) continue;
                if (!is_color(colors[0]))      colors[0] = piece.color;
                else if (!is_color(colors[1])) colors[1] = piece.color;
                else return {};
            }
        }
        if (!is_color(colors[0]) || !is_color(colors[1])) return {};

        // Draws each output from its input's pieces, one colour standing still
        // and the other sliding up against it, noting whether anything moved at
        // all.
        auto draws_every_output = [&](int8_t stays, int8_t slides, bool& moved) {
            return redraws_every_output(train, [&](const example& pair, grid& drawn) {
                vector<object_segmentation::object_info> info;
                vector<object> pieces;
                object_segmentation::segment_for_correspondence(pair.input,
                                                                correspondence.segmentation, info, &pieces,
                                                                0, 0, tiles.of(pair));
                drawn = grid(pair.output.width, pair.output.height);
                // WHAT THE OP SEES OF THE CANVAS is the box around everything
                // already painted, read once, before the first piece slides --
                // so what slides is measured against the pieces that stay, and
                // never against another piece that is still on its way.
                cell_box target;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != stays) continue;
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.row, pieces[i].summary.col)) return false;
                    target.include(pieces[i].summary.box());
                }
                // An empty canvas is nothing to touch, and the op answers that
                // by leaving every piece exactly where it stood.
                if (target.empty()) return false;
                bool anything_slid = false;
                for (size_t i = 0; i < pieces.size(); ++i) {
                    if (info[i].color != slides) continue;
                    anything_slid = true;
                    const auto [down, across] = shift_to_touch(pieces[i].summary.box(), target);
                    if (!paint_piece(drawn, pieces[i], pieces[i].summary.row + down,
                                     pieces[i].summary.col + across)) return false;
                    moved = moved || down != 0 || across != 0;
                }
                return anything_slid;
            });
        };

        // Which colour stays is the reading's to find, not to assume.
        for (const int which : { 0, 1 }) {
            bool moved = false;
            const int8_t stays = colors[which], slides = colors[1 - which];
            // Pieces already touching were never slid up against anything.
            if (draws_every_output(stays, slides, moved) && moved) return { stays, slides };
        }
        return {};
    }
};

// THE ORDER THE DIRECTIONS ARE ASKED IN, so the order a tie between two that say as
// much is broken in: the four sides before the diagonals, the plainer move
// first, each clockwise from up.
constexpr int direction_asked(int at) { return at < 4 ? 2 * at : 2 * (at - 4) + 1; }

// PIECES THAT SLIDE TO AN EDGE, AND THE REST LEFT WHERE IT LIES. Every piece,
// or one color's pieces, go one of the eight ways until they meet the edge of
// the canvas, passing over whatever is in the way, and land over or under what
// stays. That is MoveToEdge on a pick, so the replay steps each piece as the
// op does (steps_to_edge), and how far is the canvas's to say -- it differs
// from piece to piece, and no count of cells is written into the sentence.
//
// THE ELECTED CUT FIRST, THEN THE OTHERS. The election keeps the shortest
// reading, which can join the pieces that slide to the ones that stay: in
// 3618c87e it joins each blue dot to the gray under it and reads a recolor.
// Plainest first after that: every piece before one color, the sides before
// the diagonals, over before under.
//
// A SLIDE THE CANVAS HIDES IS NO SLIDE. Where the pieces laid where they lie
// already draw every output, nothing has to go anywhere -- 1cf80156 is a crop,
// and every piece pushed up and then cropped reads the same, falsely.
struct edge_slide_rule {
    segmentation_methods segmentation = segmentation_methods::unset;
    // The color whose pieces slide, or -1 when every piece does.
    int8_t color = -1;
    int8_t slide_direction = direction::up;
    // Laid over what stays, as Union lays it, or under it, as Underlay does.
    bool over = true;
    bool held = false;
    bool holds() const { return held; }

    // SelectColor picks a piece that carries the color anywhere.
    bool slides(const object& piece) const { return !is_color(color) || (palette(piece) >> color & 1) != 0; }

    // One input's output as the sentences paint it: what stays, where it lies,
    // then each sliding piece in the cut's own order, later over earlier.
    bool draw(const grid& input, const vector<object>& pieces, const output_dimension_rule& size,
              grid& drawn) const {
        return draw_on_canvas(input, size, drawn, [&](grid& canvas) {
            for (const object& piece : pieces)
                if (!slides(piece) && !paint_piece(canvas, piece, piece.summary.row, piece.summary.col))
                    return false;
            const direction_step step = step_of(slide_direction);
            for (const object& piece : pieces) {
                if (!slides(piece)) continue;
                const int steps = steps_to_edge(piece.summary.box(), canvas.width, canvas.height, slide_direction);
                if (!paint_piece(canvas, piece, piece.summary.row + steps * step.down,
                                 piece.summary.col + steps * step.across, over)) return false;
            }
            return true;
        });
    }

    static edge_slide_rule detect(const vector<example>& train, segmentation_methods elected,
                                  const output_dimension_rule& size, const tiled_pieces& tiles) {
        if (train.empty()) return {};
        vector<segmentation_methods> cuts{ elected };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::all_cells_of_one_color,
                                segmentation_methods::connected_multicolor })
            if (cut != elected) cuts.push_back(cut);
        for (const auto cut : cuts) {
            if (!boxed_by_its_cells(cut)) continue;
            // Each pair cut once; every reading below is asked of the same pieces.
            vector<vector<object>> pieces;
            uint16_t colors = 0;
            for (const example& pair : train) {
                pieces.push_back(pieces_as_said(pair.input, cut, tiles.of(pair)));
                for (const object& piece : pieces.back()) colors = (uint16_t)(colors | palette(piece));
            }
            bool as_they_lie = true;
            for (size_t i = 0; i < train.size() && as_they_lie; ++i) {
                grid drawn;
                as_they_lie = draw_on_canvas(train[i].input, size, drawn, [&](grid& canvas) {
                    for (const object& piece : pieces[i])
                        if (!paint_piece(canvas, piece, piece.summary.row, piece.summary.col)) return false;
                    return true;
                }) && drawn == train[i].output;
            }
            if (as_they_lie) continue;
            for (int color = -1; color < COLOR_COUNT; ++color) {
                if (color >= 0 && !(colors >> color & 1)) continue;
                for (int at = 0; at < direction::count; ++at)
                    for (bool over : { true, false }) {
                        // With every piece sliding nothing stays to lie under.
                        if (!over && color < 0) continue;
                        const edge_slide_rule trying{ cut, (int8_t)color, (int8_t)direction_asked(at), over, true };
                        bool every = true;
                        for (size_t i = 0; i < train.size() && every; ++i) {
                            grid drawn;
                            every = trying.draw(train[i].input, pieces[i], size, drawn) &&
                                    drawn == train[i].output;
                        }
                        if (every) return trying;
                    }
            }
        }
        return {};
    }
};

// ONE FIGURE COPIED TO THE OTHER PIECES. The largest connected figure supplies
// the drawing, the others supply frames, and the examples agree on one anchor.
// Try the input's pieces first, then the already-elected panes without their
// separators. No color, template size, destination count or coordinates are
// baked into the claim. Trim gives the op the same tight frames read here.
// The op itself accepts any selection of source figures;
// this reading only names the unambiguous largest one.
struct copy_to_rule {
    bool held = false;
    bool panes = false;
    int8_t anchor = 4;
    bool holds() const { return held; }

    static vector<object> pieces(const grid& input, const pane_split_facts& split, bool use_panes, tile_size tiles = {}) {
        grid source = input;
        if (use_panes) {
            vector<pane_split_facts::pane> frames;
            if (!split.panes_of(input, frames) || frames.size() > MAX_OBJECTS_IN_A_LIST) return {};
            for (int r = 0; r < source.height; ++r)
                for (int c = 0; c < source.width; ++c) source.data[r][c] = TRANSPARENT;
            for (const auto& frame : frames)
                for (int r = frame.top; r < frame.top + frame.height; ++r)
                    for (int c = frame.left; c < frame.left + frame.width; ++c)
                        source.data[r][c] = input.data[r][c];
        }
        return pieces_as_said(source, segmentation_methods::connected_multicolor, tiles);
    }

    static int source_index(const vector<object>& objects) {
        if (objects.size() < 2 || objects.size() > MAX_OBJECTS_IN_A_LIST) return -1;
        int largest = 0;
        for (int i = 1; i < (int)objects.size(); ++i)
            if (objects[i].summary.count > objects[largest].summary.count) largest = i;
        for (int i = 0; i < (int)objects.size(); ++i)
            if (i != largest && objects[i].summary.count == objects[largest].summary.count) return -1;
        return largest;
    }

    static grid draw(const grid& input, const vector<object>& objects, int source, int anchor) {
        grid drawn = input;
        const auto& pattern = objects[source];
        for (int i = 0; i < (int)objects.size(); ++i) {
            if (i == source) continue;
            const auto& target = objects[i].summary;
            const int row = aligned_copy_start(pattern.summary.height, target.row, target.height, anchor / 3);
            const int col = aligned_copy_start(pattern.summary.width, target.col, target.width, anchor % 3);
            paint_piece(drawn, pattern, row, col, /*over=*/true, /*clip=*/true);
        }
        return drawn;
    }

    static copy_to_rule detect(const vector<example>& train,
                               const pane_split_facts& split, const tiled_pieces& tiles) {
        if (train.empty()) return {};
        bool changed = false;
        for (const auto& pair : train) {
            if (pair.input.width != pair.output.width || pair.input.height != pair.output.height) return {};
            changed |= pair.input != pair.output;
        }
        if (!changed) return {};
        for (bool use_panes : { false, true }) {
            if (use_panes && !split.holds()) continue;
            vector<vector<object>> cut;
            vector<int> sources;
            bool readable = true;
            for (const auto& pair : train) {
                cut.push_back(pieces(pair.input, split, use_panes, tiles.of(pair)));
                sources.push_back(source_index(cut.back()));
                if (sources.back() < 0) { readable = false; break; }
            }
            if (!readable) continue;
            for (int anchor = 0; anchor < 9; ++anchor) {
                bool every = true;
                for (size_t i = 0; i < train.size() && every; ++i)
                    every = draw(train[i].input, cut[i], sources[i], anchor) == train[i].output;
                if (every) return { true, use_panes, (int8_t)anchor };
            }
        }
        return {};
    }
};

// THE SEEDS THE OP WILL BE HANDED. The DSL fills its seed list off the input
// before a program runs (17_fixed_dsl.hpp), and this says the same list from
// this side of the fence, so a replay below can be checked against what the op
// will really do. The rule itself is not restated -- 4_objects.hpp's read_seeds
// is the one copy of it -- only the grid it is asked about.
struct marker_piece {
    object_summary summary;
    int8_t color = 0;
};

inline vector<marker_piece> seeds_the_op_will_see(const grid& input, tile_size tiles = {}) {
    vector<marker_piece> pieces;
    object_walk_scratch scratch;
    each_object(input, TRANSPARENT, grouping::touching_same_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int found) {
        if (found <= 0) return;
        pieces.push_back({ box, (int8_t)input.data[cell_row(cells[0])][cell_col(cells[0])] });
    }, tiles);
    vector<object_summary> boxes;
    for (const marker_piece& piece : pieces) boxes.push_back(piece.summary);
    vector<marker_piece> seeds;
    for (int which : read_seeds(boxes).seeds) seeds.push_back(pieces[which]);
    return seeds;
}

// THE TEMPLATE STAMPED ALONG EVERY MARKER. Each marker stands away from the
// template and so says a direction; the template lands flush against that
// marker, then repeats at the same step, wearing the marker's colour, until it
// has left the grid. The input stays underneath it all. Only held when
// replaying that redraws every output exactly, which is what makes the claim
// checkable rather than a guess at what the markers meant.
//
// OR EVERY STAMP WEARS ONE COLOUR, AND THE INPUT IS LAID BACK OVER THEM, so a
// stamp shows only where the input had ground. That is a run carried on to the
// edge in a colour of its own: 0b17323b's dots step on at their own spacing
// until they leave the grid, and the new ones come out red. The colour is read
// off the first stamp that shows and held to every other, as the interior fill
// reads its own. The markers' colours are asked first, being the plainer claim.
struct project_on_seeds_rule {
    bool held = false;
    // The one colour every stamp wears when the input is laid back over them;
    // not a colour when each stamp wears its marker's, with the input beneath.
    int8_t worn = TRANSPARENT;
    bool holds() const { return held; }

    // Calls land(row, col, marker colour) for each cell of each stamp the words
    // lay on this pair's canvas, and says how many stamps they laid -- none when
    // there is no template or no marker to lay one along.
    template <class Land>
    static int stamp(const example& pair, segmentation_methods cut, Land land, tile_size tiles) {
        const auto pieces = pieces_of(pair.input, cut, tiles);
        if (pieces.empty()) return 0;
        // The template is whichever piece SortBySize[0] puts first, and that
        // word ranks by cells -- not the box that picked the markers. The op
        // reads the two off different lists, so this has to as well.
        // Tied for most cells: the op's list order picks, which we can't see.
        const object& tpl = pieces[0];
        if (pieces.size() > 1 && pieces[1].summary.count == tpl.summary.count) return 0;
        // ProjectOnSeeds hands back the copy of the input it draws on plus one
        // object per stamp, and they share one list, so the drawing runs out one
        // stamp before the list does.
        constexpr int most_the_op_stamps = MAX_OBJECTS_IN_A_LIST - 1;
        int stamped = 0;
        for (const marker_piece& marker : seeds_the_op_will_see(pair.input, tiles))
            project_along(tpl.summary.box(), marker.summary.box(),
                          pair.input.width, pair.input.height, [&](int at_row, int at_col) {
                if (stamped >= most_the_op_stamps) return false;
                // A stamp hanging over the edge is drawn as far as it fits.
                for (int r = 0; r < tpl.summary.height; ++r)
                    for (int c = 0; c < tpl.summary.width; ++c) {
                        const int row = at_row + r, col = at_col + c;
                        if (tpl.data[r][c] != TRANSPARENT && pair.input.contains(row, col))
                            land(row, col, marker.color);
                    }
                ++stamped;
                return true;
            });
        return stamped;
    }

    static project_on_seeds_rule detect(const vector<example>& train,
                                        const correspondence_election& correspondence,
                                        const tiled_pieces& tiles) {
        // The template is handed over in the frame its cells make, which is
        // what the Trim in front of the word gives it. Stamping paints over a
        // copy of the input, so a canvas of another size is a different drawing
        // than the one replayed here.
        const segmentation_methods cut = correspondence.segmentation;
        if (!boxed_by_its_cells(cut) || !pairs_overlay(train)) return {};
        // Nothing stamped is the input handed straight back, which every other
        // reading already says without a word this long.
        const bool in_the_markers_colors = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = pair.input;
            return stamp(pair, cut, [&](int row, int col, int8_t color) {
                drawn.data[row][col] = color;
            }, tiles.of(pair)) > 0;
        });
        if (in_the_markers_colors) return { true, TRANSPARENT };

        // With the input laid back over the stamps, its ground shows wherever no
        // stamp went, and the outputs' elected ground is what fills it -- or,
        // where a plan deleted the ground, nothing does until the plan's fill.
        int8_t worn = TRANSPARENT;
        bool one_color = true;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = grid(pair.input.width, pair.input.height);
            pair.input.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
                drawn.data[row][col] = (int8_t)color;
            });
            const int stamped = stamp(pair, cut, [&](int row, int col, int8_t) {
                if (pair.input.data[row][col] != TRANSPARENT) return;
                const int8_t shown = (int8_t)pair.output.data[row][col];
                if (!is_color(worn)) worn = shown;
                one_color = one_color && shown == worn;
                drawn.data[row][col] = worn;
            }, tiles.of(pair));
            return stamped > 0 && one_color;
        });
        // A stamp in the ground's own colour draws nothing at all.
        if (!redraws || !is_color(worn) || worn == TRANSPARENT) return {};
        return { true, worn };
    }
};

// THE WAYS A BRIDGE IS ASKED IN, the narrowest first -- one direction before
// both; only across the grid, then the nearest, then every one -- so the bars
// are said no wider than the pairs show them: a699fb00 bridges along rows, and
// nothing in it says a column would be bridged too; 0d87d2a6's marks face each
// other from opposite edges, and nothing in its pairs says two marks on one
// edge would be joined, as the nearest joins them on its test.
constexpr int bridge_ways_narrowest_first[BRIDGE_WAYS] = { 7, 8, 6, 1, 2, 0, 4, 5, 3 };
static_assert(row_of(node_type::Bridge).high + 1 == BRIDGE_VALUES, "Bridge takes one value for each way it reaches");

// THE GAP BETWEEN TWO PIECES OF ONE COLOUR, FILLED, and laid under what is
// already there, so a bar that crosses a wall goes behind it. That is what
// Bridge does, and this reads it off the pairs by doing it: the input's pieces,
// cut the way Split cuts them and in the order Split hands them over, each
// reaching for its nearest partner along a row or a column. Only held when
// replaying that redraws every output exactly.
//
// THE CUT IS THE ELECTED ONE WHEN IT CAN SAY THIS, and otherwise the first
// connected cut that can -- a rule about one way of cutting is read off that
// cut, elected or not. Colors is never asked: it hands one piece per colour,
// and a colour has no partner of its own to reach for. 070dd51e elects Colors
// and 22eb0ac0 the whole grid, and both are bridges under Split[1].
struct bridge_rule {
    bool held = false;
    segmentation_methods segmentation = segmentation_methods::unset;
    // How the bridge reaches, Bridge's value (bridge_bars).
    int8_t way = 0;
    bool holds() const { return held; }

    static bridge_rule detect(const vector<example>& train,
                              const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // The bars fill cells that were ground, and the rest of the grid stays
        // where it was, so the two grids lie over one another.
        if (!pairs_overlay(train)) return {};
        vector<segmentation_methods> cuts{ correspondence.segmentation };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::connected_multicolor })
            if (cut != correspondence.segmentation) cuts.push_back(cut);
        // Each cut the bridge can be read under, each way it can reach, the
        // narrowest first.
        for (const auto cut : cuts) {
            if (cut != segmentation_methods::connected_monochrome &&
                cut != segmentation_methods::connected_multicolor) continue;
            for (const int way : bridge_ways_narrowest_first) {
                bool bridged = false;
                const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    // The pieces as Split hands them over: the one walk, in its order.
                    const vector<object> pieces = pieces_as_said(pair.input, cut, tiles.of(pair));
                    // One list is all the op is handed, and the bars take one place
                    // more in it, all of them one drawing.
                    if (pieces.empty() || pieces.size() >= MAX_OBJECTS_IN_A_LIST) return false;
                    drawn = grid(pair.output.width, pair.output.height);
                    bool painted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
                    for (const object& piece : pieces)
                        piece.each_cell([&](int row, int col, int8_t color) {
                            drawn.data[row][col] = color;
                            painted[row][col] = true;
                        });
                    // Each bar is laid under whatever is already drawn. Every
                    // piece is picked, as SelectAll picks them before Bridge.
                    hand_answers fresh;
                    bridge_bars(way, pair.output.width, pair.output.height,
                                partner_pieces(pieces, vector<bool>(pieces.size(), true), fresh),
                                [&](const cell_box& bar, int8_t color) {
                        for (int row = bar.row0; row <= bar.row1; ++row)
                            for (int col = bar.col0; col <= bar.col1; ++col) {
                                if (!drawn.contains(row, col) || painted[row][col]) continue;
                                drawn.data[row][col] = color;
                                painted[row][col] = true;
                                bridged = true;
                            }
                        return true;
                    });
                    return true;
                });
                // No bar laid anywhere is the input handed straight back.
                if (redraws && bridged) return { true, cut, (int8_t)way };
            }
        }
        return {};
    }
};

// THE MARKS AS LINES, REPEATED DOWN THE GRID. Every mark becomes a line
// across the grid, and the lines together come round again, the same gap
// after them as between them, to the far edge. That is two Tiles: `lines`,
// which lays a mark's first column across the grid, then, the lines made one
// drawing, `lattice`, which lays it down by its own height and its gap; with
// NormalizeOrientation[0] in front, a pair lying on its side is read turned,
// so its lines cross the short way. Only held when replaying those words
// redraws every output exactly, which is what makes the lattice a rule read
// off each input rather than rows remembered from the pairs.
//
// THE CUT IS THE ELECTED ONE WHEN IT CAN SAY THIS, and otherwise the first
// that can, as for the bridge: a mark is whatever the word hands over, and it
// is replayed in the order the word hands it, since a later line paints over
// an earlier one where two come round on the same row.
//
// A TURN IS WORN ONLY WHEN IT DOES WORK. The grid as given is asked first, so
// pairs that all stand upright never say it, and a pair lying on its side only
// says it when its lines cross the short way.
//
// AND IT HAS TO LAY SOMETHING. An output the input already was is marks
// sitting where their own lattice would put them, which says nothing about one.
struct mark_lattice_rule {
    bool held = false;
    segmentation_methods segmentation = segmentation_methods::unset;
    // Whether a grid lying on its side is read turned: NormalizeOrientation[0].
    bool portrait = false;
    bool holds() const { return held; }

    // The Tiles that say it: each mark's first column laid across the grid
    // both ways, a step at a time; then those lines, made one drawing
    // (UnionList), laid down from where it stands onward, by its own height and
    // the gap between its lines -- the whole thing again, its gap after it.
    static constexpr tile_value lines{ .steps = { { .down = { 0 }, .across = { 1 } } },
                                       .start = { .edge = true }, .end = { .edge = true }, .spread = tile_spread::lines };
    static constexpr tile_value lattice{
        .steps = { { .down = measured(tile_measure::own_height, tile_measure::row_gap), .across = { 0 } } },
        .start = {}, .end = { .edge = true }, .spread = tile_spread::lines };

    // What the words draw on one input: the cut's marks, trimmed and read the
    // way the pair is turned, each laid as its line, the lines laid again
    // wherever the lattice puts them, the rest left empty. False where the
    // lines would not fit the one list the op is handed.
    static bool draw(const grid& input, segmentation_methods cut, bool portrait, grid& drawn, tile_size tiles = {}) {
        const vector<object> marks = pieces_as_said(input, cut, tiles);
        // Read turned, the marks' rows are their columns, and the other way round.
        const bool turned = portrait && lies_on_its_side(input.width, input.height);
        auto top = [&](const object& mark) { return (int)(turned ? mark.summary.col : mark.summary.row); };
        auto left = [&](const object& mark) { return (int)(turned ? mark.summary.row : mark.summary.col); };
        auto tall = [&](const object& mark) { return (int)(turned ? mark.summary.width : mark.summary.height); };
        auto wide = [&](const object& mark) { return (int)(turned ? mark.summary.height : mark.summary.width); };
        const int rows = turned ? input.width : input.height;
        const int across = turned ? input.height : input.width;
        // The lines, as the one drawing UnionList makes of them, and the rows it spans.
        int8_t picture[MAX_GRID_DIM][MAX_GRID_DIM];
        for (auto& row : picture) std::fill(std::begin(row), std::end(row), (int8_t)TRANSPARENT);
        int first_row = rows, last_row = -1, laid_lines = 0;
        for (const object& mark : marks) {
            const tile_moves spread = moves_of(lines, {});
            const auto [line_rows, line_columns] = copy_size(spread, tall(mark), wide(mark));
            each_tile_offset(lines, spread, top(mark), left(mark), line_rows, line_columns, across, rows,
                             [&](int, int sideways) {
                ++laid_lines;
                for (int row = 0; row < line_rows; ++row)
                    for (int col = 0; col < line_columns; ++col) {
                        const int8_t color = turned ? mark.data[col][row] : mark.data[row][col];
                        const int at_row = top(mark) + row, at_col = left(mark) + sideways + col;
                        if (color != TRANSPARENT && at_row >= 0 && at_row < rows && at_col >= 0 && at_col < across)
                            picture[at_row][at_col] = color;
                    }
                return true;
            });
            first_row = min(first_row, top(mark));
            last_row = max(last_row, top(mark) + tall(mark) - 1);
        }
        if (last_row < first_row || laid_lines > MAX_OBJECTS_IN_A_LIST) return false;
        const int height = last_row - first_row + 1;
        const tile_moves down = moves_of(lattice, measures_of(lattice, height, across, [&](int row, int col) {
            return picture[first_row + row][col] != TRANSPARENT; }, 0));
        const auto [copy_rows, copy_columns] = copy_size(down, height, across);
        drawn = grid(input.width, input.height);
        auto at = [&](int row, int col) -> int8_t& {
            return turned ? drawn.data[col][row] : drawn.data[row][col];
        };
        each_tile_offset(lattice, down, first_row, 0, copy_rows, copy_columns, across, rows, [&](int lower, int) {
            for (int row = 0; row < copy_rows && first_row + lower + row < rows; ++row)
                for (int col = 0; col < copy_columns; ++col)
                    if (const int8_t color = picture[first_row + row][col]; color != TRANSPARENT)
                        at(first_row + lower + row, col) = color;
            return true;
        });
        return true;
    }

    static mark_lattice_rule detect(const vector<example>& train,
                                    const correspondence_election& correspondence, const tiled_pieces& tiles) {
        // Lines go down on an empty canvas the size of the input, and whatever
        // they miss stays empty, for the plan to fill.
        if (!pairs_overlay(train)) return {};
        vector<segmentation_methods> cuts{ correspondence.segmentation };
        for (const auto cut : { segmentation_methods::connected_monochrome,
                                segmentation_methods::connected_multicolor,
                                segmentation_methods::all_cells_of_one_color })
            if (cut != correspondence.segmentation) cuts.push_back(cut);
        for (const auto cut : cuts) {
            if (!boxed_by_its_cells(cut)) continue;
            for (const bool portrait : { false, true }) {
                bool laid = false;
                const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                    if (!draw(pair.input, cut, portrait, drawn, tiles.of(pair)))
                        return false;
                    laid = laid || drawn != pair.input;
                    return true;
                });
                if (redraws && laid) return { true, cut, portrait };
            }
        }
        return {};
    }
};

// A shape keeps its boundary and spreads its own rarest color over its core.
// Read each marker from the input, so neither colors nor shapes are memorized.
struct interior_spread_rule {
    bool held = false;
    bool holds() const { return held; }

    static interior_spread_rule detect(const vector<example>& train, const tiled_pieces& tiles) {
        if (train.empty() || !pairs_overlay(train)) return {};
        bool changed = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = pair.input;
            object_walk_scratch scratch;
            int pieces = 0;
            each_object(pair.input, TRANSPARENT, grouping::touching_any_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    ++pieces;
                    const object piece = object::from_cells(pair.input, box, cells, count);
                    const int8_t color = object_paint_color(box.height, box.width, COLOR_COUNT,
                        [&](int row, int col) { return (int8_t)piece.data[row][col]; });
                    piece.each_cell([&](int row, int col, int8_t was) {
                        if (on_object_edge(row, col, [&](int r, int c) { return piece.holds(r, c); })) return;
                        drawn.data[row][col] = color;
                        changed = changed || color != was;
                    });
                }, tiles.of(pair));
            return pieces > 0 && pieces <= MAX_OBJECTS_IN_A_LIST;
        });
        return {redraws && changed};
    }
};

// Every enclosed space takes one color, independently of the boundary's shape.
// Read that color from the interiors, then check the whole drawing, including
// unchanged bodies and the elected outside color. No exact-shape keys are needed.
struct interior_fill_rule {
    int8_t color = -1;
    bool holds() const { return is_color(color); }

    static interior_fill_rule detect(const vector<example>& train, const tiled_pieces& tiles) {
        if (train.empty() || !pairs_overlay(train)) return {};
        interior_fill_rule rule;
        bool changed = false;
        const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            drawn = grid(pair.input.width, pair.input.height);
            object_walk_scratch scratch;
            bool consistent = true;
            int pieces = 0;
            each_object(pair.input, TRANSPARENT, grouping::touching_any_color, scratch,
                [&](const object_summary& box, const int16_t* cells, int count) {
                    if (!consistent) return;
                    ++pieces;
                    const object piece = object::from_cells(pair.input, box, cells, count);
                    bool outside[MAX_GRID_DIM][MAX_GRID_DIM];
                    mark_outside(box.height, box.width,
                        [&](int r, int c) { return is_color(piece.data[r][c]); }, outside);
                    // Separate marker objects inside a hole keep their own
                    // colors. They are not evidence for the interior's fill.
                    for (int r = 0; r < box.height; ++r)
                        for (int c = 0; c < box.width; ++c) {
                            const int y = box.row + r, x = box.col + c;
                            if (is_color(piece.data[r][c]) || outside[r][c] ||
                                pair.input.data[y][x] != TRANSPARENT) continue;
                            const int fill = pair.output.data[y][x];
                            if (!is_color(fill) || (rule.holds() && rule.color != fill)) {
                                consistent = false;
                                return;
                            }
                            rule.color = (int8_t)fill;
                            changed = changed || fill != pair.input.data[y][x];
                        }
                    // Replay in object order, as Union does: a later marker
                    // can restore its body over an earlier object's fill.
                    for (int r = 0; r < box.height; ++r)
                        for (int c = 0; c < box.width; ++c) {
                            const int y = box.row + r, x = box.col + c;
                            if (is_color(piece.data[r][c]))
                                drawn.data[y][x] = piece.data[r][c];
                            else if (!outside[r][c] && rule.holds())
                                drawn.data[y][x] = rule.color;
                        }
                }, tiles.of(pair));
            return consistent && pieces > 0 && pieces <= MAX_OBJECTS_IN_A_LIST;
        });
        return redraws && changed ? rule : interior_fill_rule{};
    }
};

// A grid of trimmed connected pieces, selected by removing its most common colors.
// Only held when the shared Layout and ScaleDown geometry redraws every output.
struct ranked_layout_rule {
    bool held = false;
    int8_t discarded_colors = 0;
    int8_t split_mode = 0;
    int8_t cell_size = 0;
    bool holds() const { return held; }

    bool draw(const example& pair, grid& drawn) const {
        if (!holds() || pair.input.empty() || discarded_colors < 0 || discarded_colors > COLOR_COUNT ||
            cell_size < 2 || cell_size > 10 || (split_mode != 0 && split_mode != 1)) return false;

        const color_tally frequencies(pair.input);
        bool removed[COLOR_COUNT] = {};
        for (int rank = 0; rank < discarded_colors; ++rank) {
            int common = -1;
            for (int color = 0; color < COLOR_COUNT; ++color) {
                if (removed[color] || frequencies[color] == 0) continue;
                if (common < 0 || frequencies[color] > frequencies[common]) common = color;
            }
            if (common >= 0) removed[common] = true;
        }

        grid remaining(pair.input.width, pair.input.height);
        for (int row = 0; row < pair.input.height; ++row)
            for (int col = 0; col < pair.input.width; ++col) {
                const int color = pair.input.data[row][col];
                if (is_color(color) && !removed[color])
                    remaining.data[row][col] = (int8_t)color;
            }

        vector<object> pieces;
        object_walk_scratch scratch;
        const grouping groups = split_mode == 0 ? grouping::touching_any_color
                                                 : grouping::touching_same_color;
        each_object(remaining, TRANSPARENT, groups, scratch,
            [&](const object_summary& box, const int16_t* cells, int count) {
                if ((int)pieces.size() < MAX_OBJECTS_IN_A_LIST)
                    pieces.push_back(object::from_cells(remaining, box, cells, count));
            });
        if (pieces.empty()) return false;

        int8_t row_of[MAX_OBJECTS_IN_A_LIST], col_of[MAX_OBJECTS_IN_A_LIST];
        auto piece_box = [&](int i) { return pieces[(size_t)i].summary.box(); };
        const int rows = bands_of((int)pieces.size(), true, piece_box, row_of);
        const int cols = bands_of((int)pieces.size(), false, piece_box, col_of);
        const int width = min(cols * (int)cell_size, MAX_GRID_DIM);
        const int height = min(rows * (int)cell_size, MAX_GRID_DIM);
        if (width <= 0 || height <= 0) return false;

        object arranged;
        arranged.summary = {0, 0, (int8_t)width, (int8_t)height, 0};
        for (int i = 0; i < (int)pieces.size(); ++i) {
            const object& piece = pieces[i];
            const int top = row_of[i] * cell_size;
            const int left = col_of[i] * cell_size;
            for (int row = 0; row < piece.summary.height && row < cell_size; ++row)
                for (int col = 0; col < piece.summary.width && col < cell_size; ++col) {
                    const int destination_row = top + row, destination_col = left + col;
                    if (destination_row < height && destination_col < width &&
                        piece.data[row][col] != TRANSPARENT)
                        arranged.data[destination_row][destination_col] = piece.data[row][col];
                }
        }

        const int reduced_width = max((width + cell_size - 1) / cell_size, 1);
        const int reduced_height = max((height + cell_size - 1) / cell_size, 1);
        grid painted(MAX_GRID_DIM, MAX_GRID_DIM);
        for (int row = 0; row < reduced_height; ++row)
            for (int col = 0; col < reduced_width; ++col)
                painted.data[row][col] = majority_vote(arranged.data, row, col, cell_size, height, width);

        cell_box bounds;
        for (int row = 0; row < reduced_height; ++row)
            for (int col = 0; col < reduced_width; ++col)
                if (painted.data[row][col] != TRANSPARENT) bounds.include_cell(row, col);
        if (bounds.empty()) return false;

        drawn = grid(bounds.width(), bounds.height());
        for (int row = 0; row < drawn.height; ++row)
            for (int col = 0; col < drawn.width; ++col) {
                const int color = painted.data[bounds.row0 + row][bounds.col0 + col];
                drawn.data[row][col] = (int8_t)color;
            }
        return true;
    }

    static ranked_layout_rule detect(const vector<example>& train, const output_dimension_rule& size) {
        if (train.empty() || size.Chosen != output_dimension_rule::BoundingBox) return {};
        for (int discarded = 0; discarded <= COLOR_COUNT; ++discarded)
            for (int mode = 0; mode <= 1; ++mode)
                for (int factor = 2; factor <= 10; ++factor) {
                    ranked_layout_rule candidate;
                    candidate.held = true;
                    candidate.discarded_colors = (int8_t)discarded;
                    candidate.split_mode = (int8_t)mode;
                    candidate.cell_size = (int8_t)factor;
                    const bool redraws = redraws_every_output(train, [&](const example& pair, grid& drawn) {
                        return candidate.draw(pair, drawn);
                    });
                    if (redraws) return candidate;
                }
        return {};
    }
};

// The output is the least common color, where it lies. Nothing crops it but the
// size rule: under BoundingBox the canvas closes on what was painted. Read per
// grid, so the color may change from pair to pair; a tie for least common, or a
// grid with one color and nothing to rank it against, is no reading at all.
struct rarest_color_rule {
    bool held = false;
    bool holds() const { return held; }

    static bool draw(const grid& input, grid& drawn) {
        const color_tally counts(input);
        int rarest = -1, colors = 0;
        bool tied = false;
        for (int color = 0; color < COLOR_COUNT; ++color) {
            if (!counts[color]) continue;
            ++colors;
            if (rarest >= 0 && counts[color] == counts[rarest]) tied = true;
            else if (rarest < 0 || counts[color] < counts[rarest]) { rarest = color; tied = false; }
        }
        if (colors < 2 || tied) return false;
        cell_box bounds;
        for (int row = 0; row < input.height; ++row)
            for (int col = 0; col < input.width; ++col)
                if (input.data[row][col] == rarest) bounds.include_cell(row, col);
        drawn = grid(bounds.width(), bounds.height());
        for (int row = 0; row < drawn.height; ++row)
            for (int col = 0; col < drawn.width; ++col)
                drawn.data[row][col] = input.data[bounds.row0 + row][bounds.col0 + col] == rarest
                                     ? (int8_t)rarest : (int8_t)TRANSPARENT;
        return true;
    }

    static rarest_color_rule detect(const vector<example>& train, const output_dimension_rule& size) {
        if (train.empty() || size.Chosen != output_dimension_rule::BoundingBox) return {};
        rarest_color_rule rule;
        rule.held = redraws_every_output(train, [&](const example& pair, grid& drawn) {
            return draw(pair.input, drawn);
        });
        return rule;
    }
};

// THE OUTPUT IS THE INPUT SEEN THROUGH THE SIZE RULE'S BOX, AT ONE CORNER.
// The canvas is the box, so painting the input crops it, and pushing the input
// down or right is what brings its bottom or right side into view: `Input
// SelectAll MoveToEdge[..] Union`. No lattice, no pieces and no count of cells
// -- the box says how much. A box larger than the input pads instead, which is
// the same sentence. Corners are tried with the fewest pushes first, and one
// corner must hold for every pair.
struct corner_crop_rule {
    bool held = false;
    // The pushes that bring this corner into view.
    bool down = false, right = false;
    bool holds() const { return held; }

    bool draw(const grid& input, const output_dimension_rule& size, grid& drawn) const {
        if (input.empty()) return false;
        return draw_on_canvas(input, size, drawn, [&](grid& canvas) {
            // The input is one piece as big as itself, so its cells are its box.
            const cell_box whole{ 0, 0, input.height - 1, input.width - 1 };
            const int top = down ? steps_to_edge(whole, canvas.width, canvas.height, direction::down) : 0;
            const int left = right ? steps_to_edge(whole, canvas.width, canvas.height, direction::right) : 0;
            for (int r = 0; r < input.height; ++r)
                for (int c = 0; c < input.width; ++c)
                    if (canvas.contains(top + r, left + c)) canvas.data[top + r][left + c] = input.data[r][c];
            return true;
        });
    }

    static corner_crop_rule detect(const vector<example>& train,
                                   const output_dimension_rule& size) {
        if (train.empty()) return {};
        for (bool down : { false, true })
            for (bool right : { false, true }) {
                const corner_crop_rule trying{ true, down, right };
                if (redraws_every_output(train, [&](const example& pair, grid& drawn) {
                        return trying.draw(pair.input, size, drawn);
                    })) return trying;
            }
        return {};
    }
};

// THE FRAME A SIZE CHANGE IS READ THROUGH: how the output's grid is made from
// the input's, so that what is left is a same-size puzzle. Either the piece a
// pick takes from a cut, cut out and moved to the corner of the canvas the size
// rule gives -- the pick, every other piece deleted, then Origin -- or the
// whole input scaled up `factor` times, scaled down `factor` times by the
// vote, tiled over the canvas, turned to the pose `factor` names, made into a
// fractal of itself, or reduced to the unit it repeats; or the pieces a pick
// takes laid out on graph paper of cells `factor` wide, by the bands they sit in;
// or the canvas filled with the one color the pieces a pick takes come out in.
struct piece_frame_rule {
    enum kind_of : int8_t { cut_out, scaled_up, scaled_down, tiled, posed, fractal, repeat_unit,
                            laid_out, filled } kind = cut_out;
    int8_t factor = 0;
    bool held = false;
    // Whether the pick takes several pieces in some pair, cut out together:
    // merged into one first (UnionList), so they keep their places to each other.
    bool merged = false;
    segmentation_methods cut = segmentation_methods::unset;
    piece_pick pick;
    // What the filled canvas's color is read off: each piece's own color, -1, or
    // the color the puzzle's table names for its key of this kind (shape_key).
    int8_t keyed = -1;
    bool holds() const { return held; }
    // No cut means Input's whole box, including its empty margins.
    vector<object> pieces(const grid& input, tile_size tiles = {}) const {
        if (cut != segmentation_methods::unset) return pieces_as_said(input, cut, tiles);
        object whole;
        whole.summary = { 0, 0, (int8_t)input.width, (int8_t)input.height, 0 };
        for (int row = 0; row < input.height; ++row)
            for (int col = 0; col < input.width; ++col) {
                whole.data[row][col] = (int8_t)input.data[row][col];
                whole.summary.count += input.data[row][col] != TRANSPARENT;
            }
        return { whole };
    }
};

// Collects facts, rules, and elections about one puzzle.
struct inference {
    output_dimension_rule size;
    color_facts colors;
    shape_color_map_rule shape_color_map;
    hole_color_rule hole_colors;
    unchanged_cells_facts unchanged;
    color_diff_facts color_diff;
    color_swap_candidates color_swaps;
    motif_continuation_rule motif_continuation;
    diagonal_continuation_rule diagonal_continuation;
    periodic_denoise_rule periodic_denoise;
    piece_slam_rule piece_slam;
    overlap_corner_rule overlap_corner;
    slide_to_touch_rule slide_to_touch;
    edge_slide_rule edge_slide;
    project_on_seeds_rule project_on_seeds;
    snap_to_anchors_rule snap_to_anchors;
    copy_to_rule copy_to;
    bridge_rule bridge;
    mark_lattice_rule mark_lattice;
    interior_spread_rule interior_spread;
    interior_fill_rule interior_fill;
    input_blocks_rule input_blocks;
    ranked_layout_rule ranked_layout;
    rarest_color_rule rarest_color;
    corner_crop_rule corner_crop;
    ray_extension_rule ray_extension;
    pane_split_facts pane_split;
    pane_mask_rule pane_mask;
    pane_pick_rule pane_pick;
    pane_gather_rule pane_gather;
    correspondence_election correspondence;
    // A SIZE CHANGE, READ THROUGH ITS FRAME: the frame, and everything read
    // from each framed input to its output, as a same-size puzzle of its own.
    piece_frame_rule frame;
    std::shared_ptr<const inference> framed;
    // WHICH PASS THIS READING IS (read_the_rest): the first reads the input,
    // and each pass after it the picture the passes before it drew.
    int8_t pass = 1;
    // WHETHER ITS WORDS AND PAINTS ARE ASKED WITH THE PARTNER PICKS too
    // (partner_picks): a last resort infer takes only where no reading
    // without them says every cell. Its passes and its framed reading read
    // the same way.
    bool partners = false;
    // WHAT THE LINE DRAWS, pair by pair, before the plan fills the ground: the
    // hand as its words leave it and every cell a paint drew, in the color the
    // output shows there (finish_the_line). None where the line cannot know
    // it: a hand it could not carry, or pairs that do not lie over each other.
    vector<grid> picture;
    // THE REST OF THE CHANGE, READ ON THAT PICTURE: the next pass, a puzzle of
    // its own from each picture to its output (read_the_rest), and whether it
    // leaves fewer cells unsaid -- else it is only the readout's, which names
    // the words the rest lacks.
    std::shared_ptr<const inference> rest;
    bool rest_says_more = false;
    // Whether the cut's pieces, painted where they lie on the canvas the size
    // rule gives, draw every output: what a line with no words claims.
    bool pieces_redraw = true;
    // The plan for this pass, and its input before background removal.
    // Room cuts read the colored cells that removal would discard.
    ground_plan plan;
    vector<grid> before_ground;
    vector<int8_t> elected_backgrounds;
    background_election ground_election;
    // EACH PLAN READ, in the order infer reads them (says_the_most): which plan,
    // the cells it left unsaid, the passes it took, the words its line said,
    // and whether with the partner picks -- the readout's [plans read], which
    // shows why one plan was kept over another (a61f2674's lost line was found
    // this way).
    struct plan_read { int plan = 0; int left = 0, passes = 0; string words; bool partners = false; };
    vector<plan_read> plans_read;
};

// THE RULES THAT SAY A WHOLE PROGRAM OF THEIR OWN, each an explanation of every
// pair by itself and so a rival to the cut's own line rather than a part of it.
// Listed in this order, which is the order a tie between them goes by: a split
// is a fact about the frame, so the panes lead, and a pane chosen only for where
// it sits is the corner that crops it, so the corner comes before the pick.
enum class own_sentences : int8_t { panes, corner_crop, pane_pick, pane_gather, ray_extension,
    mark_lattice, input_blocks, periodic_denoise, bridge, interior_fill, interior_spread,
    hole_colors, ranked_layout, edge_slide, copy_to, rarest_color, motif_continuation, count };

// One rule for the elected reading's list: its name, what it says, how it is said.
inline correspondence_election::fired_rule listed(const char* name, const char* says,
                                                  correspondence_election::fired_rule::said_as as,
                                                  int8_t value = 0) {
    correspondence_election::fired_rule one;
    one.name = name;
    one.says = says;
    one.as = as;
    one.value = value;
    return one;
}

// EVERY RULE THAT SAYS A WHOLE PROGRAM, ON THE LIST. Each holds over every pair
// by itself, so it claims no cells from the cut's line: what it explains is
// everything, and it is weighed against that line whole.
inline void list_the_rivals(inference& inf) {
    auto& election = inf.correspondence;
    if (election.readings.empty()) return;
    auto& read = election.readings[election.chosen];
    using rule = correspondence_election::fired_rule;
    auto rival = [&](bool holds, own_sentences which, const char* name, const char* says) {
        if (!holds) return;
        rule one = listed(name, says, rule::own_sentences, (int8_t)which);
        one.stands_alone = true;
        read.fired.push_back(std::move(one));
    };
    const bool panes = inf.pane_split.holds();
    rival(panes && inf.pane_mask.holds(), own_sentences::panes, "Panes", "what the panes come to together");
    rival(inf.corner_crop.holds(), own_sentences::corner_crop, "Corner Crop", "the input seen through one corner");
    rival(panes && inf.pane_pick.holds(), own_sentences::pane_pick, "Pane Pick", "one pane is the answer");
    rival(panes && inf.pane_gather.holds(), own_sentences::pane_gather, "Pane Gather",
          "every pane's drawing where its pane was");
    rival(inf.ray_extension.holds(), own_sentences::ray_extension, "Rays", "pieces grow rays");
    rival(inf.mark_lattice.holds(), own_sentences::mark_lattice, "Mark Lattice", "marks laid as lines that take turns");
    rival(inf.input_blocks.holds(), own_sentences::input_blocks, "Copies Of The Input",
          "the output is the input laid out in blocks");
    rival(inf.periodic_denoise.holds(), own_sentences::periodic_denoise, "Denoise", "the repeats voted clean");
    rival(inf.bridge.holds(), own_sentences::bridge, "Bridge", "the gaps between pieces of one color filled");
    rival(inf.interior_fill.holds(), own_sentences::interior_fill, "Interior Fill", "every hollow filled with one color");
    rival(inf.interior_spread.holds(), own_sentences::interior_spread, "Interior Spread", "cores painted, outlines kept");
    rival(inf.hole_colors.holds(), own_sentences::hole_colors, "Hole Colors", "each piece recolored by its holes");
    rival(inf.ranked_layout.holds(), own_sentences::ranked_layout, "Ranked Layout", "pieces laid out by rank");
    rival(inf.edge_slide.holds(), own_sentences::edge_slide, "Edge Slide", "pieces slid to an edge");
    rival(inf.copy_to.holds(), own_sentences::copy_to, "Copy To", "the largest figure copied onto the others");
    rival(inf.rarest_color.holds(), own_sentences::rarest_color, "Rarest Color", "the least common color alone");
    rival(inf.motif_continuation.holds(), own_sentences::motif_continuation, "The Input Repeats",
          "the output carries the repeating block");
}

}  // namespace arc
