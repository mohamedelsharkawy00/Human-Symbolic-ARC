// Where could an object from one grid appear in the other?
// It may have moved, turned, changed color, or become partly hidden.
// We find possible matches, reject likely coincidences, and choose matches
// that fit together. Chapter 6 compares this evidence across examples.
#pragma once
#include "4_objects.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// Diagnostic calls and volumes, reset per puzzle by --said. Matching never
// reads these counters when making a decision.
struct matching_work_counts {
    unsigned long long calls = 0, rematches = 0, gathers = 0;
    unsigned long long fixed_scans = 0, whole_grid_scans = 0, placement_cache_hits = 0;
    unsigned long long fitting_edges = 0;
};
inline thread_local matching_work_counts matching_work;

// WHICH COLOR EACH COLOR BECAME: UNSET for a color that was never seen. Two
// colors may become one, so this is a table, not a swap. Empty is a value it
// names too, both ways -- a cell that was empty, a color that became empty --
// because where a plan deleted the ground, empty is what the ground reads as
// (a ray drawn where nothing was, a figure and its ground trading places).
// Indexed by the value itself; TRANSPARENT names the slot past the colors.
struct recoloring {
    static constexpr int8_t UNSET = -2;
    int8_t to[COLOR_COUNT + 1] = {UNSET, UNSET, UNSET, UNSET, UNSET, UNSET, UNSET, UNSET, UNSET, UNSET, UNSET};

    static int slot(int from) { return is_color(from) ? from : COLOR_COUNT; }
    int8_t& operator[](int from) { return to[slot(from)]; }
    int8_t operator[](int from) const { return to[slot(from)]; }
    bool operator==(const recoloring&) const = default;

    // Seen at all: as a color, or as becoming empty.
    bool named(int from) const { return to[slot(from)] != UNSET; }
    // Seen, and became another value.
    bool changes(int from) const { return named(from) && to[slot(from)] != from; }
    // What a cell of this value becomes: itself, where the table says nothing.
    int8_t apply(int from) const { return named(from) ? to[slot(from)] : (int8_t)from; }
};

// Whether two boxes share their top-left corner: for two objects, the same place.
template <class Framed>
inline bool same_corner(const Framed& one, const Framed& other) {
    return one.col == other.col && one.row == other.row;
}

// WHAT A MATCH MAY CHANGE, as the readout names what the matching used: where
// the piece lies, which way round, its colors, how much of it the grid shows,
// its size -- a whole number of times bigger, each cell a square block
// (same_shape's scale, 1_readings.hpp). Every match may change any of them,
// and pays for each it does (freedom_spent).
inline constexpr const char* const freedom_names[] = { "Move", "Turn", "Color", "Hang Off", "Scale" };

// A match links an object to another object or a position on the other grid.
struct object_matching {
    class placement_cache;
    using object_info = object_segmentation::object_info;

    // Connects a source object to a target object or position, with its match details.
    struct edge {
        // True: input to output. False: output to input.
        bool from_input = true;
        // Index of the source object in its grid's object list.
        int16_t from = -1;
        // Index of the target object; -1 means a position without an object link.
        int16_t to = -1;
        // Top-left corner of the object's placement in the target grid.
        int8_t row = 0, col = 0;
        // Rotation or reflection, numbered 0 to 7; 0 means unchanged.
        int8_t pose = 0;
        // How many times bigger it was found: each of its cells, turned by
        // `pose`, a square block this many cells wide; 1 at its own size.
        int8_t scale = 1;
        // Whether the correspondence preserves the object's colors.
        bool exact = true;
        // Whether the object was found at its original position.
        bool stayed_put = false;
        // False means only an overlap with the target object, not the same shape.
        bool same_shape = true;
        // Number of cells covered by another object; 0 means fully visible.
        // Cell counts use 16 bits because a grid can contain up to 900 cells.
        int16_t hidden = 0;
        // Number of cells extending beyond the target grid's border.
        int16_t outside = 0;
        // Target color for each source color; -1 means that source color is absent.
        recoloring color_map;
        // Every pose that lands the object here in these colors, its own among
        // them; bit p is pose p. Said for blocks, where a symmetric copy lands
        // several ways and the rule that places the copies may need any of them.
        uint8_t poses = 0;
        // The one color a piece of one color, `source_color`, came out in; -1
        // where the match kept its colors or the piece wears several, and where it
        // became empty, which is a deletion and not a recolor.
        int8_t recolored_to(int source_color) const {
            return !exact && is_color(source_color) && is_color(color_map[source_color])
                 ? color_map[source_color] : (int8_t)-1;
        }
    };
    // Why a match that was found did not survive.
    enum refusal_reason : int8_t { lost_to_chance, cells_already_explained, the_other_examples_disagreed,
                                   a_freedom_no_cell_needs };
    // A rejected match and its reason travel together through filtering and reporting.
    struct rejection {
        edge match;
        refusal_reason reason;
    };
    // Said the same way wherever it is said. Each one finishes "it was refused
    // because", so it has to be a reason on its own and not a name for one.
    static constexpr const char* const refusal_names[] = {
        "IT WAS LIKELY BY LUCK",
        "ITS CELLS WERE ALREADY EXPLAINED",
        "THE OTHER EXAMPLES DID NOT AGREE",
        "IT USES A FREEDOM NO CELL NEEDS" };
    // Store each color's move for this training pair, only if every read pair has
    // exactly one move for that color. The move may differ between pairs.
    struct what_the_pairs_agreed {
        bool color_speaks[COLOR_COUNT] = {};
        int8_t down[COLOR_COUNT] = {}, across[COLOR_COUNT] = {};
    };

    // Operations used by the later chapters.
    // Returns the cells used to work out what a match covers in the target grid.
    // Uses the matched target object if known; otherwise, the source object turned or flipped as matched.
    static grid edge_footprint(const edge& edge,
                               const vector<object>& from_objects,
                               const vector<object>& into_objects) {
        if (edge.to >= 0 && (size_t)edge.to < into_objects.size()) return into_objects[edge.to].drawn();
        return posed_cells(from_objects[edge.from].drawn(), edge.pose);
    }

    // A cell of a landing that the grid actually shows: on it, and not behind
    // another object. Off the edge or covered, the object did not put it there.
    static bool landing_shows(const edge& edge, const grid& cells,
                              const grid& into, int row, int col) {
        if (cells.data[row][col] == TRANSPARENT) return false;
        const int into_row = edge.row + row, into_col = edge.col + col;
        if (!into.contains(into_row, into_col)) return false;
        return !edge.hidden || into.data[into_row][into_col] == cells.data[row][col];
    }

    // THE BOX AN EDGE LANDS IN: its source's, turned the way it landed and
    // grown as many times as it was found bigger.
    static pair<int, int> landing_size(const edge& edge, const object_summary& source) {
        const auto [width, height] = posed_size(edge.pose, source.width, source.height);
        return { width * edge.scale, height * edge.scale };
    }

    // EVERY CELL ONE EDGE CARRIED, as the two places it joins: where the cell
    // sat on the input, in the color it had there, and where it sat on the
    // output, in the color it came out in -- handed to `visit(in_row, in_col,
    // was, out_row, out_col, now)` whichever grid the edge was read from. Only
    // where the landing shows the piece: a cell behind something else is not
    // one it carried. Found bigger, a cell lands on its whole block.
    template <class Visit>
    static void each_carried_cell(const edge& link, const object& piece, const grid& landing, Visit&& visit) {
        if (!link.same_shape) return;
        const int scale = link.scale;
        piece.each_cell([&](int row, int col, int8_t color) {
            const auto [down, across] = posed_place(link.pose, piece.width, piece.height,
                                                    row - piece.row, col - piece.col);
            for (int block_row = 0; block_row < scale; ++block_row)
                for (int block_col = 0; block_col < scale; ++block_col) {
                    const int at_row = link.row + down * scale + block_row;
                    const int at_col = link.col + across * scale + block_col;
                    if (!landing.contains(at_row, at_col)) continue;
                    const int8_t there = (int8_t)landing.data[at_row][at_col];
                    if (there != link.color_map.apply(color)) continue;
                    if (link.from_input) visit(row, col, color, at_row, at_col, there);
                    else                 visit(at_row, at_col, there, row, col, color);
                }
        });
    }

    // Movement from input to output in the original grid coordinates.
    // Reverse an output-to-input correspondence so both directions agree.
    static pair<int, int> move_of(const edge& edge,
                                  const object_info& one_object) {
        const int down = edge.row - one_object.summary.row;
        const int across = edge.col - one_object.summary.col;
        return edge.from_input ? pair<int, int>{down, across}
                               : pair<int, int>{-down, -across};
    }

    // Lower scores prefer matches that did not move, change color, turn or flip,
    // or show only part of themselves, in that order of importance. Hanging off
    // the grid and hiding behind another object are the one freedom: some of the
    // piece is not shown. Free to hide, a piece half behind a frame cost no more
    // than the same piece in plain sight beside it.
    static int freedom_spent(const edge& edge) {
        return (edge.stayed_put ? 0 : 8) + (edge.exact ? 0 : 4) +
               (edge.pose == 0 ? 0 : 2) + (edge.scale == 1 ? 0 : 2) +
               (edge.outside == 0 && edge.hidden == 0 ? 0 : 1);
    }

    // Reject shape matches whose movement disagrees with the color's agreed move.
    // Stationary matches are always allowed.
    static bool the_pairs_disagree(const what_the_pairs_agreed& agreed,
                                   const edge& edge,
                                   const object_info& one_object) {
        if (!edge.same_shape) return false;
        const auto went = move_of(edge, one_object);
        if (!went.first && !went.second) return false;
        int input_color = one_object.color;
        if (!is_color(input_color)) return true;
        // A backward recoloring maps the output color back to the input color.
        if (!edge.from_input && !edge.exact) input_color = edge.color_map[input_color];
        if (!is_color(input_color) || !agreed.color_speaks[input_color]) return true;
        return agreed.down[input_color] != went.first ||
               agreed.across[input_color] != went.second;
    }

    // Find and select matches from one grid to the other. Swap grids for the reverse direction.
    static void collect_edges(const vector<object_info>& from_info,
                              const vector<object>& from_objects,
                              const vector<object_info>& into_info,
                              const vector<object>& into_objects,
                              const grid& into, bool from_input,
                              vector<edge>& edges,
                              vector<rejection>* discarded = nullptr,
                              const what_the_pairs_agreed* agreed = nullptr,
                              placement_cache* placements = nullptr) {
        ++matching_work.calls;
        matching_work.rematches += agreed != nullptr;
        vector<edge> found = gather_matches(from_info, from_objects, into_info, into, discarded, placements);
        for (auto& edge : found) edge.from_input = from_input;

        // Remove conflicting matches before fitting so their alternatives can still be chosen.
        if (agreed)
            for (size_t i = found.size(); i-- > 0; ) {
                const auto& one_object = from_info[found[i].from];
                if (!the_pairs_disagree(*agreed, found[i], one_object)) continue;
                throw_away(found[i], object_matching::the_other_examples_disagreed,
                           discarded);
                found.erase(found.begin() + i);
            }

        fitting laid_out = set_up_the_fitting(found, from_objects, into_objects, into);
        if (drop_freedoms_no_cell_needs(found, laid_out, from_info, discarded))
            laid_out = set_up_the_fitting(found, from_objects, into_objects, into);
        matching_work.fitting_edges += found.size();
        keep_what_adds_a_cell(found, laid_out, edges, discarded);
    }

    // Match each unmatched object to the same-color object with the most overlap,
    // requiring that the other object has at least as many cells.
    static void sit_on_top(const vector<object_info>& from_info,
                           const vector<object>& from_objects,
                           const vector<object_info>& into_info,
                           const vector<object>& into_objects,
                           vector<edge>& edges) {
        for (size_t from_index = 0; from_index < from_info.size(); ++from_index) {
            // Only objects nothing else accounted for. Asked of the edges themselves,
            // so this does not depend on a tally someone remembered to keep.
            if (ranges::any_of(edges, [&](const edge& edge) {
                    return edge.from == (int16_t)from_index; }))
                continue;
            const auto& one_object = from_info[from_index];
            const object& cells = from_objects[from_index];
            int best = -1, most_shared = 0;
            for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
                const auto& landing = into_info[into_index];
                if (landing.color != one_object.color || landing.summary.count < one_object.summary.count) continue;
                int shared = 0;
                cells.each_cell([&](int row, int col, int8_t) {
                    shared += into_objects[into_index].holds(row, col);
                });
                if (shared > most_shared) { most_shared = shared; best = (int)into_index; }
            }
            if (best < 0) continue;
            object_matching::edge edge;
            edge.from = (int16_t)from_index;
            edge.to = (int16_t)best;
            edge.row = into_info[best].summary.row;
            edge.col = into_info[best].summary.col;
            edge.exact = false;
            edge.same_shape = false;
            edges.push_back(edge);
        }
    }

    // Link only the topmost, then leftmost complete motifs.
    static void collect_representative_edge(
            const vector<object_info>& from_info, const vector<object>& from_objects,
            const vector<object_info>& into_info, const grid& into,
            int step_w, int step_h, vector<edge>& edges) {
        auto first = [&](const vector<object_info>& info) {
            int best = -1;
            for (int i = 0; i < (int)info.size(); ++i) {
                const auto& box = info[i].summary;
                if (box.width != step_w || box.height != step_h) continue;
                if (best < 0 || pair{box.row, box.col} <
                    pair{info[best].summary.row, info[best].summary.col}) best = i;
            }
            return best;
        };
        const int source = first(from_info), target = first(into_info);
        if (source < 0 || target < 0) return;
        const auto& landing = into_info[target].summary;
        const bool stayed = same_corner(from_info[source].summary, landing);
        const auto matches = find_in(from_objects[source].drawn(), into, false, landing.row, landing.col, true);
        const object_match* best = nullptr;
        for (const auto& hit : matches.hits) {
            if (hit.hidden || hit.outside) continue;
            if (!best || (hit.exact && !best->exact)) best = &hit;
        }
        if (!best) return;
        auto link = edge_from_match(*best, source, stayed);
        link.to = (int16_t)target;
        edges.push_back(link);
    }

    // Match repeated blocks in both directions so every matching block gets a partner.
    // Prefer a source block at the same position when possible.
    static void collect_motif_edges(
            const vector<object_info>& from_info,
            const vector<object>& from_objects,
            const vector<object_info>& into_info,
            const grid& into, int step_w, int step_h,
            vector<edge>& edges) {
        if (step_w <= 0 || step_h <= 0) return;

        // Does this from block sit at that place on the other grid? The answer is
        // the same question both sweeps ask, so it is written once. `poses` holds
        // every pose in which it sits there in the colors of the one kept.
        auto sits_at = [&](int from_index, const object_info& target,
                           object_match& found, uint8_t& poses) {
            const grid cells = from_objects[from_index].drawn();
            const object_matches matches = find_in(cells, into, true, target.summary.row, target.summary.col, true);
            // Prefer a match with unchanged colors, even if it uses a later rotation or reflection.
            const object_match* best = nullptr;
            for (const object_match& hit : matches.hits) {
                if (!best || (hit.exact && !best->exact)) best = &hit;
                if (best->exact) break;
            }
            if (!best) return false;
            found = *best;
            // A symmetric block sits several ways round, and find_in tries each
            // distinct drawing once, so the ways that draw alike are read off the
            // drawings themselves.
            poses = 0;
            for (const object_match& hit : matches.hits) {
                if (hit.exact != best->exact || hit.color_map != best->color_map) continue;
                const grid landed = posed_cells(cells, hit.pose);
                for (int pose = 0; pose < POSE_COUNT; ++pose)
                    if (posed_cells(cells, pose) == landed) poses |= (uint8_t)(1 << pose);
            }
            return true;
        };
        auto draw = [&](int from_index, int target_index, const object_match& hit, uint8_t poses) {
            const auto& target = into_info[target_index];
            auto edge = edge_from_match(hit, from_index, same_corner(from_info[from_index].summary, target.summary));
            edge.to = (int16_t)target_index;
            edge.poses = poses;
            edges.push_back(edge);
        };

        vector<bool> answered(from_info.size(), false);
        // Prefer a matching source at the target's position, then the first match elsewhere.
        for (int target_index = 0; target_index < (int)into_info.size(); ++target_index) {
            const auto& target = into_info[target_index];
            int source = -1;
            object_match hit;
            uint8_t poses = 0;
            for (int i = 0; i < (int)from_info.size(); ++i)
                if (same_corner(from_info[i].summary, target.summary) && sits_at(i, target, hit, poses)) {
                    source = i;
                    break;
                }
            for (int i = 0; source < 0 && i < (int)from_info.size(); ++i)
                if (sits_at(i, target, hit, poses)) source = i;
            if (source < 0) continue;
            draw(source, target_index, hit, poses);
            answered[source] = true;
        }
        // WHERE DID EACH OBJECT GO. Only the ones no landing claimed, so a from side
        // bigger than the into side is not silently dropped.
        for (int i = 0; i < (int)from_info.size(); ++i) {
            if (answered[i]) continue;
            object_match hit;
            uint8_t poses = 0;
            for (int target_index = 0; target_index < (int)into_info.size(); ++target_index)
                if (sits_at(i, into_info[target_index], hit, poses)) {
                    draw(i, target_index, hit, poses);
                    answered[i] = true;
                    break;
                }
        }
    }

private:
    // 5. How possible placements become the matches kept.

    // A search hit: placement, rotation or reflection, colors, and clipped or hidden cells.
    struct object_match {
        // Top-left corner of the object's placement in the target grid.
        int row = 0, col = 0;
        // Rotation or reflection, numbered 0 to 7; 0 means unchanged.
        int pose = 0;
        // Whether the correspondence preserves the object's colors.
        bool exact = false;
        // Number of cells covered by another object; 0 means fully visible.
        int hidden = 0;
        // Number of cells extending beyond the target grid's border.
        int outside = 0;
        // Target color for each source color; -1 means that source color is absent.
        recoloring color_map;
    };

    // Search hits and the placements tried, counting distinct poses only.
    struct object_matches {
        vector<object_match> hits;
        // THE PLACES A HIT COULD HAVE BEEN LUCK IN, by the freedoms it used:
        // [turned][hanging off]. As it is, its own way round alone; turned,
        // every way round tried. On the grid, the places on it; hanging off,
        // the places off it too.
        int places[2][2] = {};
        int places_of(const object_match& hit) const { return places[hit.pose != 0][hit.outside > 0]; }
    };
public:
    // Owned by one segmentation's reading, separately for each pair and direction.
    // Only raw whole-grid placements are shared, never acceptance or fitting.
    class placement_cache {
        friend struct object_matching;
        struct drawing {
            grid cells;
            bool clipped;
            object_matches matches;
        };
        grid target;
        vector<drawing> drawings;
    };
private:
    // An object's cells drawn out in one of the eight ways round (1_readings.hpp).
    static grid posed_cells(const grid& cells, int pose) {
        const auto [posed_width, posed_height] = posed_size(pose, cells.width, cells.height);
        grid posed(posed_width, posed_height);
        for (int row = 0; row < posed_height; ++row)
            for (int col = 0; col < posed_width; ++col) {
                const auto [source_row, source_col] =
                    pose_source(pose, cells.width, cells.height, row, col);
                posed.data[row][col] = cells.data[source_row][source_col];
            }
        return posed;
    }
    // An object's cells drawn `factor` times bigger, each a square block.
    static grid scaled_cells(const grid& cells, int factor) {
        grid big(cells.width * factor, cells.height * factor);
        for (int row = 0; row < big.height; ++row)
            for (int col = 0; col < big.width; ++col) big.data[row][col] = cells.data[row / factor][col / factor];
        return big;
    }
    static int lit_cells(const grid& cells) {
        int lit = 0;
        for (int row = 0; row < cells.height; ++row)
            for (int col = 0; col < cells.width; ++col)
                lit += (cells.data[row][col] != TRANSPARENT);
        return lit;
    }

    // One place being tried: the posed object laid on `where` with its top-left
    // at (row, col). Only the cells that land on the grid are evidence.
    struct placement {
        const grid& posed;
        const grid& where;
        int row, col;
        bool lands(int r, int c) const { return where.contains(row + r, col + c); }
        int under(int r, int c) const { return where.data[row + r][col + c]; }
    };

    // THE OBJECT AS DRAWN, or one consistent recoloring of it: every lit cell
    // lands on a color, and one color always lands on the same color. Matching a
    // footprint, the blank cells must land on empty cells and the lit ones off
    // them. Otherwise a lit cell landing on an empty cell lands on the ground a
    // plan deleted, said as becoming empty.
    static bool fits_as_drawn(const placement& at, bool match_footprint, object_match& hit) {
        hit.color_map = {};
        hit.exact = true;
        for (int row = 0; row < at.posed.height; ++row)
            for (int col = 0; col < at.posed.width; ++col) {
                if (!at.lands(row, col)) continue;
                const int object_color = at.posed.data[row][col];
                const int there_color = at.under(row, col);
                if (object_color == TRANSPARENT) {
                    if (match_footprint && there_color != TRANSPARENT) return false;
                    continue;
                }
                if (match_footprint && there_color == TRANSPARENT) return false;
                if (!is_color(object_color) || (!is_color(there_color) && there_color != TRANSPARENT)) return false;
                if (hit.color_map[object_color] == recoloring::UNSET) hit.color_map[object_color] = (int8_t)there_color;
                if (hit.color_map[object_color] != there_color) return false;
                if (object_color != there_color) hit.exact = false;
            }
        // A BLOCK IS THE SAME PICTURE only while its colors are renamed one for
        // one. Two colors run together is a picture that lost something, and
        // would match a grid to any recoloring of it.
        if (match_footprint) {
            bool named[COLOR_COUNT] = {};
            for (int i = 0; i < COLOR_COUNT; ++i) {
                if (hit.color_map[i] < 0) continue;
                if (named[hit.color_map[i]]) return false;
                named[hit.color_map[i]] = true;
            }
        }
        return true;
    }

    // THE WEAKER QUESTION, asked only where the object as drawn failed: itself,
    // with the cells it is missing hidden behind other cells. Counts them.
    static bool fits_behind(const placement& at, object_match& hit) {
        for (int i = 0; i < COLOR_COUNT; ++i) hit.color_map[i] = (int8_t)i;
        hit.exact = true;
        hit.hidden = 0;
        for (int row = 0; row < at.posed.height; ++row)
            for (int col = 0; col < at.posed.width; ++col) {
                const int object_color = at.posed.data[row][col];
                if (object_color == TRANSPARENT || !at.lands(row, col)) continue;
                const int there_color = at.under(row, col);
                if (there_color == TRANSPARENT) return false;
                hit.hidden += (object_color != there_color);
            }
        return true;
    }

    // A BLOCK IS A WHOLE PICTURE, ITS GROUND INCLUDED. Where the footprints
    // disagree it may still be the same picture with its colors traded, and a
    // negative is exactly that: the ground and the figure swapping, which the
    // Color freedom already says. One to one, or a block of one color would
    // match every block. The ground is empty, so the negative trades a figure
    // with empty.
    static bool fits_as_negative(const placement& at, object_match& hit) {
        recoloring came_from;
        hit.color_map = {};
        hit.exact = true;
        for (int row = 0; row < at.posed.height; ++row)
            for (int col = 0; col < at.posed.width; ++col) {
                if (!at.lands(row, col)) continue;
                const int was = at.posed.data[row][col], now = at.under(row, col);
                if (hit.color_map[was] == recoloring::UNSET && came_from[now] == recoloring::UNSET) {
                    hit.color_map[was] = (int8_t)now;
                    came_from[now] = (int8_t)was;
                }
                if (hit.color_map[was] != now || came_from[now] != was) return false;
                if (was != now) hit.exact = false;
            }
        return true;
    }

    // Searches for an object in another grid.
    static object_matches find_in(const grid& cells, const grid& where,
                                  bool may_hang_off, int fixed_row = -1, int fixed_col = -1,
                                  bool match_footprint = false) {
        object_matches found;
        if (cells.empty()) return found;
        const bool fixed = fixed_row >= 0 && fixed_col >= 0;
        if (fixed) ++matching_work.fixed_scans;
        else ++matching_work.whole_grid_scans;

        // Whether the grid searched has a ground at all: empty cells, where a plan
        // deleted it. A grid read whole has none, and so no negative.
        bool empty_ground = false;
        for (int row = 0; row < where.height && !empty_ground; ++row)
            for (int col = 0; col < where.width && !empty_ground; ++col)
                empty_ground = where.data[row][col] == TRANSPARENT;
        vector<grid> tried;
        for (int pose = 0; pose < POSE_COUNT; ++pose) {
            grid posed = posed_cells(cells, pose);
            // A symmetric object reads the same several ways round; count it once.
            if (ranges::find(tried, posed) != tried.end()) continue;
            if (!may_hang_off &&
                (posed.width > where.width || posed.height > where.height)) continue;
            if (fixed && !may_hang_off &&
                (fixed_col + posed.width > where.width ||
                 fixed_row + posed.height > where.height))
                continue;
            tried.push_back(posed);
            const int first_row = fixed ? fixed_row : may_hang_off ? 1 - posed.height : 0;
            const int first_col = fixed ? fixed_col : may_hang_off ? 1 - posed.width  : 0;
            const int last_row = fixed ? fixed_row
                               : may_hang_off ? where.height - 1 : where.height - posed.height;
            const int last_col = fixed ? fixed_col
                               : may_hang_off ? where.width - 1 : where.width - posed.width;
            const int places = (last_row - first_row + 1) * (last_col - first_col + 1);
            const int on_grid = fixed ? (int)(fixed_row + posed.height <= where.height &&
                                              fixed_col + posed.width <= where.width)
                                      : max(0, where.height - posed.height + 1) * max(0, where.width - posed.width + 1);
            for (const bool turned : { false, true })
                if (turned || pose == 0) {
                    found.places[turned][0] += on_grid;
                    found.places[turned][1] += places;
                }
            const int lit = lit_cells(posed);

            for (int at_row = first_row; at_row <= last_row; ++at_row)
                for (int at_col = first_col; at_col <= last_col; ++at_col) {
                    const placement at{ posed, where, at_row, at_col };
                    // Only visible cells provide evidence for a match.
                    int shown = 0;
                    for (int row = 0; row < posed.height; ++row)
                        for (int col = 0; col < posed.width; ++col)
                            shown += (posed.data[row][col] != TRANSPARENT && at.lands(row, col));
                    if (!shown) continue;
                    object_match hit;
                    hit.row = at_row; hit.col = at_col; hit.pose = pose;
                    hit.outside = lit - shown;
                    // As drawn; failing that, the whole object partly hidden, or, for
                    // a block on a ground, its negative.
                    if (fits_as_drawn(at, match_footprint, hit) ||
                        (!match_footprint && fits_behind(at, hit)) ||
                        (match_footprint && empty_ground && fits_as_negative(at, hit)))
                        found.hits.push_back(hit);
                }
        }
        return found;
    }

    // Did any cell of the object, drawn the way the edge turned it, land on a
    // figure? The empty ground has no shape, so every shape fits it, and a match
    // there is no evidence.
    static bool landed_on_a_figure(const edge& edge, const grid& posed, const grid& into) {
        for (int row = 0; row < posed.height; ++row)
            for (int col = 0; col < posed.width; ++col) {
                if (posed.data[row][col] == TRANSPARENT) continue;
                const int into_row = edge.row + row, into_col = edge.col + col;
                if (!into.contains(into_row, into_col)) continue;
                if (into.data[into_row][into_col] != TRANSPARENT) return true;
            }
        return false;
    }

    // Measures how common each color is to help estimate the chance of an accidental match.
    // Out of the whole grid: an empty cell is ground a match could land on as
    // well, so a ground already deleted still counts against chance.
    // The last share is empty's, read like a color where the ground is empty.
    static void color_shares(const grid& one_grid, double share[COLOR_COUNT + 1]) {
        const color_tally counts(one_grid);
        const int area = one_grid.width * one_grid.height;
        int painted = 0;
        for (int i = 0; i < COLOR_COUNT; ++i) painted += counts[i];
        for (int i = 0; i < COLOR_COUNT; ++i) share[i] = area > 0 ? (double)counts[i] / area : 0.0;
        share[COLOR_COUNT] = area > 0 ? (double)(area - painted) / area : 0.0;
    }

    // A tiny blue object can match many places just because blue is common.
    // Reject likely coincidences before treating a match as evidence of correspondence.
    static bool beats_chance(int places, double odds) {
        return places > 0 && places * odds < 1.0;
    }

    // A match that was found and then thrown away, with the reason.
    static void throw_away(const edge& match, refusal_reason reason,
                           vector<rejection>* discarded) {
        if (discarded) discarded->push_back({match, reason});
    }

    // Copy a search hit into a correspondence, including its color changes.
    static edge edge_from_match(const object_match& hit, int from_index, bool stayed_put) {
        object_matching::edge edge;
        edge.from = (int16_t)from_index;
        edge.row = (int8_t)hit.row; edge.col = (int8_t)hit.col;
        edge.pose = (int8_t)hit.pose;
        edge.exact = hit.exact;
        edge.stayed_put = stayed_put;
        edge.hidden = (int16_t)hit.hidden;
        edge.outside = (int16_t)hit.outside;
        if (!hit.exact) edge.color_map = hit.color_map;
        return edge;
    }

    // Find possible matches for each object in the other grid.
    static vector<edge> gather_matches(
            const vector<object_info>& from_info,
            const vector<object>& from_objects,
            const vector<object_info>& into_info,
            const grid& into,
            vector<rejection>* discarded,
            placement_cache* shared_placements) {
        ++matching_work.gathers;
        double share[COLOR_COUNT + 1];
        color_shares(into, share);

        // Check for repeated shapes in the target grid before allowing clipped matches.
        auto twins_over_there = [&](uint32_t shape) {
            int seen = 0;
            for (const auto& landing : into_info)
                if (landing.shape == shape && ++seen > 1) return true;
            return false;
        };

        vector<edge> found;
        // Whether each found match lands on a figure, asked as it is found, while
        // its source is drawn. Refused below, in the order found.
        vector<bool> on_figures;

        // A caller can retain these answers for the pair's rematch
        // (reconcile_examples), invalidated when the target changes; copies of
        // one drawing share them too. Each source still accepts hits in its own
        // order.
        placement_cache local_placements;
        auto& placements = shared_placements ? *shared_placements : local_placements;
        if (!(placements.target == into)) {
            placements.drawings.clear();
            placements.target = into;
        }
        auto places_for = [&](const grid& cells, bool clipped) -> const object_matches& {
            for (const auto& prior : placements.drawings)
                if (prior.clipped == clipped && prior.cells == cells) {
                    ++matching_work.placement_cache_hits;
                    return prior.matches;
                }
            placements.drawings.push_back({ cells, clipped, find_in(cells, into, clipped) });
            return placements.drawings.back().matches;
        };

        for (size_t from_index = 0; from_index < from_objects.size(); ++from_index) {
            const auto& one_object = from_info[from_index];
            const grid cells = from_objects[from_index].drawn();
            // The source turned the way a hit turned it. The search hands over its
            // hits one pose at a time, so each pose is drawn once, not once a hit.
            grid turned;
            int turned_as = -1;
            auto posed_for = [&](int pose) -> const grid& {
                if (pose != turned_as) { turned = posed_cells(cells, pose); turned_as = pose; }
                return turned;
            };
            // Only this source's accepted placements can duplicate its shape
            // matches. Keep their corners once instead of rescanning all edges.
            cell_set placed;
            auto keep = [&](const edge& match) {
                if (into.contains(match.row, match.col)) placed.add(match.row, match.col);
                // Grown, it landed on a piece of the other grid, a figure by being one.
                on_figures.push_back(match.scale != 1 || landed_on_a_figure(match, posed_for(match.pose), into));
                found.push_back(match);
            };

            // Positional: own place first, then anywhere only if it was not there.
            auto keep_positional = [&](const object_matches& found_here, bool stationary) {
                for (const object_match& hit : found_here.hits) {
                    if (!stationary && hit.pose == 0 &&
                        hit.row == one_object.summary.row && hit.col == one_object.summary.col)
                        continue;
                    // Estimate accidental-match odds using visible cells only.
                    const grid& posed = posed_for(hit.pose);
                    double odds = 1.0;
                    for (int row = 0; row < posed.height; ++row)
                        for (int col = 0; col < posed.width; ++col) {
                            const int object_color = posed.data[row][col];
                            if (object_color == TRANSPARENT) continue;
                            const int into_row = hit.row + row, into_col = hit.col + col;
                            if (!into.contains(into_row, into_col)) continue;
                            const int into_color = into.data[into_row][into_col];
                            if (hit.hidden && into_color != object_color) continue;
                            odds *= share[recoloring::slot(hit.color_map[object_color])];
                        }
                    auto edge = edge_from_match(hit, (int)from_index, stationary);
                    const bool stands_alone = beats_chance(found_here.places_of(hit), odds);
                    // EXPERIMENT: every match must pass the chance test, not only the
                    // stationary, hidden, or recolored ones.
                    if (!stands_alone) {
                        throw_away(edge, object_matching::lost_to_chance, discarded);
                        continue;
                    }
                    keep(edge);
                }
            };

            // Search both the original position and other positions: an object can stay and be copied.
            keep_positional(find_in(cells, into, false, one_object.summary.row, one_object.summary.col), true);
            // Allow clipped matches only when copies establish the shape on either grid.
            keep_positional(places_for(cells, one_object.motif >= 0 || twins_over_there(one_object.shape)), false);

            // Source 3 — the same shape at another size: a piece there whose shape
            // at any size (scaled_shape) is this one's, grown a whole number of
            // times -- the factor read off the two boxes, either way round where
            // it may turn -- and every cell, drawn that big, lying on the piece as
            // a match by placement asks. A dot or a filled box has no size but
            // its own (sized_blocks), so none is ever found at another.
            for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
                const auto& landing = into_info[into_index].summary;
                if (into_info[into_index].scaled_shape != one_object.scaled_shape) continue;
                for (const bool sideways : { false, true }) {
                    const int wide = sideways ? cells.height : cells.width, tall = sideways ? cells.width : cells.height;
                    const int factor = landing.width / wide;
                    if (factor < 2 || landing.width != wide * factor || landing.height != tall * factor) continue;
                    const object_matches grown = find_in(scaled_cells(cells, factor), into, false, landing.row,
                                                         landing.col, true);
                    const object_match* best = nullptr;
                    for (const object_match& hit : grown.hits) {
                        const auto [hit_width, hit_height] = posed_size(hit.pose, cells.width * factor,
                                                                        cells.height * factor);
                        if (hit_width != landing.width || hit_height != landing.height) continue;
                        if (!best || (hit.exact && !best->exact)) best = &hit;
                    }
                    if (!best) continue;
                    auto edge = edge_from_match(*best, (int)from_index, false);
                    edge.to = (int16_t)into_index;
                    edge.scale = (int8_t)factor;
                    keep(edge);
                    break;
                }
            }

            // Source 2 — shape: objects there with the same canonical hash (count
            // equality guards hash collisions). Skip placements source 1 found.
            for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
                const auto& landing = into_info[into_index];
                if (landing.shape != one_object.shape || landing.summary.count != one_object.summary.count) continue;
                if (same_corner(landing.summary, one_object.summary)) continue;
                if (placed.has(landing.summary.row, landing.summary.col)) continue;
                // Estimate chance from the target color count, or all grid colors for multicolor objects.
                double odds = 1.0;
                if (landing.color >= 0)
                    for (int i = 0; i < landing.summary.count; ++i) odds *= share[landing.color];
                else
                    into.for_each_colored_cell(TRANSPARENT,
                                               [&](int, int, int color) { odds *= share[color]; });
                object_matching::edge edge;
                edge.from = (int16_t)from_index;
                edge.to = (int16_t)into_index;
                edge.row = landing.summary.row; edge.col = landing.summary.col;
                edge.exact = (landing.color == one_object.color);
                if (!edge.exact && is_color(one_object.color))
                    edge.color_map[one_object.color] = landing.color;
                const bool stands_alone = beats_chance(
                    (into.width - landing.summary.width + 1) * (into.height - landing.summary.height + 1), odds);
                // Same bar as source 1: a recolor is the weaker claim either way it
                // was found, so it earns its place alone or not at all.
                if (!edge.exact && !stands_alone) {
                    throw_away(edge, object_matching::lost_to_chance, discarded);
                    continue;
                }
                keep(edge);
            }
        }
        // Refused once the whole list is in one place, so the refusals keep its order.
        vector<edge> on_a_figure;
        for (size_t i = 0; i < found.size(); ++i) {
            if (on_figures[i]) {
                on_a_figure.push_back(found[i]);
                continue;
            }
            throw_away(found[i], object_matching::lost_to_chance,
                       discarded);
        }
        return on_a_figure;
    }

    // One candidate object: the match it came from and the cells it paints.
    struct candidate {
        size_t match_index = 0;
        vector<int> cells;
    };

    // The objects, and which cells each of them could paint.
    struct fitting {
        // Number of target cells, empty ones included.
        size_t cell_count = 0;
        // One per match, in match order; empty cells mean it shows nothing
        vector<candidate> objects;
        // Every object, best first (comes_first); a tie keeps match order.
        vector<int> best_first;
        // How far each match's piece travelled to land, by match index: travel_of.
        vector<int> travel;

        // Whether candidate `left` goes before `right`: says_more, below.
        bool comes_first(int left, int right, const vector<edge>& found) const {
            const size_t mine = objects[left].match_index, theirs = objects[right].match_index;
            return says_more(found[mine], objects[left].cells.size(),
                             found[theirs], objects[right].cells.size(),
                             travel[mine], travel[theirs]);
        }
    };
    // HOW FAR A PIECE TRAVELLED to land here: a slide along one line before one
    // that needs both, then the shorter. Asked only once cells and freedoms tie,
    // which is where the order the pieces were listed in used to decide which
    // copy a landing names -- 06df4c85 credited a block between two reds to a
    // red in another row entirely, for being listed first.
    static int travel_of(const edge& edge, const object& source) {
        const int down = edge.row - source.row, across = edge.col - source.col;
        return (down != 0 && across != 0 ? MAX_GRID_DIM * 2 : 0) + abs(down) + abs(across);
    }

    // Prefer more cells; break ties with the lower freedom_spent score, then
    // with the shorter travel.
    static bool says_more(const edge& left, size_t left_cells,
                          const edge& right, size_t right_cells,
                          int left_travel, int right_travel) {
        if (left_cells != right_cells) return left_cells > right_cells;
        if (freedom_spent(left) != freedom_spent(right))
            return freedom_spent(left) < freedom_spent(right);
        return left_travel < right_travel;
    }

    // For each match, the target cells it paints -- on the grid and not behind
    // another object -- and the order the matches are tried in.
    static fitting set_up_the_fitting(const vector<edge>& found,
                             const vector<object>& from_objects,
                             const vector<object>& into_objects,
                             const grid& into) {
        fitting laid_out;
        // Index all target cells, empty ones included, where new objects may appear.
        laid_out.cell_count = (size_t)into.width * into.height;
        laid_out.travel.resize(found.size());
        for (size_t found_index = 0; found_index < found.size(); ++found_index)
            laid_out.travel[found_index] =
                travel_of(found[found_index], from_objects[found[found_index].from]);
        // Include recolored matches as candidates: their shapes still need to fit.
        for (size_t found_index = 0; found_index < found.size(); ++found_index) {
            const edge& link = found[found_index];
            const grid cells = edge_footprint(link, from_objects, into_objects);
            candidate candidate{ found_index };
            for (int row = 0; row < cells.height; ++row)
                for (int col = 0; col < cells.width; ++col)
                    if (landing_shows(link, cells, into, row, col))
                        candidate.cells.push_back((link.row + row) * into.width + link.col + col);
            laid_out.objects.push_back(std::move(candidate));
        }
        // Larger matches first, then lower change scores, then the shorter travel.
        laid_out.best_first.resize(laid_out.objects.size());
        iota(laid_out.best_first.begin(), laid_out.best_first.end(), 0);
        stable_sort(laid_out.best_first.begin(), laid_out.best_first.end(), [&](int left, int right) {
            return laid_out.comes_first(left, right, found);
        });
        return laid_out;
    }

    // A FREEDOM IS USED ONLY WHERE A CELL NEEDS IT: where every match reaching
    // some cell uses it. A match using one no cell needs is dropped, the
    // plainer matches reaching all it would: 18286ef8's ring, found half turned
    // and half hidden where it is found standing. Says whether any went.
    static bool drop_freedoms_no_cell_needs(vector<edge>& found, const fitting& laid_out,
                                            const vector<object_info>& from_info, vector<rejection>* discarded) {
        // The freedoms each match uses, a bit each in freedom_names' order.
        vector<unsigned> uses(found.size());
        for (size_t at = 0; at < found.size(); ++at) {
            const edge& link = found[at];
            const auto& source = from_info[link.from].summary;
            uses[at] = (unsigned)(link.row != source.row || link.col != source.col) | (unsigned)(link.pose != 0) << 1 |
                       (unsigned)!link.exact << 2 | (unsigned)(link.outside > 0) << 3 | (unsigned)(link.scale != 1) << 4;
        }
        // For each cell some match reaches, the freedoms every match reaching it uses.
        vector<unsigned> every_reaching(laid_out.cell_count, ~0u);
        for (const candidate& one : laid_out.objects)
            for (const int id : one.cells) every_reaching[(size_t)id] &= uses[one.match_index];
        unsigned needed = 0;
        for (const unsigned all : every_reaching)
            if (all != ~0u) needed |= all;
        size_t kept = 0;
        for (size_t at = 0; at < found.size(); ++at)
            if (uses[at] & ~needed) throw_away(found[at], a_freedom_no_cell_needs, discarded);
            else found[kept++] = found[at];
        const bool dropped = kept != found.size();
        found.resize(kept);
        return dropped;
    }

    // THE MATCHES KEPT: the best first (says_more), each kept where it paints
    // a cell no match before it did; then, the weakest first, each dropped
    // whose every cell the others kept paint -- fe45cba4's frame, found again a
    // step over inside the block its partner fills, where its own place and
    // the partner say every cell. A match that shows nothing is refused too.
    static void keep_what_adds_a_cell(const vector<edge>& found, const fitting& laid_out,
                                      vector<edge>& edges, vector<rejection>* discarded) {
        // How many kept matches paint each cell.
        vector<int> painters(laid_out.cell_count, 0);
        vector<int> kept;
        for (int object_index : laid_out.best_first) {
            const auto& candidate = laid_out.objects[object_index];
            if (ranges::none_of(candidate.cells, [&](int id) { return painters[id] == 0; })) {
                throw_away(found[candidate.match_index], cells_already_explained, discarded);
                continue;
            }
            for (int id : candidate.cells) ++painters[id];
            kept.push_back(object_index);
        }
        for (size_t at = kept.size(); at-- > 0;) {
            const auto& candidate = laid_out.objects[kept[at]];
            if (ranges::any_of(candidate.cells, [&](int id) { return painters[id] == 1; })) continue;
            for (int id : candidate.cells) --painters[id];
            throw_away(found[candidate.match_index], cells_already_explained, discarded);
            kept[at] = -1;
        }
        for (const int object_index : kept)
            if (object_index >= 0) edges.push_back(found[laid_out.objects[object_index].match_index]);
    }

};

} // namespace arc
