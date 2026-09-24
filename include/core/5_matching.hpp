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
#include <string>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// WHICH COLOR EACH COLOR BECAME: -1 for a color that was never seen. Two
// colors may become one, so this is a table, not a swap.
struct recoloring {
    int8_t to[COLOR_COUNT] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    int8_t& operator[](int from) { return to[from]; }
    int8_t operator[](int from) const { return to[from]; }
    bool operator==(const recoloring&) const = default;

    // Seen, and became another color.
    bool changes(int from) const { return to[from] >= 0 && to[from] != from; }
    // What a cell of this color becomes: itself, where the table says nothing.
    int8_t apply(int from) const { return to[from] >= 0 ? to[from] : (int8_t)from; }
};

// Whether two boxes share their top-left corner: for two objects, the same place.
inline bool same_corner(const object_summary& one, const object_summary& other) {
    return one.x0 == other.x0 && one.y0 == other.y0;
}

// A set of permissions said as words: the names of the ones it may use, or
// `none` when it may use none. Freedoms lists freedom_names and answers may().
template <class Freedoms>
string freedoms_said(const Freedoms& allowed, const char* none) {
    string words;
    for (int freedom = 0; freedom < Freedoms::FREEDOMS; ++freedom) {
        if (!allowed.may(freedom)) continue;
        if (!words.empty()) words += ", ";
        words += Freedoms::freedom_names[freedom];
    }
    return words.empty() ? none : words;
}

// A reading combines a segmentation with the changes allowed when matching objects.
struct degrees_of_freedom {
    bool translate = true;
    bool pose = true;
    bool recolor = true;
    // Allow part of an object to extend beyond the target grid's border.
    bool hang_off = true;
    // Each of the four permissions can be on or off, giving 2^4 = 16 combinations.
    static constexpr int FREEDOMS = 4;
    static constexpr int COMBINATIONS = 1 << FREEDOMS;
    static constexpr degrees_of_freedom from_index(int packed) {
        return { (packed & 1) != 0, (packed & 2) != 0, (packed & 4) != 0,
                 (packed & 8) != 0 };
    }
    constexpr int index() const {
        return (translate ? 1 : 0) | (pose ? 2 : 0) | (recolor ? 4 : 0) |
               (hang_off ? 8 : 0);
    }
    // How many of the eight ways round a match may try: all, or only as it is.
    constexpr int poses() const { return pose ? POSE_COUNT : 1; }

    // Heading and permission labels used to describe a reading.
    static constexpr const char* freedom_names_lead = "And It May";
    static constexpr const char* const freedom_names[FREEDOMS] =
        { "Move", "Turn", "Color", "Hang Off" };
    constexpr bool may(int freedom) const {
        return freedom == 0 ? translate : freedom == 1 ? pose
             : freedom == 2 ? recolor   : hang_off;
    }
    string label() const { return freedoms_said(*this, "As Is"); }
};

// A match links an object to another object or a position on the other grid.
struct object_matching {
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
        // Target color for a recolored single-color object; -1 if not applicable.
        int8_t recolored_to = -1;
        // Target color for each source color; -1 means that source color is absent.
        recoloring color_map;
        // Every pose that lands the object here in these colors, its own among
        // them; bit p is pose p. Said for blocks, where a symmetric copy lands
        // several ways and the rule that places the copies may need any of them.
        uint8_t ways_round = 0;
    };
    // Why a match that was found did not survive.
    enum refusal_reason : int8_t { lost_to_chance, cells_already_explained, overlaps_a_kept_object,
                    the_other_examples_disagreed };
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
        "IT OVERLAPS AN OBJECT THAT WAS KEPT",
        "THE OTHER EXAMPLES DID NOT AGREE" };
    // Store each color's move for this training pair, only if every read pair has
    // exactly one move for that color. The move may differ between pairs.
    struct what_the_pairs_agreed {
        bool color_speaks[COLOR_COUNT] = {};
        int8_t down[COLOR_COUNT] = {}, across[COLOR_COUNT] = {};
    };

    // How a pass decided, so a giveup can never be mistaken for an answer.
    enum class fit_verdict : int8_t { fitted, no_fit, gave_up };

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

    // Movement from input to output in the original grid coordinates.
    // Reverse an output-to-input correspondence so both directions agree.
    static pair<int, int> move_of(const edge& edge,
                                  const object_info& one_object) {
        const int down = edge.row - one_object.summary.y0;
        const int across = edge.col - one_object.summary.x0;
        return edge.from_input ? pair<int, int>{down, across}
                               : pair<int, int>{-down, -across};
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
    static fit_verdict collect_edges(const vector<object_info>& from_info,
                              const vector<object>& from_objects,
                              const vector<object_info>& into_info,
                              const vector<object>& into_objects,
                              const grid& into, bool from_input,
                              int8_t background, bool may_hang_off,
                              vector<edge>& edges,
                              vector<rejection>* discarded = nullptr,
                              degrees_of_freedom allowed = {},
                              const what_the_pairs_agreed* agreed = nullptr) {
        vector<edge> found =
            gather_matches(from_info, from_objects, into_info, into,
                           background, may_hang_off, discarded, allowed);
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

        const fitting laid_out = set_up_the_fitting(found, from_objects, into_objects, into);

        vector<int> chosen;
        const fit_verdict searched =
            fit_the_objects_together(laid_out, chosen);
        if (searched == fit_verdict::fitted) {
            keep_what_fitted(chosen, laid_out, found, edges, discarded);
            return fit_verdict::fitted;
        }

        fall_back_to_greedy(found, from_objects, into_objects, into,
                            edges, discarded);
        return searched;
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
            edge.row = into_info[best].summary.y0;
            edge.col = into_info[best].summary.x0;
            edge.exact = false;
            edge.same_shape = false;
            edges.push_back(edge);
        }
    }

    // Link only the topmost, then leftmost complete motifs.
    static fit_verdict collect_representative_edge(
            const vector<object_info>& from_info, const vector<object>& from_objects,
            const vector<object_info>& into_info, const grid& into, int8_t background,
            int step_w, int step_h, vector<edge>& edges, degrees_of_freedom allowed) {
        auto first = [&](const vector<object_info>& info) {
            int best = -1;
            for (int i = 0; i < (int)info.size(); ++i) {
                const auto& box = info[i].summary;
                if (box.w != step_w || box.h != step_h) continue;
                if (best < 0 || pair{box.y0, box.x0} <
                    pair{info[best].summary.y0, info[best].summary.x0}) best = i;
            }
            return best;
        };
        const int source = first(from_info), target = first(into_info);
        if (source < 0 || target < 0) return fit_verdict::no_fit;
        const auto& landing = into_info[target].summary;
        const bool stayed = same_corner(from_info[source].summary, landing);
        if (!allowed.translate && !stayed) return fit_verdict::no_fit;
        const auto matches = find_in(from_objects[source].drawn(), into, background,
                                    false, landing.y0, landing.x0, true, allowed.poses());
        const object_match* best = nullptr;
        for (const auto& hit : matches.hits) {
            if (hit.hidden || hit.outside || (!allowed.recolor && !hit.exact)) continue;
            if (!best || (hit.exact && !best->exact)) best = &hit;
        }
        if (!best) return fit_verdict::no_fit;
        auto link = edge_from_match(*best, source, from_info[source].color, stayed);
        link.to = (int16_t)target;
        edges.push_back(link);
        return fit_verdict::fitted;
    }

    // Match repeated blocks in both directions so every matching block gets a partner.
    // Prefer a source block at the same position when possible.
    static fit_verdict collect_motif_edges(
            const vector<object_info>& from_info,
            const vector<object>& from_objects,
            const vector<object_info>& into_info,
            const grid& into, int8_t background, int step_w, int step_h,
            vector<edge>& edges, degrees_of_freedom allowed = {}) {
        if (step_w <= 0 || step_h <= 0) return fit_verdict::no_fit;

        // Does this from block sit at that place on the other grid? The answer is
        // the same question both sweeps ask, so it is written once. `ways` is
        // every way round it sits there in the colors of the one kept.
        auto sits_at = [&](int from_index, const object_info& target,
                           object_match& found, uint8_t& ways) {
            // A copy on another block has moved, so it needs the freedom to --
            // the same rule the representative edge above keeps.
            if (!allowed.translate && !same_corner(from_info[from_index].summary, target.summary))
                return false;
            const grid cells = from_objects[from_index].drawn();
            const object_matches matches =
                find_in(cells, into, background, true,
                        target.summary.y0, target.summary.x0, true, allowed.poses());
            // Prefer a match with unchanged colors, even if it uses a later rotation or reflection.
            const object_match* best = nullptr;
            for (const object_match& hit : matches.hits) {
                if (!allowed.recolor && !hit.exact) continue;
                if (!best || (hit.exact && !best->exact)) best = &hit;
                if (best->exact) break;
            }
            if (!best) return false;
            found = *best;
            // A symmetric block sits several ways round, and find_in tries each
            // distinct drawing once, so the ways that draw alike are read off the
            // drawings themselves.
            ways = 0;
            for (const object_match& hit : matches.hits) {
                if (hit.exact != best->exact || hit.color_map != best->color_map) continue;
                const grid landed = posed_cells(cells, hit.pose);
                for (int pose = 0; pose < POSE_COUNT; ++pose)
                    if (posed_cells(cells, pose) == landed) ways |= (uint8_t)(1 << pose);
            }
            return true;
        };
        auto draw = [&](int from_index, int target_index, const object_match& hit, uint8_t ways) {
            const auto& target = into_info[target_index];
            auto edge = edge_from_match(hit, from_index, from_info[from_index].color,
                                        same_corner(from_info[from_index].summary, target.summary));
            edge.to = (int16_t)target_index;
            edge.ways_round = ways;
            edges.push_back(edge);
        };

        vector<bool> answered(from_info.size(), false);
        // Prefer a matching source at the target's position, then the first match elsewhere.
        for (int target_index = 0; target_index < (int)into_info.size(); ++target_index) {
            const auto& target = into_info[target_index];
            int source = -1;
            object_match hit;
            uint8_t ways = 0;
            for (int i = 0; i < (int)from_info.size(); ++i)
                if (same_corner(from_info[i].summary, target.summary) && sits_at(i, target, hit, ways)) {
                    source = i;
                    break;
                }
            for (int i = 0; source < 0 && i < (int)from_info.size(); ++i)
                if (sits_at(i, target, hit, ways)) source = i;
            if (source < 0) continue;
            draw(source, target_index, hit, ways);
            answered[source] = true;
        }
        // WHERE DID EACH OBJECT GO. Only the ones no landing claimed, so a from side
        // bigger than the into side is not silently dropped.
        for (int i = 0; i < (int)from_info.size(); ++i) {
            if (answered[i]) continue;
            object_match hit;
            uint8_t ways = 0;
            for (int target_index = 0; target_index < (int)into_info.size(); ++target_index)
                if (sits_at(i, into_info[target_index], hit, ways)) {
                    draw(i, target_index, hit, ways);
                    answered[i] = true;
                    break;
                }
        }
        return edges.empty() ? fit_verdict::no_fit : fit_verdict::fitted;
    }

private:
    // 5. How possible placements become a consistent set of matches.

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

    // Search hits and the number of placements tried, counting distinct poses only.
    struct object_matches {
        vector<object_match> hits;
        // Placements summed over the poses that are actually different
        int places = 0;
    };
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
    // footprint, the blank cells must land on background and the lit ones off it.
    static bool fits_as_drawn(const placement& at, int8_t background, bool match_footprint,
                              object_match& hit) {
        hit.color_map = {};
        hit.exact = true;
        for (int row = 0; row < at.posed.height; ++row)
            for (int col = 0; col < at.posed.width; ++col) {
                if (!at.lands(row, col)) continue;
                const int object_color = at.posed.data[row][col];
                const int there_color = at.under(row, col);
                if (object_color == TRANSPARENT) {
                    if (match_footprint && there_color != background) return false;
                    continue;
                }
                if (there_color == TRANSPARENT) return false;
                if (match_footprint && there_color == background) return false;
                if (!is_color(object_color) || !is_color(there_color)) return false;
                if (hit.color_map[object_color] == -1) hit.color_map[object_color] = (int8_t)there_color;
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
    static bool fits_behind(const placement& at, int8_t background, object_match& hit) {
        for (int i = 0; i < COLOR_COUNT; ++i) hit.color_map[i] = (int8_t)i;
        hit.exact = true;
        hit.hidden = 0;
        for (int row = 0; row < at.posed.height; ++row)
            for (int col = 0; col < at.posed.width; ++col) {
                const int object_color = at.posed.data[row][col];
                if (object_color == TRANSPARENT || !at.lands(row, col)) continue;
                const int there_color = at.under(row, col);
                if (there_color == TRANSPARENT || there_color == background) return false;
                hit.hidden += (object_color != there_color);
            }
        return true;
    }

    // A BLOCK IS A WHOLE PICTURE, ITS GROUND INCLUDED. Where the footprints
    // disagree it may still be the same picture with its colors traded, and a
    // negative is exactly that: the ground and the figure swapping, which the
    // Color freedom already says. One to one, or a block of one color would
    // match every block.
    static bool fits_as_negative(const placement& at, int8_t background, object_match& hit) {
        recoloring came_from;
        hit.color_map = {};
        hit.exact = true;
        for (int row = 0; row < at.posed.height; ++row)
            for (int col = 0; col < at.posed.width; ++col) {
                if (!at.lands(row, col)) continue;
                const int was = at.posed.data[row][col] == TRANSPARENT ? (int)background
                                                                       : at.posed.data[row][col];
                const int now = at.under(row, col);
                if (!is_color(was) || !is_color(now)) return false;
                if (hit.color_map[was] == -1 && came_from[now] == -1) {
                    hit.color_map[was] = (int8_t)now;
                    came_from[now] = (int8_t)was;
                }
                if (hit.color_map[was] != now || came_from[now] != was) return false;
                if (was != now) hit.exact = false;
            }
        return true;
    }

    // Searches for an object in another grid.
    static object_matches find_in(const grid& cells, const grid& where, int8_t background,
                                  bool may_hang_off, int fixed_row = -1, int fixed_col = -1,
                                  bool match_footprint = false, int max_poses = -1) {
        object_matches found;
        if (cells.empty()) return found;
        const bool fixed = fixed_row >= 0 && fixed_col >= 0;
        const int pose_limit = max_poses >= 0 ? max_poses : (fixed ? 1 : POSE_COUNT);

        vector<grid> tried;
        for (int pose = 0; pose < pose_limit; ++pose) {
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
            found.places += (last_row - first_row + 1) * (last_col - first_col + 1);
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
                    // a block on a known ground, its negative.
                    if (fits_as_drawn(at, background, match_footprint, hit) ||
                        (!match_footprint && fits_behind(at, background, hit)) ||
                        (match_footprint && is_color(background) && fits_as_negative(at, background, hit)))
                        found.hits.push_back(hit);
                }
        }
        return found;
    }

    // Did any cell of the object land on a non-background cell? Background has no
    // shape, so every shape fits it, and a match there is no evidence.
    static bool landed_on_non_background(const edge& edge,
                                         const grid& cells, const grid& into,
                                         int8_t background) {
        const grid posed = posed_cells(cells, edge.pose);
        for (int row = 0; row < posed.height; ++row)
            for (int col = 0; col < posed.width; ++col) {
                if (posed.data[row][col] == TRANSPARENT) continue;
                const int into_row = edge.row + row, into_col = edge.col + col;
                if (!into.contains(into_row, into_col)) continue;
                if (into.data[into_row][into_col] != background) return true;
            }
        return false;
    }

    // Measures how common each color is to help estimate the chance of an accidental match.
    static void color_shares(const grid& one_grid, double share[COLOR_COUNT]) {
        int counts[COLOR_COUNT], total = 0;
        count_colors(one_grid, counts);
        for (int i = 0; i < COLOR_COUNT; ++i) total += counts[i];
        for (int i = 0; i < COLOR_COUNT; ++i) share[i] = total ? (double)counts[i] / total : 0.0;
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
    static edge edge_from_match(const object_match& hit, int from_index,
                                int source_color, bool stayed_put) {
        object_matching::edge edge;
        edge.from = (int16_t)from_index;
        edge.row = (int8_t)hit.row; edge.col = (int8_t)hit.col;
        edge.pose = (int8_t)hit.pose;
        edge.exact = hit.exact;
        edge.stayed_put = stayed_put;
        edge.hidden = (int16_t)hit.hidden;
        edge.outside = (int16_t)hit.outside;
        edge.recolored_to = (!hit.exact && source_color >= 0)
                          ? hit.color_map[source_color] : (int8_t)-1;
        if (!hit.exact) edge.color_map = hit.color_map;
        return edge;
    }

    // Find possible matches for each object in the other grid.
    static vector<edge> gather_matches(
            const vector<object_info>& from_info,
            const vector<object>& from_objects,
            const vector<object_info>& into_info,
            const grid& into, int8_t background, bool may_hang_off,
            vector<rejection>* discarded, degrees_of_freedom allowed) {
        double share[COLOR_COUNT];
        color_shares(into, share);

        // Check for repeated shapes in the target grid before allowing clipped matches.
        auto twins_over_there = [&](uint32_t shape) {
            int seen = 0;
            for (const auto& landing : into_info)
                if (landing.shape == shape && ++seen > 1) return true;
            return false;
        };

        vector<edge> found;

        for (size_t from_index = 0; from_index < from_objects.size(); ++from_index) {
            const auto& one_object = from_info[from_index];
            const grid cells = from_objects[from_index].drawn();

            // Positional: own place first, then anywhere only if it was not there.
            auto keep_positional = [&](const object_matches& found_here, bool stationary) {
                for (const object_match& hit : found_here.hits) {
                    if (!allowed.recolor && !hit.exact) continue;
                    if (!stationary && hit.pose == 0 &&
                        hit.row == one_object.summary.y0 && hit.col == one_object.summary.x0)
                        continue;
                    // Estimate accidental-match odds using visible cells only.
                    const grid posed = posed_cells(cells, hit.pose);
                    double odds = 1.0;
                    for (int row = 0; row < posed.height; ++row)
                        for (int col = 0; col < posed.width; ++col) {
                            const int object_color = posed.data[row][col];
                            if (object_color == TRANSPARENT) continue;
                            const int into_row = hit.row + row, into_col = hit.col + col;
                            if (!into.contains(into_row, into_col)) continue;
                            const int into_color = into.data[into_row][into_col];
                            if (hit.hidden && into_color != object_color) continue;
                            odds *= share[hit.color_map[object_color]];
                        }
                    auto edge = edge_from_match(hit, (int)from_index, one_object.color, stationary);
                    const bool stands_alone = beats_chance(found_here.places, odds);
                    // EXPERIMENT: every match must pass the chance test, not only the
                    // stationary, hidden, or recolored ones.
                    if (!stands_alone) {
                        throw_away(edge, object_matching::lost_to_chance, discarded);
                        continue;
                    }
                    found.push_back(edge);
                }
            };

            // Search both the original position and other positions: an object can stay and be copied.
            keep_positional(
                find_in(cells, into, background, false,
                        one_object.summary.y0, one_object.summary.x0, false, allowed.poses()), true);
            // Allow clipped matches only when copies establish the shape on either grid.
            if (allowed.translate)
                keep_positional(
                    find_in(cells, into, background,
                            may_hang_off && (one_object.motif >= 0 ||
                                             twins_over_there(one_object.shape)),
                            -1, -1, false, allowed.poses()), false);

            // Source 2 — shape: objects there with the same canonical hash (count
            // equality guards hash collisions). Skip placements source 1 found.
            if (!allowed.translate) continue;
            for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
                const auto& landing = into_info[into_index];
                if (landing.shape != one_object.shape || landing.summary.count != one_object.summary.count) continue;
                if (same_corner(landing.summary, one_object.summary)) continue;
                if (ranges::any_of(found, [&](const edge& seen) {
                        return seen.from == (int16_t)from_index &&
                               seen.row == landing.summary.y0 && seen.col == landing.summary.x0;
                    })) continue;
                // Estimate chance from the target color count, or all grid colors for multicolor objects.
                double odds = 1.0;
                if (landing.color >= 0)
                    for (int i = 0; i < landing.summary.count; ++i) odds *= share[landing.color];
                else
                    walk_colored_cells(into, TRANSPARENT,
                                       [&](int, int, int color) { odds *= share[color]; });
                object_matching::edge edge;
                edge.from = (int16_t)from_index;
                edge.to = (int16_t)into_index;
                edge.row = landing.summary.y0; edge.col = landing.summary.x0;
                edge.exact = (landing.color == one_object.color);
                if (!allowed.recolor && !edge.exact) continue;
                edge.recolored_to = (!edge.exact && one_object.color >= 0) ? landing.color : (int8_t)-1;
                if (!edge.exact && is_color(one_object.color))
                    edge.color_map[one_object.color] = landing.color;
                const bool stands_alone = beats_chance(
                    (into.width - landing.summary.w + 1) * (into.height - landing.summary.h + 1), odds);
                // Same bar as source 1: a recolor is the weaker claim either way it
                // was found, so it earns its place alone or not at all.
                if (!edge.exact && !stands_alone) {
                    throw_away(edge, object_matching::lost_to_chance, discarded);
                    continue;
                }
                found.push_back(edge);
            }
        }
        // Asked once, of every source, where the whole list is in one place.
        vector<edge> on_non_background;
        for (size_t i = 0; i < found.size(); ++i) {
            if (landed_on_non_background(found[i], from_objects[found[i].from].drawn(),
                                         into, background)) {
                on_non_background.push_back(found[i]);
                continue;
            }
            throw_away(found[i], object_matching::lost_to_chance,
                       discarded);
        }
        return on_non_background;
    }

    // One candidate object: the match it came from, the cells it paints, and the
    // cells of its own that something else paints — which is what puts it BEHIND.
    struct candidate {
        size_t match_index = 0;
        vector<int> cells, behind;
    };

    // The objects, and which cells each of them could paint.
    struct fitting {
        // Number of target cells, including background.
        size_t cell_count = 0;
        // One per match with visible cells in the grid
        vector<candidate> objects;
        // Per cell: the objects that can paint it, best first
        vector<vector<int>> options;
        // Cells an EXACT object could account for — only these drive the search
        vector<bool> required;
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
    // Lower scores prefer matches that did not move, change color, turn or flip,
    // or show only part of themselves, in that order of importance. Hanging off
    // the grid and hiding behind another object are the one freedom: some of the
    // piece is not shown. Free to hide, a piece half behind a frame cost no more
    // than the same piece in plain sight beside it.
    static int freedom_spent(const edge& edge) {
        return (edge.stayed_put ? 0 : 8) + (edge.exact ? 0 : 4) +
               (edge.pose == 0 ? 0 : 2) + (edge.outside == 0 && edge.hidden == 0 ? 0 : 1);
    }

    // HOW FAR A PIECE TRAVELLED to land here: a slide along one line before one
    // that needs both, then the shorter. Asked only once cells and freedoms tie,
    // which is where the order the pieces were listed in used to decide which
    // copy a landing names -- 06df4c85 credited a block between two reds to a
    // red in another row entirely, for being listed first.
    static int travel_of(const edge& edge, const object_summary& source) {
        const int down = edge.row - source.y0, across = edge.col - source.x0;
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

    // For each target cell, list the matches that can cover it.
    static fitting set_up_the_fitting(const vector<edge>& found,
                             const vector<object>& from_objects,
                             const vector<object>& into_objects,
                             const grid& into) {
        fitting laid_out;
        // Index all target cells, including background where new objects may appear.
        // Only cells reached by exact matches become required.
        laid_out.cell_count = (size_t)into.width * into.height;
        laid_out.travel.resize(found.size());
        for (size_t found_index = 0; found_index < found.size(); ++found_index)
            laid_out.travel[found_index] =
                travel_of(found[found_index], from_objects[found[found_index].from].summary);

        // Include recolored matches as candidates: their shapes still need to fit.
        for (size_t found_index = 0; found_index < found.size(); ++found_index) {
            const grid cells = edge_footprint(found[found_index], from_objects, into_objects);
            candidate candidate{ found_index, {}, {} };
            for (int row = 0; row < cells.height; ++row)
                for (int col = 0; col < cells.width; ++col) {
                    if (cells.data[row][col] == TRANSPARENT) continue;
                    const int into_row = found[found_index].row + row;
                    const int into_col = found[found_index].col + col;
                    // Off the edge is not a cell of this grid at all: it is
                    // neither painted nor hidden, it was simply never shown.
                    if (!into.contains(into_row, into_col)) continue;
                    const int id = into_row * into.width + into_col;
                    // The cells it is hidden at are its own too — kept apart,
                    // because they are what puts it BEHIND whoever paints them.
                    if (!landing_shows(found[found_index], cells, into, row, col)) {
                        candidate.behind.push_back(id);
                        continue;
                    }
                    candidate.cells.push_back(id);
                }
            if (!candidate.cells.empty()) laid_out.objects.push_back(std::move(candidate));
        }

        laid_out.options.resize(laid_out.cell_count);
        for (int object_index = 0; object_index < (int)laid_out.objects.size(); ++object_index)
            for (int id : laid_out.objects[object_index].cells)
                laid_out.options[id].push_back(object_index);
        // Try larger matches first, then lower change scores; the search takes its first solution.
        for (auto& options : laid_out.options)
            stable_sort(options.begin(), options.end(), [&](int left, int right) {
                return laid_out.comes_first(left, right, found);
            });

        // Require only cells reached by exact matches; other cells may be newly created.
        laid_out.required.assign(laid_out.cell_count, false);
        for (const auto& candidate : laid_out.objects)
            if (found[candidate.match_index].exact)
                for (int id : candidate.cells) laid_out.required[id] = true;
        return laid_out;
    }

    // Find objects that cover each required cell exactly once; store them in `chosen`.
    static fit_verdict fit_the_objects_together(const fitting& laid_out,
                                    vector<int>& chosen) {
        if (!ranges::any_of(laid_out.required, [](bool needed) { return needed; }))
            return fit_verdict::no_fit;

        vector<int> painted_by(laid_out.cell_count, -1);
        long long work = 0;
        // Limit the search based on problem size, counting each cell checked.
        // Report when the limit is reached; a solution may still exist.
        const long long work_budget =
            (long long)(laid_out.objects.size() + 1) * (long long)(laid_out.cell_count + 1);
        bool out_of_budget = false;

        // Reject an object if it would be both in front of and behind another object.
        auto contradicts = [&](int object_index) {
            for (int behind_id : laid_out.objects[object_index].behind) {
                const int in_front = painted_by[behind_id];
                if (in_front < 0) continue;
                for (int its_behind_id : laid_out.objects[in_front].behind)
                    if (painted_by[its_behind_id] == object_index) return true;
            }
            return false;
        };

        auto cells_are_free = [&](int object_index) {
            for (int id : laid_out.objects[object_index].cells) {
                ++work;
                if (painted_by[id] >= 0) return false;
            }
            return true;
        };

        // WHETHER AN OBJECT IS STILL FREE, asked once a step. It covers many
        // cells, and asked again at every one of them a step cost the square of
        // the biggest object: a frame of 273 cells asked 273 times is 74,529
        // checks before anything is chosen. 06df4c85 spent its whole budget on
        // that first look in 206 of its 240 fittings, one of them with a single
        // object to choose, and the greedy pass decided those readings instead.
        // Stamped with the step that asked, so starting a step clears nothing.
        vector<int> asked_at(laid_out.objects.size(), 0);
        vector<bool> free_now(laid_out.objects.size(), false);
        int steps = 0;
        auto fits_together = [&](auto&& self) -> bool {
            if (work > work_budget) { out_of_budget = true; return false; }
            const int this_step = ++steps;
            auto is_free = [&](int object_index) {
                if (asked_at[object_index] != this_step) {
                    asked_at[object_index] = this_step;
                    free_now[object_index] = cells_are_free(object_index);
                }
                return (bool)free_now[object_index];
            };
            // The cell with the fewest objects still able to paint it decides next;
            // zero options means this line of choices is dead. Only required cells
            // drive the search — a made cell is answered by being left alone.
            int target = -1, fewest = 0;
            for (size_t id = 0; id < laid_out.cell_count; ++id) {
                if (painted_by[id] >= 0 || !laid_out.required[id]) continue;
                int options = 0;
                for (int object_index : laid_out.options[id]) {
                    ++work;
                    options += is_free(object_index);
                }
                if (options == 0) return false;
                if (target < 0 || options < fewest) { target = (int)id; fewest = options; }
                // One object left to paint it is no choice at all: take it now.
                // A cell further on with none left is still found, one step later.
                if (fewest == 1) break;
                if (work > work_budget) { out_of_budget = true; return false; }
            }
            if (target < 0) return true;   // every carried-over cell accounted for
            // Try candidates in preference order; an object does not have to be selected.
            for (int object_index : laid_out.options[target]) {
                if (!cells_are_free(object_index)) continue;
                for (int id : laid_out.objects[object_index].cells) painted_by[id] = object_index;
                chosen.push_back(object_index);
                if (!contradicts(object_index) && self(self)) return true;
                chosen.pop_back();
                for (int id : laid_out.objects[object_index].cells) painted_by[id] = -1;
                if (out_of_budget) return false;
            }
            return false;
        };

        if (fits_together(fits_together)) return fit_verdict::fitted;
        return out_of_budget ? fit_verdict::gave_up : fit_verdict::no_fit;
    }

    // Keep the chosen matches and any extras that do not overlap; log rejected overlaps.
    static void keep_what_fitted(vector<int>& chosen, const fitting& laid_out,
                               const vector<edge>& found,
                               vector<edge>& edges,
                               vector<rejection>* discarded) {
        sort(chosen.begin(), chosen.end());
        vector<bool> kept(found.size(), false);
        vector<bool> painted(laid_out.cell_count, false);
        for (int object_index : chosen) {
            const size_t match_index = laid_out.objects[object_index].match_index;
            edges.push_back(found[match_index]);
            kept[match_index] = true;
            for (int id : laid_out.objects[object_index].cells) painted[id] = true;
        }
        // Add matches that cover cells the solution left free, best matches first.
        vector<int> leftovers;
        for (int object_index = 0; object_index < (int)laid_out.objects.size(); ++object_index)
            if (!kept[laid_out.objects[object_index].match_index] &&
                !laid_out.objects[object_index].cells.empty())
                leftovers.push_back(object_index);
        stable_sort(leftovers.begin(), leftovers.end(), [&](int left, int right) {
            return laid_out.comes_first(left, right, found);
        });
        for (int object_index : leftovers) {
            const size_t match_index = laid_out.objects[object_index].match_index;
            if (kept[match_index]) continue;
            if (ranges::any_of(laid_out.objects[object_index].cells,
                               [&](int id) { return painted[id]; })) continue;
            edges.push_back(found[match_index]);
            kept[match_index] = true;
            for (int id : laid_out.objects[object_index].cells) painted[id] = true;
        }
        for (size_t found_index = 0; found_index < found.size(); ++found_index)
            if (!kept[found_index])
                throw_away(found[found_index], object_matching::overlaps_a_kept_object,
                           discarded);
    }

    // Try better matches first; keep each only if it covers at least one new cell.
    static void fall_back_to_greedy(const vector<edge>& found,
                                    const vector<object>& from_objects,
                                    const vector<object>& into_objects,
                                    const grid& into,
                                    vector<edge>& edges,
                                    vector<rejection>* discarded) {
        auto footprint_of = [&](const edge& edge) {
            return edge_footprint(edge, from_objects, into_objects);
        };
        // Use the same match preference as the fitting search, and the same
        // measure of it: the cells a match SHOWS, not every cell of its footprint.
        // Counted whole, a piece half behind a frame tied with the one in plain
        // sight beside it, and whichever the scan met first was kept.
        auto shown_cells = [&](const edge& edge) {
            const grid cells = footprint_of(edge);
            size_t shown = 0;
            for (int row = 0; row < cells.height; ++row)
                for (int col = 0; col < cells.width; ++col)
                    shown += landing_shows(edge, cells, into, row, col);
            return shown;
        };
        auto travel = [&](const edge& edge) {
            return travel_of(edge, from_objects[edge.from].summary);
        };
        vector<size_t> strongest_first(found.size());
        iota(strongest_first.begin(), strongest_first.end(), (size_t)0);
        stable_sort(strongest_first.begin(), strongest_first.end(),
                    [&](size_t left, size_t right) {
            return says_more(found[left], shown_cells(found[left]),
                             found[right], shown_cells(found[right]),
                             travel(found[left]), travel(found[right]));
        });

        bool taken[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        for (size_t found_index : strongest_first) {
            const grid cells = footprint_of(found[found_index]);
            const int at_row = found[found_index].row, at_col = found[found_index].col;
            bool adds = false;
            for (int row = 0; row < cells.height && !adds; ++row)
                for (int col = 0; col < cells.width; ++col)
                    if (landing_shows(found[found_index], cells, into, row, col) &&
                        !taken[at_row + row][at_col + col]) {
                        adds = true; break;
                    }
            if (!adds) {
                throw_away(found[found_index], object_matching::cells_already_explained,
                           discarded);
                continue;
            }
            for (int row = 0; row < cells.height; ++row)
                for (int col = 0; col < cells.width; ++col)
                    if (landing_shows(found[found_index], cells, into, row, col))
                        taken[at_row + row][at_col + col] = true;
            edges.push_back(found[found_index]);
        }
    }

};

} // namespace arc
