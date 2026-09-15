// Segment grids into objects and match them across training pairs.
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
// Connected objects use 8-way connectivity: edges or corners.
// Surrounding space and holes use 4-way connectivity: edges only.
// Segmentation methods are compared by how well they explain the training pairs.
enum class segmentation_methods : int8_t {
    connected_monochrome, connected_multicolor, all_cells_of_one_color,
    minimal_repeating_set, smallest_of_two_grids, count
};

// Label used for GUI
inline constexpr const char* segmentation_names_lead = "An Object Is";

// GUI and logs
inline constexpr const char* const segmentation_names[] = {
    "8-Connected Monochrome Cells",
    "8-Connected Multicolor Cells",
    "All Cells Of One Color",
    "Minimal Repeating Set Of Cells",
    "Smallest Of The Two Grids"
};

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
    // Heading and permission labels used to describe a reading.
    static constexpr const char* freedom_names_lead = "And It May";
    static constexpr const char* const freedom_names[FREEDOMS] =
        { "Move", "Turn", "Color", "Hang Off" };
    constexpr bool may(int freedom) const {
        return freedom == 0 ? translate : freedom == 1 ? pose
             : freedom == 2 ? recolor   : hang_off;
    }
    string label() const {
        string words;
        for (int freedom = 0; freedom < FREEDOMS; ++freedom) {
            if (!may(freedom)) continue;
            if (!words.empty()) words += ", ";
            words += freedom_names[freedom];
        }
        return words.empty() ? "As Is" : words;
    }
};

// Correspondences match objects from input to output and from output to input.

// This structure collects the correspondences and summaries of the readings tried.
struct correspondence_facts {
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
        int8_t color_map[COLOR_COUNT] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    };
    struct object_info {
        // Component color; -1 = the whole-grid object
        int8_t color = -1;
        // Bounding box and cell count, defined in 1_readings.hpp.
        object_summary summary;
        // Canonical shape hash (orientationless, positionless)
        uint32_t shape = 0;
        // Number of matches touching this object, counted after the matches are final.
        int edges_touching = 0;
        // Whether a match identifies a partner on the other grid.
        bool has_partner = false;
        // Copy group on this grid; -1 means no group.
        int16_t motif = -1;
        // True for the group member with the fewest cell differences from the others.
        bool medoid = false;
        // Background region on the other grid touched by an unmatched object.
        // Width 0 means no region was found; enclosed means a hole.
        int8_t blank_left = 0, blank_top = 0, blank_width = 0, blank_height = 0;
        int16_t blank_cells = 0;
        bool blank_enclosed = false;
    };
    // An uncovered cell. made_or_gone means its object has no identified partner.
    struct left_over_cell {
        bool from_input = true;
        int8_t row = 0, col = 0;
        bool made_or_gone = false;
    };
    // Why a match that was found did not survive.
    enum : int8_t { lost_to_chance, cells_already_explained, overlaps_a_kept_object,
                    the_other_examples_disagreed };
    // Said the same way wherever it is said. Each one finishes "it was refused
    // because", so it has to be a reason on its own and not a name for one.
    static constexpr const char* const refusal_names[] = {
        "IT WAS LIKELY BY LUCK",
        "ITS CELLS WERE ALREADY EXPLAINED",
        "IT OVERLAPS AN OBJECT THAT WAS KEPT",
        "THE OTHER EXAMPLES DID NOT AGREE" };
    struct pair_info {
        // Objects from both grids, segmented using the same method and block size.
        vector<object_info> input_info;
        vector<object_info> output_info;
        // Matches in both directions, with duplicate object pairs merged.
        vector<edge> edges;
        // Matches that were found and then thrown away, with why — otherwise
        // "never matched" and "matched and lost" look the same in the log.
        vector<edge> discarded;
        vector<int8_t> discard_why;
        // 0 = fitted, 1 = greedy fallback, 2 = search budget exhausted.
        int8_t verdict = 0;
        // Unexplained cells on both grids and the number of matches used.
        vector<left_over_cell> left_over;
        int objects_used = 0;
        // Movements shared by at least two input objects.
        // Color is their shared input color, or -1 if their colors differ.
        struct moved_together { int8_t down, across, color; int16_t objects; };
        vector<moved_together> agreed_moves;
        // Average compression cost across both grids; independent of matching.
        double describe_cost = 1.0;
        // The same three questions from each end: where an object went, where a
        // landing came from. Nowhere/nothing, one, several.
        int deleted = 0, persisted = 0, copied = 0;
        int invented = 0, from_one = 0, from_many = 0;
        // Maximum matches leaving one input object and arriving at one output object.
        int most_landings = 0, most_sources = 0;
        // Counts of exact, recolored, overlap-only, hidden, rotated/flipped, and rejected matches.
        int found_exact = 0, found_recolored = 0, sits_on = 0;
        int partly_hidden = 0, turned = 0, dropped = 0;
        // Non-background output cells covered by color-preserving matches.
        int exact_covered_cells = 0;
        // Non-background into cells no exact edge covers — made, not carried
        int created_cells = 0;
    };
    // One entry per train pair
    vector<pair_info> pairs;
    // Total non-background cells in all training grids, used to normalize leftover counts.
    int cells = 0;
    // The selected segmentation, shared by all training pairs.
    segmentation_methods segmentation = segmentation_methods::connected_monochrome;
    // Scores for each segmentation and permission combination.
    // Table index: segmentation * degrees_of_freedom::COMBINATIONS + allowed.index().
    struct reading {
        // Which segmentation, and what a match was allowed to change
        segmentation_methods segmentation = segmentation_methods::connected_monochrome;
        degrees_of_freedom allowed;
        // How many objects the segmentation FOUND, and how many it USED — the log prints
        // them as "segmented into X, used Y"
        int objects_found = 0, objects_used = 0;
        int left_over = 0, box_cells = 0;
        // Leftover cells classified as created or removed because their object has no partner.
        int made = 0, gone = 0;
        // Average compression cost across training pairs, independent of matching.
        double describe_cost = 1.0;
        // Whether any untransformed object fits the other grid in each pair.
        // Used for preliminary ordering and logging; matching still runs when false.
        bool objects_can_fit = true;
        int exact = 0, recolored = 0, sits_on = 0;
    };
    vector<reading> readings;
    // Index of the selected reading in readings.
    int8_t chosen = 0;
    // Indices of other segmentations whose best readings tie with the winner.
    vector<int8_t> tied_with_chosen;
    // Cached results for every reading, in readings order.
    // pairs holds a copy of reading_pairs[chosen].
    vector<vector<pair_info>> reading_pairs;
    bool valid = false;
};

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
    int8_t color_map[COLOR_COUNT];
};

// Search hits and the number of placements tried, counting distinct poses only.
struct object_matches {
    vector<object_match> hits;
    // Placements summed over the poses that are actually different
    int places = 0;
};

// An object's cells drawn out in one of the eight ways round (1_readings.hpp).
inline grid posed_cells(const grid& cells, int pose) {
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

inline int lit_cells(const grid& cells) {
    int lit = 0;
    for (int row = 0; row < cells.height; ++row)
        for (int col = 0; col < cells.width; ++col)
            lit += (cells.data[row][col] != TRANSPARENT);
    return lit;
}

// Searches for an object in another grid.
inline object_matches find_in(const grid& cells, const grid& where, int8_t background,
                              bool may_hang_off, int fixed_row = -1, int fixed_col = -1,
                              bool match_footprint = false, int max_poses = -1) {
    object_matches found;
    if (cells.width <= 0 || cells.height <= 0) return found;
    const bool fixed = fixed_row >= 0 && fixed_col >= 0;
    const int pose_limit = max_poses >= 0 ? max_poses : (fixed ? 1 : POSE_COUNT);

    vector<grid> tried;
    for (int pose = 0; pose < pose_limit; ++pose) {
        grid posed = posed_cells(cells, pose);
        // A symmetric object reads the same several ways round; count it once.
        bool repeat = false;
        for (const grid& already : tried) {
            if (already.width != posed.width || already.height != posed.height) continue;
            bool same = true;
            for (int row = 0; same && row < posed.height; ++row)
                for (int col = 0; col < posed.width; ++col)
                    if (already.data[row][col] != posed.data[row][col]) { same = false; break; }
            if (same) { repeat = true; break; }
        }
        if (repeat) continue;
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
                // Only visible cells provide evidence for a match.
                auto off_grid = [&](int row, int col) {
                    return at_row + row < 0 || at_row + row >= where.height ||
                           at_col + col < 0 || at_col + col >= where.width;
                };
                int shown = 0;
                for (int row = 0; row < posed.height; ++row)
                    for (int col = 0; col < posed.width; ++col)
                        shown += (posed.data[row][col] != TRANSPARENT && !off_grid(row, col));
                if (!shown) continue;
                int8_t color_map[COLOR_COUNT];
                for (int i = 0; i < COLOR_COUNT; ++i) color_map[i] = -1;
                bool fits = true, exact = true;
                for (int row = 0; fits && row < posed.height; ++row)
                    for (int col = 0; fits && col < posed.width; ++col) {
                        int object_color = posed.data[row][col];
                        if (off_grid(row, col)) continue;
                        int there_color = where.data[at_row + row][at_col + col];
                        if (object_color == TRANSPARENT) {
                            if (match_footprint && there_color != background) {
                                fits = false; break;
                            }
                            continue;
                        }
                        if (there_color == TRANSPARENT) { fits = false; break; }
                        if (match_footprint && there_color == background) { fits = false; break; }
                        if (!is_color(object_color) ||
                            !is_color(there_color)) { fits = false; break; }
                        if (color_map[object_color] == -1)
                            color_map[object_color] = (int8_t)there_color;
                        if (color_map[object_color] != there_color) { fits = false; break; }
                        if (object_color != there_color) exact = false;
                    }
                // The weaker question, asked only where the whole object failed:
                // itself, with the missing cells hidden behind other cells.
                int hidden = 0;
                if (!fits && !match_footprint) {
                    for (int i = 0; i < COLOR_COUNT; ++i) color_map[i] = (int8_t)i;
                    fits = true; exact = true;
                    for (int row = 0; fits && row < posed.height; ++row)
                        for (int col = 0; fits && col < posed.width; ++col) {
                            int object_color = posed.data[row][col];
                            if (object_color == TRANSPARENT || off_grid(row, col)) continue;
                            int there_color = where.data[at_row + row][at_col + col];
                            if (there_color == TRANSPARENT || there_color == background) {
                                fits = false; break;
                            }
                            hidden += (object_color != there_color);
                        }
                }
                if (!fits) continue;
                object_match hit;
                hit.row = at_row; hit.col = at_col; hit.pose = pose;
                hit.exact = exact; hit.hidden = hidden;
                hit.outside = lit - shown;
                for (int i = 0; i < COLOR_COUNT; ++i) hit.color_map[i] = color_map[i];
                found.hits.push_back(hit);
            }
    }
    return found;
}

// Did any cell of the object land on a non-background cell? Background has no
// shape, so every shape fits it, and a match there is no evidence.
inline bool landed_on_non_background(const correspondence_facts::edge& edge,
                                     const grid& cells, const grid& into,
                                     int8_t background) {
    const grid posed = posed_cells(cells, edge.pose);
    for (int row = 0; row < posed.height; ++row)
        for (int col = 0; col < posed.width; ++col) {
            if (posed.data[row][col] == TRANSPARENT) continue;
            const int into_row = edge.row + row, into_col = edge.col + col;
            if (into_row < 0 || into_row >= into.height ||
                into_col < 0 || into_col >= into.width) continue;
            if (into.data[into_row][into_col] != background) return true;
        }
    return false;
}

// Measures how common each color is to help estimate the chance of an accidental match.
inline void color_shares(const grid& one_grid, double share[COLOR_COUNT]) {
    int counts[COLOR_COUNT], total = 0;
    count_colors(one_grid, counts);
    for (int i = 0; i < COLOR_COUNT; ++i) total += counts[i];
    for (int i = 0; i < COLOR_COUNT; ++i) share[i] = total ? (double)counts[i] / total : 0.0;
}

// A tiny blue object can match many places just because blue is common.
// Reject likely coincidences before treating a match as evidence of correspondence.
inline bool beats_chance(int places, double odds) {
    return places > 0 && places * odds < 1.0;
}

// A match that was found and then thrown away, with the reason.
inline void throw_away(const vector<correspondence_facts::edge>& found, size_t found_index,
                       int8_t why, vector<correspondence_facts::edge>* discarded,
                       vector<int8_t>* discard_why) {
    if (!discarded || !discard_why) return;
    discarded->push_back(found[found_index]);
    discard_why->push_back(why);
}

// Copy a search hit into a correspondence, including its color changes.
inline correspondence_facts::edge edge_from_match(const object_match& hit, int from_index,
                                                  int source_color, bool stayed_put) {
    correspondence_facts::edge edge;
    edge.from = (int16_t)from_index;
    edge.row = (int8_t)hit.row; edge.col = (int8_t)hit.col;
    edge.pose = (int8_t)hit.pose;
    edge.exact = hit.exact;
    edge.stayed_put = stayed_put;
    edge.hidden = (int16_t)hit.hidden;
    edge.outside = (int16_t)hit.outside;
    edge.recolored_to = (!hit.exact && source_color >= 0)
                      ? hit.color_map[source_color] : (int8_t)-1;
    if (!hit.exact) copy_n(hit.color_map, COLOR_COUNT, edge.color_map);
    return edge;
}

// Find possible matches for each object in the other grid.
inline vector<correspondence_facts::edge> gather_matches(
        const vector<correspondence_facts::object_info>& from_info,
        const vector<object>& from_objects,
        const vector<correspondence_facts::object_info>& into_info,
        const grid& into, int8_t background, bool may_hang_off,
        vector<correspondence_facts::edge>* discarded,
        vector<int8_t>* discard_why, degrees_of_freedom allowed) {
    double share[COLOR_COUNT];
    color_shares(into, share);

    // Check for repeated shapes in the target grid before allowing clipped matches.
    auto twins_over_there = [&](uint32_t shape) {
        int seen = 0;
        for (const auto& landing : into_info)
            if (landing.shape == shape && ++seen > 1) return true;
        return false;
    };

    vector<correspondence_facts::edge> found;

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
                    if (into_row < 0 || into_row >= into.height ||
                        into_col < 0 || into_col >= into.width) continue;
                    const int into_color = into.data[into_row][into_col];
                    if (hit.hidden && into_color != object_color) continue;
                    odds *= share[hit.color_map[object_color]];
                }
            auto edge = edge_from_match(hit, (int)from_index, one_object.color, stationary);
            const bool stands_alone = beats_chance(found_here.places, odds);
            // Stationary, hidden, or recolored matches must pass the chance test on their own.
            if ((stationary || hit.hidden || !hit.exact) && !stands_alone) {
                if (discarded && discard_why) {
                    discarded->push_back(edge);
                    discard_why->push_back(correspondence_facts::lost_to_chance);
                }
                continue;
            }
            found.push_back(edge);
        }
        };

        // Search both the original position and other positions: an object can stay and be copied.
        const int poses = allowed.pose ? 8 : 1;
        keep_positional(
            find_in(cells, into, background, false,
                    one_object.summary.y0, one_object.summary.x0, false, poses), true);
        // Allow clipped matches only when copies establish the shape on either grid.
        if (allowed.translate)
            keep_positional(
                find_in(cells, into, background,
                        may_hang_off && (one_object.motif >= 0 ||
                                         twins_over_there(one_object.shape)),
                        -1, -1, false, poses), false);

        // Source 2 — shape: objects there with the same canonical hash (count
        // equality guards hash collisions). Skip placements source 1 found.
        if (!allowed.translate) continue;
        for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
            const auto& landing = into_info[into_index];
            if (landing.shape != one_object.shape || landing.summary.count != one_object.summary.count) continue;
            if (landing.summary.y0 == one_object.summary.y0 && landing.summary.x0 == one_object.summary.x0) continue;
            bool seen = false;
            for (const auto& edge : found)
                if (edge.from == (int16_t)from_index &&
                    edge.row == landing.summary.y0 && edge.col == landing.summary.x0) { seen = true; break; }
            if (seen) continue;
            // Estimate chance from the target color count, or all grid colors for multicolor objects.
            double odds = 1.0;
            if (landing.color >= 0)
                for (int i = 0; i < landing.summary.count; ++i) odds *= share[landing.color];
            else
                walk_colored_cells(into, TRANSPARENT,
                                   [&](int, int, int color) { odds *= share[color]; });
            correspondence_facts::edge edge;
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
                if (discarded && discard_why) {
                    discarded->push_back(edge);
                    discard_why->push_back(correspondence_facts::lost_to_chance);
                }
                continue;
            }
            found.push_back(edge);
        }
    }
    // Asked once, of every source, where the whole list is in one place.
    vector<correspondence_facts::edge> on_non_background;
    for (size_t i = 0; i < found.size(); ++i) {
        if (landed_on_non_background(found[i], from_objects[found[i].from].drawn(),
                                     into, background)) {
            on_non_background.push_back(found[i]);
            continue;
        }
        throw_away(found, i, correspondence_facts::lost_to_chance,
                   discarded, discard_why);
    }
    return on_non_background;
}

// Returns the cells used to work out what a match covers in the target grid.
// Uses the matched target object if known; otherwise, the source object turned or flipped as matched.
inline grid edge_footprint(const correspondence_facts::edge& edge,
                           const vector<object>& from_objects,
                           const vector<object>& into_objects) {
    if (edge.to >= 0 && (size_t)edge.to < into_objects.size()) return into_objects[edge.to].drawn();
    return posed_cells(from_objects[edge.from].drawn(), edge.pose);
}

// A cell of a landing that the grid actually shows: on it, and not behind
// another object. Off the edge or covered, the object did not put it there.
inline bool landing_shows(const correspondence_facts::edge& edge, const grid& cells,
                          const grid& into, int row, int col) {
    if (cells.data[row][col] == TRANSPARENT) return false;
    const int into_row = edge.row + row, into_col = edge.col + col;
    if (into_row < 0 || into_row >= into.height ||
        into_col < 0 || into_col >= into.width) return false;
    return !edge.hidden || into.data[into_row][into_col] == cells.data[row][col];
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
};

// Lower scores prefer matches that did not move, change color, turn or flip,
// or extend outside the grid, in that order of importance.
inline int freedom_spent(const correspondence_facts::edge& edge) {
    return (edge.stayed_put ? 0 : 8) + (edge.exact ? 0 : 4) +
           (edge.pose == 0 ? 0 : 2) + (edge.outside == 0 ? 0 : 1);
}

// Prefer more cells; break ties with the lower freedom_spent score.
inline bool says_more(const correspondence_facts::edge& left, size_t left_cells,
                      const correspondence_facts::edge& right, size_t right_cells) {
    if (left_cells != right_cells) return left_cells > right_cells;
    return freedom_spent(left) < freedom_spent(right);
}

// For each target cell, list the matches that can cover it.
inline fitting set_up_the_fitting(const vector<correspondence_facts::edge>& found,
                         const vector<object>& from_objects,
                         const vector<object>& into_objects,
                         const grid& into) {
    fitting laid_out;
    // Index all target cells, including background where new objects may appear.
    // Only cells reached by exact matches become required.
    laid_out.cell_count = (size_t)into.width * into.height;

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
                if (into_row < 0 || into_row >= into.height ||
                    into_col < 0 || into_col >= into.width) continue;
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
            return says_more(found[laid_out.objects[left].match_index],
                             laid_out.objects[left].cells.size(),
                             found[laid_out.objects[right].match_index],
                             laid_out.objects[right].cells.size());
        });

    // Require only cells reached by exact matches; other cells may be newly created.
    laid_out.required.assign(laid_out.cell_count, false);
    for (const auto& candidate : laid_out.objects)
        if (found[candidate.match_index].exact)
            for (int id : candidate.cells) laid_out.required[id] = true;
    return laid_out;
}

// How a pass decided, so a giveup can never be mistaken for an answer.
enum class fit_verdict : int8_t { fitted, no_fit, gave_up };

// Find objects that cover each required cell exactly once; store them in `chosen`.
inline fit_verdict fit_the_objects_together(const fitting& laid_out,
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

    auto fits_together = [&](auto&& self) -> bool {
        if (work > work_budget) { out_of_budget = true; return false; }
        // The cell with the fewest objects still able to paint it decides next;
        // zero options means this line of choices is dead. Only required cells
        // drive the search — a made cell is answered by being left alone.
        int target = -1, fewest = 0;
        for (size_t id = 0; id < laid_out.cell_count; ++id) {
            if (painted_by[id] >= 0 || !laid_out.required[id]) continue;
            int options = 0;
            for (int object_index : laid_out.options[id])
                options += cells_are_free(object_index);
            if (options == 0) return false;
            if (target < 0 || options < fewest) { target = (int)id; fewest = options; }
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
inline void keep_what_fitted(vector<int>& chosen, const fitting& laid_out,
                           const vector<correspondence_facts::edge>& found,
                           vector<correspondence_facts::edge>& edges,
                           vector<correspondence_facts::edge>* discarded,
                           vector<int8_t>* discard_why) {
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
        return says_more(found[laid_out.objects[left].match_index],
                         laid_out.objects[left].cells.size(),
                         found[laid_out.objects[right].match_index],
                         laid_out.objects[right].cells.size());
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
            throw_away(found, found_index, correspondence_facts::overlaps_a_kept_object,
                       discarded, discard_why);
}


// Try better matches first; keep each only if it covers at least one new cell.
inline void fall_back_to_greedy(const vector<correspondence_facts::edge>& found,
                                const vector<object>& from_objects,
                                const vector<object>& into_objects,
                                const grid& into,
                                vector<correspondence_facts::edge>& edges,
                                vector<correspondence_facts::edge>* discarded,
                                vector<int8_t>* discard_why) {
    auto footprint_of = [&](const correspondence_facts::edge& edge) {
        return edge_footprint(edge, from_objects, into_objects);
    };
    // Use the same match preference as the fitting search.
    vector<size_t> strongest_first(found.size());
    iota(strongest_first.begin(), strongest_first.end(), (size_t)0);
    stable_sort(strongest_first.begin(), strongest_first.end(),
                [&](size_t left, size_t right) {
        return says_more(found[left], (size_t)lit_cells(footprint_of(found[left])),
                         found[right], (size_t)lit_cells(footprint_of(found[right])));
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
            throw_away(found, found_index, correspondence_facts::cells_already_explained,
                       discarded, discard_why);
            continue;
        }
        for (int row = 0; row < cells.height; ++row)
            for (int col = 0; col < cells.width; ++col)
                if (landing_shows(found[found_index], cells, into, row, col))
                    taken[at_row + row][at_col + col] = true;
        edges.push_back(found[found_index]);
    }
}

// Movement from input to output in the original grid coordinates.
// Reverse an output-to-input correspondence so both directions agree.
inline pair<int, int> move_of(const correspondence_facts::edge& edge,
                              const correspondence_facts::object_info& one_object) {
    const int down = edge.row - one_object.summary.y0;
    const int across = edge.col - one_object.summary.x0;
    return edge.from_input ? pair<int, int>{down, across}
                           : pair<int, int>{-down, -across};
}

// Store each color's move for this training pair, only if every read pair has
// exactly one move for that color. The move may differ between pairs.
struct what_the_pairs_agreed {
    bool color_speaks[COLOR_COUNT] = {};
    int8_t down[COLOR_COUNT] = {}, across[COLOR_COUNT] = {};
};

// Reject shape matches whose movement disagrees with the color's agreed move.
// Stationary matches are always allowed.
inline bool the_pairs_disagree(const what_the_pairs_agreed& agreed,
                               const correspondence_facts::edge& edge,
                               const correspondence_facts::object_info& one_object) {
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
inline fit_verdict collect_edges(const vector<correspondence_facts::object_info>& from_info,
                          const vector<object>& from_objects,
                          const vector<correspondence_facts::object_info>& into_info,
                          const vector<object>& into_objects,
                          const grid& into, bool from_input,
                          int8_t background, bool may_hang_off,
                          vector<correspondence_facts::edge>& edges,
                          vector<correspondence_facts::edge>* discarded = nullptr,
                          vector<int8_t>* discard_why = nullptr,
                          degrees_of_freedom allowed = {},
                          const what_the_pairs_agreed* agreed = nullptr) {
    vector<correspondence_facts::edge> found =
        gather_matches(from_info, from_objects, into_info, into,
                       background, may_hang_off, discarded, discard_why, allowed);
    for (auto& edge : found) edge.from_input = from_input;

    // Remove conflicting matches before fitting so their alternatives can still be chosen.
    if (agreed)
        for (size_t i = found.size(); i-- > 0; ) {
            const auto& one_object = from_info[found[i].from];
            if (!the_pairs_disagree(*agreed, found[i], one_object)) continue;
            throw_away(found, i, correspondence_facts::the_other_examples_disagreed,
                       discarded, discard_why);
            found.erase(found.begin() + i);
        }

    const fitting laid_out = set_up_the_fitting(found, from_objects, into_objects, into);

    vector<int> chosen;
    const fit_verdict searched =
        fit_the_objects_together(laid_out, chosen);
    if (searched == fit_verdict::fitted) {
        keep_what_fitted(chosen, laid_out, found, edges, discarded, discard_why);
        return fit_verdict::fitted;
    }

    fall_back_to_greedy(found, from_objects, into_objects, into,
                        edges, discarded, discard_why);
    return searched;
}

// Match each unmatched object to the same-color object with the most overlap,
// requiring that the other object has at least as many cells.
inline void sit_on_top(const vector<correspondence_facts::object_info>& from_info,
                       const vector<object>& from_objects,
                       const vector<correspondence_facts::object_info>& into_info,
                       const vector<object>& into_objects,
                       vector<correspondence_facts::edge>& edges) {
    for (size_t from_index = 0; from_index < from_info.size(); ++from_index) {
        // Only objects nothing else accounted for. Asked of the edges themselves,
        // so this does not depend on a tally someone remembered to keep.
        if (ranges::any_of(edges, [&](const correspondence_facts::edge& edge) {
                return edge.from == (int16_t)from_index; }))
            continue;
        const auto& one_object = from_info[from_index];
        const object& cells = from_objects[from_index];
        int best = -1, most_shared = 0;
        for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
            const auto& landing = into_info[into_index];
            if (landing.color != one_object.color || landing.summary.count < one_object.summary.count) continue;
            const object& landing_object = into_objects[into_index];
            int shared = 0;
            for (int row = 0; row < cells.summary.h; ++row)
                for (int col = 0; col < cells.summary.w; ++col) {
                    if (cells.cells[row][col] == TRANSPARENT) continue;
                    int in_landing_row = one_object.summary.y0 + row - landing.summary.y0;
                    int in_landing_col = one_object.summary.x0 + col - landing.summary.x0;
                    if (in_landing_row < 0 || in_landing_row >= landing_object.summary.h ||
                        in_landing_col < 0 || in_landing_col >= landing_object.summary.w) continue;
                    shared += (landing_object.cells[in_landing_row][in_landing_col] != TRANSPARENT);
                }
            if (shared > most_shared) { most_shared = shared; best = (int)into_index; }
        }
        if (best < 0) continue;
        correspondence_facts::edge edge;
        edge.from = (int16_t)from_index;
        edge.to = (int16_t)best;
        edge.row = into_info[best].summary.y0;
        edge.col = into_info[best].summary.x0;
        edge.exact = false;
        edge.same_shape = false;
        edges.push_back(edge);
    }
}

// Count the matches starting from each source object.
inline void count_edges_leaving(const vector<correspondence_facts::edge>& edges,
                                vector<correspondence_facts::object_info>& from_info) {
    for (auto& one_object : from_info) one_object.edges_touching = 0;
    for (const auto& edge : edges)
        if (edge.from >= 0 && (size_t)edge.from < from_info.size()) {
            from_info[edge.from].edges_touching++;
            from_info[edge.from].has_partner = true;
        }
}

// Count the matches that overlap each target object.
inline void count_edges_arriving(const vector<correspondence_facts::edge>& edges,
                            const vector<object>& from_objects,
                            vector<correspondence_facts::object_info>& into_info,
                            const vector<object>& into_objects) {
    for (const auto& edge : edges) {
        const grid footprint = edge_footprint(edge, from_objects, into_objects);
        for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
            const auto& landing = into_info[into_index];
            const object& landing_object = into_objects[into_index];
            // Does the landing touch this object at all?
            bool touches = false;
            for (int row = 0; row < landing_object.summary.h && !touches; ++row)
                for (int col = 0; col < landing_object.summary.w; ++col) {
                    if (landing_object.cells[row][col] == TRANSPARENT) continue;
                    int in_footprint_row = landing.summary.y0 + row - edge.row;
                    int in_footprint_col = landing.summary.x0 + col - edge.col;
                    if (in_footprint_row < 0 || in_footprint_row >= footprint.height ||
                        in_footprint_col < 0 || in_footprint_col >= footprint.width) continue;
                    if (footprint.data[in_footprint_row][in_footprint_col] != TRANSPARENT) {
                        touches = true; break;
                    }
                }
            if (touches) {
                into_info[into_index].edges_touching++;
                // A shape match or an explicit target link identifies this object as a partner.
                if (edge.same_shape || (int16_t)into_index == edge.to)
                    into_info[into_index].has_partner = true;
            }
        }
    }
}

// Return the smaller grid size when all pairs share the same block size and
// whole-number size ratios. Return {0, 0} if they do not.
inline pair<int, int> repeating_step_of(const vector<example>& train) {
    if (train.empty()) return { 0, 0 };
    // Keep the first pair's growth direction and smaller grid size across all pairs.
    const bool output_grows =
        train[0].output.width >= train[0].input.width &&
        train[0].output.height >= train[0].input.height;
    const grid& first_block = output_grows ? train[0].input : train[0].output;
    if (first_block.width <= 0 || first_block.height <= 0) return { 0, 0 };
    for (const auto& ex : train) {
        const grid& block = output_grows ? ex.input : ex.output;
        const grid& tiled = output_grows ? ex.output : ex.input;
        if (block.width != first_block.width || block.height != first_block.height)
            return { 0, 0 };
        if (tiled.width % block.width || tiled.height % block.height)
            return { 0, 0 };
        if (tiled.width == block.width && tiled.height == block.height)
            return { 0, 0 };
    }
    return { first_block.width, first_block.height };
}

// Divides a grid into objects, excluding the background if known.
inline void segment_into_objects(const grid& one_grid, int8_t background, segmentation_methods segmentation,
                                vector<correspondence_facts::object_info>& info,
                                vector<object>* objects,
                                int step_width = 0, int step_height = 0) {
    // Allow up to one object per grid cell.
    static constexpr int MAX_OBJECTS = MAX_GRID_DIM * MAX_GRID_DIM;

    // Cut on a fixed step: equal blocks, one object each, cells of any color kept
    // together. Blocks with no colored cells at all are not objects.
    if (segmentation == segmentation_methods::smallest_of_two_grids ||
        segmentation == segmentation_methods::minimal_repeating_set) {
        if (step_width <= 0 || step_height <= 0) return;
        bool occupied[MAX_GRID_DIM * MAX_GRID_DIM];
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
                    int color = one_grid.data[block_top + row][block_left + col];
                    bool is_colored = (color >= 0 && color != background);
                    occupied[row * block_width + col] = is_colored;
                    if (is_colored) piece.cells[row][col] = (int8_t)color;
                    lit += is_colored;
                }
            if (!lit) continue;
            correspondence_facts::object_info one_object;
            one_object.color = -1;
            one_object.summary.x0 = (int8_t)block_left; one_object.summary.y0 = (int8_t)block_top;
            one_object.summary.w = (int8_t)block_width; one_object.summary.h = (int8_t)block_height;
            one_object.summary.count = (int16_t)lit;
            one_object.shape = canonical_shape_hash(block_width, block_height, occupied);
            info.push_back(one_object);
            if (objects) { piece.summary = one_object.summary; objects->push_back(piece); }
        }
        // Repeating-block segmentation needs at least two nonempty blocks.
        if (segmentation != segmentation_methods::smallest_of_two_grids && info.size() < 2) {
            info.clear();
            if (objects) objects->clear();
        }
        return;
    }
    // Whether an object has to be one color is a question of its OWN, asked apart
    // from what counts as joined: the one walk in 1_readings.hpp answers both.
    const grouping how =
        segmentation == segmentation_methods::connected_multicolor   ? grouping::touching_any_color :
        segmentation == segmentation_methods::all_cells_of_one_color ? grouping::all_of_one_color :
                                                                       grouping::touching_same_color;
    bool occupied[MAX_GRID_DIM * MAX_GRID_DIM];
    object_walk_scratch scratch;
    each_object(one_grid, background, how, scratch,
                [&](const object_summary& found, const int16_t* found_cells, int how_many) {
        if ((int)info.size() >= MAX_OBJECTS) return;
        auto row_of = [](int16_t at) { return at / MAX_GRID_DIM; };
        auto col_of = [](int16_t at) { return at % MAX_GRID_DIM; };
        const int color = one_grid.data[row_of(found_cells[0])][col_of(found_cells[0])];
        // An object wears one color only if every cell of it does; -1 says several.
        bool one_color = true;
        for (int i = 1; i < how_many && one_color; ++i)
            one_color = (one_grid.data[row_of(found_cells[i])][col_of(found_cells[i])] == color);
        correspondence_facts::object_info one_object;
        one_object.color = one_color ? (int8_t)color : (int8_t)-1;
        one_object.summary = found;
        for (int i = 0; i < found.w * found.h; ++i) occupied[i] = false;
        for (int i = 0; i < how_many; ++i)
            occupied[(row_of(found_cells[i]) - found.y0) * found.w +
                     (col_of(found_cells[i]) - found.x0)] = true;
        one_object.shape = canonical_shape_hash(found.w, found.h, occupied);
        info.push_back(one_object);
        if (objects) {
            // Each cell keeps its own color, so an object of several reads as it
            // looks. For a one-color object that is the color it already had.
            object piece;
            piece.summary = found;
            piece.cells.clear();  // every cell TRANSPARENT until the walk claims it
            for (int i = 0; i < how_many; ++i)
                piece.cells[row_of(found_cells[i]) - found.y0][col_of(found_cells[i]) - found.x0] =
                    (int8_t)one_grid.data[row_of(found_cells[i])][col_of(found_cells[i])];
            objects->push_back(piece);
        }
    });
}

// Divides a grid into objects, then sorts them by size (largest first) and position.
inline void segment_for_correspondence(const grid& one_grid, int8_t background, segmentation_methods segmentation,
                                vector<correspondence_facts::object_info>& info,
                                vector<object>* objects,
                                int step_width = 0, int step_height = 0) {
    segment_into_objects(one_grid, background, segmentation, info, objects, step_width, step_height);
    vector<int> reading_order(info.size());
    iota(reading_order.begin(), reading_order.end(), 0);
    stable_sort(reading_order.begin(), reading_order.end(), [&](int left, int right) {
        if (info[left].summary.count != info[right].summary.count)
            return info[left].summary.count > info[right].summary.count;
        if (info[left].summary.y0 != info[right].summary.y0)
            return info[left].summary.y0 < info[right].summary.y0;
        return info[left].summary.x0 < info[right].summary.x0;
    });
    vector<correspondence_facts::object_info> sorted;
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
inline double describe_cost(const grid& one_grid, int8_t background,
                            const vector<correspondence_facts::object_info>& objects) {
    const double cell_bits = log2((double)COLOR_COUNT);
    const double place_bits =
        log2((double)max(1, one_grid.width * one_grid.height));
    int colored = 0;
    walk_colored_cells(one_grid, background, [&](int, int, int) { ++colored; });
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

// Return the repeating block size measured from this grid.
inline pair<int, int> block_measured_on(const grid& one_grid, int8_t background) {
    int width, height;
    repeat_periods(one_grid, background, width, height);
    return { width, height };
}

// Choose a shared block size using whichever grid compresses better.
// Uses both training grids, so this must not supply a block to an input-only operation.
inline pair<int, int> block_of(const grid& input, const grid& output, int8_t background,
                               segmentation_methods segmentation, int sizes_width, int sizes_height) {
    if (segmentation != segmentation_methods::minimal_repeating_set)
        return { sizes_width, sizes_height };
    const auto from_input =
        block_measured_on(input, background);
    const auto from_output =
        block_measured_on(output, background);
    // Cut each grid on its own candidate and ask what that costs to say; the
    // cheaper block is the one that found real repetition rather than noise.
    auto cost_of = [&](const grid& one_grid, pair<int, int> block) {
        vector<correspondence_facts::object_info> objects;
        segment_for_correspondence(one_grid, background, segmentation, objects, nullptr,
                                   block.first, block.second);
        return describe_cost(one_grid, background, objects);
    };
    return cost_of(output, from_output) < cost_of(input, from_input) ? from_output
                                                                    : from_input;
}

// Counts differing cells when two objects are aligned at their top-left corners.
inline int count_differing_cells(const object& left, const object& right) {
    const int width = max(left.summary.w, right.summary.w);
    const int height = max(left.summary.h, right.summary.h);
    int apart = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int here = (row < left.summary.h && col < left.summary.w)
                           ? left.cells[row][col] : TRANSPARENT;
            const int there = (row < right.summary.h && col < right.summary.w)
                            ? right.cells[row][col] : TRANSPARENT;
            apart += (here != there);
        }
    return apart;
}

// Groups copies of an object and chooses the most typical copy in each group (the medoid).
inline void find_copies(vector<correspondence_facts::object_info>& info,
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

inline void mark_painted(const vector<correspondence_facts::edge>& edges,
                         const vector<object>& from_objects,
                         const vector<object>& into_objects,
                         const grid& into,
                         bool covered[MAX_GRID_DIM][MAX_GRID_DIM],
                         bool exact_only = true,
                         bool same_shape_only = false) {
    for (const auto& edge : edges) {
        if (exact_only && !edge.exact) continue;
        if (same_shape_only && !edge.same_shape) continue;
        const grid footprint = edge_footprint(edge, from_objects, into_objects);
        for (int row = 0; row < footprint.height; ++row)
            for (int col = 0; col < footprint.width; ++col)
                if (landing_shows(edge, footprint, into, row, col))
                    covered[edge.row + row][edge.col + col] = true;
    }
}

// Match repeated blocks in both directions so every matching block gets a partner.
// Prefer a source block at the same position when possible.
inline fit_verdict collect_motif_edges(
        const vector<correspondence_facts::object_info>& from_info,
        const vector<object>& from_objects,
        const vector<correspondence_facts::object_info>& into_info,
        const grid& into, int8_t background, int step_w, int step_h,
        vector<correspondence_facts::edge>& edges, degrees_of_freedom allowed = {}) {
    if (step_w <= 0 || step_h <= 0) return fit_verdict::no_fit;
    const int poses = allowed.pose ? 8 : 1;

    // Does this from block sit at that place on the other grid? The answer is
    // the same question both sweeps ask, so it is written once.
    auto sits_at = [&](int from_index, const correspondence_facts::object_info& target,
                       object_match& found) {
        const object_matches matches =
            find_in(from_objects[from_index].drawn(), into, background, true,
                    target.summary.y0, target.summary.x0, true, poses);
        // Prefer a match with unchanged colors, even if it uses a later rotation or reflection.
        const object_match* best = nullptr;
        for (const object_match& hit : matches.hits) {
            if (!allowed.recolor && !hit.exact) continue;
            if (!best || (hit.exact && !best->exact)) best = &hit;
            if (best->exact) break;
        }
        if (!best) return false;
        found = *best;
        return true;
    };
    auto draw = [&](int from_index, int target_index, const object_match& hit) {
        const auto& target = into_info[target_index];
        auto edge = edge_from_match(hit, from_index, from_info[from_index].color,
            from_info[from_index].summary.y0 == target.summary.y0 &&
            from_info[from_index].summary.x0 == target.summary.x0);
        edge.to = (int16_t)target_index;
        edges.push_back(edge);
    };

    vector<bool> answered(from_info.size(), false);
    // Prefer a matching source at the target's position, then the first match elsewhere.
    for (int target_index = 0; target_index < (int)into_info.size(); ++target_index) {
        const auto& target = into_info[target_index];
        int source = -1;
        object_match hit;
        for (int i = 0; i < (int)from_info.size(); ++i)
            if (from_info[i].summary.y0 == target.summary.y0 && from_info[i].summary.x0 == target.summary.x0 &&
                sits_at(i, target, hit)) { source = i; break; }
        for (int i = 0; source < 0 && i < (int)from_info.size(); ++i)
            if (sits_at(i, target, hit)) source = i;
        if (source < 0) continue;
        draw(source, target_index, hit);
        answered[source] = true;
    }
    // WHERE DID EACH OBJECT GO. Only the ones no landing claimed, so a from side
    // bigger than the into side is not silently dropped.
    for (int i = 0; i < (int)from_info.size(); ++i) {
        if (answered[i]) continue;
        object_match hit;
        for (int target_index = 0; target_index < (int)into_info.size(); ++target_index)
            if (sits_at(i, into_info[target_index], hit)) {
                draw(i, target_index, hit);
                answered[i] = true;
                break;
            }
    }
    return edges.empty() ? fit_verdict::no_fit : fit_verdict::fitted;
}

// An empty region that an object may have filled or left behind.
struct blank_patch {
    int8_t left = 0, top = 0, width = 0, height = 0;
    int16_t cells = 0;
    bool enclosed = false;
};

// Finds empty regions to help describe where objects appeared or disappeared.
inline void find_blanks(const grid& one_grid, int8_t background,
                        vector<blank_patch>& patches,
                        int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM]) {
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) patch_of[row][col] = -1;
    vector<int> to_visit;
    for (int row = 0; row < one_grid.height; ++row)
        for (int col = 0; col < one_grid.width; ++col) {
            if (one_grid.data[row][col] != background) continue;
            if (patch_of[row][col] >= 0) continue;
            const int16_t mine = (int16_t)patches.size();
            int left = col, right = col, top = row, bottom = row, cells = 0;
            bool open = false;
            to_visit.assign(1, row * MAX_GRID_DIM + col);
            patch_of[row][col] = mine;
            while (!to_visit.empty()) {
                const int at = to_visit.back();
                to_visit.pop_back();
                const int at_row = at / MAX_GRID_DIM, at_col = at % MAX_GRID_DIM;
                ++cells;
                left = min(left, at_col);   right = max(right, at_col);
                top = min(top, at_row);     bottom = max(bottom, at_row);
                open = open || at_row == 0 || at_col == 0 ||
                       at_row == one_grid.height - 1 || at_col == one_grid.width - 1;
                for (int side = 0; side < 4; ++side) {
                    const int next_row = at_row + neighbour_row_4(side);
                    const int next_col = at_col + neighbour_col_4(side);
                    if (next_row < 0 || next_row >= one_grid.height ||
                        next_col < 0 || next_col >= one_grid.width) continue;
                    if (one_grid.data[next_row][next_col] != background) continue;
                    if (patch_of[next_row][next_col] >= 0) continue;
                    patch_of[next_row][next_col] = mine;
                    to_visit.push_back(next_row * MAX_GRID_DIM + next_col);
                }
            }
            blank_patch patch;
            patch.left = (int8_t)left;  patch.top = (int8_t)top;
            patch.width = (int8_t)(right - left + 1);
            patch.height = (int8_t)(bottom - top + 1);
            patch.cells = (int16_t)cells;
            patch.enclosed = !open;
            patches.push_back(patch);
        }
}

// Objects and grid-only scores shared by every permission combination.
struct prepared_segmentation {
    correspondence_facts::reading look;
    vector<correspondence_facts::pair_info> pairs;
    vector<vector<object>> input_objects, output_objects;
    vector<int> step_width, step_height;
    vector<bool> read;
};

inline prepared_segmentation prepare_segmentation(const vector<example>& train,
                                                  int8_t background, segmentation_methods segmentation) {
    prepared_segmentation prepared;
    prepared.look.segmentation = segmentation;
    prepared.pairs.resize(train.size());
    prepared.input_objects.resize(train.size());
    prepared.output_objects.resize(train.size());
    prepared.step_width.resize(train.size());
    prepared.step_height.resize(train.size());
    for (const auto& pair : train)
        prepared.read.push_back(pair.input.width > 0 && pair.input.height > 0 &&
                                pair.output.width > 0 && pair.output.height > 0);
    const auto [sizes_step_width, sizes_step_height] = repeating_step_of(train);
    double shares = 0.0;
    int counted = 0;
    // Segment before matching so another matching pass can reuse the objects.
    for (size_t pair_index = 0; pair_index < train.size(); ++pair_index) {
        if (!prepared.read[pair_index]) continue;
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = prepared.pairs[pair_index];
        // Use the same block size for both grids.
        const auto [step_width, step_height] = block_of(input, output, background,
                                                        segmentation, sizes_step_width,
                                                        sizes_step_height);
        vector<object>& input_objects = prepared.input_objects[pair_index];
        vector<object>& output_objects = prepared.output_objects[pair_index];
        segment_for_correspondence(input, background, segmentation, pair.input_info,
                                   &input_objects, step_width, step_height);
        segment_for_correspondence(output, background, segmentation, pair.output_info,
                                   &output_objects, step_width, step_height);
        // Group copies and choose the most typical member while their cells are available.
        const bool one_set_for_the_grid = (segmentation == segmentation_methods::minimal_repeating_set);
        find_copies(pair.input_info, input_objects, one_set_for_the_grid);
        find_copies(pair.output_info, output_objects, one_set_for_the_grid);
        // What the segmentation cost to say, on each grid. The pair is described by both.
        pair.describe_cost =
            0.5 * (describe_cost(input, background, pair.input_info) +
                   describe_cost(output, background, pair.output_info));
        shares += pair.describe_cost;
        ++counted;
        auto any_fits = [](const auto& objects, const grid& other) {
            return ranges::any_of(objects, [&](const auto& one) {
                return one.summary.w <= other.width && one.summary.h <= other.height;
            });
        };
        // Preliminary ordering ignores rotations and clipping; all readings are still searched.
        if (!any_fits(pair.input_info, output) && !any_fits(pair.output_info, input))
            prepared.look.objects_can_fit = false;
        prepared.step_width[pair_index] = step_width;
        prepared.step_height[pair_index] = step_height;
    }

    if (counted) prepared.look.describe_cost = shares / counted;
    return prepared;
}

// Match prepared objects using one permission combination.
inline correspondence_facts detect_correspondence(const vector<example>& train, int8_t background,
                                                  const prepared_segmentation& prepared,
                                                  degrees_of_freedom allowed) {
    correspondence_facts facts;
    if (train.empty()) return facts;
    facts.pairs = prepared.pairs;
    facts.segmentation = prepared.look.segmentation;
    auto pair_was_read = [&](size_t which) { return prepared.read[which]; };
    const bool pattern_reading =
        facts.segmentation == segmentation_methods::minimal_repeating_set ||
        facts.segmentation == segmentation_methods::smallest_of_two_grids;

    // Match each pair, then rematch if its results conflict with agreement across pairs.
    auto match_one_pair = [&](size_t pair_index, const what_the_pairs_agreed* agreed) {
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        const vector<object>& input_objects = prepared.input_objects[pair_index];
        const vector<object>& output_objects = prepared.output_objects[pair_index];
        const int step_width = prepared.step_width[pair_index];
        const int step_height = prepared.step_height[pair_index];
        pair.edges.clear(); pair.discarded.clear(); pair.discard_why.clear();
        pair.agreed_moves.clear(); pair.verdict = 0;
        // THE SAME PASS, TWICE, WITH THE SIDES SWAPPED. Nothing below cares
        // which grid it was handed; the side is stamped on the edges afterwards.
        auto read_one_way = [&](bool from_input) {
            const auto& from_info = from_input ? pair.input_info : pair.output_info;
            const auto& into_info = from_input ? pair.output_info : pair.input_info;
            const auto& from_objects = from_input ? input_objects : output_objects;
            const auto& into_objects = from_input ? output_objects : input_objects;
            const grid& into = from_input ? output : input;
            vector<correspondence_facts::edge> edges, discarded;
            vector<int8_t> discard_why;
            fit_verdict verdict;
            if (pattern_reading) {
                verdict = collect_motif_edges(from_info, from_objects, into_info,
                                              into, background, step_width, step_height,
                                              edges, allowed);
            } else {
                verdict = collect_edges(from_info, from_objects, into_info,
                                        into_objects, into, from_input,
                                        background, allowed.hang_off,
                                        edges, &discarded, &discard_why, allowed, agreed);
                sit_on_top(from_info, from_objects, into_info, into_objects, edges);
            }
            for (auto& edge : edges) edge.from_input = from_input;
            for (auto& edge : discarded) edge.from_input = from_input;
            // The weaker of the two readings names the pass: a fit both ways is
            // a fit, and a giveup anywhere is still a giveup.
            pair.verdict = max(pair.verdict, (int8_t)verdict);
            pair.edges.insert(pair.edges.end(), edges.begin(), edges.end());
            pair.discarded.insert(pair.discarded.end(), discarded.begin(), discarded.end());
            pair.discard_why.insert(pair.discard_why.end(),
                                    discard_why.begin(), discard_why.end());
        };
        read_one_way(true);
        read_one_way(false);

        // Merge matches linking the same input and output objects.
        // Prefer exact over recolored over overlap-only; ties prefer the forward match.
        // Matches without a target object stay separate.
        struct two_ends { int16_t on_input, on_output; };
        auto both_ends = [](const correspondence_facts::edge& edge) {
            return edge.from_input ? two_ends{ edge.from, edge.to }
                                   : two_ends{ edge.to, edge.from };
        };
        auto joins_the_same = [&](const correspondence_facts::edge& left,
                                  const correspondence_facts::edge& right) {
            const two_ends mine = both_ends(left), theirs = both_ends(right);
            return mine.on_input >= 0 && mine.on_output >= 0 &&
                   mine.on_input == theirs.on_input &&
                   mine.on_output == theirs.on_output;
        };
        auto claim_strength = [](const correspondence_facts::edge& edge) {
            return !edge.same_shape ? 0 : edge.exact ? 2 : 1;
        };
        vector<correspondence_facts::edge> kept;
        for (const auto& edge : pair.edges) {
            auto same = ranges::find_if(kept, [&](const correspondence_facts::edge& seen) {
                return joins_the_same(edge, seen); });
            if (same == kept.end()) { kept.push_back(edge); continue; }
            const int mine = claim_strength(edge), theirs = claim_strength(*same);
            if (mine > theirs || (mine == theirs && edge.from_input && !same->from_input))
                *same = edge;
        }
        pair.edges.swap(kept);

        // Sort by source grid, source position, then target position for display.
        stable_sort(pair.edges.begin(), pair.edges.end(),
                    [&](const correspondence_facts::edge& left,
                        const correspondence_facts::edge& right) {
            if (left.from_input != right.from_input) return left.from_input;
            const auto& objects = left.from_input ? pair.input_info
                                                  : pair.output_info;
            const auto& here = objects[left.from];
            const auto& there = objects[right.from];
            if (here.summary.y0 != there.summary.y0)   return here.summary.y0 < there.summary.y0;
            if (here.summary.x0 != there.summary.x0) return here.summary.x0 < there.summary.x0;
            if (left.row != right.row)   return left.row < right.row;
            return left.col < right.col;
        });

        // Group equal input-to-output movements within this pair.
        // Count each input object once per movement, using its input color.
        // A backward positional correspondence needs an identifiable input object.
        auto input_object_of = [&](const correspondence_facts::edge& edge) {
            if (edge.from_input) return (int)edge.from;
            if (edge.to >= 0) return (int)edge.to;
            if (edge.hidden || edge.outside) return -1;
            const auto& source = pair.output_info[edge.from];
            const auto [width, height] = posed_size(edge.pose, source.summary.w, source.summary.h);
            int found = -1;
            for (size_t i = 0; i < pair.input_info.size(); ++i) {
                const auto& target = pair.input_info[i];
                if (target.summary.y0 != edge.row || target.summary.x0 != edge.col ||
                    target.summary.w != width || target.summary.h != height ||
                    target.summary.count != source.summary.count || target.shape != source.shape)
                    continue;
                if (found >= 0) return -1;
                found = (int)i;
            }
            return found;
        };
        vector<correspondence_facts::pair_info::moved_together> moves;
        vector<vector<int>> who_moved;
        for (const auto& edge : pair.edges) {
            if (!edge.same_shape) continue;
            const auto& objects = edge.from_input ? pair.input_info : pair.output_info;
            if (edge.from < 0 || (size_t)edge.from >= objects.size()) continue;
            const auto& one_object = objects[edge.from];
            const auto went = move_of(edge, one_object);
            const int8_t down = (int8_t)went.first, across = (int8_t)went.second;
            const int speaker = input_object_of(edge);
            if (speaker < 0 || (size_t)speaker >= pair.input_info.size()) continue;
            const int8_t input_color = pair.input_info[speaker].color;
            size_t at = 0;
            while (at < moves.size() &&
                   (moves[at].down != down || moves[at].across != across)) ++at;
            if (at == moves.size()) {
                moves.push_back({ down, across, input_color, 0 });
                who_moved.push_back({});
            }
            if (ranges::find(who_moved[at], speaker) == who_moved[at].end()) {
                who_moved[at].push_back(speaker);
                moves[at].objects++;
                if (moves[at].color != input_color) moves[at].color = -1;
            }
        }
        for (const auto& move : moves)
            if (move.objects >= 2) pair.agreed_moves.push_back(move);
        ranges::sort(pair.agreed_moves, [](const auto& left, const auto& right) {
            return left.objects > right.objects; });
    };
    for (size_t pair_index = 0; pair_index < train.size(); ++pair_index)
        if (pair_was_read(pair_index)) match_one_pair(pair_index, nullptr);

    // Use a color to constrain matches only when every read pair has one shared,
    // nonzero movement for it. The movement may differ between pairs.
    // Return null when this pair has no such movement or has several.
    auto lone_move_of = [&](size_t which, int color) {
        const correspondence_facts::pair_info::moved_together* only = nullptr;
        for (const auto& said : facts.pairs[which].agreed_moves) {
            if (said.color != color || (!said.down && !said.across)) continue;
            if (only) return (const correspondence_facts::pair_info::moved_together*)nullptr;
            only = &said;
        }
        return only;
    };
    vector<what_the_pairs_agreed> agreed(train.size());
    bool anything_agreed = false;
    for (int color = 0; color < COLOR_COUNT; ++color) {
        int read = 0, said_once = 0;
        for (size_t which = 0; which < facts.pairs.size(); ++which) {
            if (!pair_was_read(which)) continue;
            ++read;
            said_once += (lone_move_of(which, color) != nullptr);
        }
        if (read == 0 || said_once != read) continue;
        anything_agreed = true;
        for (size_t which = 0; which < facts.pairs.size(); ++which) {
            if (!pair_was_read(which)) continue;
            const auto* rule = lone_move_of(which, color);
            agreed[which].color_speaks[color] = true;
            agreed[which].down[color] = rule->down;
            agreed[which].across[color] = rule->across;
        }
    }
    if (anything_agreed)
        for (size_t which = 0; which < facts.pairs.size(); ++which) {
            if (!pair_was_read(which)) continue;
            // Rematch only if an existing match disagrees.
            const auto& pair = facts.pairs[which];
            bool any_disagrees = false;
            for (const auto& edge : pair.edges) {
                const auto& objects = edge.from_input ? pair.input_info
                                                      : pair.output_info;
                if (edge.from < 0 || (size_t)edge.from >= objects.size()) continue;
                if (!the_pairs_disagree(agreed[which], edge, objects[edge.from])) continue;
                any_disagrees = true;
                break;
            }
            if (any_disagrees) match_one_pair(which, &agreed[which]);
        }

    // SECOND PASS — every number, read off the edges the pairs let through.
    for (size_t pair_index = 0; pair_index < train.size(); ++pair_index) {
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        if (!pair_was_read(pair_index)) continue;
        const vector<object>& input_objects = prepared.input_objects[pair_index];
        const vector<object>& output_objects = prepared.output_objects[pair_index];

        // The two halves of the merged list, each in its own frame, so every
        // helper below stays the one-direction function it always was.
        vector<correspondence_facts::edge> forward, backward;
        for (const auto& edge : pair.edges)
            (edge.from_input ? forward : backward).push_back(edge);

        // Count matches at both ends so arrivals and departures both identify partners.
        count_edges_leaving(forward, pair.input_info);
        count_edges_leaving(backward, pair.output_info);
        count_edges_arriving(forward, input_objects, pair.output_info, output_objects);
        count_edges_arriving(backward, output_objects, pair.input_info, input_objects);
        pair.objects_used = (int)pair.edges.size();

        // Keep rejection records only for objects that still have no partner.
        size_t still_refused = 0;
        for (size_t gone = 0; gone < pair.discarded.size(); ++gone) {
            const auto& thrown = pair.discarded[gone];
            const auto& objects = thrown.from_input ? pair.input_info
                                                    : pair.output_info;
            if (thrown.from >= 0 && (size_t)thrown.from < objects.size() &&
                objects[thrown.from].has_partner) continue;
            pair.discarded[still_refused] = thrown;
            pair.discard_why[still_refused] = pair.discard_why[gone];
            ++still_refused;
        }
        pair.discarded.resize(still_refused);
        pair.discard_why.resize(still_refused);

        // Record the background region touched by each unmatched object on the other grid.
        auto note_blanks = [&](vector<correspondence_facts::object_info>& info,
                               const vector<object>& objects, const grid& other) {
            if (ranges::all_of(info, [](const auto& one) {
                    return one.has_partner; })) return;
            vector<blank_patch> patches;
            int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM];
            find_blanks(other, background, patches, patch_of);
            for (size_t i = 0; i < info.size() && i < objects.size(); ++i) {
                auto& one = info[i];
                if (one.has_partner) continue;
                for (int row = 0; row < objects[i].summary.h; ++row)
                    for (int col = 0; col < objects[i].summary.w; ++col) {
                        if (objects[i].cells[row][col] == TRANSPARENT) continue;
                        const int at_row = one.summary.y0 + row, at_col = one.summary.x0 + col;
                        if (at_row < 0 || at_row >= other.height ||
                            at_col < 0 || at_col >= other.width) continue;
                        const int16_t which = patch_of[at_row][at_col];
                        if (which < 0) continue;
                        const auto& patch = patches[which];
                        one.blank_left = patch.left;   one.blank_top = patch.top;
                        one.blank_width = patch.width; one.blank_height = patch.height;
                        one.blank_cells = patch.cells;
                        one.blank_enclosed = patch.enclosed;
                        row = objects[i].summary.h; break;
                    }
            }
        };
        note_blanks(pair.output_info, output_objects, input);
        note_blanks(pair.input_info, input_objects, output);

        for (const auto& one_object : pair.input_info) {
            if (one_object.edges_touching == 0)      pair.deleted++;
            else if (one_object.edges_touching == 1) pair.persisted++;
            else                                pair.copied++;
            pair.most_landings = max(pair.most_landings, one_object.edges_touching);
        }
        for (const auto& landing : pair.output_info) {
            if (landing.edges_touching == 0)      pair.invented++;
            else if (landing.edges_touching == 1) pair.from_one++;
            else                                  pair.from_many++;
            pair.most_sources = max(pair.most_sources, landing.edges_touching);
        }
        for (const auto& edge : pair.edges) {
            if (!edge.same_shape)  pair.sits_on++;
            else if (edge.exact)   pair.found_exact++;
            else                   pair.found_recolored++;
            pair.partly_hidden += (edge.hidden > 0);
            pair.turned += (edge.pose != 0);
        }
        pair.dropped = (int)pair.discarded.size();

        // Find uncovered cells on both grids, counting recolored shape matches as coverage.
        auto unexplained = [&](const grid& one_grid,
                               const vector<correspondence_facts::edge>& landing_here,
                               const vector<object>& their_objects,
                               const vector<object>& our_objects,
                               const vector<correspondence_facts::object_info>& ours,
                               const vector<correspondence_facts::edge>& leaving_here,
                               bool from_input) {
            // Mark cells actually covered by incoming shape matches.
            bool carried[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            mark_painted(landing_here, their_objects, our_objects, one_grid,
                         carried, false, true);
            // DOES ANYTHING OVER THERE CLAIM TO BE THIS OBJECT — it went
            // somewhere, or something arrived that named it by name.
            vector<bool> has_a_partner(ours.size(), false);
            for (const auto& edge : leaving_here)
                if (edge.from >= 0 && (size_t)edge.from < has_a_partner.size())
                    has_a_partner[edge.from] = true;
            for (const auto& edge : landing_here)
                if (edge.to >= 0 && (size_t)edge.to < has_a_partner.size())
                    has_a_partner[edge.to] = true;
            // Also mark source cells when their shape matched without extending outside the target.
            for (const auto& edge : leaving_here) {
                if (!edge.same_shape || edge.outside != 0) continue;
                if (edge.from < 0 || (size_t)edge.from >= ours.size()) continue;
                const auto& one_object = ours[edge.from];
                for (int row = 0; row < one_object.summary.h; ++row)
                    for (int col = 0; col < one_object.summary.w; ++col)
                        if (our_objects[edge.from].cells[row][col] != TRANSPARENT)
                            carried[one_object.summary.y0 + row][one_object.summary.x0 + col] = true;
            }
            // Which object each cell belongs to, so a cell can be asked about
            // the object it is part of.
            int16_t object_at[MAX_GRID_DIM][MAX_GRID_DIM];
            for (auto& row : object_at) ranges::fill(row, (int16_t)-1);
            for (size_t i = 0; i < ours.size() && i < our_objects.size(); ++i)
                for (int row = 0; row < ours[i].summary.h; ++row)
                    for (int col = 0; col < ours[i].summary.w; ++col)
                        if (our_objects[i].cells[row][col] != TRANSPARENT)
                            object_at[ours[i].summary.y0 + row][ours[i].summary.x0 + col] = (int16_t)i;
            walk_colored_cells(one_grid, background, [&](int row, int col, int) {
                if (carried[row][col]) return;
                // An uncovered cell is created or removed if its object has no partner.
                const int16_t which = object_at[row][col];
                const bool nobody_claims_the_object =
                    which < 0 || !has_a_partner[which];
                pair.left_over.push_back({ from_input, (int8_t)row, (int8_t)col,
                                           nobody_claims_the_object });
            });
        };
        unexplained(output, forward, input_objects, output_objects,
                    pair.output_info, backward, false);
        unexplained(input, backward, output_objects, input_objects,
                    pair.input_info, forward, true);

        // Count output cells covered by exact matches and those left uncovered.
        bool covered[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        mark_painted(forward, input_objects, output_objects, output, covered);
        walk_colored_cells(output, background, [&](int row, int col, int) {
            if (covered[row][col]) pair.exact_covered_cells++;
            else                   pair.created_cells++;
        });
    }
    facts.valid = true;
    return facts;
}

// Try every segmentation and allowed transformation combination on the training pairs.
// Score the results and return the selected reading. Called by infer().
inline correspondence_facts select_segmentation(const vector<example>& train, int8_t background) {

    // HOW BIG THE THING BEING EXPLAINED IS. Counted once, off the pairs alone,
    // because every reading's leftovers are a share of it.
    int facts_cells = 0;
    for (const auto& train_pair : train)
        for (const grid* one : { &train_pair.input, &train_pair.output })
            walk_colored_cells(*one, background, [&](int, int, int) { ++facts_cells; });

    vector<prepared_segmentation> preparations;
    for (int method = 0; method < (int)segmentation_methods::count; ++method)
        preparations.push_back(prepare_segmentation(train, background, (segmentation_methods)method));
    stable_sort(preparations.begin(), preparations.end(), [](const auto& left, const auto& right) {
        if (left.look.objects_can_fit != right.look.objects_can_fit) return left.look.objects_can_fit;
        return left.look.describe_cost < right.look.describe_cost;
    });

    // Keep each reading score with its matched pairs to avoid repeating the search.
    struct scored_reading {
        correspondence_facts::reading summary;
        vector<correspondence_facts::pair_info> pairs;
    };
    auto search_one = [&](const prepared_segmentation& prepared) {
        const auto& look = prepared.look;
        vector<scored_reading> pass;
        for (int packed = 0; packed < degrees_of_freedom::COMBINATIONS; ++packed) {
            const degrees_of_freedom allowed = degrees_of_freedom::from_index(packed);
            correspondence_facts tried = detect_correspondence(
                train, background, prepared, allowed);
            correspondence_facts::reading scored;
            scored.segmentation = look.segmentation;
            scored.allowed = allowed;
            scored.objects_can_fit = look.objects_can_fit;
            double shares = 0.0;
            for (const auto& pair : tried.pairs) {
                scored.left_over += (int)pair.left_over.size();
                scored.made += (int)ranges::count_if(pair.left_over,
                    [](const auto& cell) { return cell.made_or_gone && !cell.from_input; });
                scored.gone += (int)ranges::count_if(pair.left_over,
                    [](const auto& cell) { return cell.made_or_gone && cell.from_input; });
                scored.objects_used += pair.objects_used;
                shares += pair.describe_cost;
                scored.objects_found +=
                    (int)(pair.input_info.size() + pair.output_info.size());
                scored.exact += pair.found_exact;
                scored.recolored += pair.found_recolored;
                scored.sits_on += pair.sits_on;
                for (const auto& one_object : pair.input_info)
                    scored.box_cells += one_object.summary.w * one_object.summary.h;
                for (const auto& landing : pair.output_info)
                    scored.box_cells += landing.summary.w * landing.summary.h;
            }
            if (!tried.pairs.empty())
                scored.describe_cost = shares / tried.pairs.size();
            pass.push_back({ scored, std::move(tried.pairs) });
        }
        stable_sort(pass.begin(), pass.end(),
                    [](const scored_reading& left, const scored_reading& right) {
            if (left.summary.left_over != right.summary.left_over)
                return left.summary.left_over < right.summary.left_over;
            if (left.summary.objects_used != right.summary.objects_used)
                return left.summary.objects_used < right.summary.objects_used;
            return left.summary.box_cells < right.summary.box_cells;
        });
        return pass;
    };

    // Search every segmentation and keep all results for the table.
    vector<vector<scored_reading>> passes;
    for (const auto& prepared : preparations) passes.push_back(search_one(prepared));

    // Combine compression cost with the cost of unexplained cells.
    // Treat prices within this margin as tied, then compare object counts.
    const double same_price = 0.001;
    const double spelling_out = facts_cells > 0 ? 1.0 / facts_cells : 0.0;
    // Charge created or removed cells half price: their absence of a partner is known,
    // but the rule that creates or removes them is still unexplained.
    const double half_said = 0.5;
    auto price_of = [&](const correspondence_facts::reading& one) {
        const double unsaid = (one.left_over - one.made - one.gone)
                            + (one.made + one.gone) * half_said;
        return one.describe_cost + unsaid * spelling_out;
    };
    size_t elected = 0;
    for (size_t i = 1; i < passes.size(); ++i) {
        const auto& best = passes[elected].front().summary;
        const auto& here = passes[i].front().summary;
        if ((here.objects_used > 0) != (best.objects_used > 0)) {
            if (here.objects_used > 0) elected = i;
        } else if (abs(price_of(here) - price_of(best)) > same_price) {
            if (price_of(here) < price_of(best)) elected = i;
        } else if (here.objects_used < best.objects_used) {
            elected = i;
        }
    }

    // Store results in table order and record the winner separately.
    correspondence_facts facts;
    facts.cells = facts_cells;
    facts.segmentation = passes[elected].front().summary.segmentation;
    auto cell_of = [](const correspondence_facts::reading& one) {
        return (int)one.segmentation * degrees_of_freedom::COMBINATIONS + one.allowed.index();
    };
    facts.readings.resize((int)segmentation_methods::count * degrees_of_freedom::COMBINATIONS);
    facts.reading_pairs.resize(facts.readings.size());
    for (auto& pass : passes)
        for (auto& one : pass) {
            facts.readings[cell_of(one.summary)] = one.summary;
            facts.reading_pairs[cell_of(one.summary)] = std::move(one.pairs);
        }
    facts.chosen = (int8_t)cell_of(passes[elected].front().summary);
    {
        const auto& winner = passes[elected].front().summary;
        for (size_t i = 0; i < passes.size(); ++i) {
            if (i == elected) continue;
            const auto& here = passes[i].front().summary;
            if ((here.objects_used > 0) != (winner.objects_used > 0)) continue;
            if (abs(price_of(here) - price_of(winner)) > same_price) continue;
            if (here.objects_used != winner.objects_used) continue;
            facts.tied_with_chosen.push_back((int8_t)cell_of(here));
        }
    }
    facts.pairs = facts.reading_pairs[facts.chosen];
    facts.valid = true;
    return facts;
}

} // namespace arc
