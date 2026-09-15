// This file explores correspondence;
// how to segment the grid into objects,
// and where each object went across grids.
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

// After segmentation, objects can be matched across grids.
// A match between an input and an output object may allow an object to:
// move, turn, flip, change color, extend past the grid's border.
// A segmentation, paired with a combination of these permissions, is one reading.
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

// Trying a reading means segmenting each training pair and matching its objects
// using the allowed permissions. These matches are called correspondences.
// One reading can produce many correspondences, in both directions: where did
// an input object go, and where did an output object come from?

// This structure collects the correspondences and summaries of the readings tried.
struct correspondence_facts {
    // An edge connects a source object to an object or position in the target grid.
    // It records which objects correspond, along with the position, rotation,
    // and colors recorded by object_match during the search.
    // If the shape was not found, it records an overlap instead.
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
        // How many edges touch it. That one count reads as "how many places
        // this object went" on the input side and "how many objects met here" on
        // the output side — same number, two questions. Filled AFTER the edges
        // are final by count_edges_leaving / count_edges_arriving, never tallied
        // as the edges are made, so it cannot drift from `edges`.
        int edges_touching = 0;
        // SOMETHING FOUND IT. An edge that claims this is that, as opposed to a
        // sits-on edge, which only says what an object lies over and answers
        // nothing. Decided HERE, where the cells still exist: downstream there
        // are only boxes, and a box lies about a hollow shape — a ring's box
        // holds its own hole, so anything sitting in the hole looked spoken
        // for by whatever answered the ring.
        // NOT THE SAME AS ANSWERED. An object nothing partners still has an
        // answer — it was deleted, or inserted, or something covers it. What
        // it lacks is an object on the other grid that is IT.
        bool has_partner = false;
        // WHICH SET OF COPIES this object belongs to on its own grid, -1 for
        // none. A grid can hold several motifs at once, and a set is the
        // thing they are copies OF.
        int16_t motif = -1;
        // THE MEDOID of that set — the member that disagrees least with all the
        // others. A medoid is the most typical member and has to BE one of them,
        // unlike an average, which is what an exemplar has to be: a real block
        // on the grid. Not the biggest and not the first, both of which pick by
        // luck; a block the noise landed on is complete and still the odd one.
        bool medoid = false;
        // THE BLANK IT FILLED, or the blank it left behind. An object nothing
        // answers corresponds to no object — but it still covers something on
        // the other grid: a patch of background. Closed in by colored cells on every side
        // that patch is a HOLE, and filling a hole is the commonest way a
        // grid makes something where there was nothing. Only asked of the
        // objects nothing answers; width 0 means it covers no blank at all.
        int8_t blank_left = 0, blank_top = 0, blank_width = 0, blank_height = 0;
        int16_t blank_cells = 0;
        bool blank_enclosed = false;
    };
    // A cell no edge carried, named by the grid it sits on. MADE OR GONE
    // separates the two ways that happens, and it is one fact read from two
    // ends: nothing over there is this cell's object. On the output that means
    // the cell was made where it stands; on the input, that it went nowhere.
    // The other way is an object whose partner IS known and still leaves these
    // cells unsaid, which is the only one of the three that is the reading
    // failing.
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
        // The objects the chosen segmentation finds, named by the grid they sit on. Both
        // grids are segmented the same way and on the same block, which is what lets
        // the two directions be read into one list of edges.
        vector<object_info> input_info;
        vector<object_info> output_info;
        // BOTH WAYS ROUND, IN ONE LIST. Every edge says which grid it starts on,
        // so "where did this object go" and "where did that landing come from"
        // are the same question asked of the same set. Where the two ends agree
        // only the stronger claim is kept.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
        vector<edge> edges;
        // Matches that were found and then thrown away, with why — otherwise
        // "never matched" and "matched and lost" look the same in the log.
        vector<edge> discarded;
        vector<int8_t> discard_why;
        // How it decided: 0 the objects fit together, 1 nothing fit so the weaker
        // greedy reading was used, 2 the search ran out of steps — which is a
        // giveup, never an answer.
        int8_t verdict = 0;
        // THE CELLS the pass leaves unexplained ON BOTH GRIDS, and how many
        // objects it used — what picks the way of cutting up. Kept as the cells
        // and not as a tally, so what is counted and what can be shown are one
        // thing: a count nobody can point at is a claim you cannot check.
        vector<left_over_cell> left_over;
        int objects_used = 0;
        // Objects with the same input-to-output movement. A move said by several
        // objects at once is the only thing two objects in this layer ever agree
        // about; said by one it is a coincidence with nobody to back it. Kept
        // as a fact and judged by nobody yet.
        // `color` is what the objects have in common, -1 when they have nothing:
        // the one thing a cluster can be named by that survives being read in
        // another pair, where the move itself is a different number.
        struct moved_together { int8_t down, across, color; int16_t objects; };
        vector<moved_together> agreed_moves;
        // What the segmentation saves on this pair's FROM grid, as a share of saying the
        // grid plainly — see describe_cost. Free here because the objects already
        // exist, and it owes nothing to the matching, which is the point of it.
        double describe_cost = 1.0;
        // The same three questions from each end: where an object went, where a
        // landing came from. Nowhere/nothing, one, several.
        int deleted = 0, persisted = 0, copied = 0;
        int invented = 0, from_one = 0, from_many = 0;
        // The extremes those three hide: the most places any one object went,
        // and the most objects that met at any one landing. The two of them ARE
        // the class — 1-1, 1-9, 3-2 — with no naming in between.
        int most_landings = 0, most_sources = 0;
        // WHAT KIND OF CLAIM each kept forward edge makes, since a line on the
        // screen looks the same whichever it is: found as itself, found wearing
        // another color, or not found at all and only sitting on something.
        // Partly hidden and turned are how many of them were covered by
        // another object / not upright. Dropped is what was found and thrown out.
        int found_exact = 0, found_recolored = 0, sits_on = 0;
        int partly_hidden = 0, turned = 0, dropped = 0;
        // Coverage counts EXACT edges only: a recolored positional hit is
        // unconstrained (free color map) and explains nothing by itself.
        // Non-background cells of the INTO grid covered by an exact edge
        int exact_covered_cells = 0;
        // Non-background into cells no exact edge covers — made, not carried
        int created_cells = 0;
    };
    // One entry per train pair
    vector<pair_info> pairs;
    // Non-background cells across BOTH grids of every train pair — what a
    // reading's leftovers are a share of. A bare count of what a reading could
    // not account for says nothing without the size of what it failed at.
    int cells = 0;
    // The way of cutting up that explained most, judged over the WHOLE train
    // set. FOR NOW one segmentation serves every pair: a transformation that segments the
    // grid up differently from pair to pair would be a strange puzzle, and
    // holding it fixed is the stronger claim. Revisit if a puzzle needs it.
    segmentation_methods segmentation = segmentation_methods::connected_monochrome;
    // EVERY READING THERE IS, IN THE ORDER THE TABLE IS DRAWN: one row per segmentation,
    // one column per freedom, so a reading's place in this list IS its cell —
    // segmentation * degrees_of_freedom::COMBINATIONS + allowed.index(). Every
    // segmentation is looked at and every freedom searched, so the table has no
    // holes and a cell never has to be looked up. Do not collapse this to the
    // winner: a segmentation cannot tell that it read too coarsely, and only a
    // consumer coming up empty finds out. The segmentation
    // ITSELF is recorded alongside the numbers, because two readings can tie on
    // every number and differ.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
    struct reading {
        // Which segmentation, and what a match was allowed to change
        segmentation_methods segmentation = segmentation_methods::connected_monochrome;
        degrees_of_freedom allowed;
        // How many objects the segmentation FOUND, and how many it USED — the log prints
        // them as "segmented into X, used Y"
        int objects_found = 0, objects_used = 0;
        int left_over = 0, box_cells = 0;
        // Of the left over, how many were MADE — nothing on the other grid is
        // that cell's object, so it was made where it stands. The rest have a
        // partner that still cannot say where they came from, which is the only
        // part that is the reading failing. Kept per reading because a table
        // that shows one number for the two cannot be argued with.
        int made = 0, gone = 0;
        // The grid-only score, AVERAGED over the pairs — it is a share of plain
        // cost, and shares add up to nothing meaningful. Owes nothing to the
        // matching, so it is had BEFORE any search — see describe_cost.
        double describe_cost = 1.0;
        // FALSE MEANS THE SECOND GLANCE KILLED IT: not one object this segmentation makes
        // would fit on the other grid, so no match can exist and searching it
        // would be paying to be told so. Read off the two grids' sizes.
        bool objects_can_fit = true;
        int exact = 0, recolored = 0, sits_on = 0;
    };
    vector<reading> readings;
    // WHICH ONE THE ELECTION CHOSE, as a place in `readings`. A fact of its own,
    // because the list is in table order and the winner can sit anywhere in it —
    // being first was the only thing that ever said so.
    int8_t chosen = 0;
    // THE ONES THE ELECTION COULD NOT SEPARATE FROM THE WINNER, as places in
    // `readings`. Saying only the winner claims a distinction the reading never
    // made -- on 009d5c81 three segmentation methods explain exactly as much and declaration
    // order picked between them.
    vector<int8_t> tied_with_chosen;
    // THE OBJECTS AND EDGES OF EVERY READING, in `readings` order, so the work is
    // done once. Without it a consumer wanting a reading other than the chosen
    // one has to run it again — and the pane wants all of them, which meant the
    // whole election happening twice on every puzzle.
    // `pairs` is reading_pairs[chosen] said again, for the consumers that only
    // ever want the one that won.
    vector<vector<pair_info>> reading_pairs;
    bool valid = false;
};

// Searching for an object from one grid in the other produces an object_match:
// where it was found, with what rotation and colors.
// It does not store which objects correspond.
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

// Every place the object turns up, tried every way round, plus how many ways
// there were to try — the two numbers belong together, because a shape that
// can sit eight different ways had eight times as many chances to fit.
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
        int lit = 0;
        for (int row = 0; row < posed.height; ++row)
            for (int col = 0; col < posed.width; ++col)
                lit += (posed.data[row][col] != TRANSPARENT);

        for (int at_row = first_row; at_row <= last_row; ++at_row)
            for (int at_col = first_col; at_col <= last_col; ++at_col) {
                // Cells past the edge were never shown. They constrain nothing
                // and they are not evidence — `shown` is what the placement
                // actually put on the table.
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

    // A SHAPE THE OTHER GRID HOLDS SEVERAL OF. Read off that grid's own objects,
    // before anything is matched, so it can license a hang-off without the
    // matches vouching for themselves.
    // A scan per object, so this is objects x landings. Small on everything seen
    // so far and cheap enough to leave plain; if a grid ever arrives with
    // hundreds of objects this wants to be a set built once instead.
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
            // Odds of this landing: every cell the grid SHOWED had to find the
            // color it actually found there. A hidden cell found nothing —
            // something is in front of it — and a cell off the edge was never
            // shown at all, so neither is evidence and neither counts. What is
            // left is what the placement really put on the table.
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
            correspondence_facts::edge edge;
            edge.from = (int16_t)from_index;
            edge.row = (int8_t)hit.row; edge.col = (int8_t)hit.col;
            edge.pose = (int8_t)hit.pose;
            edge.exact = hit.exact;
            edge.stayed_put = stationary;
            edge.hidden = (int16_t)hit.hidden;
            edge.outside = (int16_t)hit.outside;
            edge.recolored_to =
                (!hit.exact && one_object.color >= 0)
                    ? hit.color_map[one_object.color] : (int8_t)-1;
            if (!hit.exact) for (int c = 0; c < COLOR_COUNT; ++c) edge.color_map[c] = hit.color_map[c];
            const bool stands_alone = beats_chance(found_here.places, odds);
            // ANYTHING LESS THAN THE WHOLE OBJECT AS ITSELF MUST EARN ITS PLACE
            // ALONE — only the full identity claim may lean on the set test.
            // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
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

        // STAYING PUT AND BEING COPIED ARE NOT RIVALS. Finding an object where it
        // already was used to end the hunt, so a shape that stayed and was also
        // stamped out elsewhere could never show its copies — the one thing this
        // whole layer is for. Both questions are asked now, and the judges that
        // already exist decide: a lucky landing dies to beats_chance, and one
        // that explains nothing new dies in the fitting.
        const int poses = allowed.pose ? 8 : 1;
        keep_positional(
            find_in(cells, into, background, false,
                    one_object.summary.y0, one_object.summary.x0, false, poses), true);
        // HANGING OFF THE EDGE IS ONLY HONEST ON AN OBJECT YOU ALREADY KNOW. Most
        // of an object sitting outside the grid is a claim about cells nobody can
        // see, and it costs nothing to make. What makes it checkable is knowing
        // the whole shape from somewhere else: an object with IDENTICAL TWINS on
        // its own grid is one of a set, the twins show what it looks like, and
        // the wall is then the only reason this one is cut short. An object with
        // no twin has nothing but the wall to lean on.
        // The twins are a fact about ONE GRID, found before anything is matched
        // — see find_copies. Licensing a match on something read off the
        // matches would be the matches vouching for themselves, and a wrong
        // licence lets in the unbounded thing while a wrong refusal only costs
        // one true match.
        // ON EITHER GRID, because a thing stamped out many times is ONE object
        // where it came from and a set where it landed. 045e512c's key shape has
        // no twin of its own — its copies are all on the other side — and asking
        // only its own grid refused every one of them.
        // The block segmentation methods do not come through here: a grid of blocks says where
        // every copy sits, so a block cut short by the wall is already placed.
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
            // Same odds, read off the object that was matched: a single-color
            // body is `count` cells all wearing its color; the whole-grid
            // object is the grid itself.
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
    // Cells of the target grid that need painting
    vector<pair<int, int>> cells;
    // One per match that lands wholly inside the grid
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
    // EVERY CELL OF THE GRID IS A CELL, background included. Left out, an object
    // landing on background was painting on nothing and got thrown out of the
    // fitting — so anything that was not in one grid and appears in the other,
    // out of empty space, could never be answered.
    // ADDING CELLS FORCES NOTHING. Only cells an EXACT object reaches are
    // REQUIRED, and an exact object can never land on background: an object's own
    // background is cut out as transparent, so it paints no background cell.
    // What this adds is somewhere for a new thing to land, not an obligation.
    int cell_id[MAX_GRID_DIM][MAX_GRID_DIM];
    for (int row = 0; row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < MAX_GRID_DIM; ++col) cell_id[row][col] = -1;
    for (int row = 0; row < into.height && row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < into.width && col < MAX_GRID_DIM; ++col) {
            cell_id[row][col] = (int)laid_out.cells.size();
            laid_out.cells.push_back({ row, col });
        }

    // Every match is a candidate object, recolored ones included: a recolor
    // keeps the SHAPE, and shape is what has to fit. Fitting is the strong
    // claim here, so it does the filtering that per-match judging could not.
    for (size_t found_index = 0; found_index < found.size(); ++found_index) {
        const grid cells = edge_footprint(found[found_index], from_objects, into_objects);
        candidate candidate{ found_index, {}, {} };
        bool inside = true;
        for (int row = 0; row < cells.height && inside; ++row)
            for (int col = 0; col < cells.width; ++col) {
                if (cells.data[row][col] == TRANSPARENT) continue;
                const int into_row = found[found_index].row + row;
                const int into_col = found[found_index].col + col;
                // Off the edge is not a cell of this grid at all: it is
                // neither painted nor hidden, it was simply never shown.
                if (into_row < 0 || into_row >= into.height ||
                    into_col < 0 || into_col >= into.width) continue;
                const int id = cell_id[into_row][into_col];
                // The cells it is hidden at are its own too — kept apart,
                // because they are what puts it BEHIND whoever paints them.
                if (!landing_shows(found[found_index], cells, into, row, col)) {
                    if (id >= 0) candidate.behind.push_back(id);
                    continue;
                }
                if (id < 0) { inside = false; break; }   // paints outside the grid
                candidate.cells.push_back(id);
            }
        if (inside && !candidate.cells.empty()) laid_out.objects.push_back(std::move(candidate));
    }

    laid_out.options.resize(laid_out.cells.size());
    for (int object_index = 0; object_index < (int)laid_out.objects.size(); ++object_index)
        for (int id : laid_out.objects[object_index].cells)
            laid_out.options[id].push_back(object_index);
    // SAME CELLS EXPLAINED, FEWER WORDS SPENT. The search takes the first fitting
    // it reaches, so this order IS the preference, and a 1x1 object can tile
    // anything. Two tilings can cover exactly the same cells and the fitting is
    // happy with either — what separates them is how much had to be said. One
    // object covering eight cells is one thing said; four objects covering the
    // same eight is four, and four claims that happen to agree are not better
    // evidence than the one claim they agree with.
    // BIGGEST FIRST, THEN PLAINEST — says_more, the one order all three passes
    // ask for. Asked the other way round, a three-cell stick that matched
    // exactly beat an eight-cell shape that matched in another color, so the
    // ring in 045e512c was explained four times over by the little stick lying
    // inside its edges instead of once by the ring it is a copy of.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
    for (auto& options : laid_out.options)
        stable_sort(options.begin(), options.end(), [&](int left, int right) {
            return says_more(found[laid_out.objects[left].match_index],
                             laid_out.objects[left].cells.size(),
                             found[laid_out.objects[right].match_index],
                             laid_out.objects[right].cells.size());
        });

    // SOME CELLS ARE SIMPLY NEW. Only cells an exact object could account for are
    // REQUIRED; demanding the rest be explained makes the search invent a source.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
    laid_out.required.assign(laid_out.cells.size(), false);
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

    vector<int> painted_by(laid_out.cells.size(), -1);
    long long work = 0;
    // Limit the search based on problem size, counting each cell checked.
    // Report when the limit is reached; a solution may still exist.
    const long long work_budget =
        (long long)(laid_out.objects.size() + 1) * (long long)(laid_out.cells.size() + 1);
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

    auto fits_together = [&](auto&& self) -> bool {
        if (work > work_budget) { out_of_budget = true; return false; }
        // The cell with the fewest objects still able to paint it decides next;
        // zero options means this line of choices is dead. Only required cells
        // drive the search — a made cell is answered by being left alone.
        int target = -1, fewest = 0;
        for (size_t id = 0; id < laid_out.cells.size(); ++id) {
            if (painted_by[id] >= 0 || !laid_out.required[id]) continue;
            int options = 0;
            for (int object_index : laid_out.options[id]) {
                bool free_here = true;
                for (int other : laid_out.objects[object_index].cells) {
                    ++work;
                    if (painted_by[other] >= 0) { free_here = false; break; }
                }
                options += free_here;
            }
            if (options == 0) return false;
            if (target < 0 || options < fewest) { target = (int)id; fewest = options; }
            if (work > work_budget) { out_of_budget = true; return false; }
        }
        if (target < 0) return true;   // every carried-over cell accounted for
        // AN OBJECT WITH NO GOOD LANDING SIMPLY STANDS ALONE. Every object used
        // to be offered a first landing before any object was offered a second,
        // so an object that corresponds to nothing was handed something to be
        // wrong about — in 045e512c a three-cell stick took a whole ring,
        // because the ring it belonged to had already landed and was skipped.
        // Fairness is not a finding. The options are in preference order and
        // nothing overrides it now; an object nothing wanted says nothing, which
        // is the truth and is what ALONE and LEFT OVER are for.
        for (int object_index : laid_out.options[target]) {
            bool free_here = true;
            for (int id : laid_out.objects[object_index].cells) {
                ++work;
                if (painted_by[id] >= 0) { free_here = false; break; }
            }
            if (!free_here) continue;
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
    vector<bool> painted(laid_out.cells.size(), false);
    for (int object_index : chosen) {
        const size_t match_index = laid_out.objects[object_index].match_index;
        edges.push_back(found[match_index]);
        kept[match_index] = true;
        for (int id : laid_out.objects[object_index].cells) painted[id] = true;
    }
    // THE FITTING SETTLES WHAT COMPETES AND SAYS NOTHING ABOUT THE REST. It drives
    // only on cells an EXACT object could reach, so an object landing where nothing
    // was — a thing that appears out of empty space — is never needed and never
    // taken. Not needed is not wrong: painting only cells the fitting left alone,
    // it takes nothing from anyone and accounts for cells nothing else does.
    // STRONGEST FIRST, so when two of these want one cell the better claim has
    // it — the same order the fitting offers its own choices in.
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

inline int lit_cells(const grid& cells) {
    int lit = 0;
    for (int row = 0; row < cells.height; ++row)
        for (int col = 0; col < cells.width; ++col)
            lit += (cells.data[row][col] != TRANSPARENT);
    return lit;
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
    // THE SAME PREFERENCE THE FITTING USES — says_more, asked in a hurry. Said
    // differently in the two places, a pair would disagree about which reading
    // was better depending on whether the search finished.
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

    // WHAT THE OTHER EXAMPLES DISAGREE WITH IS NEVER OFFERED. It has to go
    // BEFORE the fitting and not after: the fitting uses every candidate to
    // knock out its rivals, so striking a winner afterwards leaves the rival it
    // beat dead and its cells bare — one red object of 05a7bcf2 pair 0 went
    // missing in exactly that way.
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
                // AN EDGE THAT NAMES AN OBJECT SPEAKS FOR IT. Matching runs both
                // ways and the merge keeps one of every agreeing pair, so the
                // far end has no edge of its own to be answered by — and a
                // landing that names an object was always a claim about two
                // things. Only a landing on bare ground needs the shapes to
                // agree, because there is no object there to have been meant.
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
    // The block is whichever side is smaller, decided once on the first pair
    // and held to by every other — two pairs disagreeing about which end grows
    // is not a tiling, it is two different puzzles.
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
    // As many objects as an object index can name — edges carry it in one signed
    // byte. The finest segmentation is one object per cell, so a small grid must be able
    // to come apart completely; past this a reading simply cannot be written
    // down, and it sinks on its own by leaving the rest unexplained.
    // AS MANY OBJECTS AS THERE ARE CELLS. Every cell its own object is the most a
    // grid can ever be segmented into, so that is the only honest ceiling; any
    // number below it stops a segmentation part way and hands back a corner of the
    // grid as though it were the whole of it.
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
        // A CUT THAT DOES NOT CUT IS NOT A READING — one object means there is no
        // motif, and it would win the election for free. The step segmentation is exempt.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
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
    // A segmentation that finds nothing has said nothing: you still spell the grid out
    // cell by cell, and you paid a position for the privilege. Never the cheap
    // answer, because it is over 1 by construction.
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
    // The shortest period that holds, or the whole size when none does. Past
    // halfway there is too little left to compare, so a long period passes on a
    // cell or two and means nothing.
    // Why two whole copies, and what it costs:
    // notes/4 - To Do/A Repeat Has To Fit Twice.md
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
inline pair<int, int> block_measured_on(const grid& one_grid, int8_t background,
                                        int sizes_width, int sizes_height) {
    int width = sizes_width, height = sizes_height;
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
        block_measured_on(input, background, sizes_width, sizes_height);
    const auto from_output =
        block_measured_on(output, background, sizes_width, sizes_height);
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
        // ITSELF BEATS ITSELF-IN-ANOTHER-COLOR. The hits come back in pose order,
        // so the first one can be a recolor of an early pose while a later pose
        // sits there exactly — and taking the first would report the weaker claim
        // about the same two blocks.
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
        correspondence_facts::edge edge;
        edge.from = (int16_t)from_index;
        edge.to = (int16_t)target_index;
        edge.row = (int8_t)target.summary.y0;
        edge.col = (int8_t)target.summary.x0;
        edge.pose = (int8_t)hit.pose;
        edge.exact = hit.exact;
        edge.stayed_put = (from_info[from_index].summary.y0 == target.summary.y0 &&
                           from_info[from_index].summary.x0 == target.summary.x0);
        edge.outside = (int16_t)hit.outside;
        edge.recolored_to =
            (!hit.exact && from_info[from_index].color >= 0)
                ? hit.color_map[from_info[from_index].color] : (int8_t)-1;
        if (!hit.exact) for (int c = 0; c < COLOR_COUNT; ++c) edge.color_map[c] = hit.color_map[c];
        edges.push_back(edge);
    };

    vector<bool> answered(from_info.size(), false);
    // WHERE DID EACH LANDING COME FROM. Its own place first, so a grid whose
    // two sides hold the same blocks reads as each staying put rather than as
    // everything matching everything.
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

// Segment each training pair and match objects in both directions.
// Merge duplicate matches and store the results in correspondence_facts.
inline correspondence_facts detect_correspondence(const vector<example>& train,
                                                  int8_t background,
                                                  segmentation_methods segmentation,
                                                  degrees_of_freedom allowed = {}) {
    correspondence_facts facts;
    const auto [sizes_step_width, sizes_step_height] = repeating_step_of(train);
    if (train.empty()) return facts;
    facts.pairs.resize(train.size());
    facts.segmentation = segmentation;

    // The objects are wanted twice — once to match on, and again to read the
    // numbers off after the pairs have been set beside one another — so they
    // are segmented once and kept.
    vector<vector<object>> input_objects_of(train.size()), output_objects_of(train.size());

    auto pair_was_read = [&](size_t which) {
        return train[which].input.width > 0 && train[which].input.height > 0 &&
               train[which].output.width > 0 && train[which].output.height > 0;
    };
    // A pattern reading is read by a different collector: a block is a window
    // onto a tiling that carries on, a shape is a thing that does not.
    // HANGING OFF THE EDGE IS NOT WHAT SEPARATES THEM — every reading may do
    // that now, because a wall cutting a copy in half is a fact about the grid
    // and not about the object.
    const bool pattern_reading = (segmentation == segmentation_methods::minimal_repeating_set ||
                                  segmentation == segmentation_methods::smallest_of_two_grids);
    vector<int> step_width_of(train.size(), 0), step_height_of(train.size(), 0);

    // THE CUTTING, ONCE. The objects owe nothing to what any other pair said, so
    // they are made here and the matching may then be run over them more than
    // once.
    for (size_t pair_index = 0; pair_index < train.size(); ++pair_index) {
        if (!pair_was_read(pair_index)) continue;
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        // ONE CUT, ONE BLOCK, BOTH GRIDS. Cut once here and both readings work
        // off the same objects — measured per direction the two would disagree
        // about what an object is, and their edges could not be put in one list.
        const auto [step_width, step_height] = block_of(input, output, background,
                                                        segmentation, sizes_step_width,
                                                        sizes_step_height);
        vector<object>& input_objects = input_objects_of[pair_index];
        vector<object>& output_objects = output_objects_of[pair_index];
        segment_for_correspondence(input, background, segmentation, pair.input_info,
                                   &input_objects, step_width, step_height);
        segment_for_correspondence(output, background, segmentation, pair.output_info,
                                   &output_objects, step_width, step_height);
        // Which objects are copies of one another, and which speaks for each
        // set. Here, where the cells still exist — a shape hash and a cell count
        // cannot tell one block from another, and downstream that is all there
        // is left of them.
        const bool one_set_for_the_grid = (segmentation == segmentation_methods::minimal_repeating_set);
        find_copies(pair.input_info, input_objects, one_set_for_the_grid);
        find_copies(pair.output_info, output_objects, one_set_for_the_grid);
        // What the segmentation cost to say, on each grid. The pair is described by both.
        pair.describe_cost =
            0.5 * (describe_cost(input, background, pair.input_info) +
                   describe_cost(output, background, pair.output_info));
        step_width_of[pair_index] = step_width;
        step_height_of[pair_index] = step_height;
    }

    // ASKED ONCE, THEN ASKED AGAIN. Once to hear what each pair says on its
    // own, and again once the pairs have been set beside one another, because a
    // match the others disagree with has to be gone before the fitting.
    auto match_one_pair = [&](size_t pair_index, const what_the_pairs_agreed* agreed) {
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        const vector<object>& input_objects = input_objects_of[pair_index];
        const vector<object>& output_objects = output_objects_of[pair_index];
        const int step_width = step_width_of[pair_index];
        const int step_height = step_height_of[pair_index];
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

        // WHERE THE TWO ENDS SAY THE SAME THING, SAY IT ONCE. An input object
        // found at an output object and that output object found back at the input
        // object are one correspondence, not two, and counting it twice would
        // make every object look twice as busy as it is. The stronger claim
        // survives — found as itself beats found in another color beats only
        // sitting on something — and a tie goes to the forward edge, because
        // that is the one you could actually run.
        // WHICH TWO OBJECTS AN EDGE JOINS, said the same way whichever end it was
        // read from: the input object, then the output object. A landing with no
        // object under it has no other end, so it agrees with nothing.
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

        // IN THE ORDER THE SCREEN READS: the input grid first, then down it.
        // WHERE AN OBJECT SITS, not its place in the segmentation — the segmentation is biggest
        // first, so walking by place hopped up and down the grid.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
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

    // WHAT THE PAIRS AGREE ABOUT. A move several objects say at once is evidence
    // inside one pair; a move every pair says is a rule. THE NUMBER CANNOT BE
    // WHAT REPEATS — the reds in 05a7bcf2 fall 9, then 8, then 7, because the
    // band they fall from is a different thickness each time — so what carries
    // from pair to pair is what the objects have in common, which is their color.
    // One move per color per pair, or that color is not stating anything: the
    // yellows there split four ways in one pair and are absent from another.
    // STAYING PUT NEEDS NO NAME. It is the same move in every frame and it is
    // what "nothing happened here" looks like, so it is always agreed.
    // THE SIEVE ONLY SPEAKS WHEN THE PAIRS DO. Where no color clusters in every
    // pair the pairs have agreed about nothing, and the edges are left alone —
    // a puzzle answered one object to many places has no clusters at all, and a
    // rule that cannot see it must not be allowed to empty it.
    // The one move a color makes in a pair, or nothing when it makes none or several.
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
            // NOTHING TO TAKE BACK, NOTHING TO ASK AGAIN. A pair whose every
            // match already agrees would be read a second time to the same
            // answer, and the matching is the expensive half of this file.
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
        const vector<object>& input_objects = input_objects_of[pair_index];
        const vector<object>& output_objects = output_objects_of[pair_index];

        // The two halves of the merged list, each in its own frame, so every
        // helper below stays the one-direction function it always was.
        vector<correspondence_facts::edge> forward, backward;
        for (const auto& edge : pair.edges)
            (edge.from_input ? forward : backward).push_back(edge);

        // HOW MANY EDGES TOUCH EACH OBJECT, counting both ends. An object is busy
        // whether it went somewhere or something arrived on it, and it is
        // answered by either one as well — asking only who arrived left an object
        // that plainly went somewhere reading as though nothing spoke for it.
        count_edges_leaving(forward, pair.input_info);
        count_edges_leaving(backward, pair.output_info);
        count_edges_arriving(forward, input_objects, pair.output_info, output_objects);
        count_edges_arriving(backward, output_objects, pair.input_info, input_objects);
        pair.objects_used = (int)pair.edges.size();

        // A REFUSAL IS ONLY A REFUSAL IF ITS OBJECT NEVER GOT FOUND. Both
        // readings are kept and only the surviving edges were merged, so every
        // drawn line had its thrown-away twin sitting in this list beside it.
        // An object something found was not refused — what else it might have
        // matched is a road not taken. Asked of the OBJECT and not of the match,
        // because most matches name no object on the other side at all: they
        // are a place, and two places cannot be told to mean one thing.
        // ASKED HERE AND WITH `has_partner`, which is the same question the screen
        // asks. Asked earlier it had to be answered a second way — is this object
        // a named end of some edge — and that misses an object found by a landing
        // covering it, so an object could read as found and have refusals listed
        // as though nothing had found it.
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

        // WHERE THE ONES STANDING ALONE CAME FROM, OR WENT. An object nothing
        // answers has no partner object, but it is not nowhere: it covers a
        // patch of background on the other grid. Asked only of those objects,
        // because for the rest the partner IS the answer. See object_info's
        // blank fields, and find_blanks for what closed in means.
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

        // What the reading leaves unexplained — the number the way of cutting up
        // is judged on. A cell is explained when a REAL edge touches it: one that
        // found the shape, not one that only sits on top of something. A
        // RECOLORED match counts as explaining. ASKED OF BOTH GRIDS, and now
        // answered from both ends — a cell can be explained either by something
        // landing on it or by the object it belongs to having gone somewhere.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
        auto unexplained = [&](const grid& one_grid,
                               const vector<correspondence_facts::edge>& landing_here,
                               const vector<object>& their_objects,
                               const vector<object>& our_objects,
                               const vector<correspondence_facts::object_info>& ours,
                               const vector<correspondence_facts::edge>& leaving_here,
                               bool from_input) {
            // CARRIED — a real edge actually lands on this cell and brings it
            // from the other grid. Nothing else counts: an object used to be
            // credited with its WHOLE footprint the moment anything spoke for
            // it, so one edge on an object of two hundred cells accounted for all
            // two hundred, and a reading could score a perfect nothing-left-over
            // while carrying two cells. That is why the three had to come apart
            // before any of them could be believed.
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
            // AN OBJECT THAT WENT SOMEWHERE WHOLE TOOK ITS OWN CELLS WITH IT.
            // Nothing lands back on the side an object leaves from, so its cells
            // are painted by nobody however completely it was found — the input
            // of 00576224 is one block matched to all nine blocks of the output
            // and its own four cells still read as unaccounted for.
            // WHOLE is the whole of it: the edge carries the object's own shape
            // and none of it fell off the grid. An object credited for ANY edge
            // touching it is the leak this file just closed — one match on a
            // two-hundred-cell object cannot speak for two hundred cells.
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
                // MADE, or UNEXPLAINED. Nothing over there is this object, so
                // the cell was made where it stands (or, read from the input
                // side, it went nowhere) — a real answer, and a different
                // one from an object whose partner is known and still cannot
                // say where these cells came from.
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

        // Coverage ledger, output side: cells carried over by an exact edge vs
        // made from nothing. Still an output-side question — what the answer
        // owes to what it was given.
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
    const auto [sizes_step_width, sizes_step_height] = repeating_step_of(train);

    // HOW BIG THE THING BEING EXPLAINED IS. Counted once, off the pairs alone,
    // because every reading's leftovers are a share of it.
    int facts_cells = 0;
    for (const auto& train_pair : train)
        for (const grid* one : { &train_pair.input, &train_pair.output })
            walk_colored_cells(*one, background, [&](int, int, int) { ++facts_cells; });

    // THE LOOKING. One number per segmentation, off the grids alone. Scored on BOTH
    // grids: the segmentation has to make sense of the pair, and comparing a total taken
    // on one grid against a total taken on the other was never a fair
    // question — which is why direction is not decided here, or anywhere.
    vector<correspondence_facts::reading> looks;
    for (int segmentation = 0; segmentation < (int)segmentation_methods::count; ++segmentation) {
        correspondence_facts::reading look;
        look.segmentation = (segmentation_methods)segmentation;
        double shares = 0.0;
        int counted = 0;
        for (const auto& train_pair : train) {
            const grid& input = train_pair.input;
            const grid& output = train_pair.output;
            if (input.width <= 0 || input.height <= 0 ||
                output.width <= 0 || output.height <= 0) continue;
            const auto [step_width, step_height] =
                block_of(input, output, background, look.segmentation,
                         sizes_step_width, sizes_step_height);
            vector<correspondence_facts::object_info> input_info, output_info;
            segment_for_correspondence(input, background, look.segmentation, input_info,
                                       nullptr, step_width, step_height);
            segment_for_correspondence(output, background, look.segmentation, output_info,
                                       nullptr, step_width, step_height);
            shares += 0.5 * (describe_cost(input, background, input_info) +
                             describe_cost(output, background, output_info));
            ++counted;
            look.objects_found += (int)(input_info.size() + output_info.size());
            // THE SECOND GLANCE, at the grid the objects would have to turn up
            // on. One that fits somewhere is all this asks for — a segmentation where not
            // one object could sit on the other grid cannot match, and no amount
            // of freedom rescues it. Asked both ways, because both are read.
            // IT IS NOW STRICTER THAN THE MATCHING IT GUARDS, and here is what
            // it would cost: a pair whose grids are a turn of each other, say a
            // 10-wide 2-tall input against a 2-wide 10-tall output. Neither
            // object fits the other grid as it sits, so the segmentation is killed here —
            // and yet a quarter turn lands it exactly, which is the whole answer
            // to that puzzle. This asks about the object as it sits, while the
            // matching may turn it and may hang it off the edge.
            // It only fires when NOTHING fits either way, so nothing seen so far
            // has tripped it. It is wrong on the day a turn or a wall is the
            // answer, and that is the day to delete it.
            auto any_fits = [](const vector<correspondence_facts::object_info>& objects,
                               const grid& other) {
                return ranges::any_of(objects, [&](const auto& one_object) {
                    return one_object.summary.w <= other.width && one_object.summary.h <= other.height; });
            };
            if (!any_fits(input_info, output) && !any_fits(output_info, input))
                look.objects_can_fit = false;
        }
        if (counted) look.describe_cost = shares / counted;
        looks.push_back(look);
    }
    // Cheapest look first, but everything the second glance killed sinks to the
    // bottom — still there, because a dead segmentation is the only honest answer when
    // every segmentation is dead.
    stable_sort(looks.begin(), looks.end(),
                [](const correspondence_facts::reading& left,
                   const correspondence_facts::reading& right) {
        if (left.objects_can_fit != right.objects_can_fit) return left.objects_can_fit;
        return left.describe_cost < right.describe_cost;
    });

    // THE MATCHING, and going back round. One pass = one segmentation's eight freedoms.
    // A pass that found nothing is the signal to take the next segmentation, so the walk
    // stops the moment a segmentation actually says something.
    // A reading and the pairs it was scored from, kept together so nothing has
    // to be read a second time to get at the edges.
    struct scored_reading {
        correspondence_facts::reading summary;
        vector<correspondence_facts::pair_info> pairs;
    };
    auto search_one = [&](const correspondence_facts::reading& look) {
        vector<scored_reading> pass;
        for (int packed = 0; packed < degrees_of_freedom::COMBINATIONS; ++packed) {
            const degrees_of_freedom allowed = degrees_of_freedom::from_index(packed);
            correspondence_facts tried = detect_correspondence(
                train, background, look.segmentation, allowed);
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

    // EVERY CUT IS MATCHED. No early stop, so no reading is ever elected over
    // something nobody tried, and there is no line in the table between what
    // competed and what was only shown.
    //
    // The pane wants every reading too, so each is READ ONCE and kept — see
    // reading_pairs. The looking decides who wins; it no longer decides who
    // gets a turn, and it does not decide the ORDER either — see below.
    vector<vector<scored_reading>> passes;
    for (const auto& look : looks) passes.push_back(search_one(look));

    // WHAT A READING FAILS TO EXPLAIN, IT STILL HAS TO SPELL OUT. So there is
    // ONE number and not two keys: what the segmentation costs to say, plus a plain
    // cell's price for every cell it could not account for.
    //
    // TWO KEYS ALWAYS BREAK, whichever goes first, because the winner is then
    // whoever ignores the other one. Fewest left over first: segment into single
    // cells and every cell finds a cell, so nothing is ever left over and a
    // reading twice as long as the grid wins. Cheapest to say first: a segmentation that
    // explains almost nothing wins by being a few percent cheaper to describe.
    // Priced together neither trick pays. The flood pays for its words, and the
    // silent segmentation pays for its silence.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
    //
    // A MARGIN, because these are ratios and two readings that cost the same
    // should fall through to the object count rather than be split by rounding.
    const double same_price = 0.001;
    const double spelling_out = facts_cells > 0 ? 1.0 / facts_cells : 0.0;
    // MADE IS HALF SAID, and that is why it is priced at half. A cell nothing
    // over there is the object of was made where it stands, and that IS the
    // answer for the half of a puzzle that is drawn rather than carried -- so
    // charging it as a failure would beat the honest readings with the
    // dishonest ones. But knowing a cell was not carried is not knowing why it
    // is there, so it is not free either.
    //
    // FREE, IT UNDID THE PRICING ABOVE IT. 00dbd492 went back to electing a
    // reading at 108% that uses 8 objects and makes 120 cells, over one at 111%
    // that uses 18 and leaves nothing -- the same trade, on the same puzzle,
    // that pricing left over was written to stop. It also cost that puzzle its
    // answer: the thin reading sees no recolor, so nothing asks the detector
    // that had already learned the map, and the search spent 4.3 seconds
    // finding a word inference was holding.
    // At half, both readings behave. The drawn half is not called a failure,
    // and no segmentation can buy a cheaper description by drawing more out of thin air.
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

    // KEPT IN THE ORDER THE TABLE IS DRAWN, one row per segmentation and one column per
    // freedom, so a reading's place IS its cell. Score order put the winner
    // first and everything after it in an order nothing on screen shows, so
    // stepping the pane hopped about a table sitting right there. Which one won
    // is a fact of its own now, said by `chosen`. Nothing is read again: the
    // winner's pairs are the ones the election already had.
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

}  // namespace arc
