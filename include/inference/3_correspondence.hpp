// 3_correspondence.hpp — where each piece went, and how to cut the grid up.
// Design story in notes/1 - Overall Design/Inference Master.md, sections 3d-3e.
//
// The whole channel: the facts, the machinery that finds them — pose a glyph,
// find it somewhere, cut a grid into pieces, then fit the matches together over
// the other grid — and at the bottom the detector, pick_split, that reads it
// all off the train pairs. infer() calls that one name and nothing else here.
// Shaped like 2_background.hpp, its sibling.

#pragma once
#include "inference/1_readings.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// correspondence · across_input_output — the EDGE level of correspondence.
// One edge = "this object appears there on the other side", read both ways and
// at two levels (whole grid, then objects). Facts only, evidence not verdicts.
// Story in notes/1 - Overall Design/Inference Master.md, section 3d.

// The ways a grid can be cut into pieces. Every kind is a CANDIDATE, not a
// setting — pick_split judges them. Order matters: one_cell_each sits last
// because the finest reading explains everything and must lose on piece count.
// Story in notes/1 - Overall Design/Inference Master.md, section 3e.
// The BIGGEST repeating patch is gone (2026-08-12). Periods behave like
// multiples: once a grid repeats every p it repeats every multiple of p, so
// asking for the biggest is asking for a greatest common multiple, which does
// not exist. It only had an answer at all under an added rule — must divide the
// grid exactly — and what it then found was always a whole number of the
// smallest one. A coarser view of the same fact, bought with a caveat.
enum class split_kind : int8_t {
    touching_8, touching_any_color, one_per_color, one_cell_each,
    repeating_motif, repeating_step, split_count
};

// THE CUTS, EACH SAYING WHAT AN OBJECT IS. They complete the sentence "AN
// OBJECT IS ..." — which is what the reader wants to know and the only thing
// the cut decides. NO COUNT IS WRITTEN ANYWHERE HERE: one is a fact the list
// already carries, and every time the list changed the count stayed behind and
// lied. Nothing here is understood by comparing it with its neighbours either:
// a name that only means something beside the others is a name you have to
// hold the whole list in your head to read.
// ONE HOME, and one list: a short form beside a long one was the same fact in
// two cases, and a rename only ever fixed one of them. The font atlas carries
// no lowercase, so uppercase is the form that serves the log and the pane both.
// CORNERS ARE NOT MENTIONED. Every cut groups through walk_out, which is
// 8-connected and has no alternative, so saying it would offer a choice that
// does not exist. See notes/4 - To Do/Connectivity Is Two Questions.md.
inline constexpr const char* const split_names[] = {
    "A GROUP OF CELLS THAT TOUCH, ALL ONE COLOR",
    "A GROUP OF CELLS THAT TOUCH, ANY COLORS",
    "EVERY CELL OF ONE COLOR, WHEREVER IT EXISTS",
    "MADE OF ONE CELL",
    "THE SMALLEST REPEATING MOTIF",
    "A MOTIF THE SIZE OF THE SMALLER GRID" };
// The sentence every one of them finishes.
inline constexpr const char* split_names_lead = "AN OBJECT IS";
static_assert(sizeof(split_names) / sizeof(char*) ==
              (size_t)split_kind::split_count);

struct match_mode {
    bool translate = true;
    bool pose = true;
    bool recolor = true;
    // HANGING OFF THE EDGE IS A FREEDOM LIKE THE REST, not a property of the
    // cut. Granted to everything, one blob was free to sit mostly outside the
    // grid, recolored and flipped, and claim whatever few cells were left
    // showing — a claim that costs nothing to make and cannot be checked.
    // Withheld from everything, every copy a wall cut in half went unexplained.
    // Neither is true of all puzzles, which is what makes it a freedom: it goes
    // in the list and the judge decides it per puzzle, the way MOVE and TURN and
    // COLOR already are.
    bool hang_off = true;
    static constexpr int COUNT = 16;
    static constexpr match_mode from_index(int packed) {
        return { (packed & 1) != 0, (packed & 2) != 0, (packed & 4) != 0,
                 (packed & 8) != 0 };
    }
    constexpr int index() const {
        return (translate ? 1 : 0) | (pose ? 2 : 0) | (recolor ? 4 : 0) |
               (hang_off ? 8 : 0);
    }
    // THE FREEDOMS, IN THE WORDS EVERYTHING USES. Said once here so the log and
    // the screen cannot drift apart. Words and not initials: initials read as
    // noise, and R could be taken for either turning or recoloring.
    // They finish the sentence "AND IT MAY".
    static constexpr int FREEDOMS = 4;
    static constexpr const char* const freedom_names[FREEDOMS] =
        { "MOVE", "TURN", "COLOR", "HANG OFF" };
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
        return words.empty() ? "AS IS" : words;
    }
};

struct correspondence_facts {
    struct edge {
        // WHICH GRID THIS EDGE STARTS ON. A reading runs both ways round and
        // keeps the edges of both, so the side is a property of the edge and
        // never of the reading. True: `from` indexes input_objects and the
        // landing is on the output. False: the other way about.
        bool from_input = true;
        // Object the edge starts from, indexed on its OWN side
        int16_t from = -1;
        // Matched object on the other side; -1 = positional hit, no object there
        int16_t to = -1;
        // Placement on the other side (glyph top-left / the piece's box corner)
        int8_t row = 0, col = 0;
        // Which of the eight ways round it was sitting; 0 = as it started
        int8_t pose = 0;
        // False = same shape, different color
        bool exact = true;
        // False = the shape itself changed, so this is not a find at all but the
        // piece it sits on top of the most (see sit_on_top).
        bool same_shape = true;
        // Cells of the piece another piece is drawn on top of; 0 = all of it
        // shows. Those cells are painted by the thing on top, not by this edge.
        // WIDER THAN A BYTE ON PURPOSE: a piece can run to 900 cells, and
        // landing_shows reads this as "is any of me covered" — a count that
        // wrapped to exactly 0 would silently call a buried piece fully visible.
        int16_t hidden = 0;
        // Cells of a pattern window that continue beyond the displayed grid.
        // Same reason: a block hanging off the edge can leave hundreds outside.
        int16_t outside = 0;
        // The uniform target color when !exact; -1 for multi-color glyphs
        int8_t recolored_to = -1;
    };
    struct object_info {
        // Component color; -1 = the whole-grid object
        int8_t color = -1;
        // Its box: top-left corner, then how big
        int8_t left = 0, top = 0, width = 0, height = 0;
        int16_t count = 0;
        // Canonical shape hash (orientationless, positionless)
        uint32_t shape = 0;
        // How many edges touch it. That one count reads as "how many places
        // this piece went" on the input side and "how many pieces met here" on
        // the output side — same number, two questions. Filled AFTER the edges
        // are final by count_edges_leaving / count_edges_arriving, never tallied
        // as the edges are made, so it cannot drift from `edges`.
        int edges_touching = 0;
        // SOMETHING FOUND IT. An edge that claims this is that, as opposed to a
        // sits-on edge, which only says what a piece lies over and answers
        // nothing. Decided HERE, where the cells still exist: downstream there
        // are only boxes, and a box lies about a hollow shape — a ring's box
        // holds its own hole, so anything sitting in the hole looked answered
        // by whatever answered the ring.
        bool answered = false;
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
    // A cell no reading accounted for, named by the grid it sits on.
    struct left_over_cell {
        bool from_input = true;
        int8_t row = 0, col = 0;
    };
    // Why a match that was found did not survive.
    enum : int8_t { lost_to_chance, cells_already_explained, overlaps_a_kept_piece };
    // Said the same way wherever it is said. Each one finishes "it was refused
    // because", so it has to be a reason on its own and not a name for one.
    static constexpr const char* const refusal_names[] = {
        "IT WAS LIKELY BY LUCK",
        "ITS CELLS WERE ALREADY EXPLAINED",
        "IT OVERLAPS A PIECE THAT WAS KEPT" };
    struct pair_info {
        // The pieces the chosen split finds, named by the grid they sit on. Both
        // grids are cut the same way and on the same block, which is what lets
        // the two directions be read into one list of edges.
        vector<object_info> input_objects;
        vector<object_info> output_objects;
        // BOTH WAYS ROUND, IN ONE LIST. Every edge says which grid it starts on,
        // so "where did this piece go" and "where did that landing come from"
        // are the same question asked of the same set. Where the two ends agree
        // only the stronger claim is kept.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
        vector<edge> edges;
        // Matches that were found and then thrown away, with why — otherwise
        // "never matched" and "matched and lost" look the same in the log.
        vector<edge> discarded;
        vector<int8_t> discard_why;
        // How it decided: 0 the pieces fit together, 1 nothing fit so the weaker
        // greedy reading was used, 2 the search ran out of steps — which is a
        // giveup, never an answer.
        int8_t verdict = 0;
        // THE CELLS the pass leaves unexplained ON BOTH GRIDS, and how many
        // pieces it used — what picks the way of cutting up. Kept as the cells
        // and not as a tally, so what is counted and what can be shown are one
        // thing: a count nobody can point at is a claim you cannot check.
        vector<left_over_cell> left_over;
        int pieces_used = 0;
        // What the cut saves on this pair's FROM grid, as a share of saying the
        // grid plainly — see describe_cost. Free here because the pieces already
        // exist, and it owes nothing to the matching, which is the point of it.
        double describe_cost = 1.0;
        // The same three questions from each end: where a piece went, where a
        // landing came from. Nowhere/nothing, one, several.
        int deleted = 0, persisted = 0, copied = 0;
        int invented = 0, from_one = 0, from_many = 0;
        // The extremes those three hide: the most places any one piece went,
        // and the most pieces that met at any one landing. The two of them ARE
        // the class — 1-1, 1-9, 3-2 — with no naming in between.
        int most_landings = 0, most_sources = 0;
        // WHAT KIND OF CLAIM each kept forward edge makes, since a line on the
        // screen looks the same whichever it is: found as itself, found wearing
        // another color, or not found at all and only sitting on something.
        // Partly hidden and turned are how many of them were covered by
        // another piece / not upright. Dropped is what was found and thrown out.
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
    // The way of cutting up that explained most, judged over the WHOLE train
    // set. FOR NOW one split serves every pair: a transformation that cuts the
    // grid up differently from pair to pair would be a strange puzzle, and
    // holding it fixed is the stronger claim. Revisit if a puzzle needs it.
    split_kind cut_kind = split_kind::touching_8;
    // EVERY READING THERE IS, IN THE ORDER THE TABLE IS DRAWN: one row per cut,
    // one column per freedom, so a reading's place in this list IS its cell —
    // cut_kind * match_mode::COUNT + match_kind. Every cut is looked at and every
    // freedom searched, so the table has no holes and a cell never has to be
    // looked up. Do not collapse this to the winner: a cut cannot tell that it
    // read too coarsely, and only a consumer coming up empty finds out. The cut
    // ITSELF is recorded alongside the numbers, because two readings can tie on
    // every number and differ.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
    struct reading {
        // Which cut, and what a match was allowed to change
        split_kind cut_kind = split_kind::touching_8;
        match_mode match_kind;
        // How many pieces the cut FOUND, and how many it USED — the log prints
        // them as "cut into X, used Y"
        int pieces_found = 0, pieces_used = 0;
        int left_over = 0, box_cells = 0;
        // The grid-only score, AVERAGED over the pairs — it is a share of plain
        // cost, and shares add up to nothing meaningful. Owes nothing to the
        // matching, so it is had BEFORE any search — see describe_cost.
        double describe_cost = 1.0;
        // FALSE MEANS THE SECOND GLANCE KILLED IT: not one piece this cut makes
        // would fit on the other grid, so no match can exist and searching it
        // would be paying to be told so. Read off the two grids' sizes.
        bool pieces_can_fit = true;
        int exact = 0, recolored = 0, sits_on = 0;
    };
    vector<reading> readings;
    // WHICH ONE THE ELECTION CHOSE, as a place in `readings`. A fact of its own,
    // because the list is in table order and the winner can sit anywhere in it —
    // being first was the only thing that ever said so.
    int8_t chosen = 0;
    // THE PIECES AND EDGES OF EVERY READING, in `readings` order, so the work is
    // done once. Without it a consumer wanting a reading other than the chosen
    // one has to run it again — and the pane wants all of them, which meant the
    // whole election happening twice on every puzzle.
    // `pairs` is reading_pairs[chosen] said again, for the consumers that only
    // ever want the one that won.
    vector<vector<pair_info>> reading_pairs;
    bool valid = false;
};

// The "first look" primitive — "where does this OBJECT turn up in that grid,
// maybe recolored?" Not a fact itself; detectors in any direction may call it,
// either way round (an input object looked for in the output, an output object
// looked for in the input). Objects only: the glyph is the piece cut out of
// its own box, with
// TRANSPARENT in the holes, and holes match anything. The whole-grid reading
// (a color as "empty") is gone — it was a per-puzzle hypothesis nobody made.
struct object_match {
    // The glyph's top-left corner where it was found
    int row = 0, col = 0;
    // Which of the eight ways it was sitting when it matched
    int pose = 0;
    // Matched cells color-identical
    bool exact = false;
    // Cells of the glyph another piece is sitting on top of; 0 = all of it shows
    int hidden = 0;
    // Cells that fell off the edge of the grid — never shown, so no evidence
    int outside = 0;
    // Glyph color -> found color; -1 = unused
    int8_t color_map[10];
};

// One of the EIGHT ways a glyph can sit: four turns by two mirrors. Pose 0 is
// the glyph as given. Same index walk canonical_shape_hash uses, so the two
// agree on what "the same shape, turned" means.
inline grid posed_glyph(const grid& glyph, int pose) {
    const bool turned = (pose & 1) != 0;
    const int posed_width  = turned ? glyph.height : glyph.width;
    const int posed_height = turned ? glyph.width  : glyph.height;
    grid posed(posed_width, posed_height, 0, MAX_GRID_DIM);
    for (int row = 0; row < posed_height; ++row)
        for (int col = 0; col < posed_width; ++col) {
            int source_row = turned ? col : row, source_col = turned ? row : col;
            if (pose & 2) source_row = glyph.height - 1 - source_row;
            if (pose & 4) source_col = glyph.width  - 1 - source_col;
            posed.data[row][col] = glyph.data[source_row][source_col];
        }
    return posed;
}

// Every place the glyph turns up, tried every way round, plus how many ways
// there were to try — the two numbers belong together, because a shape that
// can sit eight different ways had eight times as many chances to fit.
struct object_matches {
    vector<object_match> hits;
    // Placements summed over the poses that are actually different
    int places = 0;
};

// PARTLY HIDDEN COUNTS AS THERE, and so does PARTLY OFF THE EDGE. A wall does
// not undo a thing: half a shape at the edge is that shape, put where the grid
// ran out, and refusing to look there left every copy the wall had cut in half
// with nothing that could ever explain it. What the grid does show still has to
// match, and how much fell off is kept on the edge, so a landing that is mostly
// gone can be told from one that is barely clipped.
// Background showing through is never hiding: it says the piece is not there.
// Story in notes/1 - Overall Design/Inference Master.md, section 3f.
inline object_matches find_in(const grid& glyph, const grid& where, int8_t background,
                              bool may_hang_off, int fixed_row = -1, int fixed_col = -1,
                              bool match_footprint = false, int max_poses = -1) {
    object_matches found;
    if (glyph.width <= 0 || glyph.height <= 0) return found;
    const bool fixed = fixed_row >= 0 && fixed_col >= 0;
    const int pose_limit = max_poses >= 0 ? max_poses : (fixed ? 1 : 8);

    vector<grid> tried;
    for (int pose = 0; pose < pose_limit; ++pose) {
        grid posed = posed_glyph(glyph, pose);
        // A symmetric glyph reads the same several ways round; count it once.
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
                int8_t color_map[10];
                for (int i = 0; i < 10; ++i) color_map[i] = -1;
                bool fits = true, exact = true;
                for (int row = 0; fits && row < posed.height; ++row)
                    for (int col = 0; fits && col < posed.width; ++col) {
                        int glyph_color = posed.data[row][col];
                        if (off_grid(row, col)) continue;
                        int there_color = where.data[at_row + row][at_col + col];
                        if (glyph_color == TRANSPARENT) {
                            if (match_footprint && there_color != background) {
                                fits = false; break;
                            }
                            continue;
                        }
                        if (there_color == TRANSPARENT) { fits = false; break; }
                        if (match_footprint && there_color == background) { fits = false; break; }
                        if (glyph_color < 0 || glyph_color > 9 ||
                            there_color < 0 || there_color > 9) { fits = false; break; }
                        if (color_map[glyph_color] == -1)
                            color_map[glyph_color] = (int8_t)there_color;
                        if (color_map[glyph_color] != there_color) { fits = false; break; }
                        if (glyph_color != there_color) exact = false;
                    }
                // The weaker question, asked only where the whole glyph failed:
                // itself, with the missing cells hidden behind other cells.
                int hidden = 0;
                if (!fits && !match_footprint) {
                    for (int i = 0; i < 10; ++i) color_map[i] = (int8_t)i;
                    fits = true; exact = true;
                    for (int row = 0; fits && row < posed.height; ++row)
                        for (int col = 0; fits && col < posed.width; ++col) {
                            int glyph_color = posed.data[row][col];
                            if (glyph_color == TRANSPARENT || off_grid(row, col)) continue;
                            int there_color = where.data[at_row + row][at_col + col];
                            if (there_color == TRANSPARENT || there_color == background) {
                                fits = false; break;
                            }
                            hidden += (glyph_color != there_color);
                        }
                }
                if (!fits) continue;
                object_match hit;
                hit.row = at_row; hit.col = at_col; hit.pose = pose;
                hit.exact = exact; hit.hidden = hidden;
                hit.outside = lit - shown;
                for (int i = 0; i < 10; ++i) hit.color_map[i] = color_map[i];
                found.hits.push_back(hit);
            }
    }
    return found;
}

// Segment one grid for correspondence the given way, background excluded (background
// < 0 means "unknown" -> nothing excluded). Emits object_info with canonical
// shape hashes and, when asked, the glyphs (each piece cut out of its own
// box, -1 = empty).
inline void cut_into_pieces(const grid& one_grid, int8_t background, split_kind cut_kind,
                                vector<correspondence_facts::object_info>& pieces,
                                vector<grid>* glyphs,
                                int step_width = 0, int step_height = 0) {
    // As many pieces as a piece index can name — edges carry it in one signed
    // byte. The finest cut is one piece per cell, so a small grid must be able
    // to come apart completely; past this a reading simply cannot be written
    // down, and it sinks on its own by leaving the rest unexplained.
    // AS MANY PIECES AS THERE ARE CELLS. Every cell its own piece is the most a
    // grid can ever be cut into, so that is the only honest ceiling; any
    // number below it stops a cut part way and hands back a corner of the
    // grid as though it were the whole of it.
    static constexpr int MAX_PIECES = MAX_GRID_DIM * MAX_GRID_DIM;

    // Every cell its own piece. Nothing is joined to anything, so this is the
    // one reading that never merges two things that behave differently.
    if (cut_kind == split_kind::one_cell_each) {
        for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
                int color = one_grid.data[row][col];
                if (color < 0 || color == background) continue;
                if ((int)pieces.size() >= MAX_PIECES) return;
                correspondence_facts::object_info piece;
                piece.color = (int8_t)color;
                piece.left = (int8_t)col; piece.top = (int8_t)row;
                piece.width = 1; piece.height = 1; piece.count = 1;
                bool one_cell = true;
                piece.shape = canonical_shape_hash(1, 1, &one_cell);
                pieces.push_back(piece);
                if (glyphs) {
                    grid glyph(1, 1, 0, MAX_GRID_DIM);
                    glyph.data[0][0] = (int8_t)color;
                    glyphs->push_back(glyph);
                }
            }
        return;
    }

    // Cut on a fixed step: equal blocks, one piece each, cells of any color kept
    // together. Blocks with no colored cells at all are not pieces.
    if (cut_kind == split_kind::repeating_step ||
        cut_kind == split_kind::repeating_motif) {
        if (step_width <= 0 || step_height <= 0) return;
        bool occupied[MAX_GRID_DIM * MAX_GRID_DIM];
        for (int block_top = 0; block_top < one_grid.height; block_top += step_height)
        for (int block_left = 0; block_left < one_grid.width; block_left += step_width) {
            if ((int)pieces.size() >= MAX_PIECES) return;
            // A step that does not divide the grid leaves a short block at the
            // edge; it is still a block, and dropping it would lose real cells.
            const int block_width  = min(step_width,  one_grid.width  - block_left);
            const int block_height = min(step_height, one_grid.height - block_top);
            grid glyph(block_width, block_height, 0, MAX_GRID_DIM);
            int lit = 0;
            for (int row = 0; row < block_height; ++row)
                for (int col = 0; col < block_width; ++col) {
                    int color = one_grid.data[block_top + row][block_left + col];
                    bool is_colored = (color >= 0 && color != background);
                    occupied[row * block_width + col] = is_colored;
                    glyph.data[row][col] = is_colored ? (int8_t)color : (int8_t)TRANSPARENT;
                    lit += is_colored;
                }
            if (!lit) continue;
            correspondence_facts::object_info piece;
            piece.color = -1;
            piece.left = (int8_t)block_left; piece.top = (int8_t)block_top;
            piece.width = (int8_t)block_width; piece.height = (int8_t)block_height;
            piece.count = (int16_t)lit;
            piece.shape = canonical_shape_hash(block_width, block_height, occupied);
            pieces.push_back(piece);
            if (glyphs) glyphs->push_back(glyph);
        }
        // A CUT THAT DOES NOT CUT IS NOT A READING — one piece means there is no
        // motif, and it would win the election for free. The step cut is exempt.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
        if (cut_kind != split_kind::repeating_step && pieces.size() < 2) {
            pieces.clear();
            if (glyphs) glyphs->clear();
        }
        return;
    }
    // Whether a piece has to be one color is a question of its OWN, asked apart
    // from what counts as joined: the one walk answers both.
    const bool same_color = (cut_kind != split_kind::touching_any_color);
    bool visited[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    int  queue_row[MAX_GRID_DIM * MAX_GRID_DIM], queue_col[MAX_GRID_DIM * MAX_GRID_DIM];
    bool occupied[MAX_GRID_DIM * MAX_GRID_DIM];

    for (int start_row = 0; start_row < one_grid.height && start_row < MAX_GRID_DIM; ++start_row)
    for (int start_col = 0; start_col < one_grid.width && start_col < MAX_GRID_DIM; ++start_col) {
        if (visited[start_row][start_col]) continue;
        int color = one_grid.data[start_row][start_col];
        if (color < 0 || color == background) {
            visited[start_row][start_col] = true;
            continue;
        }
        int seeded = 1;
        queue_row[0] = start_row; queue_col[0] = start_col;
        visited[start_row][start_col] = true;
        int left = start_col, top = start_row, right = start_col, bottom = start_row;
        // One piece per color takes every cell of the color at once, wherever
        // it is; the touching kinds walk out from this cell only.
        const bool whole_color = (cut_kind == split_kind::one_per_color);
        if (whole_color) {
            for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
                for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
                    if (visited[row][col] || one_grid.data[row][col] != color) continue;
                    visited[row][col] = true;
                    queue_row[seeded] = row; queue_col[seeded] = col; seeded++;
                }
        }
        const int size = walk_out(one_grid.height, one_grid.width,
                                  queue_row, queue_col, seeded,
            [&](int next_row, int next_col) {
                if (whole_color || visited[next_row][next_col]) return false;
                const int next_color = one_grid.data[next_row][next_col];
                if (next_color < 0 || next_color == background) return false;
                if (same_color && next_color != color) return false;
                visited[next_row][next_col] = true; return true;
            },
            [&](int row, int col) {
                left = min(left, col);   top = min(top, row);
                right = max(right, col); bottom = max(bottom, row);
            });
        if ((int)pieces.size() >= MAX_PIECES) return;
        // A piece wears one color only if every cell of it does; -1 says several.
        bool one_color = true;
        for (int i = 1; i < size && one_color; ++i)
            one_color = (one_grid.data[queue_row[i]][queue_col[i]] == color);
        correspondence_facts::object_info piece;
        piece.color = one_color ? (int8_t)color : (int8_t)-1;
        piece.left = (int8_t)left; piece.top = (int8_t)top;
        piece.width = (int8_t)(right - left + 1); piece.height = (int8_t)(bottom - top + 1);
        piece.count = (int16_t)size;
        for (int i = 0; i < piece.width * piece.height; ++i) occupied[i] = false;
        for (int i = 0; i < size; ++i)
            occupied[(queue_row[i] - top) * piece.width + (queue_col[i] - left)] = true;
        piece.shape = canonical_shape_hash(piece.width, piece.height, occupied);
        pieces.push_back(piece);
        if (glyphs) {
            // Each cell keeps its own color, so a piece of several reads as it
            // looks. For a one-color piece that is the color it already had.
            grid glyph(piece.width, piece.height, 0, MAX_GRID_DIM);
            for (int row = 0; row < piece.height; ++row)
                for (int col = 0; col < piece.width; ++col) glyph.data[row][col] = TRANSPARENT;
            for (int i = 0; i < size; ++i)
                glyph.data[queue_row[i] - top][queue_col[i] - left] =
                    one_grid.data[queue_row[i]][queue_col[i]];
            glyphs->push_back(glyph);
        }
    }
}

// How far one object is from another, in cells that do not agree. Laid over
// each other at their top-left corners, and a cell one has where the other has
// nothing counts as a disagreement — so a block cut short at the edge is far
// from every whole one and can never speak for the set.
inline int glyphs_apart(const grid& left, const grid& right) {
    const int width = max(left.width, right.width);
    const int height = max(left.height, right.height);
    int apart = 0;
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const int here = (row < left.height && col < left.width)
                           ? left.data[row][col] : TRANSPARENT;
            const int there = (row < right.height && col < right.width)
                            ? right.data[row][col] : TRANSPARENT;
            apart += (here != there);
        }
    return apart;
}

// COPIES OF ONE THING, AND WHICH OF THEM SPEAKS FOR THE SET. Objects sharing a
// canonical shape are copies of each other; when the cut ran on a VERIFIED
// period the whole grid is one set, stub at the edge included, because cutting
// on a period is what makes them copies and the stub is the motif cut short.
// The set's exemplar is its MEDOID — see object_info::medoid.
inline void find_copies(vector<correspondence_facts::object_info>& objects,
                        const vector<grid>& glyphs, bool one_set_for_the_grid) {
    vector<uint32_t> sets;
    for (size_t i = 0; i < objects.size(); ++i) {
        const uint32_t key = one_set_for_the_grid ? 0u : objects[i].shape;
        const bool has_company = one_set_for_the_grid ||
            ranges::any_of(objects, [&](const auto& other) {
                return &other != &objects[i] && other.shape == objects[i].shape &&
                       other.count == objects[i].count; });
        if (!has_company) continue;
        auto at = ranges::find(sets, key);
        if (at == sets.end()) { sets.push_back(key); at = sets.end() - 1; }
        objects[i].motif = (int16_t)(at - sets.begin());
    }
    for (int16_t set = 0; set < (int16_t)sets.size(); ++set) {
        int best = -1, least_apart = 0;
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].motif != set) continue;
            int apart = 0;
            for (size_t j = 0; j < objects.size(); ++j)
                if (j != i && objects[j].motif == set)
                    apart += glyphs_apart(glyphs[i], glyphs[j]);
            if (best < 0 || apart < least_apart) { best = (int)i; least_apart = apart; }
        }
        if (best >= 0) objects[best].medoid = true;
    }
}

// A PATCH OF NOTHING: where it is, how big, and whether the grid's own colored
// cells close it in on every side. Not an object_info — a patch has no color,
// no shape and no copies, and lending it that struct said it had.
struct blank_patch {
    int8_t left = 0, top = 0, width = 0, height = 0;
    int16_t cells = 0;
    bool enclosed = false;
};

// THE PATCHES OF NOTHING IN A GRID. Background cells joined side to side —
// ADJACENCY IS 4, see core_types.hpp — each patch given a number, and every
// cell told which patch it belongs to. A patch that reaches the border of the
// grid is open to the outside; one that never does is closed in, and that is a
// hole. Story in notes/1 - Overall Design/Inference Master.md, 3d.
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
                    const int next_row = at_row + NEIGHBOUR_ROW_4[side];
                    const int next_col = at_col + NEIGHBOUR_COL_4[side];
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

// Cut it up, then put the pieces in the order a person reads them: BIGGEST
// FIRST, then down the grid. The glyphs ride along — they are the pieces.
// Story in notes/1 - Overall Design/Inference Master.md, section 3d.
inline void segment_for_correspondence(const grid& one_grid, int8_t background, split_kind cut_kind,
                                vector<correspondence_facts::object_info>& pieces,
                                vector<grid>* glyphs,
                                int step_width = 0, int step_height = 0) {
    cut_into_pieces(one_grid, background, cut_kind, pieces, glyphs, step_width, step_height);
    vector<int> reading_order(pieces.size());
    iota(reading_order.begin(), reading_order.end(), 0);
    stable_sort(reading_order.begin(), reading_order.end(), [&](int left, int right) {
        if (pieces[left].count != pieces[right].count)
            return pieces[left].count > pieces[right].count;
        if (pieces[left].top != pieces[right].top)
            return pieces[left].top < pieces[right].top;
        return pieces[left].left < pieces[right].left;
    });
    vector<correspondence_facts::object_info> sorted;
    sorted.reserve(pieces.size());
    for (int i : reading_order) sorted.push_back(pieces[i]);
    pieces.swap(sorted);
    if (glyphs) {
        vector<grid> moved;
        moved.reserve(glyphs->size());
        for (int i : reading_order) moved.push_back((*glyphs)[i]);
        glyphs->swap(moved);
    }
}

// WHAT THE CUT SAVES YOU. Saying the grid under a cut means spelling each
// different shape out once, then saying where each copy goes. One piece per cell
// spells nothing out and then pays a position for every cell; the whole grid
// as one piece pays for every cell at once. The cut that tells you the most is
// the cheap one in between.
//
// COUNTED IN BITS, so neither price is a matter of taste: a cell has to name a
// color out of the palette, a position has to name a square out of the grid.
//
// THEN DIVIDED BY WHAT THE GRID COSTS SAID PLAINLY, cell by cell. A bit total on
// its own answers nothing, and totals off two grids of different size are not
// the same question — a 2x2 is cheap to describe however you cut it, so it beats
// a 6x6 on size alone and the reading gets taken from the side with no structure
// on it. As a share of plain cost the size cancels: under 1 the cut earned its
// place, over 1 it made things worse, and a small grid can no longer win by
// being small.
//
// Read off the pieces of ONE grid — no matching, no other side, no search. That
// is the whole point: this can be had BEFORE anything is searched.
// Story in notes/1 - Overall Design/Inference Master.md, section 3e.
inline double describe_cost(const grid& one_grid, int8_t background,
                            const vector<correspondence_facts::object_info>& pieces) {
    const double cell_bits = log2(10.0);
    const double place_bits =
        log2((double)max(1, one_grid.width * one_grid.height));
    int colored = 0;
    for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
        for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
            const int color = one_grid.data[row][col];
            colored += (color >= 0 && color != background);
        }
    // Nothing on the grid, so there is nothing to save and nothing to lose.
    const double plain = colored * cell_bits;
    if (plain <= 0.0) return 1.0;
    // A cut that finds nothing has said nothing: you still spell the grid out
    // cell by cell, and you paid a position for the privilege. Never the cheap
    // answer, because it is over 1 by construction.
    if (pieces.empty()) return (plain + place_bits) / plain;
    // A position per piece, plus the cells of each shape the first time it is
    // met. Same shape turned round is the same shape — the hash is canonical.
    double cost = pieces.size() * place_bits;
    for (size_t i = 0; i < pieces.size(); ++i) {
        bool said_before = false;
        for (size_t j = 0; j < i && !said_before; ++j)
            said_before = (pieces[j].shape == pieces[i].shape);
        if (!said_before) cost += pieces[i].count * cell_bits;
    }
    return cost / plain;
}

// What share of a grid's cells each color holds — the odds a cell picked at
// random wears that color.
inline void color_shares(const grid& one_grid, double share[10]) {
    int counts[10], total = 0;
    count_colors(one_grid, counts);
    for (int i = 0; i < 10; ++i) total += counts[i];
    for (int i = 0; i < 10; ++i) share[i] = total ? (double)counts[i] / total : 0.0;
}

// Would this match have happened anyway? `odds` is the chance one placement
// lands on exactly these colors; `places` is how many placements the footprint
// has. Their product is how many such matches to EXPECT from a grid of this
// size and coloring — under one means the match is telling us something.
// The FALLBACK judge only: see set_up_the_fitting / fit_the_pieces_together for why one match at
// a time is the weaker question.
inline bool beats_chance(int places, double odds) {
    return places > 0 && places * odds < 1.0;
}

// The footprint a match lays down. A match that names an object there wears
// that object's cells; one that only names a place wears the glyph's, as it
// was sitting when it matched.
inline grid edge_footprint(const correspondence_facts::edge& edge,
                           const vector<grid>& from_glyphs,
                           const vector<grid>& into_glyphs) {
    if (edge.to >= 0 && (size_t)edge.to < into_glyphs.size()) return into_glyphs[edge.to];
    return posed_glyph(from_glyphs[edge.from], edge.pose);
}

// A cell of a landing that the grid actually shows: on it, and not behind
// another piece. Off the edge or covered, the piece did not put it there.
inline bool landing_shows(const correspondence_facts::edge& edge, const grid& glyph,
                          const grid& into, int row, int col) {
    if (glyph.data[row][col] == TRANSPARENT) return false;
    const int into_row = edge.row + row, into_col = edge.col + col;
    if (into_row < 0 || into_row >= into.height ||
        into_col < 0 || into_col >= into.width) return false;
    return !edge.hidden || into.data[into_row][into_col] == glyph.data[row][col];
}

inline void mark_painted(const vector<correspondence_facts::edge>& edges,
                         const vector<grid>& from_glyphs,
                         const vector<grid>& into_glyphs,
                         const grid& into,
                         bool covered[MAX_GRID_DIM][MAX_GRID_DIM],
                         bool exact_only = true,
                         bool same_shape_only = false) {
    for (const auto& edge : edges) {
        if (exact_only && !edge.exact) continue;
        if (same_shape_only && !edge.same_shape) continue;
        const grid footprint = edge_footprint(edge, from_glyphs, into_glyphs);
        for (int row = 0; row < footprint.height; ++row)
            for (int col = 0; col < footprint.width; ++col)
                if (landing_shows(edge, footprint, into, row, col))
                    covered[edge.row + row][edge.col + col] = true;
    }
}

// How a pass decided, so a giveup can never be mistaken for an answer.
enum class fit_verdict : int8_t { fitted, no_fit, gave_up };

// BLOCKS THAT REPEAT, ASKED FROM BOTH ENDS. One block standing for every place
// it turns up is only half the story: it is the right half when the from side
// holds one block and the into side many, and it collapses to a single edge the
// moment the counts are the other way round. So the two questions are asked in
// turn, and neither end is the privileged one.
//   Where did each landing come from — every into block takes the from block
//   that sits at its own place if that one matches, else the first that does.
//   Where did each piece go — every from block still unanswered takes the first
//   into block it matches.
// One block to one block for equal counts, one to many or many to one when they
// differ, and never more edges than the two sides have blocks between them.
inline fit_verdict collect_motif_edges(
        const vector<correspondence_facts::object_info>& from_objects,
        const vector<grid>& from_glyphs,
        const vector<correspondence_facts::object_info>& into_objects,
        const grid& into, int8_t background, int step_w, int step_h,
        vector<correspondence_facts::edge>& edges, match_mode match = {}) {
    if (step_w <= 0 || step_h <= 0) return fit_verdict::no_fit;
    const int poses = match.pose ? 8 : 1;

    // Does this from block sit at that place on the other grid? The answer is
    // the same question both sweeps ask, so it is written once.
    auto sits_at = [&](int from_index, const correspondence_facts::object_info& target,
                       object_match& found) {
        const object_matches matches =
            find_in(from_glyphs[from_index], into, background, true,
                    target.top, target.left, true, poses);
        // ITSELF BEATS ITSELF-IN-ANOTHER-COLOR. The hits come back in pose order,
        // so the first one can be a recolor of an early pose while a later pose
        // sits there exactly — and taking the first would report the weaker claim
        // about the same two blocks.
        const object_match* best = nullptr;
        for (const object_match& hit : matches.hits) {
            if (!match.recolor && !hit.exact) continue;
            if (!best || (hit.exact && !best->exact)) best = &hit;
            if (best->exact) break;
        }
        if (!best) return false;
        found = *best;
        return true;
    };
    auto draw = [&](int from_index, int target_index, const object_match& hit) {
        const auto& target = into_objects[target_index];
        correspondence_facts::edge edge;
        edge.from = (int16_t)from_index;
        edge.to = (int16_t)target_index;
        edge.row = (int8_t)target.top;
        edge.col = (int8_t)target.left;
        edge.pose = (int8_t)hit.pose;
        edge.exact = hit.exact;
        edge.outside = (int16_t)hit.outside;
        edge.recolored_to =
            (!hit.exact && from_objects[from_index].color >= 0)
                ? hit.color_map[from_objects[from_index].color] : (int8_t)-1;
        edges.push_back(edge);
    };

    vector<bool> answered(from_objects.size(), false);
    // WHERE DID EACH LANDING COME FROM. Its own place first, so a grid whose
    // two sides hold the same blocks reads as each staying put rather than as
    // everything matching everything.
    for (int target_index = 0; target_index < (int)into_objects.size(); ++target_index) {
        const auto& target = into_objects[target_index];
        int source = -1;
        object_match hit;
        for (int i = 0; i < (int)from_objects.size(); ++i)
            if (from_objects[i].top == target.top && from_objects[i].left == target.left &&
                sits_at(i, target, hit)) { source = i; break; }
        for (int i = 0; source < 0 && i < (int)from_objects.size(); ++i)
            if (sits_at(i, target, hit)) source = i;
        if (source < 0) continue;
        draw(source, target_index, hit);
        answered[source] = true;
    }
    // WHERE DID EACH PIECE GO. Only the ones no landing claimed, so a from side
    // bigger than the into side is not silently dropped.
    for (int i = 0; i < (int)from_objects.size(); ++i) {
        if (answered[i]) continue;
        object_match hit;
        for (int target_index = 0; target_index < (int)into_objects.size(); ++target_index)
            if (sits_at(i, into_objects[target_index], hit)) {
                draw(i, target_index, hit);
                answered[i] = true;
                break;
            }
    }
    return edges.empty() ? fit_verdict::no_fit : fit_verdict::fitted;
}

inline int lit_cells(const grid& glyph) {
    int lit = 0;
    for (int row = 0; row < glyph.height; ++row)
        for (int col = 0; col < glyph.width; ++col)
            lit += (glyph.data[row][col] != TRANSPARENT);
    return lit;
}

// A match that was found and then thrown away, with the reason.
inline void throw_away(const vector<correspondence_facts::edge>& found, size_t found_index,
                       int8_t why, vector<correspondence_facts::edge>* discarded,
                       vector<int8_t>* discard_why) {
    if (!discarded || !discard_why) return;
    discarded->push_back(found[found_index]);
    discard_why->push_back(why);
}

// One candidate piece: the match it came from, the cells it paints, and the
// cells of its own that something else paints — which is what puts it BEHIND.
struct candidate {
    size_t match_index = 0;
    vector<int> cells, behind;
};

// The pieces, and which cells each of them could paint.
struct fitting {
    // Cells of the target grid that need painting
    vector<pair<int, int>> cells;
    // One per match that lands wholly inside the grid
    vector<candidate> pieces;
    // Per cell: the pieces that can paint it, best first
    vector<vector<int>> options;
    // Cells an EXACT piece could account for — only these drive the search
    vector<bool> required;
};

// DID THE MATCH LAND ON ANYTHING AT ALL. Empty space has no shape, so every
// shape fits it in the background's own color — a match there can never fail,
// and a claim that cannot fail is not evidence. It reads as "this was already
// here, in nothing", when what happened is that it was made. Being made is
// already recorded, as came-from-nothing and as created cells.
// ONE CELL OF SOMETHING IS ENOUGH, and asking for a WHOLE object was too much:
// several pieces making one picture is a fragment relationship too, so that
// rule could not tell an assembly from a coincidence and refused both. Whether
// fragments add up to the thing is the fitting's question, and it already
// asks it — see set_up_the_fitting.
inline bool landed_on_something(const correspondence_facts::edge& edge,
                                const grid& glyph, const grid& into,
                                int8_t background) {
    const grid posed = posed_glyph(glyph, edge.pose);
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

// STAGE 1 — every match that could stand on its own, unjudged as a set. Two
// sources: the glyph found at a place, and an object there with the same
// canonical shape. Story in notes/1 - Overall Design/Inference Master.md, 3d.
inline vector<correspondence_facts::edge> gather_matches(
        const vector<correspondence_facts::object_info>& from_objects,
        const vector<grid>& from_glyphs,
        const vector<correspondence_facts::object_info>& into_objects,
        const grid& into, int8_t background, bool may_hang_off,
        vector<correspondence_facts::edge>* discarded,
        vector<int8_t>* discard_why, match_mode match) {
    double share[10];
    color_shares(into, share);

    vector<correspondence_facts::edge> found;

    for (size_t from_index = 0; from_index < from_glyphs.size(); ++from_index) {
        const auto& piece = from_objects[from_index];
        const grid& glyph = from_glyphs[from_index];

        // Positional: own place first, then anywhere only if it was not there.
        auto keep_positional = [&](const object_matches& found_here, bool stationary) {
        for (const object_match& hit : found_here.hits) {
            if (!match.recolor && !hit.exact) continue;
            if (!stationary && hit.pose == 0 &&
                hit.row == piece.top && hit.col == piece.left)
                continue;
            // Odds of this landing: every cell the grid SHOWED had to find the
            // color it actually found there. A hidden cell found nothing —
            // something is in front of it — and a cell off the edge was never
            // shown at all, so neither is evidence and neither counts. What is
            // left is what the placement really put on the table.
            const grid posed = posed_glyph(glyph, hit.pose);
            double odds = 1.0;
            for (int row = 0; row < posed.height; ++row)
                for (int col = 0; col < posed.width; ++col) {
                    const int glyph_color = posed.data[row][col];
                    if (glyph_color == TRANSPARENT) continue;
                    const int into_row = hit.row + row, into_col = hit.col + col;
                    if (into_row < 0 || into_row >= into.height ||
                        into_col < 0 || into_col >= into.width) continue;
                    const int into_color = into.data[into_row][into_col];
                    if (hit.hidden && into_color != glyph_color) continue;
                    odds *= share[hit.color_map[glyph_color]];
                }
            correspondence_facts::edge edge;
            edge.from = (int16_t)from_index;
            edge.row = (int8_t)hit.row; edge.col = (int8_t)hit.col;
            edge.pose = (int8_t)hit.pose;
            edge.exact = hit.exact;
            edge.hidden = (int16_t)hit.hidden;
            edge.outside = (int16_t)hit.outside;
            edge.recolored_to =
                (!hit.exact && piece.color >= 0) ? hit.color_map[piece.color] : (int8_t)-1;
            const bool stands_alone = beats_chance(found_here.places, odds);
            // ANYTHING LESS THAN THE WHOLE PIECE AS ITSELF MUST EARN ITS PLACE
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

        // STAYING PUT AND BEING COPIED ARE NOT RIVALS. Finding a piece where it
        // already was used to end the hunt, so a shape that stayed and was also
        // stamped out elsewhere could never show its copies — the one thing this
        // whole layer is for. Both questions are asked now, and the judges that
        // already exist decide: a lucky landing dies to beats_chance, and one
        // that explains nothing new dies in the fitting.
        const int poses = match.pose ? 8 : 1;
        keep_positional(
            find_in(glyph, into, background, false, piece.top, piece.left, false, poses), true);
        if (match.translate)
            keep_positional(
                find_in(glyph, into, background, may_hang_off, -1, -1, false, poses), false);

        // Source 2 — shape: objects there with the same canonical hash (count
        // equality guards hash collisions). Skip placements source 1 found.
        if (!match.translate) continue;
        for (size_t into_index = 0; into_index < into_objects.size(); ++into_index) {
            const auto& landing = into_objects[into_index];
            if (landing.shape != piece.shape || landing.count != piece.count) continue;
            if (landing.top == piece.top && landing.left == piece.left) continue;
            bool seen = false;
            for (const auto& edge : found)
                if (edge.from == (int16_t)from_index &&
                    edge.row == landing.top && edge.col == landing.left) { seen = true; break; }
            if (seen) continue;
            // Same odds, read off the object that was matched: a single-color
            // body is `count` cells all wearing its color; the whole-grid
            // object is the grid itself.
            double odds = 1.0;
            if (landing.color >= 0)
                for (int i = 0; i < landing.count; ++i) odds *= share[landing.color];
            else
                for (int row = 0; row < into.height; ++row)
                    for (int col = 0; col < into.width; ++col) {
                        int color = into.data[row][col];
                        if (color >= 0 && color < 10) odds *= share[color];
                    }
            correspondence_facts::edge edge;
            edge.from = (int16_t)from_index;
            edge.to = (int16_t)into_index;
            edge.row = landing.top; edge.col = landing.left;
            edge.exact = (landing.color == piece.color);
            if (!match.recolor && !edge.exact) continue;
            edge.recolored_to = (!edge.exact && piece.color >= 0) ? landing.color : (int8_t)-1;
            const bool stands_alone = beats_chance(
                (into.width - landing.width + 1) * (into.height - landing.height + 1), odds);
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
    vector<correspondence_facts::edge> on_something;
    for (size_t i = 0; i < found.size(); ++i) {
        if (landed_on_something(found[i], from_glyphs[found[i].from], into, background)) {
            on_something.push_back(found[i]);
            continue;
        }
        throw_away(found, i, correspondence_facts::lost_to_chance,
                   discarded, discard_why);
    }
    return on_something;
}

// STAGE 2 — lay the matches out as a fitting problem: index the cells that need
// painting, turn each match into a piece, and list per cell which pieces could
// paint it. Story in notes/1 - Overall Design/Inference Master.md, section 3d.
inline fitting set_up_the_fitting(const vector<correspondence_facts::edge>& found,
                         const vector<grid>& from_glyphs,
                         const vector<grid>& into_glyphs,
                         const grid& into, int8_t background) {
    fitting laid_out;
    // EVERY CELL OF THE GRID IS A CELL, background included. Left out, a piece
    // landing on background was painting on nothing and got thrown out of the
    // fitting — so anything that was not in one grid and appears in the other,
    // out of empty space, could never be answered.
    // ADDING CELLS FORCES NOTHING. Only cells an EXACT piece reaches are
    // REQUIRED, and an exact piece can never land on background: a piece's own
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

    // Every match is a candidate piece, recolored ones included: a recolor
    // keeps the SHAPE, and shape is what has to fit. Fitting is the strong
    // claim here, so it does the filtering that per-match judging could not.
    for (size_t found_index = 0; found_index < found.size(); ++found_index) {
        const grid glyph = edge_footprint(found[found_index], from_glyphs, into_glyphs);
        candidate candidate{ found_index, {}, {} };
        bool inside = true;
        for (int row = 0; row < glyph.height && inside; ++row)
            for (int col = 0; col < glyph.width; ++col) {
                if (glyph.data[row][col] == TRANSPARENT) continue;
                const int into_row = found[found_index].row + row;
                const int into_col = found[found_index].col + col;
                // Off the edge is not a cell of this grid at all: it is
                // neither painted nor hidden, it was simply never shown.
                if (into_row < 0 || into_row >= into.height ||
                    into_col < 0 || into_col >= into.width) continue;
                const int id = cell_id[into_row][into_col];
                // The cells it is hidden at are its own too — kept apart,
                // because they are what puts it BEHIND whoever paints them.
                if (!landing_shows(found[found_index], glyph, into, row, col)) {
                    if (id >= 0) candidate.behind.push_back(id);
                    continue;
                }
                if (id < 0) { inside = false; break; }   // paints outside the grid
                candidate.cells.push_back(id);
            }
        if (inside && !candidate.cells.empty()) laid_out.pieces.push_back(std::move(candidate));
    }

    laid_out.options.resize(laid_out.cells.size());
    for (int piece_index = 0; piece_index < (int)laid_out.pieces.size(); ++piece_index)
        for (int id : laid_out.pieces[piece_index].cells)
            laid_out.options[id].push_back(piece_index);
    // SAME CELLS EXPLAINED, FEWER WORDS SPENT. The search takes the first fitting
    // it reaches, so this order IS the preference, and a 1x1 piece can tile
    // anything. Two tilings can cover exactly the same cells and the fitting is
    // happy with either — what separates them is how much had to be said. One
    // piece covering eight cells is one thing said; four pieces covering the
    // same eight is four, and four claims that happen to agree are not better
    // evidence than the one claim they agree with.
    // BIGGEST FIRST, THEN AS ITSELF. Asked the other way round — exact before
    // big — a three-cell stick that matched exactly beat an eight-cell shape
    // that matched in another color, so the ring in 045e512c was explained four
    // times over by the little stick lying inside its edges instead of once by
    // the ring it is a copy of. Being the whole thing matters more than being
    // the right color; the color question only decides a tie between equals.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
    for (auto& options : laid_out.options)
        stable_sort(options.begin(), options.end(), [&](int left, int right) {
            const size_t left_cells = laid_out.pieces[left].cells.size();
            const size_t right_cells = laid_out.pieces[right].cells.size();
            if (left_cells != right_cells) return left_cells > right_cells;
            return found[laid_out.pieces[left].match_index].exact &&
                   !found[laid_out.pieces[right].match_index].exact;
        });

    // SOME CELLS ARE SIMPLY NEW. Only cells an exact piece could account for are
    // REQUIRED; demanding the rest be explained makes the search invent a source.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
    laid_out.required.assign(laid_out.cells.size(), false);
    for (const auto& candidate : laid_out.pieces)
        if (found[candidate.match_index].exact)
            for (int id : candidate.cells) laid_out.required[id] = true;
    return laid_out;
}

// STAGE 3 — PIECES THAT FIT TOGETHER: every cell that needs painting is painted
// exactly once, no overlaps, nothing outside. Not one-at-a-time and not
// greediest-first — both pick locally and ties poison them. That is a search,
// so it searches: every cell, fewest options first, backtracking. Fills
// `chosen` on success. Story in notes/1 - Overall Design/Inference Master.md, 3d.
inline fit_verdict fit_the_pieces_together(const fitting& laid_out,
                                const vector<correspondence_facts::edge>& found,
                                vector<int>& chosen) {
    if (!ranges::any_of(laid_out.required, [](bool needed) { return needed; }))
        return fit_verdict::no_fit;

    vector<int> painted_by(laid_out.cells.size(), -1);
    long long steps = 0;
    // WHEN THE PIECES STOP COMING TOGETHER, THEY ARE NOT GOING TO. There are
    // only this many ways to commit a piece to a cell; past that the search is
    // not making new commitments, it is re-ordering ones it already disproved.
    // Taken from the problem's own size, so a small grid is not given a
    // budget meant for a big one, and no number here was chosen by taste. Not a
    // proof — backtracking may legitimately revisit — but a stop that scales.
    // Running out is reported, never hidden.
    const long long step_budget =
        (long long)(laid_out.pieces.size() + 1) * (long long)(laid_out.cells.size() + 1);
    bool out_of_budget = false;

    // ONE OF THEM IS IN FRONT. Nothing can be both in front of and behind the
    // same piece. Checked as the piece goes down, so the search backs out of the
    // contradiction instead of returning it.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
    auto contradicts = [&](int piece_index) {
        for (int behind_id : laid_out.pieces[piece_index].behind) {
            const int in_front = painted_by[behind_id];
            if (in_front < 0) continue;
            for (int its_behind_id : laid_out.pieces[in_front].behind)
                if (painted_by[its_behind_id] == piece_index) return true;
        }
        return false;
    };

    auto fits_together = [&](auto&& self) -> bool {
        if (++steps > step_budget) { out_of_budget = true; return false; }
        // The cell with the fewest pieces still able to paint it decides next;
        // zero options means this line of choices is dead. Only required cells
        // drive the search — a made cell is answered by being left alone.
        int target = -1, fewest = 0;
        for (size_t id = 0; id < laid_out.cells.size(); ++id) {
            if (painted_by[id] >= 0 || !laid_out.required[id]) continue;
            int options = 0;
            for (int piece_index : laid_out.options[id]) {
                bool free_here = true;
                for (int other : laid_out.pieces[piece_index].cells)
                    if (painted_by[other] >= 0) { free_here = false; break; }
                options += free_here;
            }
            if (options == 0) return false;
            if (target < 0 || options < fewest) { target = (int)id; fewest = options; }
        }
        if (target < 0) return true;   // every carried-over cell accounted for
        // A PIECE WITH NO GOOD LANDING SIMPLY GOES UNANSWERED. Every piece used
        // to be offered a first landing before any piece was offered a second,
        // so a piece that corresponds to nothing was handed something to be
        // wrong about — in 045e512c a three-cell stick took a whole ring,
        // because the ring it belonged to had already landed and was skipped.
        // Fairness is not a finding. The options are in preference order and
        // nothing overrides it now; a piece nothing wanted says nothing, which
        // is the truth and is what UNANSWERED and LEFT OVER are for.
        for (int piece_index : laid_out.options[target]) {
            bool free_here = true;
            for (int id : laid_out.pieces[piece_index].cells)
                if (painted_by[id] >= 0) { free_here = false; break; }
            if (!free_here) continue;
            for (int id : laid_out.pieces[piece_index].cells) painted_by[id] = piece_index;
            chosen.push_back(piece_index);
            if (!contradicts(piece_index) && self(self)) return true;
            chosen.pop_back();
            for (int id : laid_out.pieces[piece_index].cells) painted_by[id] = -1;
            if (out_of_budget) return false;
        }
        return false;
    };

    if (fits_together(fits_together)) return fit_verdict::fitted;
    return out_of_budget ? fit_verdict::gave_up : fit_verdict::no_fit;
}

// STAGE 4a — the pieces fit, so the fitting IS the reading, plus whatever it
// left room for. What is refused after that overlaps something kept, and the
// log has to be able to say so.
inline void keep_what_fitted(vector<int>& chosen, const fitting& laid_out,
                           const vector<correspondence_facts::edge>& found,
                           vector<correspondence_facts::edge>& edges,
                           vector<correspondence_facts::edge>* discarded,
                           vector<int8_t>* discard_why) {
    sort(chosen.begin(), chosen.end());
    vector<bool> kept(found.size(), false);
    vector<bool> painted(laid_out.cells.size(), false);
    for (int piece_index : chosen) {
        const size_t match_index = laid_out.pieces[piece_index].match_index;
        edges.push_back(found[match_index]);
        kept[match_index] = true;
        for (int id : laid_out.pieces[piece_index].cells) painted[id] = true;
    }
    // THE FITTING SETTLES WHAT COMPETES AND SAYS NOTHING ABOUT THE REST. It drives
    // only on cells an EXACT piece could reach, so a piece landing where nothing
    // was — a thing that appears out of empty space — is never needed and never
    // taken. Not needed is not wrong: painting only cells the fitting left alone,
    // it takes nothing from anyone and accounts for cells nothing else does.
    // STRONGEST FIRST, so when two of these want one cell the better claim has
    // it — the same order the fitting offers its own choices in.
    vector<int> leftovers;
    for (int piece_index = 0; piece_index < (int)laid_out.pieces.size(); ++piece_index)
        if (!kept[laid_out.pieces[piece_index].match_index] &&
            !laid_out.pieces[piece_index].cells.empty())
            leftovers.push_back(piece_index);
    stable_sort(leftovers.begin(), leftovers.end(), [&](int left, int right) {
        const bool left_exact = found[laid_out.pieces[left].match_index].exact;
        const bool right_exact = found[laid_out.pieces[right].match_index].exact;
        if (left_exact != right_exact) return left_exact;
        return laid_out.pieces[left].cells.size() > laid_out.pieces[right].cells.size();
    });
    for (int piece_index : leftovers) {
        const size_t match_index = laid_out.pieces[piece_index].match_index;
        if (kept[match_index]) continue;
        if (ranges::any_of(laid_out.pieces[piece_index].cells,
                           [&](int id) { return painted[id]; })) continue;
        edges.push_back(found[match_index]);
        kept[match_index] = true;
        for (int id : laid_out.pieces[piece_index].cells) painted[id] = true;
    }
    for (size_t found_index = 0; found_index < found.size(); ++found_index)
        if (!kept[found_index])
            throw_away(found, found_index, correspondence_facts::overlaps_a_kept_piece,
                       discarded, discard_why);
}

// STAGE 4b — nothing fits together, so the grid is not a tiling of one thing.
// The weaker reading: strongest claim first, keep a match only if it explains a
// cell no kept match explains. Nothing here needs to ask about chance — anything
// that could not stand alone never got this far.
inline void fall_back_to_greedy(const vector<correspondence_facts::edge>& found,
                                const vector<grid>& from_glyphs,
                                const vector<grid>& into_glyphs,
                                const grid& into,
                                vector<correspondence_facts::edge>& edges,
                                vector<correspondence_facts::edge>* discarded,
                                vector<int8_t>* discard_why) {
    auto glyph_of = [&](const correspondence_facts::edge& edge) {
        return edge_footprint(edge, from_glyphs, into_glyphs);
    };
    // THE SAME PREFERENCE THE FITTING USES, because this is the same question
    // asked in a hurry: biggest first, and being the right color only separates
    // two of a size. Said differently in the two places, a pair would disagree
    // about which reading was better depending on whether the search finished.
    vector<size_t> strongest_first(found.size());
    iota(strongest_first.begin(), strongest_first.end(), (size_t)0);
    stable_sort(strongest_first.begin(), strongest_first.end(),
                [&](size_t left, size_t right) {
        const int left_cells = lit_cells(glyph_of(found[left]));
        const int right_cells = lit_cells(glyph_of(found[right]));
        if (left_cells != right_cells) return left_cells > right_cells;
        return found[left].exact && !found[right].exact;
    });

    bool taken[MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (size_t found_index : strongest_first) {
        const grid glyph = glyph_of(found[found_index]);
        const int at_row = found[found_index].row, at_col = found[found_index].col;
        bool adds = false;
        for (int row = 0; row < glyph.height && !adds; ++row)
            for (int col = 0; col < glyph.width; ++col)
                if (landing_shows(found[found_index], glyph, into, row, col) &&
                    !taken[at_row + row][at_col + col]) {
                    adds = true; break;
                }
        if (!adds) {
            throw_away(found, found_index, correspondence_facts::cells_already_explained,
                       discarded, discard_why);
            continue;
        }
        for (int row = 0; row < glyph.height; ++row)
            for (int col = 0; col < glyph.width; ++col)
                if (landing_shows(found[found_index], glyph, into, row, col))
                    taken[at_row + row][at_col + col] = true;
        edges.push_back(found[found_index]);
    }
}

// ONE direction of the look: for every object on the FROM side, every place it
// turns up on the INTO side. Both edge sources, deduplicated by placement.
// Direction-neutral by construction — the forward and backward passes are this
// same function with the sides swapped. THE SET TEST decides which matches
// survive, not the cost of any one match — the four stages above are it.
// Story in notes/1 - Overall Design/Inference Master.md, section 3d.
inline fit_verdict collect_edges(const vector<correspondence_facts::object_info>& from_objects,
                          const vector<grid>& from_glyphs,
                          const vector<correspondence_facts::object_info>& into_objects,
                          const vector<grid>& into_glyphs,
                          const grid& into, int8_t background, bool may_hang_off,
                          vector<correspondence_facts::edge>& edges,
                          vector<correspondence_facts::edge>* discarded = nullptr,
                          vector<int8_t>* discard_why = nullptr,
                          match_mode match = {}) {
    const vector<correspondence_facts::edge> found =
        gather_matches(from_objects, from_glyphs, into_objects, into,
                       background, may_hang_off, discarded, discard_why, match);

    const fitting laid_out = set_up_the_fitting(found, from_glyphs, into_glyphs, into, background);

    vector<int> chosen;
    const fit_verdict searched =
        fit_the_pieces_together(laid_out, found, chosen);
    if (searched == fit_verdict::fitted) {
        keep_what_fitted(chosen, laid_out, found, edges, discarded, discard_why);
        return fit_verdict::fitted;
    }

    fall_back_to_greedy(found, from_glyphs, into_glyphs, into,
                        edges, discarded, discard_why);
    return searched;
}

// A piece nothing found still corresponds to something: WHICH PIECE OF MY OWN
// COLOR AND MY OWN SIZE DO I SIT ON TOP OF THE MOST. Same stuff, rearranged —
// the freedom left when moving, turning and recoloring have all failed, and the
// only one that can speak for a piece that was squashed, since no rigid motion
// changes a size.
// IT ONLY EVER POINTS ONE WAY: I AM PART OF THAT, never that is part of me. A
// piece may be answered by the piece that covers it — a seed sitting under the
// first tile stamped over it is answered by that tile, and neither has to rival
// the other for the cells they share. Pointed the other way the same words let
// an eighteen-cell blob explain itself by touching a two-cell marker, overlap
// cashed as ownership, and a reading that grew copies out of nothing scored a
// perfect nothing-left-over. So the piece must be no bigger than what it sits
// on. Not bigger, rather than exactly equal: equal is the squashed case, where
// the stuff is conserved and only its shape moved.
// IT CANNOT FLOOD — one edge per piece, only where pieces overlap, only for
// pieces nothing accounted for, and only between pieces of one color and one
// weight: without those gates any two boxes that touch could be called one thing.
// Story in notes/1 - Overall Design/Inference Master.md, section 3d.
inline void sit_on_top(const vector<correspondence_facts::object_info>& from_objects,
                       const vector<grid>& from_glyphs,
                       const vector<correspondence_facts::object_info>& into_objects,
                       const vector<grid>& into_glyphs,
                       vector<correspondence_facts::edge>& edges) {
    for (size_t from_index = 0; from_index < from_objects.size(); ++from_index) {
        // Only pieces nothing else accounted for. Asked of the edges themselves,
        // so this does not depend on a tally someone remembered to keep.
        if (ranges::any_of(edges, [&](const correspondence_facts::edge& edge) {
                return edge.from == (int16_t)from_index; }))
            continue;
        const auto& piece = from_objects[from_index];
        const grid& glyph = from_glyphs[from_index];
        int best = -1, most_shared = 0;
        for (size_t into_index = 0; into_index < into_objects.size(); ++into_index) {
            const auto& landing = into_objects[into_index];
            if (landing.color != piece.color || landing.count < piece.count) continue;
            const grid& landing_glyph = into_glyphs[into_index];
            int shared = 0;
            for (int row = 0; row < glyph.height; ++row)
                for (int col = 0; col < glyph.width; ++col) {
                    if (glyph.data[row][col] == TRANSPARENT) continue;
                    int in_landing_row = piece.top + row - landing.top;
                    int in_landing_col = piece.left + col - landing.left;
                    if (in_landing_row < 0 || in_landing_row >= landing_glyph.height ||
                        in_landing_col < 0 || in_landing_col >= landing_glyph.width) continue;
                    shared += (landing_glyph.data[in_landing_row][in_landing_col] != TRANSPARENT);
                }
            if (shared > most_shared) { most_shared = shared; best = (int)into_index; }
        }
        if (best < 0) continue;
        correspondence_facts::edge edge;
        edge.from = (int16_t)from_index;
        edge.to = (int16_t)best;
        edge.row = into_objects[best].top;
        edge.col = into_objects[best].left;
        edge.exact = false;
        edge.same_shape = false;
        edges.push_back(edge);
    }
}

// EDGES LEAVING each input piece — how many places it went. Every edge names
// its piece in `from`, so this is a count, not a tally kept as the edges were
// made. One writer, and it cannot disagree with `edges`.
inline void count_edges_leaving(const vector<correspondence_facts::edge>& edges,
                                vector<correspondence_facts::object_info>& from_objects) {
    for (auto& piece : from_objects) piece.edges_touching = 0;
    for (const auto& edge : edges)
        if (edge.from >= 0 && (size_t)edge.from < from_objects.size()) {
            from_objects[edge.from].edges_touching++;
            from_objects[edge.from].answered = true;
        }
}

// EDGES ARRIVING at each output piece — how many pieces met there. What the
// other end has to say, without asking it: a landing is credited to every piece
// its footprint touches, which is all the far side ever added.
// Story in notes/1 - Overall Design/Inference Master.md, section 3d.
inline void count_edges_arriving(const vector<correspondence_facts::edge>& edges,
                            const vector<grid>& from_glyphs,
                            vector<correspondence_facts::object_info>& into_objects,
                            const vector<grid>& into_glyphs) {
    for (const auto& edge : edges) {
        const grid footprint = edge_footprint(edge, from_glyphs, into_glyphs);
        for (size_t into_index = 0; into_index < into_objects.size(); ++into_index) {
            const auto& landing = into_objects[into_index];
            const grid& landing_glyph = into_glyphs[into_index];
            // Does the landing touch this piece at all?
            bool touches = false;
            for (int row = 0; row < landing_glyph.height && !touches; ++row)
                for (int col = 0; col < landing_glyph.width; ++col) {
                    if (landing_glyph.data[row][col] == TRANSPARENT) continue;
                    int in_footprint_row = landing.top + row - edge.row;
                    int in_footprint_col = landing.left + col - edge.col;
                    if (in_footprint_row < 0 || in_footprint_row >= footprint.height ||
                        in_footprint_col < 0 || in_footprint_col >= footprint.width) continue;
                    if (footprint.data[in_footprint_row][in_footprint_col] != TRANSPARENT) {
                        touches = true; break;
                    }
                }
            if (touches) {
                into_objects[into_index].edges_touching++;
                // AN EDGE THAT NAMES AN OBJECT SPEAKS FOR IT. Matching runs both
                // ways and the merge keeps one of every agreeing pair, so the
                // far end has no edge of its own to be answered by — and a
                // landing that names a piece was always a claim about two
                // things. Only a landing on bare ground needs the shapes to
                // agree, because there is no piece there to have been meant.
                if (edge.same_shape || (int16_t)into_index == edge.to)
                    into_objects[into_index].answered = true;
            }
        }
    }
}

// The step to cut on, when the sizes name one: one side a whole number of the
// other across and down, by the SAME count in every pair, and actually bigger.
// Then the small side is one block and the big side is a grid of them. WHICH
// side is small is not fixed — a grid tiled out and a tiling read back are
// the same fact seen from two ends, and a step that only knew how to grow would
// have nothing to say about the second. 0,0 = no step, so the repeating_step
// split has nothing to offer and loses on its own.
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

// The block one pair is cut on under this cut. ONE BLOCK FOR THE PAIR — not for
// a side. Pieces of different size cannot correspond, so the two grids must come
// apart on the same block; and a reading that runs both ways round would have
// the two directions cutting differently the moment the block was measured on
// "the side being read from".
//
// WHICH grid it is measured on is itself a question for the looking: measure on
// each, keep whichever describes its own grid more cheaply. A noisy grid and
// its clean counterpart both get cut on the clean one's block, which is the
// whole reason a denoise puzzle reads at all.
// Story in notes/1 - Overall Design/Inference Master.md, section 3e.
inline pair<int, int> block_measured_on(const grid& one_grid, int8_t background,
                                        int sizes_width, int sizes_height) {
    int width = sizes_width, height = sizes_height;
    repeat_periods(one_grid, background, width, height);
    return { width, height };
}

// THIS BLOCK IS NOT FOR AN OP TO USE. It is elected by looking at BOTH grids,
// which is why the strict rule can afford to be strict: a unit that fits only
// once in the input usually fits twice in a taller output, and the output side
// carries the proof. 017c7c7b pair 1 is the case -- a 4-tall unit, 6-tall input
// (fits once and a bit), 9-tall output (fits twice), found on the output.
// An op sees the input only, and the test pair has no output at all, so it can
// never reach this block. Hand an op the elected block and it solves train off
// the answer and misses on test -- a good-looking solution right up until it is
// scored.
//
// SO THE Pattern OP DIVERGES FROM THIS ON PURPOSE. Same function, different
// question: this one proves a grid repeats and has two grids to prove it with,
// Pattern only names the unit to continue with and has one. It calls
// repeat_periods with must_fit_twice=false. The two agree wherever a period is
// real and can only part where the evidence is one-and-a-half copies thin --
// which is exactly where false periods live, so the divergence is a standing
// risk, not a settled thing. If Pattern ever reads a motif nobody believes,
// this is the first place to look.
inline pair<int, int> block_of(const grid& input, const grid& output, int8_t background,
                               split_kind cut_kind, int sizes_width, int sizes_height) {
    if (cut_kind != split_kind::repeating_motif)
        return { sizes_width, sizes_height };
    const auto from_input =
        block_measured_on(input, background, sizes_width, sizes_height);
    const auto from_output =
        block_measured_on(output, background, sizes_width, sizes_height);
    // Cut each grid on its own candidate and ask what that costs to say; the
    // cheaper block is the one that found real repetition rather than noise.
    auto cost_of = [&](const grid& one_grid, pair<int, int> block) {
        vector<correspondence_facts::object_info> pieces;
        segment_for_correspondence(one_grid, background, cut_kind, pieces, nullptr,
                                   block.first, block.second);
        return describe_cost(one_grid, background, pieces);
    };
    return cost_of(output, from_output) < cost_of(input, from_input) ? from_output
                                                                    : from_input;
}

// THE DETECTOR · the one pass that reads all of the above off the train pairs.


// Fills correspondence_facts: cut both sides of each train pair ONCE, then read
// the correspondence BOTH WAYS ROUND and keep the edges of both. No same-size
// gate: every edge source works across any output size.
//
// THERE IS NO DIRECTION TO CHOOSE. Which side to read from was the one question
// neither grid could answer and every score got wrong, so it is not asked: both
// answers are taken and the ones that agree are collapsed to the stronger claim.
// Every edge carries the side it starts on, and nothing above here needs to.
// Story in notes/1 - Overall Design/Inference Master.md, sections 3d-3e.
inline correspondence_facts detect_correspondence(const vector<example>& train,
                                                  int8_t background,
                                                  split_kind cut_kind,
                                                  match_mode match = {}) {
    correspondence_facts facts;
    const auto [sizes_step_width, sizes_step_height] = repeating_step_of(train);
    if (train.empty()) return facts;
    facts.pairs.resize(train.size());
    facts.cut_kind = cut_kind;

    for (size_t pair_index = 0; pair_index < train.size(); ++pair_index) {
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        if (input.width <= 0 || input.height <= 0 ||
            output.width <= 0 || output.height <= 0) continue;

        // ONE CUT, ONE BLOCK, BOTH GRIDS. Cut once here and both readings work
        // off the same pieces — measured per direction the two would disagree
        // about what a piece is, and their edges could not be put in one list.
        const auto [step_width, step_height] = block_of(input, output, background,
                                                        cut_kind, sizes_step_width,
                                                        sizes_step_height);
        vector<grid> input_glyphs, output_glyphs;
        segment_for_correspondence(input, background, cut_kind, pair.input_objects,
                                   &input_glyphs, step_width, step_height);
        segment_for_correspondence(output, background, cut_kind, pair.output_objects,
                                   &output_glyphs, step_width, step_height);
        // Which objects are copies of one another, and which speaks for each
        // set. Here, where the cells still exist — a shape hash and a cell count
        // cannot tell one block from another, and downstream that is all there
        // is left of them.
        const bool one_set_for_the_grid = (cut_kind == split_kind::repeating_motif);
        find_copies(pair.input_objects, input_glyphs, one_set_for_the_grid);
        find_copies(pair.output_objects, output_glyphs, one_set_for_the_grid);
        // What the cut cost to say, on each grid. The pair is described by both.
        pair.describe_cost =
            0.5 * (describe_cost(input, background, pair.input_objects) +
                   describe_cost(output, background, pair.output_objects));
        // A pattern reading is read by a different collector: a block is a window
        // onto a tiling that carries on, a shape is a thing that does not.
        // HANGING OFF THE EDGE IS NOT WHAT SEPARATES THEM — every reading may do
        // that now, because a wall cutting a copy in half is a fact about the
        // grid and not about the piece.
        const bool pattern_reading = (cut_kind == split_kind::repeating_motif ||
                                      cut_kind == split_kind::repeating_step);

        // THE SAME PASS, TWICE, WITH THE SIDES SWAPPED. Nothing below cares
        // which grid it was handed; the side is stamped on the edges afterwards.
        auto read_one_way = [&](bool from_input) {
            const auto& from_objects = from_input ? pair.input_objects : pair.output_objects;
            const auto& into_objects = from_input ? pair.output_objects : pair.input_objects;
            const auto& from_glyphs = from_input ? input_glyphs : output_glyphs;
            const auto& into_glyphs = from_input ? output_glyphs : input_glyphs;
            const grid& into = from_input ? output : input;
            vector<correspondence_facts::edge> edges, discarded;
            vector<int8_t> discard_why;
            fit_verdict verdict;
            if (pattern_reading) {
                verdict = collect_motif_edges(from_objects, from_glyphs, into_objects,
                                              into, background, step_width, step_height,
                                              edges, match);
            } else {
                verdict = collect_edges(from_objects, from_glyphs, into_objects,
                                        into_glyphs, into, background, match.hang_off,
                                        edges, &discarded, &discard_why, match);
                sit_on_top(from_objects, from_glyphs, into_objects, into_glyphs, edges);
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

        // WHERE THE TWO ENDS SAY THE SAME THING, SAY IT ONCE. An input piece
        // found at an output piece and that output piece found back at the input
        // piece are one correspondence, not two, and counting it twice would
        // make every object look twice as busy as it is. The stronger claim
        // survives — found as itself beats found in another color beats only
        // sitting on something — and a tie goes to the forward edge, because
        // that is the one you could actually run.
        // WHICH TWO PIECES AN EDGE JOINS, said the same way whichever end it was
        // read from: the input piece, then the output piece. A landing with no
        // piece under it has no other end, so it agrees with nothing.
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
        // WHERE A PIECE SITS, not its place in the cut — the cut is biggest
        // first, so walking by place hopped up and down the grid.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3d.
        stable_sort(pair.edges.begin(), pair.edges.end(),
                    [&](const correspondence_facts::edge& left,
                        const correspondence_facts::edge& right) {
            if (left.from_input != right.from_input) return left.from_input;
            const auto& pieces = left.from_input ? pair.input_objects
                                                 : pair.output_objects;
            const auto& here = pieces[left.from];
            const auto& there = pieces[right.from];
            if (here.top != there.top)   return here.top < there.top;
            if (here.left != there.left) return here.left < there.left;
            if (left.row != right.row)   return left.row < right.row;
            return left.col < right.col;
        });

        // The two halves of the merged list, each in its own frame, so every
        // helper below stays the one-direction function it always was.
        vector<correspondence_facts::edge> forward, backward;
        for (const auto& edge : pair.edges)
            (edge.from_input ? forward : backward).push_back(edge);

        // HOW MANY EDGES TOUCH EACH PIECE, counting both ends. A piece is busy
        // whether it went somewhere or something arrived on it, and it is
        // answered by either one as well — asking only who arrived left a piece
        // that plainly went somewhere reading as though nothing spoke for it.
        count_edges_leaving(forward, pair.input_objects);
        count_edges_leaving(backward, pair.output_objects);
        count_edges_arriving(forward, input_glyphs, pair.output_objects, output_glyphs);
        count_edges_arriving(backward, output_glyphs, pair.input_objects, input_glyphs);
        pair.pieces_used = (int)pair.edges.size();

        // A REFUSAL IS ONLY A REFUSAL IF ITS PIECE NEVER GOT FOUND. Both
        // readings are kept and only the surviving edges were merged, so every
        // drawn line had its thrown-away twin sitting in this list beside it.
        // A piece something found was not refused — what else it might have
        // matched is a road not taken. Asked of the PIECE and not of the match,
        // because most matches name no object on the other side at all: they
        // are a place, and two places cannot be told to mean one thing.
        // ASKED HERE AND WITH `answered`, which is the same question the screen
        // asks. Asked earlier it had to be answered a second way — is this piece
        // a named end of some edge — and that misses a piece found by a landing
        // covering it, so a piece could read as found and have refusals listed
        // as though nothing had found it.
        size_t still_refused = 0;
        for (size_t gone = 0; gone < pair.discarded.size(); ++gone) {
            const auto& thrown = pair.discarded[gone];
            const auto& objects = thrown.from_input ? pair.input_objects
                                                    : pair.output_objects;
            if (thrown.from >= 0 && (size_t)thrown.from < objects.size() &&
                objects[thrown.from].answered) continue;
            pair.discarded[still_refused] = thrown;
            pair.discard_why[still_refused] = pair.discard_why[gone];
            ++still_refused;
        }
        pair.discarded.resize(still_refused);
        pair.discard_why.resize(still_refused);

        // WHERE THE UNANSWERED ONES CAME FROM, OR WENT. An object nothing
        // answers has no partner object, but it is not nowhere: it covers a
        // patch of background on the other grid. Asked only of those objects,
        // because for the rest the partner IS the answer. See object_info's
        // blank fields, and find_blanks for what closed in means.
        auto note_blanks = [&](vector<correspondence_facts::object_info>& objects,
                               const vector<grid>& glyphs, const grid& other) {
            if (ranges::all_of(objects, [](const auto& one) {
                    return one.answered; })) return;
            vector<blank_patch> patches;
            int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM];
            find_blanks(other, background, patches, patch_of);
            for (size_t i = 0; i < objects.size() && i < glyphs.size(); ++i) {
                auto& one = objects[i];
                if (one.answered) continue;
                for (int row = 0; row < glyphs[i].height; ++row)
                    for (int col = 0; col < glyphs[i].width; ++col) {
                        if (glyphs[i].data[row][col] == TRANSPARENT) continue;
                        const int at_row = one.top + row, at_col = one.left + col;
                        if (at_row < 0 || at_row >= other.height ||
                            at_col < 0 || at_col >= other.width) continue;
                        const int16_t which = patch_of[at_row][at_col];
                        if (which < 0) continue;
                        const auto& patch = patches[which];
                        one.blank_left = patch.left;   one.blank_top = patch.top;
                        one.blank_width = patch.width; one.blank_height = patch.height;
                        one.blank_cells = patch.cells;
                        one.blank_enclosed = patch.enclosed;
                        row = glyphs[i].height; break;
                    }
            }
        };
        note_blanks(pair.output_objects, output_glyphs, input);
        note_blanks(pair.input_objects, input_glyphs, output);

        for (const auto& piece : pair.input_objects) {
            if (piece.edges_touching == 0)      pair.deleted++;
            else if (piece.edges_touching == 1) pair.persisted++;
            else                                pair.copied++;
            pair.most_landings = max(pair.most_landings, piece.edges_touching);
        }
        for (const auto& landing : pair.output_objects) {
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
        // landing on it or by the piece it belongs to having gone somewhere.
        // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
        auto unexplained = [&](const grid& one_grid,
                               const vector<correspondence_facts::edge>& landing_here,
                               const vector<grid>& their_glyphs,
                               const vector<grid>& our_glyphs,
                               const vector<correspondence_facts::object_info>& ours,
                               const vector<correspondence_facts::edge>& leaving_here,
                               bool from_input) {
            bool by_pieces[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            mark_painted(landing_here, their_glyphs, our_glyphs, one_grid,
                         by_pieces, false, true);
            // A piece explains its own cells once anything speaks for it:
            // it went somewhere, or something arrived that named it by name.
            vector<bool> answered(ours.size(), false);
            for (const auto& edge : leaving_here)
                if (edge.from >= 0 && (size_t)edge.from < answered.size())
                    answered[edge.from] = true;
            for (const auto& edge : landing_here)
                if (edge.to >= 0 && (size_t)edge.to < answered.size())
                    answered[edge.to] = true;
            for (size_t i = 0; i < ours.size(); ++i) {
                if (!answered[i]) continue;
                const auto& piece = ours[i];
                for (int row = 0; row < piece.height; ++row)
                    for (int col = 0; col < piece.width; ++col)
                        if (our_glyphs[i].data[row][col] != TRANSPARENT)
                            by_pieces[piece.top + row][piece.left + col] = true;
            }
            for (int row = 0; row < one_grid.height && row < MAX_GRID_DIM; ++row)
                for (int col = 0; col < one_grid.width && col < MAX_GRID_DIM; ++col) {
                    const int color = one_grid.data[row][col];
                    if (color >= 0 && color != background && !by_pieces[row][col])
                        pair.left_over.push_back({ from_input, (int8_t)row, (int8_t)col });
                }
        };
        unexplained(output, forward, input_glyphs, output_glyphs,
                    pair.output_objects, backward, false);
        unexplained(input, backward, output_glyphs, input_glyphs,
                    pair.input_objects, forward, true);

        // Coverage ledger, output side: cells carried over by an exact edge vs
        // made from nothing. Still an output-side question — what the answer
        // owes to what it was given.
        bool covered[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        mark_painted(forward, input_glyphs, output_glyphs, output, covered);
        for (int row = 0; row < output.height && row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < output.width && col < MAX_GRID_DIM; ++col) {
                const int color = output.data[row][col];
                if (color < 0 || color == background) continue;
                if (covered[row][col]) pair.exact_covered_cells++;
                else                   pair.created_cells++;
            }
    }
    facts.valid = true;
    return facts;
}

// LOOK, GLANCE BACK, MATCH, LOOK AGAIN — the order a person reads a pair in,
// and they go round it a handful of times, not once and not exhaustively.
//
// FIRST GLANCE, at one grid: how does this grid come apart, and which side
// is worth reading from? Both are questions about the GRIDS, so looking
// answers them — cut each grid every way and keep the cut that says the most in
// the least (describe_cost). 14 cheap looks, no matching of any kind.
//
// SECOND GLANCE, at the other grid: could the pieces I just made even BE over
// there? A piece taller or wider than the other grid can never be found on it,
// so that cut is dead before it costs anything. Two grid sizes, no search.
//
// THEN MATCH, and only then: what a piece was ALLOWED TO DO to get where it went
// is invisible in either grid. So the freedoms alone are searched — 8, for the
// one cut the glances chose. Among them the old judge stands:
// fewest cells left unexplained on both grids, fewest pieces to break a tie.
//
// THEN LOOK AGAIN. A reading that leaves the grid unexplained has said
// "wrong cut" as loudly as it can, so the glance goes back and takes the next
// cut down. Only a cut that accounts for the WHOLE pair ends the walk; anything
// short of that keeps looking and lets the matching pick between what it found.
//
// EVERY READING KEPT WAS MATCHED AND SCORED. There is no second class
// of reading here and nothing in the table that never competed — the winner
// beat all of them on the same numbers.
// Story in notes/1 - Overall Design/Inference Master.md, section 3e.
inline correspondence_facts pick_split(const vector<example>& train, int8_t background) {
    const auto [sizes_step_width, sizes_step_height] = repeating_step_of(train);

    // THE LOOKING. One number per cut, off the grids alone. Scored on BOTH
    // grids: the cut has to make sense of the pair, and comparing a total taken
    // on one grid against a total taken on the other was never a fair
    // question — which is why direction is not decided here, or anywhere.
    vector<correspondence_facts::reading> looks;
    for (int cut_kind = 0; cut_kind < (int)split_kind::split_count; ++cut_kind) {
        correspondence_facts::reading look;
        look.cut_kind = (split_kind)cut_kind;
        double shares = 0.0;
        int counted = 0;
        for (const auto& train_pair : train) {
            const grid& input = train_pair.input;
            const grid& output = train_pair.output;
            if (input.width <= 0 || input.height <= 0 ||
                output.width <= 0 || output.height <= 0) continue;
            const auto [step_width, step_height] =
                block_of(input, output, background, look.cut_kind,
                         sizes_step_width, sizes_step_height);
            vector<correspondence_facts::object_info> input_pieces, output_pieces;
            segment_for_correspondence(input, background, look.cut_kind, input_pieces,
                                       nullptr, step_width, step_height);
            segment_for_correspondence(output, background, look.cut_kind, output_pieces,
                                       nullptr, step_width, step_height);
            shares += 0.5 * (describe_cost(input, background, input_pieces) +
                             describe_cost(output, background, output_pieces));
            ++counted;
            look.pieces_found += (int)(input_pieces.size() + output_pieces.size());
            // THE SECOND GLANCE, at the grid the pieces would have to turn up
            // on. One that fits somewhere is all this asks for — a cut where not
            // one piece could sit on the other grid cannot match, and no amount
            // of freedom rescues it. Asked both ways, because both are read.
            // IT IS NOW STRICTER THAN THE MATCHING IT GUARDS, and here is what
            // it would cost: a pair whose grids are a turn of each other, say a
            // 10-wide 2-tall input against a 2-wide 10-tall output. Neither
            // piece fits the other grid as it sits, so the cut is killed here —
            // and yet a quarter turn lands it exactly, which is the whole answer
            // to that puzzle. This asks about the piece as it sits, while the
            // matching may turn it and may hang it off the edge.
            // It only fires when NOTHING fits either way, so nothing seen so far
            // has tripped it. It is wrong on the day a turn or a wall is the
            // answer, and that is the day to delete it.
            auto any_fits = [](const vector<correspondence_facts::object_info>& pieces,
                               const grid& other) {
                return ranges::any_of(pieces, [&](const auto& piece) {
                    return piece.width <= other.width && piece.height <= other.height; });
            };
            if (!any_fits(input_pieces, output) && !any_fits(output_pieces, input))
                look.pieces_can_fit = false;
        }
        if (counted) look.describe_cost = shares / counted;
        looks.push_back(look);
    }
    // Cheapest look first, but everything the second glance killed sinks to the
    // bottom — still there, because a dead cut is the only honest answer when
    // every cut is dead.
    stable_sort(looks.begin(), looks.end(),
                [](const correspondence_facts::reading& left,
                   const correspondence_facts::reading& right) {
        if (left.pieces_can_fit != right.pieces_can_fit) return left.pieces_can_fit;
        return left.describe_cost < right.describe_cost;
    });

    // THE MATCHING, and going back round. One pass = one cut's eight freedoms.
    // A pass that found nothing is the signal to take the next cut, so the walk
    // stops the moment a cut actually says something.
    // A reading and the pairs it was scored from, kept together so nothing has
    // to be read a second time to get at the edges.
    struct scored_reading {
        correspondence_facts::reading summary;
        vector<correspondence_facts::pair_info> pairs;
    };
    auto search_one = [&](const correspondence_facts::reading& look) {
        vector<scored_reading> pass;
        for (int match_kind = 0; match_kind < match_mode::COUNT; ++match_kind) {
            const match_mode match = match_mode::from_index(match_kind);
            correspondence_facts tried = detect_correspondence(
                train, background, look.cut_kind, match);
            correspondence_facts::reading scored;
            scored.cut_kind = look.cut_kind;
            scored.match_kind = match;
            scored.pieces_can_fit = look.pieces_can_fit;
            double shares = 0.0;
            for (const auto& pair : tried.pairs) {
                scored.left_over += (int)pair.left_over.size();
                scored.pieces_used += pair.pieces_used;
                shares += pair.describe_cost;
                scored.pieces_found +=
                    (int)(pair.input_objects.size() + pair.output_objects.size());
                scored.exact += pair.found_exact;
                scored.recolored += pair.found_recolored;
                scored.sits_on += pair.sits_on;
                for (const auto& piece : pair.input_objects)
                    scored.box_cells += piece.width * piece.height;
                for (const auto& landing : pair.output_objects)
                    scored.box_cells += landing.width * landing.height;
            }
            if (!tried.pairs.empty())
                scored.describe_cost = shares / tried.pairs.size();
            pass.push_back({ scored, std::move(tried.pairs) });
        }
        stable_sort(pass.begin(), pass.end(),
                    [](const scored_reading& left, const scored_reading& right) {
            if (left.summary.left_over != right.summary.left_over)
                return left.summary.left_over < right.summary.left_over;
            if (left.summary.pieces_used != right.summary.pieces_used)
                return left.summary.pieces_used < right.summary.pieces_used;
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

    // THE GLANCE PICKS THE CUT, THE MATCHING PICKS THE FREEDOM. That is what
    // this file has said all along, and the judge now says it too: drawing an
    // edge beats drawing none, then SAYING IT IN THE LEAST, then explaining the
    // most, then fewest pieces.
    //
    // FEWEST CELLS LEFT OVER CANNOT COME FIRST, because one contestant can never
    // lose it: cut into single cells, every cell finds a cell, and nothing is
    // ever left over on any puzzle. A key one player always maxes out ranks
    // nobody. It said the grid at 271% of plain here and won against a reading
    // that said it at 121% and missed four cells — twice the words for four
    // cells' worth of silence.
    // ASKED SECOND IT STILL DECIDES EVERYTHING IT SHOULD: one cut's eight
    // freedoms all cost the same to say, so within a cut the matching alone
    // picks, which is the one question the looking cannot answer.
    // Story in notes/1 - Overall Design/Inference Master.md, section 3e.
    //
    // A MARGIN, because these are ratios and two cuts that cost the same to say
    // should fall through to what they explain rather than be split by rounding.
    const double same_price = 0.001;
    size_t elected = 0;
    for (size_t i = 1; i < passes.size(); ++i) {
        const auto& best = passes[elected].front().summary;
        const auto& here = passes[i].front().summary;
        if ((here.pieces_used > 0) != (best.pieces_used > 0)) {
            if (here.pieces_used > 0) elected = i;
        } else if (abs(here.describe_cost - best.describe_cost) > same_price) {
            if (here.describe_cost < best.describe_cost) elected = i;
        } else if (here.left_over != best.left_over) {
            if (here.left_over < best.left_over) elected = i;
        } else if (here.pieces_used < best.pieces_used) {
            elected = i;
        }
    }

    // KEPT IN THE ORDER THE TABLE IS DRAWN, one row per cut and one column per
    // freedom, so a reading's place IS its cell. Score order put the winner
    // first and everything after it in an order nothing on screen shows, so
    // stepping the pane hopped about a table sitting right there. Which one won
    // is a fact of its own now, said by `chosen`. Nothing is read again: the
    // winner's pairs are the ones the election already had.
    correspondence_facts facts;
    facts.cut_kind = passes[elected].front().summary.cut_kind;
    auto cell_of = [](const correspondence_facts::reading& one) {
        return (int)one.cut_kind * match_mode::COUNT + one.match_kind.index();
    };
    facts.readings.resize((int)split_kind::split_count * match_mode::COUNT);
    facts.reading_pairs.resize(facts.readings.size());
    for (auto& pass : passes)
        for (auto& one : pass) {
            facts.readings[cell_of(one.summary)] = one.summary;
            facts.reading_pairs[cell_of(one.summary)] = std::move(one.pairs);
        }
    facts.chosen = (int8_t)cell_of(passes[elected].front().summary);
    facts.pairs = facts.reading_pairs[facts.chosen];
    facts.valid = true;
    return facts;
}

}  // namespace arc
