// Segment grids into objects and match them across training pairs.
//
// Looking at an input and its output, we ask: what happened to each object?
// For example, a red square might move to another position, turn blue,
// or appear several times. A link between an input object and an output
// object is called a correspondence.
//
// First we need to decide which cells belong together as an object.
// We try several ways of grouping them, then look for matches in both
// directions: where did each input object go, and where did each output
// object come from? We compare these matches across the training examples
// and choose the grouping and allowed changes that best explain them.
#pragma once
#include "5_matching.hpp"
#include "8_vocabulary.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// A tiled frame reads each copy's pieces separately. Only this reading carries
// the boundaries: grids and their copies contain drawings alone. Both lists
// live throughout the nested reading; `pair` belongs to the framed list.
struct tiled_pieces {
    const example* framed = nullptr;
    const example* originals = nullptr;

    tile_size of(const example& pair) const {
        if (!originals) return {};
        const grid& input = originals[&pair - framed].input;
        return { input.width, input.height };
    }
};

// Which pieces a word or frame takes. A color mask picks any piece wearing
// one of its colors; `how` instead names a pick by shape, size or place.
// `part` optionally takes those as walls and picks the pieces lying in that
// part of what they fence off instead (SelectPartition), and `partner` the
// partners those have, on a side or any (SelectPartner). `but_those` turns it
// round, every piece but the ones it names, as InvertSelection says it. `fits`
// optionally keeps only pieces that fit their partner (WouldFit), and `like`
// only pieces like a piece of the key, the ones not picked (SelectLikeKey).
// The empty mask picks nothing; all() picks every color, of_color() just one.
struct piece_pick {
    enum picked_by : int8_t { by_holes, largest, smallest, row_band, column_band, most_common,
                             at_least_cells, largest_ranks, smallest_ranks, middle_row, middle_column,
                             only_one, pick_kinds };
    uint16_t colors = 0;
    int8_t how = -1, value = 0, fits = -1;
    bool but_those = false;
    int8_t part = -1, like = -1, partner = -1;

    static constexpr piece_pick all() { return { (uint16_t)((1u << COLOR_COUNT) - 1) }; }
    static constexpr piece_pick of_color(int color) { return { (uint16_t)(1u << color) }; }
    bool every() const { return by_color() && colors == all().colors && fits < 0 && like < 0; }
    // Whether it picks pieces by the colors they wear alone, before any filter.
    bool by_color() const { return how < 0 && part < 0 && partner < 0 && !but_those; }
    // A pick by shape, size or place may take any color.
    bool may_take_color(int color) const { return !by_color() || (colors >> color & 1) != 0; }
    bool operator==(const piece_pick& other) const {
        return how == other.how && fits == other.fits && part == other.part && like == other.like &&
               partner == other.partner && but_those == other.but_those &&
               (how >= 0 ? value == other.value : colors == other.colors);
    }
};

// The colors a piece wears, a bit each: what a pick by color reads off it.
inline uint16_t palette(const object& piece) {
    uint16_t colors = 0;
    piece.each_cell([&](int, int, int8_t color) { colors = (uint16_t)(colors | 1u << color); });
    return colors;
}

// Correspondences match objects from input to output and from output to input.

// This structure collects the correspondences and summaries of the readings tried.
struct correspondence_election {
    using object_info = object_segmentation::object_info;
    using edge = object_matching::edge;
    using rejection = object_matching::rejection;

    // WHAT A CHANGE CAN LEAVE UNSAID, a word each: where a carried piece went,
    // where it was drawn again, how it turned, the colors it came out in, the
    // cells drawn on the ground about it, how it changed its shape, and how
    // much bigger or smaller it came out.
    enum class missing : int8_t { where_it_went, where_it_was_copied, how_it_turned, a_recolor,
                                  drawn_on_the_ground, how_it_changed_shape, how_it_was_scaled, count };
    static constexpr const char* missing_words[] = {
        "where each piece went", "where each copy was drawn", "how each piece turned",
        "a recolor", "the cells drawn on the ground", "how a piece changed its shape",
        "how each piece was scaled" };

    // A CELL NOTHING HAS EXPLAINED YET, in the color its own grid shows it in.
    // Either no match covers it at all -- made_or_gone when its object has no
    // identified partner -- or a match carried it through a change, and it
    // `waits` on the word for that change: about the input piece `about`, -1
    // when no one piece is, having shown `other` on the other grid -- what an
    // input cell became, what an output cell was. `piece` is its object on its
    // own grid, -1 for a cell no piece holds, and `claimed_by` the fired rule
    // that accounts for it after all, -1 while nothing does. `in_place` is the
    // color the other grid shows at this very place when that is another color
    // -- the cell was recolored where it lies, however the matching read it --
    // and -1 otherwise.
    struct left_over_cell {
        bool from_input = true;
        int8_t row = 0, col = 0;
        bool made_or_gone = false;
        int16_t piece = -1;
        int8_t claimed_by = -1;
        int8_t color = -1;
        int8_t waits = -1;
        int16_t about = -1;
        int8_t other = -1;
        int8_t in_place = -1;
        bool waits_for(missing word) const { return waits == (int8_t)word; }
        // WHETHER THE WORD FOR `word` CAN SAY THIS CELL: the one it waits on, and
        // a recolor or a move wherever its place came out in another color,
        // however the matching read it -- 5582e5ca's 9 read as grown over the
        // others, where every other color turned 9; 18286ef8's 9 read as
        // recolored where it stood, where it stepped toward the 6.
        bool answers_to(missing word) const {
            return waits_for(word) ||
                   ((word == missing::a_recolor || word == missing::where_it_went) && in_place >= 0);
        }
    };
    using enum object_matching::refusal_reason;
    static constexpr auto& refusal_names = object_matching::refusal_names;
    struct pair_info {
        // Objects from both grids, segmented using the same method and block size.
        vector<object_info> input_info;
        vector<object_info> output_info;
        // Matches in both directions, with duplicate object pairs merged.
        vector<edge> edges;
        // Matches that were found and then thrown away, with why — otherwise
        // "never matched" and "matched and lost" look the same in the log.
        vector<rejection> discarded;
        // How the fitting decided, the weaker of the two ways it was read: it
        // fitted, it did not and the greedy pass decided, or it gave up.
        object_matching::fit_verdict verdict = object_matching::fit_verdict::fitted;
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
        // Figure cells of the output covered by color-preserving matches.
        int exact_covered_cells = 0;
        // Figure cells of the output no exact edge covers — made, not carried
        int created_cells = 0;

        // One side's objects: the input's, or the output's.
        const vector<object_info>& objects_on(bool input) const {
            return input ? input_info : output_info;
        }
        // The object an edge leaves from, or null when its index names none.
        const object_info* source_of(const edge& link) const {
            const auto& objects = objects_on(link.from_input);
            return link.from >= 0 && (size_t)link.from < objects.size() ? &objects[link.from] : nullptr;
        }
        // The object an edge lands on, or null when it landed on a place instead.
        const object_info* target_of(const edge& link) const {
            const auto& objects = objects_on(!link.from_input);
            return link.to >= 0 && (size_t)link.to < objects.size() ? &objects[link.to] : nullptr;
        }
        // THE INPUT PIECE AN EDGE IS ABOUT, whichever grid it was read from: the
        // one it left, the one it named on landing, or the one its landing sits
        // exactly on. A backward positional correspondence needs an identifiable
        // input object, so -1 when the landing is partly hidden, hangs off the
        // grid, or fits two pieces.
        int input_piece_of(const edge& link) const {
            if (link.from_input) return link.from;
            if (link.to >= 0) return link.to;
            if (link.hidden || link.outside) return -1;
            const auto& source = output_info[link.from];
            const auto [width, height] = object_matching::landing_size(link, source.summary);
            int found = -1;
            for (size_t i = 0; i < input_info.size(); ++i) {
                const auto& target = input_info[i];
                if (target.summary.row != link.row || target.summary.col != link.col ||
                    target.summary.width != width || target.summary.height != height ||
                    target.summary.count != source.summary.count || target.shape != source.shape)
                    continue;
                if (found >= 0) return -1;
                found = (int)i;
            }
            return found;
        }
        // Left-over cells whose object has no partner: made on the output side,
        // gone from the input side.
        int made() const { return partnerless_cells(false); }
        int gone() const { return partnerless_cells(true); }
        // Left-over cells no fired rule accounts for: what is still unexplained
        // once every rule has said what it says. A cell waiting on two words is
        // one cell unexplained, so each place counts once.
        int unclaimed() const {
            bool counted[2][MAX_GRID_DIM][MAX_GRID_DIM] = {};
            int left = 0;
            for (const auto& cell : left_over) {
                bool& seen = counted[cell.from_input][cell.row][cell.col];
                if (cell.claimed_by < 0 && !seen) { seen = true; ++left; }
            }
            return left;
        }
        // Left-over cells no match covered at all, as the matching left them.
        int unmatched() const {
            return (int)ranges::count_if(left_over, [](const left_over_cell& cell) {
                return cell.waits < 0; });
        }
        int partnerless_cells(bool on_input) const {
            return (int)ranges::count_if(left_over, [&](const left_over_cell& cell) {
                return cell.made_or_gone && cell.from_input == on_input;
            });
        }
        // HOW MANY PLACES AN INPUT PIECE WENT, each counted once. A piece matched
        // from both grids has two edges for one landing -- the one it left on,
        // and the one that found it from the output -- because a landing found by
        // placement names no object and the two never merge. So its edges are no
        // count of where it went: edges_touching reads 2 for a piece that only
        // slid, and 2 for one that never moved at all.
        int places_went(int input_index) const {
            if (input_index < 0 || (size_t)input_index >= input_info.size()) return 0;
            const auto& piece = input_info[input_index].summary;
            vector<pair<int, int>> places;
            auto went = [&](int row, int col) {
                if (ranges::find(places, pair<int, int>{ row, col }) == places.end())
                    places.push_back({ row, col });
            };
            for (const auto& link : edges) {
                if (!link.same_shape) continue;
                if (link.from_input) {
                    if (link.from == input_index) went(link.row, link.col);
                    continue;
                }
                // Found from the output side: it names this piece when it lands on it.
                const object_info* from = source_of(link);
                if (!from) continue;
                const auto [width, height] = object_matching::landing_size(link, from->summary);
                const bool on_this_piece = link.to == input_index ||
                    (link.to < 0 && link.row == piece.row && link.col == piece.col &&
                     width == piece.width && height == piece.height);
                if (on_this_piece) went(from->summary.row, from->summary.col);
            }
            return (int)places.size();
        }
    };
    // Total figure cells in all training grids, used to normalize leftover counts.
    int cells = 0;
    // The selected segmentation, shared by all training pairs.
    segmentation_methods segmentation = segmentation_methods::unset;
    // A RULE A READING WAS TAKEN TO SAY, and what it accounts for: the input
    // pieces its edges are about and the cells on both grids they carry, or,
    // for a rule that explains a choice rather than cells, what it says.
    // `under` is the rule whose evidence it reads, -1 for one that stands on
    // the matches alone. `stands_alone` marks a rule said in a sentence of its
    // own rather than as part of the cut's line: a rival explanation, not a part.
    struct fired_rule {
        string name;
        int pieces = 0, cells = 0;
        int8_t under = -1;
        string says;
        bool stands_alone = false;
        // HOW IT IS SAID, when it is: an act on the pieces of `pick` -- turned by
        // pose `value`, moved `value` down or across, recolored to `value`,
        // painted the color of the partner partner_by `value` finds, moved level
        // with that partner's rows or columns, or drawn again mirrored around its
        // middle (AlignTo's and MirrorAround's `value`), or grown rays along the
        // diagonals for `value` past 32 and otherwise up, down and to the sides,
        // `value` % 32 long, painted `paint` (paint_value); the bars bridged between
        // pieces of one color or the hollows filled, in `paint` or their own for -1;
        // or hollowed to the outline; or the pieces a pick by holes or by size takes,
        // recolored to `value`; or the picked pieces removed, slid to the edge,
        // moved their own size or grown a trail the direction `value` names,
        // their boxes filled in `paint`, or every room they fence off, or the
        // one `value` names by size or by part, filled in `paint`, or every
        // room in the color its key names (filled_room, ROOM_PARTS),
        // copied onto the pieces not picked, lined up CopyTo's way `value`, or
        // repeated as the Tile in `tile` says (tiled_along), or folded into one figure and
        // drawn again turned in its box by pose `value`, or moved toward their
        // partner as MoveToward's `value` says (move_toward), or recolored to
        // the color its own shape names (shape_recolor), or the picture so far
        // made symmetric the way symmetry `value` names (symmetric), or their
        // colors traded as a key drawn in the grid pairs them, the pieces not
        // picked read the way BidiColorMap's `value` reads them (key_swap), or
        // one color traded with the empty cells of their boxes (empty_swap) -- or a
        // swap of the pair numbered `value`, or the word a learned table or the size
        // ranking picks; or one of the rules read off the cut's own pieces that is
        // the whole rest of its line; or a rule that stands alone and says a whole
        // program of its own, the one numbered `value`. 7_infer groups the kinds by
        // what they change (group_of); 9_sentences spells them.
        enum said_as : int8_t { unsaid, turn, move_down, move_across, recolor, swap_colors,
                                learned_table, size_rank, color_from, align_to, mirror_around,
                                extend_rays, bridged, filled, outlined, recolor_picked, removed, to_edge,
                                shifted, trailed, boxed, filled_room, copied_onto, tiled_along, diagonal_tile,
                                piece_slam,
                                corner_chain, slide_to_touch, project_on_seeds, snap_to_anchors,
                                turned_copy, move_toward, shape_recolor, symmetric, key_swap,
                                empty_swap, own_sentences, kind_count } as = unsaid;
        int8_t value = 0;
        // The Tile it says, where it says one (tiled_along).
        tile_value tile;
        piece_pick pick = piece_pick::all();
        // The color it paints what it draws in, when it draws in one of its own.
        int8_t paint = -1;
        // HOW A PAINT SAID AFTER THE LINE LIES on what is down: over it, under it
        // -- only where nothing is -- or only onto it, where something is; or
        // only onto one color of it, or only off one color of it, as MaskBy and
        // MaskOut of that color keep it: laid_onto_color and laid_off_color
        // plus the color.
        enum laid : int8_t { laid_over, laid_under, laid_onto, laid_onto_color,
                             laid_off_color = laid_onto_color + COLOR_COUNT, laid_count = laid_off_color + COLOR_COUNT };
        int8_t lay = laid_over;
        // A WORD READ OFF THE PICTURE THE LINE HAS DRAWN SO FAR -- the hand and
        // every paint said before it -- rather than off the hand: said after
        // those paints, in a sentence of its own that opens on the picture.
        bool on_the_picture = false;
        // What the language has no word for, when this fired and cannot be said.
        string no_word_for;
    };
    // The one rule the matching fires that no freedom names.
    static constexpr const char* sits_on_name = "Sits On";
    // Scores for each segmentation and permission combination.
    // Table index: segmentation * degrees_of_freedom::COMBINATIONS + allowed.index().
    struct reading {
        // The matched evidence belongs to the interpretation it describes.
        vector<pair_info> pairs;
        // Which segmentation, and what a match was allowed to change
        segmentation_methods segmentation = segmentation_methods::unset;
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
        // WHAT IT WAS TAKEN TO SAY, one rule each, listed for the reading each
        // segmentation kept -- the one that entered the election -- and empty
        // for the combinations that lost to it.
        vector<fired_rule> fired;
        // THE CUT'S OWN LINE, in the order it is said: indices into `fired`. A
        // rule that fired and has no word sits where it would have been said.
        vector<int8_t> line;
        // ONE MORE RULE, listed under the one it reads, said next on the line,
        // and accounting for the cells it explains: `claims(pair, cell)` names
        // them. Each cell is claimed once, by the first rule that asks, and
        // counted as that rule's. Hands back where the rule sits on the list.
        template <class Claims>
        int8_t say_fired(fired_rule rule, int8_t under, Claims claims) {
            const int8_t index = (int8_t)fired.size();
            rule.under = under;
            for (size_t which = 0; which < pairs.size(); ++which)
                for (auto& cell : pairs[which].left_over)
                    if (cell.claimed_by < 0 && claims(which, cell)) {
                        cell.claimed_by = index;
                        ++rule.cells;
                    }
            fired.push_back(std::move(rule));
            line.push_back(index);
            return index;
        }
        int8_t say_fired(fired_rule rule, int8_t under) {
            return say_fired(std::move(rule), under, [](size_t, const left_over_cell&) { return false; });
        }
        // Lists a rule that claims cells the words on the line already account
        // for together, and says nothing of its own: it is not put on the line.
        // Listed only where it claims a cell; -1 where it claims none.
        template <class Claims>
        int8_t list_claims(fired_rule rule, Claims claims) {
            const int8_t index = (int8_t)fired.size();
            for (size_t which = 0; which < pairs.size(); ++which)
                for (auto& cell : pairs[which].left_over)
                    if (cell.claimed_by < 0 && claims(which, cell)) {
                        cell.claimed_by = index;
                        ++rule.cells;
                    }
            if (rule.cells == 0) return -1;
            fired.push_back(std::move(rule));
            return index;
        }
        // The rule this reading fired by that name, -1 when it fired none.
        int8_t fired_named(const string& name) const {
            for (size_t i = 0; i < fired.size(); ++i)
                if (fired[i].name == name) return (int8_t)i;
            return -1;
        }
        // THE CELLS THE LINE DRAWS WRONG that no cell left over stands for: a
        // word said where it brings every pair nearer its output, though it
        // lays some cells wrong (brings_nearer, 7_infer), and none said after it
        // set them right. Left unsaid, for the next pass to read on the picture.
        int drawn_wrong = 0;
        // What every rule together still leaves unexplained, on every pair: the
        // cells nothing claims, whether no match covered them or a change they
        // were carried through still waits on its word, and the cells the line
        // drew wrong.
        int unclaimed() const {
            int left = drawn_wrong;
            for (const auto& pair : pairs) left += pair.unclaimed();
            return left;
        }
        // The rules in the order they are read: each one, then the rules that
        // read its evidence, a step deeper. Hands back (rule, depth) pairs.
        vector<std::pair<int, int>> fired_in_order() const {
            vector<std::pair<int, int>> order;
            auto visit = [&](auto&& self, int under, int depth) -> void {
                for (int i = 0; i < (int)fired.size(); ++i)
                    if (fired[i].under == under) {
                        order.push_back({ i, depth });
                        self(self, i, depth + 1);
                    }
            };
            visit(visit, -1, 0);
            return order;
        }
    };
    vector<reading> readings;
    // A READING'S PLACE IN readings: one for every cut and every set of
    // freedoms, so past what eight bits hold once the freedoms are five.
    using reading_index = int16_t;
    // Index of the selected reading in readings.
    reading_index chosen = 0;
    // Each segmentation's best reading, as its index in readings, so a rule
    // about one way of cutting can read that cut's own evidence, elected or not.
    vector<int16_t> best_of;
    const reading* best_reading_of(segmentation_methods method) const {
        const size_t index = (size_t)method;
        if (index >= best_of.size() || best_of[index] < 0) return nullptr;
        return &readings[best_of[index]];
    }
    // Indices of other segmentations whose best readings tie with the winner.
    vector<reading_index> tied_with_chosen;
    // THE CUTS IN THE ORDER THE MATCHING RANKS THEM: each competing way of
    // seeing objects' best reading, the plainest first, beside the others whose
    // matching ties with it. Which one is said is the words' to decide
    // (read_figures): the matching only orders them and breaks a tie. And the
    // cells each one's words left unsaid, -1 for one never read.
    vector<reading_index> ranked;
    vector<vector<reading_index>> ties_of_ranked;
    vector<int> left_by_rank;
    // Reads on the cut ranked `rank`, with the ties its matching has.
    void choose(size_t rank) {
        chosen = ranked[rank];
        segmentation = readings[chosen].segmentation;
        tied_with_chosen = ties_of_ranked[rank];
    }
    // Read the winner directly, without keeping a second copy of its evidence.
    // A default election has no readings yet.
    const vector<pair_info>& chosen_pairs() const {
        static const vector<pair_info> empty;
        return readings.empty() ? empty : readings[chosen].pairs;
    }
    // WHAT EACH WAY OF SEEING OBJECTS STILL HAS NO WORD FOR, once every rule
    // that says anything has claimed its cells: an entry for each word some
    // cell still waits on, under the freedom it would be said with and on the
    // line where the word would go, with the pieces and cells still waiting.
    // Asked last, so a rule read off the elected reading after the acts has
    // taken back what it says before anything is called missing.
    void list_what_is_missing() {
        for (size_t method = 0; method < best_of.size(); ++method) {
            if (best_of[method] < 0 || !competes((segmentation_methods)method)) continue;
            auto& read = readings[best_of[method]];
            for (int word = 0; word < (int)missing::count; ++word) {
                fired_rule rule;
                rule.name = "No Word";
                rule.no_word_for = missing_words[word];
                // Each piece once: the input piece a cell is about, or failing
                // one, the output piece it is.
                vector<std::pair<size_t, int>> whose;
                for (size_t which = 0; which < read.pairs.size(); ++which)
                    for (const auto& cell : read.pairs[which].left_over) {
                        if (cell.claimed_by >= 0 || cell.waits != word) continue;
                        ++rule.cells;
                        const int piece = cell.about >= 0 ? cell.about
                                        : !cell.from_input && cell.piece >= 0 ? -2 - cell.piece : -1;
                        if (piece != -1 && ranges::find(whose, std::pair{ which, piece }) == whose.end())
                            whose.push_back({ which, piece });
                    }
                if (!rule.cells) continue;
                rule.pieces = (int)whose.size();
                read.say_fired(std::move(rule), said_under(read, (missing)word));
            }
        }
    }
    // The rule a missing word would be said under: the freedom it needs.
    static int8_t said_under(const reading& read, missing word) {
        const auto& freedom = degrees_of_freedom::freedom_names;
        switch (word) {
            case missing::where_it_went:
            case missing::where_it_was_copied:  return read.fired_named(freedom[0]);
            case missing::how_it_turned:        return read.fired_named(freedom[1]);
            case missing::a_recolor:            return read.fired_named(freedom[2]);
            case missing::how_it_was_scaled:    return read.fired_named(freedom[4]);
            case missing::how_it_changed_shape: return read.fired_named(sits_on_name);
            default:                            return -1;
        }
    }

    // A WAY OF SEEING OBJECTS, AS OPPOSED TO A CLAIM ABOUT THEM. Cells grouped by
    // touch or by color can be unhelpful but never wrong, so those compete in the
    // election. The two fixed-step block segmentations are claims the pairs can
    // prove wrong -- the grid repeats, one grid is copies of the other -- so their
    // matching is evidence for the rules that make those claims, read whichever
    // way of seeing won.
    static constexpr bool competes(segmentation_methods segmentation) {
        return segmentation != segmentation_methods::minimal_repeating_set &&
               segmentation != segmentation_methods::smallest_of_two_grids &&
               segmentation != segmentation_methods::unset;
    }

    // TEMPORARY: HOW MANY PIECES AN INPUT MAY BE CUT INTO FOR ITS CUT TO BE READ.
    // A speed limit and nothing else -- a program holds a piece for every cell
    // (MAX_OBJECTS_IN_A_LIST), but reading the words of a cut of hundreds of
    // pieces cost minutes a run (a full run took 137 s with no limit, against
    // about a minute without those cuts). f0df5ff0's inputs cut into 81. Lift it
    // once reading a big cut is cheap.
    static constexpr int MOST_PIECES_READ = 128;

    // 1. Compare the readings and keep the one that best explains the examples.
    // Try every segmentation and permission combination, then return the selected reading.
    // The pairs are figures: the plan deleted any ground (3_background.hpp).
    static correspondence_election infer(const vector<example>& train, const tiled_pieces& tiles = {}) {

        // HOW BIG THE THING BEING EXPLAINED IS. Counted once, off the pairs alone,
        // because every reading's leftovers are a share of it.
        int facts_cells = 0;
        for (const auto& train_pair : train)
            for (const grid* one : { &train_pair.input, &train_pair.output })
                one->for_each_colored_cell(TRANSPARENT, [&](int, int, int) { ++facts_cells; });

        vector<prepared_segmentation> preparations;
        for (int method = 0; method < (int)segmentation_methods::count; ++method)
            preparations.push_back(prepare_segmentation(train, (segmentation_methods)method, tiles));
        // THE CUT BY SIDES ALONE, READ ONLY WHERE IT IS ANOTHER READING THE READING
        // CAN READ. Where every grid cuts into the same pieces by corners, it is
        // that reading said again; where an input cuts into more pieces than the
        // reading reads (MOST_PIECES_READ, read_in_time below), it would not be
        // read. Either way, matched, it would only cost.
        auto prepared_by = [&](segmentation_methods method) {
            return ranges::find_if(preparations, [&](const auto& one) { return one.look.segmentation == method; });
        };
        if (const auto sides = prepared_by(segmentation_methods::four_connected_monochrome);
            sides != preparations.end()) {
            const auto corners = prepared_by(segmentation_methods::connected_monochrome);
            auto same_pieces = [](const vector<object>& one, const vector<object>& other) {
                return ranges::equal(one, other, [](const object& a, const object& b) {
                    return a.summary.col == b.summary.col && a.summary.row == b.summary.row &&
                           a.summary.width == b.summary.width && a.summary.height == b.summary.height &&
                           a.summary.count == b.summary.count; });
            };
            bool cuts_the_same = true, held = true;
            for (size_t i = 0; i < sides->pairs.size(); ++i) {
                const auto& by_sides = sides->pairs[i];
                const auto& by_corners = corners->pairs[i];
                cuts_the_same = cuts_the_same && same_pieces(by_sides.input_objects, by_corners.input_objects) &&
                                same_pieces(by_sides.output_objects, by_corners.output_objects);
                held = held && (int)by_sides.input_objects.size() <= MOST_PIECES_READ;
            }
            if (cuts_the_same || !held) preparations.erase(sides);
        }
        stable_sort(preparations.begin(), preparations.end(), [](const auto& left, const auto& right) {
            if (left.look.objects_can_fit != right.look.objects_can_fit) return left.look.objects_can_fit;
            return left.look.describe_cost < right.look.describe_cost;
        });

        correspondence_election facts;
        facts.cells = facts_cells;
        facts.readings.resize((int)segmentation_methods::count * degrees_of_freedom::COMBINATIONS);
        auto better_match = [](const reading& left, const reading& right) {
            if (left.left_over != right.left_over) return left.left_over < right.left_over;
            if (left.objects_used != right.objects_used) return left.objects_used < right.objects_used;
            return left.box_cells < right.box_cells;
        };
        // Keep every reading in its final table slot. Only the best permission
        // combination per segmentation enters the election; ties keep the first.
        // A claim's best reading is kept too, for its rule to read, but it does
        // not stand.
        vector<int> best_of_method;
        facts.best_of.assign((size_t)segmentation_methods::count, -1);
        for (const auto& prepared : preparations) {
            int best = -1;
            const bool by_size = any_match_by_size(prepared);
            matching_placements placements(train.size());
            for (int packed = 0; packed < degrees_of_freedom::COMBINATIONS; ++packed) {
                const int cell = (int)prepared.look.segmentation * degrees_of_freedom::COMBINATIONS + packed;
                reading& scored = facts.readings[cell];
                const degrees_of_freedom allowed = degrees_of_freedom::from_index(packed);
                // Where no match at another size can be found, Scale finds what
                // its twin without it does: read once, the twin's reading kept.
                if (allowed.scale && !by_size) {
                    scored = facts.readings[cell - degrees_of_freedom::SCALE_BIT];
                    scored.allowed = allowed;
                } else scored = detect_correspondence(train, prepared, allowed, placements);
                if (best < 0 || better_match(scored, facts.readings[best])) best = cell;
            }
            facts.best_of[(size_t)prepared.look.segmentation] = (int16_t)best;
            if (!competes(prepared.look.segmentation)) continue;
            list_what_fired(train, prepared, facts.readings[best]);
            read_the_acts(train, prepared, facts.readings[best]);
            best_of_method.push_back(best);
        }

        // Combine compression cost with the cost of unexplained cells.
        // Treat prices within this margin as tied, then compare object counts.
        const double same_price = 0.001;
        const double spelling_out = facts_cells > 0 ? 1.0 / facts_cells : 0.0;
        // Charge created or removed cells half price: their absence of a partner is known,
        // but the rule that creates or removes them is still unexplained.
        const double half_said = 0.5;
        auto price_of = [&](const correspondence_election::reading& one) {
            const double unsaid = (one.left_over - one.made - one.gone)
                                + (one.made + one.gone) * half_said;
            return one.describe_cost + unsaid * spelling_out;
        };
        // A CUT THE READING READS IN TIME: no input cut into more pieces than
        // MOST_PIECES_READ (TEMPORARY, see there). A program holds every piece a
        // grid can be cut into, so this is the reading's limit, not the program's.
        auto read_in_time = [](const reading& one) {
            return ranges::all_of(one.pairs, [](const pair_info& pair) {
                return (int)pair.input_info.size() <= MOST_PIECES_READ;
            });
        };
        // Which of two readings the election prefers: negative for the first,
        // positive for the second, 0 for a tie. Using objects at all comes first,
        // then a cut the reading reads in time, then the lower price, and only
        // prices within same_price of each other are settled by using fewer
        // objects.
        auto order = [&](const reading& one, const reading& other) {
            if ((one.objects_used > 0) != (other.objects_used > 0)) return one.objects_used > 0 ? -1 : 1;
            if (read_in_time(one) != read_in_time(other)) return read_in_time(one) ? -1 : 1;
            if (abs(price_of(one) - price_of(other)) > same_price)
                return price_of(one) < price_of(other) ? -1 : 1;
            if (one.objects_used != other.objects_used) return one.objects_used < other.objects_used ? -1 : 1;
            return 0;
        };
        // Ranked one at a time, each the first of the rest that none beats, so
        // the first is the one a single election would have chosen.
        vector<int> unranked = best_of_method;
        while (!unranked.empty()) {
            size_t first = 0;
            for (size_t i = 1; i < unranked.size(); ++i)
                if (order(facts.readings[unranked[i]], facts.readings[unranked[first]]) < 0) first = i;
            facts.ranked.push_back((reading_index)unranked[first]);
            unranked.erase(unranked.begin() + (ptrdiff_t)first);
        }
        // THE CUT BY SIDES ALONE COMES LAST. It sees the cells the cut by corners
        // sees, only finer, so its words are said only where they say more
        // (aedd82e4's lone cells, which touch the rest at a corner). Read ahead of
        // the cut by corners -- four-connected until eight is needed, as the
        // partners are asked -- it took the ties, and lost 4938f0c2, 64a7c07e and
        // 9172f3a0: a test cuts by corners where its pairs cut either way.
        ranges::stable_partition(facts.ranked, [&](reading_index one) {
            return facts.readings[one].segmentation != segmentation_methods::four_connected_monochrome; });
        // A cut the reading would not read in time is no candidate at all while
        // one it would is.
        if (ranges::any_of(facts.ranked, [&](reading_index one) { return read_in_time(facts.readings[one]); }))
            erase_if(facts.ranked, [&](reading_index one) { return !read_in_time(facts.readings[one]); });
        for (const reading_index one : facts.ranked) {
            vector<reading_index> ties;
            for (const int other : best_of_method)
                if (other != one && order(facts.readings[other], facts.readings[one]) == 0)
                    ties.push_back((reading_index)other);
            facts.ties_of_ranked.push_back(std::move(ties));
        }
        if (!facts.ranked.empty()) facts.choose(0);
        return facts;
    }

private:
    using matching_placements = vector<array<object_matching::placement_cache, 2>>;
    // Objects and grid-only scores shared by every permission combination.
    struct prepared_segmentation {
        correspondence_election::reading look;
        struct prepared_pair {
            pair_info info;
            vector<object> input_objects, output_objects;
            int step_width = 0, step_height = 0;
            bool read = false;
        };
        vector<prepared_pair> pairs;
    };

    using what_the_pairs_agreed = object_matching::what_the_pairs_agreed;
    using fit_verdict = object_matching::fit_verdict;

    // WHETHER A MATCH AT ANOTHER SIZE CAN BE FOUND ON A CUT: some piece of one
    // grid and a piece of the other whose shape at any size is its, grown a
    // whole number of times either way round -- what gather_matches' Source 3
    // looks for.
    static bool any_match_by_size(const prepared_segmentation& prepared) {
        auto grows_into = [](const object_summary& small, const object_summary& big) {
            for (const bool turned : { false, true }) {
                const int wide = turned ? small.height : small.width, tall = turned ? small.width : small.height;
                if (wide > 0 && big.width % wide == 0 && big.width / wide >= 2 && big.height == tall * (big.width / wide))
                    return true;
            }
            return false;
        };
        for (const auto& saved : prepared.pairs) {
            if (!saved.read) continue;
            for (const bool from_input : { true, false })
                for (const auto& one : saved.info.objects_on(from_input))
                    for (const auto& landing : saved.info.objects_on(!from_input))
                        if (landing.scaled_shape == one.scaled_shape && grows_into(one.summary, landing.summary))
                            return true;
        }
        return false;
    }

    // 2. Prepare each segmentation once, before trying its permissions.
    static prepared_segmentation prepare_segmentation(const vector<example>& train,
                                                      segmentation_methods segmentation, const tiled_pieces& tiles) {
        prepared_segmentation prepared;
        prepared.look.segmentation = segmentation;
        prepared.pairs.resize(train.size());
        double shares = 0.0;
        int counted = 0;
        // Segment before matching so another matching pass can reuse the objects.
        for (size_t pair_index = 0; pair_index < train.size(); ++pair_index) {
            auto& saved = prepared.pairs[pair_index];
            const grid& input = train[pair_index].input;
            const grid& output = train[pair_index].output;
            saved.read = !input.empty() && !output.empty();
            if (!saved.read) continue;
            auto& pair = saved.info;
            // Use the same block size for both grids.
            const auto [smaller_width, smaller_height] =
                object_segmentation::smaller_grid_step(train[pair_index]);
            const auto [step_width, step_height] = object_segmentation::block_of(input, output,
                                                            segmentation, smaller_width, smaller_height);
            vector<object>& input_objects = saved.input_objects;
            vector<object>& output_objects = saved.output_objects;
            object_segmentation::segment_for_correspondence(input, segmentation, pair.input_info,
                                       &input_objects, step_width, step_height, tiles.of(train[pair_index]));
            object_segmentation::segment_for_correspondence(output, segmentation, pair.output_info,
                                       &output_objects, step_width, step_height);
            // Group copies and choose the most typical member while their cells are available.
            const bool one_set_for_the_grid = (segmentation == segmentation_methods::minimal_repeating_set);
            object_segmentation::find_copies(pair.input_info, input_objects, one_set_for_the_grid);
            object_segmentation::find_copies(pair.output_info, output_objects, one_set_for_the_grid);
            // What the segmentation cost to say, on each grid. The pair is described by both.
            pair.describe_cost =
                0.5 * (object_segmentation::describe_cost(input, pair.input_info) +
                       object_segmentation::describe_cost(output, pair.output_info));
            shares += pair.describe_cost;
            ++counted;
            auto any_fits = [](const auto& objects, const grid& other) {
                return ranges::any_of(objects, [&](const auto& one) {
                    return one.summary.width <= other.width && one.summary.height <= other.height;
                });
            };
            // Preliminary ordering ignores rotations and clipping; all readings are still searched.
            if (!any_fits(pair.input_info, output) && !any_fits(pair.output_info, input))
                prepared.look.objects_can_fit = false;
            saved.step_width = step_width;
            saved.step_height = step_height;
        }

        if (counted) prepared.look.describe_cost = shares / counted;
        return prepared;
    }

    // 3. Follow one reading through the examples.
    // One reading: match each example, reconcile their evidence, then describe what survived.
    static reading detect_correspondence(const vector<example>& train,
                                                       const prepared_segmentation& prepared,
                                                       degrees_of_freedom allowed,
                                                       matching_placements& placements) {
        reading facts = prepared.look;
        facts.allowed = allowed;
        if (train.empty()) return facts;
        for (const auto& pair : prepared.pairs) facts.pairs.push_back(pair.info);
        for (size_t i = 0; i < train.size(); ++i)
            if (prepared.pairs[i].read)
                match_one_pair(train, prepared, allowed, facts, i, nullptr, placements);
        reconcile_examples(train, prepared, allowed, facts, placements);
        for (size_t i = 0; i < train.size(); ++i)
            if (prepared.pairs[i].read) summarize_pair(train, prepared, facts, i);
        add_up_the_pairs(facts);
        return facts;
    }

    // The reading's totals, summed over its pairs, for the election to compare.
    static void add_up_the_pairs(reading& facts) {
        double shares = 0.0;
        for (const auto& pair : facts.pairs) {
            facts.left_over += (int)pair.left_over.size();
            facts.made += pair.made();
            facts.gone += pair.gone();
            facts.objects_used += pair.objects_used;
            shares += pair.describe_cost;
            facts.exact += pair.found_exact;
            facts.recolored += pair.found_recolored;
            facts.sits_on += pair.sits_on;
            for (const bool input : { true, false })
                for (const auto& one_object : pair.objects_on(input)) {
                    facts.objects_found++;
                    facts.box_cells += one_object.summary.width * one_object.summary.height;
                }
        }
        if (!facts.pairs.empty()) facts.describe_cost = shares / facts.pairs.size();
    }

    // Match both directions, merge duplicate links, and record shared movements.
    static void match_one_pair(const vector<example>& train,
                               const prepared_segmentation& prepared, degrees_of_freedom allowed,
                               reading& facts, size_t pair_index,
                               const what_the_pairs_agreed* agreed,
                               matching_placements& placements) {
        const bool pattern_reading =
            facts.segmentation == segmentation_methods::minimal_repeating_set ||
            facts.segmentation == segmentation_methods::smallest_of_two_grids;
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        const vector<object>& input_objects = prepared.pairs[pair_index].input_objects;
        const vector<object>& output_objects = prepared.pairs[pair_index].output_objects;
        const int step_width = prepared.pairs[pair_index].step_width;
        const int step_height = prepared.pairs[pair_index].step_height;
        pair.edges.clear(); pair.discarded.clear();
        pair.agreed_moves.clear(); pair.verdict = fit_verdict::fitted;
        // THE SAME PASS, TWICE, WITH THE SIDES SWAPPED. Nothing below cares
        // which grid it was handed; the side is stamped on the edges afterwards.
        auto read_one_way = [&](bool from_input) {
            const auto& from_info = pair.objects_on(from_input);
            const auto& into_info = pair.objects_on(!from_input);
            const auto& from_objects = from_input ? input_objects : output_objects;
            const auto& into_objects = from_input ? output_objects : input_objects;
            const grid& into = from_input ? output : input;
            vector<edge> edges;
            vector<rejection> discarded;
            fit_verdict verdict;
            if (facts.segmentation == segmentation_methods::minimal_repeating_set) {
                verdict = object_matching::collect_representative_edge(from_info, from_objects,
                    into_info, into, step_width, step_height, edges, allowed);
            } else if (pattern_reading) {
                verdict = object_matching::collect_motif_edges(from_info, from_objects, into_info,
                                              into, step_width, step_height, edges, allowed);
            } else {
                verdict = object_matching::collect_edges(from_info, from_objects, into_info,
                                        into_objects, into, from_input, allowed.hang_off,
                                        edges, &discarded, allowed, agreed,
                                        &placements[pair_index][from_input ? 0 : 1]);
                object_matching::sit_on_top(from_info, from_objects, into_info, into_objects, edges);
            }
            for (auto& edge : edges) edge.from_input = from_input;
            for (auto& rejected : discarded) rejected.match.from_input = from_input;
            // The weaker of the two readings names the pass: a fit both ways is
            // a fit, and a giveup anywhere is still a giveup.
            pair.verdict = max(pair.verdict, verdict);
            pair.edges.insert(pair.edges.end(), edges.begin(), edges.end());
            pair.discarded.insert(pair.discarded.end(), discarded.begin(), discarded.end());
        };
        read_one_way(true);
        read_one_way(false);

        // Merge matches linking the same input and output objects.
        // Prefer exact over recolored over overlap-only; ties prefer the forward match.
        // Matches without a target object stay separate.
        struct two_ends { int16_t on_input, on_output; };
        auto both_ends = [](const edge& edge) {
            return edge.from_input ? two_ends{ edge.from, edge.to }
                                   : two_ends{ edge.to, edge.from };
        };
        auto joins_the_same = [&](const edge& left,
                                  const edge& right) {
            const two_ends mine = both_ends(left), theirs = both_ends(right);
            return mine.on_input >= 0 && mine.on_output >= 0 &&
                   mine.on_input == theirs.on_input &&
                   mine.on_output == theirs.on_output;
        };
        auto claim_strength = [](const edge& edge) {
            return !edge.same_shape ? 0 : edge.exact ? 2 : 1;
        };
        vector<edge> kept;
        for (const auto& link : pair.edges) {
            auto same = ranges::find_if(kept, [&](const edge& seen) {
                return joins_the_same(link, seen); });
            if (same == kept.end()) { kept.push_back(link); continue; }
            const int mine = claim_strength(link), theirs = claim_strength(*same);
            if (mine > theirs || (mine == theirs && link.from_input && !same->from_input))
                *same = link;
        }
        pair.edges.swap(kept);

        // Sort by source grid, source position, then target position for display.
        stable_sort(pair.edges.begin(), pair.edges.end(), [&](const edge& left, const edge& right) {
            if (left.from_input != right.from_input) return left.from_input;
            const auto& here = pair.source_of(left)->summary;
            const auto& there = pair.source_of(right)->summary;
            if (here.row != there.row) return here.row < there.row;
            if (here.col != there.col) return here.col < there.col;
            if (left.row != right.row) return left.row < right.row;
            return left.col < right.col;
        });

        // Group equal input-to-output movements within this pair.
        // Count each input object once per movement, using its input color.
        vector<pair_info::moved_together> moves;
        vector<vector<int>> who_moved;
        for (const auto& edge : pair.edges) {
            const object_info* one_object = pair.source_of(edge);
            if (!edge.same_shape || !one_object) continue;
            const auto went = object_matching::move_of(edge, *one_object);
            const int8_t down = (int8_t)went.first, across = (int8_t)went.second;
            const int speaker = pair.input_piece_of(edge);
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
    }

    // Let movements supported by all examples constrain ambiguous matches.
    static void reconcile_examples(const vector<example>& train,
                                   const prepared_segmentation& prepared, degrees_of_freedom allowed,
                                   reading& facts, matching_placements& placements) {
        // Use a color to constrain matches only when every read pair has one shared,
        // nonzero movement for it. The movement may differ between pairs.
        // Return null when this pair has no such movement or has several.
        auto lone_move_of = [&](size_t which, int color) {
            const pair_info::moved_together* only = nullptr;
            for (const auto& said : facts.pairs[which].agreed_moves) {
                if (said.color != color || (!said.down && !said.across)) continue;
                if (only) return (const pair_info::moved_together*)nullptr;
                only = &said;
            }
            return only;
        };
        vector<what_the_pairs_agreed> agreed(train.size());
        bool anything_agreed = false;
        for (int color = 0; color < COLOR_COUNT; ++color) {
            int read = 0, said_once = 0;
            for (size_t which = 0; which < facts.pairs.size(); ++which) {
                if (!prepared.pairs[which].read) continue;
                ++read;
                said_once += (lone_move_of(which, color) != nullptr);
            }
            if (read == 0 || said_once != read) continue;
            anything_agreed = true;
            for (size_t which = 0; which < facts.pairs.size(); ++which) {
                if (!prepared.pairs[which].read) continue;
                const auto* rule = lone_move_of(which, color);
                agreed[which].color_speaks[color] = true;
                agreed[which].down[color] = rule->down;
                agreed[which].across[color] = rule->across;
            }
        }
        if (anything_agreed)
            for (size_t which = 0; which < facts.pairs.size(); ++which) {
                if (!prepared.pairs[which].read) continue;
                // Rematch only if an existing match disagrees.
                const auto& pair = facts.pairs[which];
                const bool any_disagrees = ranges::any_of(pair.edges, [&](const edge& link) {
                    const object_info* source = pair.source_of(link);
                    return source && object_matching::the_pairs_disagree(agreed[which], link, *source);
                });
                if (any_disagrees) match_one_pair(train, prepared, allowed, facts, which, &agreed[which], placements);
            }

    }

    // Describe the final links, unmatched objects, and unexplained cells of one pair.
    static void summarize_pair(const vector<example>& train,
                               const prepared_segmentation& prepared,
                               reading& facts, size_t pair_index) {
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
        const vector<object>& input_objects = prepared.pairs[pair_index].input_objects;
        const vector<object>& output_objects = prepared.pairs[pair_index].output_objects;

        // The two halves of the merged list, each in its own frame, so every
        // helper below stays the one-direction function it always was.
        vector<edge> forward, backward;
        for (const auto& edge : pair.edges)
            (edge.from_input ? forward : backward).push_back(edge);

        // Count matches at both ends so arrivals and departures both identify partners.
        count_edges_leaving(forward, pair.input_info);
        count_edges_leaving(backward, pair.output_info);
        count_edges_arriving(forward, input_objects, pair.output_info, output_objects);
        count_edges_arriving(backward, output_objects, pair.input_info, input_objects);
        pair.objects_used = (int)pair.edges.size();

        // Validate instances against their own matched motif, including blank cells.
        // Short boundary blocks must equal the visible prefix of the full motif.
        if (facts.segmentation == segmentation_methods::minimal_repeating_set) {
            auto instances = [](vector<object_info>& info, const vector<object>& objects) {
                for (int rep = 0; rep < (int)info.size(); ++rep) {
                    if (!info[rep].has_partner) continue;
                    const auto& whole = objects[rep];
                    for (int i = 0; i < (int)info.size(); ++i) {
                        if (i == rep || info[i].has_partner) continue;
                        const auto& part = objects[i];
                        if (part.summary.width > whole.summary.width || part.summary.height > whole.summary.height) continue;
                        bool equal = true;
                        for (int r = 0; r < part.summary.height; ++r)
                            for (int c = 0; c < part.summary.width; ++c)
                                if (part.data[r][c] != whole.data[r][c]) equal = false;
                        if (equal) info[i].instance_of = (int16_t)rep;
                    }
                }
            };
            instances(pair.input_info, input_objects);
            instances(pair.output_info, output_objects);
        }

        // Keep rejection records only for objects that still have no partner.
        erase_if(pair.discarded, [&](const rejection& rejected) {
            const object_info* source = pair.source_of(rejected.match);
            return source && source->has_partner;
        });

        // Record the empty region touched by each unmatched object on the other grid.
        auto note_blanks = [&](vector<object_info>& info,
                               const vector<object>& objects, const grid& other) {
            if (ranges::all_of(info, [](const auto& one) {
                    return one.has_partner; })) return;
            vector<blank_patch> patches;
            int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM];
            find_blanks(other, patches, patch_of);
            for (size_t i = 0; i < info.size() && i < objects.size(); ++i) {
                auto& one = info[i];
                if (one.has_partner || one.instance_of >= 0) continue;
                // The patch under the first of its cells, in reading order, that has one.
                objects[i].each_cell([&](int row, int col, int8_t) {
                    if (one.blank.found() || !other.contains(row, col) || patch_of[row][col] < 0) return;
                    one.blank = patches[patch_of[row][col]];
                });
            }
        };
        note_blanks(pair.output_info, output_objects, input);
        note_blanks(pair.input_info, input_objects, output);

        // The same three questions from each end: nowhere or nothing, one, several.
        auto count_ends = [](const vector<object_info>& info, int& none, int& one, int& many, int& most) {
            for (const auto& piece : info) {
                if (piece.instance_of >= 0) continue;
                if (piece.edges_touching == 0)      none++;
                else if (piece.edges_touching == 1) one++;
                else                                many++;
                most = max(most, piece.edges_touching);
            }
        };
        count_ends(pair.input_info, pair.deleted, pair.persisted, pair.copied, pair.most_landings);
        count_ends(pair.output_info, pair.invented, pair.from_one, pair.from_many, pair.most_sources);
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
                               const vector<edge>& landing_here,
                               const vector<object>& their_objects,
                               const vector<object>& our_objects,
                               const vector<object_info>& ours,
                               const vector<edge>& leaving_here,
                               const grid& other_grid,
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
            // Also mark the cells each piece took with it to its landing.
            for (const auto& edge : leaving_here) {
                if (edge.from < 0 || (size_t)edge.from >= ours.size()) continue;
                cells_it_took(edge, our_objects[edge.from], other_grid,
                              [&](int row, int col) { carried[row][col] = true; });
            }
            // Which object each cell belongs to, so a cell can be asked about
            // the object it is part of.
            int16_t object_at[MAX_GRID_DIM][MAX_GRID_DIM];
            for (auto& row : object_at) ranges::fill(row, (int16_t)-1);
            for (size_t i = 0; i < ours.size() && i < our_objects.size(); ++i)
                our_objects[i].each_cell([&](int row, int col, int8_t) { object_at[row][col] = (int16_t)i; });
            one_grid.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int color) {
                if (carried[row][col]) return;
                if (object_at[row][col] >= 0 && ours[object_at[row][col]].instance_of >= 0) return;
                // An uncovered cell is created or removed if its object has no partner.
                const int16_t which = object_at[row][col];
                const bool nobody_claims_the_object =
                    which < 0 || !has_a_partner[which];
                pair.left_over.push_back({ from_input, (int8_t)row, (int8_t)col,
                                           nobody_claims_the_object, which, -1, (int8_t)color });
            });
        };
        unexplained(output, forward, input_objects, output_objects,
                    pair.output_info, backward, input, false);
        unexplained(input, backward, output_objects, input_objects,
                    pair.input_info, forward, output, true);

        // Count output cells covered by exact matches and those left uncovered.
        bool covered[MAX_GRID_DIM][MAX_GRID_DIM] = {};
        mark_painted(forward, input_objects, output_objects, output, covered);
        output.for_each_colored_cell(TRANSPARENT, [&](int row, int col, int) {
            if (covered[row][col]) pair.exact_covered_cells++;
            else                   pair.created_cells++;
        });
    }

    // 4. Count what the final matches explain and describe the empty regions.
    // Count the matches starting from each source object.
    static void count_edges_leaving(const vector<edge>& edges,
                                    vector<object_info>& from_info) {
        for (auto& one_object : from_info) one_object.edges_touching = 0;
        for (const auto& edge : edges)
            if (edge.from >= 0 && (size_t)edge.from < from_info.size()) {
                from_info[edge.from].edges_touching++;
                from_info[edge.from].has_partner = true;
            }
    }

    // Count the matches that overlap each target object.
    static void count_edges_arriving(const vector<edge>& edges,
                                const vector<object>& from_objects,
                                vector<object_info>& into_info,
                                const vector<object>& into_objects) {
        for (const auto& edge : edges) {
            const grid footprint = object_matching::edge_footprint(edge, from_objects, into_objects);
            for (size_t into_index = 0; into_index < into_info.size(); ++into_index) {
                // Does the landing touch this object at all?
                bool touches = false;
                into_objects[into_index].each_cell([&](int row, int col, int8_t) {
                    const int in_row = row - edge.row, in_col = col - edge.col;
                    touches = touches || (footprint.contains(in_row, in_col) &&
                                          footprint.data[in_row][in_col] != TRANSPARENT);
                });
                if (touches) {
                    into_info[into_index].edges_touching++;
                    // A shape match or an explicit target link identifies this object as a partner.
                    if (edge.same_shape || (int16_t)into_index == edge.to)
                        into_info[into_index].has_partner = true;
                }
            }
        }
    }

    // THE CELLS A PIECE TOOK WITH IT TO ITS LANDING, handed to `mark` in grid
    // coordinates. Every one when it went whole; none when some of it hangs off
    // the grid, since that part never arrived; and when it was found partly
    // behind something, only the ones its landing shows. A hidden cell is a claim
    // that something else is in front of it, and nothing here explains that
    // something or says which of the two is on top -- credited, a piece whose
    // cells had simply changed read as a piece that never changed at all.
    template <class Mark>
    static void cells_it_took(const edge& link, const object& piece, const grid& landing, Mark mark) {
        if (!link.same_shape || link.outside != 0) return;
        piece.each_cell([&](int row, int col, int8_t color) {
            if (link.hidden) {
                const auto [down, across] = posed_place(link.pose, piece.summary.width, piece.summary.height,
                                                        row - piece.summary.row, col - piece.summary.col);
                if (!landing.contains(link.row + down, link.col + across) ||
                    landing.data[link.row + down][link.col + across] != color) return;
            }
            mark(row, col);
        });
    }

    static void mark_painted(const vector<edge>& edges,
                             const vector<object>& from_objects,
                             const vector<object>& into_objects,
                             const grid& into,
                             bool covered[MAX_GRID_DIM][MAX_GRID_DIM],
                             bool exact_only = true,
                             bool same_shape_only = false) {
        for (const auto& edge : edges) {
            if (exact_only && !edge.exact) continue;
            if (same_shape_only && !edge.same_shape) continue;
            const grid footprint = object_matching::edge_footprint(edge, from_objects, into_objects);
            for (int row = 0; row < footprint.height; ++row)
                for (int col = 0; col < footprint.width; ++col)
                    if (object_matching::landing_shows(edge, footprint, into, row, col))
                        covered[edge.row + row][edge.col + col] = true;
        }
    }

    // Finds empty regions to help describe where objects appeared or disappeared.
    static void find_blanks(const grid& one_grid, vector<blank_patch>& patches,
                            int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM]) {
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < MAX_GRID_DIM; ++col) patch_of[row][col] = -1;
        int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
        for (int row = 0; row < one_grid.height; ++row)
            for (int col = 0; col < one_grid.width; ++col) {
                if (one_grid.data[row][col] != TRANSPARENT) continue;
                if (patch_of[row][col] >= 0) continue;
                const int16_t mine = (int16_t)patches.size();
                cell_box box;
                bool open = false;
                queue_row[0] = (int16_t)row; queue_col[0] = (int16_t)col;
                patch_of[row][col] = mine;
                const int cells = walk_out<4>(one_grid.height, one_grid.width, queue_row, queue_col, 1,
                    [&](int next_row, int next_col) {
                        if (one_grid.data[next_row][next_col] != TRANSPARENT ||
                            patch_of[next_row][next_col] >= 0) return false;
                        patch_of[next_row][next_col] = mine;
                        return true;
                    },
                    [&](int at_row, int at_col) {
                        box.include_cell(at_row, at_col);
                        open = open || on_border(at_row, at_col, one_grid.height, one_grid.width);
                    });
                patches.push_back({ summary_of(box, cells), !open });
            }
    }

    // 5. What a kept reading's edges were taken to say, rule by rule. An edge
    // came across as it was, or needed some of the freedoms, or was found partly
    // behind something, or only sits on a piece; each rule names the input pieces
    // its edges are about and the cells on both grids they carry, counted the way
    // left over counts them. An edge that moved and was recolored counts under
    // both, so the rules can add up past the whole; left over is still what none
    // of them carried.
    static void list_what_fired(const vector<example>& train,
                                const prepared_segmentation& prepared, reading& facts) {
        constexpr int freedoms = degrees_of_freedom::FREEDOMS;
        // As it was, then each freedom in its own place, then the two that are not.
        enum : int { as_it_was, partly_hidden = 1 + freedoms, sitting_on, kinds };
        auto kinds_of = [](const pair_info& pair, const edge& link) {
            if (!link.same_shape) return 1u << sitting_on;
            const object_info* source = pair.source_of(link);
            const auto went = source ? object_matching::move_of(link, *source) : std::pair<int, int>{};
            // In the order degrees_of_freedom numbers them: move, turn, color, hang off, scale.
            const bool needed[freedoms] = { went.first != 0 || went.second != 0, link.pose != 0,
                                            !link.exact, link.outside > 0, link.scale != 1 };
            unsigned said = link.hidden ? 1u << partly_hidden : 0u;
            for (int freedom = 0; freedom < freedoms; ++freedom)
                if (needed[freedom]) said |= 1u << (1 + freedom);
            return said ? said : 1u << as_it_was;
        };
        int pieces[kinds] = {}, cells[kinds] = {};
        bool fired[kinds] = {};
        for (size_t which = 0; which < facts.pairs.size(); ++which) {
            if (!prepared.pairs[which].read) continue;
            const pair_info& pair = facts.pairs[which];
            const grid* on_grid[2] = { &train[which].input, &train[which].output };
            const vector<object>* objects[2] = { &prepared.pairs[which].input_objects,
                                                 &prepared.pairs[which].output_objects };
            // Per kind, the cells its edges carry on each grid, and the input pieces they name.
            bool carried[kinds][2][MAX_GRID_DIM][MAX_GRID_DIM] = {};
            vector<vector<bool>> named(kinds, vector<bool>(pair.input_info.size(), false));
            for (const auto& link : pair.edges) {
                const unsigned said = kinds_of(pair, link);
                const int from = link.from_input ? 0 : 1, into = 1 - from;
                const int piece = pair.input_piece_of(link);
                for (int kind = 0; kind < kinds; ++kind) {
                    if (!(said >> kind & 1)) continue;
                    fired[kind] = true;
                    // Where it landed, as far as the grid shows it, and the piece it
                    // left when it went whole -- the two things left over credits.
                    mark_painted({ link }, *objects[from], *objects[into], *on_grid[into],
                                 carried[kind][into], false, true);
                    cells_it_took(link, (*objects[from])[link.from], *on_grid[into],
                                  [&](int row, int col) { carried[kind][from][row][col] = true; });
                    if (piece >= 0 && (size_t)piece < named[kind].size()) named[kind][piece] = true;
                }
            }
            for (int kind = 0; kind < kinds; ++kind) {
                for (int side = 0; side < 2; ++side)
                    on_grid[side]->for_each_colored_cell(TRANSPARENT, [&](int row, int col, int) {
                        cells[kind] += carried[kind][side][row][col]; });
                pieces[kind] += (int)count(named[kind].begin(), named[kind].end(), true);
            }
        }
        auto name_of = [](int kind) -> string {
            if (kind == as_it_was) return degrees_of_freedom::from_index(0).label();
            if (kind == partly_hidden) return "Partly Hidden";
            if (kind == sitting_on) return sits_on_name;
            return degrees_of_freedom::freedom_names[kind - 1];
        };
        facts.fired.clear();
        facts.line.clear();
        for (int kind = 0; kind < kinds; ++kind)
            if (fired[kind]) facts.fired.push_back({ name_of(kind), pieces[kind], cells[kind] });
    }

    // 6. THE ACTS THE CUT'S OWN LINE SAYS: what happened to the pieces of one
    // color, the same way in every pair -- turned, moved down or across,
    // recolored -- each listed under the freedom it needed and said on the line
    // in the order it was read. A line is a selection and what happened to it,
    // so one piece that turns AND moves AND changes colour is several acts.
    //
    // EACH ACT STANDS ON ITS OWN EVIDENCE. Asked as one key, a turn that differs
    // between pairs took the colour down with it -- 017c7c7b recolours 1 to 2 in
    // every pair and says so on every edge, and the reading lost it because the
    // poses read 0, 5 and 5. Turn, move and colour are three questions, so a
    // disagreement about one says nothing about the others.
    //
    // AN ACT MUST HOLD IN EVERY PAIR, or it is one example's accident.
    static void read_the_acts(const vector<example>& train,
                              const prepared_segmentation& prepared, reading& facts) {
        struct seen_act {
            int8_t color, what, value;
            // GCC 11 (Kaggle) crashes when ranges instantiates a local class's
            // defaulted equality. Spell out the same three comparisons.
            bool operator==(const seen_act& other) const {
                return color == other.color && what == other.what && value == other.value;
            }
        };
        // Which piece of which pair each act was seen on, to count what it covers.
        struct seen_on { seen_act act; size_t pair; int piece; };
        vector<seen_act> acts_in_every_pair;
        vector<seen_on> where;
        for (size_t which = 0; which < facts.pairs.size(); ++which) {
            const auto& pair = facts.pairs[which];
            vector<seen_act> here;
            auto note = [&](int8_t color, fired_rule::said_as what, int8_t value, int piece) {
                if (value == 0 && what != fired_rule::recolor) return;   // nothing happened
                const seen_act act{ color, (int8_t)what, value };
                where.push_back({ act, which, piece });
                if (ranges::find(here, act) == here.end()) here.push_back(act);
            };
            // A piece found at another size is no act's: none of them grows it.
            for (const auto& edge : pair.edges) {
                const auto* source = pair.source_of(edge);
                if (!edge.from_input || !edge.same_shape || edge.scale != 1 || !source) continue;
                const auto& piece = *source;
                note(piece.color, fired_rule::turn, edge.pose, edge.from);
                // A copy's landing is not a movement of the single original. Asked
                // as places and not edges: a piece found from both grids has two
                // edges for its one landing (places_went), and read as a copy, so
                // not one move was ever said.
                if (pair.places_went(edge.from) == 1) {
                    note(piece.color, fired_rule::move_down,   (int8_t)(edge.row - piece.summary.row), edge.from);
                    note(piece.color, fired_rule::move_across, (int8_t)(edge.col - piece.summary.col), edge.from);
                }
                // A color that became empty was deleted, which no recolor says.
                if (!edge.exact)
                    for (int8_t was = 0; was < COLOR_COUNT; ++was)
                        if (edge.color_map.changes(was) && is_color(edge.color_map[was]))
                            note(was, fired_rule::recolor, edge.color_map[was], edge.from);
            }
            if (which == 0) { acts_in_every_pair = here; continue; }
            erase_if(acts_in_every_pair, [&](const seen_act& act) {
                return ranges::find(here, act) == here.end(); });
        }

        // AN ACT IS SAID OF EVERY PIECE IT PICKS. The line picks by color -- every
        // piece that holds a cell of it, or every piece there is for a piece of
        // several colors -- so an act that happened to some of them and not the
        // rest says something false of the rest, and is not the act. Each picked
        // piece must have been carried, and carried that way: turned by that pose,
        // moved that far and only once, or recolored so that all of it wears the
        // new color, since Recolor paints a piece whole.
        auto holds_a_cell_of = [&](size_t which, int piece, int color) {
            return (palette(prepared.pairs[which].input_objects[piece]) >> color & 1) != 0;
        };
        auto said_of_every_piece = [&](const seen_act& act) {
            for (size_t which = 0; which < facts.pairs.size(); ++which) {
                const auto& pair = facts.pairs[which];
                for (int piece = 0; piece < (int)pair.input_info.size(); ++piece) {
                    if (act.color >= 0 && !holds_a_cell_of(which, piece, act.color)) continue;
                    const auto& box = pair.input_info[piece].summary;
                    bool carried = false;
                    for (const auto& edge : pair.edges) {
                        if (!edge.from_input || !edge.same_shape || edge.from != piece) continue;
                        carried = true;
                        if (edge.scale != 1) return false;
                        bool that_way = true;
                        switch (act.what) {
                            case fired_rule::turn:
                                that_way = edge.pose == act.value;
                                break;
                            case fired_rule::move_down:
                                that_way = pair.places_went(piece) == 1 && edge.row - box.row == act.value;
                                break;
                            case fired_rule::move_across:
                                that_way = pair.places_went(piece) == 1 && edge.col - box.col == act.value;
                                break;
                            default:
                                prepared.pairs[which].input_objects[piece].each_cell([&](int, int, int8_t cell) {
                                    that_way = that_way && edge.color_map.apply(cell) == act.value; });
                                break;
                        }
                        if (!that_way) return false;
                    }
                    if (!carried) return false;
                }
            }
            return true;
        };
        erase_if(acts_in_every_pair, [&](const seen_act& act) { return !said_of_every_piece(act); });

        // EVERY CHANGE IS WRITTEN DOWN BEFORE ANY ACT IS SAID, cell by cell, so
        // an act accounts for what it says by claiming those cells, and whatever
        // no rule claims is what the reading has no word for.
        write_down_what_changed(train, prepared, facts);

        // WHAT THE ACTS SAY OF ONE PIECE. An act accounts only for the pieces it
        // picks, so one said move says nothing of a piece that went somewhere
        // else. A piece's turn is said when every pose it landed in is, and its
        // move when it went to one place and every landing is said down and
        // across. An act has to hold in every pair, so a slide that differs from
        // piece to piece leaves no act, and the line would paint the piece
        // straight back: the language missing a word, not the piece standing
        // still. Nor does any act say a landing found only from the output side,
        // or a piece drawn again where it stayed -- 06df4c85 said exactly that,
        // with nothing to show a word was missing.
        auto said_by_an_act = [&](fired_rule::said_as what, int value, size_t which, int piece) {
            return ranges::any_of(acts_in_every_pair, [&](const seen_act& act) {
                return act.what == what && act.value == value &&
                       (act.color < 0 || holds_a_cell_of(which, piece, act.color)); });
        };
        auto acts_say = [&](missing word, size_t which, int piece) {
            const auto& pair = facts.pairs[which];
            if (piece < 0 || (size_t)piece >= pair.input_info.size()) return false;
            if (word == missing::where_it_went && pair.places_went(piece) != 1) return false;
            const auto& box = pair.input_info[piece].summary;
            bool carried = false;
            for (const auto& link : pair.edges) {
                if (!link.from_input || !link.same_shape || link.from != piece) continue;
                carried = true;
                if (link.scale != 1) return false;
                if (word == missing::how_it_turned) {
                    if (link.pose != 0 && !said_by_an_act(fired_rule::turn, link.pose, which, piece)) return false;
                    continue;
                }
                const int down = link.row - box.row, across = link.col - box.col;
                if ((down != 0 && !said_by_an_act(fired_rule::move_down, down, which, piece)) ||
                    (across != 0 && !said_by_an_act(fired_rule::move_across, across, which, piece)))
                    return false;
            }
            return carried;
        };

        const auto& freedom = degrees_of_freedom::freedom_names;
        const int8_t moves = facts.fired_named(freedom[0]), turns = facts.fired_named(freedom[1]),
                     colors = facts.fired_named(freedom[2]);
        auto what_it_did = [](const seen_act& act) -> string {
            switch (act.what) {
                case fired_rule::turn:        return "turns, pose " + to_string(act.value);
                case fired_rule::move_down:   return act.value > 0 ? "moves " + to_string(act.value) + " down"
                                                                   : "moves " + to_string(-act.value) + " up";
                case fired_rule::move_across: return act.value > 0 ? "moves " + to_string(act.value) + " right"
                                                                   : "moves " + to_string(-act.value) + " left";
                default:                      return "becomes " + to_string(act.value);
            }
        };
        for (const auto& act : acts_in_every_pair) {
            fired_rule rule;
            rule.name = act.color < 0 ? string("Every Piece") : "Color " + to_string(act.color);
            rule.says = what_it_did(act);
            rule.as = (fired_rule::said_as)act.what;
            rule.pick = act.color < 0 ? piece_pick::all() : piece_pick::of_color(act.color);
            rule.value = act.value;
            // The pieces it was seen on, each once.
            vector<std::pair<size_t, int>> counted;
            for (const auto& seen : where) {
                if (seen.act != act || ranges::find(counted, std::pair{ seen.pair, seen.piece }) != counted.end())
                    continue;
                counted.push_back({ seen.pair, seen.piece });
                ++rule.pieces;
            }
            // The cells it accounts for. A turn or a move: those of the pieces it
            // picks, once the acts say their turn or their move whole. A recolor:
            // every cell that was its color and came out in its value, on either
            // grid, since a recolor is said of colors.
            const missing word = act.what == fired_rule::turn    ? missing::how_it_turned
                               : act.what == fired_rule::recolor ? missing::a_recolor
                                                                 : missing::where_it_went;
            const int8_t under = act.what == fired_rule::turn    ? turns
                               : act.what == fired_rule::recolor ? colors : moves;
            facts.say_fired(std::move(rule), under, [&](size_t which, const left_over_cell& cell) {
                if (!cell.waits_for(word)) return false;
                if (word == missing::a_recolor)
                    return cell.from_input ? cell.color == act.color && cell.other == act.value
                                           : cell.color == act.value && cell.other == act.color;
                return acts_say(word, which, cell.about) &&
                       (act.color < 0 || holds_a_cell_of(which, cell.about, act.color));
            });
        }
    }

    // 7. EVERY CHANGE THE MATCHES FOUND, CELL BY CELL, in the colors both grids
    // show: a carried piece that went somewhere, was drawn again or turned; a
    // cell that came out in another color, or was drawn where the input had
    // ground; a piece that only sits on its landing. Written down whether or not
    // anything says it yet: each cell waits on the word for its change, a rule
    // that says the change claims it when it is said, and what none claims is
    // what the reading has no word for. A cell waits on each word once.
    static void write_down_what_changed(const vector<example>& train,
                                        const prepared_segmentation& prepared, reading& facts) {
        for (size_t which = 0; which < facts.pairs.size(); ++which) {
            if (!prepared.pairs[which].read) continue;
            auto& pair = facts.pairs[which];
            // Both grids and their pieces, by side: [true] is the input's.
            const grid* grids[2] = { &train[which].output, &train[which].input };
            const vector<object>* pieces[2] = { &prepared.pairs[which].output_objects,
                                                &prepared.pairs[which].input_objects };
            int16_t piece_at[2][MAX_GRID_DIM][MAX_GRID_DIM];
            for (const bool side : { false, true }) {
                for (auto& row : piece_at[side]) ranges::fill(row, (int16_t)-1);
                for (size_t i = 0; i < pieces[side]->size(); ++i)
                    (*pieces[side])[i].each_cell([&](int row, int col, int8_t) {
                        piece_at[side][row][col] = (int16_t)i; });
            }
            bool written[(int)missing::count][2][MAX_GRID_DIM][MAX_GRID_DIM] = {};
            auto wait = [&](missing word, bool from_input, int row, int col, int about, int8_t other) {
                const grid& on = *grids[from_input];
                if (!on.contains(row, col)) return;
                const int8_t color = (int8_t)on.data[row][col];
                // An empty cell, where the ground was, is no cell of anything.
                if (!is_color(color)) return;
                bool& done = written[(int)word][from_input][row][col];
                if (done) return;
                done = true;
                left_over_cell cell;
                cell.from_input = from_input;
                cell.row = (int8_t)row;
                cell.col = (int8_t)col;
                cell.piece = piece_at[from_input][row][col];
                cell.color = color;
                cell.waits = (int8_t)word;
                cell.about = (int16_t)about;
                cell.other = other;
                pair.left_over.push_back(cell);
            };
            // The cells one edge speaks for: every cell of the piece it leaves,
            // on its own grid, whatever its landing shows -- a piece found partly
            // behind something still went there whole -- then both ends of every
            // cell the landing does show. `visit(from_input, row, col, was, now)`
            // sees each, `was` its color on the input and `now` on the output.
            auto each_end = [&](const edge& link, auto&& visit) {
                const auto& from = *pieces[link.from_input];
                if (link.from < 0 || (size_t)link.from >= from.size()) return;
                from[link.from].each_cell([&](int row, int col, int8_t color) {
                    const int8_t mapped = (int8_t)link.color_map.apply(color);
                    if (link.from_input) visit(true, row, col, color, mapped);
                    else                 visit(false, row, col, mapped, color);
                });
                each_carried_cell(link, from[link.from], *grids[!link.from_input],
                                  [&](int in_row, int in_col, int8_t was, int out_row, int out_col, int8_t now) {
                    visit(true, in_row, in_col, was, now);
                    visit(false, out_row, out_col, was, now);
                });
            };

            // A CARRIED PIECE: where it went, where it was drawn again, how it
            // turned, how much bigger or smaller it came out, read off every
            // landing it has -- one found only from the output side moved, turned
            // or grew it as surely.
            for (int piece = 0; piece < (int)pair.input_info.size(); ++piece) {
                const auto& box = pair.input_info[piece].summary;
                bool carried = false, moved = false, turned = false, scaled = false;
                bool landed_elsewhere = false, landed_turned = false;
                for (const auto& link : pair.edges) {
                    if (!link.same_shape || pair.input_piece_of(link) != piece) continue;
                    scaled = scaled || link.scale != 1;
                    if (link.from_input) {
                        carried = true;
                        moved = moved || link.row != box.row || link.col != box.col;
                        turned = turned || link.pose != 0;
                        continue;
                    }
                    const auto went_by = object_matching::move_of(link, *pair.source_of(link));
                    landed_elsewhere = landed_elsewhere || went_by.first != 0 || went_by.second != 0;
                    landed_turned = landed_turned || link.pose != 0;
                }
                const bool copied = carried && pair.places_went(piece) > 1;
                const bool went = carried ? moved && !copied : landed_elsewhere;
                if (!carried) turned = landed_turned;
                const bool happened[] = { facts.allowed.translate && went, facts.allowed.translate && copied,
                                          facts.allowed.pose && turned, facts.allowed.scale && scaled };
                const missing words[] = { missing::where_it_went, missing::where_it_was_copied,
                                          missing::how_it_turned, missing::how_it_was_scaled };
                for (const auto& link : pair.edges) {
                    if (!link.same_shape || pair.input_piece_of(link) != piece) continue;
                    each_end(link, [&](bool from_input, int row, int col, int8_t was, int8_t now) {
                        for (int i = 0; i < (int)size(words); ++i)
                            if (happened[i]) wait(words[i], from_input, row, col, piece, from_input ? now : was);
                    });
                }
            }

            // A LANDING FOUND FROM THE OUTPUT SIDE THAT NAMES NO INPUT PIECE --
            // partly hidden, hanging off the grid, or fitting two -- still says
            // what is drawn there came from somewhere else. Found elsewhere or
            // turned, its cells wait on the word for that as surely: 18286ef8's 9,
            // found one step up and to the left and partly hidden, was otherwise
            // said by nothing, and the line drew it nowhere. Both ends of it, as
            // every edge's (each_end): the cells it came from on the input, which
            // the matching counts as carried, wait on that word too where they
            // show the piece itself, so where no word says it they are left over
            // (4364c1c4's last pass: its block, found inside a piece of nine
            // cells and a step to the left, left the column it stood on drawn
            // where the output shows the ground, and no cell said so). A cell it
            // was found over in another color waits on the recolor as before,
            // not the copy (b8825c91's 4s, the hidden parts of its picture found
            // in them, would have joined the colors its copies are picked by).
            for (const auto& link : pair.edges) {
                if (link.from_input || !link.same_shape || pair.input_piece_of(link) >= 0) continue;
                const object_info* source = pair.source_of(link);
                if (!source || link.from < 0 || (size_t)link.from >= pieces[false]->size()) continue;
                const auto [down, across] = object_matching::move_of(link, *source);
                const bool copied = facts.allowed.translate && (down != 0 || across != 0);
                const bool turned = facts.allowed.pose && link.pose != 0;
                if (!copied && !turned) continue;
                each_end(link, [&](bool from_input, int row, int col, int8_t was, int8_t now) {
                    if (from_input && was != now) return;
                    if (copied) wait(missing::where_it_was_copied, from_input, row, col, -1, now);
                    if (turned) wait(missing::how_it_turned, from_input, row, col, -1, now);
                });
            }

            // A CELL THAT CAME OUT IN ANOTHER COLOR, read from either grid. Where
            // the input's own edge already speaks for the piece, it is that edge
            // that is asked. A cell the input had as ground was drawn there, and
            // no piece held it to recolor: that is a word of its own. A piece
            // recolored with no color named waits on a recolor all the same, one
            // no act or swap can say, since neither color is known.
            for (const auto& link : pair.edges) {
                if (!link.same_shape || link.exact) continue;
                const int piece = pair.input_piece_of(link);
                if (!link.from_input && piece >= 0 && ranges::any_of(pair.edges, [&](const edge& other) {
                        return other.from_input && other.same_shape && !other.exact && other.from == piece; }))
                    continue;
                bool named = false;
                each_end(link, [&](bool from_input, int row, int col, int8_t was, int8_t now) {
                    if (was == now) return;
                    named = true;
                    if (was != TRANSPARENT)
                        wait(missing::a_recolor, from_input, row, col, piece, from_input ? now : was);
                    else if (!from_input)
                        wait(missing::drawn_on_the_ground, false, row, col, piece, was);
                });
                if (!named)
                    each_end(link, [&](bool from_input, int row, int col, int8_t, int8_t) {
                        if (from_input == link.from_input) wait(missing::a_recolor, from_input, row, col, piece, -1);
                    });
            }

            // A PIECE THAT ONLY SITS ON ITS LANDING changed its shape, and
            // nothing says how: both pieces, every cell but those the other grid
            // shows in the same color at the same place, which stayed as they
            // were and are no change at all (5582e5ca's 9s, where every other
            // color around them turned 9).
            for (const auto& link : pair.edges) {
                if (link.same_shape) continue;
                const int piece = pair.input_piece_of(link);
                for (const bool side : { link.from_input, !link.from_input }) {
                    const int index = side == link.from_input ? link.from : link.to;
                    if (index < 0 || (size_t)index >= pieces[side]->size()) continue;
                    const grid& other = *grids[!side];
                    (*pieces[side])[index].each_cell([&](int row, int col, int8_t color) {
                        if (other.contains(row, col) && other.data[row][col] == color) return;
                        wait(missing::how_it_changed_shape, side, row, col, piece, -1); });
                }
            }

            // A CELL RECOLORED WHERE IT LIES, however the matching read it: the
            // other grid shows another color at its very place. Noted on every
            // cell left over there, so a recolor said of that place can claim it.
            if (grids[false]->same_size(*grids[true]))
                for (auto& cell : pair.left_over) {
                    const int8_t there = (int8_t)grids[!cell.from_input]->data[cell.row][cell.col];
                    if (is_color(there) && there != cell.color) cell.in_place = there;
                }
        }
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
            const auto [down, across] = posed_place(link.pose, piece.summary.width, piece.summary.height,
                                                    row - piece.summary.row, col - piece.summary.col);
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

};

} // namespace arc
