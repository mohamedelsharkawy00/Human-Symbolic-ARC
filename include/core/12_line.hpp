// The line: the acts and paints said after the cut's own acts, the one that
// says the most first (finish_the_line).
#pragma once
#include "11_replays.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace arc {

// A SWAP A FIRED RULE ALREADY SAYS. Every cell it changes is one a rule on the
// elected reading accounts for, so saying it too would do that work twice -- on
// 009d5c81 the 0<->1 swap only ever painted away the key ShapeColorMap uses up.
// A swap that changes nothing at all is not one this can speak for.
inline bool swap_already_claimed(const inference& inf, int variant) {
    const auto [first, second] = color_pair::from_index(variant);
    const auto& pairs = inf.correspondence.chosen_pairs();
    const auto& lost = inf.color_diff.lost;
    bool changes_anything = false;
    for (size_t which = 0; which < pairs.size() && which < lost.size(); ++which)
        for (const int color : { first, second })
            for (const auto& [row, col] : lost[which][color]) {
                changes_anything = true;
                if (!ranges::any_of(pairs[which].left_over, [&](const auto& cell) {
                        return cell.from_input && cell.row == row && cell.col == col &&
                               cell.claimed_by >= 0; }))
                    return false;
            }
    return changes_anything;
}

// Each rule belongs to one group. Acts change the hand; paints draw from the
// cut again; whole-program rules have their own spelling. The act groups say
// which later picks remain usable. No group carries another group's fields.
enum class rule_group { unsaid, recolors, changes_pieces, removes, paints, whole_program };
inline constexpr rule_group group_of(correspondence_election::fired_rule::said_as kind) {
    using rule = correspondence_election::fired_rule;
    switch (kind) {
        // Acts: only the colors change.
        case rule::recolor: case rule::swap_colors: case rule::size_rank:
        case rule::color_from: case rule::recolor_picked: case rule::shape_recolor: case rule::key_swap:
            return rule_group::recolors;
        // Acts: pieces move, turn, change shape, or gain copies -- the learned
        // tables too: the fill by shape adds each piece's hollows, and the key
        // whose shape names the figure's color goes.
        case rule::turn: case rule::move_down: case rule::move_across:
        case rule::learned_table: case rule::align_to: case rule::mirror_around:
        case rule::outlined: case rule::to_edge: case rule::shifted: case rule::move_toward:
        case rule::empty_swap:
            return rule_group::changes_pieces;
        case rule::removed: return rule_group::removes;
        // Paints: the hand on the line is left alone.
        case rule::extend_rays: case rule::bridged: case rule::filled: case rule::trailed:
        case rule::boxed: case rule::filled_room: case rule::copied_onto: case rule::tiled_along:
        case rule::repeated_past: case rule::turned_copy: case rule::symmetric:
            return rule_group::paints;
        // These rules redraw the whole output and spell their own programs.
        case rule::diagonal_tile: case rule::piece_slam: case rule::corner_chain:
        case rule::slide_to_touch: case rule::project_on_seeds: case rule::snap_to_anchors:
        case rule::own_sentences: return rule_group::whole_program;
        case rule::unsaid: case rule::kind_count: return rule_group::unsaid;
    }
    return rule_group::unsaid;
}
static_assert([] {
    using rule = correspondence_election::fired_rule;
    for (int kind = rule::unsaid + 1; kind < rule::kind_count; ++kind)
        if (group_of((rule::said_as)kind) == rule_group::unsaid) return false;
    return true;
}(), "every rule belongs to a group; duplicate entries fail the switch");

// Rays keep their length below 32, plus 32 for the diagonals. This is their
// stored value, not the argument written in the DSL.
struct paint_value {
    static constexpr int second_bit = 32;
    static constexpr int packed(bool second, int number) { return (second ? second_bit : 0) + number; }
    static constexpr bool second(int stored) { return stored >= second_bit; }
    static constexpr int number(int stored) { return stored % second_bit; }
};

// The one pose that does `first` and then `then`. Read off where the corners of
// a square land, since the eight ways round are exactly the ways to move them.
inline int compose_poses(int first, int then) {
    for (int pose = 0; pose < POSE_COUNT; ++pose) {
        bool same = true;
        for (int corner = 0; corner < 4 && same; ++corner) {
            const auto [row, col] = posed_place(first, 2, 2, corner / 2, corner % 2);
            same = posed_place(then, 2, 2, row, col) == posed_place(pose, 2, 2, corner / 2, corner % 2);
        }
        if (same) return pose;
    }
    return 0;
}

// THE CUT'S PAINTS, READ ONCE FOR EVERY LINE READ ON IT: what a paint reads
// is the cut opened again, the same for each line read on one cut -- with its
// acts or the rules alone, plainly or reading ahead (read_the_cut) -- so the
// cut is opened once, the pieces each draw pick takes off it handed over once,
// and each way of asking a paint -- the kind, the pick, the value -- walked
// once (paint_walk) for all of them. A walk is counted once for each canvas
// it is laid on (walk_tally), each canvas of every line its own number.
// `asked_picks` are the draw picks a paint asks, in order: each takes, in some
// pair, pieces no pick before it takes (takes_the_same); `asked_partner_picks`
// the partner picks past them, which a line asks last (partner_picks). The
// rays of each pick, one way round, are fanned out once for every length
// (rays_fan).
struct walked_paint { optional<paint_walk> walk; int counted_on = -1; walk_tally tally; };
struct cut_paints {
    optional<replay_facts> opened;
    vector<piece_pick> draw_picks;
    vector<vector<partner_pieces>> handed_by_pick;
    vector<vector<uint16_t>> colors_by_pick;
    // The draw picks asked, and past them the partner picks a line asks last
    // (partner_picks): indices into draw_picks.
    vector<size_t> asked_picks, asked_partner_picks;
    size_t plain_draws = 0;
    std::map<std::tuple<int, size_t, int>, walked_paint> walks;
    std::map<std::pair<size_t, bool>, optional<ray_fan>> fans;
    int canvases = 0;
};

// THE REST OF THE ELECTED CUT'S LINE, after its own acts: a rule read off the
// cut's own pieces that drew every output by itself, or else the rules its
// recolor is said with and what it has no rule for. Listed on the reading under
// the rule whose evidence each one reads, and put on the line in the order it is
// said. Reading ahead, a paint is weighed with what the picture lets be read
// right after it (read_figures asks both ways).
inline void finish_the_line(inference& inf, const vector<example>& train, const tiled_pieces& boundaries,
                            bool read_ahead, cut_paints& paints) {
    ++inference_work.lines;
    const time_into clock{ inference_time.ms[inf.pass > 1 ? inference_times::lines_later : inference_times::lines_first] };
    using rule = correspondence_election::fired_rule;
    auto& election = inf.correspondence;
    if (election.chosen_pairs().empty()) return;
    auto& read = election.readings[election.chosen];
    const int8_t colors = read.fired_named(degrees_of_freedom::freedom_names[2]);

    // A RULE THAT DREW EVERY OUTPUT BY ITSELF from the cut's own pieces is the
    // whole rest of the line: the acts have nothing left to explain, so they stay
    // listed and are not said, and it accounts for every cell there is. Asked in
    // this order, the first that holds.
    auto whole_line = [&](bool holds, rule::said_as as, const char* name, const char* says) {
        if (!holds) return false;
        read.line.clear();
        read.say_fired(listed(name, says, as), -1, [](size_t, const auto&) { return true; });
        return true;
    };
    // A pass after the first says one change in one sentence, and these spell
    // programs of their own, some reading the input again; the first pass alone
    // asks them.
    if (inf.pass == 1 &&
        (whole_line(inf.diagonal_continuation.holds(), rule::diagonal_tile, "Diagonal Tile",
                   "the colors cycle along the diagonals") ||
        whole_line(inf.piece_slam.holds(), rule::piece_slam, "Slam", "each piece piled up inside its own box") ||
        whole_line(inf.overlap_corner.holds(), rule::corner_chain, "Corner Chain",
                   "each piece hung off the last one's corner") ||
        whole_line(inf.slide_to_touch.holds(), rule::slide_to_touch, "Slide To Touch",
                   "pieces slid up against the ones that stay") ||
        whole_line(inf.project_on_seeds.holds(), rule::project_on_seeds, "Project On Seeds",
                   "the template stamped along every marker") ||
        whole_line(inf.snap_to_anchors.holds(), rule::snap_to_anchors, "Snap To Anchors",
                   "each body turned and set down on the anchors its keys match")))
        return;
    using missing = correspondence_election::missing;
    const auto& shapes = inf.shape_color_map;
    const bool recolors = read.allowed.recolor && ranges::any_of(read.pairs, [](const auto& pair) {
        return pair.found_recolored > 0; });

    // THE KEY'S SHAPE NAMES THE TARGET'S COLOR. The rule read the elected
    // reading's recolor, so it says so on that reading, under the recolor --
    // freedom 2, Color -- accounting for the target's recolored cells and for
    // the keys it used up: their cells are the ones the reading had left over
    // as gone. Said ahead of the swaps, since a swap is only said where no rule
    // already accounts for what it would change.
    const bool by_shape = recolors && shapes.holds() && shapes.claimed_by == shape_color_map_rule::body_recolor;
    if (by_shape) {
        rule one = listed("Shape Picks Color", "the shape of the key names the color", rule::learned_table);
        one.pieces = (int)shapes.read.size();
        // Each pair's key off the cut the matching numbered it in, asked for its
        // own cells, which another piece may share.
        vector<object> keys;
        for (size_t which = 0; which < shapes.read.size() && which < train.size(); ++which) {
            const vector<object> cut = pieces_of(train[which].input, election.segmentation, boundaries.of(train[which]),
                                                 &election.shape);
            const int key = shapes.read[which].key;
            keys.push_back(key >= 0 && (size_t)key < cut.size() ? cut[(size_t)key] : object{});
        }
        read.say_fired(std::move(one), colors, [&](size_t which, const auto& cell) {
            if (which >= keys.size()) return false;
            return (cell.from_input && cell.waits < 0 && keys[which].holds(cell.row, cell.col)) ||
                   (cell.waits_for(missing::a_recolor) && cell.about == shapes.read[which].target);
        });
    }

    // A swap is said over the pieces, so it only has to hold where a piece is.
    // Candidates that fail even that remain available to search.
    //
    // AND ONLY WHERE THE READING LET A COLOUR CHANGE AT ALL. A swap IS a
    // recolour, so a reading that matched every piece without one has already
    // said these pieces kept their colours -- and over_the_pieces is in no
    // position to argue, because the cells it reads are the ones a piece LEFT.
    // 05f2a901 is the puzzle that shows it: the shape slides up against the
    // block, [Move] matches all six pieces exactly, 0 recoloured, and 0<->2
    // holds over the pieces for one reason only -- a shape that moves away
    // leaves ground where it stood and covers ground where it lands, which
    // reads cell for cell as the two trading places. Said, it paints the shape
    // away and the reading draws an empty grid, while its own list has already
    // named the thing it cannot say: where each piece went.
    // Nor where a fired rule already accounts for every cell it would change.
    // A swap with the ground cannot come up: the plan deleted it, so no piece
    // wears it (aabf363d's marker was painted away by one, before the shape could
    // take its color). It accounts for every recolored cell that traded one of
    // its two colors for the other.
    // Nor a swap with a color a rule already said paints: the swap would trade
    // back what that rule laid down (9565186b: 8 recolored to 5, then 1 and 5
    // trading places turns those 5s to 1s, while each rule alone claimed its
    // cells). A swap paints both its colors, so a second swap of either one
    // trades back what the first laid down (9565186b again: 1 and 5, then 2
    // and 5, turns the 5s the first laid down to 2s).
    auto painted_by_the_line = [&](int color) {
        return ranges::any_of(read.fired, [&](const auto& one) {
            if (one.as == rule::swap_colors) {
                const auto [traded, other] = color_pair::from_index(one.value);
                return traded == color || other == color;
            }
            return ((one.as == rule::recolor || one.as == rule::recolor_picked) && one.value == color) ||
                   one.paint == color;
        });
    };
    if (read.allowed.recolor)
        for (int variant = 0; variant < COLOR_PAIR_COUNT; ++variant)
            if (const auto [first, second] = color_pair::from_index(variant);
                inf.color_swaps.over_the_pieces[variant] &&
                !painted_by_the_line(first) && !painted_by_the_line(second) &&
                !swap_already_claimed(inf, variant)) {
                const string says = to_string(first) + " and " + to_string(second) + " trade places";
                read.say_fired(listed("Swap", says.c_str(), rule::swap_colors, (int8_t)variant), colors,
                               [&](size_t, const auto& cell) {
                    return cell.waits_for(missing::a_recolor) &&
                           ((cell.color == first && cell.other == second) ||
                            (cell.color == second && cell.other == first));
                });
            }

    // THE HAND AS THE LINE LEAVES IT: the cut's pieces carried through every rule
    // already on the line, the hand the rules asked below are replayed on, laid
    // under what is down where a piece sits on another, as the line is said.
    replay_facts facts(inf, train, boundaries);
    picks_on_hand hand_picks;
    facts.laid_under = ranges::any_of(read.pairs, [](const auto& pair) { return pair.sits_on > 0; });
    for (const int8_t which : read.line)
        facts.hand_known = facts.hand_known && carry_the_hand(read.fired[which], inf, facts);
    // Carries the hand through a rule just said on the line. Once it cannot,
    // the line no longer draws its picture, and no rule after lays a cell wrong
    // (the sure rules first, below).
    auto carry = [&](int8_t which) {
        hand_picks.clear();
        facts.hand_known = facts.hand_known && carry_the_hand(read.fired[which], inf, facts);
    };

    // Validate the derived sequence on a separate hand before saying any of it.
    // The existing replays check each rule; they do not choose its pick or value.
    if (const auto rules = size_changes_from_matches(facts, read); !rules.empty()) {
        replay_facts checked = facts;
        vector<replayed> seen;
        for (const rule& one : rules) {
            const auto handed = handed_pieces(checked, one.pick);
            auto replay = one.as == rule::removed ? removal_holds(checked, handed)
                : picked_recolor_holds(checked, handed, one.value);
            if (!replay || !carry_the_hand(one, inf, checked)) break;
            seen.push_back(std::move(*replay));
        }
        if (seen.size() == rules.size())
            for (size_t index = 0; index < rules.size(); ++index) {
                rule one = rules[index];
                one.pieces = seen[index].pieces;
                const bool removes = one.as == rule::removed;
                const cell_marks claimed(removes ? seen[index].left : seen[index].drawn);
                carry(read.say_fired(std::move(one), removes ? -1 : colors,
                    [removes, claimed](size_t which, const auto& cell) {
                        return (removes ? cell.from_input : cell.answers_to(missing::a_recolor)) &&
                            on_cells(claimed, which, cell.row, cell.col);
                    }));
            }
    }

    // THE HOLLOWS FILLED BY SHAPE, a step of its own: it paints space the pieces
    // enclose, not the pieces, so it is no rule for a recolor. Said where a replay
    // on the hand shows every fill it paints, and it accounts for those cells.
    if (shapes.holds() && shapes.claimed_by == shape_color_map_rule::interior_fill && facts.hand_known)
        if (const auto filled = interior_fills_hold(facts, shapes)) {
            rule one = listed("Shape Picks Fill", "the shape of a piece names the color inside it",
                              rule::learned_table);
            one.pieces = filled->pieces;
            carry(read.say_fired(std::move(one), -1, [&](size_t which, const auto& cell) {
                return !cell.from_input && on_cells(filled->drawn, which, cell.row, cell.col);
            }));
        }

    // A RECOLOR THE ACTS DID NOT SAY, said by size rank where the palette ranks
    // the same in every output. The replay on the hand showed every piece in its
    // rank's color, so it accounts for every recolored cell.
    const bool a_recolor_was_said = ranges::any_of(read.line, [&](int8_t which) {
        return read.fired[which].as == rule::recolor; });
    if (recolors && !by_shape && !a_recolor_was_said && facts.hand_known)
        if (const auto ranked = size_rank_holds(facts, inf.colors, election.segmentation)) {
            rule one = listed("Size Picks Color", "the colors go by size", rule::size_rank);
            one.pieces = ranked->pieces;
            carry(read.say_fired(std::move(one), colors,
                                 [](size_t, const auto& cell) { return cell.waits_for(missing::a_recolor); }));
        }

    // THE RULES READ OFF THE PIECES: a piece takes its color from its partner,
    // moves level with it or by one step every pair agrees on, is drawn again
    // mirrored around a middle, is hollowed to its edge or recolored, grows rays,
    // or has its hollows filled; the gaps between pieces of one color are bridged.
    // Each is asked the same way. It picks the pieces the ways every rule picks --
    // by the colors an unsaid change starts from, by holes, by size -- and replays
    // each value it takes on every pair. Of the picks and values every pair bears
    // out, the one that accounts for the most cells still unexplained is said, a
    // tie going to the plainer: the colors before holes and size, the smaller
    // value before the larger. Where every piece of every pair is picked, it picks
    // every piece, since a test can hold a color no pair did (496994bd).
    // How a piece is like the key (SelectLikeKey), each likeness of shape in turn.
    static constexpr const char* likenesses[] = { "", " Any Way Round", " At Any Size", " At Any Size, Any Way Round",
                                                  " Stretched", " Stretched, Any Way Round" };
    static_assert(size(likenesses) == (size_t)KEY_LIKENESSES);
    // A partner by name (partner_way_names): each way partner_by finds one, and,
    // of any kind (partner_of_kind), then the piece of each color.
    auto partners = [](int way) { return string(partner_way_names[way]); };
    auto partner_named = [&](int kind) {
        return kind < (int)partner_by::count ? partners(kind)
                                             : "the piece of color " + to_string(kind - (int)partner_by::count);
    };
    // Each kind of pick by name, then turned round, every piece but those.
    static constexpr const char* pick_names[][2] = { { "Holes", "" }, { "Largest", "" }, { "Smallest", "" },
        { "Row Band", "" }, { "Column Band", "" }, { "Most Common Color", "Not Most Common Color" },
        { "Cells At Least", "Cells Fewer Than" }, { "The Largest", "All But The Largest" },
        { "The Smallest", "All But The Smallest" }, { "Row From The Middle", "Off The Row From The Middle" },
        { "Column From The Middle", "Off The Column From The Middle" },
        { "The One With No Twin By", "All But The One With No Twin By" }, { "Running", "Not Running" } };
    static_assert(size(pick_names) == (size_t)piece_pick::pick_kinds);
    static constexpr const char* running_ways[] = { "Along Its Rows", "Along Its Columns",
        "Along Its Diagonals Down To The Right", "Along Its Diagonals Up To The Right", "No Way" };
    static_assert(size(running_ways) == (size_t)runs_along::count);
    using left_over_cell = correspondence_election::left_over_cell;
    using claim = std::function<bool(size_t, const left_over_cell&)>;
    // The best a rule heard: its value and pick, how many cells nothing else
    // claims it accounts for, what its replay saw, and the cells it claims.
    struct heard {
        int value = -1, said = 0;
        piece_pick pick;
        replayed seen;
        claim claims;
        const marked_paint* marked = nullptr;
        heard() = default;
        heard(int value, int said, const piece_pick& pick, const replayed& seen, claim claims)
            : value(value), said(said), pick(pick), seen(seen), claims(std::move(claims)) {}
        heard(int value, int said, const piece_pick& pick, const marked_paint& seen, claim claims)
            : value(value), said(said), pick(pick), claims(std::move(claims)), marked(&seen) {}
        // Only the best lay of a kind needs lists, for saying it or reading
        // ahead on its drawing. Every candidate used the same lay_paint judge.
        bool materialize(const replay_facts& facts) {
            if (value < 0) return false;
            if (marked) {
                seen = *lay_paint(facts, *marked->walk, marked->how);
                marked = nullptr;
            }
            return true;
        }
    };
    // Whether the line already says `as` with this pick and value. Said again of a
    // hand that changed under it, the pick takes some other piece each time --
    // 1a07d186 removed "the third largest" four times over, peeling pieces off
    // one by one -- so it names nothing steady, and it is not asked again.
    auto already_said = [&](rule::said_as as, const piece_pick& pick, int value) {
        return ranges::any_of(read.line, [&](int8_t which) {
            const rule& step = read.fired[which];
            return step.as == as && step.value == value && step.pick == pick;
        });
    };
    // What a paint said after the line reads: the cut opened again, as it is said,
    // once for every line read on the cut (cut_paints). It lays cells wrong where
    // the rules may (replay_facts::tolerant).
    const bool cut_opened = paints.opened.has_value();
    if (!cut_opened) paints.opened.emplace(inf, train, boundaries);
    replay_facts& opened = *paints.opened;
    opened.tolerant = false;
    // THE CELLS A RULE SAID LAID WRONG, every rule's: where the picture the line
    // draws still shows them wrong at its end, they are left unsaid.
    colors_by_pair laid_wrong(facts.train.size());
    // The colors of the input cells nothing claims yet that `says` takes, a bit each.
    auto colors_left = [&](auto says) {
        uint16_t found = 0;
        for (const auto& pair : read.pairs)
            for (const auto& cell : pair.left_over)
                if (cell.claimed_by < 0 && cell.from_input && says(cell))
                    found = (uint16_t)(found | 1u << cell.color);
        return found;
    };
    // The colors an unsaid `change` starts from.
    auto starts_from = [&](missing change) {
        return colors_left([change](const auto& cell) { return cell.answers_to(change); });
    };
    // Keeps `value` when it accounts for more cells nothing claims yet than the
    // best so far, less the cells it lays wrong, with what its replay saw. A
    // finished pair (replay_facts::finished) lets every rule off, so none is
    // credited with a cell there: the cells left over in it are the line's to
    // draw (cd3c21df's framed pieces, each already its output).
    auto weigh = [&](heard& best, int value, const piece_pick& pick, const auto& seen, claim claims) {
        if (facts.any_finished())
            claims = [&facts, claims = std::move(claims)](size_t which, const left_over_cell& cell) {
                return !facts.finished(which) && claims(which, cell);
            };
        int said = -seen.wrong_cells();
        for (size_t which = 0; which < read.pairs.size(); ++which)
            for (const auto& cell : read.pairs[which].left_over)
                said += cell.claimed_by < 0 && claims(which, cell);
        if (said > best.said) best = { value, said, pick, seen, std::move(claims) };
    };
    // Says what a rule heard best, under the rule whose freedom it needs, named and
    // picking by every piece, by its colors, or by its pick of holes or size.
    auto say = [&](const heard& best, rule::said_as as, const string& says, int8_t under) {
        string name;
        const bool every_piece = best.seen.every_piece;
        if (every_piece) name = "Every Piece";
        else if (best.pick.how >= 0) {
            name = pick_names[best.pick.how][best.pick.but_those];
            const bool by_count = (best.pick.how == piece_pick::largest || best.pick.how == piece_pick::smallest)
                                      ? best.pick.value >= SIZE_RANKS
                                      : best.pick.how == piece_pick::only_one && best.pick.value >= (int)twin_by::by_count;
            if (by_count) {
                const int kind = best.pick.value - (best.pick.how == piece_pick::only_one ? (int)twin_by::by_count : SIZE_RANKS);
                name += string(best.pick.how == piece_pick::smallest ? " Fewest " : best.pick.how == piece_pick::largest
                                                                                         ? " Most " : " Count Of ") +
                        (kind == (int)count_kind::cells ? "Cells" : kind == (int)count_kind::colors ? "Colors"
                         : kind == (int)count_kind::holes ? "Holes"
                         : "Cells Of Color " + to_string(kind - (int)count_kind::of_color));
            } else if (best.pick.how == piece_pick::running_along) name += string(" ") + running_ways[best.pick.value];
            else if (best.pick.how != piece_pick::most_common) name += " " + to_string(best.pick.value);
        }
        else if (best.pick.partner >= 0) {
            // SelectPartner: the partners its base's pieces have, on a side or any.
            const int partner_direction = partner_direction_of(best.pick.partner);
            name = best.pick.but_those ? "All But " : "";
            if (best.pick.colors == piece_pick::all().colors) name += "Every Piece's";
            else {
                name += "Colors";
                for (int color = 0; color < COLOR_COUNT; ++color)
                    if (best.pick.colors >> color & 1) name += " " + to_string(color);
                name += "'";
            }
            name += string(" Partners ") + (partner_direction < 0 ? "Any Side" : direction_names[partner_direction]) + " (" +
                    partners((int)partner_way_of(best.pick.partner)) + ")";
        }
        else {
            name = best.pick.part < 0 ? string(best.pick.but_those ? "All But " : "") + "Colors"
                 : string(best.pick.but_those ? "Off Part " : "Part ") + to_string(best.pick.part) + " Fenced By";
            for (int color = 0; color < COLOR_COUNT; ++color)
                if (best.pick.colors >> color & 1) name += " " + to_string(color);
        }
        if (!every_piece && best.pick.fits >= 0)
            name += string(" That Would Fit ") + partners(best.pick.fits);
        if (!every_piece && best.pick.like >= 0)
            name += string(" Shaped Like The Key") + likenesses[best.pick.like];
        rule one = listed(name.c_str(), says.c_str(), as, (int8_t)best.value);
        one.pick = every_piece ? piece_pick::all() : best.pick;
        one.pieces = best.seen.pieces;
        one.paint = best.seen.paint;
        for (size_t which = 0; which < best.seen.wrong.size() && which < laid_wrong.size(); ++which)
            laid_wrong[which].insert(laid_wrong[which].end(), best.seen.wrong[which].begin(),
                                     best.seen.wrong[which].end());
        return read.say_fired(std::move(one), under, best.claims);
    };
    const int8_t moves = read.fired_named(degrees_of_freedom::freedom_names[0]);
    constexpr int ways = (int)partner_by::count;
    // Claims for a move: the cells the pieces left, on the input, and landed on,
    // on both grids -- an input cell a landing covers is one the move drew over
    // (18286ef8's 9 stepping onto the 5 beside it) -- however the matching read
    // them, since the replay saw each one right (ae3edfdc's pieces, some read as
    // copies, so their removal said more than either move).
    auto move_claims = [](const replayed& seen) -> claim {
        return [left = cell_marks(seen.left), landed = cell_marks(seen.drawn)](size_t which, const left_over_cell& cell) {
            return on_cells(landed, which, cell.row, cell.col) ||
                   (cell.from_input && on_cells(left, which, cell.row, cell.col));
        };
    };
    // Claims for a recolor: the cells it repainted that wait on one.
    auto recolor_claims = [](const replayed& seen) -> claim {
        return [recolored = cell_marks(seen.drawn)](size_t which, const left_over_cell& cell) {
            return cell.answers_to(missing::a_recolor) && on_cells(recolored, which, cell.row, cell.col);
        };
    };

    // THE ACTS READ OFF THE PIECES, THE ONE THAT SAYS THE MOST FIRST. Each kind --
    // a color taken from a partner, a move, a mirrored copy, an outline, a recolor
    // of a pick, a removal -- hears its best of the hand the line leaves, and the
    // kind whose best accounts for the most cells nothing claims yet is said, a tie
    // going in that order; then every kind is asked again, of the hand that act
    // leaves, since an act said late can leave the pieces as another needs them.
    // Asked in a fixed order instead, the first kind took what a later one says
    // better: 18286ef8's 9 was recolored to the 5 around it before its step toward
    // the 6 was weighed. Every act said claims cells nothing had, and each kind
    // says at most `rounds` acts, as many as the rounds it was once asked in --
    // a line of a hundred acts, each claiming a cell or two, says nothing (and a
    // reading lists at most 127 rules); where the hand could not be carried, only
    // the kinds that need no hand are asked. An offer is what a kind would say:
    // how much it accounts for, and the saying.
    // What a kind offers: how many cells it says, how to say it, and for a
    // paint, the cells it would lay, in what it lays there.
    struct offer { int said = 0; std::function<void()> say; colors_by_pair drawn; };

    // A RECOLOR ANOTHER PIECE DECIDES, READ OFF THE PAIRS: the pieces the output
    // shows in other colors where they lie (pieces_recolored); the first find
    // that colors every one of them as the output does -- or, where none does,
    // the one that colors the most of their cells -- its partner's color, the
    // key's at its place, the key's one color, or its colors cell by cell from
    // a piece of its shape (colored_right); the pieces it colors right named as
    // every pair names them (pick_for_group); and that rule replayed once. It
    // accounts for the cells it recolored.
    auto color_from_rule = [&]() -> offer {
        if (!facts.hand_known || !starts_from(missing::a_recolor)) return {};
        const vector<vector<bool>> recolored = pieces_recolored(facts);
        auto cells_of = [&](const vector<vector<bool>>& group) {
            int cells = 0;
            for (size_t which = 0; which < group.size(); ++which)
                for (size_t index = 0; index < group[which].size(); ++index)
                    cells += group[which][index] ? (int)facts.cuts()[which][index].summary.count : 0;
            return cells;
        };
        int found = -1, most = 0;
        vector<vector<bool>> group;
        for (int how = 0; how <= row_of(node_type::ColorFrom).high && group != recolored; ++how) {
            if (how >= COLORS_FROM_ITS_SHAPE ? !facts.asks_likeness(how - COLORS_FROM_ITS_SHAPE)
                                             : how < KEY_BY_PLACE && !facts.asks_way((partner_by)how))
                continue;
            vector<vector<bool>> right = colored_right(facts, recolored, how);
            if (const int cells = cells_of(right); cells > most || right == recolored) {
                most = cells;
                found = how;
                group = std::move(right);
            }
        }
        if (found < 0) return {};
        // The group by its name, else by the colors it wears, which may take a
        // piece the rule leaves as it is (b7256dcd's shape with no dot beside it).
        vector<piece_pick> picks;
        if (const auto named = pick_for_group(facts, group)) picks.push_back(*named);
        uint16_t worn = 0;
        for (size_t which = 0; which < group.size(); ++which)
            for (size_t index = 0; index < group[which].size(); ++index)
                if (group[which][index]) worn = (uint16_t)(worn | palette(facts.cuts()[which][index]));
        if (worn && same_colors_every_pair(facts, worn) && ranges::find(picks, piece_pick{ worn }) == picks.end())
            picks.push_back(piece_pick{ worn });
        for (const piece_pick& pick : picks) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            const auto seen = found >= COLORS_FROM_ITS_SHAPE ? colors_from_its_shape_holds(facts, handed, found - COLORS_FROM_ITS_SHAPE)
                            : found == KEY_BY_PLACE ? key_place_holds(facts, handed)
                            : found == KEY_COLOR ? key_color_holds(facts, handed)
                                                 : color_from_holds(facts, handed, (partner_by)found);
            if (!seen) continue;
            heard best;
            weigh(best, found, pick, *seen, recolor_claims(*seen));
            if (best.value < 0) continue;
            return { best.said, [&, best] {
                carry(say(best, rule::color_from,
                          best.value >= COLORS_FROM_ITS_SHAPE
                              ? string("takes its colors cell by cell from ") +
                                    partners((int)partner_by::same_shape + best.value - COLORS_FROM_ITS_SHAPE)
                              : "takes the color of " + value_written(node_type::ColorFrom, best.value),
                          colors)); } };
        }
        return {};
    };

    // A MOVE THE PIECES DECIDE: every picked piece slides level with its partner,
    // into its rows or its columns; slides to an edge of the grid; moves its own
    // size one of eight ways; or, picked by anything but its colors alone -- holes,
    // size, band, part, partner -- the same step in every pair, which a pick by
    // color already says as an act; or moves toward
    // its partner. Of the kinds, the one that accounts for the most is said, a tie
    // going in that order -- a move toward a partner, the newest rule, after every
    // older one. It accounts for the cells each piece left and the cells it landed
    // on.
    static_assert(direction::count == row_of(node_type::MoveToEdge).high + 1 &&
                  direction::count == row_of(node_type::Shift).high + 1);
    auto move_rule = [&]() -> offer {
        const uint16_t from = starts_from(missing::where_it_went);
        if (!facts.hand_known || !from) return {};
        heard aligned, edged, sized, stepped, toward;
        asked_piece_sets asked, axis_steps, diagonal_steps;
        int best_down = 0, best_across = 0;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (!asked.before(handed)) {
                for (int value = 0; value < 2 * ways; ++value)
                    if (!facts.asks_way((partner_by)(value % ways))) continue;
                    else if (const auto seen = align_holds(facts, handed, value >= ways, (partner_by)(value % ways)))
                        weigh(aligned, value, pick, *seen, move_claims(*seen));
                for (int value = 0; value <= row_of(node_type::MoveToward).high; ++value)
                    if (!facts.asks_kind(value % PARTNER_KINDS)) continue;
                    else if (const auto seen = toward_holds(facts, handed, value))
                        weigh(toward, value, pick, *seen, move_claims(*seen));
                for (int at = 0; at < direction::count; ++at) {
                    const int move_direction = direction_asked(at);
                    if (const auto seen = to_edge_holds(facts, handed, move_direction)) weigh(edged, move_direction, pick, *seen, move_claims(*seen));
                    if (const auto seen = shift_holds(facts, handed, move_direction)) weigh(sized, move_direction, pick, *seen, move_claims(*seen));
                }
            }
            if (pick.by_color()) continue;
            const bool axis_before = axis_steps.before(handed);
            const bool diagonal_before = pick.partner >= 0 || diagonal_steps.before(handed);
            if (axis_before && diagonal_before) continue;
            for (const auto& [down, across] : offsets_to_ask(facts, handed)) {
                // A step both ways is said as two sentences, the second picking
                // again on the hand the first moved: a pick that reads a partner
                // may take other pieces there, so it steps one way only.
                if (down != 0 && across != 0 ? diagonal_before : axis_before) continue;
                const auto seen = move_holds(facts, handed, down, across);
                if (!seen) continue;
                const int before = stepped.said;
                weigh(stepped, 0, pick, *seen, move_claims(*seen));
                if (stepped.said > before) { best_down = down; best_across = across; }
            }
        }
        const int most = max({ aligned.said, edged.said, sized.said, stepped.said, toward.said });
        if (most == 0) return {};
        return { most, [&, aligned, edged, sized, stepped, toward, best_down, best_across, most]() mutable {
            if (aligned.said == most)
                carry(say(aligned, rule::align_to,
                          string("moves level with the ") + (aligned.value >= ways ? "columns" : "rows") + " of " +
                              partners(aligned.value % ways), moves));
            else if (edged.said == most)
                carry(say(edged, rule::to_edge, string("slides ") + direction_names[edged.value] + " to the edge", moves));
            else if (sized.said == most)
                carry(say(sized, rule::shifted, string("moves its own size ") + direction_names[sized.value], moves));
            else if (stepped.said < most) {
                const int kind = toward.value % PARTNER_KINDS;
                carry(say(toward, rule::move_toward,
                          string("moves ") + toward_names[toward.value / PARTNER_KINDS] + " toward " + partner_named(kind),
                          moves));
            } else {
                // Said as the acts a pick by color says: down, then across, each only
                // when it moves at all -- the pick by holes or size is not changed by
                // a move, so the second finds what the first moved.
                heard across = stepped;
                across.claims = [](size_t, const left_over_cell&) { return false; };
                if (best_down) {
                    stepped.value = best_down;
                    carry(say(stepped, rule::move_down, best_down > 0 ? "moves " + to_string(best_down) + " down"
                                                                      : "moves " + to_string(-best_down) + " up",
                              moves));
                }
                if (best_across) {
                    across.value = best_across;
                    if (!best_down) across.claims = stepped.claims;
                    carry(say(across, rule::move_across,
                              best_across > 0 ? "moves " + to_string(best_across) + " right"
                                              : "moves " + to_string(-best_across) + " left", moves));
                }
            }
        } };
    };

    // A COPY ANOTHER PIECE DECIDES: every picked piece is drawn again mirrored
    // around its partner's middle, or the grid's -- left to right, top to bottom,
    // or all four ways. The copies are drawn turned, so it accounts for the copies
    // and the turns of the pieces it mirrors, and for the cells no match covered
    // that the copies land on. All four ways is asked first, so it takes a tie:
    // it holds only where both its halves hold, every copy borne out, and one way
    // alone that accounts for as much says only as much as the pairs' own gaps
    // needed -- as the longer ray takes a tie (b8825c91's test hides a patch on
    // its middle column, which only the copy top to bottom restores).
    static constexpr mirror_by mirrors_asked[] = { mirror_by::four_ways, mirror_by::left_right,
                                                   mirror_by::top_bottom };
    static_assert(size(mirrors_asked) == (size_t)mirror_by::count);
    // Claims for a copy: the cells waiting on a copy or a turn where what it
    // copies lies (`kept`) or its copies land, and the cells no match covered
    // that its copies land on.
    auto copy_claims = [](cells_by_pair kept, cells_by_pair landed) -> claim {
        return [kept = cell_marks(kept), landed = cell_marks(landed)](size_t which, const left_over_cell& cell) {
            const bool mine = on_cells(kept, which, cell.row, cell.col) || on_cells(landed, which, cell.row, cell.col);
            if (cell.waits_for(missing::where_it_was_copied) || cell.waits_for(missing::how_it_turned))
                return mine;
            return cell.waits < 0 && !cell.from_input && on_cells(landed, which, cell.row, cell.col);
        };
    };
    auto mirror_rule = [&]() -> offer {
        const uint16_t from = starts_from(missing::where_it_was_copied);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            for (const mirror_by mirror : mirrors_asked)
                for (int how = 0; how < MIRROR_PARTNERS; ++how) {
                    if (how < (int)partner_by::count && !facts.asks_way((partner_by)how)) continue;
                    const int value = (int)mirror * MIRROR_PARTNERS + how;
                    const auto seen = mirror_holds(facts, handed, mirror, how);
                    if (!seen) continue;
                    weigh(best, value, pick, *seen, copy_claims(seen->left, seen->drawn));
                }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const int how = best.value % MIRROR_PARTNERS;
            carry(say(best, rule::mirror_around,
                      string("drawn again mirrored ") + mirror_names[best.value / MIRROR_PARTNERS] + " around the middle of " +
                          (how < ways ? partners(how) : string("the grid")), moves)); } };
    };

    // A PIECE HOLLOWED TO ITS EDGE: every picked piece keeps only its outline. It
    // accounts for the cells those pieces keep and drop that wait on a change of
    // shape or on nothing at all, and it is asked wherever such a cell is left:
    // asked only for a change of shape, it never saw 4347f46a, whose rectangles'
    // cores are simply gone, with no rule named for them.
    auto outline_rule = [&]() -> offer {
        auto says = [](const left_over_cell& cell) {
            return cell.waits < 0 || cell.waits_for(missing::how_it_changed_shape); };
        const uint16_t from = colors_left(says);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const auto handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            const auto seen = outline_holds(facts, handed);
            if (!seen) continue;
            weigh(best, 0, pick, *seen, [says, kept = cell_marks(seen->drawn), dropped = cell_marks(seen->left)](size_t which,
                                                                                         const left_over_cell& cell) {
                return says(cell) &&
                       (on_cells(kept, which, cell.row, cell.col) || on_cells(dropped, which, cell.row, cell.col));
            });
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            carry(say(best, rule::outlined, "keeps only its outline",
                      read.fired_named(correspondence_election::sits_on_name))); } };
    };

    // A RECOLOR A PICK DECIDES: the pieces it picks all come out in one color --
    // the pick, then Recolor. The colors an unsaid recolor starts from are asked
    // too, alone and with a fit: an act says a recolor by color only where the
    // line opens with the acts, and the rules alone say it themselves (18286ef8's
    // 6 becomes 9 once its 9 has stepped). Asked of the hand the line leaves --
    // or, where the hand could not be carried, only of a line nothing on it moves
    // or reshapes a piece, with no pick by color, which a recolor on it changes,
    // and the color most cells wear, which a recolor changes, only of a bare line.
    auto recolor_rule = [&]() -> offer {
        if (!facts.hand_known && ranges::any_of(read.line, [&](int8_t which) {
                return group_of(read.fired[which].as) == rule_group::changes_pieces; }))
            return {};
        heard best;
        array<asked_piece_sets, COLOR_COUNT> asked;
        const uint16_t from = starts_from(missing::a_recolor);
        for (const piece_pick& pick : hand_picks.get(facts, facts.hand_known ? from : 0, from)) {
            if (pick.how == piece_pick::most_common &&
                !facts.hand_known && !read.line.empty())
                continue;
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            for (int8_t color = 0; color < COLOR_COUNT; ++color) {
                if (already_said(rule::recolor_picked, pick, color)) continue;
                if (asked[(size_t)color].before(handed)) continue;
                if (const auto seen = picked_recolor_holds(facts, handed, color))
                    weigh(best, color, pick, *seen, recolor_claims(*seen));
            }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { carry(say(best, rule::recolor_picked, "becomes " + to_string(best.value), colors)); } };
    };

    // A RECOLOR THE PIECES' OWN KEYS DECIDE: every picked piece takes the color
    // its key names -- its shape, the same whichever way round (e8593010's holes:
    // one cell 3, two 2, three 1), or its symmetry -- ShapeColorMap, its table
    // read off the pairs (read_off), one color a key, the color of the rest for
    // every key it does not name. The puzzle has one table, the first pass's
    // (read_the_cut), so this is asked only there, while nothing else holds it.
    // It accounts for the cells it recolored.
    static constexpr const char* keys_named[] = { "shape", "symmetry" };
    static_assert(size(keys_named) == (size_t)shape_key::count);
    auto shape_rule = [&]() -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (inf.pass > 1 || !facts.hand_known || !from || !facts.overlays || inf.shape_color_map.holds()) return {};
        heard best;
        shape_color_map_rule best_table;
        asked_piece_sets asked;
        vector<uint16_t> shown;
        for (const example& pair : facts.train) shown.push_back(shape_color_map_rule::colors_shown(pair.input));
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            for (int kind = 0; kind < (int)shape_key::count; ++kind) {
                // Each picked piece's key and the one color it comes out in.
                vector<shape_color_map_rule::seen_key> seen;
                bool reads = true;
                for (size_t which = 0; which < handed.size() && reads; ++which) {
                    const grid& output = facts.train[which].output;
                    const partner_pieces& cut = handed[which];
                    for (int i = 0; i < cut.count() && reads; ++i) {
                        if (!cut.picked[i]) continue;
                        // -2 until a cell is read, -3 once two cells disagree.
                        int worn = -2;
                        cut.pieces[i].each_cell([&](int row, int col, int8_t) {
                            const int there = output.data[row][col];
                            worn = worn == -2 || worn == there ? there : -3;
                        });
                        reads = is_color(worn) &&
                                shape_color_map_rule::see(seen, shape_color_map_rule::key_of((shape_key)kind,
                                                                                             cut.pieces[i]),
                                                          (int8_t)worn, which, shown[which]);
                    }
                }
                if (!reads) continue;
                const shape_color_map_rule table = shape_color_map_rule::read_off(seen);
                if (!table.holds()) continue;
                const auto repainted = recolors_hold(facts, handed, [&](size_t, const partner_pieces& cut, int i,
                                                                        int8_t& color) {
                    const int named = table.color_of((shape_key)kind, cut.pieces[i]);
                    color = named >= 0 ? (int8_t)named : LEFT_AS_IT_IS;
                    return true;
                });
                if (!repainted) continue;
                const int before = best.said;
                weigh(best, kind, pick, *repainted, recolor_claims(*repainted));
                if (best.said > before) best_table = table;
            }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best, best_table] {
            inf.shape_color_map = best_table;
            const string rest = best_table.otherwise >= 0 ? ", any other " + to_string(best_table.otherwise) : "";
            carry(say(best, rule::shape_recolor,
                      string("takes the color its ") + keys_named[best.value] + " names" + rest, colors)); } };
    };

    // BidiColorMap PAST ITS FIXED PAIRS. A RECOLOR A KEY DECIDES: the colors of
    // every picked piece trade places as a key drawn in the grid pairs them --
    // the pieces the pick leaves, read along their rows or their columns
    // (key_swaps). Each grid brings its own key, so no swap fixed for every pair
    // says it (0becf7df's corner: 1 and 3, 2 and 8, then 4 and 2, 3 and 7). It
    // accounts for the cells it recolored. OR A COLOR TRADING PLACES WITH EMPTY
    // over each picked piece's box (empty_swap_holds), a color an unsaid recolor
    // starts from (d511f180's 8s and the ground the plan deleted). It accounts for
    // the cells it emptied and the ones it drew where the box was empty.
    static constexpr const char* keys_read[] = { "rows", "columns" };
    static_assert(size(keys_read) == (size_t)(KEY_PAIRS_ALONG_COLUMNS - KEY_PAIRS_ALONG_ROWS + 1) &&
                  EMPTY_PAIRS + COLOR_COUNT - 1 == row_of(node_type::BidiColorMap).high);
    auto swap_claims = [](const replayed& seen) -> claim {
        return [changed = cell_marks(seen.drawn)](size_t which, const left_over_cell& cell) {
            return (cell.answers_to(missing::a_recolor) || !cell.from_input) && on_cells(changed, which, cell.row, cell.col);
        };
    };
    auto key_rule = [&](bool changes_allowed) -> offer {
        const uint16_t from = starts_from(missing::a_recolor);
        if (!facts.hand_known || !from) return {};
        heard best;
        asked_piece_sets asked;
        for (const piece_pick& pick : hand_picks.get(facts, from)) {
            const vector<partner_pieces> handed = handed_pieces(facts, pick);
            if (asked.before(handed)) continue;
            for (int value = KEY_PAIRS_ALONG_ROWS; value <= KEY_PAIRS_ALONG_COLUMNS; ++value)
                if (const auto seen = key_swap_holds(facts, handed, value))
                    weigh(best, value, pick, *seen, recolor_claims(*seen));
            for (int color = 0; color < COLOR_COUNT; ++color)
                if (changes_allowed && (from >> color & 1))
                    if (const auto seen = empty_swap_holds(facts, handed, color))
                        weigh(best, EMPTY_PAIRS + color, pick, *seen, swap_claims(*seen));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            if (best.value >= EMPTY_PAIRS)
                carry(say(best, rule::empty_swap,
                          "its " + to_string(best.value - EMPTY_PAIRS) + "s and the empty cells of its box trade places",
                          colors));
            else
                carry(say(best, rule::key_swap,
                          string("trades colors as the key's ") + keys_read[best.value - KEY_PAIRS_ALONG_ROWS] +
                              " pair them", colors)); } };
    };

    // Paints are emitted after the hand, in this order, even when read first.
    colors_by_pair painted(opened.cuts().size());
    // Derive the whole removal group, describe it once, and verify that rule.
    auto removal_rule = [&]() -> offer {
        const auto pick = pick_for_group(facts, pieces_to_remove(facts, painted));
        if (!pick || already_said(rule::removed, *pick, 0)) return {};
        const auto seen = removal_holds(facts, handed_pieces(facts, *pick), painted);
        if (!seen) return {};
        heard best;
        weigh(best, 0, *pick, *seen, [gone = cell_marks(seen->left)](size_t which, const left_over_cell& cell) {
            return cell.from_input && on_cells(gone, which, cell.row, cell.col);
        });
        if (best.value < 0) return {};
        return { best.said, [&, best] { carry(say(best, rule::removed, "is removed", -1)); } };
    };

    // The picks a paint is asked with: each color on its own, then the picks by
    // holes, size and band.
    vector<piece_pick>& draw_picks = paints.draw_picks;
    if (!cut_opened) {
        for (int color = 0; color < COLOR_COUNT; ++color) draw_picks.push_back(piece_pick::of_color(color));
        for (const piece_pick& pick : picks_to_ask(opened, 0)) draw_picks.push_back(pick);
        paints.plain_draws = draw_picks.size();
        for (const piece_pick& pick : partner_picks(opened)) draw_picks.push_back(pick);
    }

    // WHAT A PAINT SAID AFTER THE LINE READS: it opens the cut again, so it reads
    // the pieces as the cut hands them over, as its replay does, and paints only
    // what it draws. Laid over what is down -- rays, trails, copies, repeats -- it needs
    // nothing of the line at all. Laid under it -- bars, filled hollows, filled
    // boxes -- it needs to know what the line covers: the cut's pieces, where
    // nothing on the line moves, reshapes or removes one, or else the hand the
    // line leaves, where it was carried. Each accounts for the cells it draws, and
    // is asked for every pick that accounts for any.
    auto drawn_claims = [](const cells_by_pair& drawn) -> claim {
        return [drawn = cell_marks(drawn)](size_t which, const left_over_cell& cell) {
            return !cell.from_input && on_cells(drawn, which, cell.row, cell.col);
        };
    };
    // Laid over what is down, a paint explains what it covers as well: an input
    // cell under it came out in the paint's color, its replay saw to that, so a
    // piece under 4f537728's repeated marker needs no recolor of its own.
    auto covered_claims = [](const auto& drawn) -> claim {
        if constexpr (std::is_same_v<std::decay_t<decltype(drawn)>, cell_marks>)
            // These marks live in the line's cache, through weighing and saying.
            return [&drawn](size_t which, const left_over_cell& cell) {
                return on_cells(drawn, which, cell.row, cell.col);
            };
        else return [drawn = cell_marks(drawn)](size_t which, const left_over_cell& cell) {
            return on_cells(drawn, which, cell.row, cell.col);
        };
    };
    // Whether the line leaves every piece where the cut finds it: none of its
    // rules moves, reshapes or removes one, though it may recolor them.
    auto still = [&] {
        return ranges::all_of(read.line, [&](int8_t which) {
            const auto group = group_of(read.fired[which].as);
            return group == rule_group::recolors || group == rule_group::paints; });
    };
    // Whether what is down is known: the cut's pieces, where a still line leaves
    // them where they lie, or the hand the rules carried.
    auto down_known = [&] { return still() || facts.hand_known; };
    // WHAT IS DOWN WHERE A PAINT LANDS: the pieces as the line leaves them so far
    // -- the cut's, where the line is still, else the hand the rules carried
    // (18286ef8's 5s paint, laid under, the hollow their 9 stepped off) -- and
    // every paint said before it, each cell in the color it shows: in the colors
    // the line draws, where it draws its picture, and what each paint laid
    // there. Laid under it or only onto it, a paint reads it, so those ways are
    // asked only where it is known; laid over it, a paint needs nothing of it.
    // Laid again before every rule is asked, and counted each time it changes.
    vector<cells> canvas(opened.cuts().size());
    // How often it changed, and how often where something is; and its number
    // among every canvas a line on the cut laid (cut_paints).
    int canvases = 0, occupancies = 0, canvas_number = 0;
    // WHAT THE LINE DRAWS SO FAR, CELL BY CELL: the hand as the rules leave it,
    // laid as its sentence lays it -- over what is down, or under it where a
    // piece sits on another (draw_the_hand) -- pair by pair, where the rules
    // carried the hand, on pairs that lie over their outputs; and whether it
    // shows a cell's place as the output shows it, which the line then says
    // itself, however the matching paired the pieces.
    vector<grid> line_drawn;
    auto draw_the_line = [&] {
        line_drawn.clear();
        if (!facts.hand_known || !facts.overlays) return;
        for (size_t which = 0; which < facts.cuts().size(); ++which) {
            const grid& output = facts.train[which].output;
            grid drawn(output.width, output.height);
            draw_the_hand(drawn, facts.cuts()[which], facts.laid_under, /*clip=*/true,
                          [](size_t) { return direction_step{ 0, 0 }; });
            line_drawn.push_back(std::move(drawn));
        }
    };
    auto line_shows = [&](size_t which, const left_over_cell& cell) {
        return which < line_drawn.size() && line_drawn[which].contains(cell.row, cell.col) &&
               line_drawn[which].data[cell.row][cell.col] == facts.train[which].output.data[cell.row][cell.col];
    };
    auto lay_the_canvas = [&] {
        draw_the_line();
        vector<cells> now = pieces_down(still() ? opened : facts);
        for (size_t which = 0; which < now.size() && which < line_drawn.size(); ++which)
            for (int row = 0; row < line_drawn[which].height; ++row)
                for (int col = 0; col < line_drawn[which].width; ++col)
                    if (now[which][row][col] != (int8_t)TRANSPARENT &&
                        line_drawn[which].data[row][col] != TRANSPARENT)
                        now[which][row][col] = (int8_t)line_drawn[which].data[row][col];
        for (size_t which = 0; which < painted.size() && which < now.size(); ++which)
            for (const cell_color& cell : painted[which]) now[which][cell.row][cell.col] = cell.color;
        if (now.size() != canvas.size() || memcmp(now.data(), canvas.data(), now.size() * sizeof(cells)) != 0) {
            // Laid under or onto anything, a paint reads only where something
            // is, so it is read again only where that changed.
            bool moved = now.size() != canvas.size();
            for (size_t which = 0; which < now.size() && !moved; ++which)
                for (int row = 0; row < MAX_GRID_DIM && !moved; ++row)
                    for (int col = 0; col < MAX_GRID_DIM && !moved; ++col)
                        moved = (now[which][row][col] != (int8_t)TRANSPARENT) !=
                                (canvas[which][row][col] != (int8_t)TRANSPARENT);
            occupancies += moved;
            canvas = std::move(now);
            ++canvases;
            canvas_number = ++paints.canvases;
        }
    };
    // What a paint lays, pair by pair: the cells it lays right, in the colors
    // the output shows there, and the cells it lays wrong, in what it lays.
    auto laid_by = [&](const replayed& seen) {
        colors_by_pair laid(seen.drawn.size());
        for (size_t which = 0; which < seen.drawn.size(); ++which) {
            for (const auto& [row, col] : seen.drawn[which])
                laid[which].push_back({ (int8_t)row, (int8_t)col, (int8_t)facts.train[which].output.data[row][col] });
            if (which < seen.wrong.size())
                laid[which].insert(laid[which].end(), seen.wrong[which].begin(), seen.wrong[which].end());
        }
        return laid;
    };
    // Says a paint, and keeps what it lays for the canvas.
    auto say_paint = [&](const heard& best, rule::said_as as, const string& says, int8_t under = -1) {
        const int8_t said = say(best, as, says, under);
        const colors_by_pair laid = laid_by(best.seen);
        for (size_t which = 0; which < laid.size() && which < painted.size(); ++which)
            painted[which].insert(painted[which].end(), laid[which].begin(), laid[which].end());
        return said;
    };
    // THE WAYS A PAINT MAY BE LAID on what is down now: over it; where it is
    // known, under it and only onto it; and where the line draws its picture,
    // only onto each color down in every pair, or only off it (MaskBy and
    // MaskOut of the color: bdad9b1f's crossing, one line laid only onto the
    // other's color).
    auto lays = [&] {
        vector<int8_t> ways = { rule::laid_over };
        if (!down_known()) return ways;
        ways.insert(ways.end(), { rule::laid_under, rule::laid_onto });
        if (line_drawn.size() != facts.train.size()) return ways;
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (ranges::all_of(canvas, [&](const cells& down) {
                    return ranges::find(down.values, (int8_t)color) != ranges::end(down.values);
                }))
                ways.insert(ways.end(), { (int8_t)(rule::laid_onto_color + color), (int8_t)(rule::laid_off_color + color) });
        return ways;
    };
    // EACH PAINT REPLAYED ONCE: it reads only the cut opened again and, laid under
    // or onto what is down, the canvas, so one way of asking it -- the kind, the
    // pick, the lay, the value -- is replayed once laid over, and once for each
    // canvas otherwise -- where something is, laid under or onto anything, and
    // in what colors, laid onto or off one -- however often the rules around it
    // are asked; and again once it may no longer lay a cell wrong.
    using replay_key = std::tuple<int, size_t, int, int, int, bool>;
    auto replay_key_of = [&](int kind, size_t pick, int8_t lay, int value) -> replay_key {
        const int read_on = lay == rule::laid_over ? 0
                          : lay == rule::laid_under || lay == rule::laid_onto ? occupancies : canvases;
        return { kind, pick, (int)lay, value, read_on, opened.tolerant };
    };
    std::map<replay_key, optional<replayed>> replays;
    std::map<replay_key, optional<marked_paint>> marked_replays;
    auto replayed_once = [&](auto& cache, int kind, size_t pick, int8_t lay, int value, auto replay) -> const auto& {
        const auto [at, fresh] = cache.try_emplace(replay_key_of(kind, pick, lay, value));
        if (fresh) at->second = replay();
        return at->second;
    };
    // EACH PAINT WALKED ONCE (paint_walk): its walk reads only the cut opened
    // again, so one way of asking it -- the kind, the pick, the value -- is
    // walked once for every line on the cut (cut_paints), however often the
    // rules around it are asked, and laid from that walk every way, on every
    // canvas, in `in` (lay_paint) -- counted once for each canvas it is laid on
    // (walk_tally), so a lay that cannot hold is known from the counts, and only
    // one that may is laid cell by cell.
    auto laid_once = [&](int kind, size_t pick, int8_t lay, int value, laying::colors in,
                         auto walk) -> const optional<marked_paint>& {
        const auto [walked, fresh] = paints.walks.try_emplace({ kind, pick, value });
        if (fresh) walked->second.walk = walk();
        return replayed_once(marked_replays, kind, pick, lay, value, [&]() -> optional<marked_paint> {
            walked_paint& paint = walked->second;
            if (!paint.walk) return nullopt;
            if (paint.counted_on != canvas_number) {
                paint.tally = tally_walk(opened, *paint.walk, canvas);
                paint.counted_on = canvas_number;
            }
            if (!may_hold(opened, paint.tally, { lay, &canvas, in })) return nullopt;
            return lay_paint<marked_paint>(opened, *paint.walk, { lay, &canvas, in });
        });
    };
    // How a paint was laid, in words.
    auto laid_words = [](int8_t lay) -> string {
        if (lay == rule::laid_under) return ", laid under what is down";
        if (lay == rule::laid_onto) return ", only onto what is down";
        if (lay >= rule::laid_off_color) return ", only off the " + to_string(lay - rule::laid_off_color) + "s down";
        if (lay >= rule::laid_onto_color) return ", only onto the " + to_string(lay - rule::laid_onto_color) + "s down";
        return "";
    };

    // THE PAINTS, THE ONE THAT SAYS THE MOST FIRST. Each kind -- rays, trails,
    // copies, repeats, the turned figure, bars, filled hollows, filled boxes --
    // offers its best over every pick, the kind whose best accounts for the most
    // cells nothing claims yet is said, a tie going in that order, and then every
    // kind is asked again of the claims and the canvas that paint leaves, so a
    // kind can be said again: a piece grows rays to the sides and then along the
    // diagonals. Asked in a fixed order instead, the first kind took what a later
    // one says whole: e729b7be's row from the middle repeated down the grid, a
    // coincidence of the pairs, was said ahead of the half turn that draws every
    // output. Each kind's pieces are handed over once, the cut opened again.
    // A PAINT IS WEIGHED ONLY WHERE IT EXPLAINS A CELL IN EVERY PAIR that
    // nothing claims yet. One that explains a cell or two of one pair redoes, by
    // coincidence, what the line already did in the others: 91714a58's rays one
    // cell long, in the ground's color, three of them, finished a line of color
    // coincidences that is wrong on its test. A pair finished before the line
    // (replay_facts::finished) has no cell to explain, and asks none.
    // AND A PAINT ACCOUNTS ONLY FOR WHAT THE LINE DOES NOT SHOW ALREADY: a cell
    // the line so far draws as the output shows it is the line's own to say. A
    // paint asked beside the acts could take it otherwise: 22168020's pieces,
    // each repeated along its rows only onto what is down, redrew the pieces
    // the line leaves where they lie -- and the bars beside them -- and did
    // nothing else, true of the pairs and wrong on its test.
    auto weigh_paint = [&](heard& best, int value, const piece_pick& pick, const auto& seen, claim claims) {
        claim unshown = [&line_shows, claims = std::move(claims)](size_t which, const left_over_cell& cell) {
            return claims(which, cell) && !line_shows(which, cell);
        };
        for (size_t which = 0; which < read.pairs.size(); ++which)
            if (!opened.finished(which) && ranges::none_of(read.pairs[which].left_over, [&](const left_over_cell& cell) {
                    return cell.claimed_by < 0 && unshown(which, cell); }))
                return;
        weigh(best, value, pick, seen, std::move(unshown));
    };

    // Own-color paints cannot claim a target color absent from their sources.
    // Read actual output colors: an input-side leftover carries its old color.
    vector<uint16_t> paint_colors(read.pairs.size());
    auto colors_of_pick = [](const vector<partner_pieces>& handed) {
        vector<uint16_t> colors(handed.size());
        for (size_t which = 0; which < handed.size(); ++which)
            for (int i = 0; i < handed[which].count(); ++i)
                if (handed[which].picked[i])
                    colors[which] |= palette(handed[which].pieces[i]);
        return colors;
    };
    auto paint_has_color = [&](const vector<uint16_t>& colors) {
        for (size_t which = 0; which < read.pairs.size(); ++which)
            if (!opened.finished(which) && !(colors[which] & paint_colors[which])) return false;
        return true;
    };
    vector<vector<partner_pieces>>& handed_by_pick = paints.handed_by_pick;
    if (!cut_opened)
        for (const piece_pick& pick : draw_picks) {
            handed_by_pick.push_back(handed_pieces(opened, pick));
            paints.colors_by_pick.push_back(colors_of_pick(handed_by_pick.back()));
        }
    // THE DRAW PICKS A PAINT ASKS, in order. A pick that takes the same pieces
    // as one before it, in every pair, is handed the same pieces: every paint
    // draws for it what it drew for that one, which keeps a tie, so it is not
    // asked.
    vector<size_t>& asked_picks = paints.asked_picks;
    if (!cut_opened)
        for (size_t at = 0; at < draw_picks.size(); ++at) {
            auto same = [&](size_t before) { return takes_the_same(handed_by_pick[before], handed_by_pick[at]); };
            if (ranges::any_of(asked_picks, same) || ranges::any_of(paints.asked_partner_picks, same)) continue;
            (at < paints.plain_draws ? asked_picks : paints.asked_partner_picks).push_back(at);
        }
    // The draw picks a paint is asked with now: the partner picks too, once the
    // line has turned to its last resort (partner_picks).
    vector<size_t> asked_now = asked_picks;
    vector<size_t> own_color_picks;

    // RAYS A PIECE GROWS: up, down and to the sides, or along the diagonals, as
    // long as every pair bears out, painted in one color, its own or one the line
    // names, which is a question of its own. A tie goes to the longer ray: a ray
    // that explains no more for being longer has run into the edge, and a larger
    // test grid is exactly where the shorter reading stops too soon (ea786f4a).
    // Asked again while a ray accounts for any cell, so one piece can grow both
    // kinds: dc1df850's ring is the four to the sides and the four along the
    // diagonals, and said once, the diagonals were left to any other pick that
    // happened to take the same cells.
    const auto rays = row_of(node_type::ExtendOrthogonal);
    static_assert(row_of(node_type::ExtendOrthogonal).low == row_of(node_type::ExtendDiagonal).low &&
                  row_of(node_type::ExtendOrthogonal).high == row_of(node_type::ExtendDiagonal).high &&
                  row_of(node_type::ExtendDiagonal).high < paint_value::second_bit,
                  "one ray length is said the same both ways round, and fits under the diagonal's 32");
    // THE FARTHEST A RAY RUNS on any pair's grid: past it, every longer ray draws
    // as the longest, asked first, so it is replayed once.
    int farthest = 0;
    for (const example& pair : opened.train) farthest = max(farthest, max(pair.output.height, pair.output.width) - 1);
    // EVERY RAY LENGTH LAID AT ONCE (lay_every_length): the rays of a pick, one
    // way round, fanned out once for every line on the cut (rays_fan), every
    // length laid the first time one is asked on a canvas, and each kept as
    // laid_once keeps one. A length is walked on its own (rays_walk), once for
    // every line, only where it ties for its color or holds.
    auto rays_laid_once = [&](size_t at, int8_t lay, bool diagonal, int reach) -> const optional<marked_paint>& {
        const replay_key key = replay_key_of(0, at, lay, paint_value::packed(diagonal, reach));
        if (const auto found = marked_replays.find(key); found != marked_replays.end()) return found->second;
        const auto [fanned, fresh] = paints.fans.try_emplace({ at, diagonal });
        if (fresh) fanned->second = rays_fan(opened, handed_by_pick[at], diagonal, farthest);
        vector<optional<marked_paint>> laid;
        if (fanned->second)
            laid = lay_every_length(opened, *fanned->second, { lay, &canvas, laying::in_one }, farthest,
                                    [&](int length) -> const paint_walk* {
                const auto [walked, unwalked] = paints.walks.try_emplace({ 0, at, paint_value::packed(diagonal, length) });
                if (unwalked) walked->second.walk = rays_walk(opened, handed_by_pick[at], diagonal, length);
                return walked->second.walk ? &*walked->second.walk : nullptr;
            });
        laid.resize((size_t)farthest + 1);
        for (int length = 0; length <= farthest; ++length)
            marked_replays.try_emplace(replay_key_of(0, at, lay, paint_value::packed(diagonal, length)),
                                       std::move(laid[(size_t)length]));
        return marked_replays.at(key);
    };
    auto ray_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : asked_now)
            for (const int8_t lay : lays())
                for (const bool diagonal : { false, true })
                    for (int radius = rays.high; radius >= rays.low; --radius) {
                        if (radius < rays.high && radius > farthest) continue;
                        const int reach = min(radius, farthest);
                        const auto& seen = rays_laid_once(at, lay, diagonal, reach);
                        if (!seen) continue;
                        const int before = best.said;
                        weigh_paint(best, paint_value::packed(diagonal, radius), draw_picks[at], *seen,
                                    covered_claims(seen->drawn));
                        if (best.said > before) best_lay = lay;
                    }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const bool own = best.pick.by_color() && best.pick.colors == (uint16_t)(1u << best.seen.paint);
            const int8_t said = say_paint(best, rule::extend_rays,
                string("grows rays ") + (paint_value::second(best.value) ? "along the diagonals" : "up, down and to the sides") + ", " +
                    to_string(paint_value::number(best.value)) + " long, in " +
                    (own ? string("its own color") : "color " + to_string(best.seen.paint)) + laid_words(best_lay));
            read.fired[said].lay = best_lay; }, laid_by(best.seen) };
    };

    // A TRAIL A PIECE GROWS one of the eight ways, in its own color, until it meets
    // the edge of the grid or another piece -- or, past those, every picked piece
    // grown in step with the others along its track until they meet, the eight
    // ways asked first and taking a tie -- asked again, like the rays, while a
    // trail accounts for any cell, laid each way a paint may be.
    static_assert(direction::count + (int)grown_along::count == row_of(node_type::ExtendTo).high + 1);
    static constexpr const char* grown_names[] = { "along its rows", "along its columns", "to every side",
                                                   "along the grid's edge", "along its diagonals down to the right",
                                                   "along its diagonals up to the right", "along both its diagonals",
                                                   "along its rows and columns", "every way, straight and diagonal",
                                                   "to every side and corner" };
    static_assert(size(grown_names) == (size_t)grown_along::count);
    auto trail_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : own_color_picks)
            for (const int8_t lay : lays())
                for (int asked = 0; asked <= row_of(node_type::ExtendTo).high; ++asked) {
                    const int grow_direction = asked < direction::count ? direction_asked(asked) : asked;
                    const auto& seen = laid_once(1, at, lay, grow_direction, laying::as_drawn, [&] {
                        return trail_walk(opened, handed_by_pick[at], grow_direction); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, grow_direction, draw_picks[at], *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const string grows = best.value < direction::count
                ? string("grows a trail ") + direction_names[best.value] + " until it meets something"
                : string("grows ") + grown_names[best.value - direction::count] + " in step with the others until they meet";
            const int8_t said = say_paint(best, rule::trailed, grows + laid_words(best_lay));
            read.fired[said].lay = best_lay;
        }, laid_by(best.seen) };
    };

    // THE PICKED PIECES COPIED ONTO THE OTHERS, lined up one of CopyTo's nine ways.
    static_assert(row_of(node_type::CopyTo).high + 1 == 9);
    static constexpr const char* lined_up[] = { "top-left", "top", "top-right", "left", "middle", "right",
                                                "bottom-left", "bottom", "bottom-right" };
    auto copy_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : own_color_picks)
            for (const int8_t lay : lays())
                for (int anchor = 0; anchor < (int)size(lined_up); ++anchor) {
                    const auto& seen = laid_once(2, at, lay, anchor, laying::as_drawn, [&] {
                        return copy_walk(opened, handed_by_pick[at], anchor); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, anchor, draw_picks[at], *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            const int8_t said = say_paint(best, rule::copied_onto,
                string("is copied onto the other pieces, lined up ") + lined_up[best.value] + laid_words(best_lay));
            read.fired[said].lay = best_lay;
        }, laid_by(best.seen) };
    };

    // A PIECE REPEATED ALONG ITS ROWS OR DOWN ITS COLUMNS, to the edges both ways
    // -- as far apart as its own width along its rows, or its own height down
    // its columns, read on each grid, or every so many cells -- asked again,
    // like the rays, while a repeat accounts for any cell, so one piece can be
    // repeated both ways. A tie goes to its own width or height, which grows
    // with the piece on a test, then to the tighter spacing the pairs bear out.
    // THE NARROWEST PAIR'S GRID, across and down: so many cells apart as that
    // lays no copy beside the piece itself there, and is not at work.
    int narrowest[2] = { MAX_GRID_DIM, MAX_GRID_DIM };
    for (const example& pair : opened.train) {
        narrowest[0] = min(narrowest[0], pair.output.width);
        narrowest[1] = min(narrowest[1], pair.output.height);
    }
    // The repeats asked, in the order asked; a repeat's value is its place here.
    // ITS OWN WIDTH AND HEIGHT ARE A SHAPE'S: a whole color layer's are only the
    // span its scattered cells happen to cover, and a layer repeated by them fits
    // the pairs by coincidence (62b74c02's last nine cells, wrong on its test).
    const bool cut_into_shapes = election.segmentation != segmentation_methods::all_cells_of_one_color;
    vector<tile_value> repeats;
    for (const bool down : { false, true }) {
        auto apart = [&](tile_number cells) {
            tile_value repeat{ .start = { .edge = true }, .end = { .edge = true },
                               .spread = tile_spread::lines };
            repeat.steps[0] = down ? tile_value::step{ cells, { 0 } } : tile_value::step{ { 0 }, cells };
            repeats.push_back(repeat);
        };
        if (cut_into_shapes) apart(measured(down ? tile_measure::own_height : tile_measure::own_width));
        for (int cells = 1; cells < narrowest[down]; ++cells) apart({ (int8_t)cells });
    }
    // THE FIGURE REPEATED FROM PAST ITSELF, along its rows every key width or
    // down its columns every key height, either way: every piece but one color's,
    // folded into one figure, moved its own size the way the copies go and
    // repeated from there to the grid's edge (past_walk), the key the pieces of
    // that color, held in every pair (12422b43's 5s, the rows beside them
    // stacked under the figure).
    vector<std::pair<int, tile_value>> past_repeats;
    for (const int shift : { direction::right, direction::left, direction::down, direction::up }) {
        const direction_step step = step_of(shift);
        const tile_number apart{ .terms = { { step.down + step.across < 0,
                                              step.down ? tile_measure::key_height : tile_measure::key_width } } };
        tile_value repeat{ .end = { .edge = true }, .spread = tile_spread::lines };
        repeat.steps[0] = step.down ? tile_value::step{ apart, { 0 } } : tile_value::step{ { 0 }, apart };
        past_repeats.push_back({ shift, repeat });
    }
    vector<std::pair<piece_pick, vector<partner_pieces>>> figures;
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (every_pair_holds(opened, color)) {
            piece_pick figure = piece_pick::of_color(color);
            figure.but_those = true;
            figures.push_back({ figure, handed_pieces(opened, figure) });
        }
    auto repeat_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        for (const size_t at : own_color_picks)
            for (const int8_t lay : lays())
                for (int asked = 0; asked < (int)repeats.size(); ++asked) {
                    const auto& seen = laid_once(3, at, lay, asked, laying::as_drawn, [&] {
                        return tile_walk(opened, handed_by_pick[at], repeats[(size_t)asked]); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, asked, draw_picks[at], *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        for (size_t at = 0; at < figures.size(); ++at) {
            const piece_pick& figure = figures[at].first;
            const vector<partner_pieces>& handed = figures[at].second;
            if (!paint_has_color(colors_of_pick(handed))) continue;
            for (const int8_t lay : lays())
                for (int asked = 0; asked < (int)past_repeats.size(); ++asked) {
                    const auto& seen = laid_once(9, at, lay, asked, laying::as_drawn, [&] {
                        return past_walk(opened, handed, past_repeats[(size_t)asked].first,
                                         past_repeats[(size_t)asked].second); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, (int)repeats.size() + asked, figure, *seen, covered_claims(seen->drawn));
                    if (best.said > before) best_lay = lay;
                }
        }
        if (!best.materialize(opened)) return {};
        return { best.said, [&, best, best_lay] {
            if (best.value >= (int)repeats.size()) {
                const auto& [shift, repeat] = past_repeats[(size_t)best.value - repeats.size()];
                heard past = best;
                past.value = shift;
                const int8_t said = say_paint(past, rule::repeated_past,
                    string("as one figure, is repeated ") + direction_names[shift] + " from past itself, the key's " +
                        (repeat.steps[0].down != tile_number{} ? "height" : "width") + " apart" + laid_words(best_lay));
                read.fired[said].lay = best_lay;
                read.fired[said].tile = repeat;
                return;
            }
            const tile_value& repeat = repeats[(size_t)best.value];
            const bool down = repeat.steps[0].down != tile_number{};
            const tile_number& cells = down ? repeat.steps[0].down : repeat.steps[0].across;
            const int8_t said = say_paint(best, rule::tiled_along,
                "is repeated " +
                    (cells.terms[0].measure == tile_measure::none
                         ? "every " + to_string(cells.count) + " cells "
                         : "its " + string(tile_measure_names[(int)cells.terms[0].measure]) + " apart ") +
                    (down ? "down its columns" : "along its rows") + laid_words(best_lay));
            read.fired[said].lay = best_lay;
            read.fired[said].tile = repeat; }, laid_by(best.seen) };
    };

    // THE WHOLE FIGURE DRAWN AGAIN TURNED in its own box -- a quarter turn
    // right, a half turn, a quarter turn left, each a copy of its own -- asked
    // again while a turn accounts for any cell, so it can be turned all three
    // ways (11852cab's design, the same from all four sides). Every piece, as one
    // figure: a part of the drawing turned in a box of its own fit the pairs by
    // accident (2546ccf6's quadrants, which mirror across walls; 52df9849's
    // hidden corner). It accounts for the figure where it lies as well as for its
    // copies, so it is asked only of a still line.
    const int right = pose_of(node_type::TurnRight);
    const std::pair<int, const char*> turns[] = { { right, "a quarter turn right" },
                                                  { compose_poses(right, right), "a half turn" },
                                                  { pose_of(node_type::TurnLeft), "a quarter turn left" } };
    const piece_pick figure = piece_pick::all();
    const vector<partner_pieces> figure_handed = handed_pieces(opened, figure);
    const vector<uint16_t> figure_colors = colors_of_pick(figure_handed);
    auto turned_paint = [&]() -> offer {
        if (!still() || !paint_has_color(figure_colors)) return {};
        heard best;
        for (const auto& [pose, name] : turns) {
            if (const auto& seen = replayed_once(replays, 4, 0, rule::laid_over, pose, [&] {
                    return turned_holds(opened, figure_handed, pose); }))
                weigh_paint(best, pose, figure, *seen, covered_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const auto turn = ranges::find_if(turns, [&](const auto& one) { return one.first == best.value; });
            say_paint(best, rule::turned_copy, string("is drawn again as one figure, ") + turn->second + " in its box");
        }, laid_by(best.seen) };
    };

    // BARS BETWEEN PIECES OF ONE COLOR, in their own color or one of the line's,
    // reaching each way a bridge can, the narrowest first, so a wider way is said
    // only where it accounts for more. Between every piece first, then between
    // the pieces each pick takes, as the other paints are asked: Bridge joins
    // only the picked pieces (0d87d2a6's blue marks, past red blocks that line
    // up with one another and must not be joined).
    // THEN AN L from each picked piece to its one partner of each kind
    // (partner_of_kind), every pick asked for every straight bar first, so an
    // L is said only where it accounts for more. It leaves along the picked
    // piece's row, so the pick says which way it turns (0e671a1a: the 2 runs to
    // the piece of color 4, and the 4 to the piece of color 3).
    const vector<partner_pieces> every_piece_handed = handed_pieces(opened, piece_pick::all());
    auto bar_paint = [&]() -> offer {
        if (!down_known()) return {};
        heard best;
        for (const bool ells : { false, true })
            for (size_t at = 0; at <= draw_picks.size(); ++at) {
                if (at > 0 && !ranges::binary_search(asked_now, at - 1)) continue;
                const piece_pick pick = at == 0 ? piece_pick::all() : draw_picks[at - 1];
                const vector<partner_pieces>& handed = at == 0 ? every_piece_handed : handed_by_pick[at - 1];
                auto ask = [&](int way) {
                    if (const auto& seen = replayed_once(replays, 5, at, rule::laid_under, way, [&] {
                            return bridge_holds(opened, handed, canvas, way); }))
                        weigh_paint(best, way, pick, *seen, drawn_claims(seen->drawn));
                };
                if (!ells) ranges::for_each(bridge_ways_narrowest_first, ask);
                else for (int way = BRIDGE_TO_PARTNER; way < BRIDGE_VALUES; ++way)
                    if (opened.asks_kind(way - BRIDGE_TO_PARTNER)) ask(way);
            }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::bridged,
                (best.value < BRIDGE_TO_PARTNER
                     ? string("bars filled between pieces of one color, reaching ") + bridge_reach_names[best.value]
                     : "an L run along its row to the column of " + partner_named(best.value - BRIDGE_TO_PARTNER) +
                           ", then along that column to it") +
                    ", in " + (best.seen.paint < 0 ? string("their own color") : "color " + to_string(best.seen.paint)));
        }, laid_by(best.seen) };
    };

    // THE REST OF THE CHANGE, READ OFF THE PICTURE SO FAR. Once a paint is said,
    // what it drew is in the picture the line has drawn -- the hand as the acts
    // leave it, and every cell a paint drew, in the color its replay saw the
    // output show there -- and the rest of the change may read it: the picture
    // is cut again as the line cuts the input, and each picked piece takes the
    // color of its partner, a piece a paint drew among them (0d87d2a6's red
    // blocks, blue where a bar the bridge drew touches one). Said after the
    // paints said before it, in a sentence of its own that opens on the picture
    // (Output), it lays only the pieces it picked, over what is down, where they
    // lie. Only a recolor is read so: a piece moved or removed would leave its
    // old cells drawn. And every act on the hand is drawn before the paints,
    // under the picture this read, so none is asked after it.
    // The picture with `also` drawn in as well: the cells a paint not said yet
    // would lay, in what it lays there. Every paint lays its cells in the order
    // said, the ones it lays wrong in what it lays, so a cell one lays wrong and
    // a later one sets right shows right.
    auto the_picture = [&](const colors_by_pair& also) {
        vector<grid> pictures = line_drawn;
        for (size_t which = 0; which < pictures.size(); ++which) {
            auto shown = [&](const vector<cell_color>& cells) {
                for (const cell_color& cell : cells) pictures[which].data[cell.row][cell.col] = cell.color;
            };
            if (which < painted.size()) shown(painted[which]);
            if (which < also.size()) shown(also[which]);
        }
        return pictures;
    };
    // A PIECE OF THE PICTURE THE LINE KNOWS: one of the hand's own pieces, the
    // same cells where they lie. A paint may draw a partner beside a piece, but
    // a piece a paint cut into or ran into is not one the line cut: 63613498's
    // diagonal rays, laid onto a shape, left fragments of it for a recolor to
    // finish, a coincidence of its pairs that is wrong on its test.
    auto whole = [&](const vector<partner_pieces>& handed) {
        for (size_t which = 0; which < handed.size(); ++which)
            for (int i = 0; i < handed[which].count(); ++i) {
                if (!handed[which].is_picked(i)) continue;
                const object& piece = handed[which].pieces[i];
                const bool known = ranges::any_of(facts.cuts()[which], [&](const object& other) {
                    const object_summary& one = piece.summary, & two = other.summary;
                    if (one.row != two.row || one.col != two.col || one.width != two.width ||
                        one.height != two.height || one.count != two.count) return false;
                    bool same = true;
                    piece.each_cell([&](int row, int col, int8_t) { same = same && other.holds(row, col); });
                    return same;
                });
                if (!known) return false;
            }
        return true;
    };
    // The best recolor read off the picture, with `also` drawn in: none before
    // any paint, since the picture is then the hand the acts read already.
    auto read_off_the_picture = [&](const colors_by_pair& also) {
        heard best;
        const bool painted_yet = !also.empty() || ranges::any_of(read.line, [&](int8_t which) {
            return group_of(read.fired[which].as) == rule_group::paints; });
        const uint16_t from = starts_from(missing::a_recolor);
        if (!painted_yet || !from || line_drawn.size() != facts.train.size()) return best;
        const replay_facts picture(facts.train, the_picture(also), election.segmentation, boundaries, &election.shape);
        for (const piece_pick& pick : picks_to_ask(picture, from)) {
            const vector<partner_pieces> handed = handed_pieces(picture, pick);
            if (!whole(handed)) continue;
            for (int how = 0; how < ways; ++how)
                if (!picture.asks_way((partner_by)how)) continue;
                else if (const auto seen = color_from_holds(picture, handed, (partner_by)how))
                    weigh(best, how, pick, *seen, recolor_claims(*seen));
        }
        return best;
    };
    auto picture_rule = [&]() -> offer {
        const heard best = read_off_the_picture({});
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const int8_t said = say_paint(best, rule::color_from,
                string("takes the color of ") + partners(best.value) + ", read off the picture so far", colors);
            read.fired[said].on_the_picture = true;
        } };
    };

    // THE PICTURE MADE SYMMETRIC (symmetric_holds): the picture so far laid over
    // itself mirrored or turned around the grid's middle, the narrowest symmetry
    // first, a wider one said only where it accounts for more. One rule for the
    // whole symmetry: said a pose at a time, an early pose can finish one pair
    // and leave a later one nothing to do there (9ddd00f0's first pair needs the
    // mirrors alone, its second the quarter turn too). It reads the picture, so
    // it is said after the paints before it, and no act on the hand follows it.
    // A copy, it accounts as the mirror rule does (copy_claims), what it copies
    // being the whole picture: weighed as a paint, it could not count the cells
    // of the pieces it copies, which the line draws already, and lost them to
    // a mirror of some of the pieces. It completes the input, asked only before
    // anything else is said: asked after 21f83797's rays, it mirrored them, a
    // symmetry its pairs' marks share by coincidence, and went wrong on its test.
    auto symmetry_paint = [&]() -> offer {
        if (!read.line.empty() || line_drawn.size() != facts.train.size()) return {};
        replay_facts as_one = facts;
        const vector<grid> pictures = the_picture({});
        for (size_t which = 0; which < as_one.cuts().size(); ++which) {
            const grid& picture = pictures[which];
            object piece;
            int count = 0;
            for (int row = 0; row < picture.height; ++row)
                for (int col = 0; col < picture.width; ++col)
                    if (picture.data[row][col] != (int8_t)TRANSPARENT) {
                        piece.data[row][col] = (int8_t)picture.data[row][col];
                        ++count;
                    }
            piece.summary = { 0, 0, (int8_t)picture.width, (int8_t)picture.height, (int16_t)count };
            as_one.change(which).hand = { piece };
        }
        const vector<partner_pieces> handed = handed_pieces(as_one, piece_pick::all());
        cells_by_pair copied(as_one.cuts().size());
        for (size_t which = 0; which < as_one.cuts().size(); ++which)
            as_one.cuts()[which][0].each_cell([&](int row, int col, int8_t) { copied[which].push_back({ row, col }); });
        heard best;
        for (int kept = 0; kept < (int)size(symmetries); ++kept)
            if (const auto seen = symmetric_holds(as_one, handed, symmetries[kept]))
                weigh(best, kept, piece_pick::all(), *seen, copy_claims(copied, seen->drawn));
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            const int8_t said = say_paint(best, rule::symmetric,
                                          string("the picture so far ") + symmetries[best.value].says);
            read.fired[said].on_the_picture = true;
        }, laid_by(best.seen) };
    };

    // THE HOLLOWS OF THE PICKED PIECES, FILLED, in their own color or one of the
    // line's -- or, with value 1, THEIR INSIDE: hollowed to the edge first, the
    // core and the hollows in one color, laid over (inside_holds).
    auto fill_paint = [&]() -> offer {
        if (!down_known()) return {};
        heard best;
        for (const size_t at : asked_now) {
            if (const auto& seen = replayed_once(replays, 6, at, rule::laid_under, 0, [&] {
                    return fill_holds(opened, handed_by_pick[at], canvas); }))
                weigh_paint(best, 0, draw_picks[at], *seen, drawn_claims(seen->drawn));
            if (const auto& seen = replayed_once(replays, 6, at, rule::laid_over, 1, [&] {
                    return inside_holds(opened, handed_by_pick[at]); }))
                weigh_paint(best, 1, draw_picks[at], *seen, covered_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] {
            say_paint(best, rule::filled,
                      string(best.value == 1 ? "fills its inside in " : "fills its hollows in ") +
                          (best.seen.paint < 0 ? string("its own color") : "color " + to_string(best.seen.paint)));
        }, laid_by(best.seen) };
    };

    // THE BOXES OF THE PICKED PIECES, FILLED where the pieces leave them empty, in
    // one color.
    auto box_paint = [&]() -> offer {
        if (!down_known()) return {};
        heard best;
        for (const size_t at : asked_now) {
            if (const auto& seen = replayed_once(replays, 7, at, rule::laid_under, 0, [&] {
                    return box_holds(opened, handed_by_pick[at], canvas); }))
                weigh_paint(best, 0, draw_picks[at], *seen, drawn_claims(seen->drawn));
        }
        if (best.value < 0) return {};
        return { best.said, [&, best] { say_paint(best, rule::boxed, "fills its box in color " + to_string(best.seen.paint)); }, laid_by(best.seen) };
    };

    // THE ROOMS THE PICKED PIECES FENCE OFF, ONE FILLED in one color: the
    // largest or smallest room, or a part's room (room_holds). The walls are
    // every piece, or the pieces of one color, which the program can pick again
    // once Partition hands them back. A room is cut from the input, not from what is down, so it is
    // laid over what is down, and under it where that is known, a tie going
    // over. A pick's rooms are walked once, for every room and canvas the line
    // asks.
    vector<optional<vector<vector<fenced_room>>>> rooms_by_pick(draw_picks.size());
    auto room_named = [](int chosen) {
        if (chosen == ROOM_EVERY) return string("every room");
        if (!room_by_size(chosen)) return "the room of part " + to_string(room_part(chosen));
        const int rank = room_rank(chosen);
        return string("the ") + (rank > 0 ? to_string(rank + 1) + (rank == 1 ? "nd " : rank == 2 ? "rd " : "th ") : "") +
               (room_largest(chosen) ? "largest" : "smallest") + " room";
    };
    // EVERY ROOM AT ONCE, EACH IN THE COLOR ITS OWN KEY NAMES (rooms_table): the
    // puzzle's one table, so asked only in the first pass while nothing else
    // holds one (shape_rule), and said, it is the table the program carries.
    // AND THE GROUND'S PLACES SHAPED LIKE THE SHAPE THE CHANGES TAKE, every one
    // at once (shaped_rooms), where the pairs learned one: no walls fence them.
    optional<vector<vector<fenced_room>>> rooms_of_every_piece, rooms_of_the_shape;
    auto room_paint = [&]() -> offer {
        heard best;
        int8_t best_lay = rule::laid_over;
        shape_color_map_rule best_table;
        const bool tables = inf.pass == 1 && !inf.shape_color_map.holds();
        // The rooms the walls `pick` takes fence off, `at` its place among the
        // picks a paint is replayed for.
        auto ask = [&](size_t at, const piece_pick& pick, const vector<partner_pieces>& handed,
                       optional<vector<vector<fenced_room>>>& rooms) {
            if (!rooms) rooms = rooms_fenced(opened, handed, inf.before_ground);
            for (const int8_t lay : { rule::laid_over, rule::laid_under }) {
                if (lay == rule::laid_under && !down_known()) continue;
                for (int chosen = 0; chosen < (tables ? ROOM_VALUES : ROOM_CHOICES); ++chosen) {
                    shape_color_map_rule table;
                    if (chosen >= ROOM_TABLE) {
                        table = rooms_table(opened, *rooms, chosen - ROOM_TABLE);
                        if (!table.holds()) continue;
                    } else if (ranges::any_of(*rooms, [&](const auto& pair) {
                                   return chosen == ROOM_EVERY ? pair.empty() : !room_chosen(pair, chosen); }))
                        continue;
                    const auto& seen = replayed_once(replays, 8, at, lay, chosen, [&] {
                        return chosen >= ROOM_TABLE
                            ? room_table_holds(opened, handed, *rooms, canvas, table, chosen - ROOM_TABLE, lay)
                            : room_holds(opened, handed, *rooms, canvas, chosen, lay); });
                    if (!seen) continue;
                    const int before = best.said;
                    weigh_paint(best, chosen, pick, *seen,
                                lay == rule::laid_over ? covered_claims(seen->drawn) : drawn_claims(seen->drawn));
                    if (best.said > before) {
                        best_lay = lay;
                        best_table = table;
                    }
                }
            }
        };
        // The walls: every piece -- 83302e8f's are 1s, 8s and 9s, one color a
        // pair -- then the pieces of each one color that are not every piece.
        ask(draw_picks.size(), piece_pick::all(), every_piece_handed, rooms_of_every_piece);
        for (const size_t at : asked_now) {
            const piece_pick& pick = draw_picks[at];
            if (has_single_bit(pick.colors) && pick == piece_pick::of_color(countr_zero(pick.colors)) &&
                !takes_the_same(handed_by_pick[at], every_piece_handed))
                ask(at, pick, handed_by_pick[at], rooms_by_pick[at]);
        }
        if (election.shape.holds()) {
            if (!rooms_of_the_shape) rooms_of_the_shape = shaped_rooms(opened, election.shape, inf.before_ground);
            for (const int8_t lay : { rule::laid_over, rule::laid_under }) {
                if (lay == rule::laid_under && !down_known()) continue;
                const auto& seen = replayed_once(replays, 8, draw_picks.size() + 1, lay, ROOM_SHAPE, [&] {
                    return room_holds(opened, every_piece_handed, *rooms_of_the_shape, canvas, ROOM_EVERY, lay); });
                if (!seen) continue;
                const int before = best.said;
                weigh_paint(best, ROOM_SHAPE, piece_pick::all(), *seen,
                            lay == rule::laid_over ? covered_claims(seen->drawn) : drawn_claims(seen->drawn));
                if (best.said > before) {
                    best_lay = lay;
                    best_table = {};
                }
            }
        }
        if (best.value < 0) return {};
        return { best.said, [&, best, best_lay, best_table] {
            const bool by_table = best.value >= ROOM_TABLE && best.value < ROOM_SHAPE;
            if (by_table) inf.shape_color_map = best_table;
            const string paint = ", filled in color " + to_string(best.seen.paint);
            const int8_t said = say_paint(best, rule::filled_room,
                (by_table ? "fences off its rooms, each taking the color its " +
                                string(keys_named[best.value - ROOM_TABLE]) + " names"
                 : best.value == ROOM_SHAPE ? "cuts the changed shape's places off the ground" + paint
                                            : "fences off " + room_named(best.value) + paint) +
                    laid_words(best_lay));
            read.fired[said].lay = best_lay;
        }, laid_by(best.seen) };
    };

    // THE ACTS AND THE PAINTS, THE ONE THAT SAYS THE MOST FIRST. Every kind of
    // act and of paint offers its best on the hand and the canvas the line
    // leaves so far, the one accounting for the most cells nothing claims yet is
    // said, a tie going in the order below -- the acts before the paints -- and
    // every kind is asked again. Asked after the acts instead, a paint lost what
    // it says whole to the acts said first: 178fcbfb's rows and columns, each
    // repeated across the grid, came out as a string of mirrored cells the
    // repeats then finished. Spelled, every paint follows the line, laid over
    // what it leaves -- so a paint laid over needs nothing of the acts around
    // it -- or under it or onto it, reading the canvas: once a paint that reads
    // it is said, no act that moves, copies, reshapes or removes a piece is
    // asked after it, so the line leaves the canvas as that paint read it. Each
    // kind of act says at most so many acts (rounds). A rule read off the
    // picture so far is said after the line too, last on a tie; once one is
    // said, no act on the hand is asked: every act is drawn before the
    // paints, under the picture it read.
    bool frozen = false, picture_read = false;
    struct asked { std::function<offer()> offers; bool paint, changes_pieces; };
    const asked kinds[] = { { color_from_rule, false, false }, { move_rule, false, true },
                            { mirror_rule, false, true }, { outline_rule, false, true },
                            { recolor_rule, false, false }, { shape_rule, false, false },
                            { [&] { return key_rule(!frozen); }, false, false },
                            { removal_rule, false, true }, { ray_paint, true, false },
                            { trail_paint, true, false }, { copy_paint, true, false },
                            { repeat_paint, true, false }, { turned_paint, true, false },
                            { bar_paint, true, false }, { fill_paint, true, false }, { box_paint, true, false },
                            { room_paint, true, false }, { symmetry_paint, true, false }, { picture_rule, true, false } };
    static_assert(size(kinds) == size(inference_times::names) - inference_times::first_kind);
    // Whether a paint said reads what is down: laid under it or onto it, or the
    // whole figure where it lies.
    auto reads_what_is_down = [](const rule& said) {
        return group_of(said.as) == rule_group::paints &&
               (said.lay != rule::laid_over || said.as == rule::bridged || said.as == rule::filled ||
                said.as == rule::boxed || said.as == rule::turned_copy || said.as == rule::symmetric);
    };
    constexpr int rounds = 8;
    int rules[size(kinds)] = {};
    // A PAINT IS WORTH WHAT IT SAYS AND WHAT IT LETS BE READ OFF THE PICTURE
    // RIGHT AFTER IT: 0d87d2a6's bars say 37 cells, and with them drawn, the
    // blocks they touch take their color, 224 more; weighed alone, the bars
    // came after "the largest becomes 1" (140), which left the first pair's
    // one touched block nothing for the picture to change.
    // A paint said for what it lets be read is followed at once by that reading,
    // so the line keeps the worth it was chosen for: 833966f4's rays, said ahead
    // on that promise, were followed by acts on the hand instead, a coincidence
    // of its pairs that is wrong on its test. The picture can say no more than
    // every recolored cell still unclaimed, so a paint that could not win even
    // so is not read ahead.
    // A PASS AFTER THE FIRST SAYS ONE CHANGE (read_the_rest): the acts on its
    // hand, which lay the hand in one sentence, or else one paint. Its sentence
    // reads the picture the passes before it drew, as the program hands it over
    // (Output); a second sentence would read a picture the first had changed,
    // one this reading never looked at, and the next pass looks at it instead.
    const bool one_change = inf.pass > 1;
    auto hand_said = [&] {
        return ranges::any_of(read.line, [&](int8_t which) {
            const auto group = group_of(read.fired[which].as);
            return group == rule_group::recolors || group == rule_group::changes_pieces ||
                   group == rule_group::removes;
        });
    };
    auto said_last = [&] {
        frozen = frozen || reads_what_is_down(read.fired[read.line.back()]);
        picture_read = picture_read || read.fired[read.line.back()].on_the_picture;
    };
    // THE PARTNER PICKS, where the reading asks them (inference::partners):
    // every act and every paint is asked with them too.
    if (inf.partners) {
        hand_picks.partners = true;
        asked_now.insert(asked_now.end(), paints.asked_partner_picks.begin(), paints.asked_partner_picks.end());
    }
    while (true) {
        lay_the_canvas();
        ranges::fill(paint_colors, (uint16_t)0);
        for (size_t which = 0; which < read.pairs.size(); ++which) {
            const grid& output = opened.train[which].output;
            for (const auto& cell : read.pairs[which].left_over)
                if (cell.claimed_by < 0 && !line_shows(which, cell) && output.contains(cell.row, cell.col)) {
                    const int color = output.data[cell.row][cell.col];
                    if (is_color(color)) paint_colors[which] |= (uint16_t)(1u << color);
                }
        }
        own_color_picks.clear();
        for (const size_t at : asked_now)
            if (paint_has_color(paints.colors_by_pick[at])) own_color_picks.push_back(at);
        int recolors_left = 0;
        for (const auto& pair : read.pairs)
            for (const auto& cell : pair.left_over)
                recolors_left += cell.claimed_by < 0 && cell.answers_to(missing::a_recolor);
        offer best;
        int best_worth = 0, best_unlocks = 0;
        size_t chosen = 0;
        // THE SURE RULES FIRST: every kind is asked with rules that lay no cell
        // wrong, and only where none says anything, again with rules that may,
        // where the line draws its picture (brings_nearer). Asked together, a
        // rule that lays cells wrong said so much that it went first and
        // crowded out the exact rules (f0df5ff0's 1s grown every way until they
        // meet, over its rays one cell long); never asked, a change painted
        // whole and its exception set right after could not be said
        // (516b51b7's blocks, filled inside in 2, the rings within set right
        // by the passes after).
        for (const bool lays_wrong : { false, true }) {
            facts.tolerant = lays_wrong && facts.hand_known && facts.overlays;
            if (lays_wrong && !facts.tolerant) break;
            opened.tolerant = facts.tolerant;
            for (size_t kind = 0; kind < size(kinds); ++kind) {
                if ((!kinds[kind].paint && (rules[kind] >= rounds || picture_read)) ||
                    (frozen && kinds[kind].changes_pieces) || (one_change && kinds[kind].paint && hand_said()))
                    continue;
                offer one;
                {
                    const time_into offer_clock{ inference_time.ms[inference_times::first_kind + kind] };
                    one = kinds[kind].offers();
                }
                const bool may_unlock = read_ahead && one.said > 0 && !one.drawn.empty() &&
                                        one.said + recolors_left > best_worth;
                int unlocks = 0;
                if (may_unlock) {
                    const time_into ahead_clock{ inference_time.ms[inference_times::reading_ahead] };
                    unlocks = read_off_the_picture(one.drawn).said;
                }
                if (one.said + unlocks > best_worth) {
                    best = std::move(one);
                    best_worth = best.said + unlocks;
                    best_unlocks = unlocks;
                    chosen = kind;
                }
            }
            if (best.said > 0) break;
        }
        if (best.said == 0) break;
        best.say();
        ++rules[chosen];
        said_last();
        if (one_change && kinds[chosen].paint) break;
        if (best_unlocks > 0)
            if (offer then = picture_rule(); then.said > 0) {
                then.say();
                said_last();
            }
    }

    // WHAT THE LINE DRAWS, as the rules leave it (draw_the_line): it explains
    // every cell where it shows what the output shows. d23f8c26's middle column,
    // which the matching paired with pieces the rules remove, is drawn by the
    // line where it stands.
    draw_the_line();
    read.drawn_wrong = 0;
    // Where a rule said laid a cell wrong, pair by pair.
    vector<array<array<bool, MAX_GRID_DIM>, MAX_GRID_DIM>> laid_wrong_at(laid_wrong.size());
    for (size_t which = 0; which < laid_wrong.size(); ++which) {
        for (auto& row : laid_wrong_at[which]) row.fill(false);
        for (const cell_color& cell : laid_wrong[which]) laid_wrong_at[which][cell.row][cell.col] = true;
    }
    if (!line_drawn.empty()) {
        // THE PICTURE THE LINE DRAWS: the hand as the acts leave it, and every
        // paint laid over it in the order said.
        const vector<grid> pictures = the_picture({});
        auto shows_right = [&](size_t which, int row, int col) {
            return pictures[which].data[row][col] == facts.train[which].output.data[row][col];
        };
        // WHAT THE ACTS CLAIMED, HELD TO WHAT THE LINE DRAWS. Each act claimed
        // its cells on the hand as it was said, and an act said after it can undo
        // them: 50c07299's diagonal, slid its own size and then a step more,
        // leaves the cell the first slide landed on. A cell an act on the line
        // claimed stays its own only where the picture bears it out -- on the
        // output, the picture shows the cell's color; on the input, the picture
        // shows what the output shows there, or nothing, which a paint may say.
        // So does a cell a paint claimed, where a rule laid that cell wrong.
        for (size_t which = 0; which < read.pairs.size() && which < pictures.size(); ++which)
            for (auto& cell : read.pairs[which].left_over) {
                if (cell.claimed_by < 0 || ranges::find(read.line, cell.claimed_by) == read.line.end()) continue;
                const auto group = group_of(read.fired[cell.claimed_by].as);
                if (group != rule_group::recolors && group != rule_group::changes_pieces && group != rule_group::removes &&
                    !(which < laid_wrong_at.size() && laid_wrong_at[which][cell.row][cell.col]))
                    continue;
                const int8_t there = (int8_t)pictures[which].data[cell.row][cell.col];
                if (there == (int8_t)facts.train[which].output.data[cell.row][cell.col] ||
                    (cell.from_input && there == (int8_t)TRANSPARENT))
                    continue;
                --read.fired[cell.claimed_by].cells;
                cell.claimed_by = -1;
            }
        read.list_claims(listed("Drawn By The Line", "the rules draw it as the output shows it", rule::unsaid),
                         [&](size_t which, const left_over_cell& cell) {
                             return line_shows(which, cell) && shows_right(which, cell.row, cell.col);
                         });
        // THE CELLS THE LINE DRAWS WRONG: each one a rule laid wrong that the
        // picture still shows wrong, where no cell left over on the output
        // stands for it already.
        for (size_t which = 0; which < laid_wrong.size() && which < pictures.size(); ++which) {
            bool counted[MAX_GRID_DIM][MAX_GRID_DIM] = {};
            for (const auto& cell : read.pairs[which].left_over)
                if (!cell.from_input) counted[cell.row][cell.col] = true;
            for (const cell_color& cell : laid_wrong[which])
                if (!counted[cell.row][cell.col] && !shows_right(which, cell.row, cell.col)) {
                    counted[cell.row][cell.col] = true;
                    ++read.drawn_wrong;
                }
        }
    }
    else
        // No picture to look at: every cell a rule laid wrong stays wrong.
        for (const auto& wrong_here : laid_wrong_at)
            for (const auto& row : wrong_here) read.drawn_wrong += (int)ranges::count(row, true);
    // What the line draws, for the pass after it to read (read_the_rest).
    inf.picture.clear();
    if (line_drawn.size() == facts.train.size()) inf.picture = the_picture({});
}

}  // namespace arc
