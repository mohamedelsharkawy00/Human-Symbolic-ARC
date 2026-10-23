// Which pieces a word picks, asked of the hand each pair holds (replay_facts),
// and the frame a size change is read through, chosen (choose_piece_frame).
#pragma once
#include "9_rules.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace arc {

// THE PIECES A PICK BY HOLES OR BY SIZE TAKES, as the op takes them from the
// pieces as the cut hands them over: every one with `value` holes
// (object_hole_count, the count SelectObjectsWithHoles makes), or the one `value`
// down the order of painted cells, largest or smallest first, ties keeping the
// order the pieces came in -- the insertion sort SelectLargestObject and
// SelectSmallestObject run -- or the first `value` of that order, one selector
// a rank, or every one painting at least `value` cells, or every one with a
// cell on the row or the column `value` lines past the middle of all the
// pieces paint (line_from_middle). A pick's but_those turns any of them round.
inline vector<bool> picked_pieces(const vector<object>& pieces, int8_t how, int value) {
    vector<bool> picked(pieces.size(), false);
    if (how == piece_pick::middle_row || how == piece_pick::middle_column) {
        const bool rows = how == piece_pick::middle_row;
        int lo = MAX_GRID_DIM * 2, hi = -MAX_GRID_DIM * 2;
        for (const object& piece : pieces)
            piece.each_cell([&](int row, int col, int8_t) {
                lo = min(lo, rows ? row : col);
                hi = max(hi, rows ? row : col);
            });
        const int line = line_from_middle(lo, hi, value);
        for (size_t i = 0; i < pieces.size() && line >= 0; ++i)
            pieces[i].each_cell([&](int row, int col, int8_t) { picked[i] = picked[i] || (rows ? row : col) == line; });
        return picked;
    }
    // What piece i counts, as count_of or twin_key reads it, read once a piece.
    auto keys_of = [&](auto key) {
        vector<int> keys;
        for (const object& piece : pieces)
            keys.push_back(key(piece.summary.height, piece.summary.width,
                               [&](int row, int col) { return (int)piece.data[row][col]; }));
        return keys;
    };
    if (how == piece_pick::only_one) {
        // The one piece with no twin, as `value` reads twins (unique_piece, the
        // count SelectUnique makes): by a key read once a piece where twins
        // share one (twin_key), else shape against shape.
        int found = -1;
        if (value >= (int)twin_by::colors) {
            const vector<int> keys = keys_of([&](int height, int width, auto cell) {
                return twin_key(value, height, width, cell); });
            found = lone_one((int)keys.size(), [&](int i, int j) { return keys[(size_t)i] == keys[(size_t)j]; });
        } else
            found = unique_piece(value, (int)pieces.size(),
                [&](int i) { return (int)pieces[(size_t)i].summary.width; },
                [&](int i) { return (int)pieces[(size_t)i].summary.height; },
                [&](int i, int row, int col) { return (int)pieces[(size_t)i].data[row][col]; });
        if (found >= 0) picked[(size_t)found] = true;
        return picked;
    }
    if ((how == piece_pick::largest || how == piece_pick::smallest) && value >= SIZE_RANKS) {
        // Every piece with the most -- or the fewest -- of a count (count_of),
        // ties and all, as the op picks them past its ranks.
        const vector<int> counts = keys_of([&](int height, int width, auto cell) {
            return count_of(value - SIZE_RANKS, height, width, cell); });
        each_extreme((int)counts.size(), how == piece_pick::largest, [&](int i) { return counts[(size_t)i]; },
                     [&](int i) { picked[(size_t)i] = true; });
        return picked;
    }
    if (how == piece_pick::most_common) {
        // The color most of the pieces' cells wear, the lowest on a tie, and every
        // piece wearing it -- the count SelectMostCommonColor makes.
        color_tally worn;
        for (const object& piece : pieces) piece.each_cell([&](int, int, int8_t color) { worn.add(color); });
        const int most = worn.most();
        if (most >= 0)
            for (size_t i = 0; i < pieces.size(); ++i)
                pieces[i].each_cell([&](int, int, int8_t color) { picked[i] = picked[i] || color == most; });
        return picked;
    }
    if (how == piece_pick::at_least_cells) {
        // Every piece painting at least `value` cells (paints_at_least, the count
        // SelectAtLeastCells makes).
        for (size_t i = 0; i < pieces.size(); ++i) {
            const object& piece = pieces[i];
            picked[i] = paints_at_least(piece.summary.height, piece.summary.width, [&](int row, int col) {
                return piece.data[row][col] != (int8_t)TRANSPARENT; }, value);
        }
        return picked;
    }
    if (how == piece_pick::row_band || how == piece_pick::column_band) {
        // The band of rows or of columns each piece's box sits in, as
        // overlapping_bands groups them for SelectRow, SelectColumn and Layout.
        const int count = (int)pieces.size();
        if (count == 0) return picked;
        vector<int8_t> band(count);
        bands_of(count, how == piece_pick::row_band, [&](int i) { return pieces[(size_t)i].summary.box(); },
                 band.data());
        for (int i = 0; i < count; ++i) picked[i] = band[i] == value;
        return picked;
    }
    if (how == piece_pick::by_holes) {
        for (size_t i = 0; i < pieces.size(); ++i) {
            const object& piece = pieces[i];
            picked[i] = object_hole_count(piece.summary.height, piece.summary.width, [&](int row, int col) {
                return piece.data[row][col] != (int8_t)TRANSPARENT; }) == value;
        }
        return picked;
    }
    vector<int> order(pieces.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    const bool largest = how == piece_pick::largest || how == piece_pick::largest_ranks;
    stable_sort(order.begin(), order.end(), [&](int one, int other) {
        return largest ? pieces[one].summary.count > pieces[other].summary.count
                       : pieces[one].summary.count < pieces[other].summary.count;
    });
    if (how == piece_pick::largest || how == piece_pick::smallest) {
        if (value >= 0 && value < (int)order.size()) picked[order[value]] = true;
        return picked;
    }
    for (int rank = 0; rank < value && rank < (int)order.size(); ++rank) picked[order[rank]] = true;
    return picked;
}

static_assert(row_of(node_type::WouldFit).high + 1 == (int)partner_by::count,
              "WouldFit takes one value for each way partner_by finds a partner");
static_assert(row_of(node_type::SelectLikeKey).high + 1 == KEY_LIKENESSES,
              "SelectLikeKey takes one value for each likeness to the key");
static_assert(row_of(node_type::SelectPartner).high + 1 == SELECT_PARTNER_VALUES,
              "SelectPartner takes each partner way on any side, then on each of the eight directions");

// THE PIECES A PICK TAKES, as its words take them on a canvas the size of
// `canvas`, in the order pick_words says them: its colors or its kind; the
// pieces in the part those fence off, where it names a part; their partners,
// where it names a partner; every piece but those, where it turns them round;
// then WouldFit's and SelectLikeKey's keeps.
inline vector<bool> picked_pieces(const vector<object>& pieces, const piece_pick& pick, tile_size canvas,
                                  hand_answers& answers) {
    vector<bool> picked(pieces.size(), false);
    if (pick.how >= 0) picked = picked_pieces(pieces, pick.how, pick.value);
    else
        for (size_t i = 0; i < pieces.size(); ++i) picked[i] = (palette(pieces[i]) & pick.colors) != 0;
    if (pick.part >= 0) {
        // SelectPartition: the pick as it stands is the walls, and the pieces
        // lying in the part `part` they fence off are picked instead.
        const partner_pieces walls(pieces, picked, answers);
        int16_t part[MAX_GRID_DIM][MAX_GRID_DIM];
        parts_the_pick_fences(walls, canvas.width, canvas.height, part);
        for (size_t i = 0; i < pieces.size(); ++i) picked[i] = lies_in_part(walls, (int)i, pick.part, part);
    }
    if (pick.partner >= 0) {
        // SelectPartner: the pick moves onto every partner the pieces it took
        // have, among every piece, by the way and on the side it names.
        const partner_pieces taken(pieces, picked, answers);
        vector<bool> partners(pieces.size(), false);
        each_partner_picked(pick.partner, taken, [&](int other) { partners[(size_t)other] = true; });
        picked = std::move(partners);
    }
    // Every piece but those, as InvertSelection turns them round.
    if (pick.but_those) picked.flip();
    if (pick.fits >= 0) {
        // WouldFit keeps those that would fit into a gap of their partner among
        // the pieces not picked, every partner read off the pick as it stood, as
        // the op reads its marks before it drops any.
        const partner_pieces places(pieces, picked, answers);
        for (size_t i = 0; i < pieces.size(); ++i)
            if (places.picked[i]) picked[i] = places.fits((partner_by)pick.fits, (int)i);
    }
    if (pick.like >= 0) {
        // SelectLikeKey keeps those like a piece of the key, the pieces not
        // picked, read off the pick as it stood.
        const partner_pieces key(pieces, picked, answers);
        for (size_t i = 0; i < pieces.size(); ++i)
            if (key.picked[i]) picked[i] = like_the_key(pick.like, (int)i, key);
    }
    return picked;
}

inline partner_pieces::partner_pieces(const vector<object>& cut, const piece_pick& pick, tile_size canvas,
                                      hand_answers& kept)
    : partner_pieces(cut, picked_pieces(cut, pick, canvas, kept), kept) {}

static_assert(row_of(node_type::SelectLargestObject).low == row_of(node_type::SelectSmallestObject).low &&
              row_of(node_type::SelectLargestObject).high == row_of(node_type::SelectSmallestObject).high,
              "the largest and the smallest are asked as far down the order");

// WHAT EVERY REPLAY BELOW IS HANDED: the training pairs, each input's pieces as
// the cut hands them over, cut once, and the grounds the pairs were read on.
//
// AND WHAT EVERY ONE ASKS OF A WORD: that it is seen at work in every pair. A
// word that does nothing in some pair holds there only by standing idle -- a
// band of rows that pair's pieces do not make, a color it does not show -- and
// a word that holds by standing idle is a word for the pairs, not a rule: the
// test lays its bands and colors out afresh (22a4bbc2's rows, e3fe1151's
// colors). At work is the one thing every replay asks the same way: at some
// cell the word acts on, the output shows what its picked pieces did not show
// there before -- a piece recolored, moved, copied, grown or cut back, never
// only drawn again as it was. A copy laid on its own likeness is at work all
// the same: 4938f0c2's last pair is already whole.
//
// BUT A FINISHED PAIR, whose grid read already shows its output, has nothing
// left for any word to do, so it asks only that a word lays nothing wrong
// there (the user): a pass after the first finds some pairs done and others
// not (1190bc91's pair 2, whole once its rays are drawn, while each marker of
// the other pairs still fills its part).
struct replay_facts {
    const vector<example>& train;
    // Whether each pair's grid read is already its output, and how many pairs'
    // are not (finished).
    vector<bool> done;
    int open = 0;
    bool overlays = true;
    // Whether `cuts` is still the hand the line leaves: false once a word on it
    // is one carry_the_hand cannot carry the pieces through.
    bool hand_known = true;
    // Whether the line lays its hand under what is down, as Underlay does, where
    // some piece sits on another -- else over it, as Union does (draw_the_hand).
    bool laid_under = false;
    // Whether a word may lay cells wrong where it brings every pair nearer
    // (brings_nearer): only on a line that draws its picture, so the next pass
    // can read the cells it left wrong. Off everywhere else.
    bool tolerant = false;
    // The pieces a named cut hands over, rather than the elected one.
    replay_facts(const vector<example>& pairs, segmentation_methods cut,
        const tiled_pieces& tiles) : train(pairs) {
        for (const example& pair : train) {
            hands.push_back(pieces_as_said(pair.input, cut, tiles.of(pair)));
            done.push_back(pair.input == pair.output);
            open += !done.back();
            overlays = overlays && pair.input.same_size(pair.output);
        }
        answers.resize(hands.size());
    }
    replay_facts(const inference& inf, const vector<example>& pairs, const tiled_pieces& tiles)
        : replay_facts(pairs, inf.correspondence.segmentation, tiles) {}
    // The pieces a named cut hands over of `grids`, one a pair, rather than of
    // the inputs: the picture the line has drawn so far, cut again.
    replay_facts(const vector<example>& pairs, const vector<grid>& grids, segmentation_methods cut,
        const tiled_pieces& tiles) : train(pairs) {
        for (size_t which = 0; which < train.size() && which < grids.size(); ++which) {
            hands.push_back(pieces_as_said(grids[which], cut, tiles.of(train[which])));
            done.push_back(grids[which] == train[which].output);
            open += !done.back();
            overlays = overlays && train[which].input.same_size(train[which].output);
        }
        answers.resize(hands.size());
    }
    // The canvas pair `which`'s pieces lie on: its input's, as the ops take it.
    tile_size canvas(size_t which) const { return { train[which].input.width, train[which].input.height }; }
    // Whether pair `which` is finished: its grid read already its output,
    // while some pair's is not. Where every pair's is, no word has anything to
    // do anywhere, and none is let off (cd3c21df's framed pieces, each already
    // its output: a mirror that drew them again was said).
    bool finished(size_t which) const { return open > 0 && which < done.size() && done[which]; }
    bool any_finished() const { return open > 0 && ranges::find(done, true) != done.end(); }
    // WHETHER A GRID HOLDS ONE SHAPE AT TWO SIZES: two pieces of one pair's
    // hand whose boxes differ while their shape at any size -- scaled
    // [0], or stretched [1] -- is one (key_of_shape). Only there does a
    // likeness at any size, or stretched, say what the likeness at its own
    // size does not: elsewhere the two find the same partners, and the one at
    // its own size is asked first. Read once, and again once the hand changes.
    bool shape_at_two_sizes(bool stretched) const {
        if (sizes_seen < 0) {
            sizes_seen = 0;
            for (const vector<object>& cut : hands)
                for (const shape_freedom free : { shape_freedom{ true, true }, shape_freedom{ true, false, true } }) {
                    vector<uint32_t> keys;
                    for (const object& piece : cut)
                        keys.push_back(key_of_shape(piece.summary.width, piece.summary.height,
                                                    [&](int row, int col) { return (int)piece.data[row][col]; }, free));
                    for (size_t i = 0; i < cut.size(); ++i)
                        for (size_t j = i + 1; j < cut.size(); ++j)
                            if (keys[i] == keys[j] && (cut[i].summary.width != cut[j].summary.width ||
                                                       cut[i].summary.height != cut[j].summary.height))
                                sizes_seen |= (int8_t)(free.stretch ? 2 : 1);
                }
        }
        return (sizes_seen >> (stretched ? 1 : 0) & 1) != 0;
    }
    // Whether a likeness, a partner way, or a partner of any kind
    // (partner_of_kind) is worth asking of this hand (shape_at_two_sizes).
    bool asks_likeness(int like) const { return !sized_likeness(like) || shape_at_two_sizes(like / 2 == 2); }
    bool asks_way(partner_by how) const { return likeness_of(how) < 0 || asks_likeness(likeness_of(how)); }
    bool asks_kind(int kind) const { return kind >= (int)partner_by::count || asks_way((partner_by)kind); }
    mutable int8_t sizes_seen = -1;
    // A hand changes only through change(), so nothing read off it outlives it.
    const vector<vector<object>>& cuts() const { return hands; }
    hand_answers& answers_of(size_t which) const { return answers[which]; }
    vector<bool> picked(size_t which, const piece_pick& pick) const {
        return picked_pieces(hands[which], pick, canvas(which), answers[which]);
    }
    partner_pieces handed(size_t which, const piece_pick& pick) const {
        return { hands[which], pick, canvas(which), answers[which] };
    }
    // Pair `which`'s hand, open to change; what was read off it is forgotten after.
    class hand_change {
    public:
        vector<object>& hand;
        hand_change(replay_facts& facts, size_t which) : hand(facts.hands[which]), facts(facts), which(which) {}
        hand_change(const hand_change&) = delete;
        hand_change& operator=(const hand_change&) = delete;
        ~hand_change() { facts.forget(which); }
    private:
        replay_facts& facts;
        size_t which;
    };
    hand_change change(size_t which) { return { *this, which }; }
private:
    vector<vector<object>> hands;
    mutable vector<hand_answers> answers;
    void forget(size_t which) {
        answers[which].forget();
        sizes_seen = -1;
    }
};

// THE PIECES A PICK TAKES, pair by pair, made ready once for a word asked many
// ways over -- every length, both ways round, every lay -- whose tries differ in
// everything but the pieces. Kept only while that word is asked, and only while
// nothing carries the hand: `facts` must not change under it.
inline vector<partner_pieces> handed_pieces(const replay_facts& facts, const piece_pick& pick) {
    vector<partner_pieces> handed;
    handed.reserve(facts.cuts().size());
    for (size_t which = 0; which < facts.cuts().size(); ++which) handed.push_back(facts.handed(which, pick));
    return handed;
}

// Whether two picks take the same pieces in every pair: each replay is handed
// the same pieces by both, and sees the same.
inline bool takes_the_same(const vector<partner_pieces>& one, const vector<partner_pieces>& other) {
    return ranges::equal(one, other, [](const partner_pieces& a, const partner_pieces& b) { return a.picked == b.picked; });
}

// Local to one word on one unchanged hand. Record only eligible picks, so the
// first pick allowed to ask a value keeps its tie, even if another was skipped.
struct asked_piece_sets {
    vector<vector<partner_pieces>> asked;
    bool before(const vector<partner_pieces>& handed) {
        if (ranges::any_of(asked, [&](const auto& prior) { return takes_the_same(prior, handed); })) return true;
        asked.push_back(handed);
        return false;
    }
};

// Whether a pick by `colors` picks by the same colors in every pair: each pair's
// pieces hold the same ones of them. A pick whose colors change from pair to pair
// names each pair's own colors rather than a rule, and a test brings its own.
inline bool same_colors_every_pair(const replay_facts& facts, uint16_t colors) {
    bool first = true;
    uint16_t seen = 0;
    for (const auto& cut : facts.cuts()) {
        uint16_t held = 0;
        for (const object& piece : cut) held = (uint16_t)(held | (palette(piece) & colors));
        if (first) seen = held;
        else if (held != seen) return false;
        first = false;
    }
    return true;
}

// THE PARTNERS OF A PICK (SelectPartner): of every piece, and of each color
// every pair holds, every partner each has by a way partner_by finds one -- on
// any side, then wholly on each side and each corner in turn (direction_asked) --
// and every piece but those, each asked only where it takes a piece in every
// pair and leaves one in some, and takes other pieces somewhere than its base
// and every partner pick asked before it (4364c1c4: of every piece, the
// partners above, the upper piece of each stack). Each way's partners are found
// once a pair (each_partner, among every piece, as each_partner_picked finds
// them), for every base and side.
// A LAST RESORT: a line asks them only once nothing else it asks says any more
// (finish_the_line), so a line the other picks finish reads as it did -- asked
// with the rest, they said more at once in lines the others finish, and steered
// them wrong (22168020, 272f95fa, 62b74c02, 9344f635, a61f2674, f823c43c).
inline vector<piece_pick> partner_picks(const replay_facts& facts) {
    vector<piece_pick> picks;
    // Each partner found with the sides of its piece it lies on, a bit for each direction.
    struct found_partner { int mine, other; uint8_t partner_directions; };
    vector<vector<vector<found_partner>>> partners_by_way((size_t)partner_by::count);
    for (int way = 0; way < (int)partner_by::count; ++way) {
        if (!facts.asks_way((partner_by)way)) continue;
        for (size_t which = 0; which < facts.cuts().size(); ++which) {
            const vector<object>& cut = facts.cuts()[which];
            const partner_pieces every(cut, vector<bool>(cut.size(), true), facts.answers_of(which));
            vector<found_partner>& found = partners_by_way[(size_t)way].emplace_back();
            for (int mine = 0; mine < every.count(); ++mine)
                each_partner((partner_by)way, mine, every, [&](int other) {
                    uint8_t partner_directions = 0;
                    for (int partner_direction = 0; partner_direction < direction::count; ++partner_direction)
                        if (lies_toward(every.box(mine), every.box(other), partner_direction)) partner_directions = (uint8_t)(partner_directions | 1u << partner_direction);
                    found.push_back({ mine, other, partner_directions });
                }, true);
        }
    }
    vector<uint16_t> bases = { piece_pick::all().colors };
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (ranges::all_of(facts.cuts(), [&](const vector<object>& cut) {
                return ranges::any_of(cut, [&](const object& piece) { return (palette(piece) >> color & 1) != 0; });
            }))
            bases.push_back(piece_pick::of_color(color).colors);
    // The ways asked touch first, at a side and then at a corner, and then in
    // partner_by's order: a piece it touches is its partner whatever lies in
    // line with it, which a move can change (4364c1c4's test: once each upper
    // piece stepped left, the lower stack's top piece came into the columns of
    // the upper stack's bottom one, which "a piece in its columns" then took for
    // its partner too, and it did not step right).
    vector<int> ways_asked = { (int)partner_by::touching_sides, (int)partner_by::touching };
    for (int way = 0; way < (int)partner_by::count; ++way)
        if (ranges::find(ways_asked, way) == ways_asked.end()) ways_asked.push_back(way);
    vector<vector<vector<bool>>> partner_picks_taken;
    for (const uint16_t base : bases) {
        vector<vector<bool>> base_taken;
        for (size_t which = 0; which < facts.cuts().size(); ++which)
            base_taken.push_back(facts.picked(which, piece_pick{ base }));
        for (const int way : ways_asked) {
            if (partners_by_way[(size_t)way].empty()) continue;
            for (int at = -1; at < direction::count; ++at) {
                const int partner_direction = at < 0 ? -1 : direction_asked(at);
                vector<vector<bool>> taken;
                bool every = true, splits = false;
                for (size_t which = 0; which < facts.cuts().size(); ++which) {
                    vector<bool>& one = taken.emplace_back(facts.cuts()[which].size(), false);
                    for (const found_partner& found : partners_by_way[(size_t)way][which])
                        if (base_taken[which][(size_t)found.mine] && (partner_direction < 0 || (found.partner_directions >> partner_direction & 1u)))
                            one[(size_t)found.other] = true;
                    const bool some = ranges::find(one, true) != one.end();
                    every = every && some;
                    splits = splits || (some && ranges::find(one, false) != one.end());
                }
                if (!every || !splits || taken == base_taken) continue;
                vector<vector<bool>> turned = taken;
                for (vector<bool>& one : turned) one.flip();
                if (ranges::find(partner_picks_taken, taken) != partner_picks_taken.end() ||
                    ranges::find(partner_picks_taken, turned) != partner_picks_taken.end())
                    continue;
                partner_picks_taken.push_back(std::move(taken));
                partner_picks_taken.push_back(std::move(turned));
                piece_pick partnered{ base };
                partnered.partner = (int8_t)((partner_direction + 1) * (int)partner_by::count + way);
                for (const bool but_those : { false, true }) {
                    partnered.but_those = but_those;
                    picks.push_back(partnered);
                }
            }
        }
    }
    return picks;
}

// Every pick a word is asked with: the colors an unsaid change starts from, when
// there are any, then every pick by holes, size, band, the color most cells
// wear and how many cells the ops can make, then
// each of `fit_colors` alone kept to its pieces that would fit into a gap of
// their partner, each partner way. Colors that change from pair to pair are
// asked after those, so a pick by holes or size that accounts for as much wins
// the tie (aabf363d's shape, a new color in its test); then each of the colors
// alone; then the parts one color's walls fence off, and the keys a pick turned
// round leaves to be like.
inline vector<piece_pick> picks_to_ask(const replay_facts& facts, uint16_t colors, uint16_t fit_colors = 0,
                                       bool partners = false) {
    vector<piece_pick> picks;
    const bool steady = colors && same_colors_every_pair(facts, colors);
    if (steady) picks.push_back({ colors });
    const auto holes = row_of(node_type::SelectObjectsWithHoles);
    for (int value = holes.low; value <= holes.high; ++value) picks.push_back({ 0, piece_pick::by_holes, (int8_t)value });
    // Sort each pair once for both size orders and every rank asked below.
    vector<vector<int>> sizes;
    for (const auto& cut : facts.cuts()) {
        vector<int> counts;
        for (const object& piece : cut) counts.push_back(piece.summary.count);
        ranges::sort(counts);
        sizes.push_back(std::move(counts));
    }
    auto size_at = [](const vector<int>& counts, size_t at, bool largest) {
        return counts[largest ? counts.size() - 1 - at : at];
    };
    // By size, the nearer an end the plainer: the largest and the smallest, then
    // the second of each, so where two take the same pieces in every pair the
    // plainer is said (62ab2642's "smallest", not "ninth largest", which only
    // the pairs' own counts made the same). Each asked only where no pair's
    // pieces tie at that place in the order: a tie there is broken by the order
    // the pieces came in, a pick by where they lie and not by size (833966f4's
    // cells, one cell each, the "second largest" only the second scanned). Nor
    // anywhere before it: pieces of one size ahead of it make its place count
    // how many share a size, not how large it is (272f95fa's center, "the fifth
    // smallest" only past three panes of 8 cells; three of 18 on its test put
    // another pane fifth).
    for (int value = 0; value < SIZE_RANKS; ++value)
        for (const int8_t how : { (int8_t)piece_pick::largest, (int8_t)piece_pick::smallest })
            if (ranges::all_of(sizes, [&](const vector<int>& counts) {
                    const size_t at = (size_t)value;
                    if (at >= counts.size()) return true;
                    const bool largest = how == piece_pick::largest;
                    for (size_t i = 0; i <= at && i + 1 < counts.size(); ++i)
                        if (size_at(counts, i, largest) == size_at(counts, i + 1, largest)) return false;
                    return true;
                }))
                picks.push_back({ 0, how, (int8_t)value });
    // The bands, as many as some pair's pieces make.
    for (const int8_t how : { (int8_t)piece_pick::row_band, (int8_t)piece_pick::column_band }) {
        int bands = 0;
        for (const auto& cut : facts.cuts())
            for (int value = 0; value < (int)cut.size(); ++value)
                if (ranges::any_of(picked_pieces(cut, how, value), [](bool one) { return one; }))
                    bands = max(bands, value + 1);
        for (int value = 0; value < bands && value <= row_of(node_type::SelectRow).high; ++value)
            picks.push_back({ 0, how, (int8_t)value });
    }
    for (const bool but_those : { false, true }) picks.push_back({ 0, piece_pick::most_common, 0, -1, but_those });
    // By how many cells, at least so many or fewer, each count asked only where
    // it splits some pair's pieces: one at least taken there and one left.
    const auto cells = row_of(node_type::SelectAtLeastCells);
    for (int value = cells.low; value <= cells.high; ++value) {
        if (ranges::none_of(facts.cuts(), [&](const vector<object>& cut) {
                const vector<bool> taken = picked_pieces(cut, piece_pick::at_least_cells, value);
                return ranges::find(taken, true) != taken.end() && ranges::find(taken, false) != taken.end();
            }))
            continue;
        for (const bool but_those : { false, true })
            picks.push_back({ 0, piece_pick::at_least_cells, (int8_t)value, -1, but_those });
    }
    // The first so many by size, largest or smallest first, and every piece but
    // those (9caf5b84: every color but the two most common becomes 7), each asked
    // only where in every pair the order parts there, the last piece taken
    // painting more cells than the first left -- or fewer -- so no tie the order
    // the pieces came in breaks says which. The first one alone is the pick by
    // size above, so the first so many are asked from two.
    for (const bool largest : { true, false })
        for (int value = 1; value <= SIZE_RANKS; ++value) {
            if (!ranges::all_of(sizes, [&](const vector<int>& counts) {
                    if ((int)counts.size() <= value) return false;
                    return size_at(counts, (size_t)value - 1, largest) != size_at(counts, (size_t)value, largest);
                }))
                continue;
            const int8_t first_so_many = largest ? piece_pick::largest_ranks : piece_pick::smallest_ranks;
            if (value >= 2) picks.push_back({ 0, first_so_many, (int8_t)value });
            picks.push_back({ 0, first_so_many, (int8_t)value, -1, true });
        }
    // The row or the column so many lines past the middle of all the pieces
    // paint, and every piece but those (d23f8c26 keeps its middle column), the
    // middle first and then out a line at a time, each asked only where every
    // pair has that line and it splits some pair's pieces: one at least taken
    // there and one left.
    constexpr auto from_middle = row_of(node_type::SelectRowFromMiddle);
    static_assert(from_middle.high == (MAX_GRID_DIM - 1) / 2 && from_middle.low == -from_middle.high &&
                  row_of(node_type::SelectColumnFromMiddle).low == from_middle.low &&
                  row_of(node_type::SelectColumnFromMiddle).high == from_middle.high,
                  "a line from the middle reaches either edge of the largest grid");
    for (const bool rows : { true, false })
        for (int at = 0; at <= 2 * from_middle.high; ++at) {
            // 0, -1, 1, -2, 2, ...
            const int value = (at + 1) / 2 * (at % 2 ? -1 : 1);
            const int8_t how = rows ? piece_pick::middle_row : piece_pick::middle_column;
            bool every = true, splits = false;
            for (const auto& cut : facts.cuts()) {
                const vector<bool> taken = picked_pieces(cut, how, value);
                const bool some = ranges::find(taken, true) != taken.end();
                every = every && some;
                splits = splits || (some && ranges::find(taken, false) != taken.end());
            }
            if (!every || !splits) continue;
            for (const bool but_those : { false, true }) picks.push_back({ 0, how, (int8_t)value, -1, but_those });
        }
    // Whether holds(which, the pieces `pick` takes) in every pair.
    auto every_pair = [&](const piece_pick& pick, auto holds) {
        for (size_t which = 0; which < facts.cuts().size(); ++which)
            if (!holds(which, facts.picked(which, pick))) return false;
        return true;
    };
    // A fit is asked only where it is at work in every pair: it keeps one piece
    // of the color at least and drops one, or the color alone says as much.
    for (int color = 0; color < COLOR_COUNT; ++color) {
        if (!(fit_colors >> color & 1)) continue;
        const piece_pick alone = piece_pick::of_color(color);
        for (int way = 0; way < (int)partner_by::count; ++way) {
            if (!facts.asks_way((partner_by)way)) continue;
            const piece_pick fit{ alone.colors, -1, 0, (int8_t)way };
            if (every_pair(fit, [&](size_t which, const vector<bool>& kept) {
                    return ranges::find(kept, true) != kept.end() && kept != facts.picked(which, alone);
                }))
                picks.push_back(fit);
        }
    }
    if (colors && !steady) picks.push_back({ colors });
    // Then each of those colors alone, where there are several and every pair
    // holds it or none does: a change of one of them is said by its own pick,
    // where the matching's acts are left unsaid (18286ef8's 9 steps toward the
    // 6 while the 5 it steps onto and the 6 change too).
    if (popcount(colors) > 1)
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (const piece_pick alone = piece_pick::of_color(color);
                colors >> color & 1 && same_colors_every_pair(facts, alone.colors))
                picks.push_back(alone);
    // THE PARTS ONE COLOR'S WALLS FENCE OFF (parts_the_pick_fences): the pieces
    // lying in one part, and every piece but those, asked only where every pair
    // holds the color and its walls fence two parts at least in every pair, and
    // only for a part every pair has that holds a piece in every pair
    // (63613498's grey corner, its key alone in part 0). Then, on every pick
    // that turns its pieces round, SelectLikeKey: the ones it keeps that are
    // like a piece of the key, the pieces it turned away, asked only where it
    // keeps one in every pair and drops one in some -- a key fenced off by walls
    // first, the surer key, so it takes a tie (63613498's corner, not the column
    // three left of the middle, which leaves its key and an 8 besides). The
    // newest picks, asked last, so an older pick that says as much takes the tie.
    auto takes_one = [](size_t, const vector<bool>& taken) { return ranges::find(taken, true) != taken.end(); };
    for (int color = 0; color < COLOR_COUNT && !facts.cuts().empty(); ++color) {
        const piece_pick walls = piece_pick::of_color(color);
        if (!every_pair(walls, takes_one)) continue;
        int parts = row_of(node_type::SelectPartition).high + 1;
        for (size_t which = 0; which < facts.cuts().size(); ++which) {
            const tile_size canvas = facts.canvas(which);
            int16_t part[MAX_GRID_DIM][MAX_GRID_DIM];
            parts = min(parts, parts_the_pick_fences(facts.handed(which, walls), canvas.width, canvas.height, part));
        }
        if (parts < 2) continue;
        for (int which_part = 0; which_part < parts; ++which_part) {
            piece_pick in_part = walls;
            in_part.part = (int8_t)which_part;
            if (!every_pair(in_part, takes_one)) continue;
            for (const bool but_those : { false, true }) {
                in_part.but_those = but_those;
                picks.push_back(in_part);
            }
        }
    }
    const size_t asked = picks.size();
    for (const bool fenced : { true, false })
        for (size_t at = 0; at < asked; ++at) {
            if (!picks[at].but_those || picks[at].fits >= 0 || (picks[at].part >= 0) != fenced) continue;
            for (int like = 0; like < KEY_LIKENESSES; ++like) {
                if (!facts.asks_likeness(like)) continue;
                piece_pick liked = picks[at];
                liked.like = (int8_t)like;
                bool drops = false;
                const bool keeps = every_pair(liked, [&](size_t which, const vector<bool>& kept) {
                    drops = drops || kept != facts.picked(which, picks[at]);
                    return takes_one(which, kept);
                });
                if (keeps && drops) picks.push_back(liked);
            }
        }
    // THE NEWEST PICKS, ASKED LAST, so an older pick that says as much takes the
    // tie (a61f2674's bars: a count asked ahead of the ranks by size took
    // their words and left its line unfinished).
    // The one piece with no twin, each way twins are read (unique_piece): asked
    // only where it takes a piece in every pair, a twin being what tells the
    // rest apart from it.
    for (int value = 0; value < (int)twin_by::count; ++value)
        if (ranges::all_of(facts.cuts(), [&](const vector<object>& cut) {
                return ranges::any_of(picked_pieces(cut, piece_pick::only_one, value), [](bool one) { return one; });
            }))
            for (const bool but_those : { false, true })
                picks.push_back({ 0, piece_pick::only_one, (int8_t)value, -1, but_those });
    // The most and the fewest of each count past size (count_of): the pieces
    // with the most colors, holes or cells of one color, or the fewest, and
    // every piece but those, asked only where the count parts some pair's
    // pieces: one at least taken there and one left. A count of one color's
    // cells only where some piece mixes that color with another: among pieces
    // of one color each it is the size of the ones of that color, which the
    // ranks by size say -- asked anyway, it peeled a61f2674's bars off one at
    // a time and left its line unfinished.
    auto mixes = [&](int color) {
        return ranges::any_of(facts.cuts(), [&](const vector<object>& cut) {
            return ranges::any_of(cut, [&](const object& piece) {
                const uint16_t worn = palette(piece);
                return (worn >> color & 1) != 0 && (worn & ~(1u << color)) != 0;
            });
        });
    };
    for (int kind = 0; kind < (int)count_kind::count; ++kind) {
        if (kind == (int)count_kind::cells) continue;
        if (kind >= (int)count_kind::of_color && !mixes(kind - (int)count_kind::of_color)) continue;
        for (const int8_t how : { (int8_t)piece_pick::largest, (int8_t)piece_pick::smallest }) {
            const int value = SIZE_RANKS + kind;
            if (ranges::none_of(facts.cuts(), [&](const vector<object>& cut) {
                    const vector<bool> taken = picked_pieces(cut, how, value);
                    return ranges::find(taken, true) != taken.end() && ranges::find(taken, false) != taken.end();
                }))
                continue;
            for (const bool but_those : { false, true })
                picks.push_back({ 0, how, (int8_t)value, -1, but_those });
        }
    }
    if (partners)
        for (const piece_pick& pick : partner_picks(facts)) picks.push_back(pick);
    return picks;
}

// Picks depend on the hand and the requested colors, not on claims or the
// canvas painted beside it. Keep their order, and forget them when carried.
// With `partners`, once the line has turned to its last resort, the partner
// picks too (partner_picks).
struct picks_on_hand {
    bool partners = false;
    std::map<std::tuple<uint16_t, uint16_t, bool>, vector<piece_pick>> asked;
    const vector<piece_pick>& get(const replay_facts& facts, uint16_t colors, uint16_t fit_colors = 0) {
        const auto [at, fresh] = asked.try_emplace({ colors, fit_colors, partners });
        if (fresh) at->second = picks_to_ask(facts, colors, fit_colors, partners);
        return at->second;
    }
    void clear() { asked.clear(); }
};

// Whether two pieces draw the same thing: the same cells in the same colors,
// wherever each lies.
inline bool same_drawing(const object& one, const object& other) {
    return same_shape(one.summary.width, one.summary.height, [&](int row, int col) { return (int)one.data[row][col]; },
                      other.summary.width, other.summary.height,
                      [&](int row, int col) { return (int)other.data[row][col]; }, DRAWING);
}

// WHETHER A CANVAS SHOWS WHAT IT FRAMES: all of `drawing`, or the corner of it
// that says the rest -- half of it as tall or as wide, rounded up, on every
// side the canvas cuts, the drawing coming back on itself there: mirrored
// across the cut, or a half turn for a half, a quarter turn for a quarter, as
// 2013d3e2's figure does in its corner. Any other cut shows whatever happened
// to lie in the corner: 39a8645d's shapes, laid out, with one of them there, or
// its layer, scattered, with one copy in it.
inline bool shows_what_it_frames(const grid& drawing, int height, int width) {
    const int tall = drawing.height, wide = drawing.width;
    const bool cut_down = height < tall, cut_across = width < wide;
    if (!cut_down && !cut_across) return true;
    auto half = [](int whole, int kept) { return (whole + 1) / 2 == kept; };
    if ((cut_down && !half(tall, height)) || (cut_across && !half(wide, width))) return false;
    // Whether the drawing is the same read through `at`, cell for cell.
    auto comes_back = [&](auto at) {
        for (int row = 0; row < tall; ++row)
            for (int col = 0; col < wide; ++col) {
                const auto [from_row, from_col] = at(row, col);
                if (drawing.data[row][col] != drawing.data[from_row][from_col]) return false;
            }
        return true;
    };
    const bool across_mirror = comes_back([&](int row, int col) { return std::pair{ row, wide - 1 - col }; });
    const bool down_mirror = comes_back([&](int row, int col) { return std::pair{ tall - 1 - row, col }; });
    const bool half_turn = comes_back([&](int row, int col) { return std::pair{ tall - 1 - row, wide - 1 - col }; });
    if (cut_down && cut_across)
        return (across_mirror && down_mirror) ||
               (tall == wide && comes_back([&](int row, int col) { return std::pair{ col, wide - 1 - row }; }));
    return half_turn || (cut_across ? across_mirror : down_mirror);
}

// Whether a pick by size took piece `one` on a tie with a piece that draws
// something else: the op breaks the tie by where the pieces lie, so what it
// takes is the order's to say and not the rule's.
inline bool size_pick_ties(const vector<object>& pieces, const piece_pick& pick, int one) {
    if ((pick.how != piece_pick::largest && pick.how != piece_pick::smallest) || pick.value >= SIZE_RANKS) return false;
    for (size_t i = 0; i < pieces.size(); ++i)
        if ((int)i != one && pieces[i].summary.count == pieces[one].summary.count &&
            !same_drawing(pieces[i], pieces[one]))
            return true;
    return false;
}

// THE INPUT FRAMED: the pieces a pick takes, moved as one to the corner of the
// canvas the size rule gives, and ground everywhere else -- what the frame's
// words leave on the canvas. Several go together, keeping their places to each
// other, as UnionList merges them; `several` says whether this pair's pick took
// more than one. False when the pick takes no piece; when the canvas does not
// show what it frames (shows_what_it_frames); and when a pick by size lands on a
// tie (size_pick_ties), since the crop would be of whichever came first.
inline bool framed_input(const vector<object>& pieces, const piece_pick& pick, const grid& input,
                         const output_dimension_rule& size, grid& framed, bool& several) {
    hand_answers fresh;
    const vector<bool> picked = picked_pieces(pieces, pick, { input.width, input.height }, fresh);
    cell_box box;
    int taken = 0, one = -1;
    for (size_t i = 0; i < picked.size(); ++i) {
        if (!picked[i]) continue;
        ++taken;
        one = (int)i;
        box.include(pieces[i].summary.box());
    }
    several = taken > 1;
    if (taken == 0 || size_pick_ties(pieces, pick, one)) return false;
    grid drawing(box.width(), box.height());
    for (size_t i = 0; i < pieces.size(); ++i)
        if (picked[i])
            pieces[i].each_cell([&](int row, int col, int8_t color) {
                drawing.data[row - box.row0][col - box.col0] = color;
            });
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        if (!shows_what_it_frames(drawing, canvas.height, canvas.width)) return false;
        for (int row = 0; row < drawing.height && row < canvas.height; ++row)
            for (int col = 0; col < drawing.width && col < canvas.width; ++col)
                if (drawing.data[row][col] != TRANSPARENT) canvas.data[row][col] = drawing.data[row][col];
        return true;
    });
}

// THE PIECES LAID OUT: the ones a pick takes, as many as one list holds, each
// drawn unresized into its own cell of graph paper `cell` wide -- or, for 0, as
// wide as the widest and as tall as the tallest -- in the row band and column
// band its box sits in (overlapping_bands, the grouping Layout and the band
// picks read), clipped to the cell, the paper laid at the corner of the canvas
// -- where each cell shows the piece it frames, and the canvas the paper
// (shows_what_it_frames).
inline bool laid_out_input(const vector<object>& pieces, const piece_pick& pick, int cell, const grid& input,
                           const output_dimension_rule& size, grid& framed) {
    hand_answers fresh;
    const vector<bool> picked = picked_pieces(pieces, pick, { input.width, input.height }, fresh);
    vector<const object*> kept;
    for (size_t i = 0; i < pieces.size() && kept.size() < (size_t)MAX_OBJECTS_IN_A_LIST; ++i)
        if (picked[i]) kept.push_back(&pieces[i]);
    const int count = (int)kept.size();
    if (count == 0) return false;
    int cell_width = cell, cell_height = cell;
    if (cell == 0)
        for (const object* piece : kept) {
            cell_width = max(cell_width, (int)piece->summary.width);
            cell_height = max(cell_height, (int)piece->summary.height);
        }
    if (cell_width <= 0 || cell_height <= 0) return false;
    vector<int8_t> row_band(count), col_band(count);
    auto kept_box = [&](int i) { return kept[(size_t)i]->summary.box(); };
    const int rows = bands_of(count, true, kept_box, row_band.data());
    const int cols = bands_of(count, false, kept_box, col_band.data());
    const int paper_width = min(cols * cell_width, MAX_GRID_DIM), paper_height = min(rows * cell_height, MAX_GRID_DIM);
    grid paper(paper_width, paper_height);
    for (int i = 0; i < count; ++i) {
        const object& piece = *kept[i];
        grid drawing(piece.summary.width, piece.summary.height);
        for (int row = 0; row < piece.summary.height; ++row)
            for (int col = 0; col < piece.summary.width; ++col) drawing.data[row][col] = piece.data[row][col];
        if (!shows_what_it_frames(drawing, cell_height, cell_width)) return false;
        const int top = row_band[i] * cell_height, left = col_band[i] * cell_width;
        for (int row = 0; row < cell_height && row < piece.summary.height; ++row)
            for (int col = 0; col < cell_width && col < piece.summary.width; ++col) {
                const int at_row = top + row, at_col = left + col;
                const int8_t color = piece.data[row][col];
                if (at_row < paper_height && at_col < paper_width && color != (int8_t)TRANSPARENT)
                    paper.data[at_row][at_col] = color;
            }
    }
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        if (!shows_what_it_frames(paper, canvas.height, canvas.width)) return false;
        for (int row = 0; row < paper.height && row < canvas.height; ++row)
            for (int col = 0; col < paper.width && col < canvas.width; ++col)
                if (paper.data[row][col] != TRANSPARENT) canvas.data[row][col] = paper.data[row][col];
        return true;
    });
}

// THE CANVAS FILLED, as the filled frame's words leave it: the pieces a pick
// takes, each recolored by the puzzle's table where `keyed` names a key
// (ShapeColorMap[keyed]), filled to its box with its most common color
// (AddColor[11]) and stretched over the canvas the size rule gives (Fit[1]),
// laid in turn. Held only where every piece the pick takes comes out in one
// color, the same for all, so the canvas is that color whichever lies last;
// false when the pick takes no piece, or a pick by size lands on a tie
// (size_pick_ties).
inline bool filled_input(const vector<object>& pieces, const piece_pick& pick, int8_t keyed,
                         const shape_color_map_rule& table, const grid& input, const output_dimension_rule& size,
                         grid& framed) {
    hand_answers fresh;
    const vector<bool> picked = picked_pieces(pieces, pick, { input.width, input.height }, fresh);
    int color = -1, one = -1;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (!picked[i]) continue;
        one = (int)i;
        // The one color the piece comes out in: -2 until a cell is read, -1 once
        // two disagree.
        int worn = -2;
        if (keyed >= 0) worn = table.color_of((shape_key)keyed, pieces[i]);
        else pieces[i].each_cell([&](int, int, int8_t was) { worn = worn == -2 || worn == was ? was : -1; });
        if (!is_color(worn) || (color >= 0 && worn != color)) return false;
        color = worn;
    }
    if (!is_color(color) || size_pick_ties(pieces, pick, one)) return false;
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        canvas.fill((int8_t)color);
        return true;
    });
}

// THE WHOLE INPUT FRAMED: scaled up `factor` times, each cell a block, as ScaleUp
// scales the input's one object; scaled down `factor` times, each block the
// color most of it wears, as ScaleDown votes (majority_vote); or tiled from the
// corner over the canvas the size rule gives, a copy per input, as Tile lays
// them by its own size (tiled_by_its_own_size) -- as many as one list holds;
// turned to a pose (posed_place, the pose the
// turns and flips said for it make); a fractal, each cell of the input scaled
// up to a copy of the input where it holds a figure, as Fractal draws the input
// without its ground; or the unit it repeats (repeat_periods), as Pattern cuts it.
// What falls off the canvas is clipped. Where a plan deleted the ground, the
// input framed is its figures alone, and every frame acts on them as they are.
inline bool framed_whole(const grid& input, const piece_frame_rule& frame, const output_dimension_rule& size,
                         grid& framed) {
    using kind = piece_frame_rule::kind_of;
    const int factor = frame.factor;
    return draw_on_canvas(input, size, framed, [&](grid& canvas) {
        switch (frame.kind) {
        case kind::posed:
            for (int row = 0; row < input.height; ++row)
                for (int col = 0; col < input.width; ++col) {
                    const auto [at_row, at_col] = posed_place(factor, input.width, input.height, row, col);
                    if (canvas.contains(at_row, at_col)) canvas.data[at_row][at_col] = input.data[row][col];
                }
            return true;
        case kind::fractal: {
            // A figure is what is not empty, as a plan that deleted the ground
            // leaves the input.
            auto figure = [&](int row, int col) { return input.data[row][col] != TRANSPARENT; };
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    if (figure(row * input.height / canvas.height, col * input.width / canvas.width) &&
                        figure(row % input.height, col % input.width))
                        canvas.data[row][col] = input.data[row % input.height][col % input.width];
            return true;
        }
        case kind::repeat_unit: {
            int unit_width = 0, unit_height = 0;
            repeat_periods(input, unit_width, unit_height);
            for (int row = 0; row < unit_height; ++row)
                for (int col = 0; col < unit_width; ++col) {
                    if (!canvas.contains(row, col)) return false;
                    canvas.data[row][col] = input.data[row][col];
                }
            return true;
        }
        case kind::scaled_up:
            if (factor < 2) return false;
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    if (row / factor < input.height && col / factor < input.width)
                        canvas.data[row][col] = input.data[row / factor][col / factor];
            return true;
        case kind::scaled_down:
            if (factor < 2) return false;
            for (int row = 0; row < canvas.height; ++row)
                for (int col = 0; col < canvas.width; ++col)
                    canvas.data[row][col] = majority_vote(input.data, row, col, factor, input.height, input.width);
            return true;
        case kind::tiled: {
            const tile_moves moves = moves_of(tiled_by_its_own_size, { input.width, input.height });
            int copies = 0;
            each_tile_offset(tiled_by_its_own_size, moves, 0, 0, input.height, input.width, canvas.width, canvas.height,
                             [&](int, int) { return ++copies <= MAX_OBJECTS_IN_A_LIST; });
            if (copies > MAX_OBJECTS_IN_A_LIST) return false;
            each_tile_offset(tiled_by_its_own_size, moves, 0, 0, input.height, input.width, canvas.width, canvas.height,
                             [&](int down, int across) {
                for (int row = 0; row < input.height; ++row)
                    for (int col = 0; col < input.width; ++col)
                        if (canvas.contains(down + row, across + col))
                            canvas.data[down + row][across + col] = input.data[row][col];
                return true;
            });
            return true;
        }
        default: return false;
        }
    });
}

// THE FRAME A SIZE CHANGE IS READ THROUGH, chosen. Of every cut with a word and
// every pick a word is asked with -- each color on its own, then the picks by
// holes, size, band and the color most cells wear -- the one whose framed
// inputs share the most cells with their outputs, counted in the pair that
// shares the fewest; a tie goes to the cut and the pick asked first. And ahead
// of them the whole input scaled or tiled, where the size rule scales it; and
// after them all the canvas filled in one color. Asked only of a size change,
// and held only where the frame lands on the output's size in every pair. A
// filled frame that reads its color off a table read off the pairs makes that
// the puzzle's table (inf.shape_color_map), the one ShapeColorMap looks up.
inline piece_frame_rule choose_piece_frame(const vector<example>& train, inference& inf, const tiled_pieces& tiles) {
    if (train.empty() || inf.size.Chosen == output_dimension_rule::Identity) return {};
    // The framed grid is of figures: no ground is filled into it, so what a
    // frame leaves unpainted is empty (draw_on_canvas).
    // A frame that leaves a cell empty is right only where the output is
    // empty -- what a plan fills last. An output with no empty cell at all has
    // nothing to fill one, so there such a frame is no frame; elsewhere an empty
    // cell facing a color is one more cell the frame does not share, which the
    // reading through it may still explain (28bf18c6's copies).
    auto shared = [](const grid& one, const grid& other) {
        bool other_has_empty = false;
        for (int row = 0; row < other.height && !other_has_empty; ++row)
            for (int col = 0; col < other.width && !other_has_empty; ++col)
                other_has_empty = other.data[row][col] == TRANSPARENT;
        int same = 0;
        for (int row = 0; row < one.height; ++row)
            for (int col = 0; col < one.width; ++col) {
                if (one.data[row][col] == TRANSPARENT && !other_has_empty) return 0.0;
                same += one.data[row][col] == other.data[row][col];
            }
        return (double)same / (one.width * one.height);
    };
    piece_frame_rule best;
    double best_share = 0;
    auto weigh = [&](const piece_frame_rule& frame, auto framing) {
        double share = 1;
        for (size_t which = 0; which < train.size() && share > best_share; ++which) {
            grid framed;
            if (!framing(which, framed) || !framed.same_size(train[which].output)) {
                share = 0;
                break;
            }
            share = min(share, shared(framed, train[which].output));
        }
        if (share > best_share) {
            best_share = share;
            best = frame;
        }
    };
    // The whole input, where the size rule scales it: up or down by one whole
    // factor both ways, which ScaleUp and ScaleDown take; tiled by any whole count.
    vector<piece_frame_rule> wholes;
    if (inf.size.Chosen == output_dimension_rule::Scale) {
        const auto& ratio = inf.size;
        const bool square = ratio.width_numerator * ratio.height_denominator ==
                            ratio.height_numerator * ratio.width_denominator;
        const auto up = row_of(node_type::ScaleUp), down = row_of(node_type::ScaleDown);
        if (square && ratio.width_denominator == 1 && ratio.width_numerator >= up.low &&
            ratio.width_numerator <= up.high)
            wholes.push_back({ piece_frame_rule::scaled_up, (int8_t)ratio.width_numerator, true });
        if (square && ratio.width_numerator == 1 && ratio.width_denominator >= down.low &&
            ratio.width_denominator <= down.high)
            wholes.push_back({ piece_frame_rule::scaled_down, (int8_t)ratio.width_denominator, true });
        if (ratio.width_denominator == 1 && ratio.height_denominator == 1)
            wholes.push_back({ piece_frame_rule::tiled, 0, true });
        for (const piece_frame_rule& frame : wholes)
            weigh(frame, [&](size_t which, grid& framed) {
                return framed_whole(train[which].input, frame, inf.size, framed);
            });
    }
    // The whole input turned, where every output's sides are its input's traded;
    // a fractal of itself, where every output is its input times its own size;
    // or the unit it repeats. Asked whatever the size rule reads, since the rule
    // cannot say "the input's own size" and the canvas it gives is checked anyway.
    vector<piece_frame_rule> reads;
    if (ranges::all_of(train, [](const example& pair) {
            return pair.output.width == pair.input.height && pair.output.height == pair.input.width; }))
        for (const int pose : { 1, 3, 5, 7 }) reads.push_back({ piece_frame_rule::posed, (int8_t)pose, true });
    if (ranges::all_of(train, [](const example& pair) {
            return pair.output.width == pair.input.width * pair.input.width &&
                   pair.output.height == pair.input.height * pair.input.height; }))
        reads.push_back({ piece_frame_rule::fractal, 0, true });
    reads.push_back({ piece_frame_rule::repeat_unit, 0, true });
    for (const piece_frame_rule& frame : reads)
        weigh(frame, [&](size_t which, grid& framed) {
            return framed_whole(train[which].input, frame, inf.size, framed);
        });
    static constexpr segmentation_methods frame_cuts[] = { segmentation_methods::connected_multicolor,
                                                           segmentation_methods::connected_monochrome,
                                                           segmentation_methods::all_cells_of_one_color };
    // Every piece first, then each color on its own, then the picks by kind --
    // and the partner picks, where the reading asks them (inference::partners).
    auto picks_of = [&](const replay_facts& facts) {
        vector<piece_pick> picks = { piece_pick::all() };
        for (int color = 0; color < COLOR_COUNT; ++color) picks.push_back(piece_pick::of_color(color));
        for (const piece_pick& pick : picks_to_ask(facts, 0, 0, inf.partners)) picks.push_back(pick);
        return picks;
    };
    for (const segmentation_methods cut : frame_cuts) {
        const replay_facts facts(train, cut, tiles);
        const vector<piece_pick> picks = picks_of(facts);
        for (const piece_pick& pick : picks) {
            piece_frame_rule frame;
            frame.held = true;
            frame.cut = cut;
            frame.pick = pick;
            for (size_t which = 0; which < train.size(); ++which) {
                grid framed;
                bool several = false;
                if (!framed_input(facts.cuts()[which], pick, train[which].input, inf.size, framed,
                                  several))
                    break;
                frame.merged = frame.merged || several;
            }
            weigh(frame, [&](size_t which, grid& framed) {
                bool several = false;
                return framed_input(facts.cuts()[which], pick, train[which].input, inf.size, framed,
                                    several);
            });
        }
        // Laid out on graph paper: every piece, or the pieces of one color, in
        // cells of every size Layout takes, the size read off the pieces first.
        const auto cells = row_of(node_type::Layout);
        for (size_t which_pick = 0; which_pick <= (size_t)COLOR_COUNT; ++which_pick) {
            const piece_pick pick = picks[which_pick];
            for (int cell = cells.low; cell <= cells.high; ++cell) {
                piece_frame_rule frame;
                frame.kind = piece_frame_rule::laid_out;
                frame.factor = (int8_t)cell;
                frame.held = true;
                frame.cut = cut;
                frame.pick = pick;
                weigh(frame, [&](size_t which, grid& framed) {
                    return laid_out_input(facts.cuts()[which], pick, cell, train[which].input, inf.size, framed);
                });
            }
        }
    }
    // THE CANVAS IN ONE COLOR, the color a word names off the input, where every
    // output is one color all over: the pieces a pick takes, each in one color --
    // its own, or the one its shape or its symmetry names in a table read off the
    // pairs (ShapeColorMap, read_off) -- filled to its box and stretched over the
    // canvas, as a plan lays its fill (filled_input). Each piece's own color is
    // asked before a table; a table only while the puzzle has none, and only
    // where it is at work in every pair: some piece it takes comes out in a
    // color it did not wear (44f52bb0's picture, 1 where a turn or flip leaves
    // it as it was, else 7). The size rule gives the canvas, so not under
    // BoundingBox.
    const bool one_color = ranges::all_of(train, [](const example& pair) {
        const int first = pair.output.data[0][0];
        for (int row = 0; row < pair.output.height; ++row)
            for (int col = 0; col < pair.output.width; ++col)
                if (pair.output.data[row][col] != first) return false;
        return is_color(first);
    });
    shape_color_map_rule best_table;
    if (one_color && inf.size.Chosen != output_dimension_rule::BoundingBox) {
        const int last_kind = inf.shape_color_map.holds() ? -1 : (int)shape_key::count - 1;
        vector<uint16_t> shown;
        for (const example& pair : train) shown.push_back(shape_color_map_rule::colors_shown(pair.input));
        for (size_t which_cut = 0; which_cut <= size(frame_cuts); ++which_cut) {
            piece_frame_rule source;
            source.cut = which_cut < size(frame_cuts) ? frame_cuts[which_cut] : segmentation_methods::unset;
            replay_facts facts(train, source.cut, tiles);
            const bool whole = source.cut == segmentation_methods::unset;
            if (whole)
                for (size_t which = 0; which < train.size(); ++which)
                    facts.change(which).hand = source.pieces(train[which].input);
            const vector<piece_pick> picks = whole ? vector<piece_pick>{ piece_pick::all() } : picks_of(facts);
            for (const piece_pick& pick : picks)
                for (int keyed = -1; keyed <= last_kind; ++keyed) {
                    shape_color_map_rule table;
                    if (keyed >= 0) {
                        vector<shape_color_map_rule::seen_key> seen;
                        bool learned = true;
                        for (size_t which = 0; which < train.size() && learned; ++which) {
                            const int8_t wanted = (int8_t)train[which].output.data[0][0];
                            const vector<object>& pieces = facts.cuts()[which];
                            const vector<bool> picked = facts.picked(which, pick);
                            bool at_work = false;
                            for (size_t i = 0; i < pieces.size() && learned; ++i) {
                                if (!picked[i]) continue;
                                learned = shape_color_map_rule::see(
                                    seen, shape_color_map_rule::key_of((shape_key)keyed, pieces[i]), wanted, which,
                                    shown[which]);
                                pieces[i].each_cell([&](int, int, int8_t was) { at_work = at_work || was != wanted; });
                            }
                            learned = learned && at_work;
                        }
                        if (!learned) continue;
                        table = shape_color_map_rule::read_off(seen);
                        if (!table.holds()) continue;
                    }
                    piece_frame_rule frame;
                    frame.kind = piece_frame_rule::filled;
                    frame.held = true;
                    frame.cut = source.cut;
                    frame.pick = pick;
                    frame.keyed = (int8_t)keyed;
                    const double before = best_share;
                    weigh(frame, [&](size_t which, grid& framed) {
                        return filled_input(facts.cuts()[which], pick, (int8_t)keyed, table, train[which].input,
                                            inf.size, framed);
                    });
                    if (best_share > before) best_table = table;
                }
        }
    }
    if (best.kind == piece_frame_rule::filled && best.keyed >= 0) inf.shape_color_map = best_table;
    return best;
}

}  // namespace arc
