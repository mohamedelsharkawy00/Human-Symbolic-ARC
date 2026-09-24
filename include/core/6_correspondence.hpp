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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// Correspondences match objects from input to output and from output to input.

// This structure collects the correspondences and summaries of the readings tried.
struct correspondence_election {
    using object_info = object_segmentation::object_info;
    using edge = object_matching::edge;
    using rejection = object_matching::rejection;

    // An uncovered cell. made_or_gone means its object has no identified partner.
    struct left_over_cell {
        bool from_input = true;
        int8_t row = 0, col = 0;
        bool made_or_gone = false;
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
        // Left-over cells whose object has no partner: made on the output side,
        // gone from the input side.
        int made() const { return partnerless_cells(false); }
        int gone() const { return partnerless_cells(true); }
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
                const auto [width, height] = posed_size(link.pose, from->summary.w, from->summary.h);
                const bool on_this_piece = link.to == input_index ||
                    (link.to < 0 && link.row == piece.y0 && link.col == piece.x0 &&
                     width == piece.w && height == piece.h);
                if (on_this_piece) went(from->summary.y0, from->summary.x0);
            }
            return (int)places.size();
        }
    };
    // Total non-background cells in all training grids, used to normalize leftover counts.
    int cells = 0;
    // The selected segmentation, shared by all training pairs.
    segmentation_methods segmentation = segmentation_methods::unset;
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
    };
    vector<reading> readings;
    // Index of the selected reading in readings.
    int8_t chosen = 0;
    // Each segmentation's best reading, as its index in readings, so a rule
    // about one way of cutting can read that cut's own evidence, elected or not.
    vector<int16_t> best_of;
    const reading* best_reading_of(segmentation_methods method) const {
        const size_t index = (size_t)method;
        if (index >= best_of.size() || best_of[index] < 0) return nullptr;
        return &readings[best_of[index]];
    }
    // Indices of other segmentations whose best readings tie with the winner.
    vector<int8_t> tied_with_chosen;
    // Read the winner directly, without keeping a second copy of its evidence.
    // A default election has no readings yet.
    const vector<pair_info>& chosen_pairs() const {
        static const vector<pair_info> empty;
        return readings.empty() ? empty : readings[chosen].pairs;
    }

    // 1. Compare the readings and keep the one that best explains the examples.
    // Try every segmentation and permission combination, then return the selected reading.
    static correspondence_election infer(const vector<example>& train, int8_t background) {

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
        vector<int> best_of_method;
        facts.best_of.assign((size_t)segmentation_methods::count, -1);
        for (const auto& prepared : preparations) {
            int best = -1;
            for (int packed = 0; packed < degrees_of_freedom::COMBINATIONS; ++packed) {
                const int cell = (int)prepared.look.segmentation * degrees_of_freedom::COMBINATIONS + packed;
                reading& scored = facts.readings[cell];
                scored = detect_correspondence(train, background, prepared,
                                                degrees_of_freedom::from_index(packed));
                if (best < 0 || better_match(scored, facts.readings[best])) best = cell;
            }
            best_of_method.push_back(best);
            facts.best_of[(size_t)prepared.look.segmentation] = (int16_t)best;
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
        // Which of two readings the election prefers: negative for the first,
        // positive for the second, 0 for a tie. Using objects at all comes first,
        // then the lower price, and only prices within same_price of each other
        // are settled by using fewer objects.
        auto order = [&](const reading& one, const reading& other) {
            if ((one.objects_used > 0) != (other.objects_used > 0)) return one.objects_used > 0 ? -1 : 1;
            if (abs(price_of(one) - price_of(other)) > same_price)
                return price_of(one) < price_of(other) ? -1 : 1;
            if (one.objects_used != other.objects_used) return one.objects_used < other.objects_used ? -1 : 1;
            return 0;
        };
        size_t elected = 0;
        for (size_t i = 1; i < best_of_method.size(); ++i)
            if (order(facts.readings[best_of_method[i]], facts.readings[best_of_method[elected]]) < 0)
                elected = i;

        facts.chosen = (int8_t)best_of_method[elected];
        facts.segmentation = facts.readings[facts.chosen].segmentation;
        for (size_t i = 0; i < best_of_method.size(); ++i)
            if (i != elected && order(facts.readings[best_of_method[i]], facts.readings[facts.chosen]) == 0)
                facts.tied_with_chosen.push_back((int8_t)best_of_method[i]);
        return facts;
    }

private:
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

    // 2. Prepare each segmentation once, before trying its permissions.
    static prepared_segmentation prepare_segmentation(const vector<example>& train,
                                                      int8_t background, segmentation_methods segmentation) {
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
            const auto [step_width, step_height] = object_segmentation::block_of(input, output, background,
                                                            segmentation, smaller_width, smaller_height);
            vector<object>& input_objects = saved.input_objects;
            vector<object>& output_objects = saved.output_objects;
            object_segmentation::segment_for_correspondence(input, background, segmentation, pair.input_info,
                                       &input_objects, step_width, step_height);
            object_segmentation::segment_for_correspondence(output, background, segmentation, pair.output_info,
                                       &output_objects, step_width, step_height);
            // Group copies and choose the most typical member while their cells are available.
            const bool one_set_for_the_grid = (segmentation == segmentation_methods::minimal_repeating_set);
            object_segmentation::find_copies(pair.input_info, input_objects, one_set_for_the_grid);
            object_segmentation::find_copies(pair.output_info, output_objects, one_set_for_the_grid);
            // What the segmentation cost to say, on each grid. The pair is described by both.
            pair.describe_cost =
                0.5 * (object_segmentation::describe_cost(input, background, pair.input_info) +
                       object_segmentation::describe_cost(output, background, pair.output_info));
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
            saved.step_width = step_width;
            saved.step_height = step_height;
        }

        if (counted) prepared.look.describe_cost = shares / counted;
        return prepared;
    }

    // 3. Follow one reading through the examples.
    // One reading: match each example, reconcile their evidence, then describe what survived.
    static reading detect_correspondence(const vector<example>& train, int8_t background,
                                                       const prepared_segmentation& prepared,
                                                       degrees_of_freedom allowed) {
        reading facts = prepared.look;
        facts.allowed = allowed;
        if (train.empty()) return facts;
        for (const auto& pair : prepared.pairs) facts.pairs.push_back(pair.info);
        for (size_t i = 0; i < train.size(); ++i)
            if (prepared.pairs[i].read)
                match_one_pair(train, background, prepared, allowed, facts, i, nullptr);
        reconcile_examples(train, background, prepared, allowed, facts);
        for (size_t i = 0; i < train.size(); ++i)
            if (prepared.pairs[i].read) summarize_pair(train, background, prepared, facts, i);
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
                    facts.box_cells += one_object.summary.w * one_object.summary.h;
                }
        }
        if (!facts.pairs.empty()) facts.describe_cost = shares / facts.pairs.size();
    }

    // Match both directions, merge duplicate links, and record shared movements.
    static void match_one_pair(const vector<example>& train, int8_t background,
                               const prepared_segmentation& prepared, degrees_of_freedom allowed,
                               reading& facts, size_t pair_index,
                               const what_the_pairs_agreed* agreed) {
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
        pair.agreed_moves.clear(); pair.verdict = 0;
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
                    into_info, into, background, step_width, step_height, edges, allowed);
            } else if (pattern_reading) {
                verdict = object_matching::collect_motif_edges(from_info, from_objects, into_info,
                                              into, background, step_width, step_height,
                                              edges, allowed);
            } else {
                verdict = object_matching::collect_edges(from_info, from_objects, into_info,
                                        into_objects, into, from_input,
                                        background, allowed.hang_off,
                                        edges, &discarded, allowed, agreed);
                object_matching::sit_on_top(from_info, from_objects, into_info, into_objects, edges);
            }
            for (auto& edge : edges) edge.from_input = from_input;
            for (auto& rejected : discarded) rejected.match.from_input = from_input;
            // The weaker of the two readings names the pass: a fit both ways is
            // a fit, and a giveup anywhere is still a giveup.
            pair.verdict = max(pair.verdict, (int8_t)verdict);
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
            if (here.y0 != there.y0) return here.y0 < there.y0;
            if (here.x0 != there.x0) return here.x0 < there.x0;
            if (left.row != right.row) return left.row < right.row;
            return left.col < right.col;
        });

        // Group equal input-to-output movements within this pair.
        // Count each input object once per movement, using its input color.
        // A backward positional correspondence needs an identifiable input object.
        auto input_object_of = [&](const edge& edge) {
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
        vector<pair_info::moved_together> moves;
        vector<vector<int>> who_moved;
        for (const auto& edge : pair.edges) {
            const object_info* one_object = pair.source_of(edge);
            if (!edge.same_shape || !one_object) continue;
            const auto went = object_matching::move_of(edge, *one_object);
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
    }

    // Let movements supported by all examples constrain ambiguous matches.
    static void reconcile_examples(const vector<example>& train, int8_t background,
                                   const prepared_segmentation& prepared, degrees_of_freedom allowed,
                                   reading& facts) {
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
                if (any_disagrees) match_one_pair(train, background, prepared, allowed, facts, which, &agreed[which]);
            }

    }

    // Describe the final links, unmatched objects, and unexplained cells of one pair.
    static void summarize_pair(const vector<example>& train, int8_t background,
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
                        if (part.summary.w > whole.summary.w || part.summary.h > whole.summary.h) continue;
                        bool equal = true;
                        for (int r = 0; r < part.summary.h; ++r)
                            for (int c = 0; c < part.summary.w; ++c)
                                if (part.cells[r][c] != whole.cells[r][c]) equal = false;
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

        // Record the background region touched by each unmatched object on the other grid.
        auto note_blanks = [&](vector<object_info>& info,
                               const vector<object>& objects, const grid& other) {
            if (ranges::all_of(info, [](const auto& one) {
                    return one.has_partner; })) return;
            vector<blank_patch> patches;
            int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM];
            find_blanks(other, background, patches, patch_of);
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
                our_objects[edge.from].each_cell([&](int row, int col, int8_t) { carried[row][col] = true; });
            }
            // Which object each cell belongs to, so a cell can be asked about
            // the object it is part of.
            int16_t object_at[MAX_GRID_DIM][MAX_GRID_DIM];
            for (auto& row : object_at) ranges::fill(row, (int16_t)-1);
            for (size_t i = 0; i < ours.size() && i < our_objects.size(); ++i)
                our_objects[i].each_cell([&](int row, int col, int8_t) { object_at[row][col] = (int16_t)i; });
            walk_colored_cells(one_grid, background, [&](int row, int col, int) {
                if (carried[row][col]) return;
                if (object_at[row][col] >= 0 && ours[object_at[row][col]].instance_of >= 0) return;
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
    static void find_blanks(const grid& one_grid, int8_t background,
                            vector<blank_patch>& patches,
                            int16_t patch_of[MAX_GRID_DIM][MAX_GRID_DIM]) {
        for (int row = 0; row < MAX_GRID_DIM; ++row)
            for (int col = 0; col < MAX_GRID_DIM; ++col) patch_of[row][col] = -1;
        int16_t queue_row[MAX_GRID_CELLS], queue_col[MAX_GRID_CELLS];
        for (int row = 0; row < one_grid.height; ++row)
            for (int col = 0; col < one_grid.width; ++col) {
                if (one_grid.data[row][col] != background) continue;
                if (patch_of[row][col] >= 0) continue;
                const int16_t mine = (int16_t)patches.size();
                cell_box box;
                bool open = false;
                queue_row[0] = (int16_t)row; queue_col[0] = (int16_t)col;
                patch_of[row][col] = mine;
                const int cells = walk_out<4>(one_grid.height, one_grid.width, queue_row, queue_col, 1,
                    [&](int next_row, int next_col) {
                        if (one_grid.data[next_row][next_col] != background ||
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

};

// Entry point used by infer(); the result owns the correspondence machinery.
inline correspondence_election select_segmentation(const vector<example>& train, int8_t background) {
    return correspondence_election::infer(train, background);
}

} // namespace arc
