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
struct correspondence_facts {
    using object_info = object_segmentation::object_info;
    using edge = object_matching::edge;

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
        // Matches that were found and then thrown away, with why â€” otherwise
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
        // Non-background into cells no exact edge covers â€” made, not carried
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
        // How many objects the segmentation FOUND, and how many it USED â€” the log prints
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

    // 1. Compare the readings and keep the one that best explains the examples.
    // Try every segmentation and permission combination, then return the selected reading.
    static correspondence_facts infer(const vector<example>& train, int8_t background) {

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

private:
    // Objects and grid-only scores shared by every permission combination.
    struct prepared_segmentation {
        correspondence_facts::reading look;
        vector<correspondence_facts::pair_info> pairs;
        vector<vector<object>> input_objects, output_objects;
        vector<int> step_width, step_height;
        vector<bool> read;
    };

    using what_the_pairs_agreed = object_matching::what_the_pairs_agreed;
    using fit_verdict = object_matching::fit_verdict;

    // 2. Prepare each segmentation once, before trying its permissions.
    static prepared_segmentation prepare_segmentation(const vector<example>& train,
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
        const auto [sizes_step_width, sizes_step_height] = object_segmentation::repeating_step_of(train);
        double shares = 0.0;
        int counted = 0;
        // Segment before matching so another matching pass can reuse the objects.
        for (size_t pair_index = 0; pair_index < train.size(); ++pair_index) {
            if (!prepared.read[pair_index]) continue;
            const grid& input = train[pair_index].input;
            const grid& output = train[pair_index].output;
            auto& pair = prepared.pairs[pair_index];
            // Use the same block size for both grids.
            const auto [step_width, step_height] = object_segmentation::block_of(input, output, background,
                                                            segmentation, sizes_step_width,
                                                            sizes_step_height);
            vector<object>& input_objects = prepared.input_objects[pair_index];
            vector<object>& output_objects = prepared.output_objects[pair_index];
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
            prepared.step_width[pair_index] = step_width;
            prepared.step_height[pair_index] = step_height;
        }

        if (counted) prepared.look.describe_cost = shares / counted;
        return prepared;
    }

    // 3. Follow one reading through the examples.
    // One reading: match each example, reconcile their evidence, then describe what survived.
    static correspondence_facts detect_correspondence(const vector<example>& train, int8_t background,
                                                       const prepared_segmentation& prepared,
                                                       degrees_of_freedom allowed) {
        correspondence_facts facts;
        if (train.empty()) return facts;
        facts.pairs = prepared.pairs;
        facts.segmentation = prepared.look.segmentation;
        for (size_t i = 0; i < train.size(); ++i)
            if (prepared.read[i])
                match_one_pair(train, background, prepared, allowed, facts, i, nullptr);
        reconcile_examples(train, background, prepared, allowed, facts);
        for (size_t i = 0; i < train.size(); ++i)
            if (prepared.read[i]) summarize_pair(train, background, prepared, facts, i);
        facts.valid = true;
        return facts;
    }

    // Match both directions, merge duplicate links, and record shared movements.
    static void match_one_pair(const vector<example>& train, int8_t background,
                               const prepared_segmentation& prepared, degrees_of_freedom allowed,
                               correspondence_facts& facts, size_t pair_index,
                               const what_the_pairs_agreed* agreed) {
        const bool pattern_reading =
            facts.segmentation == segmentation_methods::minimal_repeating_set ||
            facts.segmentation == segmentation_methods::smallest_of_two_grids;
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
                verdict = object_matching::collect_motif_edges(from_info, from_objects, into_info,
                                              into, background, step_width, step_height,
                                              edges, allowed);
            } else {
                verdict = object_matching::collect_edges(from_info, from_objects, into_info,
                                        into_objects, into, from_input,
                                        background, allowed.hang_off,
                                        edges, &discarded, &discard_why, allowed, agreed);
                object_matching::sit_on_top(from_info, from_objects, into_info, into_objects, edges);
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
        for (const auto& link : pair.edges) {
            auto same = ranges::find_if(kept, [&](const correspondence_facts::edge& seen) {
                return joins_the_same(link, seen); });
            if (same == kept.end()) { kept.push_back(link); continue; }
            const int mine = claim_strength(link), theirs = claim_strength(*same);
            if (mine > theirs || (mine == theirs && link.from_input && !same->from_input))
                *same = link;
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
            const auto went = object_matching::move_of(edge, one_object);
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
                                   correspondence_facts& facts) {
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
                if (!prepared.read[which]) continue;
                ++read;
                said_once += (lone_move_of(which, color) != nullptr);
            }
            if (read == 0 || said_once != read) continue;
            anything_agreed = true;
            for (size_t which = 0; which < facts.pairs.size(); ++which) {
                if (!prepared.read[which]) continue;
                const auto* rule = lone_move_of(which, color);
                agreed[which].color_speaks[color] = true;
                agreed[which].down[color] = rule->down;
                agreed[which].across[color] = rule->across;
            }
        }
        if (anything_agreed)
            for (size_t which = 0; which < facts.pairs.size(); ++which) {
                if (!prepared.read[which]) continue;
                // Rematch only if an existing match disagrees.
                const auto& pair = facts.pairs[which];
                bool any_disagrees = false;
                for (const auto& edge : pair.edges) {
                    const auto& objects = edge.from_input ? pair.input_info
                                                          : pair.output_info;
                    if (edge.from < 0 || (size_t)edge.from >= objects.size()) continue;
                    if (!object_matching::the_pairs_disagree(agreed[which], edge, objects[edge.from])) continue;
                    any_disagrees = true;
                    break;
                }
                if (any_disagrees) match_one_pair(train, background, prepared, allowed, facts, which, &agreed[which]);
            }

    }

    // Describe the final links, unmatched objects, and unexplained cells of one pair.
    static void summarize_pair(const vector<example>& train, int8_t background,
                               const prepared_segmentation& prepared,
                               correspondence_facts& facts, size_t pair_index) {
        const grid& input = train[pair_index].input;
        const grid& output = train[pair_index].output;
        auto& pair = facts.pairs[pair_index];
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
            // DOES ANYTHING OVER THERE CLAIM TO BE THIS OBJECT â€” it went
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

    // 4. Count what the final matches explain and describe the empty regions.
    // Count the matches starting from each source object.
    static void count_edges_leaving(const vector<correspondence_facts::edge>& edges,
                                    vector<correspondence_facts::object_info>& from_info) {
        for (auto& one_object : from_info) one_object.edges_touching = 0;
        for (const auto& edge : edges)
            if (edge.from >= 0 && (size_t)edge.from < from_info.size()) {
                from_info[edge.from].edges_touching++;
                from_info[edge.from].has_partner = true;
            }
    }

    // Count the matches that overlap each target object.
    static void count_edges_arriving(const vector<correspondence_facts::edge>& edges,
                                const vector<object>& from_objects,
                                vector<correspondence_facts::object_info>& into_info,
                                const vector<object>& into_objects) {
        for (const auto& edge : edges) {
            const grid footprint = object_matching::edge_footprint(edge, from_objects, into_objects);
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

    static void mark_painted(const vector<correspondence_facts::edge>& edges,
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

    // An empty region that an object may have filled or left behind.
    struct blank_patch {
        int8_t left = 0, top = 0, width = 0, height = 0;
        int16_t cells = 0;
        bool enclosed = false;
    };

    // Finds empty regions to help describe where objects appeared or disappeared.
    static void find_blanks(const grid& one_grid, int8_t background,
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

};

// Entry point used by infer(); the result owns the correspondence machinery.
inline correspondence_facts select_segmentation(const vector<example>& train, int8_t background) {
    return correspondence_facts::infer(train, background);
}

} // namespace arc
