// This file expresses the reading in the solver's own words.
// Design story in notes/1 - Overall Design/Inference Master.md
//
// THE LOOKING AND THE SEARCHING SHARE ONE VOCABULARY, and this is where that
// is made true. A reading already knows what an object is, what the pieces were
// allowed to do, and what became of the cells. Said in its own words that is a
// table nobody can run; said in the solver's words it is a program.
//
// TWO LISTS COME BACK, and the second is the valuable one. What can be said
// becomes sentences. What cannot names the word the language is missing, at the
// moment the looking wanted it — instead of after a search has spent minutes
// failing to find it.
//
// IT SPELLS, IT NEVER TRIES. Every word here is read off a fact the reading
// already holds; nothing is searched, and nothing is scored. The moment this
// starts trying sentences to see which fits, it has stopped being inference and
// become a second solver.

#pragma once
#include "7_infer.hpp"
#include <optional>
#include <string>
#include <vector>

namespace arc {
using namespace std;

// A beat nothing has a word for yet. Said out loud rather than left out, so the
// search knows exactly which one slot it has to fill.
inline constexpr const char* a_blank = "?";

struct said_reading {
    vector<string> sentences;
    vector<string> no_word_for;
    // OPENINGS THE READING COULD NOT RULE OUT. Only the first sentence can be
    // said more than one way -- everything after it hangs off the chosen
    // reading's own pieces, so a rival segmentation has no follow-on to go with it.
    vector<string> other_openings;
};

// WHAT AN OBJECT IS, as the operation that segments the same way. Split's argument is
// the any-color flag, so Split[0] keeps colors together and Split[1] does not.
inline const char* word_for_segmentation(segmentation_methods segmentation) {
    switch (segmentation) {
        case segmentation_methods::connected_monochrome:   return "Split[1]";
        case segmentation_methods::connected_multicolor:   return "Split[0]";
        case segmentation_methods::all_cells_of_one_color: return "Colors";
        case segmentation_methods::minimal_repeating_set:  return "Pattern";
        case segmentation_methods::smallest_of_two_grids:  return nullptr;
        default:                                          return nullptr;
    }
}

// ONE OF THE EIGHT WAYS ROUND (1_readings.hpp), said in turns and mirrors.
// Every one of the eight is sayable, four of them in one word.
inline vector<string> words_for_pose(int pose) {
    switch (pose & 7) {
        case 1:  return { "TurnRight", "FlipHorizontal" };
        case 2:  return { "FlipVertical" };
        case 3:  return { "TurnRight" };
        case 4:  return { "FlipHorizontal" };
        case 5:  return { "TurnLeft" };
        case 6:  return { "TurnRight", "TurnRight" };
        case 7:  return { "TurnRight", "FlipVertical" };
        default: return {};
    }
}

// A CELL-LEVEL COLOR MAP AS WORDS: nothing when no color changes, one
// BidiColorMap when two colors trade places, nullopt when it is anything else.
inline optional<string> word_for_color_map(const int8_t* map) {
    int first = -1, second = -1;
    for (int c = 0; c < COLOR_COUNT; ++c)
        if (map[c] >= 0 && map[c] != c) { first = c; second = map[c]; break; }
    if (first < 0) return string();
    for (int c = 0; c < COLOR_COUNT; ++c) {
        const int mapped = c == first ? second : c == second ? first : c;
        if (map[c] >= 0 && map[c] != mapped) return nullopt;
    }
    const int a = min(first, second), b = max(first, second);
    const int variant = a * (2 * COLOR_COUNT - a - 1) / 2 + b - a - 1;
    return "BidiColorMap[" + to_string(variant) + "]";
}

// ONE SENTENCE. Nothing is implicit: the first names its source, the rest carry
// on from the objects already in hand.
inline void say_sentence(said_reading& said, const string& source, const string& pick,
                         const string& act, const string& combine) {
    said.sentences.push_back((said.sentences.empty() ? source : string("CurrentObjects")) +
                             " " + pick + " " + act + " " + combine);
}

// ONE BEAT OF A LINE: the pieces picked, and the one word done to them.
struct beat { string pick, act; };

// A LINE OF BEATS, SAID AS SENTENCES. Only the last paints: a cell drawn before
// its piece has turned or slid is never taken back, so every beat before it
// withholds.
inline void say_beats(said_reading& said, const string& source,
                      const vector<beat>& beats, const string& combine) {
    for (size_t i = 0; i < beats.size(); ++i)
        say_sentence(said, source, beats[i].pick, beats[i].act,
                     i + 1 < beats.size() ? "Withhold" : combine);
}

// WHICH WORD A FIRED DETECTOR IS. The table is shared storage, so it says which
// detector filled it; feeding one op the other's table would be silent nonsense.
inline const char* word_for_learned_recolor(const shape_color_map_rule& rule) {
    if (!rule.valid) return nullptr;
    switch (rule.claimed_by) {
        case shape_color_map_rule::body_recolor:  return "ShapeColorMap";
        case shape_color_map_rule::interior_fill: return "ShapeToInteriorColorMap";
        default:                                  return nullptr;
    }
}

// THE WHOLE READING AS PROGRAMS, most likely first. The tied openings are whole
// programs of their own rather than a claim that one of them is right.
inline vector<string> programs_from(const said_reading& said) {
    vector<string> programs;
    if (said.sentences.empty()) return programs;
    string whole;
    for (const string& sentence : said.sentences)
        whole += (whole.empty() ? "" : " ") + sentence;
    programs.push_back(whole);
    for (const string& opening : said.other_openings) {
        string rest;
        for (size_t i = 1; i < said.sentences.size(); ++i) rest += " " + said.sentences[i];
        programs.push_back(opening + rest);
    }
    return programs;
}

// A PROGRAM READ BACK AS ITS SENTENCES, one to a row. What the pane draws and
// what the log writes, so neither has to know how a program is spelled. A pick
// can take several words, so a sentence ends at the word that closes it (the
// combine role in 10_dsl.cuh), not after a count of them.
inline vector<string> sentences_of(const string& program) {
    static constexpr const char* const closers[] = {
        "Union", "Underlay", "Withhold", "Intersection", "Difference" };
    vector<string> rows;
    string row;
    for (size_t at = 0, next; at < program.size(); at = next) {
        next = program.find(' ', at);
        if (next == string::npos) next = program.size();
        if (next > at) {
            const string word = program.substr(at, next - at);
            row += (row.empty() ? "" : " ") + word;
            if (ranges::find(closers, word) != end(closers)) rows.push_back(exchange(row, string()));
        }
        if (next < program.size()) ++next;
    }
    return rows;
}

// THE OUTPUT AS WHOLE INPUTS, said as which blocks hold one and how each sits.
// All of them filled is Tile; filled where the input has its figure is Fractal.
// How each sits is factored into a pose for the input, then one per row of
// blocks, then one per column, so a flipped middle row, a mirror down the middle
// and a kaleidoscope are the same sentences with different numbers. Tile hands
// back a piece per copy, which is what lets a row of them be picked; Fractal
// hands back one sheet, so its copies can only all sit one way.
// False when there is nothing to say, so the rest of the reading still speaks;
// a layout it has no words for is named on the way out.
inline bool say_input_blocks(const inference& inf, said_reading& said) {
    const auto& blocks = inf.input_blocks;
    if (!blocks.valid || inf.size.Chosen == output_dimension_rule::NotInferredYet ||
        inf.size.Chosen == output_dimension_rule::BoundingBox) return false;
    const auto color_word = word_for_color_map(blocks.color_map);
    if (!color_word) {
        said.no_word_for.push_back("the copies' cell-color mapping");
        return false;
    }
    auto cheapest = [](auto holds) {
        int found = -1;
        for (int pose = 0; pose < POSE_COUNT; ++pose)
            if (holds(pose) && (found < 0 || words_for_pose(pose).size() < words_for_pose(found).size()))
                found = pose;
        return found;
    };

    vector<beat> opening;
    if (!color_word->empty()) opening.push_back({"SelectAll", *color_word});
    auto say_pose = [](vector<beat>& beats, const string& pick, int pose) {
        for (const string& word : words_for_pose(pose)) beats.push_back({pick, word});
    };
    // What a spelling costs is how many ops it runs: source, pick, act, combine.
    auto ops_in = [](const vector<beat>& beats) {
        size_t ops = 0;
        for (const auto& one : beats) ops += 3 + ranges::count(one.pick, ' ') + 1;
        return ops;
    };

    const bool some_are_ground = ranges::any_of(blocks.pairs, [](const auto& at) {
        for (int r = 0; r < at.rows; ++r)
            for (int c = 0; c < at.cols; ++c)
                if (!at.poses[r][c]) return true;
        return false;
    });
    string source = "Input";
    vector<beat> beats;
    if (some_are_ground) {
        // The ground is punched out before stamping, so no map may repaint it.
        const int ground = inf.background.input_elected();
        const int pose = cheapest([&](int p) { return (blocks.figure_poses >> p & 1) != 0; });
        if (pose < 0 || !is_color(ground) ||
            (blocks.color_map[ground] >= 0 && blocks.color_map[ground] != ground)) {
            said.no_word_for.push_back("which blocks hold a copy of the input");
            return false;
        }
        source = "InputWithoutBackground";
        beats = opening;
        say_pose(beats, "SelectAll", pose);
        beats.push_back({"SelectAll", "Fractal"});
    } else {
        int rows = 0, cols = 0;
        bool several = false;
        for (const auto& at : blocks.pairs) {
            rows = max(rows, (int)at.rows);
            cols = max(cols, (int)at.cols);
            several = several || at.rows * at.cols > 1;
        }
        // Does the block at (r, c) sit as `input`, then `row`, then `col`, in every pair?
        auto sits = [&](int r, int c, int input, int row, int col) {
            const int pose = compose_poses(compose_poses(input, row), col);
            return ranges::all_of(blocks.pairs, [&](const auto& at) {
                return r >= at.rows || c >= at.cols || (at.poses[r][c] >> pose & 1) != 0; });
        };
        // Tile's copies keep the input's mark, so each band starts from none picked.
        auto say_bands = [&](vector<beat>& into, const vector<int>& poses, const string& select) {
            for (int pose = 1; pose < POSE_COUNT; ++pose) {
                string bands;
                for (size_t band = 0; band < poses.size(); ++band)
                    if (poses[band] == pose) bands += " " + select + "[" + to_string(band) + "]";
                if (!bands.empty()) say_pose(into, "SelectAll InvertSelection" + bands, pose);
            }
        };
        bool found = false;
        for (int input = 0; input < POSE_COUNT; ++input)
            for (int first_col = 0; first_col < POSE_COUNT; ++first_col) {
                vector<int> row(rows), col(cols);
                bool fits = true;
                for (int r = 0; r < rows && fits; ++r) {
                    row[r] = cheapest([&](int p) { return sits(r, 0, input, p, first_col); });
                    fits = row[r] >= 0;
                }
                col[0] = first_col;
                for (int c = 1; c < cols && fits; ++c) {
                    col[c] = cheapest([&](int p) {
                        for (int r = 0; r < rows; ++r)
                            if (!sits(r, c, input, row[r], p)) return false;
                        return true;
                    });
                    fits = col[c] >= 0;
                }
                if (!fits) continue;
                vector<beat> spelled = opening;
                say_pose(spelled, "SelectAll", input);
                if (several) spelled.push_back({"SelectAll", "Tile[0]"});
                say_bands(spelled, row, "SelectRow");
                say_bands(spelled, col, "SelectColumn");
                if (!found || ops_in(spelled) < ops_in(beats)) beats = move(spelled);
                found = true;
            }
        if (!found) {
            said.no_word_for.push_back("how each copy of the input sits");
            return false;
        }
    }
    // An output that is the input as given says nothing about blocks.
    if (beats.empty()) return false;
    say_beats(said, source, beats, "Union");
    return true;
}

inline said_reading say_the_reading(const inference& inf) {
    const correspondence_facts& facts = inf.correspondence;
    said_reading said;
    if (say_input_blocks(inf, said)) return said;
    if (!facts.valid || facts.pairs.empty()) return said;

    // A whole-input block is not a repeating unit extracted by Pattern. Until
    // its placement is explained, copying it has no complete sentence here.
    if (facts.segmentation == segmentation_methods::smallest_of_two_grids) {
        said.no_word_for.push_back("the placement of whole-input copies");
        return said;
    }

    // A piece that sits on what is already there is laid under it, not over.
    const bool some_sit_on = ranges::any_of(facts.pairs, [](const auto& pair) {
        return pair.sits_on > 0; });
    const char* combine = some_sit_on ? "Underlay" : "Union";

    // THE READING AS ONE LINE OF BEATS, said once it is complete, because only
    // then is it known which beat is the last and paints.
    vector<beat> beats;
    auto say = [&](const string& act) { beats.push_back({"SelectAll", act}); };
    // The segmentation methods that explained just as much. Said as openings of their own so a
    // search can try each, rather than trusting the order they were declared in.
    vector<const char*> tied;
    auto spoken = [&]() -> said_reading& {
        say_beats(said, "Input", beats, combine);
        // A tied opening closes the way the chosen one did, so the rest of the
        // line still follows it.
        const string close = beats.size() > 1 ? "Withhold" : combine;
        for (const char* segmentation : tied) {
            string opening = string("Input SelectAll ") + segmentation + " " + close;
            bool already = !said.sentences.empty() && said.sentences.front() == opening;
            for (const string& seen : said.other_openings) already = already || seen == opening;
            if (!already) said.other_openings.push_back(opening);
        }
        return said;
    };

    const degrees_of_freedom allowed = facts.chosen < (int8_t)facts.readings.size()
                             ? facts.readings[facts.chosen].allowed : degrees_of_freedom{};

    if (const char* segmentation = word_for_segmentation(facts.segmentation)) {
        say(segmentation);
        // Correspondence measured every piece in the box its cells make, and Split
        // and Colors hand pieces back in the frame they were given (only Trim
        // shrinks one), so the pieces are trimmed to those boxes first.
        if (boxed_by_its_cells(facts.segmentation)) say("Trim");
    }
    else said.no_word_for.push_back(string("the segmentation ") + segmentation_names[(int)facts.segmentation]);

    // A complete continuation has its own cell-level map. Spell that map once,
    // then repeat the motif; the generic object recolor below has another scope.
    if (inf.motif_continuation.valid &&
        inf.size.Chosen != output_dimension_rule::NotInferredYet &&
        inf.size.Chosen != output_dimension_rule::BoundingBox) {
        const auto map_word = word_for_color_map(inf.motif_continuation.color_map);
        if (!map_word) {
            said.no_word_for.push_back("the motif's cell-color mapping");
            return spoken();
        }
        if (!map_word->empty()) say(*map_word);
        say("Tile[0]");
        return spoken();
    }

    for (int8_t cell : facts.tied_with_chosen) {
        if (cell < 0 || cell >= (int8_t)facts.readings.size()) continue;
        if (const char* segmentation = word_for_segmentation(facts.readings[cell].segmentation))
            tied.push_back(segmentation);
    }

    // PIECES THAT PILED UP INSIDE THEIR OWN BOXES. The slide drew every output
    // by itself, so it is the whole of what happened.
    if (inf.piece_slam.valid) {
        say(string(inf.piece_slam.rows ? "SlamRow[" : "SlamColumn[") +
            to_string((int)inf.piece_slam.step) + "]");
        return spoken();
    }

    // A LINE IS A SELECTION AND WHAT HAPPENED TO IT. One piece can turn AND move
    // AND change colour, so a line carries several acts, and the grammar gives
    // each act its own sentence sharing the line's GET and SELECT. Only the last
    // of them paints; the rest withhold, which is what that word is for.
    //
    // EACH ACT STANDS ON ITS OWN EVIDENCE. Asked as one key, a turn that differs
    // between pairs took the colour down with it -- 017c7c7b recolours 1 to 2 in
    // every pair and says so on every edge, and the reading lost it because the
    // poses read 0, 5 and 5. Turn, move and colour are three questions, so a
    // disagreement about one says nothing about the others.
    //
    // AN ACT MUST HOLD IN EVERY PAIR, or it is one example's accident.
    struct seen_act { int8_t color; int8_t what; int8_t value; };
    auto same_act = [](const seen_act& one, const seen_act& other) {
        return one.color == other.color && one.what == other.what && one.value == other.value;
    };
    enum : int8_t { act_turn, act_down, act_across, act_recolor };
    vector<seen_act> acts_in_every_pair;
    for (size_t which = 0; which < facts.pairs.size(); ++which) {
        const auto& pair = facts.pairs[which];
        vector<seen_act> here;
        auto note = [&](int8_t color, int8_t what, int8_t value) {
            if (value == 0 && what != act_recolor) return;   // nothing happened
            for (const auto& already : here)
                if (same_act(already, {color, what, value})) return;
            here.push_back({color, what, value});
        };
        for (const auto& edge : pair.edges) {
            if (!edge.from_input || !edge.same_shape) continue;
            if (edge.from < 0 || edge.from >= (int16_t)pair.input_info.size()) continue;
            const auto& piece = pair.input_info[edge.from];
            note(piece.color, act_turn,   edge.pose);
            // A copy's landing is not a movement of the single original.
            if (piece.edges_touching == 1) {
                note(piece.color, act_down,   (int8_t)(edge.row - piece.summary.y0));
                note(piece.color, act_across, (int8_t)(edge.col - piece.summary.x0));
            }
            if (!edge.exact)
                for (int8_t was = 0; was < COLOR_COUNT; ++was)
                    if (edge.color_map[was] >= 0 && edge.color_map[was] != was)
                        note(was, act_recolor, edge.color_map[was]);
        }
        if (which == 0) { acts_in_every_pair = here; continue; }
        vector<seen_act> still;
        for (const auto& act : acts_in_every_pair)
            for (const auto& mine : here)
                if (same_act(act, mine)) { still.push_back(act); break; }
        acts_in_every_pair = still;
    }

    bool a_recolor_was_said = false;
    for (const auto& act : acts_in_every_pair) {
        if (act.what == act_recolor) a_recolor_was_said = true;
        // Picks add to what is already marked, and the Trim picked every piece, so
        // a line about one color starts from none picked.
        const string pick = act.color < 0 ? string("SelectAll")
            : "SelectAll InvertSelection SelectColor[" + to_string((int)act.color) + "]";
        vector<string> words;
        switch (act.what) {
            case act_turn:    words = words_for_pose(act.value); break;
            case act_down:    words = {"MoveColumn[" + to_string((int)act.value) + "]"}; break;
            case act_across:  words = {"MoveRow[" + to_string((int)act.value) + "]"}; break;
            case act_recolor: words = {"Recolor[" + to_string((int)act.value) + "]"}; break;
            default: break;
        }
        for (const string& word : words) beats.push_back({pick, word});
    }

    // Colours that trade places. Said before the reading's own recolor, because
    // a swap is the whole story for the cells it touches and needs no pieces.
    for (int variant = 0; variant < COLOR_PAIR_COUNT; ++variant)
        if (inf.color_swaps.variant[variant])
            say("BidiColorMap[" + to_string(variant) + "]");

    if (allowed.recolor)
        for (const auto& pair : facts.pairs)
            if (pair.found_recolored > 0) {
                if (const char* word = word_for_learned_recolor(inf.shape_color_map))
                    say(word);
                else if (!a_recolor_was_said)
                    said.no_word_for.push_back("a recolor");
                break;
            }

    // A piece that sits on its landing changed shape, and nothing here says how.
    if (some_sit_on) said.no_word_for.push_back("how a piece changed its shape");

    return spoken();
}

}  // namespace arc
