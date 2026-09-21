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
    if (!rule.holds()) return nullptr;
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
// combine role in 11_dsl.cuh), not after a count of them.
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

// EVERY SENTENCE MUST OPEN WITH A SOURCE. A sentence whose first word is not a
// GET does not start from the input: it carves whatever the sentence before it
// left in hand. That reads as a perfectly ordinary program, runs without
// complaint, draws something plausible, and is wrong -- which is exactly how it
// got past a footer, a log and two readers. So it is checked here, once, and a
// program that breaks it is named rather than handed on.
//
// CurrentObjects is a GET and says "the ones I have" on purpose; carrying the
// hand forward is only a bug when nobody said to.
inline const char* sentence_with_no_source(const string& program) {
    // The GET-role ops, named here the way sentences_of names the closers: this
    // file is read before the op table exists, so the words are spelled out. If
    // a GET is ever added to NODE_PRIMITIVES it belongs in this list too.
    static constexpr const char* const sources[] = {
        "Input", "InputWithoutBackground", "Output", "CurrentObjects", "Seeds" };
    for (const string& sentence : sentences_of(program)) {
        const size_t end = sentence.find(' ');
        const string first = sentence.substr(0, end == string::npos ? sentence.size() : end);
        bool opens = false;
        for (const char* source : sources) opens |= first == source;
        if (!opens) return "a sentence that opens with no source";
    }
    return nullptr;
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
    if (!blocks.holds() || !inf.size.holds() ||
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

// THE PANES, SAID. The split cut the input into panes and the mask says what
// they come to; both are facts, so the program is read off them and nothing is
// tried. Every sentence here opens with a GET of its own, which is why it does
// not go through say_sentence: that one carries the hand forward with
// CurrentObjects, and a sentence that carried the hand forward would carve what
// the last sentence left instead of the input -- right-looking and silently
// wrong. Origin is what stacks the panes; without it they stay where they were
// cut and nothing lines up.
inline bool say_panes(const inference& inf, said_reading& said) {
    const auto& split = inf.pane_split;
    const auto& mask = inf.pane_mask;
    if (!split.holds() || !mask.holds()) return false;
    // A PANE TURNED BEFORE IT STACKS. Panes facing each other across a wall are
    // mirrored, not shifted, and pane_mask reads every pane after the first
    // through one pose. Saying that needs those panes named, and a band names a
    // pane only while they lie in one row or one column -- so a lattice read
    // through a pose is a reading with no sentence.
    string turn;
    if (mask.posed != 0) {
        if (split.down != 1 && split.across != 1) {
            said.no_word_for.push_back("a lattice of panes read through a pose");
            return false;
        }
        // Every pane but the first: name the first's band and take the others.
        turn = (split.down == 1 ? " SelectColumn[0]" : " SelectRow[0]") + string(" InvertSelection");
        for (const string& word : words_for_pose(mask.posed)) turn += " " + word;
        turn += " SelectAll";
    }
    const int panes = mask.panes;
    if (panes < 2 || panes > 8) return false;
    const int patterns = 1 << panes;
    const auto& says = mask.by_mask;
    if ((int)says.size() != patterns || !says[0].said()) return false;
    // WHICH COLOR IS THE GROUND IS NOT OBVIOUS. The pattern where no pane has
    // anything is the ground for most shapes -- and is the one and only thing
    // NOR paints, so assuming it costs every NOR its sentence. Each color in the
    // table is tried as the ground instead, and the shape that holds names it.
    // THE ELECTED GROUND IS TRIED FIRST, and that is not a preference but the
    // whole difference between two readings of one table. A table that says
    // "8 where no pane has anything, 0 everywhere else" reads equally as NOR
    // painting 8 and as OR painting 0 -- and only the first is a program, since
    // painting the ground is not the same as leaving it there.
    int8_t grounds[COLOR_COUNT + 2];
    int ground_count = 0;
    auto offer = [&](int8_t color) {
        if (color < 0 || ground_count > COLOR_COUNT) return;
        for (int g = 0; g < ground_count; ++g) if (grounds[g] == color) return;
        grounds[ground_count++] = color;
    };
    offer(inf.background.output_elected());
    offer(inf.background.input_elected());
    for (int m = 0; m < patterns; ++m) {
        if (says[m].said()) offer(says[m].paint);
    }
    if (ground_count == 0) return false;

    // Each pane's own color is what it alone comes to.
    int8_t alone[8];
    bool one_color_each = true;
    for (int p = 0; p < panes; ++p) {
        const auto& only = says[1 << p];
        alone[p] = only.said() ? only.paint : (int8_t)-1;
        if (alone[p] < 0) one_color_each = false;
    }

    auto bits = [](int m) { int n = 0; while (m) { n += m & 1; m >>= 1; } return n; };
    const string head = "InputWithoutBackground Panes" + (turn.empty() ? " SelectAll" : turn) + " Origin";
    for (int g = 0; g < ground_count; ++g) {
        const int8_t ground = grounds[g];
        // Every pattern that paints must paint the same color for the four
        // boolean shapes; a table that does not is an order, or nothing said.
        int8_t lit_color = -1;
        bool one_lit_color = true;
        for (int m = 0; m < patterns; ++m) {
            if (!says[m].said() || says[m].paint == ground) continue;
            if (says[m].paint < 0) { one_lit_color = false; break; }
            if (lit_color < 0) lit_color = says[m].paint;
            else if (lit_color != says[m].paint) { one_lit_color = false; break; }
        }
        if (!one_lit_color || lit_color < 0) continue;
        auto shape_holds = [&](auto predicate) {
            for (int m = 0; m < patterns; ++m) {
                if (!says[m].said()) continue;
                if (predicate(m) != (says[m].paint != ground)) return false;
            }
            return true;
        };
        const string paint = to_string((int)lit_color);
        if (shape_holds([&](int m) { return m == patterns - 1; })) {
            said.sentences.push_back(head + " IntersectionList Recolor[" + paint + "] Union");
            return true;
        }
        if (shape_holds([&](int m) { return m != 0; })) {
            said.sentences.push_back(head + " UnionList Recolor[" + paint + "] Union");
            return true;
        }
        if (shape_holds([&](int m) { return bits(m) == 1; })) {
            // Everything the panes cover, less what they all agree on.
            said.sentences.push_back(head + " UnionList Recolor[" + paint + "] Union");
            said.sentences.push_back(head + " IntersectionList Difference");
            return true;
        }
        if (shape_holds([&](int m) { return m == 0; })) {
            // What the panes do NOT cover: fill the box, then keep only the fill.
            said.sentences.push_back(head + " UnionList AddColor[" + paint + "] Colors SelectColor["
                                     + paint + "] InvertSelection DeleteSelection Union");
            return true;
        }
    }

    // AN ORDER, when the panes do not agree and one of them always wins. Read
    // off the table: whoever takes the pattern where every pane is still in wins
    // first, then the same question again without it.
    const int8_t ground = says[0].paint;
    // THE PANES LAID DOWN AS THEY COME, when what wins is simply the first pane
    // that has anything. Nothing has to be named, so this says the panes whose
    // colors are not one apiece too -- including the ones that never overlap,
    // where there is no order to get wrong.
    if (ground >= 0) {
        bool list_order = true;
        for (int m = 1; m < patterns && list_order; ++m) {
            if (!says[m].said()) continue;
            int first = -1;
            for (int p = 0; p < panes && first < 0; ++p) if (m >> p & 1) first = p;
            list_order = says[m].copy_pane == first ||
                         (says[m].paint >= 0 && alone[first] == says[m].paint);
        }
        if (list_order) {
            said.sentences.push_back(head + " Underlay");
            said.sentences.push_back("Input Panes SelectAll Origin Recolor[" +
                                     to_string((int)ground) + "] Underlay");
            return true;
        }
    }
    if (one_color_each && ground >= 0) {
        int order[8], count = 0, left = patterns - 1;
        while (left) {
            if (!says[left].said()) break;
            int won = -1;
            for (int p = 0; p < panes; ++p)
                if ((left >> p & 1) && alone[p] == says[left].paint) { won = p; break; }
            if (won < 0) break;
            order[count++] = won;
            left &= ~(1 << won);
        }
        if (count == panes) {
            // Every pattern must come to the first pane in that order that has
            // anything -- otherwise the order is a coincidence of the big masks.
            bool every = true;
            for (int m = 0; m < patterns && every; ++m) {
                if (!says[m].said()) continue;
                int8_t wins = ground;
                for (int i = 0; i < panes; ++i)
                    if (m >> order[i] & 1) { wins = alone[order[i]]; break; }
                every = says[m].paint == wins;
            }
            if (every) {
                for (int i = 0; i < panes; ++i)
                    said.sentences.push_back(
                        "Input Colors SelectColor[" + to_string((int)alone[order[i]]) +
                        "] InvertSelection DeleteSelection Panes SelectAll Origin " +
                        (i == 0 ? "Union" : "Underlay"));
                // The canvas starts empty when the outputs elect no ground, so
                // the ground is laid last, under everything already painted.
                said.sentences.push_back("Input Panes SelectAll Origin Recolor[" +
                                         to_string((int)ground) + "] Underlay");
                return true;
            }
        }
    }

    said.no_word_for.push_back("what the panes come to");
    return false;
}

// WHICH PANE THE ANSWER IS, when the answer is a pane and not something drawn
// from them. Only one of the reasons has words: a place in a row of panes, said
// as the band the pane sits in. The others name a property the language has no
// selector for, so they are said as the missing word and nothing is spelled --
// which is the point of saying them at all.
inline bool say_pane_pick(const inference& inf, said_reading& said) {
    const auto& split = inf.pane_split;
    const auto& pick = inf.pane_pick;
    if (!split.holds() || !pick.holds()) return false;
    if (pick.way != pane_pick_rule::AtIndex) {
        said.no_word_for.push_back(
            pick.way == pane_pick_rule::OddColor   ? "the pane whose palette is unlike the others" :
            pick.way == pane_pick_rule::OddOneOut  ? "the pane unlike the others" :
            pick.way == pane_pick_rule::NotSymmetric || pick.way == pane_pick_rule::OnlySymmetric
                                                   ? "the pane that folds unlike the others"
                                                   : "the pane that carries most or least");
        return false;
    }
    // A band names a pane only while the panes lie in one row or one column;
    // across a lattice a place needs two bands at once, and a selector adds
    // rather than narrows, so the two together name a cross and not a pane.
    const bool one_across = split.down == 1, one_down = split.across == 1;
    if (!one_across && !one_down) {
        said.no_word_for.push_back("a place in a lattice of panes");
        return false;
    }
    const string band = one_across ? "SelectColumn[" : "SelectRow[";
    said.sentences.push_back("Input Panes " + band + to_string((int)pick.which) +
                             "] InvertSelection DeleteSelection Origin Union");
    return true;
}

inline said_reading say_the_reading(const inference& inf) {
    const correspondence_election& facts = inf.correspondence;
    said_reading said;
    // Asked first: a split is a fact about the frame, and where it holds the
    // panes are what the pairs are about. Nothing below knows they are there.
    if (say_panes(inf, said)) return said;
    if (say_pane_pick(inf, said)) return said;
    if (say_input_blocks(inf, said)) return said;
    if (facts.pairs.empty()) return said;

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
    // Read again below: a reading that turns out to be one of the layouts lays
    // the input down itself, and then nothing sits on anything.

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
    if (inf.motif_continuation.holds() && inf.size.holds() &&
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
    if (inf.piece_slam.holds()) {
        say(string(inf.piece_slam.rows ? "SlamRow[" : "SlamColumn[") +
            to_string((int)inf.piece_slam.step) + "]");
        return spoken();
    }

    // PIECES LAID INTO A CORNER CHAIN, each hung off the last one's bottom-right
    // cell. Like the slam, the layout drew every output by itself, so it is the
    // whole of what happened and the acts below have nothing left to explain.
    if (inf.overlap_corner.holds()) {
        say("OverlapCorner");
        return spoken();
    }

    // THE TEMPLATE STAMPED ALONG EVERY MARKER. SortBySize[0] hands over the
    // piece with the most cells first, which is the one ProjectOnSeeds repeats,
    // and the markers it repeats it along are the DSL's own -- this line never
    // names them. Like the slam and the chain, the stamping drew every output
    // by itself, so nothing is left for the acts below to explain.
    if (inf.project_on_seeds.holds()) {
        // ProjectOnSeeds hands back the input with the stamps over it, so the
        // pieces found sitting on one another are markers under their own
        // stamps -- the projection working, not a piece laid beneath another.
        // Underlay would answer that by leaving every marker showing through.
        combine = "Union";
        say("SortBySize[0]");
        say("ProjectOnSeeds");
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
    // Whether any piece landed anywhere but where it started, whether or not
    // the act that did it survived being asked of every pair.
    bool a_piece_moved = false;
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
                const int8_t down   = (int8_t)(edge.row - piece.summary.y0);
                const int8_t across = (int8_t)(edge.col - piece.summary.x0);
                a_piece_moved = a_piece_moved || down != 0 || across != 0;
                note(piece.color, act_down,   down);
                note(piece.color, act_across, across);
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

    bool a_recolor_was_said = false, a_move_was_said = false;
    for (const auto& act : acts_in_every_pair) {
        if (act.what == act_recolor) a_recolor_was_said = true;
        if (act.what == act_down || act.what == act_across) a_move_was_said = true;
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

    // A PIECE MOVED AND NOTHING HERE SAYS WHERE IT WENT. An act has to hold in
    // every pair, so a slide that differs from piece to piece -- or one whose
    // pieces wear a different colour in every pair, since an act is asked under
    // the colour it happened to -- leaves the table empty. Empty then reads as
    // "nothing happened", and the line says to segment the input and paint it
    // straight back. That is the language missing a word for the arrangement,
    // not the pieces standing still, and this is the moment to say so.
    if (allowed.translate && a_piece_moved && !a_move_was_said)
        said.no_word_for.push_back("where each piece went");

    // Only confirmed whole-grid swaps can be said without choosing objects.
    // Partial-swap candidates remain available to search.
    for (int variant = 0; variant < COLOR_PAIR_COUNT; ++variant)
        if (inf.color_swaps.whole_grid[variant])
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
