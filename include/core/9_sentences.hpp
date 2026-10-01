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
//
// HAND-WRITE THE PROGRAM FIRST. Before teaching the reading to say a puzzle,
// write its solution by hand in solutions/<id>.txt and check it with arc_spell.
// If the words cannot say it, the gap is the vocabulary, and no reading can
// spell what the DSL has no word for. But write it out before calling a word
// missing: 0bb8deee looked like that gap, and one sentence a pane said it in
// words there already were. Writing those out found the real gap, a smaller
// one: Layout drew each piece from its frame's corner instead of drawing what
// it holds. With that mended, any lattice of panes is one sentence.

#pragma once
#include "7_infer.hpp"
#include "8_vocabulary.hpp"
#include <bit>
#include <optional>
#include <tuple>
#include <string>
#include <vector>

namespace arc {
using namespace std;

// A beat nothing has a word for yet. Said out loud rather than left out, so the
// search knows exactly which one slot it has to fill.
inline constexpr const char* a_blank = "?";

// An op's word with its argument, as a program spells it: Recolor[3].
inline string spell(const string& op, int value) {
    return op + "[" + to_string(value) + "]";
}

// The pick for one color's pieces alone. Picks add to what is already marked,
// so it starts from none picked.
inline string pick_only_color(int color) {
    return "SelectAll InvertSelection " + spell("SelectColor", color);
}

// The pieces of any of the colors in `colors`, a bit each, picked from none.
// Selectors add, so each color is one more SelectColor.
inline string pick_colors(uint16_t colors) {
    string pick = "SelectAll InvertSelection";
    for (int color = 0; color < COLOR_COUNT; ++color)
        if (colors >> color & 1) pick += " " + spell("SelectColor", color);
    return pick;
}

// A PICK AS WORDS, picked from none: by a kind of pick -- holes, size, band,
// the color most cells wear, every color but that one -- when `how` names one,
// else by its colors.
inline string pick_words(uint16_t colors, int8_t how, int8_t value) {
    using fired_rule = correspondence_election::fired_rule;
    static constexpr const char* selectors[] = { "SelectObjectsWithHoles", "SelectLargestObject",
                                                 "SelectSmallestObject", "SelectRow", "SelectColumn",
                                                 "SelectMostCommonColor",
                                                 "SelectMostCommonColor InvertSelection" };
    static_assert(size(selectors) == (size_t)fired_rule::pick_kinds);
    // SelectMostCommonColor takes no value: which color it picks is the list's to say.
    if (how == fired_rule::most_common || how == fired_rule::not_most_common)
        return string("SelectAll InvertSelection ") + selectors[how];
    if (how >= 0) return "SelectAll InvertSelection " + spell(selectors[how], value);
    return pick_colors(colors);
}

// THE PIECES A RULE PICKS, as words, picked from none: by its pick of holes or of
// size, by its colors, by its one color -- or every piece.
inline string pick_of(const correspondence_election::fired_rule& rule) {
    if (rule.pick_by >= 0 || rule.picks) return pick_words(rule.picks, rule.pick_by, rule.pick_value);
    return rule.color < 0 ? string("SelectAll") : pick_only_color(rule.color);
}

struct said_reading {
    vector<string> sentences;
    vector<string> no_word_for;
    // OPENINGS THE READING COULD NOT RULE OUT. Only the first sentence can be
    // said more than one way -- everything after it hangs off the chosen
    // reading's own pieces, so a rival segmentation has no follow-on to go with it.
    vector<string> other_openings;
    // Cells of the pairs these sentences leave unexplained. A rule that speaks
    // holds over every cell, so only the cut's own line leaves any.
    int left_unexplained = 0;
};

// WHAT AN OBJECT IS, as the operation that segments the same way. Split's argument is
// the same-color flag: Split[1] keeps colors separate; Split[0] joins them.
inline const char* word_for_segmentation(segmentation_methods segmentation) {
    switch (segmentation) {
        case segmentation_methods::connected_monochrome:   return "Split[1]";
        case segmentation_methods::connected_multicolor:   return "Split[0]";
        case segmentation_methods::all_cells_of_one_color: return "Colors";
        case segmentation_methods::minimal_repeating_set:  return "Pattern";
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
inline optional<string> word_for_color_map(const recoloring& map) {
    int first = -1, second = -1;
    for (int c = 0; c < COLOR_COUNT; ++c)
        if (map.changes(c)) { first = c; second = map[c]; break; }
    if (first < 0) return string();
    for (int c = 0; c < COLOR_COUNT; ++c) {
        const int mapped = c == first ? second : c == second ? first : c;
        if (map[c] >= 0 && map[c] != mapped) return nullopt;
    }
    const int variant = color_pair{first, second}.index();
    if (variant < 0) return nullopt;
    return spell("BidiColorMap", variant);
}

// How much of the object list is selected after these words.
// Selectors add to the selection; carving and deletion can clear it.
enum class the_mark { none, some, all };

inline the_mark mark_after(const string& words) {
    the_mark mark = the_mark::none;
    each_program_word(words, [&](std::string_view, node_type op) {
        if (op == node_type::SelectAll) mark = the_mark::all;
        else if (op == node_type::InvertSelection)
            mark = mark == the_mark::all ? the_mark::none
                 : mark == the_mark::none ? the_mark::all : the_mark::some;
        else if (is_select_op(op)) {
            if (mark != the_mark::all) mark = the_mark::some;
        }
        else if (clears_the_marks(op)) mark = the_mark::none;
        // Bridge hands its bars back unmarked beside the pieces it was given,
        // so after it the hand is only partly picked, and a paint that means the
        // whole of it has to say so. Said here for the one op, where CLEARS
        // would say it for all of them if the table had a column for it.
        else if (op == node_type::Bridge && mark == the_mark::all) mark = the_mark::some;
    });
    return mark;
}

// ONE SENTENCE, CLOSED, AND THE PICK A PAINT NEEDS SAID HERE AND NOWHERE ELSE.
// Union, Underlay, Intersection and Difference land only what is marked, and a
// pick in these sentences is how the ACT before it was narrowed -- never what
// the paint was for: every sentence this file spells means to paint the whole
// hand. So where the words have not left all of it marked, the hand is said out
// loud before the paint. Withhold lands nothing, so it asks for nothing.
// Said at the one point every sentence goes through, for the reason the kernel
// carries its marks at one point: a branch that never heard of the rule cannot
// forget it, and the next branch written here will not either.
inline string closed_with(const string& words, const string& combine) {
    const bool paints = combine != "Withhold";
    return words + (paints && mark_after(words) != the_mark::all ? " SelectAll " : " ") + combine;
}

// A SENTENCE THAT MEANS TO PAINT ONLY WHAT IT PICKED. The closer above says the
// hand out loud because every sentence it closes means the whole of it; this one
// is for the sentences that narrow on purpose -- lay down this layer, keep the
// rest in hand for the sentence after -- which is the whole of what a gated
// paint is for. Two words for the two intents, so the call site says which it
// means rather than the closer guessing.
inline string closed_on_the_pick(const string& words, const string& combine) {
    return words + " " + combine;
}

// ONE SENTENCE THAT OPENS WITH A SOURCE OF ITS OWN, as written, closed.
inline void say_closed(said_reading& said, const string& words, const string& combine) {
    said.sentences.push_back(closed_with(words, combine));
}

// ONE SENTENCE. Nothing is implicit: the first names its source, the rest carry
// on from the objects already in hand -- unless it opens a line `afresh`, which
// reads its source again instead of what the sentences before it left in hand.
inline void say_sentence(said_reading& said, const string& source, const string& pick,
                         const string& act, const string& combine, bool afresh = false) {
    said.sentences.push_back(closed_with(
        (said.sentences.empty() || afresh ? source : string("CurrentObjects")) + " " + pick + " " + act,
        combine));
}

// ONE BEAT OF A LINE: the pieces picked, and the one word done to them.
struct beat { string pick, act; };

// A LINE OF BEATS, SAID AS SENTENCES. Only the last paints: a cell drawn before
// its piece has turned or slid is never taken back, so every beat before it
// withholds. A line said `afresh` opens with its own source, then carries on.
inline void say_beats(said_reading& said, const string& source,
                      const vector<beat>& beats, const string& combine, bool afresh = false) {
    for (size_t i = 0; i < beats.size(); ++i)
        say_sentence(said, source, beats[i].pick, beats[i].act,
                     i + 1 < beats.size() ? "Withhold" : combine, afresh && i == 0);
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

// A sentence ends at a combine operation. Keep only complete sentences,
// as before; the same rows are used by the GUI and the log.
inline vector<string> sentences_of(const string& program) {
    vector<string> rows;
    string row;
    each_program_word(program, [&](std::string_view word, node_type op) {
        if (!row.empty()) row += ' ';
        row += word;
        if (is_combine_op(op)) rows.push_back(exchange(row, string()));
    });
    return rows;
}

// Every sentence names its source, including CurrentObjects when it carries
// the previous sentence's objects forward.
inline const char* sentence_with_no_source(const string& program) {
    for (const string& sentence : sentences_of(program)) {
        const auto first = std::string_view(sentence).substr(0, sentence.find(' '));
        if (!is_get_op(operation_named(first))) return "a sentence that opens with no source";
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

    // THE NEGATIVE HAS NO WORD OF ITS OWN, but where the copies sit on the
    // input's own figure it has a spelling: grow every cell into a block of its
    // own color, then carve the copies back out. A figure cell's block is left
    // wearing the figure's color exactly where the input was ground, and a
    // ground cell's block was ground already. The carve reads the input again,
    // since what it cuts is the copies and not what the growing left in hand.
    if (blocks.negative) {
        const int pose = !some_are_ground ? -1
            : cheapest([&](int p) { return (blocks.figure_poses >> p & 1) != 0; });
        if (some_are_ground && pose < 0) {
            said.no_word_for.push_back("which blocks hold a copy of the input");
            return false;
        }
        // Grown, the input covers the output only when every pair grew by one
        // whole factor both ways, and that factor is how many blocks a side holds.
        const int grown = blocks.pairs.front().rows;
        if (pose < 0 || !takes_value(node_type::ScaleUp, grown) ||
            ranges::any_of(blocks.pairs, [&](const auto& at) {
                return at.rows != grown || at.cols != grown; })) {
            said.no_word_for.push_back("a copy with its figure and ground traded");
            return false;
        }
        vector<beat> grow, carve;
        say_pose(grow, "SelectAll", pose);
        say_pose(carve, "SelectAll", pose);
        grow.push_back({"SelectAll", spell("ScaleUp", grown)});
        carve.push_back({"SelectAll", "Fractal"});
        say_beats(said, "Input", grow, "Union");
        say_beats(said, "InputWithoutBackground", carve, "Difference", /*afresh=*/true);
        return true;
    }

    string source = "Input";
    vector<beat> beats;
    if (some_are_ground) {
        // The ground is punched out before stamping, so no map may repaint it.
        const int ground = inf.background.input_elected();
        const int pose = cheapest([&](int p) { return (blocks.figure_poses >> p & 1) != 0; });
        if (pose < 0 || !is_color(ground) || blocks.color_map.changes(ground)) {
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
                    if (poses[band] == pose) bands += " " + spell(select, (int)band);
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
        const auto& only = says[(size_t)1 << p];
        alone[p] = only.said() ? only.paint : (int8_t)-1;
        if (alone[p] < 0) one_color_each = false;
    }

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
        const string recolor = " " + spell("Recolor", lit_color);
        if (shape_holds([&](int m) { return m == patterns - 1; })) {
            say_closed(said, head + " IntersectionList" + recolor, "Union");
            return true;
        }
        if (shape_holds([&](int m) { return m != 0; })) {
            say_closed(said, head + " UnionList" + recolor, "Union");
            return true;
        }
        if (shape_holds([&](int m) { return popcount((unsigned)m) == 1; })) {
            // Everything the panes cover, less what they all agree on.
            say_closed(said, head + " UnionList" + recolor, "Union");
            say_closed(said, head + " IntersectionList", "Difference");
            return true;
        }
        if (shape_holds([&](int m) { return m == 0; })) {
            // What the panes do NOT cover: fill the box, then keep only the fill.
            say_closed(said, head + " UnionList " + spell("AddColor", lit_color) + " Colors " +
                             spell("SelectColor", lit_color) + " InvertSelection DeleteSelection", "Union");
            return true;
        }
    }

    // AN ORDER, when the panes do not agree and one of them always wins. Read
    // off the table: whoever takes the pattern where every pane is still in wins
    // first, then the same question again without it.
    const int8_t ground = says[0].paint;
    // The ground is laid last, under everything already painted, since the
    // canvas starts empty when the outputs elect no ground.
    const string lay_ground = "Input Panes SelectAll Origin " + spell("Recolor", ground);
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
            say_closed(said, head, "Underlay");
            say_closed(said, lay_ground, "Underlay");
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
                    say_closed(said, "Input Colors " + spell("SelectColor", alone[order[i]]) +
                                     " InvertSelection DeleteSelection Panes SelectAll Origin",
                               i == 0 ? "Union" : "Underlay");
                say_closed(said, lay_ground, "Underlay");
                return true;
            }
        }
    }

    said.no_word_for.push_back("what the panes come to");
    return false;
}

// THE CHOSEN PANE, directly or fitted behind the walls. Occupancy has words
// after removing the background; a place has words in a row or column. Other
// choices still report the selector they are missing.
inline bool say_pane_pick(const inference& inf, said_reading& said) {
    const auto& split = inf.pane_split;
    const auto& pick = inf.pane_pick;
    if (!split.holds() || !pick.holds()) return false;
    if (pick.fitted && (pick.way == pane_pick_rule::Emptiest || pick.way == pane_pick_rule::Fullest)) {
        const int ground = inf.background.input_elected();
        if (!is_color(ground)) return false;
        // The walls are laid back by their color, so a wall whose color each
        // grid says for itself has no word to lay it back with.
        if (!is_color(split.color)) {
            said.no_word_for.push_back("walls laid back in each grid's own color");
            return false;
        }
        // Size counts opaque cells, so remove the ground before cutting panes.
        // Restore it after painting; the selected pane keeps its full frame.
        const string selector = pick.way == pane_pick_rule::Emptiest
                              ? "SelectSmallestObject[0]" : "SelectLargestObject[0]";
        say_closed(said, "InputWithoutBackground Panes " + selector +
            " InvertSelection DeleteSelection SelectAll Fit", "Union");
        say_closed(said, "Input Colors " + spell("SelectColor", split.color) +
            " InvertSelection DeleteSelection", "Union");
        say_closed(said, "Input SelectAll " + spell("Recolor", ground), "Underlay");
        return true;
    }
    if (pick.fitted) {
        said.no_word_for.push_back("the chosen pane enlarged behind the walls");
        return false;
    }
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
    const char* band = one_across ? "SelectColumn" : "SelectRow";
    // The hand is taken up again after the delete, and not only for the paint:
    // the delete hands back the panes that were NOT picked and leaves nothing
    // marked, and Origin is an act, so with nothing picked it would leave the
    // pane sitting where it was cut instead of stacking it at the corner.
    say_closed(said, "Input Panes " + spell(band, pick.which) +
                     " InvertSelection DeleteSelection SelectAll Origin", "Union");
    return true;
}

// THE INPUT THROUGH THE SIZE RULE'S BOX, AT ONE CORNER. The canvas is the box,
// so painting the input crops it, and a push down or right brings that side into
// view. It is a place and not a reason, so a pane picked for a reason outranks
// it: a reason still names the pane on a test where it has moved, a corner does
// not. A pane picked only for its place is this, said without the lattice.
inline bool say_corner_crop(const inference& inf, said_reading& said) {
    const auto& crop = inf.corner_crop;
    if (!crop.holds()) return false;
    if (inf.pane_pick.holds() && inf.pane_pick.way != pane_pick_rule::AtIndex) return false;
    string words = "Input SelectAll";
    if (crop.down) words += " " + spell("MoveToEdge", heading::down);
    if (crop.right) words += " " + spell("MoveToEdge", heading::right);
    say_closed(said, words, "Union");
    return true;
}

// PIECES SLID TO AN EDGE. What stays is painted where it lies, then the pieces
// that slide are pushed to the edge and laid over or under it. Both sentences
// paint only their pick, since the pick is the whole point: the one lays what
// stays, the other what moved. Every piece sliding is one sentence.
inline bool say_edge_slide(const inference& inf, said_reading& said) {
    const auto& slide = inf.edge_slide;
    if (!slide.holds()) return false;
    const char* cut = word_for_segmentation(slide.segmentation);
    if (!cut) return false;
    const string pieces = string("Input ") + cut;
    const string push = spell("MoveToEdge", slide.way);
    if (!is_color(slide.color)) {
        say_closed(said, pieces + " SelectAll " + push, "Union");
        return true;
    }
    const string pick = pieces + " " + spell("SelectColor", slide.color);
    said.sentences.push_back(closed_on_the_pick(pick + " InvertSelection", "Union"));
    said.sentences.push_back(closed_on_the_pick(pick + " " + push, slide.over ? "Union" : "Underlay"));
    return true;
}

// EVERY PANE'S DRAWING, CROPPED, WHERE ITS PANE WAS. Panes hands each pane
// back in its own frame, and Layout bands pieces by their frames and draws what
// each holds, so the panes' rows and columns become the layout's -- however
// many there are -- and each cell gets its pane's figure wherever it sat.
// Layout[0] takes the cell from the drawings, so no size is written here.
inline bool say_pane_gather(const inference& inf, said_reading& said) {
    if (!inf.pane_split.holds() || !inf.pane_gather.holds()) return false;
    say_closed(said, "InputWithoutBackground Panes " + spell("Layout", 0), "Union");
    return true;
}

// The size-ranked color layers are independent sources: each gets its own
// extension, and the seed body is a separate paint when the rule calls for it.
inline bool say_ray_extension(const inference& inf, said_reading& said) {
    const auto& rule = inf.ray_extension;
    if (!rule.holds()) return false;

    for (const auto& branch : rule.branches) {
        string layer = "Input Colors " + spell("SelectSmallestObject", branch.rank) +
                       " InvertSelection DeleteSelection SelectAll Trim";
        if (branch.fill_interior) layer += " InteriorColor[-1]";
        layer += " Split[1] SelectAll Trim";
        const string extend = spell(branch.direction == ray_extension_rule::Orthogonal
                                    ? "ExtendOrthogonal" : "ExtendDiagonal", branch.radius);
        if (branch.stamp_body) {
            say_closed(said, layer, "Union");
            say_closed(said, "CurrentObjects SelectAll " + extend, "Union");
        } else {
            say_closed(said, layer + " " + extend, "Union");
        }
    }
    return true;
}

// THE MARKS AS LINES THAT TAKE TURNS, cut the way the reading replayed them.
// A pair lying on its side is turned to lay its lines and turned back before
// they paint, and only when the reading needed the turn.
inline bool say_mark_lattice(const inference& inf, said_reading& said) {
    const auto& rule = inf.mark_lattice;
    if (!rule.holds()) return false;
    const char* cut = word_for_segmentation(rule.segmentation);
    if (!cut) return false;
    string words = string("Input ") + cut + " SelectAll Trim";
    if (rule.portrait) words += " NormalizeOrientation[0]";
    words += " TileX[1] Tile[3]";
    if (rule.portrait) words += " RestoreOrientation";
    say_closed(said, words, "Union");
    return true;
}

inline bool say_ranked_layout(const inference& inf, said_reading& said) {
    const auto& rule = inf.ranked_layout;
    if (!rule.holds()) return false;

    string words = "Input Colors";
    for (int rank = 0; rank < rule.discarded_colors; ++rank)
        words += " SelectMostCommonColor DeleteSelection";
    words += " " + spell("Split", rule.split_mode) + " SelectAll Trim " +
             spell("Layout", rule.cell_size) + " SelectAll " +
             spell("ScaleDown", rule.cell_size);
    say_closed(said, words, "Union");
    return true;
}

// THE INPUT REPEATS, AND THE OUTPUT CARRIES THE REPEAT. Pattern cuts the input
// into its repeating block, the block's own cell-level map is spelled once, and
// Tile[0] carries it across. A claim about the grid, read off the block's own
// matching, so it is said whichever way of seeing objects won the election; the
// generic object recolor of the cut's own line has another scope.
inline bool say_motif_continuation(const inference& inf, said_reading& said) {
    const auto& motif = inf.motif_continuation;
    if (!motif.holds() || !inf.size.holds() ||
        inf.size.Chosen == output_dimension_rule::BoundingBox) return false;
    const auto map_word = word_for_color_map(motif.color_map);
    if (!map_word) {
        said.no_word_for.push_back("the motif's cell-color mapping");
        return false;
    }
    vector<beat> beats = { {"SelectAll", word_for_segmentation(segmentation_methods::minimal_repeating_set)} };
    if (!map_word->empty()) beats.push_back({"SelectAll", *map_word});
    beats.push_back({"SelectAll", "Tile[0]"});
    say_beats(said, "Input", beats, "Union");
    return true;
}

// A NOISY LATTICE, VOTED CLEAN. The whole input is the one piece the vote
// reads, so there is nothing to cut first and nothing left to explain.
inline bool say_periodic_denoise(const inference& inf, said_reading& said) {
    if (!inf.periodic_denoise.holds()) return false;
    say_sentence(said, "Input", "SelectAll", "VoteDenoise", "Union");
    return true;
}

// THE GAPS BETWEEN PIECES OF ONE COLOUR, FILLED. Said under the cut its
// replay held under, which need not be the elected one, and laid under what
// is there, so a bar that crosses a wall goes behind it. The paint needs the
// whole hand said out loud, since the bars come back unmarked (mark_after).
inline bool say_bridge(const inference& inf, said_reading& said) {
    if (!inf.bridge.holds()) return false;
    const char* cut = word_for_segmentation(inf.bridge.segmentation);
    if (!cut) return false;
    say_beats(said, "Input", { {"SelectAll", cut}, {"SelectAll", "Trim"},
                               {"SelectAll", "Bridge"} }, "Underlay");
    return true;
}

// Every hollow filled with the one color the rule read.
inline bool say_interior_fill(const inference& inf, said_reading& said) {
    if (!inf.interior_fill.holds()) return false;
    say_closed(said, "InputWithoutBackground SelectAll Split[0] SelectAll Trim " +
               spell("InteriorColor", inf.interior_fill.color), "Union");
    return true;
}

// Paint the cores, then restore the original boundaries. Reread the input for
// Outline: the recolored hand has lost its boundary colors.
inline bool say_interior_spread(const inference& inf, said_reading& said) {
    if (!inf.interior_spread.holds()) return false;
    say_closed(said, "Input", "Union");
    const string pieces = "InputWithoutBackground SelectAll Split[0] SelectAll";
    say_closed(said, pieces + " Recolor[10]", "Union");
    say_closed(said, pieces + " Outline", "Union");
    return true;
}

// Each piece recolored by how many holes it has.
inline bool say_hole_colors(const inference& inf, said_reading& said) {
    if (!inf.hole_colors.holds()) return false;
    vector<beat> recolors = { {"SelectAll", "Split[1]"} };
    for (int holes = 0; holes < hole_color_rule::key_count; ++holes)
        if (is_color(inf.hole_colors.colors[holes]))
            recolors.push_back({"SelectAll InvertSelection " + spell("SelectObjectsWithHoles", holes),
                                spell("Recolor", inf.hole_colors.colors[holes])});
    say_beats(said, "InputWithoutBackground", recolors, "Union");
    return true;
}

// The largest figure copied onto the others, the input laid down first.
inline bool say_copy_to(const inference& inf, said_reading& said) {
    if (!inf.copy_to.holds()) return false;
    say_closed(said, "Input", "Union");
    const string pieces = string("InputWithoutBackground ") +
        (inf.copy_to.panes ? "Panes " : "") +
        "Split[0] SelectAll Trim InvertSelection SelectLargestObject[0] ";
    said.sentences.push_back(closed_on_the_pick(pieces + spell("CopyTo", inf.copy_to.anchor), "Union"));
    return true;
}

// The least common color alone. The size rule's box is the crop, so the
// sentence only picks that color and paints it where it lies.
inline bool say_rarest_color(const inference& inf, said_reading& said) {
    if (!inf.rarest_color.holds()) return false;
    said.sentences.push_back(closed_on_the_pick("Input Colors SelectSmallestObject[0]", "Union"));
    return true;
}

// ONE RULE ON THE CUT'S LINE, AS WORDS: the beats it adds. The Trim picked every
// piece, so a line about one color picks it from none picked.
inline vector<beat> beats_of(const correspondence_election::fired_rule& rule, const inference& inf) {
    using said_as = correspondence_election::fired_rule::said_as;
    const string pick = pick_of(rule);
    vector<beat> beats;
    switch (rule.as) {
        case said_as::turn:
            for (const string& word : words_for_pose(rule.value)) beats.push_back({pick, word});
            break;
        case said_as::move_down:   beats.push_back({pick, spell("MoveColumn", rule.value)}); break;
        case said_as::move_across: beats.push_back({pick, spell("MoveRow", rule.value)}); break;
        case said_as::recolor:     beats.push_back({pick, spell("Recolor", rule.value)}); break;
        case said_as::color_from:  beats.push_back({pick, spell("ColorFrom", rule.value)}); break;
        case said_as::align_to:    beats.push_back({pick, spell("AlignTo", rule.value)}); break;
        case said_as::mirror_around: beats.push_back({pick, spell("MirrorAround", rule.value)}); break;
        case said_as::outlined:    beats.push_back({pick, "Outline"}); break;
        case said_as::recolor_picked: beats.push_back({pick, spell("Recolor", rule.value)}); break;
        case said_as::removed:     beats.push_back({pick, "DeleteSelection"}); break;
        case said_as::to_edge:     beats.push_back({pick, spell("MoveToEdge", rule.value)}); break;
        case said_as::shifted:     beats.push_back({pick, spell("Shift", rule.value)}); break;
        case said_as::swap_colors: beats.push_back({"SelectAll", spell("BidiColorMap", rule.value)}); break;
        case said_as::learned_table:
            if (const char* word = word_for_learned_recolor(inf.shape_color_map))
                beats.push_back({"SelectAll", word});
            break;
        case said_as::size_rank:
            // A color layer can hold several disconnected pieces; the rank
            // belongs to each piece, not their combined color.
            if (inf.correspondence.segmentation == segmentation_methods::all_cells_of_one_color)
                beats.push_back({"SelectAll", "Split[0]"});
            // SizeRankColorMap reads list order, so say the sort too.
            beats.push_back({"SelectAll", "SortBySize[0]"});
            beats.push_back({"SelectAll", "SizeRankColorMap"});
            break;
        default: break;
    }
    return beats;
}

// HOW A PAINT LIES ON WHAT IS DOWN, as words: over it, Union; under it,
// Underlay, which lands only where nothing is; only onto it, MaskBy[10] -- any
// color -- and then Union, so it lands only where something is.
inline string onto_words(const correspondence_election::fired_rule& rule) {
    return rule.lay == correspondence_election::fired_rule::laid_onto ? " " + spell("MaskBy", 10) : string();
}
inline const char* laid_paint(const correspondence_election::fired_rule& rule) {
    return rule.lay == correspondence_election::fired_rule::laid_under ? "Underlay" : "Union";
}

// WHAT A RULE DRAWS OFF THE PIECES, as the one beat of a sentence said after the
// line: the pieces picked and the word that draws -- rays, bars, filled hollows,
// trails, filled boxes, copies, repeats -- and whether what it draws goes over what is
// already down or under it. False for a rule that draws nothing after the line.
inline bool drawn_after(const correspondence_election::fired_rule& rule, beat& drawn, const char*& paint) {
    using said_as = correspondence_election::fired_rule::said_as;
    const string pick = pick_of(rule);
    switch (rule.as) {
        case said_as::extend_rays: {
            // Rays only, the pieces are already down; painted over in the rule's
            // own color unless it is the one color picked.
            string act = spell(rule.value >= 32 ? "ExtendDiagonal" : "ExtendOrthogonal", rule.value % 32);
            if (rule.picks != (uint16_t)(1u << rule.paint)) act += " " + spell("Recolor", rule.paint);
            drawn = { pick, act + onto_words(rule) };
            paint = laid_paint(rule);
            return true;
        }
        case said_as::bridged: {
            // Bridge reads every piece and hands the bars back unmarked, so the
            // inversion picks the bars alone, to paint them and to paint over.
            string act = "Bridge InvertSelection";
            if (rule.paint >= 0) act += " " + spell("Recolor", rule.paint);
            drawn = { "SelectAll", act };
            paint = "Underlay";
            return true;
        }
        case said_as::filled:      drawn = { pick, spell("InteriorColor", rule.paint) }; paint = "Underlay"; return true;
        case said_as::trailed:     drawn = { pick, spell("ExtendTo", rule.value) };      paint = "Union";    return true;
        case said_as::boxed:       drawn = { pick, spell("AddColor", rule.paint) };      paint = "Underlay"; return true;
        case said_as::copied_onto: drawn = { pick, spell("CopyTo", rule.value) };        paint = "Union";    return true;
        case said_as::tiled_along:
            drawn = { pick, spell(rule.value >= 32 ? "TileY" : "TileX", rule.value % 32) + onto_words(rule) };
            paint = laid_paint(rule);
            return true;
        default:                   return false;
    }
}

// THE FRAME'S WORDS IN FRONT OF A LINE: each sentence that opens on the input
// opens instead on the input framed -- the piece the frame's pick takes, cut out
// and moved to the corner (the cut, the pick, every other piece deleted, then
// Origin), or the whole input scaled up, scaled down or tiled -- and the line's
// own cut carves that. False for a line that opens any other way.
inline bool frame_the_sentences(const piece_frame_rule& frame, said_reading& said) {
    string framed;
    switch (frame.kind) {
    case piece_frame_rule::scaled_up:   framed = "Input SelectAll " + spell("ScaleUp", frame.factor); break;
    case piece_frame_rule::scaled_down: framed = "Input SelectAll " + spell("ScaleDown", frame.factor); break;
    case piece_frame_rule::tiled:       framed = "Input SelectAll " + spell("Tile", 0); break;
    case piece_frame_rule::fractal:     framed = "InputWithoutBackground SelectAll Fractal"; break;
    case piece_frame_rule::repeat_unit: framed = "Input SelectAll Pattern"; break;
    case piece_frame_rule::posed:
        framed = "Input SelectAll";
        for (const string& word : words_for_pose(frame.factor)) framed += " " + word;
        break;
    default: {
        const char* cut = word_for_segmentation(frame.cut);
        if (!cut) return false;
        // Every piece needs no pick, and nothing to delete.
        const bool every = frame.how < 0 && frame.colors == (uint16_t)((1u << COLOR_COUNT) - 1);
        framed = string("Input SelectAll ") + cut + " SelectAll Trim " +
                 (every ? string() : pick_words(frame.colors, frame.how, frame.value) +
                                     " InvertSelection DeleteSelection ") +
                 (frame.kind == piece_frame_rule::laid_out ? "SelectAll " + spell("Layout", frame.factor)
                  : string(frame.merged ? "SelectAll UnionList " : "") + "SelectAll Origin");
    }
    }
    auto reframe = [&](string& sentence) {
        if (sentence.rfind("CurrentObjects ", 0) == 0) return true;
        if (sentence.rfind("Input ", 0) != 0) return false;
        sentence = framed + sentence.substr(5);
        return true;
    };
    for (string& sentence : said.sentences)
        if (!reframe(sentence)) return false;
    for (string& opening : said.other_openings)
        if (!reframe(opening)) return false;
    return !said.sentences.empty();
}

// A RULE THAT STANDS ALONE, SAID: the whole program of the rule it names.
inline bool say_rival(const correspondence_election::fired_rule& rule, const inference& inf,
                      said_reading& said) {
    using speaker = bool (*)(const inference&, said_reading&);
    // In own_sentences order.
    static constexpr speaker by_rule[] = {
        say_panes, say_corner_crop, say_pane_pick, say_pane_gather, say_ray_extension,
        say_mark_lattice, say_input_blocks, say_periodic_denoise, say_bridge,
        say_interior_fill, say_interior_spread, say_hole_colors, say_ranked_layout,
        say_edge_slide, say_copy_to, say_rarest_color, say_motif_continuation };
    static_assert(size(by_rule) == (size_t)own_sentences::count);
    return rule.value >= 0 && rule.value < (int)size(by_rule) && by_rule[rule.value](inf, said);
}

// THE ELECTED CUT'S OWN LINE: its pieces, cut and trimmed, and the acts every
// pair agrees on. An explanation like any rule's, and what it leaves
// unexplained is the reading's left over that nothing it says accounts for --
// a rule with a sentence of its own is a rival, so what that rule claims is
// still unsaid here.
inline bool say_the_cut(const inference& inf, said_reading& said) {
    const correspondence_election& facts = inf.correspondence;
    if (facts.chosen_pairs().empty()) return false;
    const auto& read = facts.readings[facts.chosen];
    said.left_unexplained = read.unclaimed();
    // Pieces that do not redraw the output where they lie, with nothing on the
    // line to say what else happened.
    if (!inf.pieces_redraw && read.line.empty())
        said.no_word_for.push_back("what happened to the pieces the reading matched");

    // A piece that sits on what is already there is laid under it, not over.
    const bool some_sit_on = ranges::any_of(facts.chosen_pairs(), [](const auto& pair) {
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
    auto spoken = [&]() {
        say_beats(said, "Input", beats, combine);
        // A tied opening closes the way the chosen one did, so the rest of the
        // line still follows it.
        const string close = beats.size() > 1 ? "Withhold" : combine;
        for (const char* segmentation : tied) {
            string opening = closed_with(string("Input SelectAll ") + segmentation, close);
            bool already = !said.sentences.empty() && said.sentences.front() == opening;
            for (const string& seen : said.other_openings) already = already || seen == opening;
            if (!already) said.other_openings.push_back(opening);
        }
        return true;
    };

    if (const char* segmentation = word_for_segmentation(facts.segmentation)) {
        say(segmentation);
        // Correspondence measured every piece in the box its cells make, and Split
        // and Colors hand pieces back in the frame they were given (only Trim
        // shrinks one), so the pieces are trimmed to those boxes first.
        if (boxed_by_its_cells(facts.segmentation)) say("Trim");
    }
    else said.no_word_for.push_back(string("the segmentation ") + segmentation_names[(int)facts.segmentation]);
    // The cut as its words open the line, said again by every paint after it.
    const vector<beat> opening = beats;

    for (int8_t cell : facts.tied_with_chosen) {
        if (cell < 0 || cell >= (int8_t)facts.readings.size()) continue;
        if (const char* segmentation = word_for_segmentation(facts.readings[cell].segmentation))
            tied.push_back(segmentation);
    }

    // A RULE THAT DREW EVERY OUTPUT BY ITSELF from the cut's own pieces is the
    // line's first and only entry, and is said its own way.
    using said_as = correspondence_election::fired_rule::said_as;
    const said_as whole = read.line.empty() ? said_as::unsaid : read.fired[read.line.front()].as;

    // THE SAME CONTINUATION, READ ALONG THE DIAGONALS. Which color sits on which
    // diagonal is read off the grid in hand, so the sentence carries the lean and
    // nothing else -- that is what makes DiagTile a rule and not a table. Said
    // below the tied openings rather than above them, because all the op reads is
    // the input's own cells where they sat, and any opening that hands those over
    // is as good as the one the election happened to pick.
    if (whole == said_as::diagonal_tile) {
        say(spell("DiagTile", inf.diagonal_continuation.lean));
        return spoken();
    }

    // PIECES THAT PILED UP INSIDE THEIR OWN BOXES. The slide drew every output
    // by itself, so it is the whole of what happened.
    if (whole == said_as::piece_slam) {
        say(spell(inf.piece_slam.rows ? "SlamRow" : "SlamColumn", inf.piece_slam.step));
        return spoken();
    }

    // PIECES LAID INTO A CORNER CHAIN, each hung off the last one's bottom-right
    // cell. Like the slam, the layout drew every output by itself, so it is the
    // whole of what happened and the acts below have nothing left to explain.
    if (whole == said_as::corner_chain) {
        say("OverlapCorner");
        return spoken();
    }

    // PIECES SLID UP AGAINST THE ONES THAT STAY. Two sentences that both paint,
    // which is why this is not said in beats: the still pieces have to be ON the
    // canvas before the others slide, because the canvas is the whole of what
    // AlignToCanvas measures against. Each names its layer by colour and paints
    // that layer alone -- the other is still in hand, waiting for its own
    // sentence. Before a paint read the mark, saying this meant deleting every
    // other layer first and carving the input again for the next one.
    if (whole == said_as::slide_to_touch) {
        // The Trim above picked every piece, so each line picks its colour
        // from none picked.
        say_beats(said, "Input", beats, "Withhold");
        said.sentences.push_back(closed_on_the_pick(
            (said.sentences.empty() ? string("Input") : string("CurrentObjects")) +
            " " + pick_only_color(inf.slide_to_touch.stays), "Union"));
        said.sentences.push_back(closed_on_the_pick(
            "CurrentObjects " + pick_only_color(inf.slide_to_touch.slides) + " AlignToCanvas", "Union"));
        return true;
    }

    // THE TEMPLATE STAMPED ALONG EVERY MARKER. SortBySize[0] hands over the
    // piece with the most cells first, which is the one ProjectOnSeeds repeats,
    // and the markers it repeats it along are the DSL's own -- this line never
    // names them. Like the slam and the chain, the stamping drew every output
    // by itself, so nothing is left for the acts below to explain.
    if (whole == said_as::project_on_seeds) {
        // ProjectOnSeeds hands back the input with the stamps over it, so the
        // pieces found sitting on one another are markers under their own
        // stamps -- the projection working, not a piece laid beneath another.
        // Underlay would answer that by leaving every marker showing through.
        combine = "Union";
        say("SortBySize[0]");
        say("ProjectOnSeeds");
        const int8_t worn = inf.project_on_seeds.worn;
        if (!is_color(worn)) return spoken();
        // THE STAMPS IN ONE COLOUR, THE INPUT LAID BACK OVER THEM. The copy of
        // the input the word hands back holds every cell, ground and all, so it
        // is the largest thing in hand, and it goes before the recolour, which
        // would otherwise paint the whole canvas. The pick starts from none,
        // since the stamps come back marked. Then the input's own cells go on
        // top, so a stamp shows only where the input had ground.
        beats.push_back({"SelectAll InvertSelection SelectLargestObject[0]", "DeleteSelection"});
        say(spell("Recolor", worn));
        spoken();
        say_closed(said, "InputWithoutBackground", "Union");
        return true;
    }

    // BODIES SNAPPED ONTO THEIR ANCHORS. The op reads every piece the cut hands
    // over as one grid and hands back only the bodies it set down, so they are
    // all that is painted, over the ground -- nothing sits under them.
    if (whole == said_as::snap_to_anchors) {
        combine = "Union";
        say("SnapToAnchors");
        return spoken();
    }

    // THE REST OF THE LINE IS WHAT THE READING FIRED, in the order it says it:
    // its acts, then the words its recolor is said with. A rule that fired and
    // has no word names the word the language lacks. What a rule draws off the
    // pieces -- rays, bars, filled hollows, trails, filled boxes, copies -- is
    // said after the line, each in a sentence of its own that opens with the cut
    // again: it reads the pieces as the cut hands them over, which is what its
    // replay read, and paints only what it draws, over what is down or under it,
    // so it neither reads what the line or another paint changed nor undoes it.
    vector<std::pair<beat, const char*>> afterwards;
    for (const int8_t which : read.line) {
        const auto& rule = read.fired[which];
        beat drawn;
        const char* paint = nullptr;
        if (!rule.no_word_for.empty()) said.no_word_for.push_back(rule.no_word_for);
        else if (drawn_after(rule, drawn, paint)) afterwards.push_back({ drawn, paint });
        else for (const beat& one : beats_of(rule, inf)) beats.push_back(one);
    }

    spoken();
    for (const auto& [drawn, paint] : afterwards) {
        for (size_t i = 0; i < opening.size(); ++i)
            say_sentence(said, "Input", opening[i].pick, opening[i].act, "Withhold", i == 0);
        said.sentences.push_back(closed_on_the_pick("CurrentObjects " + drawn.pick + " " + drawn.act, paint));
    }
    return true;
}

// ONE CHOICE AMONG EVERY EXPLANATION ON THE LIST. Each rule that stands alone is
// an explanation of the pairs, and so is the elected cut's own line. All of them
// are asked and priced alike -- complete first, meaning no cell left unexplained
// and no word missing, then the fewest ops -- and the cheapest is said; a tie
// goes to the one listed first. The list says what fired; nothing here keeps a
// roster of its own, so a rule that is not on the list is not asked.
// THE SECOND LIST KEEPS EVERY MISSING WORD. A rule that held and could not be
// said names what the language lacks, whichever explanation is said.
inline said_reading say_the_reading(const inference& inf) {
    vector<said_reading> spoke;
    vector<string> missing;
    auto heard = [&](bool it_spoke, said_reading& one) {
        if (it_spoke) spoke.push_back(std::move(one));
        else missing.insert(missing.end(), one.no_word_for.begin(), one.no_word_for.end());
    };
    const auto& election = inf.correspondence;
    if (!election.readings.empty())
        for (const auto& rule : election.readings[election.chosen].fired) {
            if (!rule.stands_alone) continue;
            said_reading one;
            heard(say_rival(rule, inf, one), one);
        }
    said_reading line;
    heard(say_the_cut(inf, line), line);
    // A SIZE CHANGE, READ THROUGH ITS FRAME: the framed reading's line, every
    // sentence of it that opens on the input opening on the framed input -- the
    // piece cut out and moved to the corner -- instead.
    if (inf.framed) {
        said_reading framed;
        heard(say_the_cut(*inf.framed, framed) && frame_the_sentences(inf.frame, framed), framed);
    }
    auto price = [](const said_reading& one) {
        int ops = 0;
        for (const string& sentence : one.sentences)
            each_program_word(sentence, [&](std::string_view, node_type) { ++ops; });
        const bool complete = one.no_word_for.empty() && one.left_unexplained == 0;
        return tuple{ !complete, one.left_unexplained, ops };
    };
    const said_reading* cheapest = nullptr;
    for (const auto& one : spoke)
        if (!cheapest || price(one) < price(*cheapest)) cheapest = &one;
    said_reading said = cheapest ? *cheapest : said_reading{};
    said.no_word_for.insert(said.no_word_for.begin(), missing.begin(), missing.end());
    return said;
}

}  // namespace arc
