// 6_sentences.hpp — the reading, said in the solver's own words.
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
#include "5_infer.hpp"
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
inline const char* word_for_segmentation(segmentation_method segmentation) {
    switch (segmentation) {
        case segmentation_method::touching_8:         return "Split[1]";
        case segmentation_method::touching_any_color: return "Split[0]";
        case segmentation_method::one_per_color:      return "Colors";
        case segmentation_method::repeating_motif:    return "Pattern";
        case segmentation_method::repeating_step:     return "Pattern";
        default:                             return nullptr;
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

// A PROGRAM READ BACK AS ITS SENTENCES, four beats to a row. What the pane draws
// and what the log writes, so neither has to know how a program is spelled.
inline vector<string> sentences_of(const string& program) {
    vector<string> beats, rows;
    for (size_t at = 0, next; at < program.size(); at = next) {
        next = program.find(' ', at);
        if (next == string::npos) next = program.size();
        if (next > at) beats.push_back(program.substr(at, next - at));
        if (next < program.size()) ++next;
    }
    for (size_t beat = 0; beat + 3 < beats.size(); beat += 4)
        rows.push_back(beats[beat] + " " + beats[beat + 1] + " " +
                       beats[beat + 2] + " " + beats[beat + 3]);
    return rows;
}

inline said_reading say_the_reading(const inference& inf) {
    const correspondence_facts& facts = inf.correspondence;
    said_reading said;
    if (!facts.valid || facts.pairs.empty()) return said;

    // A piece that sits on what is already there is laid under it, not over.
    const char* combine = "Union";
    for (const auto& pair : facts.pairs)
        if (pair.sits_on > 0) combine = "Underlay";

    // Nothing is implicit: the first sentence names its source, the rest carry
    // on from the objects already in hand.
    auto say = [&](const string& act) {
        said.sentences.push_back(string(said.sentences.empty() ? "Input" : "CurrentObjects")
                                 + " SelectAll " + act + " " + combine);
    };

    const match_mode allowed = facts.chosen < (int8_t)facts.readings.size()
                             ? facts.readings[facts.chosen].match_kind : match_mode{};

    if (const char* segmentation = word_for_segmentation(facts.segmentation)) say(segmentation);
    else said.no_word_for.push_back(string("the segmentation ") + segmentation_name((int)facts.segmentation));

    // The segmentation methods that explained just as much. Said as openings of their own so a
    // search can try each, rather than trusting the order they were declared in.
    for (int8_t cell : facts.tied_with_chosen) {
        if (cell < 0 || cell >= (int8_t)facts.readings.size()) continue;
        const char* segmentation = word_for_segmentation(facts.readings[cell].segmentation);
        if (!segmentation) continue;
        string opening = string("Input SelectAll ") + segmentation + " " + combine;
        bool already = !said.sentences.empty() && said.sentences.front() == opening;
        for (const string& seen : said.other_openings) already = already || seen == opening;
        if (!already) said.other_openings.push_back(opening);
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
            if (edge.from < 0 || edge.from >= (int16_t)pair.input_objects.size()) continue;
            const auto& piece = pair.input_objects[edge.from];
            note(piece.color, act_turn,   edge.pose);
            note(piece.color, act_down,   (int8_t)(edge.row - piece.top));
            note(piece.color, act_across, (int8_t)(edge.col - piece.left));
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
        const string pick = act.color < 0 ? string("SelectAll")
                                          : "SelectColor[" + to_string((int)act.color) + "]";
        vector<string> words;
        switch (act.what) {
            case act_turn:    words = words_for_pose(act.value); break;
            case act_down:    words = {"MoveColumn[" + to_string((int)act.value) + "]"}; break;
            case act_across:  words = {"MoveRow[" + to_string((int)act.value) + "]"}; break;
            case act_recolor: words = {"Recolor[" + to_string((int)act.value) + "]"}; break;
            default: break;
        }
        for (size_t i = 0; i < words.size(); ++i)
            said.sentences.push_back(
                string(said.sentences.empty() ? "Input" : "CurrentObjects") + " " + pick + " " +
                words[i] + " " + (i + 1 < words.size() ? "Withhold" : combine));
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

    return said;
}

}  // namespace arc
