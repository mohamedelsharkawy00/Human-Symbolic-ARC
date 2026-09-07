// 5_sentences.hpp — the reading, said in the solver's own words.
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
#include "inference/4_infer.hpp"
#include <string>
#include <vector>

namespace arc {
using namespace std;

struct said_reading {
    vector<string> sentences;
    vector<string> no_word_for;
    // OPENINGS THE READING COULD NOT RULE OUT. Only the first sentence can be
    // said more than one way -- everything after it hangs off the chosen
    // reading's own pieces, so a rival cut has no follow-on to go with it.
    vector<string> other_openings;
};

// WHAT AN OBJECT IS, as the word that cuts the same way. Split's argument is
// the any-color flag, so Split[0] keeps colors together and Split[1] does not.
inline const char* word_for_cut(split_kind cut) {
    switch (cut) {
        case split_kind::touching_8:         return "Split[1]";
        case split_kind::touching_any_color: return "Split[0]";
        case split_kind::one_per_color:      return "Colors";
        case split_kind::repeating_motif:    return "Pattern";
        case split_kind::repeating_step:     return "Pattern";
        default:                             return nullptr;
    }
}

// ONE OF THE EIGHT WAYS ROUND, said in turns and mirrors. The bits are
// posed_glyph's own: 1 swaps rows with columns, 2 flips top to bottom, 4 flips
// left to right. Every one of the eight is sayable, four of them in one word.
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

    if (const char* cut = word_for_cut(facts.cut_kind)) say(cut);
    else said.no_word_for.push_back(string("the cut ") + split_name((int)facts.cut_kind));

    // The cuts that explained just as much. Said as openings of their own so a
    // search can try each, rather than trusting the order they were declared in.
    for (int8_t cell : facts.tied_with_chosen) {
        if (cell < 0 || cell >= (int8_t)facts.readings.size()) continue;
        const char* cut = word_for_cut(facts.readings[cell].cut_kind);
        if (!cut) continue;
        string opening = string("Input SelectAll ") + cut + " " + combine;
        bool already = !said.sentences.empty() && said.sentences.front() == opening;
        for (const string& seen : said.other_openings) already = already || seen == opening;
        if (!already) said.other_openings.push_back(opening);
    }

    // A LINE IS A SELECTION AND WHAT HAPPENED TO IT. One piece can turn AND
    // move AND change colour, so a line carries several acts -- and the grammar
    // gives each act its own sentence, all sharing the line's GET and SELECT.
    // Only the last of them paints; the rest withhold, which is what that word
    // is for. A line must hold in EVERY pair, or it is one example's accident.
    struct line_key {
        int8_t color;      // the selection, as the only thing a SELECT can name
        int8_t pose;
        int8_t down, across;
        int8_t recolor_to;
        bool operator==(const line_key& other) const {
            return color == other.color && pose == other.pose && down == other.down &&
                   across == other.across && recolor_to == other.recolor_to;
        }
    };
    vector<line_key> lines_in_every_pair;
    for (size_t which = 0; which < facts.pairs.size(); ++which) {
        const auto& pair = facts.pairs[which];
        vector<line_key> here;
        for (const auto& edge : pair.edges) {
            if (!edge.from_input || !edge.same_shape) continue;
            if (edge.from < 0 || edge.from >= (int16_t)pair.input_objects.size()) continue;
            const auto& piece = pair.input_objects[edge.from];
            line_key key{piece.color, edge.pose,
                         (int8_t)(edge.row - piece.top), (int8_t)(edge.col - piece.left),
                         edge.exact ? (int8_t)-1 : edge.recolored_to};
            if (key.pose == 0 && key.down == 0 && key.across == 0 && key.recolor_to < 0)
                continue;  // nothing happened to it, so there is no line to say
            bool seen = false;
            for (const auto& already : here) seen = seen || (already == key);
            if (!seen) here.push_back(key);
        }
        if (which == 0) { lines_in_every_pair = here; continue; }
        vector<line_key> still;
        for (const auto& key : lines_in_every_pair)
            for (const auto& mine : here)
                if (key == mine) { still.push_back(key); break; }
        lines_in_every_pair = still;
    }

    for (const auto& key : lines_in_every_pair) {
        const string pick = key.color < 0 ? string("SelectAll")
                                          : "SelectColor[" + to_string((int)key.color) + "]";
        vector<string> acts;
        for (const string& word : words_for_pose(key.pose)) acts.push_back(word);
        if (key.across != 0) acts.push_back("MoveRow[" + to_string((int)key.across) + "]");
        if (key.down != 0)   acts.push_back("MoveColumn[" + to_string((int)key.down) + "]");
        if (key.recolor_to >= 0) acts.push_back("Recolor[" + to_string((int)key.recolor_to) + "]");
        for (size_t i = 0; i < acts.size(); ++i)
            said.sentences.push_back(
                string(said.sentences.empty() ? "Input" : "CurrentObjects") + " " + pick + " " +
                acts[i] + " " + (i + 1 < acts.size() ? "Withhold" : combine));
    }

    // Said last because nothing spells it yet: a recolor needs the map from the
    // old color to the new one, and the reading carries the pieces, not the map.
    // Colours that trade places. Said before the reading's own recolor, because
    // a swap is the whole story for the cells it touches and needs no pieces.
    for (int variant = 0; variant < 45; ++variant)
        if (inf.color_swaps.variant[variant])
            say("BidiColorMap[" + to_string(variant) + "]");

    if (allowed.recolor)
        for (const auto& pair : facts.pairs)
            if (pair.found_recolored > 0) {
                if (const char* word = word_for_learned_recolor(inf.shape_color_map))
                    say(word);
                else
                    said.no_word_for.push_back("a recolor");
                break;
            }

    return said;
}

}  // namespace arc
