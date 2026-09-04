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

inline said_reading say_the_reading(const correspondence_facts& facts) {
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

    // THE TURN, only when every kept edge in every pair turned the same way. A
    // turn one piece alone saw is a coincidence with nobody to back it, and a
    // program cannot turn one piece differently from the next anyway.
    int turn = -1;
    bool turns_differ = false;
    for (const auto& pair : facts.pairs)
        for (const auto& edge : pair.edges) {
            if (!edge.same_shape) continue;
            if (turn < 0) turn = edge.pose;
            else if (turn != edge.pose) turns_differ = true;
        }
    if (turns_differ) said.no_word_for.push_back("a turn that differs from piece to piece");
    else if (turn > 0)
        for (const string& word : words_for_pose(turn)) say(word);

    // THE MOVE, only when every pair said the same one. Each pair's loudest
    // agreement is the one the most pieces backed; pairs that disagree with
    // each other have said nothing a test pair could hear.
    bool have_move = allowed.translate && !facts.pairs.empty();
    bool first_pair = true;
    int down = 0, across = 0;
    for (const auto& pair : facts.pairs) {
        const correspondence_facts::pair_info::moved_together* loudest = nullptr;
        for (const auto& moved : pair.agreed_moves)
            if (!loudest || moved.pieces > loudest->pieces) loudest = &moved;
        if (!loudest) { have_move = false; break; }  // nobody agreed here
        if (first_pair) { down = loudest->down; across = loudest->across; first_pair = false; }
        else if (loudest->down != down || loudest->across != across) {
            have_move = false;
            said.no_word_for.push_back("a move the pairs did not agree on");
            break;
        }
    }
    if (have_move) {
        // One act to a sentence, so a corner-to-corner move is two of them.
        if (across != 0) say("MoveRow[" + to_string((int)across) + "]");
        if (down != 0)   say("MoveColumn[" + to_string((int)down) + "]");
    } else if (allowed.translate) {
        said.no_word_for.push_back("a move that differs from piece to piece");
    }

    // Said last because nothing spells it yet: a recolor needs the map from the
    // old color to the new one, and the reading carries the pieces, not the map.
    if (allowed.recolor)
        for (const auto& pair : facts.pairs)
            if (pair.found_recolored > 0) {
                said.no_word_for.push_back("a recolor");
                break;
            }

    return said;
}

}  // namespace arc
