// This file writes the per-puzzle readout.
// Design story in notes/1 - Overall Design/Inference Master.md
//
// Last in the family: it reads every fact and writes none.

#pragma once
#include "8_sentences.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace arc {
using namespace std;

// Dump EVERYTHING infer() read off the puzzle into logs/inference/<id>.txt
// (one file per puzzle, rewritten on each load, so a walk across puzzles
// keeps every reading). Called by puzzle_loader after infer(), so infer()
// stays pure. Skipped headless, same reason as 20_logger.hpp: Kaggle disk quota.
//
// ONE FUNCTION PER SECTION, in the order they land in the file, so the section
// a question is about is the only one that has to be read.
#ifdef _MSC_VER

// Which grid an edge starts on. A reading holds both, so this is per edge.
inline const char* edge_side(const correspondence_facts::edge& edge) {
    return edge.from_input ? "in->out" : "out->in";
}

// What kind of edge it is, in the order the claim weakens.
inline const char* edge_kind(const correspondence_facts::edge& edge) {
    return !edge.same_shape ? "  sits on" : edge.exact ? "  exact" : "  recolored";
}

// Idx/count/flags mirror the UI header: 1-based zero-padded index over the
// loaded count, and the dataset chips (TRN/EVL/TST from the DS_* flags).
inline void report_header(ostream& report, const string& id,
                          int idx, int count, int dataset_flags) {
    report << "puzzle " << id;
    if (idx >= 0 && count > 0) {
        string position = to_string(idx + 1), total = to_string(count);
        report << "  "
          << string(total.size() - min(total.size(), position.size()), '0')
          << position << '/' << total;
    }
    if (dataset_flags) {
        report << "  ";
        bool first = true;
        for (auto [flag, chip] : {pair{1, "TRN"}, {2, "EVL"}, {4, "TST"}, {8, "CUS"}}) {
            if (!(dataset_flags & flag)) continue;
            if (!first) report << '+';
            report << chip;
            first = false;
        }
    }
    report << "\n\n";
}

inline void report_dims(ostream& report, const output_dimension_rule& size) {
    report << "[dims]\n";
    report << "rule " << (int)size.Chosen << "\n\n";
}

inline void report_colors(ostream& report, const color_facts& colors) {
    report << "[colors]\n";
    report << "consumed " << colors_as_text(colors.consumed)
      << "  introduced " << colors_as_text(colors.introduced)
      << "  preserved " << colors_as_text(colors.preserved) << "\n";
    report << "every_input " << colors_as_text(colors.in_every_input)
      << "  any_input " << colors_as_text(colors.in_any_input)
      << "  every_output " << colors_as_text(colors.in_every_output)
      << "  any_output " << colors_as_text(colors.in_any_output) << "\n";
    report << "every_grid " << colors_as_text(colors.in_every_grid)
      << "  any_grid " << colors_as_text(colors.in_any_grid)
      << "  background " << (int)colors.background_color << "\n";
    report << "output_colors ";
    for (int i = 0; i < colors.output_colors_count; ++i)
        report << (int)colors.output_colors[i];
    report << "  by_size ";
    for (int i = 0; i < colors.output_colors_by_size_count; ++i)
        report << (int)colors.output_colors_by_size[i];
    report << "\n\n";
}

inline void report_background(ostream& report, const background_hypotheses& background) {
    report << "[background] input=" << (int)background.input_elected()
      << "  output=" << (int)background.output_elected()
      << (background.is_per_pair() ? "  MODE=PER-PAIR" : "") << "\n "
      << " across_inputs=" << (int)background_hypotheses::top(background.across_inputs)
      << " across_outputs=" << (int)background_hypotheses::top(background.across_outputs)
      << " across_inputs_outputs=" << (int)background_hypotheses::top(background.across_inputs_outputs)
      << "  across_input_output";
    for (int8_t color : background.across_input_output) report << ' ' << (int)color;
    report << "\n";
    auto dump = [&](const char* name, const vector<background_hypotheses::hypothesis>& ranked) {
        report << "  " << name << " (color embeds dom in_out SCORE span gaps under):\n";
        for (const auto& candidate : ranked)
            report << "    " << (int)candidate.color << "  " << candidate.embeds << "  "
              << candidate.dominance << "  " << candidate.appears_in_any_output
              << "  " << candidate.score << "  " << candidate.span
              << "  " << candidate.gaps << "  " << candidate.underneath << "\n";
    };
    dump("across_inputs", background.across_inputs);
    dump("across_outputs", background.across_outputs);
    dump("across_inputs_outputs", background.across_inputs_outputs);
    report << "\n";
}

inline void report_color_swaps(ostream& report, const color_swap_rule& swaps) {
    report << "[color_swaps] count " << swaps.count << "\n";
    int variant = 0;
    for (int first = 0; first < COLOR_COUNT; ++first)
        for (int second = first + 1; second < COLOR_COUNT; ++second, ++variant)
            if (swaps.variant[variant])
                report << "swap " << first << "<->" << second << "\n";
    report << "\n";
}

inline void report_shape_color_map(ostream& report, const shape_color_map_rule& rule) {
    static const char* claimed_name[] = { "nobody", "body_recolor", "interior_fill" };
    report << "[shape_color_map] valid " << rule.valid
      << " count " << rule.count
      << " claimed_by " << claimed_name[rule.claimed_by] << "\n";
    for (int i = 0; i < rule.count; ++i)
        report << "shape " << rule.keys[i]
          << " -> color " << (int)rule.colors[i] << "\n";
    report << "\n";
}

inline void report_unchanged(ostream& report, const unchanged_cells_facts& unchanged) {
    report << "[unchanged] valid " << unchanged.valid;
    if (unchanged.valid) {
        report << "  min_kept " << unchanged.min_kept_fraction << "  per_pair";
        for (float kept : unchanged.kept_fraction) report << ' ' << kept;
    }
    report << "\n\n";
}

inline void report_color_diff(ostream& report, const color_diff_facts& diff) {
    report << "[color_diff] valid " << diff.valid << "\n";
    if (diff.valid)
        for (int color = 0; color < COLOR_COUNT; ++color)
            if (diff.gained_count[color] || diff.lost_count[color])
                report << "color " << color
                  << "  gained " << diff.gained_count[color]
                  << "  lost " << diff.lost_count[color] << "\n";
    report << "\n";
}

// The election's own numbers, so a tie is readable as a tie. BOX CELLS is what
// the pieces' own boxes cover — two readings can tie on pieces and left over
// and still differ here.
inline void report_correspondence_readings(ostream& report, const correspondence_facts& correspondence) {
    report << "[correspondence] valid " << correspondence.valid
      << "  segmentation " << segmentation_names[(int)correspondence.segmentation] << "\n";
    // Which readings were actually in the election. Everything else below was
    // run for the table and never competed — see select_segmentation.
    report << "  ELECTED FROM the readings marked SEARCHED; the rest never"
              " competed and are shown only so the table can be read\n";
    for (size_t i = 0; i < correspondence.readings.size(); ++i) {
        const auto& reading = correspondence.readings[i];
        report << "  reading " << i
          << "  says it in " << (int)(reading.describe_cost * 100.0 + 0.5)
          << "% of plain  segmented into " << reading.objects_found;
        report << ", used " << reading.objects_used
          << " (" << reading.exact << " exact, " << reading.recolored << " recolored, "
          << reading.sits_on << " sits on), " << reading.made << " made, "
          << reading.gone << " gone, "
          << (reading.left_over - reading.made - reading.gone) << " unexplained, " << reading.left_over
          << " left over, boxes cover " << reading.box_cells << " cells";
        if (!reading.objects_can_fit) report << "  (no object fits the other grid)";
        report << "   " << segmentation_names[(int)reading.segmentation]
          << "  [" << reading.allowed.label() << "]";
        report << ((int)i == correspondence.chosen ? "   ELECTED\n" : "\n");
    }
}

// Which way the pieces' cells piled up inside their own boxes, if they did.
inline void report_piece_slam(ostream& report, const piece_slam_rule& slam) {
    report << "[piece_slam] valid " << slam.valid;
    if (slam.valid) report << "  " << (slam.rows ? "rows" : "columns") << " step " << (int)slam.step;
    report << "\n\n";
}

// WHAT THE READING COMES TO, in the solver's own words. The second list is the
// one to read: it names the word the language is missing, at the moment the
// looking wanted it.
inline void report_said_reading(ostream& report, const inference& inf) {
    // Each block as the poses that draw it, '-' for background.
    const auto& blocks = inf.input_blocks;
    report << "[input_blocks] valid " << blocks.valid << "\n";
    for (size_t which = 0; which < blocks.pairs.size(); ++which) {
        const auto& at = blocks.pairs[which];
        report << "  pair " << which << "  " << (int)at.rows << "x" << (int)at.cols << " ";
        for (int r = 0; r < at.rows; ++r) {
            report << (r ? " /" : "");
            for (int c = 0; c < at.cols; ++c) {
                report << " ";
                if (!at.poses[r][c]) report << "-";
                for (int pose = 0; pose < POSE_COUNT; ++pose)
                    if (at.poses[r][c] >> pose & 1) report << pose;
            }
        }
        report << "\n";
    }
    report << "\n";
    const said_reading said = say_the_reading(inf);
    const vector<string> programs = programs_from(said);
    report << "[sentences] " << programs.size() << " program"
           << (programs.size() == 1 ? "" : "s") << ", "
           << said.no_word_for.size() << " with no word\n";
    for (const string& program : programs) {
        for (const string& row : sentences_of(program)) report << "  " << row << "\n";
        report << "\n";
    }
    for (const string& missing : said.no_word_for) report << "  NO WORD FOR " << missing << "\n";
}

// WHAT SEVERAL PIECES SAID AT ONCE, reading by reading and pair by pair. Read
// in each pair's own way up, so a pair drawn a quarter turn from its
// neighbours can be set beside them. Readings nobody agreed in are not listed:
// a page of empty lines says nothing that their absence does not.
inline void report_agreed_moves(ostream& report, const correspondence_facts& correspondence) {
    for (size_t i = 0; i < correspondence.reading_pairs.size(); ++i) {
        const auto& pairs = correspondence.reading_pairs[i];
        if (ranges::all_of(pairs, [](const auto& pair) {
                return pair.agreed_moves.empty(); })) continue;
        report << "  reading " << i << " moved together   "
          << segmentation_names[(int)correspondence.readings[i].segmentation]
          << "  [" << correspondence.readings[i].allowed.label() << "]\n";
        for (size_t which = 0; which < pairs.size(); ++which) {
            report << "    pair " << which << " input-to-output";
            for (const auto& agreed : pairs[which].agreed_moves) {
                report << "   down " << (int)agreed.down << " across " << (int)agreed.across
                  << " said by " << agreed.objects;
                if (agreed.color >= 0) report << " all color " << (int)agreed.color;
                else                   report << " mixed colors";
            }
            report << "\n";
        }
    }
}

// WHAT EACH READING ACTUALLY CUT AND WHERE EACH PIECE WENT, every reading, both
// sides. The numbers above can agree while the segmentation methods disagree, and only the
// pieces say how; and a reading that leaves a cell over is only readable next
// to the edges it did draw. Read straight off the readings -- it used to be
// handed a screen-shaped copy of them, and could only say what that copy kept.
inline void report_pieces_seen(ostream& report, const correspondence_facts& correspondence) {
    const char* newline = "\n";
    for (size_t i = 0; i < correspondence.reading_pairs.size(); ++i) {
        report << "  reading " << i << " pieces   "
          << segmentation_names[(int)correspondence.readings[i].segmentation]
          << "  [" << correspondence.readings[i].allowed.label() << "]" << newline;
        const auto& pairs = correspondence.reading_pairs[i];
        for (size_t which = 0; which < pairs.size(); ++which)
            for (int side = 0; side < 2; ++side) {
                const auto& list = side ? pairs[which].output_info
                                        : pairs[which].input_info;
                for (const auto& piece : list)
                    report << "    pair " << which << (side ? " out" : " in ")
                      << "  color " << (int)piece.color
                      << " at " << (int)piece.summary.x0 << "," << (int)piece.summary.y0
                      << " " << (int)piece.summary.w << "x" << (int)piece.summary.h
                      << " px " << piece.summary.count
                      << " instance_of " << piece.instance_of << " edges " << piece.edges_touching << newline;
            }
        for (size_t which = 0; which < pairs.size(); ++which)
            for (const auto& edge : pairs[which].edges) {
                const auto& mine = edge.from_input ? pairs[which].input_info
                                                   : pairs[which].output_info;
                const auto& theirs = edge.from_input ? pairs[which].output_info
                                                     : pairs[which].input_info;
                const auto& piece = mine[edge.from];
                const int landed_w = edge.to >= 0 ? theirs[edge.to].summary.w : piece.summary.w;
                const int landed_h = edge.to >= 0 ? theirs[edge.to].summary.h : piece.summary.h;
                report << "    pair " << which
                  << "  " << (int)piece.summary.w << "x" << (int)piece.summary.h
                  << " at " << (int)piece.summary.x0 << "," << (int)piece.summary.y0
                  << " -> " << landed_w << "x" << landed_h
                  << " at " << (int)edge.col << "," << (int)edge.row;
                if (edge.hidden) report << ", " << (int)edge.hidden << " hidden";
                if (edge.outside) report << ", " << (int)edge.outside << " off grid";
                report << newline;
            }
    }
}

// One pair's pieces on both sides, then every edge, then what was found and
// thrown away — a missing line is only readable next to the reason it is
// missing.
inline void report_one_pair(ostream& report, size_t pair_index,
                            const correspondence_facts::pair_info& pair) {
    report << "pair " << pair_index
      << "  in_objects " << pair.input_info.size()
      << "  out_objects " << pair.output_info.size()
      << "  class " << pair.most_sources << '-' << pair.most_landings
      << "  went nowhere " << pair.deleted << " one " << pair.persisted
      << " many " << pair.copied
      << "  came from nothing " << pair.invented << " one " << pair.from_one
      << " many " << pair.from_many
      << "  exact " << pair.found_exact << " recolored " << pair.found_recolored
      << " sits_on " << pair.sits_on << " hidden " << pair.partly_hidden
      << " turned " << pair.turned << " dropped " << pair.dropped
      << "  made " << ranges::count_if(pair.left_over, [](const auto& c) { return c.made_or_gone && !c.from_input; })
      << "  gone " << ranges::count_if(pair.left_over, [](const auto& c) { return c.made_or_gone && c.from_input; })
      << "  unexplained " << ranges::count_if(pair.left_over, [](const auto& c) { return !c.made_or_gone; })
      << "  carried_cells " << pair.exact_covered_cells
      << "  created_cells " << pair.created_cells << "\n";
    for (size_t i = 0; i < pair.input_info.size(); ++i) {
        const auto& piece = pair.input_info[i];
        report << "  in[" << i << "] color " << (int)piece.color
          << " at " << (int)piece.summary.x0 << ',' << (int)piece.summary.y0
          << " " << (int)piece.summary.w << 'x' << (int)piece.summary.h
          << " px " << piece.summary.count << " shape " << piece.shape
          << " instance_of " << piece.instance_of << " edges " << piece.edges_touching << "\n";
    }
    for (size_t i = 0; i < pair.output_info.size(); ++i) {
        const auto& landing = pair.output_info[i];
        report << "  out[" << i << "] color " << (int)landing.color
          << " at " << (int)landing.summary.x0 << ',' << (int)landing.summary.y0
          << " " << (int)landing.summary.w << 'x' << (int)landing.summary.h
          << " px " << landing.summary.count << " shape " << landing.shape
          << " instance_of " << landing.instance_of << " edges " << landing.edges_touching << "\n";
    }
    static const char* how_name[] = { "PIECES FIT", "no fit, greedy", "GAVE UP" };
    report << "  edges " << pair.edges.size() << ", "
      << how_name[pair.verdict] << "\n";
    for (const auto& edge : pair.edges) {
        report << "  " << edge_side(edge) << " [" << (int)edge.from << "] -> ";
        if (edge.to >= 0) report << '[' << (int)edge.to << ']';
        else              report << "place";
        report << " at " << (int)edge.col << ',' << (int)edge.row << edge_kind(edge);
        if (edge.hidden) report << ", " << (int)edge.hidden << " hidden";
        if (edge.recolored_to >= 0) report << " -> " << (int)edge.recolored_to;
        // Which way round it was sitting: turned, and/or mirrored
        if (edge.pose) {
            report << "  ";
            if (edge.pose & 1) report << "turned ";
            if (edge.pose & 2) report << "flipped_rows ";
            if (edge.pose & 4) report << "flipped_cols";
        }
        report << "\n";
    }

    for (size_t i = 0; i < pair.discarded.size(); ++i) {
        const auto& edge = pair.discarded[i];
        report << "  dropped " << edge_side(edge) << " [" << (int)edge.from
          << "] at " << (int)edge.col << ','
          << (int)edge.row << edge_kind(edge)
          << "  " << correspondence_facts::refusal_names[pair.discard_why[i]] << "\n";
    }
}

#endif

inline void log_inference(const string& id, const inference& inf,
                          int idx = -1, int count = 0, int dataset_flags = 0) {
#ifdef _MSC_VER
    static once_flag dir_flag;
    call_once(dir_flag, [] { filesystem::create_directories("logs/inference"); });
    ofstream report("logs/inference/" + (id.empty() ? "unknown" : id) + ".txt", ios::trunc);
    if (!report.is_open()) return;

    report_header(report, id, idx, count, dataset_flags);
    report_dims(report, inf.size);
    report_colors(report, inf.colors);
    report_background(report, inf.background);
    report_color_swaps(report, inf.color_swaps);
    report_shape_color_map(report, inf.shape_color_map);
    report_unchanged(report, inf.unchanged);
    report_color_diff(report, inf.color_diff);
    report_correspondence_readings(report, inf.correspondence);
    report_piece_slam(report, inf.piece_slam);
    report_said_reading(report, inf);
    report_agreed_moves(report, inf.correspondence);
    report_pieces_seen(report, inf.correspondence);
    for (size_t i = 0; i < inf.correspondence.pairs.size(); ++i)
        report_one_pair(report, i, inf.correspondence.pairs[i]);
#endif
}

}  // namespace arc
