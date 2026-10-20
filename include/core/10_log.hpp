// This file writes the per-puzzle readout.
// Design story in notes/1 - Overall Design/Inference Master.md
//
// Last in the family: it reads every fact and writes none.

#pragma once
#include "9_sentences.hpp"
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
// stays pure. Only when asked: a run with no one to read it (Kaggle) says
// no, because 240 files would blow its disk quota.
//
// ONE FUNCTION PER SECTION, in the order they land in the file, so the section
// a question is about is the only one that has to be read.

// Which grid an edge starts on. A reading holds both, so this is per edge.
inline const char* edge_side(const correspondence_election::edge& edge) {
    return edge.from_input ? "in->out" : "out->in";
}

// What kind of edge it is, in the order the claim weakens.
inline const char* edge_kind(const correspondence_election::edge& edge) {
    return !edge.same_shape ? "  sits on" : edge.exact ? "  exact" : "  recolored";
}

// A reading as the readout names it: how it cut, and what a match may change.
inline string reading_title(const correspondence_election::reading& reading) {
    return string(segmentation_names[(int)reading.segmentation]) + "  [" + reading.allowed.label() + "]";
}

// A box as the readout says it: width x height at column,row.
inline string box_said(int width, int height, int col, int row) {
    return to_string(width) + "x" + to_string(height) + " at " + to_string(col) + "," + to_string(row);
}

// One piece on one line. `with_shape` adds its shape hash.
inline void report_piece(ostream& report, const correspondence_election::object_info& piece,
                         bool with_shape) {
    report << "color " << (int)piece.color
      << " at " << (int)piece.summary.col << ',' << (int)piece.summary.row
      << " " << (int)piece.summary.width << 'x' << (int)piece.summary.height
      << " px " << piece.summary.count;
    if (with_shape) report << " shape " << piece.shape;
    report << " instance_of " << piece.instance_of << " edges " << piece.edges_touching << "\n";
}

// The colors a map changes, each as " was->now". False when it changes none.
inline bool report_recoloring(ostream& report, const recoloring& map) {
    bool changed = false;
    auto said = [](int value) { return is_color(value) ? to_string(value) : string("empty"); };
    for (int was = TRANSPARENT; was < COLOR_COUNT; ++was)
        if (map.changes(was)) {
            report << " " << said(was) << "->" << said(map[was]);
            changed = true;
        }
    return changed;
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

inline void report_colors(ostream& report, const color_facts& colors, int8_t input_background) {
    report << "[colors]\n";
    report << "consumed " << color_facts::as_text(colors.consumed)
      << "  introduced " << color_facts::as_text(colors.introduced)
      << "  preserved " << color_facts::as_text(colors.preserved) << "\n";
    report << "every_input " << color_facts::as_text(colors.in_every_input)
      << "  any_input " << color_facts::as_text(colors.in_any_input)
      << "  every_output " << color_facts::as_text(colors.in_every_output)
      << "  any_output " << color_facts::as_text(colors.in_any_output) << "\n";
    report << "every_grid " << color_facts::as_text(colors.in_every_grid)
      << "  any_grid " << color_facts::as_text(colors.in_any_grid)
      << "  background " << (int)input_background << "\n";
    report << "output_colors ";
    for (int i = 0; i < colors.output_colors_count; ++i)
        report << (int)colors.output_colors[i];
    report << "  by_size ";
    for (int i = 0; i < colors.output_colors_by_size_count; ++i)
        report << (int)colors.output_colors_by_size[i];
    report << "\n\n";
}

inline void report_background(ostream& report, const background_election& background) {
    report << "[background] input=" << (int)background.input_elected()
      << "  output=" << (int)background.output_elected()
      << (background.is_per_pair() ? "  MODE=PER-PAIR" : "") << "\n "
      << " across_inputs=" << (int)background_election::top(background.across_inputs)
      << " across_outputs=" << (int)background_election::top(background.across_outputs)
      << " across_inputs_outputs=" << (int)background_election::top(background.across_inputs_outputs)
      << "  across_input_output";
    for (int8_t color : background.across_input_output) report << ' ' << (int)color;
    report << "\n";
    auto dump = [&](const char* name, const vector<background_election::hypothesis>& ranked) {
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

inline void report_color_swaps(ostream& report, const color_swap_candidates& swaps) {
    report << "[color_swaps] count " << swaps.count << "\n";
    for (int variant = 0; variant < COLOR_PAIR_COUNT; ++variant) {
        if (!swaps.variant[variant]) continue;
        const auto [first, second] = color_pair::from_index(variant);
        report << "candidate " << first << "<->" << second << "\n";
    }
    report << "\n";
}

inline void report_shape_color_map(ostream& report, const shape_color_map_rule& rule) {
    static const char* claimed_name[] = { "nobody", "body_recolor", "interior_fill" };
    report << "[shape_color_map] holds " << rule.holds()
      << " count " << rule.count
      << " claimed_by " << claimed_name[rule.claimed_by] << "\n";
    for (int i = 0; i < rule.count; ++i)
        report << "key " << rule.keys[i]
          << " -> color " << (int)rule.colors[i] << "\n";
    if (rule.otherwise >= 0) report << "any other key -> color " << (int)rule.otherwise << "\n";
    // Read off the elected reading: which of its pieces keyed which, as the
    // pairs below number them.
    for (size_t which = 0; which < rule.read.size(); ++which) {
        const auto& read = rule.read[which];
        report << "  pair " << which << "  key in[" << read.key << "]  target in["
          << read.target << "] became " << (int)read.became << "\n";
    }
    report << "\n";
}

inline void report_unchanged(ostream& report, const unchanged_cells_facts& unchanged) {
    report << "[unchanged] computed " << unchanged.computed;
    if (unchanged.computed) {
        report << "  min_kept " << unchanged.min_kept_fraction << "  per_pair";
        for (float kept : unchanged.kept_fraction) report << ' ' << kept;
    }
    report << "\n\n";
}

inline void report_color_diff(ostream& report, const color_diff_facts& diff) {
    report << "[color_diff] computed " << diff.computed << "\n";
    if (diff.computed)
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
inline void report_correspondence_readings(ostream& report, const correspondence_election& correspondence) {
    report << "[correspondence] segmentation " << segmentation_names[(int)correspondence.segmentation] << "\n";
    // Which readings were actually in the election: each way of seeing objects'
    // best, as [fired] lists them. The rest never competed -- see competes.
    report << "  ELECTED FROM each way of seeing objects' best reading, the ones [fired] lists;"
              " the block segmentations never compete, their readings are evidence for rules\n";
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
        report << "   " << reading_title(reading);
        report << ((int)i == correspondence.chosen ? "   ELECTED\n" : "\n");
    }
    // The cuts the words weighed, in the matching's order: the one said is
    // starred, beside the cells each one's words left unsaid (- never read).
    report << "  [cuts]";
    for (size_t rank = 0; rank < correspondence.ranked.size(); ++rank) {
        const int cut = correspondence.ranked[rank];
        const int left = rank < correspondence.left_by_rank.size() ? correspondence.left_by_rank[rank] : -1;
        report << "  |" << (cut == correspondence.chosen ? "*" : " ") << "reading " << (int)cut << " "
               << segmentation_names[(int)correspondence.readings[cut].segmentation] << " ";
        if (left < 0) report << "-";
        else report << left << " unsaid";
    }
    report << "\n";
}

// WHAT EACH WAY OF SEEING OBJECTS WAS TAKEN TO SAY, a rule a line, a rule that
// reads another's evidence a step under it, and what all of them leave over.
// Only the segmentations that compete: the block claims show up as rules.
inline void report_fired(ostream& report, const correspondence_election& correspondence,
                         const string& title = "[fired] each segmentation's own reading, rule by rule") {
    report << "\n" << title << "\n";
    for (size_t method = 0; method < correspondence.best_of.size(); ++method) {
        const int16_t kept = correspondence.best_of[method];
        if (kept < 0 || !correspondence_election::competes((segmentation_methods)method)) continue;
        const auto& reading = correspondence.readings[kept];
        report << "  reading " << kept << "  " << segmentation_names[(int)reading.segmentation]
          << (kept == correspondence.chosen ? "   ELECTED" : "") << "\n";
        for (const auto& [which, depth] : reading.fired_in_order()) {
            const auto& rule = reading.fired[which];
            report << "    " << string((size_t)depth * 2, ' ') << rule.name;
            if (rule.stands_alone) report << "  by itself";
            if (rule.pieces || rule.cells)
                report << "  " << rule.pieces << " pieces, " << rule.cells << " cells";
            if (!rule.says.empty()) report << "  " << rule.says;
            if (!rule.no_word_for.empty()) report << "  no word for " << rule.no_word_for;
            report << "\n";
        }
        report << "    left over  " << reading.unclaimed() << " of " << correspondence.cells << " cells\n";
        if (reading.drawn_wrong)
            report << "    drawn wrong  " << reading.drawn_wrong << " of those, laid wrong by a word and never set right\n";
        // The order the cut's own line says them in, which is the program.
        if (!reading.line.empty()) {
            report << "    said as";
            for (const int8_t which : reading.line) report << "  " << reading.fired[which].name;
            report << "\n";
        }
    }
    report << "\n";
}

// THE PASSES AFTER THE LINE (read_the_rest), each read on the picture the passes
// before it drew: its own reading, rule by rule, the last one's what is left --
// the last one read saying no more, where it names only the words the rest lacks.
inline void report_the_rest(ostream& report, const inference& inf) {
    for (const inference* at = &inf; at->rest; at = at->rest.get()) {
        report_fired(report, at->rest->correspondence,
                     "[rest " + to_string((int)at->rest->pass) + "] the picture so far read again, rule by rule" +
                         (at->rest_says_more ? "" : ", saying no more than the pass before it"));
        if (!at->rest_says_more) break;
    }
}

// Which way the pieces' cells piled up inside their own boxes, if they did.
inline void report_piece_slam(ostream& report, const piece_slam_rule& slam) {
    report << "[piece_slam] holds " << slam.holds();
    if (slam.holds()) report << "  " << (slam.rows ? "rows" : "columns") << " step " << (int)slam.step;
    report << "\n\n";
}

// Which way the diagonals lean, where the colors cycle along them.
inline void report_diagonal_continuation(ostream& report, const diagonal_continuation_rule& diagonals) {
    report << "[diagonal_continuation] holds " << diagonals.holds();
    if (diagonals.holds()) report << "  lean " << (int)diagonals.lean;
    report << "\n\n";
}

// Whether the vote over the input's repeats draws every output.
inline void report_periodic_denoise(ostream& report, const periodic_denoise_rule& denoise) {
    report << "[periodic_denoise] holds " << denoise.holds() << "\n\n";
}

// Whether the pieces were laid out as one corner chain, each hung off the last.
inline void report_overlap_corner(ostream& report, const overlap_corner_rule& chain) {
    report << "[overlap_corner] holds " << chain.holds() << "\n\n";
}

// Which colour stayed put and which slid up against it.
inline void report_slide_to_touch(ostream& report, const slide_to_touch_rule& slide) {
    report << "[slide_to_touch] holds " << slide.holds();
    if (slide.holds())
        report << "  " << (int)slide.stays << " stays, " << (int)slide.slides << " slides";
    report << "\n\n";
}

// Whether the template was stamped along every marker that pointed a way.
inline void report_project_on_seeds(ostream& report, const project_on_seeds_rule& projection) {
    report << "[project_on_seeds] holds " << projection.holds();
    if (projection.holds() && is_color(projection.worn))
        report << "  every stamp wears " << (int)projection.worn << ", the input over them";
    report << "\n\n";
}

// Whether the gaps between pieces of one colour were filled, and under which cut.
inline void report_bridge(ostream& report, const bridge_rule& bridge) {
    report << "[bridge] holds " << bridge.holds();
    if (bridge.holds()) report << "  " << segmentation_names[(int)bridge.segmentation] << "  way " << (int)bridge.way;
    report << "\n\n";
}

// Whether the marks were laid as lines that take turns, under which cut, and
// whether a pair lying on its side had to be read turned.
inline void report_mark_lattice(ostream& report, const mark_lattice_rule& lattice) {
    report << "[mark_lattice] holds " << lattice.holds();
    if (lattice.holds()) {
        report << "  " << segmentation_names[(int)lattice.segmentation];
        if (lattice.portrait) report << "  read turned when lying on its side";
    }
    report << "\n\n";
}

// The frame the figures sit in: whether the input is really several grids side
// by side. One line, because the fact is one claim; the panes follow so a wrong
// cut is visible at a glance rather than inferred from what fails downstream.
inline void report_pane_split(ostream& report, const pane_split_facts& split) {
    static constexpr const char* const cut_names[] = { "wall", "blank", "bare" };
    static constexpr const char* const axis_names[] = { "across", "down", "both ways" };
    report << "[pane_split] holds " << split.holds();
    if (split.holds())
        report << "  cut " << cut_names[split.cut]
               << "  axis " << axis_names[split.axis]
               << "  color " << (split.cut == pane_split_facts::Wall && split.color < 0
                                     ? string("each grid's own") : to_string(split.color))
               << "  panes " << (int)split.down << "x" << (int)split.across;
    report << "\n";
    for (size_t which = 0; which < split.pairs.size(); ++which) {
        report << "  pair " << which << " ";
        for (const auto& one_pane : split.pairs[which])
            report << "  " << box_said(one_pane.width, one_pane.height, one_pane.left, one_pane.top);
        report << "\n";
    }
    report << "\n";
}

// What the panes come to: one line per pattern of filled panes, so the idea
// behind the rule is readable rather than inferred from sixteen color rows.
inline void report_pane_mask(ostream& report, const pane_mask_rule& mask) {
    report << "[pane_mask] holds " << mask.holds();
    if (mask.holds())
        report << "  panes " << (int)mask.panes << "  pose " << (int)mask.posed;
    report << "\n";
    for (size_t pattern = 0; pattern < mask.by_mask.size(); ++pattern) {
        const auto& says = mask.by_mask[pattern];
        if (!says.said()) continue;
        report << "  ";
        for (int p = 0; p < mask.panes; ++p) report << (pattern >> p & 1 ? '#' : '.');
        if (says.leaves_empty) report << "  empty";
        else if (says.paint >= 0) report << "  paint " << (int)says.paint;
        else report << "  copy pane " << (int)says.copy_pane;
        report << "\n";
    }
    report << "\n";
}

// Which pane the answer is, and the reason given for choosing it.
inline void report_pane_pick(ostream& report, const pane_pick_rule& pick) {
    static constexpr const char* const way_names[] = {
        "the odd one out", "the odd palette", "the one they repeat",
        "the one that breaks the fold", "the only one that folds",
        "the fullest", "the emptiest", "the most colored", "the least colored",
        "the one in that place" };
    report << "[pane_pick] holds " << pick.holds();
    if (pick.holds()) {
        report << "  " << way_names[pick.way];
        if (pick.fitted) report << "  fitted behind the walls";
        if (pick.way == pane_pick_rule::NotSymmetric ||
            pick.way == pane_pick_rule::OnlySymmetric) report << "  fold " << (int)pick.which;
        if (pick.way == pane_pick_rule::AtIndex) report << "  place " << (int)pick.which;
    }
    report << "\n\n";
}

// The output as copies of the input: each block as the poses that draw it, '-'
// for background.
inline void report_input_blocks(ostream& report, const input_blocks_rule& blocks) {
    report << "[input_blocks] holds " << blocks.holds();
    // How the copies' colors were read: a table every pair shares, or the
    // negative, whose ink each pair's own input supplies.
    if (blocks.negative) report << "  colors negative";
    else if (blocks.holds()) {
        report << "  colors";
        if (!report_recoloring(report, blocks.color_map)) report << " as given";
    }
    report << "\n";
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
}

inline void report_ray_extension(ostream& report, const ray_extension_rule& rule) {
    report << "[ray_extension] holds " << rule.holds() << "\n";
    for (const auto& branch : rule.branches) {
        report << "  color rank " << (int)branch.rank << "  "
               << (branch.ray_kind == ray_extension_rule::Orthogonal ? "orthogonal" : "diagonal")
               << " rays " << (int)branch.radius;
        if (branch.fill_interior) report << "  fill interior";
        if (branch.stamp_body) report << "  stamp body";
        report << "\n";
    }
    report << "\n";
}

// WHAT THE READING COMES TO, in the solver's own words. The second list is the
// one to read: it names the word the language is missing, at the moment the
// looking wanted it.
inline void report_said_reading(ostream& report, const inference& inf) {
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
inline void report_agreed_moves(ostream& report, const correspondence_election& correspondence) {
    for (size_t i = 0; i < correspondence.readings.size(); ++i) {
        const auto& pairs = correspondence.readings[i].pairs;
        if (ranges::all_of(pairs, [](const auto& pair) {
                return pair.agreed_moves.empty(); })) continue;
        report << "  reading " << i << " moved together   "
          << reading_title(correspondence.readings[i]) << "\n";
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
inline void report_pieces_seen(ostream& report, const correspondence_election& correspondence) {
    for (size_t i = 0; i < correspondence.readings.size(); ++i) {
        report << "  reading " << i << " pieces   " << reading_title(correspondence.readings[i]) << "\n";
        const auto& pairs = correspondence.readings[i].pairs;
        for (size_t which = 0; which < pairs.size(); ++which)
            for (const bool input : { true, false })
                for (const auto& piece : pairs[which].objects_on(input)) {
                    report << "    pair " << which << (input ? " in " : " out") << "  ";
                    report_piece(report, piece, false);
                }
        for (size_t which = 0; which < pairs.size(); ++which)
            for (const auto& edge : pairs[which].edges) {
                const auto& piece = pairs[which].source_of(edge)->summary;
                const auto* target = pairs[which].target_of(edge);
                const auto& landed = target ? target->summary : piece;
                report << "    pair " << which << "  " << box_said(piece.width, piece.height, piece.col, piece.row)
                  << " -> " << box_said(landed.width, landed.height, edge.col, edge.row);
                if (edge.hidden) report << ", " << (int)edge.hidden << " hidden";
                if (edge.outside) report << ", " << (int)edge.outside << " off grid";
                report << "\n";
            }
    }
}

// One pair's pieces on both sides, then every edge, then what was found and
// thrown away — a missing line is only readable next to the reason it is
// missing.
inline void report_one_pair(ostream& report, size_t pair_index,
                            const correspondence_election::pair_info& pair) {
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
      << "  made " << pair.made() << "  gone " << pair.gone()
      << "  unexplained " << pair.unmatched() - pair.made() - pair.gone()
      << "  carried_cells " << pair.exact_covered_cells
      << "  created_cells " << pair.created_cells << "\n";
    for (const bool input : { true, false }) {
        const auto& pieces = pair.objects_on(input);
        for (size_t i = 0; i < pieces.size(); ++i) {
            report << (input ? "  in[" : "  out[") << i << "] ";
            report_piece(report, pieces[i], true);
        }
    }
    static const char* how_name[] = { "PIECES FIT", "no fit, greedy", "GAVE UP" };
    report << "  edges " << pair.edges.size() << ", "
      << how_name[(int)pair.verdict] << "\n";
    for (const auto& edge : pair.edges) {
        report << "  " << edge_side(edge) << " [" << (int)edge.from << "] -> ";
        if (edge.to >= 0) report << '[' << (int)edge.to << ']';
        else              report << "place";
        report << " at " << (int)edge.col << ',' << (int)edge.row << edge_kind(edge);
        if (edge.hidden) report << ", " << (int)edge.hidden << " hidden";
        const auto* source = pair.source_of(edge);
        const int8_t recolored = edge.recolored_to(source ? source->color : -1);
        if (recolored >= 0) report << " -> " << (int)recolored;
        // A piece of several colors has no one color it went to, so its map is said whole.
        else if (!edge.exact) report_recoloring(report, edge.color_map);
        // Which way round it was sitting: turned, and/or mirrored
        if (edge.pose) {
            report << "  ";
            if (edge.pose & 1) report << "turned ";
            if (edge.pose & 2) report << "flipped_rows ";
            if (edge.pose & 4) report << "flipped_cols";
        }
        if (edge.scale != 1) report << "  scaled x" << (int)edge.scale;
        report << "\n";
    }

    for (const auto& [edge, reason] : pair.discarded) {
        report << "  dropped " << edge_side(edge) << " [" << (int)edge.from
          << "] at " << (int)edge.col << ','
          << (int)edge.row << edge_kind(edge)
          << "  " << correspondence_election::refusal_names[reason] << "\n";
    }
}

inline void log_inference(const string& id, const inference& inf,
                          int idx = -1, int count = 0, int dataset_flags = 0) {
    static once_flag dir_flag;
    call_once(dir_flag, [] { filesystem::create_directories("logs/inference"); });
    ofstream report("logs/inference/" + (id.empty() ? "unknown" : id) + ".txt", ios::trunc);
    if (!report.is_open()) return;

    report_header(report, id, idx, count, dataset_flags);
    report_dims(report, inf.size);
    report_colors(report, inf.colors, TRANSPARENT);
    // The reading read figures with no ground; its plan came from the election
    // it keeps beside it, and the plan it chose is starred.
    report_background(report, inf.ground_election);
    report << "[ground plans]";
    auto side = [](const background_source& ground) {
        return ground.per_grid ? string("each input's own")
             : is_color(ground.color) ? to_string((int)ground.color) : string("none");
    };
    // A plan as words: what the inputs lose -- the ground deleted, and what the
    // fill takes off them as well -- and what is filled last.
    auto plan_said = [&](const ground_plan& plan) {
        const background_source also = plan.filled_on_input();
        const string deleted = !also.removes_color() || also == plan.deleted ? side(plan.deleted)
                             : !plan.deleted.removes_color() ? side(also)
                             : side(plan.deleted) + " and " + side(also);
        return "delete " + deleted + ", fill " + side(plan.filled);
    };
    const vector<ground_plan> plans = ground_plans(inf.ground_election);
    for (const ground_plan& plan : plans) report << "  |" << (plan == inf.plan ? "*" : " ") << plan_said(plan);
    // Kept with words that may lay cells wrong (inference::tolerant), and with
    // the partner picks (inference::partners).
    if (inf.tolerant) report << "  | words lay cells wrong";
    if (inf.partners) report << "  | with the partner picks";
    report << "\n";
    // Each plan read, in order, until one said every cell with no pass after it.
    report << "[plans read] in order, the first to say the most kept\n";
    for (const auto& read : inf.plans_read)
        report << "  " << ((size_t)read.plan < plans.size() ? plan_said(plans[(size_t)read.plan]) : string("?"))
               << (read.tolerant ? ", words may lay cells wrong" : "")
               << (read.partners ? ", with the partner picks" : "") << ": " << read.left << " unsaid, "
               << read.passes << (read.passes == 1 ? " pass" : " passes")
               << (read.words.empty() ? string() : ": " + read.words) << "\n";
    report << "\n";
    report_pane_split(report, inf.pane_split);
    report_pane_mask(report, inf.pane_mask);
    report_pane_pick(report, inf.pane_pick);
    report << "[pane_gather] holds " << inf.pane_gather.holds() << "\n\n";
    report_color_swaps(report, inf.color_swaps);
    report_shape_color_map(report, inf.shape_color_map);
    report_unchanged(report, inf.unchanged);
    report_color_diff(report, inf.color_diff);
    report_correspondence_readings(report, inf.correspondence);
    report_fired(report, inf.correspondence);
    report_the_rest(report, inf);
    report_diagonal_continuation(report, inf.diagonal_continuation);
    report_periodic_denoise(report, inf.periodic_denoise);
    report_piece_slam(report, inf.piece_slam);
    report_overlap_corner(report, inf.overlap_corner);
    report_slide_to_touch(report, inf.slide_to_touch);
    report_project_on_seeds(report, inf.project_on_seeds);
    report << "[snap_to_anchors] holds " << inf.snap_to_anchors.holds() << "\n\n";
    report << "[piece_frame] holds " << inf.frame.holds();
    static constexpr const char* frame_kinds[] = { "cut out", "scaled up", "scaled down", "tiled", "posed",
                                                   "fractal", "repeat unit", "laid out", "filled" };
    const piece_pick& pick = inf.frame.pick;
    const string pick_said = "  how " + to_string((int)pick.how) + (pick.but_those ? " but those" : "") +
                             "  value " + to_string((int)pick.value);
    if (inf.frame.holds() && inf.frame.kind == piece_frame_rule::laid_out)
        report << "  laid out  cut " << segmentation_names[(int)inf.frame.cut] << "  colors " << pick.colors
               << "  cell " << (int)inf.frame.factor;
    else if (inf.frame.holds() && inf.frame.kind == piece_frame_rule::filled)
        report << "  filled  cut " << segmentation_names[(int)inf.frame.cut] << "  colors " << pick.colors
               << pick_said << "  keyed " << (int)inf.frame.keyed;
    else if (inf.frame.holds() && inf.frame.kind != piece_frame_rule::cut_out)
        report << "  " << frame_kinds[inf.frame.kind] << "  factor " << (int)inf.frame.factor;
    else if (inf.frame.holds())
        report << "  cut " << segmentation_names[(int)inf.frame.cut] << "  colors " << pick.colors << pick_said;
    report << "\n\n";
    if (inf.framed) {
        report << "[framed] the same-size puzzle read through the frame\n";
        report_fired(report, inf.framed->correspondence);
        report_the_rest(report, *inf.framed);
    }
    report << "[copy_to] holds " << inf.copy_to.holds();
    if (inf.copy_to.holds())
        report << "  panes " << inf.copy_to.panes << "  anchor " << (int)inf.copy_to.anchor;
    report << "\n\n";
    report_bridge(report, inf.bridge);
    report_mark_lattice(report, inf.mark_lattice);
    report << "[interior_spread] holds " << inf.interior_spread.holds() << "\n\n";
    report_input_blocks(report, inf.input_blocks);
    report_ray_extension(report, inf.ray_extension);
    report << "[rarest_color] holds " << inf.rarest_color.holds() << "\n\n";
    report << "[edge_slide] holds " << inf.edge_slide.holds();
    if (inf.edge_slide.holds())
        report << "  cut " << segmentation_names[(int)inf.edge_slide.segmentation]
               << "  pieces " << (is_color(inf.edge_slide.color) ? "of color " + to_string(inf.edge_slide.color)
                                                                  : string("all"))
               << "  way " << direction_names[inf.edge_slide.slide_direction]
               << "  laid " << (inf.edge_slide.over ? "over" : "under");
    report << "\n\n";
    report << "[corner_crop] holds " << inf.corner_crop.holds();
    if (inf.corner_crop.holds())
        report << "  corner " << (inf.corner_crop.down ? "bottom" : "top") << "-"
               << (inf.corner_crop.right ? "right" : "left");
    report << "\n\n";
    report_said_reading(report, inf);
    report_agreed_moves(report, inf.correspondence);
    report_pieces_seen(report, inf.correspondence);
    for (size_t i = 0; i < inf.correspondence.chosen_pairs().size(); ++i)
        report_one_pair(report, i, inf.correspondence.chosen_pairs()[i]);
}

}  // namespace arc
