# Handoff, 2026-10-08

Read AGENTS.md first: the rules, how to build and check, where things are.
This file holds only what is current; past stages are in git history
(`git log`, and this file's own history).

## Where it stands
- Training: 199 right, 0 wrong. Current run: out/base/ (out\ is not
  in git). 170 at 32d371d, 172 at a201ff6, 187 at 7740de8, 199 at abc3f0b.
- EVALUATION, 2026-10-08, inference only (arc_headless, ARC_INFERENCE_ONLY=1,
  ARC_DATASET=eval): 0 of 167 test outputs, 0 of 120 puzzles; no program
  fits any puzzle's pairs (out\eval\). Readouts: `ARC_DATASET=eval
  scripts\said.bat evalread` (out\evalread\). Missing words named: copies 99,
  recolor 71, turns 64, where pieces went 55, shape changes 36, drawn on the
  ground 13; 58% lack 3 or more words (31% of unsolved training), only 23
  lack one. The training solves bunch in the first hundred by id (42, then
  10-24 a hundred). 8f215267 names no missing word yet does not fit: cells
  drawn with no partner wait on no word, so `missing` undercounts.
- Commit 17715bb simplified files 0-11 again, every step a
  full build and a score against out/base: 199 right, 0 wrong, gained 0,
  lost 0, every program and all 1000 readouts byte-for-byte unchanged. Net
  source: 67 fewer lines (1_readings +34, 4_objects -4, 7_infer -79,
  cuda_solver.cu -18). The GPU kernel's stack 29,536 -> 29,488 bytes.
  - One box: object_summary::box() and cell_box::include; box_of is gone.
  - One numbering of directions: walk_out, on_object_edge and 3_background's
    side loops step through heading (every_nth_way<4> for the sides); the
    neighbour_row/col_4/8 helpers are gone. touches() is the one touch test
    (each_partner, move_toward); a cell on the other piece counts as touching.
  - color_tally: "most cells, lowest color on a tie, none where nothing was
    counted" for majority_vote, object_paint_color, denoise_ground, the snap
    body, the most-common pick, and the GPU's SelectMostCommonColor (it had
    its own copy). The tie rules that differ (rarest color refuses a tie,
    Pattern's unit and the denoise vote go by scan order) were left apart.
  - object verbs: painted_box, repaint, posed (the pose block moved above
    object); paint_piece moved beside empty_grid and paints the hand-written
    loops in read_the_cut, finish_the_line and size_rank_holds.
  - Shared with the GPU (HANDOFF #6 done): mirror_middle, each_mirror,
    mirrored_box, mirror_pose (MirrorAround: its op, mirror_holds,
    carry_the_hand), march_trail (ExtendTo: its op, trail_holds).
  - Replays return one result (#3 done): optional<replayed> (drawn, left,
    pieces, paint, every_piece) and cells_by_pair; bar_paint still says
    every piece itself (Bridge reads every piece whatever is picked).
  - snap_to_anchors keeps a plain range check on a key's color: the static
    analyzer (C6385) cannot see through is_color once the tally made the
    function simpler for it to walk.
- Two sessions ago grouped the words: group_of in 7_infer, just above
  carry_the_hand, puts each fired_rule::said_as kind in one word_group
  (recolors, changes_pieces, removes, paints, whole_program); a
  static_assert checks every kind has one (kind_count ends the enum). It
  replaces paints_after_the_line (6), reshapes, moves_a_pick and moves_cells
  (finish_the_line), and gates beats_of and drawn_after (9_sentences).
  paint_value names the +32 packing of rays and repeats. 199 right, 0 wrong,
  gained 0, lost 0; every program and all 1000 readouts byte-for-byte
  unchanged. Net source: 25 more lines. A first try, word_for (a full row
  per kind: op, argument, landing), added more and was dropped.
- The previous session simplified files 0-11 upstream to downstream, 226 fewer
  source lines, every step a full build and a score: 199 right, 0 wrong,
  every program and all 1000 readouts byte-for-byte unchanged, but where a
  step renumbered on purpose:
  - Directions are numbered as a compass is, clockwise from up (heading),
    for MoveToEdge, ExtendTo and Shift alike, and asked sides first, then
    the diagonals (heading_asked). Partner ways are numbered in the order
    they are asked (partner_by). The stored solutions were renumbered. Four
    unsolved readouts changed: a tie between two directions goes the new
    asking order's way (1190bc91, 3345333e, 444801d8, a2d730bd).
  - Fit's targets are named (fit_to). The partner helpers ask one pieces
    object (count, box, painted, is_picked, color, holds), as partner_pieces
    and gpu_partner_view answer them. One flood fill (walk_out) for the hole
    count, would_fit and snap_to_anchors. No reader takes a ground: every one
    was handed figures only. A pick turned round is piece_pick::but_those
    (the readout's frame line says "how N but those"). palette, empty_grid
    and bands_of replace their hand-written copies.
  - The flood fill costs the GPU's dsl_exec_and_score_kernel_v 28,192 ->
    29,536 bytes of stack: walk_out keeps two queues where the hand walks
    kept one. The user said speed can wait.
- solutions/ holds the inferred program of every solved puzzle. Left as they
  were: the manual 05a7bcf2, 10fcaaa3 (now inferred too), 32597951,
  f25fbde4, and 44d8ac46 (an old inferred one that no longer fits).
  solutions/shelf/ holds puzzles outside training.
- BUILD TIME IS THE BOTTLENECK NOW (the user: speed up the build or drop the
  GPU soon). One checked step this session took 7-10 minutes: a full build
  3-4 minutes, most of it cuda_solver.cu, which every header in include/core
  recompiles; then a score of 85-220 s. Even `said.bat` builds the GPU
  library, since arc_said links it.
- A full score takes 30 to 80 s; the laptop's clock moves with heat, so
  compare two builds by interleaving them (old, new, old, new) and taking
  each one's fastest. `ARC_THREADS=1` reads one puzzle at a time (about 15
  minutes), for when the user is gaming. scripts\profile.bat profiles.
- What blocks the unsolved (`readouts.py missing <label>`, named / the only
  word missing): copies 485 / 73, a recolor 383 / 71, turns 254 / 26, where
  pieces went 223 / 44, cells drawn on the ground 143 / 67, shape changes
  116 / 7.

## How to check a refactor
- Full build, then `scripts\said.bat <label>`, then against the last run:
  `cmp out\<old>\said.txt out\<new>\said.txt` and
  `diff -rq out\<old>\logs out\<new>\logs` (byte for byte), and
  `readouts.py solved` and `explained`. Where a change renumbers values on
  purpose, map the old run's numbers first, then compare.
- Where a helper the GPU runs changes, compare the build's `ptxas info` and
  `stack frame` lines before and after.
- Line endings are mixed in the repo: 1_readings, 3_background, 4_objects,
  5_matching and 6_correspondence are CRLF; 7_infer, 8_vocabulary, 9_sentences, 10_log,
  cuda_solver.cu, this file and solutions/ are LF, though .gitattributes
  says CRLF everywhere (the user's call). Keep each file's own. Git Bash's
  `grep -c $'\r$'` counts every line: check endings with Python.
- Delete each run once it has been compared (about 300 MB each).

## Next: the simplification, upstream to downstream
The user's order: upstream first, each change living in the file that first
uses it, global before local. Estimates are rough.
1. 3_background, the election (~100 lines; the user's call first: it
   changes which plans are read). says_the_most picks the plan now, so the
   election only proposes plans; the veto, the per-pair majority and
   is_per_pair_raw (a copy of is_per_pair) may go.
2. Word table: done as word groups (above); the ~150 lines it was
   estimated to save were not there. Left: the matching's missing-word and
   freedom mapping (read_the_acts, said_under) belongs in 6 at its first
   use; the partner value packing (value / PARTNER_KINDS) belongs with its
   shared arithmetic. The local name arrays each have one owner already;
   move them only when a second reader needs them.
3. Done this session (replays return one result).
4. Tried this session and dropped: one frame for the line's kinds (`ask`
   over picks and values, `offer_of` for the saying) passed every check but
   made finish_the_line 13 lines longer; the replay result in 3 already took
   what there was. Splitting finish_the_line into named parts would read
   better, not shorter.
5. One mechanism for a rule that redraws every output by itself (~100 over
   7, 9 and 10): today the rivals (own_sentences, list_the_rivals,
   say_rival) and the whole-line rules (whole_line, and six branches in
   say_the_cut); each rule touches six places (struct, inference field,
   read call, listing, say_X, report_X), and 10's small report_ functions
   would fold into one.
6. Done this session (mirror and trail shared with the GPU).
7. Measured this session: no old rival can go yet. With all eight off
   (bridge, hole colors, interior fill, rarest color, copy-to, edge slide,
   rays, corner crop) 12 puzzles were lost and 5 went wrong: interior fill
   00d62c1b, 7b6016b9, a5313dff (wrong without it); corner crop 5bd6f4ac,
   d10ecb37; hole colors 0a2355a6, 37d3e8b2; rarest color 1f85a75f (wrong);
   edge slide 3618c87e (wrong); rays 0962bcdd; bridge 22eb0ac0; copy-to
   363442ee. Each is the line's word to learn before its rule can go -- the
   wrong ones first: the line's fill, edge slide and least-color readings
   fit the pairs and miss the test.
7a. Smaller, from this session's report: framing dispatched once
   (choose_piece_frame and read_the_cut both switch on the frame kind,
   ~40); one walk for "the posed piece laid on a place" in 5_matching
   (written about 9 times) and an edge that holds its object_match instead
   of copying 7 fields (~60); a sentence's source as a field in 9_sentences
   instead of cutting strings (substr(5)) in frame_the_sentences,
   cut_then_tile and say_the_plan (~20); typed op names (spell(node_type))
   for the 71 spell("...") calls; one piece map for "which piece holds this
   cell" (6 twice, 7); color sets are both bool[10] and uint16_t masks;
   the packed values of AlignTo, MirrorAround and MoveToward beside
   paint_value; the two pre-existing compiler warnings (C4458 `count` in
   shape_color_map_rule, C6336 in picks_to_ask's middle-out values).
8. Think with the user: grid::tile_width/tile_height is state every grid
   carries for one frame (10fcaaa3); passing the tile to each_object where
   the frame needs it would keep the base type plain.
9. Structure: 7_infer.hpp includes 8_vocabulary.hpp, so the numbered
   reading order runs backwards there; 7_infer (about 6,300 lines) could be
   split by owner (tables, the standalone rules, picks and frames, replays,
   the line and its choices). Renumbering touches 12+, src and
   kaggle_push.py.
10. Stale text: "EXPERIMENT" on the chance test (5_matching gather_matches);
    "Detect-only for now -- nothing segments per-pair" (3_background); the
    "cannot say yet" list naming Output (8_vocabulary); "background_reading
    ::central" (9_sentences); the log's "background -1" (report_colors
    takes a ground no one has); short names in vote_denoise_cells.
11. Generality: rule indices are int8_t, 127 rules, which is why each kind
    says at most 8 words; int16_t would lift it. MOST_PIECES_READ = 128 is
    TEMPORARY.
12. GPU, when speed matters again: walk_out with one queue of cell numbers
    would undo the flood fill's stack, and the queue is each_object's cell
    list too, so its scratch loses two of three arrays. The busiest walk:
    check speed by interleaving.

## Next, from before (the user's order)
1. 18286ef8, parked (the user agreed). Sayable, checked with arc_spell
   (fits, test right): `SelectColor[9] MoveToward[15]`, then
   `SelectColor[6] Recolor[9]`, then the 5s' box `AddColor[5] Underlay`. Not
   found; it needs the move to claim the input cell its landing covers (it
   ties ColorFrom's "the 9 takes the 5 around it" at 6 cells), "6 becomes 9"
   sayable after the move (by color it is only an act, and the acts open the
   line), and the box fill asked of a line that moves pieces.
2. Speed: read the cuts and plans of one puzzle in parallel (low priority).

Smaller, open:
- Recolors: 71 unsolved lack only a recolor, mostly a color taken from
  another piece (compositions of ColorFrom's partners, picks, frames). Work
  one puzzle at a time with the user, the fewest pixels first: next
  e3fe1151 (a 7 cross cuts four panes, and each pane's stray 7 takes the
  color it lacks -- a new op, no composition says it), fc754716.
- ef26cbf6, open: "ColorFrom[4], the nearest" says it (checked in Python,
  test right) with the walls picked along with the shapes, so they are no
  shape's partner -- but every dot is as near a wall as the next, and a piece
  whose partners disagree is one ColorFrom's word refuses: let through, it
  made d23f8c26 fit and go wrong on its test.
- Four-connected before eight until eight is needed (the user): asked of the
  cuts too (the cut by sides just ahead of the cut by corners), it lost
  4938f0c2, 64a7c07e and 9172f3a0, two of them wrong, so the cut by sides
  still comes last.
- carry_the_hand cannot carry a learned table, so no word that needs the
  hand is asked after a Shape Picks Fill (6d75e8bb is solved by one).
- The freedoms inside a cut are still chosen by the matching, not the words;
  do it if the data asks. FitInto (would_fit answered with the place) when a
  puzzle needs it; WouldFit's gap is "for now" (the user: not general).

## How the reading works now (decisions to keep)
- The ground is owned once (3_background.hpp): a plan deletes its ground
  first and fills last; every reader sees figures and empty cells only, and
  none takes a ground. On a same-size puzzle the fill color is ground on the
  inputs too (fill_is_ground_too); on a size change it can be a figure the
  rule reads (1b2d62fb's 0s).
- Every choice is says_the_most's: the plan, the cut, and whether the line
  opens with the matching's acts or leaves them to the words.
- The line (finish_the_line): each kind of word offers its best on the hand
  the line leaves, the biggest offer is said, then all are asked again; each
  kind says at most 8 words (a reading lists at most 127 rules). The paints
  follow the line, each in a sentence of its own, and compete the same way,
  the biggest first (e729b7be). A paint is weighed only where it explains a
  cell in every pair (91714a58).
- What the line draws is said (Drawn By The Line, list_claims), however the
  matching paired the pieces (d23f8c26's middle column).
- A recolor or a move answers to any cell changed in place
  (left_over_cell::answers_to). The cell a moved piece leaves may show
  anything but the piece itself.
- A replay checks everything its op draws, with the op's own arithmetic: the
  trail redraws its body; the turned figure is the whole drawing.
- Every replay (`*_holds`) sits on paint_holds (recolors, mirror, rays,
  trails, copies, repeats, the turned figure, boxes, bars, fills),
  slides_hold (the moves) or drops_hold (removal, outline), handed the
  pieces its pick takes (handed_pieces). At work in every pair: at some cell
  it acts on, the output shows what its picked pieces did not show there
  before (4938f0c2's last pair is already whole).
- paint_holds owns how a paint lies: over, under (first to reach a cell
  keeps it, as Underlay) or onto, on one canvas of what is down, and in its
  own colors, one color, or its own else one. A paint is never no color.
- The cut by sides (four_connected_monochrome, Split[3]) is read after every
  other cut, only where it cuts differently and into no more pieces than the
  reading reads.
- A tiled input is copies of the input's pieces: its program cuts before
  Tile, and the reading cuts a grid laid out of copies one copy at a time
  (grid::tile_width, each_object; 10fcaaa3).
- A list holds 900 pieces (MAX_OBJECTS_IN_A_LIST); the reading reads a cut
  only where no input cuts into more than 128 (MOST_PIECES_READ, TEMPORARY).
- Partner ways are numbered in the order they are asked, surest first
  (partner_by): around it, inside it, its rows, its columns, the nearest, the
  touch at a side, the touch at a side or a corner, the same shape, turned
  or not. Directions are numbered as a compass is, clockwise from up
  (heading), asked sides first (heading_asked). A tie goes to the first
  asked.
- The partner helpers (each_partner and kin, 1_readings.hpp) ask one pieces
  object, so the op and the replay find the same partners; move_toward and
  would_fit too.
- ColorFrom leaves a picked piece with no partner as it is (LEFT_AS_IT_IS);
  one whose partners disagree it refuses (d23f8c26).
- Bridge hands back its bars as one drawing.
- ShapeColorMap[kind]: each picked piece takes the color its key names --
  [0] its shape however it turns, [1] its symmetry (shape_key_of, the one
  key of reading and op; shape_table the one lookup, with a color for every
  key it does not name). read_off reads the table off the pairs: keys seen
  in two pairs, and a key seen in one only where its color is the table's
  alone to say (27a28665; ef26cbf6 and b7256dcd went wrong otherwise); the
  rest may share one color (25d8a9c8). detect_body_recolor says a key naming
  a figure's color in three words.
- The filled frame (piece_frame_rule::filled), where every output is one
  color: the pieces a pick takes, each in its own color or its key's,
  stretched over the canvas (AddColor[11] Fit[1]; 44f52bb0).
- A pick (piece_pick, 6_correspondence.hpp) is by colors, or by a kind --
  holes, size, band, the most common color, at least so many cells, the
  first so many by size, a row or column from the middle -- and but_those
  turns a kind round, said with InvertSelection. A count or a rank is asked
  only where it splits some pair's pieces with no tie.
- Size rank and the fill by shape replay on the hand, so they are asked only
  while the hand is known.
- MoveToward[value]: value / PARTNER_KINDS is how far, value % PARTNER_KINDS
  the partner (each partner way, then each color).

## Kaggle and cloud
- Only one-shot inference is tested on Kaggle (ARC_INFERENCE_ONLY=1, no
  search). Submission 56790276 (notebook v37) scored 0.00; v38, with the
  loader fix, was never submitted. A local evaluation of 120 tasks scored 0.
  Source upload: `python scripts/kaggle_push.py -m "..."`; smoke run with
  `--smoke`. Scoring rules and links are in README.md.
- Linux cloud: `python3 scripts/cloud_setup.py` builds the readouts only (no
  GPU); never run in an actual Claude cloud session.
