# Handoff, 2026-10-06

## Next session (state on 2026-10-06, pushed to main at the user's request)
- 170 right, 0 wrong (out/order3-said.txt, out/probe/order3/). The 3 wrong
  puzzles are fixed, and 7 gained today; how, step by step, is in the to-do
  list of the "one measure" section below (items 1-3, 5).
- A full score takes about 35 s on a cool laptop and 70-80 s on a hot one;
  compare times back to back. Check a change the same way as always:
  programs, readouts and summary identical, or say what moved.
- Left, in the user's order:
  1. DRY: the rest of the replays (`*_holds`, 7_infer.hpp) onto templates,
     as paint_holds (rays, trails, copies, repeats, the turned figure) and
     slides_hold (align, step, edge, shift, toward) now are: about 15 left
     (color from a partner, picked recolor, removal, mirror, outline, fills,
     boxes, bridges, size rank...). Also "STILL OPEN" in the speed section:
     whether replay_facts keeps the pieces it made ready.
  2. Merge the score and the readouts (below); needs the user's decisions.
  3. 18286ef8, parked (the user agreed): sayable, but not found without
     three more changes to shared rules (item 5 of the to-do below).
  4. Speed: read the cuts and plans in parallel (low priority now).
- MERGE THE SCORE AND THE READOUTS (the user: "confusing and inconsistent,
  both need merging and consistency"). Today they are two tools, built and
  run two different ways, that read all 1000 puzzles the same way twice:
  - arc_said (scripts/said.cpp), the score: built by the normal build, run
    by hand (`out\x64-Release\arc_said.exe > said.txt`, or `<id>` for one),
    no .bat of its own. Needs the GPU: runs the programs on the pairs and
    the test. Prints id, SOLVED or WRONG, program, and a tally on stderr.
  - arc_probe (scripts/probe.cpp), the readouts: left out of the normal
    build; `scripts\probe.bat <label>` builds and runs it. No GPU, runs no
    program. Writes out\probe\<label>\logs\inference\<id>.txt and
    summary.txt. The Linux cloud sessions build only this one
    (scripts/cloud_setup.py), as they have no GPU.
  - scripts/readouts.py reads both: `solved` takes two said files,
    `explained` two labels and two said files, `missing` one of each.
  - Wanted: one run that scores and writes the readouts, built and run one
    way, its outputs in one place, the readout written by one owner. A
    GPU-free build that writes only the readouts must stay (the cloud).
    Agree the details with the user first: the name, the one command, where
    the outputs go. Then update AGENTS.md "Build and check" and CLAUDE.md.

## Speed of the dataset run: profiled, matching, parallel (2026-10-05)
- User paused feature work to address dataset run time and code repetition.
  CPU profiling requested first; no solver rules changed. Then agreed: the
  duplicate-placement scan, and matching, the biggest time consumer. Then,
  in the user's order: puzzles read in parallel (done), and only then the
  replay pieces (partner_pieces, below): the nothing-kept way done, the rest
  waits on the user's decisions.
- Then (2026-10-05): arc_said about 30 s cool (less since the replay
  pieces), arc_probe 32 s for all 1000 puzzles; 163 right, 3 wrong; every
  program and readout as before all this. More threads than the CPU has did
  not help (20: 26 s, 40: 27 s), and a hot laptop runs the same thing in 75 s.
- scripts/profile.bat now builds RelWithDebInfo and profiles arc_said, then
  exports function, source-line, thread and call-stack reports. Each run has
  its own profiling/output/cpu-<timestamp>/ folder (ignored by git).
- VTune user-mode launch stalled before CUDA startup on this machine. Attach
  after startup works (profiling/collect_cpu.ps1); score covers all puzzles,
  profile excludes startup/attachment. No admin needed. A control-connection
  error occurred on target exit; separate finalization recovered the capture.
  The runner now finalizes separately and requires both score and report.
  Collection skips finalization with -finalization-mode=none (this VTune
  ignores -no-auto-finalize, so every capture had been finalized twice).
- Full capture: profiling/output/cpu-20261005-164441-893/. summary.txt:
  850.629 s elapsed, 838.082 s sampled CPU. Programs/verdicts exactly match
  out/fixes-said.txt: 163 right, 3 wrong, gained 0, lost 0.
- Source attribution: 5_matching.hpp 38.04% of sampled CPU; 1_readings.hpp
  19.74%; vector 11.87%; 7_infer.hpp 1.72%. Matching's duplicate-placement
  scan at 5_matching.hpp:736-737 accounts for about 12.8% (107.1 s).
  Runtime functions include unsymbolized VCRUNTIME functions (14.89% total
  module time) and malloc_base (6.32%). Do not mistake CPU samples for GPU
  elapsed time or treat the name-based area summary as inclusive caller cost.
- MATCHING MADE FASTER, NO READING CHANGED (5_matching.hpp). Started in a
  Codex session (the first two), finished in a Claude session:
  - A source's shape matches skip a corner it already placed through a
    per-source table, not a scan of every match found.
  - The greedy fallback reuses the fitting's cells and order instead of
    redrawing every piece inside every comparison. The fitting now keeps one
    candidate per match, empty ones too (the work budget still counts only
    the ones that show cells, as before).
  - Each source is turned once per pose, not once per hit; whether a match
    lands on more than background is asked as it is kept (posed_for).
  - The fitting sorts its candidates once (best_first). Each cell's options
    are listed in that order in one flat list (options_at), and the extras
    (keep_what_fitted) and the greedy pass read best_first: three sorts per
    fitting and one per cell are now one.
  - Checked each step: programs and verdicts identical to
    out/fixes-said.txt (163 right, 3 wrong, gained 0, lost 0), and all 1000
    readouts identical to out/probe/fixes (out/probe/matchspeed after the
    first two, out/probe/matchspeed2 after all four).
  - Time: profiled 850.6 s -> 665.9 s after the first two
    (cpu-20261005-171532-192). After all four, Release arc_said run back to
    back with the last commit built apart: 1188 s -> 321 s, the same 166
    lines (out/head-said.txt, out/matchspeed3-said.txt). One earlier run of
    the new code took 209 s: this laptop varies a lot run to run (the 887 s
    above was the same code as the 1188 s), so compare times back to back.
    Readouts (arc_probe, all 1000): 218 s.
- Where the time went in cpu-20261005-171532-192 (inclusive CPU, 655 s):
  read_figures 95%; matching (collect_edges) 57%, of it gather_matches 32%
  and set_up_the_fitting 17% (the last two fixes go at those); finish_the_line
  28%, of it partner_pieces 19% (124 s). Running programs on the GPU: 4.5%.
- PUZZLES READ IN PARALLEL: arc_said and arc_probe read a puzzle on every
  CPU thread (scripts/in_parallel.hpp) and hand back their lines in the
  dataset's order. The reading shares nothing between puzzles (no global
  state, no caches). The GPU checks stay one puzzle at a time (a mutex in
  said.cpp): every run goes through one scratch slot
  (cuda_dsl_upload_scratch_example), and programs are parsed into one
  unsynchronized pool (node_pool). ARC_THREADS=n sets the threads (unset:
  all); scripts/profile.bat sets 1, so profiles compare with earlier ones.
  - Numbers: arc_said 321 s -> 30 s, arc_probe 218 s -> 32 s (without its
    build). The same 166 lines (out/parallel-said.txt), all 1000 readouts
    and the summary identical (out/probe/parallel): 163 right, 3 wrong,
    gained 0, lost 0, explained but unsolved 2 -> 2.
  - Not run: the Linux cloud build of arc_probe (cloud_setup.py already
    passes -pthread). Not done: the cuts inside one puzzle in parallel
    (to-do 4 below), and the Kaggle runner (src/kaggle_main.cpp).
- THE PIECES MADE READY ONCE A PICK, NOTHING KEPT (the user's choice, over
  a notebook that remembers). rays_hold and tile_holds were asked every
  length x both ways x every lay of a pick, and each try rebuilt every
  pair's piece boxes and picks though only the pick decides them. Now the
  caller makes them ready once a pick (handed_pieces, 7_infer.hpp), every
  try reads them, and they are dropped after. Safe because `opened` is const:
  nothing carries that hand while the pick is asked.
  - Numbers: programs identical (out/handed-said.txt), all 1000 readouts and
    the summary identical (out/probe/handed): 163 right, 3 wrong, gained 0,
    lost 0, explained but unsolved 2 -> 2. Old and new run in turn on a hot
    laptop: CPU time 538/509 s -> 415/368 s (about -25%), wall 81/71 s ->
    64/48 s (cool, the old one ran in 26 s).
- STILL OPEN, THE USER DECIDES: the other ~18 replays still make their
  pieces ready every try (about 15 s of CPU in the single-thread profile).
  Proposed and understood, not chosen yet: replay_facts keeps the pieces it
  made ready, and the one door that changes the pieces (carry_the_hand)
  wipes them, so nothing stale can be read. The user asked whether keeping
  them beats remaking them (here: big work, a few KB), and chose the
  nothing-kept way first.
- The 3 wrong cases of that day were fixed on 2026-10-06 (next section).

## Completed: one measure for every choice, WouldFit, and the fixes after (2026-10-05/06)
Pushed as a checkpoint at the user's request with 3 wrong (was 0). The
to-do items 1-3 below fixed all three on 2026-10-06, and item 5 went on to
9565186b, MoveToward and the most-first line: 170 right, 0 wrong.

- Numbers (out/fixes-said.txt, out/probe/fixes/): 163 right, 3 wrong, lost 0
  against out/speed-said.txt (152 right). Gained 11: 1acc24af, 4258a5f9,
  62ab2642, a61f2674, a699fb00, b60334d2, ce22a75a, d037b0a7, d13f3404,
  ded97339, f45f5ca7. Reworded, still solved: 06df4c85, 070dd51e, 140c817e,
  22eb0ac0, 4f537728, 5614dbcf, aabf363d, b230c067, c7d4e6ad. Wrong:
  11852cab, 5582e5ca, c35c1b4c (also the 2 "explained but unsolved").
  A full score takes 887 s (was 344 s): unsolved puzzles read every cut.
  Tests left: only tests/puzzles (the user removed the rest, see below).
- WouldFit[how] (new op, select role, 8 partner ways): of the picked pieces,
  only those that would fill a gap of their one partner among the unpicked
  stay picked; nothing moves. would_fit in 1_readings.hpp is the one helper
  (GPU op and replay). A gap, "for now" (the user: not general): a run of the
  partner's empty box cells meeting the box edge on one side only, the side
  the piece lies past; any of the 8 poses; leftover cells hang past that
  side. Asked by the recolor-a-pick word with the colors an unsaid recolor
  starts from (picks_to_ask's fit_colors), only where the fit keeps one piece
  and drops one in every pair. 1acc24af: SelectColor[5] WouldFit[0]
  Recolor[2]. Agreed later: FitInto, the real Tetris move, when a puzzle
  needs it (same helper answered with the place).
- ONE MEASURE FOR EVERY CHOICE (the user found two yardsticks: plans judged
  by the words, cuts by the matching). says_the_most (7_infer.hpp) is the one
  owner: candidates read in order, the fewest cells unsaid kept, a tie to
  the earlier, stop at 0. Plans: in infer. Cuts: in read_figures, just above
  the first word that reads the cut (read_the_cut); words that read no cut
  are asked once, above it. The election only RANKS cuts (ranked,
  ties_of_ranked, choose); a cut the program cannot hold (more than 32 pieces
  an input) is no candidate while one is (4f537728). Readout: "[cuts]" line.
  Not done: the freedoms inside a cut are still chosen by the matching (same
  inconsistency, one level down); do it if the data asks.
- Made honest along the way (each surfaced by the new choice):
  - cells_left_unsaid counts every cell where the pieces go unsaid whole
    (no word, pieces do not redraw) -- pieces_go_unsaid, shared with the
    speller's "no word for what happened" (ba1aa698).
  - write_down_what_changed: an output-side landing that names no input
    piece (partly hidden, off the grid) and is found elsewhere or turned now
    waits on a word (18286ef8's moved 9 had been "explained" by nothing).
  - A swap paints both its colors: a second swap of either is not said
    (9565186b's chain of swaps undid itself).
  - Picks by size are asked nearest an end first (largest, smallest, 2nd
    largest, ...): a tie goes to the plainer name (62ab2642: "smallest", not
    "9th largest").
  - Bridge[way] (0..5): rows+columns, rows, columns, to the nearest; the
    same to every one. Way 0 is the old Bridge. Asked narrowest first
    (bridge_ways_narrowest_first), so bars are said no wider than the pairs
    show them (a699fb00: Bridge[1]; ded97339: Bridge[3]).
- TO DO, in order (agreed analysis; the user stopped before the fixes):
  1. DONE 2026-10-06. 5582e5ca was read as "the 9 grew over the others", so
     no cell waited on a recolor and no color word was asked. Now said as
     "every color but the most common takes the color of the nearest
     piece" (SelectMostCommonColor InvertSelection ColorFrom[0]), test
     right. Two changes in write_down_what_changed (6_correspondence.hpp):
     - The agreed fix: a left-over cell whose place the other grid shows in
       another color notes that color (`in_place`), and a recolor answers
       to it however the matching read it (left_over_cell::answers_to): the
       colors the recolor words are asked with (starts_from) and the cells
       ColorFrom and a picked Recolor claim. Swaps, acts, size rank and the
       shape table still claim only cells that wait on a recolor.
     - Beyond the agreed fix (tell the user): Sits On now writes "how a
       piece changed its shape" only on cells the other grid does not show
       in the same color at the same place; a cell that stayed is no
       change. Without it the unchanged 9s were left over, CopyTo and TileX
       claimed them, and the test stayed wrong (out/inplace-said.txt).
     Numbers: 164 right, 2 wrong, gained 5582e5ca, lost 0; explained but
     unsolved 2 -> 1 (5582e5ca, now solved), none new. Every other program
     identical; readouts of solved puzzles differ only in counts. 70
     unsolved readings say more, 1 says less (45737921: two ColorFrom picks
     tie at 48 cells, and "largest" is asked before "most common color").
     Score 32 s, readouts 34 s. Evidence: out/inplace3-said.txt and
     out/probe/inplace3/.
  2. DONE 2026-10-06. c35c1b4c: the plan "delete each input's own, fill 0"
     kept black in the inputs and emptied it in the outputs, so black pieces
     looked removed and "remove the 3rd largest" claimed cells that never
     changed. Now a plan's fill color is ground on the inputs too (deleted
     first) where every input lies over its output
     (ground_plan::fill_is_ground_too, from the election's `overlays`; one
     owner for figures_only, foreground_input and the readout). The
     plan "keep everything" now says the most and the mirror alone is said:
     test right. Not on size changes: there the fill color can be a figure
     the rule reads (1b2d62fb's 0s, read through its panes); the fix on
     every puzzle lost 1b2d62fb (out/fillground-said.txt). Numbers: 165
     right, 1 wrong, gained c35c1b4c, lost 0; explained but unsolved 1 -> 1,
     none new. 7b6016b9 reworded (also deletes its 3, which its inputs lack).
     Readouts now print what a plan deletes ("delete each input's own and
     0, fill 0"). Evidence: out/fillground2-said.txt, out/probe/fillground2/.
     Seen on the way, not fixed: under the plan "delete 0, fill 0" the
     mirror word is never asked, because Shape Picks Fill (a spurious 1-cell
     table) is said first and carry_the_hand cannot carry the hand through a
     learned table.
  3. DONE 2026-10-06. 11852cab: mirroring fit training only because every
     cell to copy sat on a diagonal; the test's 4 sits straight above the
     middle. The user: quarter turns are not mirroring, and TurnRight and
     TurnLeft already exist. So no new op: a new word, "the whole figure
     drawn again turned in its own box" (said_as::turned_copy), spelled
     `CurrentObjects SelectAll UnionList Trim TurnRight Union` (and
     TurnRight TurnRight, TurnLeft), a paint said after the line, asked
     only of a still line. It claims the figure where it lies as well as
     its copies (the matching read the figure turned). Every piece only: a
     part turned in its own box fit 2546ccf6 and 52df9849 by accident, both
     wrong on test (out/turned-said.txt). Now said in the 8-connected
     multicolor cut, which the matching ranks first, so MirrorAround is
     never reached.
     - DRY (the user: templates, not copies): paint_holds (7_infer.hpp)
       is the one replay of a paint laid over what is down; rays, trails,
       copies onto pieces, repeats and the turned figure each say only what
       they draw. fired_rule::paints_after_the_line owns the list of paint
       words the line's `changes` and `moves_cells` read. The GPU's
       TurnRight/TurnLeft/FlipHorizontal/FlipVertical read through
       pose_source (pose_of in 8_vocabulary.hpp), as the replays do.
     Numbers: 167 right, 0 wrong, gained 11852cab and e729b7be (the whole
     figure, a half turn), lost 0; explained but unsolved 1 -> 0. Only 3
     readouts changed, all by the new word (5c2c9af4 still unsolved); every
     other program and readout identical, so the refactor moved nothing.
     Evidence: out/turned2-said.txt, out/probe/turned2/.
  4. Speed: read the cuts (and plans) in parallel; 887 s vs 344 s. Since
     then matching went 1188 s -> 321 s and the puzzles are read in
     parallel: 30 s (the speed section above).
  5. Re-export DONE 2026-10-06: every solved puzzle's program from
     out/turned2-said.txt written to solutions/<id>.txt (167: 47 unchanged,
     103 reworded, 16 new, and 11852cab's hand-written program -- the same
     three TurnRight copies -- replaced by the inferred one; it stays in git
     history). Left as they were, not solved now: 05a7bcf2, 10fcaaa3,
     32597951, f25fbde4 (manual) and 44d8ac46 (an old inferred one that no
     longer fits).
     9565186b DONE 2026-10-06: the act "8 becomes 5" opened the line and left
     the first pair nothing for "every color but the most common becomes 5"
     to change. Now read_the_cut asks says_the_most whether to say the acts
     or leave them unsaid (leave_the_acts_unsaid) and let the words alone say
     every change; a tie keeps the acts, and a line that says everything is
     never read the second way. pieces_redraw moved above it (it reads only
     the cut). Said now: SelectMostCommonColor InvertSelection Recolor[5].
     Surfaced and fixed: the trail paint (ExtendTo) hands back the body with
     its trail, so its replay now requires the output to show the body where
     it lies (66e6c45b's words-alone line removed a piece whose trail drew it
     back, and was "explained" while its program drew otherwise).
     Numbers: 168 right, 0 wrong, gained 9565186b, lost 0; explained but
     unsolved 0 -> 0. 21f83797 reworded (dropped a TileX it no longer
     needs), still right. 10 unsolved readings say more, 6 say less (their
     trails were over-claiming). Readouts 35 s. Evidence: out/acts2-said.txt,
     out/probe/acts2/; solutions/ synced.
     MoveToward DONE 2026-10-06 (agreed: the general form of "steps
     toward", kept DRY). MoveToward[value] (new op, last in the table):
     value / PARTNER_KINDS is how far -- one step, its own size, until it
     touches -- and value % PARTNER_KINDS the partner: each partner_by way,
     then each color (the one unpicked piece holding it). One owner of the
     arithmetic, move_toward in 1_readings.hpp (heading_toward: the compass
     sector between the boxes' middles, integer test against tan 22.5),
     called by the GPU op and the replay. DRY on the way: the five move
     replays share slides_hold (align, step, edge, shift, toward); AlignTo
     and MoveToward share goes_level/goes_toward on partner_pieces, read by
     the replay and carry_the_hand; the search's partner-op gating is one
     table (14_puzzle_loader.hpp). Asked last among the moves, so it wins
     only where it says more. Numbers: 170 right, 0 wrong, gained dc433765
     ("the 3 moves one step toward the nearest piece") and a48eeaf7 ("the
     5s move toward the nearest piece until they touch"), lost 0; explained
     but unsolved 0 -> 0; every other program identical. Evidence:
     out/toward-said.txt, out/probe/toward/; solutions/ synced.
     18286ef8 is sayable now (checked with arc_spell: fits, test right):
     `... SelectColor[9] MoveToward[14] Withhold CurrentObjects SelectAll
     InvertSelection SelectColor[6] Recolor[9] SelectAll Union` + the 5s'
     box filled (AddColor[5] Underlay). The reading does not find it.
     Tried first on their own, with the user's go, and reverted (no gain):
     moves answering to cells changed in place (answers_to), and slides_show
     letting the cell a piece leaves show anything but the piece itself.
     Still not found then, because of the line's order: in the words-alone
     line ColorFrom ("the 9 takes the color of the piece around it") is asked
     before the moves, claims the 9's cell and recolors the 9 away, so no
     move of the 9 is left to weigh. Both were applied again below, with the
     most-first line.
     Agreed and done after that (2026-10-06): (1) THE MOST FIRST in
     finish_the_line: each kind (color from a partner, move, mirror, outline,
     recolor of a pick, removal) offers its best on the hand the line leaves,
     the biggest offer is said (a tie in that order), then all are asked
     again; each kind says at most 8 words, as the old 8 rounds allowed.
     Without that cap a noisy puzzle grew a 120-word line, overflowing the
     int8_t rule index (a reading lists at most 127 rules): access
     violations in arc_said/arc_probe, seen only with puzzles in parallel.
     (2) moves answer to cells changed in place (answers_to), and (3) the
     cell a moved piece leaves may show anything but the piece itself.
     Numbers: 170 right, 0 wrong, gained 0, lost 0, explained but unsolved
     0 -> 0; a61f2674 and b230c067 reworded, still right; 16 unsolved
     readings say more, 6 less. Evidence: out/order3-said.txt,
     out/probe/order3/. A slow run then (82 s) was on a hot laptop.
     18286ef8 is still not found; it needs three more things: the move must
     claim the input cell its landing covers (now it ties ColorFrom's "the
     9 takes the 5 around it" at 6 cells, and the tie goes to ColorFrom);
     "6 becomes 9" is only an act, said before any move, so the 9 is gone
     first; and the 5-box fill behind the 9 is asked only of a still line.
- Evidence: out/wouldfit-said.txt (WouldFit alone: no change),
  out/cutchoice-said.txt and out/cutchoice2-said.txt (the choice before the
  fixes: 159 and 160 right, 6 wrong), out/fixes-said.txt, out/probe/fixes/.

## Completed: the ground owned once (central background, 2026-10-05)
- Agreed with the user: detectors see only objects. Ground cells are EMPTY
  (-1, nothing); empty is not a ground. Only the plan knows the background
  color: it deletes it first and fills it last. A detector that needs the
  color reads it from the plan. The user's earlier rule ("delete the ground,
  do what is needed, fill it back after"; both ways, keep whichever explains
  more) now lives in one place instead of 23 detectors and the frames.
- How it reads: `ground_plans(election)` (3_background.hpp) names the plans,
  each a pair (delete what first, fill what last); `figures_only` empties
  those cells; `read_figures` (7_infer.hpp) reads with no ground anywhere,
  and frames read their inside the same way (TRANSPARENT in and out, so no
  frame fills the ground in the middle); `infer` keeps the plan that leaves
  the fewest cells unsaid (a tie keeps the earlier plan; it stops at the
  first plan that leaves none); `say_the_plan` (9_sentences.hpp) writes the
  delete once at the start and the fill once at the end.
- Plans: 2.5 per puzzle on average. Chosen in 1000 puzzles: delete c, fill c
  765; keep everything 115; each input's own both ways 94; own, then fill a
  color 14; c, then fill another color 8; own, fill nothing 2; keep, then
  fill a new color 2. "Delete a color, fill nothing" was never chosen (0 of
  87) and is no longer offered.
- Deleted: the old modes (automatic/per_input/handed/fixed_color), the
  ground parameters of 25 detectors, ground-first and handed frames, the
  duality patch, the inference's cut/fill sources, `--central`, every
  `!is_color(ground)` bail-out, the hidden `has_output_ground()` bail-out
  (it blocked align/move/to-edge/shift/remove/outline replays), and the
  GPU's `ctx.background_color` (VoteDenoise uses `denoise_ground`; no hidden
  state). The search side (13_environment, 14_puzzle_loader, 16_solver_beam,
  the GUI) reads the one election, `inf.ground_election`.
- Empty as a value: the recolor table names empty (5_matching: slot 10 is
  empty, UNSET is -2; empty matches the ground only where the searched grid
  has empty cells, so raw grids read as before); match chance counted over
  the whole grid; VoteDenoise votes empty in its own slot (`vote_slot`);
  panes count empty lines as blank and a mask answer may leave cells empty.
  A swap may not trade a color that an earlier word paints (9565186b had
  claimed every cell while Recolor[5] then BidiColorMap[1,5] undid it).
- Numbers: 152 right, 0 wrong. Against the duality baseline (150 right, 1
  wrong): lost 0, gained 54d9e175 and be94b721; 44d8ac46, the 1 wrong, no
  longer fits. Explained-but-unsolved: 0. CopyTo 20/20; background checks
  14,000/14,000. A full score takes 581 s on this machine (plans are read one
  after another; about 5 minutes before this stage). ASan audit of the final
  reading: all 1000 puzzles, every plan, written and said, 0 errors (105
  minutes under the sanitizer). Its readouts match out/probe/cleanup/ in
  every reading; 87 differ only in the plan list (the dropped plan).
- Evidence: out/speed-said.txt (identical to out/cleanup3-said.txt),
  out/cleanup-tests.log, out/probe/cleanup/, out/final-audit.log and
  out/figures-audit/ (the audit's readouts).
- Removed afterward (user: too much; only the test puzzles stay): the ASan
  audit and the background checks (never in git), the CopyTo checks, the
  loader checks, and the truth lists (background, pane split, pane rule,
  pane programs), with their CMake targets and out\ scripts. The ones in
  git are in its history (HEAD 1a97d77). Older notes may still name them.
  What remains in tests/ is tests/puzzles (made-up puzzles for the app).
- Per-pair grounds are no longer the blocker: 57 puzzles are per-pair, 5
  solved (as before); the 52 unsolved name a recolor 28 times, where each
  copy was drawn 24, where each piece went 19.
- Open, not agreed:
  - (Re-exported 2026-10-06, see the "one measure" to-do item 5.) solutions/
    was not re-exported: 47 saved programs are unchanged, 103 are
    now said in other words, 54d9e175 and be94b721 have no file. Files for
    05a7bcf2, 10fcaaa3, 11852cab, 32597951, f25fbde4 (manual) and 44d8ac46
    are not inferred now.
  - f45f5ca7 was solved by an interim build, not now. The old reading never
    solved it, so it is not a loss; worth a look.
  - Speed: the plans could be read in parallel.

## Completed: the ground both ways through a frame (2026-10-04)
- User's rule: delete the ground, do what is needed, fill it back after; and
  keep both ways ("duality"), whichever explains more. Frames had broken it:
  they filled the ground back in the MIDDLE on the host, so per-pair frames
  were switched off, and a whole input was always scaled/voted with its ground.
- Whole-input frames (ScaleUp, ScaleDown, Tile[0], poses, Pattern) are also
  asked ground first: `Input Colors <ground> DeleteSelection UnionList SelectAll
  <frame>`, the figures alone framed, the ground filled only at the end by the
  existing final fill. Asked LAST in choose_piece_frame, so it wins only on a
  strictly larger share; a tie keeps the ground or the cut-out found earlier
  (asked first, it stole ties and rewrote 1cf80156, 7468f01a and 68b67ca3).
- Frames that delete the ground (cut-out, layout, fractal, ground-first) no
  longer switch off for per-pair grounds. The reading inside is HANDED each
  framed grid's ground and fill from its input (background_source::handed,
  infer's `handed` mode): no re-election from a host-filled grid, no ground
  words of its own; the outer program fills last. Fixed grounds unchanged.
- e57337a4 is now inferred: `Input Colors SelectElectedBackground
  DeleteSelection UnionList SelectAll ScaleDown[5] Colors Withhold
  CurrentObjects SelectAll Trim Union Input Colors SelectElectedBackground
  AddColor[11] Fit[1] Underlay`.
- Build passed; CopyTo 20/20, background checks 13,999/13,999: e57337a4
  end to end (including a ground no training input shows) and, on the first
  input of every training puzzle, each ground-first whole-input frame as the
  host draws it against the GPU program (7,401 new checks).
- Full score: 151 fit, 150 right, 1 wrong (44d8ac46), gained e57337a4, lost 0.
  The other 150 programs are unchanged. All 1000 readouts: explained-but-
  unsolved stays 1, none new; 8 readouts changed (e57337a4; handed cut-outs
  now hold in 20818e16, 25e02866, 7c9b52a0, 83eb0a57, c8cbb738; ground-first
  Pattern in 8e2edd66, e3f79277), all still lacking other words.
  Saved the inferred program to solutions/e57337a4.txt (156 active files).
  Evidence: out/duality-build.log, out/duality-tests.log, out/duality-said.txt,
  out/duality-score.txt, out/probe/duality/.
- OPEN, raised by the user: background handling is one central concept, yet
  it is threaded through 23 detectors (ground_in/ground_out parameters, fixed
  grounds only, so per-pair puzzles get none of them), the frames, replays and
  spelling -- and this stage added one more patch. Proposed next stage, not
  yet agreed: delete the ground ONCE from every input and output, read the
  figures-only puzzle with no notion of ground, fill ONCE at the end; read
  the untouched puzzle too and keep whichever explains more. Detectors would
  lose their ground parameters and today's frame patch would go. Needs an
  audit of how every detector treats empty cells; stage it with full checks.

## Completed: the odd pane, SelectUnique (2026-10-04)
- SelectUnique marks the one drawing in the list with no twin: same frame
  size and same cells, transparent ones included, wherever each one lies.
  Fewer than three objects, or not exactly one drawing without a twin, adds
  no mark; earlier marks stay. One ARC_HOST_DEVICE helper, unique_drawing in
  1_readings.hpp, serves both the GPU op and the reading (the pane pick's
  OddOneOut), so they cannot disagree.
- The odd pane is now said as `Input Panes SelectUnique Origin Union`: the
  whole pane, ground included, so nothing is removed or restored. That is how
  2dc579da, whose ground changes from pair to pair, is solved without any
  per-pair election. User approved the new op.
- Codex wrote and built it, then ran out of usage; Claude Code ran the checks
  and finished the stage. Codex's thread: 01a1016e-4a89-78a2-a0aa-381ce2e73a11
  under ~/.codex/sessions/2026/10/03/.
- Build passed; CopyTo 20/20, background checks 6,592/6,592 (new: SelectUnique
  cases, both odd-pane puzzles replayed, 2dc579da with a ground color it never
  shows in training).
- Full score: 150 fit, 149 right, 1 wrong (44d8ac46), gained 2dc579da and
  88a62173, lost 0. The other 148 programs are unchanged. All 1000 readouts:
  only those two changed; explained-but-unsolved stays 1, with none new.
  Saved both solutions (155 active files) and added both to
  tests/pane_program_truth.txt.
  Evidence: out/odd-pane-build.log, out/odd-pane-tests.log,
  out/odd-pane-said.txt, out/odd-pane-score.txt, out/probe/odd-pane/.

## Completed: Fit and all output-size policies (2026-10-04)
- Fit[0] stretches selected objects to input dimensions (the old default).
  Fit[1] uses the live output frame; Fit[2] uses painted output bounds and their
  offset. Sampling and target bounds are ARC_HOST_DEVICE helpers shared with
  host replay. Empty target bounds discard selected objects without selecting
  or painting surviving unselected objects.
- Identity, Fixed and Scale use Fit[1] for elected-background restoration;
  BoundingBox uses Fit[2], so filling does not expand the final crop.
  NotInferredYet records a missing frame rather than assuming input dimensions.
  output_dimension_rule::canvas_size_for is shared by host replay, the fixed
  upload context and CUDA. Invalid dimensions/ratios yield no frame, including
  zero denominators, oversize results and ratios that round to zero.
- Inference can compare local-ground readings for every inferred size policy.
  Replayed standalone explanations count as complete in that comparison;
  otherwise a partial background cut displaced 4c4377d9's complete block rule.
  Whole-input framed readings carry their elected fill through their prefix,
  targeting the outer frame. Fill is included before candidate program pricing.
- Sparse intermediate frames remain a separate boundary (see Next below).
- Build passed; CopyTo 20/20, background/size checks 6,570/6,570. Tests cover all
  size cases, nearest-neighbor grow/shrink, painted-bound offsets and empty
  paint, emitted fill against host replay, a framed fill, and every training
  pair of 140c817e, ea786f4a, 4c4377d9 and 23b5c85d.
- Full score: 148 fit, 147 right, 1 wrong (44d8ac46), gained 0, lost 0.
  All 1000 readouts checked: explained-but-unsolved remains 1, with none new.
  Refreshed accepted programs in place: 09629e4f spells Fit[0], 140c817e spells
  Fit[1]. All other accepted programs and verdicts are unchanged. 153 active
  saved solutions remain; no export backups. Whitespace check passed.
  Evidence: out/size-fit-build.log, out/size-fit-tests.log,
  out/size-fit-said.txt, out/size-fit-score.txt, out/probe/size-fit/.

## Completed: same-size per-pair inference (2026-10-04)
- background_source names a fixed color/no ground or an input-local election.
  Correspondence resolves it per input and keeps that color in each prepared
  pair; replay_facts carries each pair's output ground for every general word.
- infer compares the usual reading with an input-local-ground reading on
  same-size PER-PAIR puzzles. It chooses the latter only if it leaves strictly
  fewer cells unexplained. A tie keeps the existing all-color explanation.
  The alternative uses the general cut/word replay; older detectors that own
  fixed-color cuts remain in the usual reading, with their matching emitters.
- The chosen cut emits Colors SelectElectedBackground DeleteSelection before
  segmentation. Final restoration is Input Colors SelectElectedBackground
  AddColor[11] Underlay. Neither selection nor fill consults training at runtime.
- 140c817e now fits training and passes its test. An extra replay replaces its
  input/output ground with color 6, absent from its training inputs, and passes
  with the same program. ea786f4a and 4c4377d9 retain their programs.
- Resized puzzles, including their inner framed readings, retain the old route.
  Frame composition splices the inner cut but not its fill; enabling dynamic
  ground there without explicit restoration lost 23b5c85d. The boundary is now
  enforced by background_reading::fixed_color and covered by a regression check.
- Build passed; CopyTo 20/20 and background/inference contracts 6,507/6,507.
  Full score: 148 fit, 147 right, 1 wrong, gained 140c817e, lost none.
  All 1000 readouts checked: only the 30 chosen per-pair readings changed.
  Explained-but-unsolved stays 1, with no new cases.
  All 147 previous accepted programs/verdicts are unchanged. Saved the new
  solution to solutions/140c817e.txt (153 active files, no export backups).
  Evidence: out/per-pair-build.log, out/per-pair-tests.log,
  out/per-pair-said.txt, out/per-pair-score.txt, out/probe/per-pair/.

## Completed: shared election and SelectElectedBackground (2026-10-04)
- The selector elects from the current object layout, preserving its frame.
  It adds whole-object marks like SelectColor; abstention leaves marks alone.
  Explicit removal: Input Colors SelectElectedBackground DeleteSelection.
  A selected color layer can fill its frame with AddColor[11] then Underlay.
  No training pair, input snapshot, or puzzle background field is read by it.
- 3_background.hpp now shares ARC_HOST_DEVICE measurement, candidate scoring,
  ranking and confidence between host and GPU. Object/contour scratch is
  bounded by the 900-cell grid, with no device allocation. Across-pair choices
  remain on the host. Host per-pair election calls elect_grid_background.
- Inference does NOT emit the new selector yet. Per-pair cuts and explicit
  restoration are the next stage; do not remove ground globally just because
  the puzzle is PER-PAIR. Preserve 4c4377d9 and ea786f4a.
- Full score: 146 right, 1 wrong, 0 gained, 0 lost. All 147 accepted programs
  and verdicts, and all 1000 readouts, match the relative-AddColor baseline.
  Explained-but-unsolved remains 1, with no new cases.
- Build passed; CopyTo 20/20 and background/election checks 6,499/6,499.
- Corpus contracts compare the bounded election to the pooled host path and
  replay the GPU selector on all 3,232 training inputs. Also check abstention,
  additive marks, whole-object selection, changed paint, transparent margins,
  translated frames, deletion and relative fill.
- Evidence: out/elected-build.log, out/elected-tests.log,
  out/elected-said.txt, out/elected-score.txt, out/probe/elected-background/.

## Completed extension: relative AddColor (2026-10-04)
- User approved giving AddColor the same color arguments as Recolor:
  0..9 are literals, 10 least common, 11 most common, measured separately
  on each selected object. Ties pick the lowest color; an empty object has
  no relative color and stays empty. Resolve before changing any cells.
- The GPU uses gpu_obj_resolve_color, which calls the existing shared
  ARC_HOST_DEVICE object_paint_color helper. No second color-choice recipe.
- This is vocabulary support only. No emitter uses relative AddColor yet;
  per-pair background reading and SelectElectedBackground remain next.
- Build passed; CopyTo 20/20 and background/fill contracts 22/22. Full score:
  146 right, 1 wrong, 0 gained, 0 lost. All 147 programs and verdicts exactly
  match the explicit-background baseline. No solution files needed rewriting.
  Evidence: out/addcolor-build.log, out/addcolor-tests.log, out/addcolor-said.txt.
  All 1000 readouts checked in out/probe/addcolor-relative: explained-but-unsolved
  stays 1, no new cases.

## Completed: explicit background steps (2026-10-03)
- User chose consistency over search depth. The existing programs have been
  rewritten without changing their drawings. Per-pair election is next.
- Colors and Split now keep all real colors. Explicit `Colors SelectColor[c]
  DeleteSelection` removes a named ground; `UnionList` is used when the next
  operation needs the single framed input instead of separate color layers.
- InputWithoutBackground is retired: it fused a source and background removal.
- Output now preserves the whole canvas frame, including transparent margins
  and an entirely empty canvas. User explicitly approved this change.
- The kernel no longer fills output gaps automatically. Programs finish with
  `Output SelectAll AddColor[c] Underlay` when an output ground is elected.
  BoundingBox programs insert Trim before AddColor, so the fill stays inside
  the drawn bounds and cannot enlarge the final crop. No new operation needed.
- Emitter cleanup removes SelectAll immediately before Colors/Split and repeated
  ground removal only while the written operations prove that color absent.
  Recolor or any unrecognised act drops that proof; no runtime state is added.
- Background election itself has NOT changed. Fixed literals preserve the old
  readings; per-pair selection is the next stage. Keep ea786f4a's ground:
  its existing correct program deliberately uses a cut that keeps all colors.
- The fresh pre-change export is `out/solutions-export/said.txt`; its 147
  programs fit training, 146 right / 1 wrong. The user requested overwrite,
  not export backups. The 35 temporary export backups were deleted.
- A separate pre-existing full-build snapshot hook remains in scripts/backup.py.

- Verified full training run after the rewrite: **146 right, 1 wrong, 0 gained,
  0 lost** (147 fits). CopyTo 20/20; explicit-background contracts 16/16.
  All 1000 readouts checked: explained-but-unsolved stays 1, no new cases.
  Every accepted training drawing is still exact. All 146 correct test results
  are unchanged; 44d8ac46's wrong test prediction was also compared byte-for-byte
  with the old submission and is unchanged.
- Current score/programs: `out/explicit-background-said.txt`; checks:
  `out/explicit-background-tests.log`; readouts: `out/probe/explicit-background/`.
  `arc_background_tests` was the focused GPU contract target (Colors/Split keep
  ground, explicit delete/fill, Output frame, bbox fill, safe emitter cleanup);
  removed 2026-10-05.
- All 147 inferred programs overwrite `solutions/<id>.txt`. Five older manual
  training solutions also migrated and replayed successfully; their results
  are in `out/explicit-background-manual/`. User approved moving dbff022c, which
  is absent from training, into `solutions/shelf/`. Active folder: 152 files.
  The 35 pane reference programs in tests/pane_program_truth.txt were refreshed.

## Where it stands
- Training: 170 fit, 170 right, 0 wrong. Checks: the score (arc_said) and
  the readouts; no test programs remain.
- Current run (`out\` is not in git): `out/order3-said.txt` and
  `out/probe/order3/` (before 2026-10-06: `out/handed-said.txt` and
  `out/probe/handed/`). Last good baseline (152 right, 0 wrong):
  `out/speed-said.txt` and `out/probe/cleanup/`. Before the central
  background: `out/duality-said.txt` and `out/probe/duality/`. Before the
  explicit-background rewrite: `out/solutions-export/said.txt` and
  `out/probe/kaggle-after/`.
- What blocks the unsolved (`readouts.py missing order3 out/order3-said.txt`,
  named / the only word missing): where each copy was drawn 435 / 59; a
  recolor 397 / 122; where each piece went 225 / 42; how each piece turned
  215 / 14; the cells drawn on the ground 147 / 89; how a piece changed its
  shape 115 / 10.
- Before October 3: TileX and TileY unlocked, the repeat counted from the piece; a pick
  of every color but the most common; a paint laid over a piece explains the
  cells it covers; choosing the cut prefers one with at most 32 pieces per input
  (what a GPU list holds); rays and repeats laid under pieces (Underlay) or only
  onto them (MaskBy[10]).
- Still locked: the list at the top of `include/core/8_vocabulary.hpp`.

## Decisions and open questions from 2026-10-03
- Today's code/preparation changes were pushed to main as `1dcfc8b`; Kaggle
  scoring rules and the local evaluation baseline followed as `f6d59cf`.
  The 240 visible test tasks were checked: all are identical to training
  tasks. The 57 visible fits are not a measure of unseen-task performance.
- The next proposed solver work is the per-pair background problem below.
  The explicit-ops decisions above supersede the earlier proposal. Keep
  training as the development measure; today's evaluation was explicitly requested by the
  user, not a change to the standing rule or permission to tune on evaluation.
- Claude cloud preparation is in main, but an actual Claude cloud session
  has not been configured or verified. Connect this repo on main and set its
  environment setup script to `python3 scripts/cloud_setup.py`. It needs a
  C++20 compiler and network access to fetch public puzzle data when absent.
  Ready scope: CPU inference edits and readouts. Program execution, accuracy
  scoring, and CopyTo still need the local GPU or Kaggle.
- We considered a CPU executor to make a $100 cloud-credit offer more useful.
  The user decided it is not worth doing now. No CPU executor/prototype was
  implemented; do not start one as an assumed prerequisite for cloud work.
  Correction to the initial assessment: substantial CPU/GPU logic IS shared
  through ARC_HOST_DEVICE helpers in `1_readings.hpp` and `7_infer.hpp`:
  segmentation, matching partners, placement, rays, repeats, snapping, etc.
  The missing layer is full program execution: object selection/list state,
  canvas composition, and CPU equivalents for CUDA-specific parallel loops.
  `17_fixed_dsl.hpp` prepares data, while `18_dsl_executor.hpp` invokes CUDA.
  The interpreter mentions 76 operation types; accepted training programs
  use 67. These counts do NOT imply 67 separate algorithm rewrites.
  The quoted 2-4 working days / 2-4 hour prototype were rough estimates only,
  not measured work or a reliable dollar estimate.
- Latest user question before this handoff update: what the "project
  coordinator limit" in Claude resets. We verified what a coordinator does
  (directs worker threads), but NOT that specific limit's meaning or reset
  behavior. Asked for its exact wording/screenshot and reset tooltip; none
  received yet. Do not claim an ordinary usage reset clears every limit.
  Official reset guidance says an offer restores the session OR weekly
  allowance it specifies, not credit balances. The user's $100 offer terms
  and eligibility are also unverified. References:
  https://support.claude.com/en/articles/17007452-what-is-a-limit-reset
  https://claude.com/blog/projects-redesigned
  https://code.claude.com/docs/en/cloud-environments#setup-scripts
- No solver changes followed the completed audit/evaluation. Last verified
  training counts remain 146 right, 1 wrong, 0 gained, 0 lost; CopyTo 20/20,
  loader checks 17/17, explained-but-unsolved 1. Later work was analysis/docs.

## Kaggle and cloud preparation
- User requested a local evaluation and a fresh check of Kaggle's output rules
  on 2026-10-03. Commit `1dcfc8b`, inference only: **0/167 test outputs**,
  0/120 whole tasks. One task fit its demonstration pairs but failed its test;
  119 had no accepted program. All 120 entries valid, zero process failures,
  zero search, test answers withheld from the solver until external scoring.
  Evidence: `out/eval-one-shot/score.json`, `score.py`, `run.log`, and
  `kaggle/working/submission.json` inside that directory. This is the first
  recorded evaluation baseline here; gained/lost are not available.
  Training remains 146 right, 1 wrong, 0 gained, 0 lost; no solver edits.
  This supports a coverage/generalization problem, although the hidden zero's
  exact cause remains unverified. Fixing the JSON loader did not fix local eval.
  Kaggle counts exact matches per test OUTPUT, accepting either of two attempts;
  `arc_said` counts whole tasks. Both attempt keys, every ID, and ordered test
  entries are required. Our identical attempts are valid but offer no second
  guess. Official links and details are now in README.md's scoring section.
- Current source passed the private `mohamedelsharkawy0/arc-smoke-check` notebook
  on Kaggle's T4 / CUDA 12.8: three tasks, valid predictions, no child failures.
  This is a plumbing check, not a score or competition submission.
- Version 34 finished: 240 valid entries, zero child failures, 62 tasks fit
  their training pairs; 18,239.9 seconds solving plus 108.9 seconds compiling.
  It mistakenly used search. The user wants ONLY one-shot inference tested.
  Its files are in `out/kaggle-v34-result/`; do not treat it as an inference run.
- Version 35 was accidentally cancelled; the identical inference-only notebook
  was restarted as version 36 on 2026-10-03. Last checked: queued for L4.
  The user then requested T4: version 37 uses `NvidiaTeslaT4` and completed.
  The user reports stopping the older L4 run themselves.
  It sets `ARC_INFERENCE_ONLY=1`: each child uses `--infer-only`,
  emits programs, replays each on every training pair, accepts the first exact
  fit (the same selection as arc_said), predicts, and exits. No beam solver
  is constructed and no search loop or timeout runs in this path.
  Version 37: 93.0 seconds inference plus 108.3 seconds compilation; 240 valid
  entries, 57 training fits, zero search steps and zero child failures.
  Output is in `out/kaggle-v37-result/`.
  Submitted this exact version to `arc-prize-2026-arc-agi-2` on 2026-10-03,
  submission reference 56790276, output `submission.json`. Last checked:
  COMPLETE, public score 0.00, no error reported. Hidden run logs are not
  exposed; the zero's cause is not established.
  Claude cloud work remains deferred.
- Zero-score audit found a loader bug: training outputs before inputs were
  paired incorrectly, and spaces before field colons prevented loading.
  The loader now reads each example within its own object, independent of
  field order and whitespace. Three known solves failed with the old loader
  after harmless JSON reformatting; all three now fit in every tested layout.
  `arc_puzzle_loader_tests`: 17 checks pass; CopyTo: 20 checks pass.
  Full notebooks now reject child failures, just as smoke checks already did.
  This is a confirmed bug, not proof of the hidden zero's cause.
  Full audit: 146 right, 1 wrong, 0 gained, 0 lost. All 1000 predictions
  match the prior run with test answers entirely withheld; 147 fits, zero
  child failures, zero search. All 1000 inference summaries match; explained
  but unsolved stays 1, no new cases. Evidence: `out/submission-audit/` and
  `out/probe/submission-audit/`.
  T4 version 38 COMPLETE: 107.7s compile, 95.1s inference, 240 valid entries,
  57 fits, zero child failures/search. Predictions exactly match version 37.
  Outputs: `out/kaggle-v38-result/`. Submission was rejected: daily team
  allowance (1) already used. On October 3 Kaggle said to retry October 4 UTC
  (about 18 hours at that time; not a live countdown).
  Pending: exact notebook version 38 may be submitted after reset to test the
  loader fix on hidden data. It has NOT been submitted successfully. The later
  local evaluation still scored zero, so do not promise this retry improves
  the score; the proposed development priority is solver coverage below.
- The new inference-only runner passed all 1000 training tasks locally:
  exactly the same 147 accepted programs and verdicts as arc_said (146 right,
  1 wrong, 0 gained, 0 lost), zero search steps and zero child failures.
  All 1000 submission entries validate; CopyTo 20/20. Results:
  `out/inference-train.log`, `out/inference-train-submission.json`,
  `out/inference-only-said.txt`. Use a fixed executable for these checks;
  rebuilding arc_headless while it launches children interrupts the run.
- `python scripts/kaggle_push.py --smoke` runs the separate short notebook;
  upload source first with `python scripts/kaggle_push.py -m "message"`.
  The notebook now compiles for its assigned GPU and rejects invalid output.
- Port fixes: explicit equality for `seen_act` avoids a GCC 11 compiler crash;
  shared-memory spilling is enabled only on CUDA 13+. No rule changes.
- Rechecked: 146 right, 1 wrong, 0 gained, 0 lost; CopyTo 20/20;
  explained-but-unsolved stays 1, no new cases. All 1000 inference summaries
  match before/after. Files are `out/kaggle-original-said.txt` (before),
  `out/kaggle-after-said.txt` (after), and
  `out/probe/kaggle-before/`, `out/probe/kaggle-after/` (readouts).
- Linux CPU setup: `python3 scripts/cloud_setup.py`, then
  `out/cloud/arc_probe out/probe/cloud [id...]` from the repo root.
  `CLAUDE.md` imports AGENTS.md. This builds readouts, not GPU replay or scores.
  Tested on Kaggle's CPU Linux runner (GCC 11.4): three inference summaries
  match Windows (`00576224`, `44d8ac46`, `140c817e`). The actual Claude cloud
  session has not been started; its checkout must contain the setup files.

## Next: faithful intermediate frames
SUPERSEDED (2026-10-05) by the central background: frames now read with no
ground, so none fills the ground in the middle and nothing elects from a
host-filled image. Kept below for its history and its regressions.

Every output-size policy now has an explicit final-fill composition. A remaining
boundary is intermediate ground: cut-out, laid-out and fractal host frames can
restore ground that their DSL object stream does not contain. Their inner
readings stay fixed-color, and per-grid outer cuts decline those frames. Before
opening that route, make the intermediate fill an explicit composition and
replay the exact stream the next election sees. Do not elect from a host-only
filled image. Keep 23b5c85d, 4c4377d9 and ea786f4a as regressions.

Do not add another election, hidden fill, or removal mode to segmentation ops.
Continue to read current objects, use whole-object selection, and keep fills,
cuts and frame changes explicit in the program.

Agreed order (2026-10-04): finish this per-pair stage first, then recolors.
Work one example puzzle at a time with the user, as with 2dc579da, and first
try keeping the whole piece with its ground before removing and restoring it.
57 training puzzles are PER-PAIR: 5 solved, 52 not; 20 of those change size.
Most also lack other words (a recolor, where pieces went), so per-pair ground
alone will not unlock them all. Since the duality stage, cut-out, layout and
fractal frames run on per-pair grounds with the ground handed down, and no
frame fills ground in the middle; the central-background proposal above may
replace that plumbing.
Update 2026-10-05: the central background replaced it. Each input's own
ground is now a plan like any other, and the 52 unsolved per-pair puzzles
lack other words (above), so recolors are the next stage: agreed and started
(1acc24af with WouldFit; see the "one measure" section near the top).

## After that
- Recolors (the current stage, started with WouldFit): 118 unsolved puzzles
  lack only a recolor word. Most take the color from another piece (a key, a
  marker, a partner), so they are compositions of existing words
  (ColorFrom's partners, picks, frames), not one-property rules. Work one
  example puzzle at a time with the user (1acc24af was the first).
- Cells drawn on the ground: 93 unsolved puzzles lack only that.
- `python scripts\readouts.py missing <label> <said>` gives the current list.
