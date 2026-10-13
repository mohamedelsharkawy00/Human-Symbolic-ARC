# Handoff, 2026-10-13

Read AGENTS.md first: the rules, how to build and check, where things are.
This file holds only what is current; past stages are in git history
(`git log`, and this file's own history).

## Deadline: 2026-11-02
The user: not much time left. Prefer work that gains puzzles soon over long
redesigns; the big ideas wait in "Parked" below.

The plan (the user, 2026-10-12): training until 2026-10-25, aiming at about
350 right there, about 9 a day; then the last week on the evaluation set,
which is what Kaggle scores (the last evaluation run, 2026-10-10, fit 0 of
120). The trend, training right, the best of each day: 187 on 10-07, 199 on
10-08, 219 on 10-10, 224 on 10-11, 237 on 10-12 -- 5 to 13 a day, slowing as
the easier puzzles go; it gives about 310 to 350 by 10-25. All 1000 by then
would need about 59 a day.

THE WAY OF WORKING is AGENTS.md's (each puzzle with all of its adjacent
concepts). Open (the user, 2026-10-12): whether a batch of 50 consecutive ids
stays the unit of work. A run of 50 ids takes about a minute (62 s), its
readouts byte-identical to the full run's. Ids 1-50: 38 right; 51-100: 11
right (out/grow); out/batch2/unsolved.txt lists 51-100's unsolved before the
growth. Families of the unsolved that lack the same word (`readouts.py
missing`, out/grow): 72 lack only where a copy was drawn, 48 only the cells
drawn on the ground, 45 only a recolor.

## Paper: due 2026-11-08 (the user, 2026-10-12: let's try)
The ARC Prize paper track (kaggle.com/competitions/arc-prize-2026-paper-track),
written in the week after the Kaggle deadline, so the work comes first. A
Kaggle writeup of at most 1,500 words, a cover image and the public
notebook. It must name a competition submission: the last one before 11-02
is the one it describes (the last so far: 56790276, 10-03, 0.00). Marked 0
to 5 on accuracy (the Kaggle score), universality, progress, theory,
completeness and novelty, averaged. It reports the evaluation set and Kaggle
only, never the training count. Its main result: the words the evaluation
puzzles lack (`readouts.py missing` on an evaluation run), which the last
week reads anyway. Closest earlier work: Neumann and Pinter 2023 (no search;
each output object's properties read off the input objects), Ferre's grid
models (MADIL, 2025), Alford et al. 2021 (output objects bound to input
facts, with search).

## Parked (the user, 2026-10-10: good ideas, for later)
- THE READING ON THE GPU (the user, 2026-10-12; parked until after the
  deadline). The time is in thousands of small checks -- each pick, length
  and lay of a paint, each word's replay, grids at most 30 by 30 -- the kind
  of work a GPU does, while it sits idle as the reading runs, and the ops'
  arithmetic is shared with the device already (8b_op_math.hpp). But the
  reading around them is greedy and round by round -- claims, maps, passes --
  so only the checks could move, a round's at a time: about 200,000 calls a
  full run, 20 threads on one GPU. Weeks of work, and it speeds the runs here,
  not the Kaggle score (about 3 CPU seconds a puzzle).
- THE REST FIRES ON THE REST, EVERYWHERE. Today a word reads the hand (the
  cut's pieces carried through the words before it), a paint reads the cut
  opened again, and only a recolor read off the picture (picture_word, below)
  sees what a paint drew. The general form: after every step, cut the picture
  drawn so far again, and let every word and every paint read it -- the
  picture as the line's one state, instead of the hand beside the canvas. A
  word that moves or removes a piece then has to erase its old cells in the
  program: Union and Underlay paint, Intersection and Difference erase
  (8_vocabulary.hpp). Costs: every word's replay reads the cut-again picture,
  and many readouts move. The passes after the first (below, 2026-10-11) now
  do this a whole pass at a time, one change each; this is the same inside
  one line, without a pass for every step.

## Where it stands
- 2026-10-13: EVERY PLAN READ, IN THE READOUT; THE EVALUATION SET AGAIN (the
  user, from the session's tips: keep the per-plan print; read the
  evaluation set now and then, not only after 10-25). [plans read], under
  [ground plans]: each plan read, in order, exact words or words that may lay
  cells wrong, its cells unsaid, its passes and its line's words
  (inference::plans_read, filled by infer; plan_said in 10_log, the one way a
  plan is said). Results unchanged. Evaluation (ARC_DATASET=eval, out/
  eval_1013): still 0 of 120 fit; the elected reading leaves under 5% of its
  cells unsaid in 22, under 10% in 40, under 25% in 63 (the nearest: 8e5c0c38
  11 cells, 88e364bc 10, 409aa875 22, 135a2760 10); today's words appear in
  its programs (the growth in 4, the counts in 5, the one with no twin in 3)
  without finishing one. Missing (readouts.py missing): where a copy was drawn
  101 (8 only it), a recolor 73 (3), how a piece turned 63 (4), where it went
  51 (3), how it changed shape 47 (2), cells drawn on the ground 11 (1).
- 2026-10-13: COUNTING (the user: 2c0b0aff the most crosses, 358ba94e the
  most holes, 39a8645d the most of one color -- the reading cannot count).
  One owner of what a piece counts, count_of (8b_op_math.hpp): its cells,
  colors, holes (object_hole_count) or cells of one color (count_kind); and
  each_extreme, every one with the most or the fewest, ties and all, none
  where all count the same. SelectLargestObject and SelectSmallestObject
  keep their ranks by size [0-9] (SIZE_RANKS) and past them pick every piece
  with the most or fewest of a count [10-22]; SelectUnique past its
  likenesses reads twins by a count [4-16] (twin_by::by_count, twin_key,
  lone_one: the host reads a key once a piece, the GPU pair by pair). The
  reading asks them last of all picks (the newest last, an older pick taking
  a tie), and a count of one color's cells only where some piece mixes that
  color with another: among one-color pieces it is their size, and asked
  there it peeled a61f2674's bars off one at a time and lost it (found with a
  temporary per-plan print, since removed). 246 right, gained 358ba94e (the
  odd one out by cell count), 54db823b (fewest 9s), 8efcae92 and e50d258f
  (most 2s), 9a4bb226 (the odd one out by colors), de1cd16c (most holes),
  lost 0; explained but unsolved 0 -> 0; 6 wrong, caa06a1f joined (fewest
  holes, read off its pairs). The data on the user's three: 2c0b0aff is the
  most green crosses in every pair and the test, not the most green cells
  (pairs 1 and 3); 358ba94e is the odd one out by holes, not the most (pair
  2's has the fewest); 39a8645d is the color with the most pieces. Still
  open: a count of copies of a small shape (crosses) and of pieces a color
  has (39a8645d, whose pick takes copies a frame must take one of). GPU
  kernel 80 registers, 71,552 bytes stack; 71 #20014 warnings. Run
  out/counts2, against out/unique (no timing: a game running).
- 2026-10-13: THE ONE PIECE WITH NO TWIN, A PICK (the user: f25fbde4 is
  plain x2 scaling, suspicious; 13 unsolved size changes whose every output
  is one input piece as it lies, read with this session's own cut).
  SelectUnique[how] (0-3): the one piece of the list with no twin, twins by
  drawing [0], shape as it lies [1], any way round [2], or colors [3]
  (unique_piece and twin_by, 8b_op_math.hpp, the op's and the reading's one
  arithmetic; unique_drawing gone into it). A pick kind, piece_pick::only_one,
  asked where it takes a piece in every pair, as it is and turned round, so
  the frames, the words and the paints all read it. SelectUnique took no
  value before: the pane rule's sentence and solutions/2dc579da, 88a62173 now
  say SelectUnique[0]; 88a62173's saved program no longer fits by arc_spell
  (its plan now deletes the ground, the panes cut by blanks), the puzzle
  solved the new way. 240 right, gained 1a2e2828 (SelectUnique[3], the only
  piece of its colors) and cd3c21df (SelectUnique[0]), lost 0; explained but
  unsolved 0 -> 0; the same 5 wrong. An over-claim corrected: the table that
  promised up to 11 of the 13 checked only that the output piece was one of a
  kind, not that it was the only one; held strictly, cd3c21df alone of them
  is (48d8fb45's marker dot is one of a kind too; its piece is the one the
  marker touches). The rest of the 13 lack other picks. f25fbde4 still
  wants the frames chained: a piece cut out, then scaled (choose_piece_frame
  reads one frame at a time). Run out/unique, against out/boxrule; timing
  not taken (a game running).
- 2026-10-13: THE SHAPE WORK'S COST, FOUND AND CUT (the user closed the game:
  profile, do all and anything else found). Measured with nothing beside
  it: e681b708 alone took 283 s (49 s on 10-12), every tenth id 401 s by
  the clock (63 s). VTune (profiling/output/cpu-20261013-025320-858): 67 of
  300 s in blocks_of, every piece's blocks read again for every pair a
  likeness at another size compares, and the partner asks grown with the
  ways (ColorFrom 20 values, the L bridge's kinds, WouldFit's, the
  likenesses). Three fixes, results unchanged by the first two: (1) each
  piece's blocks read once (partner_pieces::blocks, the GPU's view reads them
  when asked; same_shape_in_blocks takes them); (2) a likeness at any size or
  stretched is asked only of a hand that holds one shape at two sizes
  (replay_facts::shape_at_two_sizes, asks_way/asks_kind/asks_likeness, read
  again once carry_the_hand changes the hand), and a Scale reading is read
  only on a cut where a match at another size can be found
  (any_match_by_size), else it is its twin without Scale; said.txt and all
  1000 readouts byte-identical (out/speed, 834 s). (3) A shape whose smallest
  drawing is one cell, a dot or a filled box, has no size but its own
  (sized_blocks, the shape owner's): 528 puzzles hold filled boxes of two
  sizes, every one asked the likenesses at another size for that alone; the
  matching's private `scales` flag is gone into it. 238 right, gained 0,
  lost 0; explained but unsolved 0 -> 0; 5 wrong, fea12743 left it (a key
  "stretched" matched boxes by luck). Full run 779 s (out/boxrule; 558 s on
  10-12). e681b708 alone 60 s, every tenth id 88.6 s by the clock and 526 s
  of CPU (414 s on 10-12): the clock's slowest puzzle is 60 s of it, so the
  workers seen waiting at a run's end are the tail, not a lock. Tried and
  dropped: the two paint caches as hash maps -- alternated A/B on e681b708
  and 92e50de0, 94.8-95.1 s against 93.4-93.5 s for the maps, 1.5% slower.
  An earlier 150 s timing of e681b708 on this build was the machine, as
  10-12's 142 s against 63 s was: time A/B in turn. GPU kernel unchanged (80
  registers, 71,552 bytes stack); 71 #20014 warnings, all host pieces asked
  through the shared helpers. Left of the cost, as on 10-12: malloc 13%,
  the cell lists of every lay that may hold (NEXT (1) of "where the time
  goes" below).
- 2026-10-13: COLORS CELL BY CELL FROM A PIECE OF ITS SHAPE -- stage 4, as
  ColorFrom (the user's pick of the owners: CopyTo fitted, ColorFrom cell by
  cell, or small ops chained; CopyTo fitted waits for a frame to fill).
  ColorFrom[14-19] (COLORS_FROM_ITS_SHAPE): each picked piece takes its
  colors cell by cell from its one partner of its shape, each likeness in
  turn (as it lies or turned, at its size, any size, stretched);
  colors_from_its_shape (8b_op_math.hpp) is the op's and the replay's one
  arithmetic, laying the two shapes one on the other as same_shape does and
  reading the partner's cell each cell lies on (shape_fit, fit_place in
  1_readings.hpp). The ways round are asked turns before flips, rights
  before lefts (pose_asked): 103eff5b's pair 0 lays its key on the azure
  shape by a quarter turn and by a flip, and only the turn gives its colors.
  posed_place moved to the pose owner (1_readings.hpp). The recolor replays
  repaint cell by cell (cells_repainted_hold; repaints_hold is the colour-map
  case of it), and carry_the_hand carries the new values. 238 right, gained
  103eff5b (`SelectColor[8] ColorFrom[17]`, its test right), lost 0;
  explained but unsolved 0 -> 0; the same 6 wrong; e76a88a6, solved before,
  is now said with ColorFrom[14]. GPU kernel 80 registers, 71,552 bytes
  stack (+16); 61 #20014 warnings (+4, the same kind: host pieces asked
  through the shared helpers). Full run 1947 s, a game beside it; at the
  tail 5 to 10 of 20 workers ran and the rest waited (Windows: "Unknown",
  most likely a lock, the allocator's by the profile's malloc share) -- to
  confirm with VTune. Run out/cellwise, against out/scale.
- 2026-10-13: SHAPE, stages 2 and 3 (the user: all but stage 4, and test
  before it). (2) The rest of the shape questions ask the owner: a drawing is
  a shape whose colors may not change (shape_freedom::recolor, DRAWING;
  cells_alike), and key_of_shape is the one key (canonical_shape_hash gone):
  the shape table (shape_key_of), the segmentation's cost (shape_of), the
  interior fill's table, the panes' and a cut's same_drawing, unique_drawing
  (SelectUnique) and the partner ways. said.txt and all 1000 readouts
  byte-identical to stage 1 (out/shape2). (3) The matching sees size:
  degrees_of_freedom's fifth freedom, Scale (32 readings a cut, so a
  reading's place is correspondence_election::reading_index, 16 bits: at 8
  bits every puzzle crashed); an edge's scale; Source 3 in gather_matches, a
  piece of the other grid whose shape at any size (object_info::scaled_shape)
  is this one's grown a whole number of times, checked cell by cell with its
  footprint, only for a piece whose smallest drawing is more than one cell
  (scales: no dot, no filled box); each_carried_cell lays a cell on its whole
  block; landing_size owns an edge's box; the new missing word "how each
  piece was scaled"; no act (turn, move, recolor) and no rule reading edges
  claims a scaled piece. 237 right, gained 0, lost 0; explained but unsolved
  0 -> 0; the same 6 wrong. Scale changed a matching in 16 puzzles (3b4c2228,
  447fd412, 57aa92db, 60c09cac, 6aa20dc0, 6b9890af, 80af3007, 8a004b2b,
  8a6d367c, 9172f3a0, ac0a08a4, b4a43f3b, b91ae062, d4b1c2b1, f0afb749,
  f25fbde4); 9 elected readings name the new word, none lacks only it.
  103eff5b is not among them: its elected plan keeps the black ground, where
  the key's blank cells land on black, and the plan that deletes it, where
  the scaled match is, says no more without a word for the scaled copy.
  GPU kernel unchanged (80 registers, 71,536 bytes stack); the same 57
  #20014 warnings. Full run 1909 s (stage 1's 1288 s, both with a game
  running): twice the readings a cut, so likely slower in fact; to measure.
  Run out/scale, against out/shape2.
- 2026-10-13: SHAPE, ONE OWNER, AT ANY SIZE -- stage 1 of 4 (the user: a
  shape is the same whatever its colors, turns, flips and size, and that runs
  through the whole solver; stretch too). 1_readings.hpp owns it: blocks_of
  (the most rows and columns a shape's cells repeat in; shrunk by them, its
  smallest drawing), shape_freedom (turn, scale, stretch; colors never enter
  a shape) and same_shape (two shapes the same as a freedom lets them
  differ). The partner ways gain the same shape at any size and stretched,
  each as it lies or turned (partner_by, now 13; SHAPE_LIKENESSES,
  likeness_freedom in 8b_op_math.hpp), so ColorFrom, WouldFit, Bridge's L,
  AlignTo, MirrorAround, MoveToward and SelectLikeKey (6 likenesses) read
  them; the old two ways ask the owner and find what they found. Values
  renumbered: ColorFrom 0-13 (13 the key at its place), WouldFit 0-12, Bridge
  0-31, AlignTo 0-25, MirrorAround 0-41, MoveToward 0-68, SelectLikeKey 0-5;
  the 7 solutions/ files that hold them renumbered, each still fits. 237
  right, gained 0, lost 0; explained but unsolved 0 -> 0; 6 wrong: fea12743
  joined (unsolved before: a piece like the key, stretched, fit its pairs by
  luck). GPU kernel unchanged (80 registers, 71,536 bytes stack), the same 57
  #20014 warnings. Run out/shape, against out/grow (1288 s, a game running).
  NEXT, the user's order: (2) the rest of the shape questions onto the
  owner, results unchanged -- canonical_shape_hash (the shape table, the
  matching's shape search, the segmentation's cost), same_drawing (panes, a
  cut's repeats), unique_drawing (SelectUnique) -- with colors a freedom of
  the owner, a drawing being a shape whose colors may not change; (3) the
  matching gets a scale freedom beside its turn, so the reading sees "this
  matches that, x3" (the user: then 103eff5b is said many ways, not only by
  a copy); (4) CopyTo fitted onto a partner of its shape (103eff5b: the key,
  turned, grown to the azure shape, laid onto it).
- 2026-10-12: GROWN IN STEP UNTIL THEY MEET (the user: 0f63c0b9's partial
  program lacks an extend at the edges, a generality issue; keep it general,
  not only grid edges). ExtendTo takes four values past its eight ways, [8-11]
  (grown_along, 8b_op_math.hpp): every picked piece grows in step with the
  others, a step a turn, along its rows, along its columns, to every side
  through the open cells, or along the canvas's edge, until they meet; a cell
  goes to the first piece to reach it, a tie to the earlier piece in the list,
  and a cell any piece holds is none to grow onto. One owner, grow_in_step
  (8b_op_math.hpp), run by the GPU op (gpu_exec_grow_in_step, its scratch
  gpu_exec_context::grow_scratch) and by the replay (grown_walk, under
  trail_walk); asked after the eight ways, which take a tie, and said "grows
  along its rows in step with the others until they meet". Why not the other
  owners (read off the op list): SelectColumn picks whole pieces by band, not
  cells; one ExtendTo way runs on to the next piece, so it never splits a gap;
  ColorFrom's nearest refuses a tie and recolors only pieces there already;
  Bridge joins one color; rays start mid-piece. 0f63c0b9 is now its lines
  (ExtendTo[8]) and, a pass on that picture, the border (ExtendTo[11]): the
  copies that drew its top and bottom rows by luck (its bottom row wrong on
  the test) are gone. 237 right, gained 0f63c0b9, 1bfc4729, 782b5218 (wrong
  before), 9edfc990, e69241bd, fc754716, lost 0; explained but unsolved 0 ->
  0; 5 wrong: d4f3cd78 joined -- its fill leaks out of the box's gap, up in
  one pair and down in the other, said as "along its columns", and leaks right
  on its test: the word it lacks is the leak through the gap -- 782b5218 left
  solved, and 29623171 and ac0c2ac3 no longer fit. Ten programs use the new
  values (178fcbfb, 21f83797, bdad9b1f solved before and still). GPU kernel:
  80 registers, 71,536 bytes stack (+32); the same 57 #20014 warnings. Full
  run 1250 s with a game running beside it, so not comparable with 558 s:
  time old and new on a few slow puzzles when the machine is quiet. Run
  out/grow, against out/picks_after_cold.
  ALSO FIXED, readouts.py: its pattern for the passes after the first ate the
  next pass's header, so it skipped every second kept pass and judged 49
  readouts on an earlier one (`missing` and `explained`); explained but
  unsolved is 0 either way.
- 2026-10-12: WHERE THE TIME GOES, MEASURED (the user: speed, pure speed;
  stopped here; nothing in the reading changed). Each puzzle timed in
  arc_said -- its reading by the clock and by its thread's CPU, and the GPU
  lock's wait and hold, into timing.txt -- kept back as
  extras/said_timing.patch (against scripts/said.cpp; extras\ is not in
  git). Every tenth id, 20 threads (out/timing_base2): 63 s, the readings
  460 s by the clock, 414 s of CPU.
  THE GPU CHECK IS NO QUEUE: its lock held 1.07 s in all, the longest wait
  0.16 s. A FEW PUZZLES TAKE THE TIME: the ten slowest are 61% of the
  readings -- e681b708 54 s, 92e50de0 36, 256b0a75 35, 6d0160f0 31, 543a7ed5
  29, ea959feb 24, ad173014 23, 9edfc990 20, b782dc8a 16, 7837ac64 15 -- and a
  sample's clock is its slowest puzzle's. THREADS COST LITTLE: e681b708
  takes 49.2 s alone, 54 s among 20 threads.
  THE MACHINE: the same ids and build took 142 s at 19:28 and 63 s at 19:42,
  every readout the same; e681b708 took 132 s then, its last 66 s with no
  other puzzle running, so something outside the reading slowed it (heat or
  another program, not known). Asking Windows not to throttle the process
  (SetProcessInformation, ProcessPowerThrottling) changed nothing, 64 s
  against 63 s, and was taken out. One thread repeats: e681b708 49.16 s, then
  49.15 s. So the paint-sharing run Codex stopped (bars, filled hollows and
  boxes walked once; reverted when its usage ran out; out/paints_shared,
  half done) may have looked twice as slow for the machine alone: never
  measured fairly. Back-to-back runs heat the laptop too (the user), so
  compare old against new on a few slow puzzles, ARC_THREADS=1, old, new,
  old, new, the best of each; check the readouts on the tenth sample.
  e681b708 PROFILED ALONE (VTune, 45 s; profiling/output/cpu-20261012-193314-382;
  inclusive times from its callstacks.csv by extras/inclusive.py): the lines
  85%; laying a paint cell by cell (lay_paint) 28%, the cell lists of every
  lay that may hold, built, weighed once and dropped (their push_back 12%);
  malloc 14% of all; the replay cache, a std::map keyed by tuples, 5% in
  lookups and 4% torn down at each line's end; the walks 15% (bars 4.6, rays
  4.4, repeats 3, copies 1.4, trails 1); matching 13%; the key words 11%
  (key_swap_holds 7, key_place_holds 4); the passes after the first 27% of
  it all.
  NEXT, NOT STARTED: (1) a lay weighed by marks and counts alone, its cell
  lists made only for the paint each kind offers (the same answers by
  construction; 20% on the slow puzzles is a guess); (2) the replay cache a
  flat table; (3) the strict and tolerant readings of one plan share its
  matching.
- 2026-10-12: PICK LISTS REUSED ON THE UNCHANGED HAND. finish_the_line keeps
  picks_on_hand, keyed by colors and fit_colors, and clears it in carry before
  carrying a word. picks_to_ask sorts each pair's sizes once per call, reused
  for both directions and every rank. No picks, order or replays removed.
  Fresh full GPU runs, both ARC_THREADS=20: out/picks_before_all versus
  out/picks_after_cold. Both 231 right, 7 wrong; gained 0, lost 0; explained
  but unsolved 0 -> 0. All 1000 readouts and said.txt byte-identical.
  Sampled process totals: wall 581.95 -> 556.77 s, CPU 5119.66 -> 4849.19 s
  (5.3% less); peak working set 5509394432 -> 5831999488 bytes (5.9% more).
  The user paused before the edited run to start cold; this is one comparison,
  not a controlled repeated speed measurement. The aborted runs are excluded.
- 2026-10-12: A WORD MAY LAY CELLS WRONG WHERE IT BRINGS EVERY PAIR NEARER
  (the user: rejecting a paint for one wrong cell is too early; bdad9b1f,
  two bars grown to full lines whose crossing is a third color). One judge
  for every word, brings_nearer (7_infer): paints (paint_holds), moves
  (slides_show), removals and outlines (drops_hold) tally, pair by pair, the
  cells they make right (gained: the output shows what was laid there and it
  was not there before) and the cells they lay wrong; a word holds where it
  gains more than it lays wrong in every pair. Its wrong cells are left
  unsaid: the reading's drawn_wrong, counted in what is left over (and
  printed as "drawn wrong" under it), drawn in the picture the next pass
  reads (`painted` keeps every paint's cells in the order said, right or
  wrong), and unclaimed from any word whose claim the picture no longer bears
  out; weigh nets them off what a word says. A paint laid in one color takes
  the color the output shows most where it lands. Paints may be laid only
  onto or off one color of what is down (MaskBy[c], MaskOut[c]; mask_keeps in
  8b_op_math is the ops' and the replay's one arithmetic; the canvas keeps
  colors), and trails and copies are laid every way rays and repeats are.
  Whether words may lay cells wrong is a choice infer makes
  (inference::tolerant): every plan read with exact words first, then with
  tolerant ones, a tolerant reading kept only where it leaves fewer cells
  unsaid, passes and all -- always tolerant, 332efdb3 was lost to "every
  cell becomes 1", which left its holes wrong and crowded out the rays that
  solve it. says_the_most breaks a tie by fewer passes (passes_said):
  bdad9b1f's tolerant reading in one pass over five passes of mirrored
  copies that drew its pairs by coincidence. 231 right, gained 23581191,
  319f2597, 41e4d17e, 6f8cd79b, bdad9b1f, d90796e8, lost 0; explained but
  unsolved 0 -> 0; 7 wrong: bdad9b1f left the list, and 782b5218, ba26e723
  (tolerant readings) and 29623171, 833966f4, ac0c2ac3 (exact, trails and
  copies laid under) joined it. 209 readings keep the tolerant words. Run
  out/fast, against out/ranks.
- 2026-10-12: SPEED KEPT BACK, AND WHAT IS ONLY SUSPECTED (the user: back to
  the safe checkpoint, every speculation and suspicion written down). The
  code is what out/shared ran: 558 s, 231 right, 7 wrong.
  KEPT BACK, saved as extras/speed_kept_back.patch (against this
  checkpoint's 7_infer.hpp; extras\ is not in git): bars, filled
  hollows and filled boxes walked once per cut like rays and laid on each
  canvas (bridge_walk, fill_walk, box_walk, with own_else_one the one helper
  for "in its own colors, or else in one"); the picks the words ask read off
  the hand once until a word changes it (picks_on_hand); and a pick that
  takes the same pieces as one asked before it, in every pair, not asked
  again (taken_before) -- every paint, bar and word, each thing a word asks of
  some picks only keeping its own record (a move's step, a recolor's color, a
  removal's may). Its full run, out/dedup, read every puzzle the same as
  out/shared. On every tenth id, the two versions run in turn: about 30% less
  CPU (306 to 308 s against 428 to 446 s) and 47 to 51 s against 64 to 71 s
  by the clock. But its full run took 698 s, against 558 s for the
  checkpoint at a quieter time, and its memory was never measured: less CPU
  may come with more memory or thrashing (the user). Not settled: a full run
  of both, one after the other, with time and memory, would settle it (one was
  begun and stopped at 744 of 1000, for the checkpoint).
  THE GPU QUEUE, RULED OUT (measured that evening, "Where the time goes"
  above): arc_said's one lock on the GPU checks (run_what_it_said) was held
  1.07 s in all over the tenth sample's 63 s, the longest wait 0.16 s.
  THE MACHINE, MEASURED (the same entry): runs at different times do not
  compare by the clock -- the same ids and build took 142 s, then 63 s. Over
  the day the NVIDIA Overlay used about 8,300 s of CPU and Edge (YouTube)
  about 4,700 s. Compare on a few slow puzzles on one thread, old and new in
  turn; a timing probe of its own slowed a run by its clock reads (it read
  the clock some 400 million times).
- 2026-10-12: A PAINT WALKED ONCE, LAID EVERY WAY (the user: the cost is
  upstream of the lines the profiler shows). Timed by layer on the 20
  slowest puzzles: about 88 line readings each, 61% of the time in the
  tolerant ones, and in each line 90% in the paints, every lay a replay of
  its own, again whenever the canvas changed. Now a paint is walked once
  (walk_paint: each cell it reaches, the color laid there first and last,
  whether two of its own disagree, what was there before) and laid from the
  walk every way, on every canvas (lay_paint, each_landing); rays, trails,
  copies and repeats keep their walks for the line (laid_once), and
  paint_holds is a walk laid one way. Also behind it: the picked pieces'
  drawing and the ray seeds drawn once per pick (partner_pieces::shown,
  ray_seeds), ray lengths past the grid read once and repeats too far apart
  for a second copy not read, claims looked up in a cell grid (cell_marks).
  Full run 818 s (1204 s before, 255 s before the tolerant readings);
  said.txt and every readout the same. Then a walk is counted once for each
  canvas it is laid on, by what lies under its cells (walk_tally), and each
  lay judged from the counts (may_hold), only a lay that may hold laid cell
  by cell: 580 s. Then the cut's paints shared by every line read on the cut
  (cut_paints, read_the_cut): 558 s, about 4% of CPU. Each the same said.txt
  and readouts. Run out/shared. Tried and dropped: words' replays kept until
  the hand changes (about 2% of CPU); one reading a plan, strict words first
  and loose ones only where no strict word says anything (lost 23581191 and
  332efdb3, and slower, 639 s: the loose words at a line's end leave the
  passes a worse picture). Measured by CPU time on every tenth id, two
  versions run in turn (wall time on a sample is its slowest puzzle).
- 2026-10-12: A RANK BY SIZE COUNTS SIZES, NOT COPIES (the user: 272f95fa,
  wrong). Its center pane was said by "the fifth smallest", which took the
  tie from "part 4" (sizes are asked before places) and landed on the center
  in both pairs only past three panes of 8 cells; on its test three panes of
  18 put another pane fifth. picks_to_ask now asks a rank by size only where
  no pair's pieces tie at that place nor anywhere before it. 225 right,
  gained 272f95fa (part 4 says the center), lost 0; explained but unsolved
  0 -> 0; 3 wrong: 7d1f7ee8 left the list and no longer fits (its fit leaned
  on "the third smallest" past pieces of one size; its word, a frame's color
  at any depth, is still missing). Run out/ranks, against out/x64-Release,
  the same code before it (out\ was wiped on 2026-10-12: the older runs are
  gone).
- 2026-10-11: LONG PROGRAMS FOLD IN THE GUI (the user: 31aa019c's program ran
  off the screen). One toggle, show_whole_programs (ui_action
  toggle_programs, also on the command port): folded, CURRENT and BEST keep
  the footer's four lines, the last box "... N MORE" (fold_of, the one answer
  for the reflow and the draw), and the facts pane shows a program's first 6
  rows, "... N MORE", each row cut to "..." at the frame; SHOW ALL on the
  pane's heading, or the box, opens them, FOLD folds them. Not saved in the
  settings, so the app starts folded. Checked by screenshots over the port
  and by mouse clicks posted to the window (from a DPI-aware process: the
  window is per-monitor aware, so an unaware sender's clicks land elsewhere).
  And the freeze the user saw (Windows logged ARC.exe hung): the pane read a
  puzzle under the lock the frame draws under, seconds where it reads its
  passes (f823c43c about 4 s). Now infer runs outside it and
  take_the_reading (14_puzzle_loader, the rest of infer_into_puzzle) under
  it; the port answers within 0.03 s while f823c43c is read. A second freeze
  (the user: the third click): with the programs open the grid band shrinks
  to nothing, and the hover preview sized its rows off it, a negative height
  that vkCmdDraw's unsigned instance count read as four billion quads.
  vulkan_context::draw now draws nothing for a size not above zero, and the
  preview skips a row with no room; checked with the solver running, a box
  hovered and the preview up, open and fold three times over, the port
  answering within 0.02 s.
- 2026-10-11: A KEY FENCED OFF, AND THE PIECES LIKE IT (the user: the key
  is the shape in its sector; something that meshes with Partition and
  Layout, "select the nth partition", numbered like rows that wrap). One
  owner of the parts walls fence off, fence_parts (8b_op_math): the cells
  off the walls joined at a side, numbered in reading order by each part's
  first cell; walls across the whole grid fence a lattice's tiles, row by
  row. parts_the_pick_fences takes the picked pieces' cells as the walls.
  Partition now cuts on these parts, any shape, not only lines across the
  grid (no program said it). Two new ops: SelectPartition[n], the picked
  pieces as walls, the pick becomes the pieces lying wholly in part n
  (lies_in_part); SelectLikeKey[how], of the picked pieces only those like
  a piece of the key -- every piece not picked, as the key words read it --
  stay picked, [0] the same shape as it lies, [1] any way round
  (like_the_key, on the partner owner's two shape ways; any_partner is the
  one "a partner at all" question, partner_pieces::partnered asks it too).
  piece_pick gains `part` and `like`; picked_pieces takes the canvas, since
  a part needs the grid's size (replay_facts::canvas, handed_pieces,
  carry_the_hand, the frames). picks_to_ask asks, last: for each color every
  pair holds whose walls fence 2 parts at least in every pair, each part
  that holds a piece in every pair, and every piece but those; then
  SelectLikeKey on every pick that turns its pieces round, where it keeps
  one piece in every pair and drops one in some, a fenced key first (the
  surer key: unfenced, 63613498 said "off the column 3 left of the middle",
  which left its key and an 8 besides, right on its test by luck).
  63613498: `SelectColor[5] SelectPartition[0] InvertSelection
  SelectLikeKey[0] Recolor[5]`. 224 right, gained 31aa019c (a pass after
  the first deletes part 0 of the 2 ring it drew), 332efdb3 (rays from part
  0), 63613498, lost 0; explained but unsolved 0 -> 0; 5 wrong: 63613498
  left the list and 272f95fa, unsolved before, joined it: four of its five
  panes said by part, the center by "the 5th smallest", the center in both
  pairs and not on the test (an older pick takes a tie, and sizes are asked
  before places, bands too). Full run 135 s (255 s last time). GPU kernel:
  80 registers, 71,504 bytes stack (+1,104); 4 more #20014 warnings of the
  same kind. Run out/partition, build log out/partition-build3.txt.
- 2026-10-11: THE REST, READ ON THE PICTURE SO FAR (the user: after the
  partial program, a second inference on what it drew, building on the output
  so far; it was parked as "re-run the inference on the rest"). read_the_rest
  (7_infer): once read_figures has chosen a pass's cut, where its line leaves
  cells unsaid and draws something new, the picture it draws (finish_the_line
  keeps it as inf.picture: the hand as the words leave it, every painted cell
  in the output's color) is the input of a puzzle of its own, picture ->
  output, read by read_figures as the next pass: at most 10 (MOST_PASSES),
  and a pass that draws nothing new ends them. It is said where it leaves
  fewer cells unsaid (rest_says_more), else shown in the readout only ("[rest
  N] ..., saying no more than the pass before it", naming what the rest
  lacks). A pass after the first says ONE change, its hand words or one paint
  (finish_the_line's one_change; no whole-line rules, rivals, reading ahead
  or frame there): each sentence reads the canvas as the program has it
  (Output), and a second sentence would read a picture the reading never
  looked at. say_the_rest (9_sentences) spells each pass after the line: a
  paint opens on Output; hand words lay the hand again, `Output SelectAll
  Withhold`, `CurrentObjects SelectAll Difference` (Difference ends a sentence
  and hands nothing on, so the picture is taken in hand first), then the
  hand's sentences from CurrentObjects. readouts.py judges the last kept pass,
  and its "explained but unsolved" now skips a program that fits the pairs.
  Tried and dropped: no picture where a paint cut into a hand piece (the
  picture word's `whole` rule). It stopped 63613498 and f823c43c, and cost
  6c434453, whose plus cuts each square and removes the leftover corners, its
  real rule (the user: turn it off). 221 right, gained 6c434453, 92e50de0,
  lost 0; explained but unsolved 0 -> 0; 5 new wrong, unsolved before, each
  the reading flailing for a word it lacks (the user): 63613498 a key by
  shape (the piece of the shape inside the grey frame turns grey), 7d1f7ee8
  everything inside an outer frame takes its color at any depth (a pass
  pushed it one frame in, and the test is one frame deeper), bdad9b1f rays to
  the edges with a third color where two cross, e40b9e2f turned around the
  body's middle and not its box, f823c43c the noise color removed and the
  lattice kept (periodic_denoise did not fire). The user: keep it on, it
  exposes the real issues; new wrong answers are allowed, a lost solve is not
  (AGENTS.md). Full run 255 s (was about 50 s). arc_spell's `--test` shows
  each test drawn, fitting or not. GPU kernel unchanged. Run out/rest, build
  log out/rest-build.txt.
- 2026-10-10: AN L BRIDGE (the user: generalize Bridge, an L as a variant,
  a start pick and an end). Bridge[9..27] = BRIDGE_TO_PARTNER + kind: from
  each picked piece to its one partner of `kind` (partner_of_kind, the owner
  MoveToward reads too: each partner_by way, then the piece of each color),
  along the picked piece's middle row to the partner's middle column, then
  down or up that column; the pick is the start, so it says which way the L
  turns. Like a bar it is only the path between: each leg is cut back off
  both pieces' boxes. bridge_bars now asks the pieces as the partner helpers
  do (gpu_partner_view, partner_pieces) instead of three lambdas, and answers
  whether an L turned. bar_paint asks every pick for every straight bar
  first, then the L kinds; bridge_holds says an L only where it turns in one
  pair at least (833966f4's 1-wide column: an L straight down to the 6 in
  every pair, a coincidence, was wrong on its test). 0e671a1a:
  `SelectColor[2] Bridge[22] Recolor[5]` then `SelectColor[4] Bridge[21]
  Recolor[5]`, the 2 to the 4 and the 4 to the 3. 219 right, 0 wrong,
  gained 0e671a1a, a2fd1cf0, d4a91cb9, lost 0; explained but unsolved
  0 -> 0; 9 unsolved readouts say an L for part of their change (e7639916
  now elects the multicolor cut). GPU kernel: 80 registers, 70,400 bytes
  stack (+48); 8 more #20014 warnings of the same kind. The run was compared
  and deleted; build log out/lbridge-build.txt.
- 2026-10-10: A KEY DRAWN IN THE GRID (the user: reading something off a key
  is general in ARC; any number of keys, any property as a key). A key is a
  table the grid holds itself, read off each grid: every piece the program
  leaves unpicked, read together as one drawing (each_key_entry, 8b_op_math).
  Its first reading is by color: two cells of two colors side by side, alone
  together along a row (or a column), are an entry. The first word on it:
  BidiColorMap past its 45 fixed pairs trades the pairs the key names, [45]
  along rows, [46] along columns (key_swaps; swaps_of is the one table of
  every BidiColorMap value, for the GPU op and the replay). In the line it is
  key_word (fired_rule::key_swap, a recolor), replayed by key_swap_holds on
  repaints_hold (recolors_hold, one color a piece, now sits on it). 0becf7df:
  the multicolor cut, key and figure, `SelectLargestObject[0]
  BidiColorMap[45]`; it had elected the colors cut by coincidence (105 vs 114
  unsaid), naming copies, turns and shape changes. 212 right, 0 wrong, gained
  0becf7df, lost 0; explained but unsolved 0 -> 0; only its readout changed
  (run out/keyswap).
  GPU kernel unchanged: 80 registers, 70,352 bytes stack; 5 more #20014
  warnings of the kind 8b_op_math already had (host pieces in a host-device
  template). Build log out/keyswap-build.txt (the run was compared and
  deleted).
  Then READ BY PLACE (each_key_cell, key_place_colors): the n-th picked piece
  in reading order -- the bands of rows SelectRow reads, then left to right --
  takes the color of the key's n-th entry in reading order. The entries are
  every cell of the key where it holds one for each picked piece, else every
  cell but its ground's, the color most of it wears (color_tally), as a
  grid's ground is: 1da012fc's gray panel, one of its dots on the panel's
  very edge, so "framed in one color" was tried and dropped. Said as
  ColorFrom[9] (KEY_BY_PLACE, past the partner ways): color_from_word asks it
  with the partners, key_place_holds replays it, carry_the_hand carries it;
  the GPU's scratch sits in a frame of its own (gpu_paint_key_places, 9,016
  bytes; the kernel's 70,352 unchanged). 216 right, 0 wrong, gained 1da012fc,
  33b52de3, fafd9572 and ef26cbf6 (its shapes take the dot of their pane; the
  walls are the key's ground, and the n-th dot and the n-th shape share a pane
  in every pair and the test), lost 0; explained but unsolved 0 -> 0; only
  those four readouts changed. The run was compared and deleted; build log
  out/keyplace-build.txt (7 more #20014 warnings, same kind).
  Next keys (unsolved training): by place in a field, each piece the key cell
  of its quadrant (58743b76: the field is the canvas outside the key's frame);
  by color one way, the rows in order (e4888269); a sequence (b457fec5).
- 2026-10-10: THE PICTURE MADE SYMMETRIC (the user: a composition; mirrored
  pieces alone left 9ddd00f0's second pair unfinished). One word, the paint
  `symmetric` (finish_the_line's symmetry_paint, symmetric_holds): the
  picture laid over itself posed around the grid's middle, pose after pose,
  each taken of what is drawn by then -- one mirror, either diagonal, the
  half turn; both mirrors, the quarter turns; all eight ways (`symmetries`,
  narrowest first). Spelled `Output SelectAll <pose words> Union`, a
  sentence a pose (Output is the whole canvas). One word, not a pose at a
  time: an early pose can finish one pair and leave a later one nothing to
  change there. It claims as the mirror word does (copy_claims, now shared):
  weighed as a paint it lost the cells of the pieces it copies to a mirror of
  some of them. It completes the input, asked only while nothing is said yet:
  asked after 21f83797's rays it mirrored them, a symmetry its pairs' marks
  share by coincidence, and went wrong. 211 right, 0 wrong, gained 5751f35e,
  9ddd00f0, lost 0; explained but unsolved 0 -> 0; e729b7be now says its
  half turn this way, still right; 8e5a5113, ac0c2ac3 readouts changed.
  Run: out/symmetry.
  Copies batch (67 lack only "where each copy was drawn"), rough groups by
  first pairs: lines and trails from pieces 15 (5c0a986e, 1f0c79e5); a
  pattern repeated along a direction 12 (9b30e358, e619ca6e); output bigger,
  the input tiled or fractal 11 (ccd554ac, a59b95c0, 8f2ea7aa); a template
  stamped at markers or into panes 9 (3e980e27, 36d67576, c444b776); output
  smaller, pick or crop 8 (f25fbde4, 662c240a); mirror into place 6
  (2546ccf6, 2b01abd0); other 6. Open, each a general case: ccd554ac needs a
  size rule scaling by a number read off the input (its side; others: its
  color count, cell count); c444b776 needs a repeat both ways as one paint
  (`TileX[10] TileY[10]` fits and is right; asked one way at a time, the
  second way does nothing in pair 0).
- 2026-10-10: THE INSIDE OF A PIECE, a composition of existing ops (the
  user: read the op list): Outline hollows a piece to its edge, InteriorColor
  fills what the edge encloses, so `pick Outline InteriorColor[c] Union`
  recolors a piece's inside. The fill paint asks it as value 1 beside the
  hollows (value 0): inside_holds replays it laid over, the edge as it was,
  the inside in one color read off the output; `outlined` is now the one
  hollowing, shared with carry_the_hand. 209 right, 0 wrong, gained
  50cb2852, 694f12f3, bb43febb, lost 0; explained but unsolved 0 -> 0;
  8886d717's readout changed too. Run: out/inside. Left in the cell-role
  group: rings by depth (516b51b7: the inside's inside, which wants the
  picture cut again), corners (b6afb2da, 15663ba9), quadrants (639f5a19).
- 2026-10-10: the batch "lacks only a recolor" (66 puzzles; ids and a rough
  grouping of all 66 by their first pairs, below). The outline word was
  gated on cells waiting for a shape change, while it claims cells waiting on
  no word too; 4347f46a's cores are simply gone, with no word named, so it
  was never asked. Its gate now takes what its claim takes (colors_left in
  finish_the_line, starts_from built on it). 206 right, 0 wrong, gained
  4347f46a, lost 0; only its readout changed. Run: out/outline. Not bugs:
  e509e548 colors a path by how it turns (one turn, a U, an S; sizes differ,
  so no shape key), a09f6c25 reads no shape table.
  Rough groups (first pairs only): not really a recolor 16 (moves, copies,
  counting, size changes, denoise: 48634b99, 9f27f097, e26a3af2); a whole
  piece by a property 12 (exact size d2abd087, shape class e509e548,
  direction 1d61978c, every other one 817e6c09, mirror partner ce039d91);
  part of a piece by the cell's role 11 (inside bb43febb, corner/edge/inside
  b6afb2da, depth 516b51b7, inside by rank 694f12f3, corner kinds 15663ba9,
  quadrants and inset 639f5a19, lower half e7dd8335, every second cell
  a5f85a15); a key or legend in the grid 8 (1da012fc, e4888269, 33b52de3);
  relation to another piece 8 (inside its box 36fdfd69, nested 7d1f7ee8,
  ef26cbf6); local patterns on the ground 7 (a8d7556c, 9f5f939b); colors
  rotating among pieces 4 (bda2d7a6, 880c1354).
- 2026-10-10: the rest of the change read off the picture so far (the
  user's "the rest fires on the rest"). Once a paint is said, ColorFrom may
  be asked on the picture the line has drawn -- line_drawn, with every
  painted cell in the output's color -- cut again as the line cuts the input
  (finish_the_line: the_picture, picture_word; replay_facts takes grids).
  Said after the paints said before it, in a sentence that opens on Output
  (fired_rule::on_the_picture; 9_sentences), laying only its picked pieces,
  over what is down; no word on the hand is asked after it (picture_read).
  It may recolor only whole pieces of the line's cut (`whole`): unguarded,
  63613498 fit by coincidence and went wrong (diagonal rays laid onto a
  shape, then its fragments recolored).
  READ AHEAD. Plainly, the line said "the largest becomes 1" (140 cells)
  before the bars (37), leaving pair 0's one touched block nothing for the
  picture to change (a word must change something in every pair). So each
  line is also read ahead (finish_the_line's read_ahead; read_figures asks
  both, says_the_most keeps the read-ahead line only where it leaves fewer
  cells unsaid): a paint is worth its cells plus what read_off_the_picture
  then says with its cells drawn in (each paint's offer carries `drawn`),
  skipped where even every unclaimed recolor cell could not make it win; a
  paint said on that promise is followed at once by the picture word.
  Unpaired, 833966f4's rays went ahead and words followed: fit, wrong. Kept
  only where better, since ae3edfdc's 8-connected line read ahead said less
  (its rays and the picture stopped its moves). A line is read ahead only
  where read plainly it left a recolor unsaid (the picture reads recolors
  alone; checked: said.txt and every readout byte-identical without it).
  0d87d2a6's line now: the bars across, then the red pieces take the color
  of the piece they share a side with, read off the picture (all 7 blocks).
  205 right, 0 wrong; against the session's start (out/snap-keys) gained
  0d87d2a6, lost 0; explained but unsolved 0 -> 0. Against the plain line,
  only 0d87d2a6 and b0722778 (unsolved, 27 -> 25 unsaid) changed. Run:
  out/lookahead2. Full runs took 101-124 s against 54 s before the picture
  word, but the laptop was hot and gaming (the user): not measured cleanly.
  Tried and dropped: reading the line with the paints first (a coincidental
  ray paint went first, 46 cells); a Bridge-only version (ColorFrom on the
  list Bridge hands on) that solved 0d87d2a6 only by re-laying a block the
  line had already made blue -- the user: no "for now".
- 2026-10-10: Bridge reaches across the grid too: Bridge[6..8] (both ways,
  rows, columns) join only pieces facing each other from opposite edges of
  the output canvas (left and right along a row, top and bottom along a
  column; bridge_bars takes the canvas size, as MoveToEdge does). Asked
  before the nearest (bridge_ways_narrowest_first, {7,8,6,1,2,0,4,5,3}): it
  asks more of a pair. 0d87d2a6's test has two marks on each side edge;
  the nearest joined them down the edge, across joins each to the mark
  facing it (checked in Python: its test bars 36 drawn, 0 wrong). 204 right,
  0 wrong, gained 0, lost 0; explained but unsolved 0 -> 0; 4 readouts
  changed: 0d87d2a6 says Bridge[6]; 22eb0ac0 says Bridge[7] for Bridge[1],
  still right (solutions/22eb0ac0 rewritten); 6bcdb01e and 6cdd2623 say
  more, unsolved. GPU kernel: 80 registers, 70,352 bytes stack, spill loads
  +20 bytes. Build log out/bridge-across-build.txt (the run was compared
  and deleted).
- 2026-10-10: Bridge joins only the picked pieces, and only its bars come
  back picked (bridge_bars takes `picked`, shared by the GPU op and the
  replay; Bridge's PICK column is now true). It joined every piece whatever
  was picked, so 0d87d2a6's red blocks that line up got red bars too. The
  line asks the bars on every piece first, then on each pick the other paints
  use (bar_paint), spelled `<pick> Bridge[v] [Recolor[c]] Underlay`: the old
  `SelectAll Bridge InvertSelection` loses its InvertSelection (22168020,
  a699fb00, aa18de87, dbc1a6ce respelled, still right; solutions/a699fb00
  rewritten). 204 right, 0 wrong, gained 0, lost 0; explained but unsolved
  0 -> 0; 14 readouts changed (5751f35e, 6bcdb01e, f8f52ecc now elect the
  4-connected cut, which leaves fewer unsaid; cf133acc says one picked bar
  for two trails). GPU kernel unchanged: 80 registers, 70,352 bytes stack.
  Build log out/bridge-pick-build.txt (the run was compared and deleted).
  0d87d2a6 then said its blue bars (SelectColor[1] Bridge[0]). Left:
  (a) each red piece a bar touches turns blue: ColorFrom by touch with the
  bars as partner, which needs a said paint to join the hand the later words
  read, and the line spelled in the order said -- done above, for the bars;
  (b) done above (Bridge across).
- 2026-10-10: the app keeps what inference said when nothing fits. Where no
  inferred program fits the pairs, try_what_inference_said (19_solver_executor)
  leaves the first (most likely) one on as BEST, drawn on every train pair,
  instead of wiping it; turn the inference pane off [I] to see it, hover its
  words for each step. score_program drops the cached verdicts of the tree it
  replaces (a mid-search import was judged by the old tree's). arc_said and
  Kaggle's inference-only run have their own loops: 204 right, 0 wrong,
  gained 0, lost 0, said.txt and all readouts byte-identical.
- 2026-10-10: SnapToAnchors retains only body cells whose color has an
  external anchor as keys (8b_op_math.hpp, shared by GPU and replay). Details
  with no possible partner used up the first 12 key slots: on 94133066's
  second and third pairs, ten detail cells displaced the third true key.
  The existing alignment and bounding-box crop now infer the solution; no
  new op, detector, or sentence handling. Full build and training score:
  204 right, 0 wrong, gained 94133066, lost 0; explained but unsolved 0 -> 0.
  Before: out/snap-before; after: out/snap-keys. The key-count limit itself
  is unchanged. Build resource log: out/snap-full-build.txt; no pre-change
  compiler resource log was available for comparison.
- Training: 231 right, 7 wrong. Current run: out/shared/ (out/faster/ reads
  the same, slower), the one before it out/ranks/, and before that
  out/x64-Release/ (out\ is not in git, and was wiped on 2026-10-12: the
  older runs are gone). 170 at 32d371d, 172 at a201ff6, 187 at 7740de8,
  199 at abc3f0b, 224 at 0b35bfb.
- This session made the line compose existing words (all in 7_infer; report
  out/opus55-compose-refined-report.md, patch and snapshots in
  out/opus55-compose-refined-work/). 203 right, 0 wrong, gained 18286ef8,
  22168020, ae3edfdc, b8825c91, lost 0, explained but unsolved 0 -> 0; six
  solved puzzles say another program, still right. The old composition
  patch (each start color alone in picks_to_ask, a move claims the input cell
  its landing covers, recolor_word asks the start colors, paints laid under
  read the hand the line leaves) broke five puzzles (out/opus55-compose-v1);
  each break named a missing distinction or a bug, fixed at its owner:
  - A size rank is asked only where no pair ties at that rank (picks_to_ask,
    sizes_in_order): a tie is broken by scan order, a pick by place
    (833966f4's one-cell pieces; 22168020's mirrors).
  - The words' claims are held to the hand drawn as the line lays it, after
    the words: a later word can undo an earlier one's landing (50c07299 slid
    one piece twice).
  - All four ways is the mirror asked first, so it takes a tie, as the longer
    ray does (b8825c91's test patch sits on the middle column).
  - Words and paints compete in ONE loop, biggest first (finish_the_line):
    asked after the words, paints lost what they say whole to coincidental
    words (178fcbfb's repeats). The canvas is laid again each round from the
    hand (or the cut, where the line is still) and the paints said; a paint
    that reads what is down (under, onto, bars, fills, boxes, the turned
    figure) freezes the pieces: no word that moves, copies, reshapes or
    removes one is asked after it. Each paint replay is cached per canvas.
  - One drawing of the line (draw_the_line, line_shows): a paint accounts
    only for cells the line does not already show (22168020's vacuous repeat
    onto what is down), and Drawn By The Line claims what it shows.
  Timed under the same conditions, the full run took 262 s against the
  baseline's 267 s (the laptop was slow then; another session was building).
  Left: a "paint cannot beat the best" bound was dropped (a pair's left-over
  list can hold one place twice, so it was inexact, and it saved nothing).
- This session fixed the move replay (host/GPU mismatch on 833966f4): Union
  draws the hand in list order, a later piece over an earlier one (Underlay
  the other way), but slides_show only asked the output to show a moved piece
  where it lands, so it claimed the 9 that AlignTo lays under the unmoved 0
  after it. Now one host owner, draw_the_hand (7_infer, beside paint_piece),
  draws the hand as the line lays it (replay_facts::laid_under, set where a
  piece sits on another); slides_show and "Drawn By The Line" both use it, and
  a moved piece must show in that drawing and in the output. No GPU change;
  the order is a loop, not arithmetic, so no ARC_HOST_DEVICE helper. Proof:
  with the reverted composition patch on top (out/opus55low/composition.patch,
  removed again), 833966f4 went from "0 unsaid" yet not fitting
  (out/opus55low) to a line that fits its pairs (out/opus55-diag; its test is
  wrong, a matter for that patch). Full run: 199 right, 0 wrong, gained 0,
  lost 0, said.txt byte-identical, explained but unsolved 0 -> 0; 23 readouts
  changed (18 unsolved, 5 solved where only a rival cut's count moved).
  Missing words: copies 484, recolor 382, turns 253, where pieces went 225.
  Left: a recolor's replay (paint_holds) still checks only the output, not
  what the line lays over a recolored piece that moved under another.
- EVALUATION, 2026-10-08, inference only (arc_headless, ARC_INFERENCE_ONLY=1,
  ARC_DATASET=eval): 0 of 167 test outputs, 0 of 120 puzzles; no program
  fits any puzzle's pairs (out\eval\). Readouts: `ARC_DATASET=eval
  scripts\said.bat evalread` (out\evalread\). Missing words named: copies 99,
  recolor 71, turns 64, where pieces went 55, shape changes 36, drawn on the
  ground 13; 58% lack 3 or more words (31% of unsolved training), only 23
  lack one. The training solves bunch in the first hundred by id (42, then
  10-24 a hundred). 8f215267 names no missing word yet does not fit: cells
  drawn with no partner wait on no word, so `missing` undercounts.
- This session moved tile boundaries out of grid (item 8 below). The tiled
  frame supplies them explicitly to the piece readers, matching and both
  replay paths; replay_facts requires the context so a caller cannot omit it.
  Full build and training run (tile_explicit, compared then deleted): 199
  right, 0 wrong, gained 0, lost 0; every program and all 1000 readouts
  byte-for-byte unchanged against out/base. Explained but unsolved: 0 -> 0.
  GPU kernel stack unchanged at 29,488 bytes; only the two existing warnings.
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
- Build time cut 2026-10-08 (CMakeLists.txt says why at each flag): /analyze
  commented out, no /GL, the GUI compiles beside the rest, and the GPU code
  builds with --Ofast-compile=min (the smem-spilling pragma is gone; the
  search may be slower, the user is moving off it). `said.bat` after an
  inference edit builds in about 30-40 s (was about 175 s), a clean full build
  in about 45 s (was 247 s). What is left is cuda_solver.cu, which every header
  in include/core recompiles; arc_said links it. Tried and dropped:
  --Ofast-compile=max (said ran 94 -> 170 s), --split-compile (349 s).
- A full score takes 30 to 80 s; the laptop's clock moves with heat, so
  compare two builds by interleaving them (old, new, old, new) and taking
  each one's fastest. `ARC_THREADS=1` reads one puzzle at a time (about 15
  minutes), for when the user is gaming. scripts\profile.bat profiles.
- What blocks the unsolved (`readouts.py missing <label>`, named / the only
  word missing): copies 485 / 73, a recolor 383 / 71, turns 254 / 26, where
  pieces went 223 / 44, cells drawn on the ground 143 / 67, shape changes
  116 / 7.

## How to check a refactor
- Keep choosing changes upstream, in numbered file order. Once chosen, wire
  the change from the lowest helper back up through every caller. Make new
  context arguments required immediately, so the compiler catches omissions
  before scoring. The tile refactor missed the second replay path (`opened`);
  its default argument hid the omission, costing a second full build and run.
  That was a wiring mistake, not a reason to reverse the reading order.
- Full build, then `scripts\said.bat --as <label>`, then against the last run:
  `cmp out\<old>\said.txt out\<new>\said.txt` and
  `diff -rq out\<old>\logs out\<new>\logs` (byte for byte), and
  `readouts.py solved` and `explained`. Where a change renumbers values on
  purpose, map the old run's numbers first, then compare.
- Where a helper the GPU runs changes, compare the build's `ptxas info` and
  `stack frame` lines before and after.
- Line endings are mixed in the repo: 1_readings, 3_background, 4_objects, 8b_op_math,
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
8. Done: grid is only dimensions and cells. each_object takes an explicit
   tile_size (1_readings); tiled_pieces (6_correspondence) supplies each framed
   input's boundaries from the original pairs, passed through matching, the
   rule readers and replays. Only the tiled frame creates this context.
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
0. Now (the user, 2026-10-11): the wrong ones, in id order, each a word the
   reading lacks: 833966f4, ba26e723, d4f3cd78 (new, the leak through a
   gap), e40b9e2f and f823c43c as the 2026-10-11 passes entry names them,
   and caa06a1f (new, 2026-10-13: fewest holes fit its pairs).
   272f95fa, bdad9b1f and 782b5218 are solved (2026-10-12); 7d1f7ee8,
   29623171 and ac0c2ac3 no longer fit, their words still missing.
1. Speed: read the cuts and plans of one puzzle in parallel (low priority).
   A full run is 558 s: about 88 line readings a slow puzzle, half of them
   tolerant; each rays and repeats length is still a walk of its own (one
   walk could judge every length). A run takes every CPU thread, so the
   machine stalls while it runs (the user). The hotspots are several now --
   the passes after the first, walking paints, partner lookups, matching --
   and most of what is left needs fewer readings, not faster ones. Open:
   the speed kept back and the GPU queue (2026-10-12, "Speed kept back").
   Lighter runs (the user, bored by a stalled machine): a batch of ids for
   quick checks, ARC_THREADS=10 to leave half the CPU free.

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
- The line (finish_the_line): each kind of word and of paint offers its best
  on the hand and the canvas the line leaves so far, the biggest offer is
  said (a tie to the words, then the paints in order), then all are asked
  again; each kind of word says at most 8 words (a reading lists at most 127
  rules). Spelled, the paints follow the line, each in a sentence of its own
  (e729b7be's half turn, 178fcbfb's repeats). A paint that reads what is down
  freezes the pieces: no word that moves, copies, reshapes or removes one
  comes after it. A paint is weighed only where it explains a cell in every
  pair (91714a58), and only on cells the line does not already show.
- The words' claims are held to what the line draws once they are said
  (50c07299): a cell the drawing does not bear out is unclaimed again.
- What the line draws is said (Drawn By The Line, list_claims), however the
  matching paired the pieces (d23f8c26's middle column).
- A recolor or a move answers to any cell changed in place
  (left_over_cell::answers_to). The cell a moved piece leaves may show
  anything but the piece itself.
- A replay checks everything its op draws, with the op's own arithmetic: the
  trail redraws its body; the turned figure is the whole drawing. A move is
  checked on the hand drawn as the line lays it (draw_the_hand): under Union
  a later piece in the list covers an earlier one.
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
  Tile, and the reading passes each input's tile_size explicitly to each_object
  through tiled_pieces; grids carry no reading state (10fcaaa3).
- A list holds 900 pieces (MAX_OBJECTS_IN_A_LIST); the reading reads a cut
  only where no input cuts into more than 128 (MOST_PIECES_READ, TEMPORARY).
- Partner ways are numbered in the order they are asked, surest first
  (partner_by): around it, inside it, its rows, its columns, the nearest, the
  touch at a side, the touch at a side or a corner, the same shape, turned
  or not. Directions are numbered as a compass is, clockwise from up
  (heading), asked sides first (heading_asked). A tie goes to the first
  asked.
- The partner helpers (each_partner and kin, 8b_op_math.hpp) ask one pieces
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
  first so many by size, a row or column from the middle -- or the part one
  color's walls fence off (SelectPartition); but_those turns it round, said
  with InvertSelection; then WouldFit, or SelectLikeKey (like a piece the
  pick left out, the key), keeps some of what it took. A count or a rank is
  asked only where it splits some pair's pieces with no tie; a size rank only
  where no pair ties at it (833966f4). The colors an unsaid change starts
  from are asked together, then each alone (18286ef8's 9). The newest picks
  are asked last, so an older one takes a tie, and a key fenced off by walls
  before any other key (63613498).
- Place is a piece's row band and column band (overlapping_bands: Layout,
  SelectRow, SelectColumn) or the part walls fence it in (fence_parts:
  Partition cuts on the parts, SelectPartition picks by them), the parts
  numbered in reading order, row by row, as rows wrap.
- A mirror all four ways is asked first and takes a tie (b8825c91).
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
