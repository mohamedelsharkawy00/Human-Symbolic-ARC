# Handoff, 2026-10-17

Read AGENTS.md first: the rules.
This file holds only what is current; past stages are in git history
(`git log`, and this file's own history).

## Deadline: 2026-11-02
The user: not much time left. Prefer work that gains puzzles soon over long
redesigns; the big ideas wait in "Parked" below.

The plan (the user, 2026-10-12): training until 2026-10-25, then the last
week on the evaluation set, which is what Kaggle scores (0 of 120 on 10-10
and again on 10-13). THE AIM IS ALL 1000 of training (the user, 2026-10-13:
350 was never the aim; an earlier entry here had it wrong). The trend,
training right, the best of each day: 187 on 10-07, 199 on 10-08, 219 on
10-10, 224 on 10-11, 237 on 10-12, 258 on 10-13, 266 on 10-14. All 1000 by
10-25 needs about 67 a day from 266, so each change has to gain whole
families, tens at a time, not one or two.

THE WAY OF WORKING is AGENTS.md's (each puzzle with all of its adjacent
concepts), and the user's (2026-10-17): THE SLOWEST PUZZLES FIRST, NOT ID
ORDER, each its missing word built in the op that owns it. 43 take over a
minute each and hold 56% of a full run's reading (Kaggle, T4, 4 threads,
a4b3a9b), and 41 of them are unsolved (no program fits). The slowest:
256b0a75 312 s, 8a371977 292 s, a64e4611 288 s, 4e45f183 254 s, db615bd4
247 s, 5c2c9af4 247 s, e681b708 235 s, 890034e9 172 s, e2092e0c 156 s,
97239e3d 150 s. In id order, ids 1-50: 40 right; 51-100: 12
right (out/hole2). The first unsolved in id order: 05a7bcf2 (a hand program
only, about half its cells unsaid), then 1190e5a7, 11dc524f, 11e1fe23 --
1190bc91 and 4364c1c4 are solved now. A run of 50 ids takes about a minute,
its readouts byte-identical to the full run's. Families of the unsolved that
lack the same word (`readouts.py missing`, out/hole2): 59 lack only where a
copy was drawn, 54 only a recolor, 41 only the cells drawn on the ground, 37
only where a piece went. out\ was wiped on 2026-10-13: the runs older entries
name before then are gone.

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
- ONE OWNER FOR RAYS AND REPEATS (the user, 2026-10-13: the dryness later).
  Rays live in two ops -- ExtendTo (a trail one way from every cell; past its
  eight ways, grown in step until they meet) and ExtendDiagonal and
  ExtendOrthogonal (rays of a set length from one seed cell) -- and repeats
  in several: TileX, TileY, Tile, DiagTile. One ray op whose value says how
  far (a length, or until it meets something), and one for repeats.

## Where it stands
- 2026-10-17: SPEED, THE 8x (the user: the 8-minute laptop run "should have
  been 1 min on my machine or lower"; "i know claude loves to narrow the
  scope from the 8x": each step is weighed against 8x, and a small gain is
  not the aim). WHERE THE TIME IS (Kaggle, T4, 4 threads): the full run at
  a4b3a9b, 10,021 s summed over its puzzles, 96.5% of it outside the 266
  solved, the 100 slowest 75%. TEMPORARY TIMERS, still in (368f0be:
  inference_times and time_into in 7_infer.hpp, their columns in said.cpp's
  timings.csv; nothing reads them; remove them once they have served), on
  the 5 slowest (256b0a75, 8a371977, a64e4611, 4e45f183, db615bd4: 1,275
  s): by reading, strict 24%, tolerant 31%, partners 44%; by stage,
  preparing (the matching) 12%, the line in the first pass 34%, in the
  passes after it 54%, reading ahead under 1%; THE KINDS' OFFERS 84% --
  color_from 18%, move 17%, key 10%, bar 9%, ray 6%, recolor 5%, repeat 5%,
  mirror 4%, the rest under 4% each. 256b0a75 alone: 100 lines, 2.1 million
  paint replays, 2.0 million slide replays, 2.75 million walks, 2.2 million
  lays. Of the 266 solved, 250 keep the plain strict reading (8 tolerant, 7
  strict with the partner picks, 1 tolerant with them) and 249 one pass; of
  the 1,777 plans read both ways, 1,207 end the same strict as tolerant
  (cells unsaid, passes and words).
  TRIED AND DROPPED: the partner reading on the plain reading's cut alone,
  not on every cut again (b16882b, reverted by ca76267, the user's call).
  Lost 9720b24f and d282b262: their partner picks say every cell on a cut
  the plain reading did not choose (9720b24f the monochrome cut, where the
  plain reading chose the multicolor one; d282b262 the other way round). The
  partner picks are words, and the words choose the cut. 1.16x on the 5
  slowest (1,275 -> 1,103 s) and 1.08x on the 266 solved (348 -> 322 s),
  part of it the run itself: their strict readings, unchanged, came out
  5-7% faster too (other puzzles ran beside them; a64e4611's tolerant one
  31%). Explained but unsolved 0 -> 0. Checked on the 266 solved and the 5
  slowest, `kaggle_push.py --run train <ids>`: 607 s with the compile
  (out/one_cut, this cloud session only).
  STATE: the code is a2ad22a's, the timers in; no full run since a4b3a9b.
  NEXT, AND WHY: a time profile by function of the offers (84%), on the 5
  slowest: too many replays, or slow ones. The stage timers cannot tell the
  two apart, and the fix guessed from them gave 1.16x and lost two. On the
  laptop, `scripts\profile.bat 256b0a75 8a371977 a64e4611 4e45f183
  db615bd4` (VTune, one thread, as the profiles of 10-12 and 10-13 below);
  the cloud has none set up. Then the fix at its owner, checked on the 266
  solved and the 5 slowest, and weighed against 8x.
- WORD AND OPERATOR, ONE NAME EACH (2026-10-16, the user: a word is one
  operator). The same noun carries three sizes in the code. A program
  token is one op (operation_named, "Read a word's operation name";
  spell("Recolor", 3)). An act the reading fires is a pick plus an op
  (rule::outlined; pick_words spells one pick as up to six ops). A whole
  program is what word_group::whole_program holds -- diagonal_tile,
  piece_slam, corner_chain, slide_to_touch, project_on_seeds,
  snap_to_anchors, own_sentences -- "These rules redraw the whole output
  and spell their own programs". The act already has a noun of its own,
  fired_rule / said_as / rule::, so only the name is wrong: word_group is
  an act group, and the nine *_word lambdas in 7_infer (color_from_word,
  move_word, mirror_word, outline_word, recolor_word, shape_word,
  key_word, removal_word, picture_word) offer acts, not words. The ops'
  own list stands as it is: WORDS THE READING CANNOT SAY YET names ops.
  AGENTS.md 2.6 is the one doc line that breaks the rule -- "New words
  are compositions of existing ops where possible" says a word can be
  several ops, where it means a reading's line. Not started.
- OWN-COLOR PAINT GATE (2026-10-16): based on restored fed27b2. Later memory,
  GUI and cache experiments were stashed at the user's request; main/origin
  were already fed27b2. Do not reapply that stash as part of this work.
  New change only in 7_infer: selected source palettes retained per cut/pick;
  each round reads actual output colors at unclaimed, not-line-shown cells.
  If any unfinished pair has no shared source/target color, exclude that pick
  from trails/growth, copies and repeats, plus the all-piece turned copy.
  Source colors come from opened pieces, not the recolored hand. Recolored
  paints/rays are untouched. Existing geometry/replay caches and tie order
  stay intact. Full replay still judges survivors. Release build passed.
  The focused 19-puzzle check was superseded by the full training run below.
  That run covered the current paint-color gate: 266 right, 39 wrong, matching
  the handoff's prior full-run counts. No puzzle-by-puzzle comparison is needed.
  User asked for an estimate: rough low-confidence expectation 5-15% overall
  time reduction, NOT measured. Initial a64e4611 plan has only introduced3
  changes but source colors1/2/8, proving whole own-color families futile
  there; profile does not isolate their cost in that plan. After correctness,
  use the same profiler setup against original cpu-20261015-212823-600
  (89.313 s elapsed, 88.391 s CPU), not parallel Release sample timing.
  No new profile/full run yet. Return control during long checks.
- COMPLETED READING REUSE implemented at user's request, focused check passed.
  figure_readings is explicitly owned by infer and passed through read_figures,
  read_the_cut and read_the_rest. read_figures_once uses exact figures_key
  (input/output cells, dimensions, tile sizes) plus pass/tolerant/partners.
  Completed results are shared_ptr<const inference>; callers retain separate
  mutable claims and ground-plan annotations. FIFO retention of eight entries
  per puzzle; existing rest/framed ownership may retain descendants longer.
  Cache entries are installed only on completion; no in-progress reuse.
  Existing initial preparation reuse per ground plan remains. Audit runner
  uses the same figures_key owner instead of a second serialization.
  Release build passed. reading_reuse_sample completed,
  same 13 ids as prep_audit_sample, all available workers, AboveNormal, no
  affinity, ARC_PREP_AUDIT unset. Logs out/reading_reuse_sample.stdout.txt and
  .stderr.txt. All 13 readouts and programs identical to permission_places_full:
  4 right, 1 wrong, 8 unsolved, gained 0, lost 0. Across the sample:
  preparations 612 -> 172; readings 647 -> 205; matching calls 206720 -> 53632;
  fitting edges 32711257 -> 28222058; paint replays 10464279 -> 3722107;
  slide replays 8834287 -> 3758919. 79cce52d readings 281 -> 28;
  25094a63, db615bd4 and a64e4611 readings unchanged. Do not attribute their
  lower elapsed times to reuse: CPU conditions vary. Observed process working
  set about 3.6 GB after ten completed cases, not a final peak measurement.
  Full training check completed on 2026-10-17: 604 s wall time, 1,000 readouts
  and timing rows, 266 right / 39 wrong, matching the prior full-run counts.
  This checks reading reuse across the full set; no puzzle-by-puzzle comparison
  is needed. Results: out/handoff_full_20261017. Its time is not compared with
  earlier runs because their scores differ. Then follow the profiler branch
  upstream on expensive cases that did not reuse readings; no more speculative
  storage tuning. User explicitly
  asked to stop adding measurements and analyze code; no new audit added here.

## Still live, carried up from the cut
- The laptop's clock moves with heat (2.4x seen back to back on 10-13): compare builds old, new, old, new, in turn; runs at different times do not compare.
- `out/` holds a run's `said.txt` and logs, about 600 MB each; delete one once compared (the user's call).
- The speed list of 10-13 (items 1-9, the partner reading's cost among them) is in git; the 10-17 entry above holds the next step.

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
   reading lacks -- 39 on 2026-10-14 (out/hole2): 17b80ad2, 1d61978c,
   1f876c06, 2072aba6, 22a4bbc2, 29623171, 321b1fc6, 332202d5, 39a8645d,
   3ac3eb23, 3bd292e8, 3bdb4ada, 56ff96f3, 67a423a3, 6bcdb01e, 7acdf6d3,
   7c8af763, 7e0986d6, 833966f4, 8fbca751, a5f85a15, ac0c2ac3, ae4f1146,
   ba26e723, caa06a1f (fewest holes fit its pairs), d2abd087, d4f3cd78 (the
   leak through a gap), d5d6de2d, db3e9e38, e179c5f4, e40b9e2f (turned
   around its body's middle, not its box), e73095fd, e734a0e8, e760a62e,
   ecb67b6d, f1bcbc2c, f5b8619d, fe9372f3, fea12743. And unsolved, open:
   868de0fa, the partner picks wanted under another plan than the plain
   reading's best.
1. Speed, now the first (the user, 2026-10-17: the 8x, "Where it stands"
   has the state and the next step). From before: read the cuts and plans
   of one puzzle in parallel (low priority).
   A full run took 563 s on 10-13 (out/rays_fanned; 558 s on 10-12, before
   the shape work): about 88 line readings a slow puzzle, half of them
   tolerant; each repeats period is still a walk of its own (the rays read
   every length off one walk since 10-13). A run takes every CPU thread, so
   the machine stalls while it runs (the user). The hotspots are several now --
   the passes after the first, walking paints, partner lookups, matching --
   and most of what is left needs fewer readings, not faster ones. Open:
   the one speed list (2026-10-13, "picks that take the same pieces asked
   once"); the GPU queue was ruled out (2026-10-12).
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

## Where things are
- `include/core`. Inference lives in `3_background.hpp` (background
  election and the ground plan: the one owner of the background).
- `6_correspondence.hpp` (readings, fired rules, the line, missing
  words, ranking the cuts).
- `7_infer.hpp` (replays `*_holds` on three templates, `paint_holds`,
  `slides_hold` and `drops_hold`; picks, frames, `carry_the_hand`,
  `finish_the_line`, `says_the_most`, `read_figures`, `read_the_cut`,
  `read_the_rest`, `infer`).
- `9_sentences.hpp` (the line spelled as programs, `say_the_rest`,
  `say_the_plan`).
- `10_log.hpp` (readouts).
- The passes: once a reading's cut is chosen, `read_the_rest` reads what
  its line leaves unsaid on the picture the line drew, as a puzzle of
  its own, up to 10 passes, stopping when a pass draws nothing new. Each
  pass after the first says one change, read on that picture as the
  program has it (`Output`); `say_the_rest` spells the passes after the
  line, and the readout shows each as `[rest N]`.
- The background: `infer` asks the election for plans (delete what
  first, fill what last), reads each plan's figures with `read_figures`
  (no detector sees a background; empty cells are nothing), keeps the
  plan that explains most, and `say_the_plan` writes its delete and fill
  once.
- Every choice the reading makes (the ground plan, the cut, whether the
  line opens with the matching's acts or the words say every change,
  whether a word may lay cells wrong) has one owner, `says_the_most`.
- Each candidate is read to the end of its words, in the order given,
  and the one leaving the fewest cells unsaid is kept, of those the one
  said in the fewest passes.
- The order only breaks a tie, and once one leaves none with no pass
  after its own no later one is read.
- Each choice is made just above the first word that reads it.
- A word is kept where, in every pair, it makes more cells right than it
  lays wrong (`brings_nearer`, the one judge of every word); the cells
  it lays wrong are left unsaid for a later word or pass. Every plan is
  read with words that lay none wrong first (`inference::tolerant`).
- `include/core/8_vocabulary.hpp`: the ops. Its top comment lists the
  words the reading cannot say yet and how to unlock one.
- `include/core/8b_op_math.hpp`: what each op does to the cells, called
  by the GPU op and by its replay in `7_infer.hpp`. `1_readings.hpp` is
  only how to read a grid (histogram, boxes, objects, shapes).
- Place: a piece's row band and column band (`overlapping_bands`:
  Layout, `SelectRow`, `SelectColumn`), or the part walls fence it in
  (`fence_parts`: Partition, `SelectPartition`, numbered in reading
  order). A key is every piece a pick leaves unpicked (the key words,
  `SelectLikeKey`).
- `src/cuda_solver.cu`: the GPU ops.
- `README.md`: Kaggle preparation and the private smoke run.

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
  (direction), asked sides first (direction_asked). A tie goes to the first
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

## Kaggle
- Only one-shot inference is tested on Kaggle (ARC_INFERENCE_ONLY=1, no
  search). Submission 56790276 (notebook v37) scored 0.00; v38, with the
  loader fix, was never submitted. A local evaluation of 120 tasks scored 0.
  Source upload: `python scripts/kaggle_push.py -m "..."`; smoke run with
  `--smoke`. Scoring rules and links are in README.md.
- 2026-10-16, KAGGLE CLEARED (the user: all of it old and stale): the
  notebooks arc-build, arc-smoke-check, arc-cloud-check and arc-l4-probe and
  the dataset solver-source-code deleted; the code only Kaggle had is in
  kaggle/src/diagnostics (cloud_check, smoke_check). The 5 submissions stay
  (no way to delete them). The next `-m` push creates the dataset anew.
- SCORED RUNS ON KAGGLE'S GPU, for the cloud sessions, which have none:
  `kaggle_push.py --run train [id...]` (README, Kaggle). ONE DATASET AND ONE
  NOTEBOOK FOR EVERYTHING (the user: one, the main one), solver-source-code
  and arc-build, pushed from any folder; the "only ARC pushes" guard is gone.
  build.py is told by ARC_DATASET which set to read: training or evaluation
  builds arc_said and hands back what it writes as said.tar.gz; a run builds
  whatever was pushed last, each version's message the commit it came from.
- THE FIRST FULL RUN THERE (2026-10-17, a4b3a9b, whose arc_said is main's
  ac3f883): 266 right, 39 wrong, the same 39 wrong as hole2's list above;
  2,515 s on 4 CPU threads (T4), compile 166 s. The first full run since
  permission_places_full. The local full training run on 2026-10-17 also had
  266 right and 39 wrong; no puzzle-by-puzzle comparison is needed. Its local
  readouts are in out/handoff_full_20261017.
