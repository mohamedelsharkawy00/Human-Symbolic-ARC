INFERENCE SAYS THE ANSWER — and the search stopped having to find it
====================================================================

Written 2026-09-07, after a session that began with a machine-gun sound and
ended with three puzzles solving at load without one search candidate.

Commits: 2058d2a, 15757ac, d277cd9, e873db0. Everything after e873db0 is
UNCOMMITTED and is described under WHERE IT STANDS.


THE HEADLINE
------------
009d5c81 took 11 minutes and 108,210,246 attempts. It now solves in 0.48
seconds with ZERO attempts, because everything needed was already worked out
when the puzzle was read and nobody was asking.

    009d5c81   11 min / 108M attempts  ->  0.48s / 0 attempts
    00d62c1b   (never tried)           ->  0.042s / 0 attempts
    00dbd492   4.3s of search          ->  0 attempts

All three had SAVED SOLUTIONS THAT THE GRAMMAR CAN NO LONGER SPELL — 10 ops,
opening with an ACT, no GET. Same trap as 007bbfb7 in the previous handoff.
Search could never have found any of them. Inference found different, legal,
shorter ones.


WHAT THE LOOKING NOW SAYS
-------------------------
Every one of these was already computed at load and thrown away.

1. THE ELECTION COULD NOT SEPARATE THREE CUTS and picked one by declaration
   order. On 009d5c81, `connected_monochrome`, `connected_multicolor` and `all_cells_of_one_color`
   all explained exactly as much — same used, same left over. Only the first
   was said out loud. All three are now said, each as a whole program of its
   own. The right one was `Colors`, third in the list.

2. A RECOLOR IT HAD NO WORD FOR was already spelled by a detector that fired.
   `shape_color_map valid 1` with the output's own colours. The emitter now
   asks that detector instead of giving up. `claimed_by` says which of the two
   detectors filled the shared table, so ShapeColorMap and
   ShapeToInteriorColorMap cannot be confused.

3. THE COLOUR SWAP sat in `color_swaps` and was never told to anyone.
   `swap 0<->1` is `BidiColorMap[0]`, the detector's variant index used
   directly as the op's argument.

4. MADE OR GONE was one number for two opposite facts. Split by the
   `from_input` flag every cell already carried. 009d5c81 reads `0 made,
   28 gone` — the erase, said out loud. Gone-with-no-made is what proves the
   cells ended up background rather than moved.

5. THE COLOUR MAP LIVES ON THE EDGE NOW. All three sites that set
   `recolored_to` gated on `piece.color >= 0`, so a cut that labels its pieces
   -1 (a motif block, anything mixed) threw away an answer the matcher already
   had in `hit.color_map`. 017c7c7b recolours 1 to 2 on every edge and the
   reading lost it. The edge carries the whole map; `recolored_to` is a view of
   it for the uniform case.

6. A LINE IS A SELECTION AND WHAT HAPPENED TO IT, and each act stands on its
   own evidence. Asked as one key, a turn that differed between pairs took the
   colour down with it. Turn, move and colour are three questions.


THE SEARCH CAN HEAR IT
----------------------
`suggested_programs` is bridged onto the puzzle like every other inference
fact, and `try_what_inference_said` runs them before the beam spends a single
candidate. They also stay in the beam (scored 5.0) so search extends them when
they do not finish the puzzle — inference and search are AND, not either-or.

Two bugs on the way in, both mine, both worth remembering:

  - Gated on `!m_puzzle.inferred`, but `init()` infers at startup with
    `with_inference` defaulting true, so the block never ran. Gate on whether
    the ATTEMPT has happened, not on when inference did.
  - Called `toggle_pause()` while holding `m_solver_state.mtx`. Unpausing takes
    that same non-recursive mutex, so the app DEADLOCKED before opening its
    port. Release the lock first.


THE PRICING FIX, WHICH IS THE ONE WITH TEETH
--------------------------------------------
`made` was free in `price_of`, and that undid the line directly above it.
00dbd492 was electing a reading at 108% that uses 8 pieces and makes 120 cells,
over one at 111% that uses 18 and leaves nothing — the exact trade that pricing
left over was written to stop.

It cost that puzzle its answer: the thin reading sees no recolor, so nothing
asked the detector that had already learned the map, and the search spent 4.3
seconds finding a word inference was holding.

MADE IS HALF SAID, so it is priced at half. A cell made where it stands IS the
answer for the half of a puzzle that is drawn rather than carried, so charging
it as a failure beats the honest readings with the dishonest ones. But knowing
a cell was not carried is not knowing why it is there.

The two comments there had been arguing with each other, and the arithmetic in
the first stopped being true the moment the second landed. They tell one story
now, without sums that rot.


RUN SHAPE
---------
    --max-attempts N     pause at exactly N attempts, not at a clock
    --start-examples N   judge against N examples from the first candidate
    --minimized          as before

`state` gained `attempts=`. A solved puzzle stops itself, whoever solved it —
the saved solution at load, what the looking said, or the beam — and unpausing
is yours, after which it hunts for a shorter one. The saved settings are
restored BEFORE the solver thread starts, so the solver has the last word on
pausing rather than being restarted by a stale "was running".


WHERE IT STANDS — UNCOMMITTED WORK
----------------------------------
Five files modified since e873db0. All of it builds warning-free and all four
puzzles behave, EXCEPT the last item.

  include/3_correspondence.hpp   colour map on the edge
  include/5_sentences.hpp        programs_from, sentences_of, a_blank,
                                           acts stand on their own evidence
  include/6_log.hpp              log lists programs, no numbering
  include/puzzle_loader.hpp                bridge asks programs_from
  include/gui.hpp                          the pane, and the panel frames

DRY, as asked: `programs_from(said)` and `sentences_of(program)` are the one
home. The bridge, the log and the pane all ask there, so what is stored, what
is written down and what is on screen cannot say three different things.

THE PANES MEASURE THEMSELVES NOW. Both faces run their body once to measure and
once to draw, so a row added needs no arithmetic kept in step with it. The two
hand-counts are deleted. Watch for two things: the measuring pass must not
register click zones twice (guarded in `button`), and the op drawer at the
bottom of gui.hpp was left alone because its height already derives from real
counts.


WHAT IS BROKEN RIGHT NOW
------------------------
THE FACTS PANE'S LABEL CHIPS COLLIDE. Matching the facts face's labels to the
lines face's was started and is not finished. A chip is `5*scale + pad` tall
drawn from `y - pad/2`; the facts rows are `line_h` (7*scale) apart, so
consecutive chips overlap. The lines face does not have this because its rows
are `row_h` apart — the chip was designed for that rhythm. The taller content
also pushes the frame down over the BEST column's last grid.

So matching the labels forces the row height; the two are not separable. Three
ways out, undecided:

  - bump the facts rows to `row_h`, faithful but the pane grows by half again
    and with ~45 rows will run off the bottom
  - keep `line_h` and shrink the chip, no longer identical to the lines face
  - revert the chips and leave facts as plain text

Chipped so far: the three election heads, SHAPE PICKS COLOR, WHAT THE READING
SAYS, COLORS GONE/NEW/KEPT, IN EVERY INPUT/OUTPUT, IN EVERY GRID/ANYWHERE,
OUTPUT COLORS/BY SIZE, BG INPUT/OUTPUT/MODE, VETOED, PER PAIR. Column headers
were deliberately left plain, as they are on the lines face.


WHERE TO PICK UP
----------------
1. Decide the chip question above, or revert it. It is the only broken thing.

2. 017c7c7b IS THE NEXT PUZZLE AND IT IS TWO THIRDS DONE. Inference says

       Input SelectAll Pattern Union
       CurrentObjects SelectColor[1] Recolor[2] Union

   and the third sentence is missing: the grid grows 6x3 -> 9x3 and the saved
   solution does it with `Fractal`. The size rule already knows — the pane says
   OUTPUT SIZE SCALED 1/1 BY 3/2 — and nothing turns that into a sentence.

3. HOLES ARE THE AGREED NEXT MECHANISM, and `a_blank = "?"` is already declared
   in 5_sentences.hpp, unused. Inference emits the hole, the SEARCH expands it,
   which is the user's call and the right one. Three kinds, cheapest first:
     - variant hole  Recolor[?]                    10-45 spellings, and
                                                   output_colors / color_swap_
                                                   variants often pin them
     - act hole      CurrentObjects SelectAll ? Union   399 minus dead_ops
     - sentence hole                                falls back to beam seeding
   017c7c7b is an act hole: the frame is certain, only the growth word missing.

4. THE OP ALLOW/BLOCK LIST, which the user wants after holes. Block needs proof
   of impossibility, allow needs proof of applicability and must only suggest.
   009d5c81 has the proof for a block: ZERO background cells ever become
   painted, and the colour map is total (0->0, 1->0, 8->{2,3,7}), so every
   geometry word is dead. MoveRow and MoveColumn alone were 41% of the tied
   sweep.


THINGS I GOT WRONG, SO NOBODY RE-DERIVES THEM
---------------------------------------------
  - Reported a 15x slowdown from the hand term and a 40x level-1 collapse. Both
    were my own spin bug: the attempt cap broke only the INNER fill loop, so
    `pump_generate` refilled pending and the worker span forever while the clock
    ran and `attempts` stayed frozen at the cap. A counter that stops moving
    while the clock does not is the tell. The clean bisect afterwards: hand
    scoring on 28.81s vs off 28.94s — nothing, and the OFF run was slower.
  - Read `already_seen=30` as proof the sweep's candidates were behaviourally
    distinct. It is a PROGRAM-TEXT hash. The real dedup is on state at merge and
    it is brutal: 627,310 of 641,080 dropped at level 0, ~1,554 distinct states.
  - Said the palette was the only way to name 017c7c7b's recolour. The edge had
    it all along; the cut had mislabelled the piece.
  - Killed a running app with a guard that closed whoever held 2026. Never send
    exit_app to an app this session did not start.


NOT DONE, FLAGGED
-----------------
  - The line-based emitter (multi-act lines, SelectColor groups, Withhold on all
    but the last) has never fired. 009d5c81 has no turns or moves, so it builds
    zero lines. It needs a puzzle where pieces turn or move differently.
  - The beam seeding has never been exercised either — every puzzle tried so far
    either solves outright or the search finds something that does not start
    with the seeded sentence.
  - `logging 1` still writes one [TREE] line per candidate under a global mutex.
    A 3.5 minute run made 4.9 GB. Left alone on purpose again.
