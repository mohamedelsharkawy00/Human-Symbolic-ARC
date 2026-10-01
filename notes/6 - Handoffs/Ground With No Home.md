GROUND WITH NO HOME — and a beam that cannot see preparation
============================================================

Written 2026-09-04, after a session that started with a puzzle search could not
solve and ended with it solved in 0.72 seconds by one new word.

Commits: a59209d, 141b155, 3ef988f, d578d8c, 660df95.


THE HEADLINE
------------
007bbfb7 had a saved solution from 2026-08-21:

    Colors SelectAll Fractal AddColor[0] Union      3.46s, 2,042,768 attempts

It could not be found any more, and not because search was slow. Read its beats:
ACT SELECT ACT ACT COMBINE. Three acts, no GET. The four-beat grammar builds
`gets x selects x acts x combines` — one act, and a GET must open. THE SENTENCE
WAS UNSPELLABLE. It predates the grammar it was saved under.

It now solves as

    InputWithoutBackground SelectAll Fractal Union  0.72s, 155,648 attempts

Four words instead of five, thirteen times fewer attempts.


THE CAUSE: THE GROUND HAD FOUR HOMES
------------------------------------
Nothing owned the question "what is the ground", so every beat answered it alone:

    GET       Input hands it back as a real colour (black is 0; TRANSPARENT is -1)
    SELECT    used to say it was a piece you could point at — now says nothing
    ACT       cuts drop it as a side effect; VoteDenoise needs it as evidence
    COMBINE   the canvas calls it whatever nobody painted, and fills it at the end

Four answers, none aware of the others. A program that wanted the ground gone
had to BORROW A CUT'S SIDE EFFECT to get it — which is what `Colors` was doing on
this puzzle. The input has one colour; there were no colour layers to make.

InputWithoutBackground says it once: the input with the elected ground punched
out and the frame kept. Dead when no ground is elected, read off the same
`invariant_colors` the kernel reads so the verdict cannot drift from it.

This is the sequel to `4 - To Do/Solution Skeleton From Inference.md`, which
moved ground REMOVAL and ground RESTORE into the engine on 2026-07-26 and deleted
`SelectInferredBackgroundColor` the same day. That was right and it paid (corpus
536 -> 393 tokens). What went with it was the ability to say either ON PURPOSE.
Removal came back as a side effect of cutting; restore came back as a property of
the canvas; neither had a word until now.

Why it was a SELECT back then, which is worth understanding: `Colors` used to
hand back the ground as its own colour layer, so the ground WAS an object and
picking it was an object-level mark. Today the cut skips it, so there is no object
to point at, and the same job has to happen at cell level — which SELECT cannot
do, because `selected` is one flag per object.


THE THING STILL OPEN: PREPARATION IS INVISIBLE
----------------------------------------------
This is the finding to carry forward. It is general, it is not fixed, and it is
why the puzzle needed a new word rather than a better search.

The scorer sees the canvas. A sentence that prepares the hand and paints nothing
scores as if it did nothing at all, which is the bottom. Measured on this puzzle,
BEAM_WIDTH 128, level-1 cut line 7.4982:

    Input SelectAll Colors Union            -17.4    1,976 candidates above it
    Input SelectAll Colors Intersection     -41.1   48,804 candidates above it

The SHORTER, better program scores WORSE, because Intersection on an empty canvas
paints nothing — which is precisely why it is better (no residue to clean up).

Worse, the surviving beam held one idea. All 128 were

    Input <one of 64 selects> ExtendTo[1] <Union|Underlay>

One GET, one ACT, 128 spellings. State-dedup kept them apart because the selects
leave different marks and state is canvas + object list — marks the next
sentence's own SELECT beat overwrites anyway. Not checked: why Union and Underlay
survived as two when both paint identically on an empty canvas.

The hard prefix idea is the answer already chosen for this: if inference emits
the opening sentence, the beam never has to score it. That is still unwired —
`5_sentences.hpp` is log-only.


TRAPS HIT TODAY
---------------
1. IMPORT COULD NOT REPORT FAILURE. `apply_solution` only ever set `is_solved`
   true, never false. Importing a failing program onto an already-solved puzzle
   still read as solved. Every import verdict before this fix is worth doubting
   unless the puzzle was reloaded first. FIXED — the old verdict now goes out
   with the old tree.

2. SCREENSHOTS OF A STALE BINARY. Two screenshots were taken of a build from
   11:31 to check a change compiled at 11:36, the new colour looked missing, and
   the wrong conclusion was nearly written down. The folder lock had correctly
   refused the second instance; nothing said the first one predated the code.
   The tell was in the receipt log the whole time — `listening on` was older than
   the exe. FIXED — `state` now ends with `build=`, and the receipt log opens
   with "running the binary built ...". Compare either against the exe on disk.

   It reads the file's MODIFICATION time, ONCE, at startup. Not `__DATE__`, which
   is only as fresh as the last time its own translation unit recompiled. Not
   creation time, which NTFS can carry across a rewrite. And not re-read on each
   question, because Windows lets the linker replace a running exe — verified,
   that is exactly what happened here, so a stale app would have reported the
   binary that superseded it.

3. NEARLY BUILT SOMETHING THAT EXISTED. `MaskOut[c]` looks like "punch out cells
   of colour c" but reads `ctx.output.data` — it is a CANVAS-keyed stencil, not a
   self-keyed one. Check what an op reads, not what its name suggests.


THE PALETTE NOW SAYS HOW MUCH IS INFERRED
-----------------------------------------
`how_much_is_inferred(node_type)` in `cuda_dsl.cuh`, beside `needs_a_pick` and
`hands_back_new_objects`. Yellow = ALL, orange = SOME, blue = none.

ALL (4) — nothing without the fact, so a dead-op guard is an honest verdict.
This IS the guard list, by construction:

    InputWithoutBackground, SizeRankColorMap, ShapeColorMap,
    ShapeToInteriorColorMap

SOME (5) — reads the fact, still does real work without one. Guarding these would
DELETE CAPABILITY, not prune waste:

    Split          still segments, just skips the erase
    Colors         still layers, ground included — which 0520fde7 needed
    VoteDenoise    falls back to its own local background, loses nothing
    Seeds          the elected ground removes whole objects from the pool before
    ProjectOnSeeds seeds are picked, which changes which one is largest and so
                   changes the whole list

Seeds and ProjectOnSeeds are empty when the grid has fewer than two objects —
geometry, not inference. So they can be no-ops without inference ever abstaining,
and a dead-op guard on them would be a verdict about the wrong thing.

The drawer used to open by mouse drag only, so the port could not see the one
panel that lists the ops. `toggle_op_list` joins the vocabulary.


THINGS VERIFIED BY IMPORT, KEPT HERE SO NOBODY RE-DERIVES THEM
--------------------------------------------------------------
    InputWithoutBackground SelectAll Fractal Union                  SOLVES
    Colors SelectAll Fractal Union                                  SOLVES (4w)
    Input SelectAll Colors Intersection / ... Fractal Union          SOLVES (8w)
    Input SelectAll Colors Union / ... Fractal / ... AddColor[0]     SOLVES (12w)
    Input SelectAll Fractal Union                                   fails
    Input SelectAll Fractal Union / ... AddColor[0] Union            fails
    Input SelectAll Colors Union / ... Fractal Union                 fails
    Input SelectAll Split[0|1|2] Intersection / ... Fractal Union    fails

Two things those rows teach. `Fractal` needs ONE object that still has the whole
frame with holes — `Split` shatters it and retightens each piece, `Colors` keeps
the source bbox. And `AddColor[0]` was never restoring the ground; the engine
does that. In the 12-word form it was covering sentence one's spill, and in the
old 5-word export it was doing nothing at all.


WHERE TO PICK UP
----------------
007bbfb7 is solved, exported and committed. The next unsolved puzzle is the next
job. Two things are worth carrying into it rather than rediscovering:

  - If a puzzle needs a preparing first sentence, search will not find it. That
    is the hard prefix's job and the emitter is still log-only.
  - Ask what an op READS before believing what it is named.

Not done, flagged, low cost: `logging 1` writes one [TREE] line per candidate
under a global mutex — 1.15 GB in 22 seconds on this puzzle. Left alone on
purpose today.
