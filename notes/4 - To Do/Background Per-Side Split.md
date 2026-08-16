# Background Per-Side Split (STARTED 2026-07-19 — steps 1 and 2 done)

> UPDATE 2026-07-26: the missing consumers now exist, one per side. The INPUT
> verdict is consumed by segmentation (`gpu_exec_colors` skips it,
> `gpu_exec_split` erases it), and the OUTPUT verdict is consumed by the closing
> frame after eval. Both sides are therefore load-bearing in the kernel, not just
> displayed. See `Solution Skeleton From Inference.md`. What remains open here is
> the per-pair case and the truth file asking the two sides separately.

The election gives ONE verdict, but input background and output background are
genuinely different facts — their relationship IS part of the transformation
rule. The data already elects both sides independently (`across_inputs`,
`across_outputs` in `background_hypotheses`); what is missing is consumers and
the truth file asking the two sides separately.

## Poster-child puzzles (the single verdict cannot say the truth)

- **20818e16** — input-bg per-pair (8,4,0), output-bg none: the transformation
  crops each object's bbox and deletes the ground. Single verdict compromises
  to per-pair.
- **25e02866** — same family as 20818e16: input-bg per-pair (2,8), output-bg
  none.
- **2f0c5170** — input-bg azure (73–89% every input), output-bg black (68–89%
  every output): the figure is cropped off the azure field and re-planted on
  black. Cleanest case: both sides elected correctly and independently; the
  single verdict says 8 (right for segmentation) but the two-sided truth is 8/0.
- **27f8ce4f** — input-bg none (tiny 3×3 figure-only grids), output-bg black:
  the 9×9 fractal tiling introduces the canvas. Handled for now by the
  weakest-tier "output-only ground with figures on it (embeds > 0)" gate, so
  the single verdict says 0 — but the honest answer is two-sided.

## The plan

1. DONE — `tests/background_truth.txt` is two-sided (`<id> <input-bg>
   <output-bg>`), checked by `scripts/check_background_truth.py` (29/29).
   Input is the PRIMARY side: where it speaks, an abstaining output is
   acceptable.
2. DONE — `input_elected()` / `output_elected()` are real per-side verdicts in
   `background_hypotheses`. The output side is gated by the same
   dominance-confidence bar (>= 500) as per_pair — one-sided evidence must
   truly dominate; embeds proved wrong as a gate (worst-grid number, zeroed by
   one grid whose figures majority-border something else). `elected()` remains
   the input-first one-answer merge for existing consumers.
3. TODO — the relation between the two sides (kept / recolored / deleted /
   introduced) becomes a learnable fact of the transformation.
4. TODO — **UP NEXT** — the per-grid vs per-pair distinction. Walk paused at
   6a1e5592 (first 400 of training) on 2026-07-19; resume there after this.
   PER-GRID background (50a16a69: only input 0 has a canvas, azure;
   the rest are backgroundless). The per_pair column already computes the
   per-grid verdicts (8,-1,-1) since it elects per input; what is missing is
   a consumer that reads it per grid instead of demanding all-pairs coherence.
   "Per-pair" in our vocabulary really means per-INPUT-grid — the distinction
   bites when an input and its own output need different canvases.

## The consumer contract: tolerate absence and error on EITHER side

Both verdicts are optional and fallible, independently. Every consumer must
hold to this:

- **-1 is "no information", never a color.** Input -1: segment nothing out
  (all cells are content). Output -1: no construction prior, no candidate
  rejection — build and verify without it. Neither side's -1 may block a
  consumer from working; it only removes a shortcut.
- **A verdict can be WRONG.** No consumer may hard-commit in a way that makes
  a wrong background unrecoverable: exclusions, priors and rejections must be
  preferences the search can climb out of. The ranked list is the recovery
  path — the solver's stall event re-runs under the runner-up lens.
- **The sides fail independently.** Input elected with output -1 (57edb29d),
  output elected with input -1 (27f8ce4f), both -1 (backgroundless, texture),
  both elected but different (2f0c5170) — all four are normal states, not
  edge cases.

## Consumer plan for the color-literal fan (decided 2026-07-19)

Measured over the 37 exported solutions: ~40% of all color-literal arguments
in real programs are just "the background" (RemoveColor[0], AddColor[0],
Filter[0], plus Foreground). Three escalation levels, in order:

1. NOW — RANKING: enumerate each color-literal fan through a per-puzzle
   permutation with the elected input bg FIRST. Wrong/absent election costs
   nothing (everything is still tried, just later); a right one moves the
   solution earlier in each level's budget. Honors the tolerance contract.
2. REJECTED (2026-07-20) — hard pruning of bg-semantic literals. Tried
   (RemoveColor -> input canvas only, AddColor -> elected canvases only): the
   13-puzzle re-export ran 2x faster overall and every stressed puzzle
   rerouted — but it trades COMPLETENESS for speed, and in ARC a lost puzzle
   outweighs any speedup. User verdict: steer, never kill.
3. LATER — CANDIDATE-SIDE STEERING instead of literal pruning: when an
   output canvas is elected, prefer/produce candidates whose OUTPUT actually
   sits on that canvas (score prior or generation bias on the result, not the
   op) — the literal fan stays complete, the beam just points at answers with
   the right ground.
4. LATER — EQUIVALENCE MERGE: RemoveColor[bg] == Foreground == filter-of-
   everything-else once bg is known; dedupe to one branch.

## Why step 3 pays: the relation hands the solver a layer of the answer

Same move as the size rule, one level up. From "every output is 3x the input"
the solver knows the test output's DIMENSIONS before any search; from "every
output sits on black" it knows the test answer's GROUND LAYER before any
search — start from that canvas and search only for the figures. Each relation
kind constrains differently:

- **kept** (most puzzles) — paint the same ground, search only figure changes.
- **introduced** (27f8ce4f: none -> 0) — the output canvas is a constant,
  known at load time.
- **deleted** (20818e16: per-pair -> none) — the answer contains no ground
  cells at all, killing every candidate program that leaves background in.
- **recolored** (140c817e: per-pair -> blue) — ground maps to a constant, the
  figures carry over.

Without the fact, the beam rediscovers each of these per puzzle, as ops, at
search prices. With it, it is a load-time verdict — the "push inference
upstream of the ops" principle applied to the ground layer.
