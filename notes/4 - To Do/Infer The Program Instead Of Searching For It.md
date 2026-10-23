# Infer the program instead of searching for it

Parked 2026-08-27, from "how about we make ALL the ops inference".

## The idea

Every op becomes something read off the train pairs, and the program falls out of
the reading. Not the search moved into the inference layer — **inferred, not
searched.**

## The line that matters

It is NOT "search versus no search". Inference searches all the time already:
`find_in` sweeps every placement of a glyph over a grid — 841 of them on a 30x30
— and `fit_the_pieces_together` is a backtracking exact cover with a step budget.

The line is:

    work bounded by THE PAIR          ok
    work bounded by HOW MANY PROGRAMS not ok

Local and exhaustive is fine. Combinatorial over compositions is the thing to
kill.

## How a chain forms without trying chains

One answer only: **the residual.**

    read the most certain thing
    apply it
    look at what is LEFT
    read the next thing from that
    stop when nothing is left

Each step is chosen by what remains, never tried against alternatives. This is
the same shape as "certain first, then multipass" — the reading of 05a7bcf2 that
started all of this — but at program level instead of match level.

## What already exists

- **The freedoms ARE ops.** `MOVE` is `Shift`, `TURN` is
  `TurnRight`/`FlipHorizontal`, `COLOR` is `Recolor`, `HANG OFF` is the canvas
  edge. Correspondence already has an op vocabulary; it is four of the seventy
  in `cuda_dsl.cuh`, and nothing says why it stops there.
- **Three detectors are already op-shaped.** `shape_color_map_rule` carries a
  `claimed_by` field naming which op will use it — `ShapeColorMap` versus
  `ShapeToInteriorColorMap`. `color_swap_rule` is a recolor. The size rule is
  `ScaleUp`/`ScaleDown`/`Layout`.
- **The residual already exists.** `pair_info::left_over` is exactly "the cells
  this reading could not account for", kept as CELLS and not a tally, on both
  grids. Today it is a penalty in the price. Under this idea it is the input to
  the next step.
- **The intent is written down.** `4_infer.hpp`: "propose() deferred (see
  notes)", and "Backward Deduction step 1, applied to detectors instead of ops".

## The one concrete gap

**Inference never applies anything.** Nothing under `include/inference/`
produces a grid. `infer()` is pure — train pairs in, facts out. So there is no
"the input after this step" and nothing to take a difference against.

That single absence is what makes this impossible today, and it is small.

## The danger, already met once

A residual loop is greedy with no backtracking, and that is exactly how the
staging experiment failed on 2026-08-26: rounds of increasing freedom, each
freezing what it explained, made the fitting WORSE — 05a7bcf2 went from 239
matches to 400 — because the fitting's whole power is that everything competes
at once, and backing out is what rejection IS.

So a step may only be taken if the examples corroborate it, not merely because
it is the best thing available. That is the cross-pair sieve, one level up.

## Interpretations that were considered and not chosen

1. **One detector per op.** 70 detectors, "do the pairs show this op". Three
   exist. Says which ops are present, not in what order with what arguments.
2. **Ops as more freedoms.** `SCALE`, `NEGATE`, `OUTLINE` beside `MOVE`/`TURN`/
   `COLOR`. Blocked by arithmetic: the table is `cuts x 2^freedoms`, 5 x 16 = 80
   today, and every freedom doubles it. Seventy ops as freedoms is 2^70, so ops
   could only ever enter singly — "which one extra op" — not as a power set.
3. **Ops as more cuts.** Rows, not columns; linear by construction. Each op is a
   way of READING the pair as a whole, priced like a reading.
4. **Inference emits the program.** Correspondence already computes "this piece
   went there, recoloured", which IS an op with arguments; emit the chain
   instead of facts for the search to re-derive.
5. **Search disappears.** The strong form of 4 — and the one picked, with the
   constraint that it must be inferred and not merely relocated.

## First step when this is picked up

Make the residual exist: apply the elected correspondence to each input, produce
the difference, and log it. One function, no election change, testable on the
six puzzles that have been read by hand (05a7bcf2, 045e512c, 03560426,
025d127b, 00dbd492, 009d5c81). It answers immediately whether what
correspondence reads is enough to take even a first step.

## See also

- `A Piece Is Credited With Cells It Never Carried.md` — the leak that lets a
  reading score a perfect zero without carrying anything, which any residual
  loop would inherit as "nothing left to do".
