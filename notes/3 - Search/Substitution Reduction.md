# Substitution Reduction (idea, not built)

Status: proposed 2026-07-11, after the 007bbfb7 lock-leanness work. We might build it. Companion to [[Program Reduction Greedy Vs Fullsubset]].

## Where the idea came from

After the full-train lock vetting + certification-based lock cleaning landed, 007bbfb7 converges to a 6-node tree:

    Target[0] Union Input RemoveColor[0] Fractal Union

The pre-locking baseline solution was 4 stored nodes:

    Input RemoveColor[0] Fractal AddColor[0] Union

Both are fully correct. The 2-node gap is structural, and deletion-only reduction can never close it.

## Why deletion-only reduction has a floor

1. Reduction today only removes ops. Every surviving op is provably load-bearing (certification: same cells explained, same colors finished).
2. The 6-node tree pays for the background with its own layer: `Target[0] Union` deposits an all-0 canvas. Deleting it loses 288 certified cells, so deletion correctly refuses.
3. The 4-node tree pays for the background differently: `AddColor[0]` fills the fractal object's transparent cells in-line, so one deposit does both jobs.
4. Getting from form A to form B is a rewrite, not a removal. No sequence of deletions reaches it. That is the floor: deletion-only reduction finds the shortest *subset* of the found tree, not the shortest *equivalent program*.

## Why the floor exists at all: layer locking

The ladder solves one color per layer and every locked layer costs its own deposit (`... Union`). So lock-mode solutions structurally carry roughly +2 nodes per layer versus what a flat search might find in-line. That is the same trade as the lock-mode time overhead on easy puzzles: spend a little on puzzles simple enough to go flat, buy tractability on puzzles that are not.

## The mechanism

Extend reduction with a small set of *rewrite moves*, tried under the exact same acceptance rule as deletions (certification preserved, fewer nodes). Each move is a local pattern -> replacement on the flat op list:

1. **Layer folding** (the motivating one): a deposit layer that paints a single uniform color everywhere transparent
   - pattern: `X Union <rest> Union` where the `X Union` layer's only certified contribution is "cells of color c"
   - replacement: `<rest> AddColor[c] Union`
   - net: -2 nodes when it fires (one layer disappears, one op appears)
2. **Inverse-pair cancellation across a barrier**: `FlipH <mid> FlipH` -> `<mid>` when `<mid>` commutes with the flip. Deletion already catches this when removal alone survives certification; the rewrite form catches cases where an *orientation-corrected* middle is needed (e.g. replace with `<mid'>` = the mirrored op variant).
3. **Constant folding**: two adjacent ops with a known composition law collapse to one (`TurnLeft TurnLeft` -> `Turn180`-style, wherever the DSL has the composite op).

Moves 2 and 3 need a small table of op algebra (which ops invert, commute, compose). Move 1 needs no table: it is checked empirically by building the candidate and re-certifying, same as every deletion today.

## Why we might want it (the reasons)

1. **Generality of the answer.** Shorter equivalent-on-train programs are better bets on test (minimum description length). Every node removed is one fewer place a train-only coincidence can hide. This is the real payoff, not aesthetics.
2. **Depth budget inside the ladder.** Lock-bound cleaning with folding makes frozen prefixes shorter, and every node saved in a lock is search depth returned to all later layers. Deletion cleaning already buys some of this; folding buys the structural +2 per layer back.
3. **It reuses everything.** Same flat-program representation, same certification acceptance, same call sites (deep mode at lock commits and finale). A move is just another candidate generator next to "drop k ops".

## Why we might not (the costs)

1. **Move tables are a maintenance liability.** Op algebra facts (inverts/commutes/composes) are per-op knowledge that must track every DSL change. Wrong algebra = wrong rewrites accepted only if certification catches them — it will, but then the table is dead weight.
2. **Search-space creep.** Deletion candidates are C(n,k); rewrites multiply that. Fine at lock commits (rare), dangerous if it leaks into the hot path.
3. **The payoff is bounded.** On correctness it buys nothing — the 6-node tree already solves test. It buys nodes (generality margin) and lock depth. If benchmarks show locks are rarely deeper than 2-3 layers, the depth argument is weak and only the MDL argument remains.

## Decision rule

Build move 1 (layer folding) alone first — it is table-free, targets the known structural cost, and its win is measurable (locked-prefix node counts across the benchmark set). Only add moves 2-3 if benchmark solutions visibly carry foldable patterns that move 1 misses.
