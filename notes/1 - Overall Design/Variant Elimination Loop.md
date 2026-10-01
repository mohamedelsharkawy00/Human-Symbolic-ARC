# The Variant Elimination Loop

> **HANDOFF 2026-07-03 — state + next build.** Layer locking (commit / freeze /
> beam-restart / enforce) is BUILT and verified firing (05a7bcf2: `Input Union`
> 692 cells → `Input BidiColorMap[24] Union` 705 → restart; every candidate
> extends the lock). Lookahead scoring is BUILT (dsl_score_kernel: hypothetical
> Union of the pending list, gain−spoil, W_LOOKAHEAD=2, GPU-only term).
> KNOWN-INSUFFICIENT: the commit criterion v1 (monotone correct-paint + kept-cell
> veto) assumes ADDITIVE solutions — it can never commit an Intersection/
> Difference layer (05a7bcf2's hand L2 clips the canvas, so no 3rd commit fires).
> DECIDED STATICALLY (2026-07-03, no experiment): KEEP the monotone ladder.
> The op-agnostic "checkpoint on beam dominance + backtrack" redesign was
> rejected by logic: dominance trusts the misleading scorer (the −12 wall) and
> makes its misjudgments sticky; backtrack under that scorer has no convergence
> story. The ladder commits on ground truth only (correct paint can't lie), and
> the corpus says 67/69 deposits are Union — ~97% native coverage. Clip-style
> solutions always have an additive equivalent (MaskBy/Filter paint instead of
> canvas clipping), so the ladder never blocks solvability — it just won't lock
> along a clipping path. Do NOT build backtrack until a stalled puzzle is
> TRACED to a bad lock (full-subset-reduction precedent: detection probe first,
> machinery second).
> Also: [TREE] log streaming produced 3.7GB in 3min — drop log mode before runs.
>
> **HANDOFF 2026-07-03 (later) — color-layer solving BUILT, end to end.** The
> ladder now solves ONE COLOR AT A TIME (user: "split the input into colors and
> solve each color as a layer"). What's in (solver_executor/solver_beam +
> cuda_solver):
> - **Commit = finish a color.** try_lock_layer commits ONLY when a root-Union
>   tree completes every target cell of a NEW color across all active pairs.
>   ("+N total cells" locking was tried and REMOVED — greedy micro-locks, one op
>   deeper each, burned the depth budget dry → busy-wait stall at 10M attempts.)
> - **Purity against the schedule** (v3, replaces the kept-cell veto): wrong
>   paint allowed ONLY on cells whose target color hasn't had its layer yet
>   (that color must repaint them to ever lock). Fully-strict purity was tried
>   and vetoes real solutions (the wall paints cells the towers cover later).
>   Hollowing undone colors is likewise legal (their own layer completes them).
> - **Focus color** (the human way): picked at start + at each lock = the
>   unfinished color with the largest residual (substrate/area first).
>   W_FOCUS=16 GPU score term — DOMINANT, not a nudge. 8 stalled
>   level-boundaries → rotate to next color (stalling discovers dependency
>   order); full circle stalled → free search.
> - **Level-boundary sweep**: top-64 root-Union beam survivors get the commit
>   check every level — a color-finishing layer does NOT need to beat the
>   global best score (the scorer is a misleading doorman).
> - **Per-layer depth budget**: depth counts from the LOCK
>   (LAYER_DEPTH_BUDGET=8 above the frozen prefix, capped near GPU_MAX_OPS) —
>   locking truly resets depth; solutions are many short layers, not one deep
>   chain.
> - **Lean locks**: qualifying trees are leave-one-out reduced before freezing
>   (raw fallback if the reduced form no longer qualifies).
> - **Promotion re-verify**: locks verified on fewer examples are re-checked
>   when active widens; on failure → "LOCK DROPPED", whole ladder rebuilds
>   (the human "my example-1 solution doesn't generalize" step).
> - **Inference**: color_diff_facts — raw per-pair per-color gained/lost cells,
>   deliberately NO interpretation (clean diff data; detectors later). Consumer
>   sketch: seed `InputFiltered[c], <verb>` chains at beam restart.
> RESULT: 05a7bcf2 example 1 solved by the ladder (~28M attempts, <5min) with a
> 5-layer chain nearly identical to the hand solution — same decomposition
> (substrate, towers, caps, wall, landed reds); its last layer is hand-L3 with
> ops reordered. Both example-1 solutions shelved in solutions/shelf/.
> Full-train generalization pending (the constructed wall layer is overfit).
> PRINCIPLE: no hand-seeding chains — the search FINDS layers, criteria judge
> them. SHELVED: background-born-done criterion (unpainted counts as correct
> under the detected background → kills the Input Union substrate copy; liked,
> but deferred — restore is 3 small edits in try_lock_layer/focus-init/
> promotion-reverify). KNOWN, unfixed: best-channel churn — steering terms
> (W_FOCUS, lookahead) inflate GPU scores so weird trees flash as "new best";
> fix = compare best-channel on the steering-free host score.
>
> **HANDOFF 2026-07-03 (final) — counterexample audit + the next two builds.**
> WHY THIS BLOCK EXISTS: color-commit worked end to end on 05a7bcf2, so the
> user asked the follow-the-precedent question (probe before machinery): does
> the corpus contain solutions the color ladder CANNOT climb? Scanned all 31
> chains in solutions/. Answer: yes, two classes — and their shape points at
> what to build next, which is why the two roadmap items below are written the
> way they are.
>
> COUNTEREXAMPLE CLASS 1 — transform/region layers. 00576224 (Fractal/Tile/
> SlamColumn deposits) and 11852cab (Input + TurnRight×3 deposits): every
> layer paints ALL colors at once (a tile, a rotation), so NO color completes
> before the final Union → the ladder never locks mid-chain, degrades to plain
> beam. Worse than mute: W_FOCUS=16 is DOMINANT and pushes toward "the largest
> unfinished color", which is meaningless steering on a symmetry puzzle —
> potential active harm, unmeasured.
> COUNTEREXAMPLE CLASS 2 — one color split across layers. 06df4c85 (palette
> boxes + Bridge in the same colors), 09629e4f: the same color arrives in two
> intentions; completion only fires at the LAST of them, so intermediate locks
> come late or not at all. Softer failure than class 1.
> (Also known: 0520fde7 uses destructive Intersection — 2/31, has an additive
> equivalent, ladder just won't lock along the clipping path. 13/31 chains are
> single-Union = single layer, ladder irrelevant there, harmless.)
>
> NEXT BUILD 1 — lock STACK + peel on promotion (replaces full ladder reset).
> WHY: promotion re-verify currently drops the ENTIRE ladder when a lock fails
> on the widened example set. That's correct but wasteful, and it's not what a
> human does — the user's own words: you keep the substrate layers that still
> hold and redo the one that broke ("repass on layers"). Mechanically: the
> ladder already climbs one lock at a time — KEEP the predecessors instead of
> discarding them (vector<node*> lock_stack). On promotion failure, pop locks
> off the top until the remaining prefix passes the same layer criteria on the
> widened set, restart the beam from that prefix, rebuild only the top layers.
> Cheap: the re-verify loop already exists (try_promote_examples), it just
> needs to walk the stack instead of nuking state.
>
> NEXT BUILD 2 — commit on PARTITIONS, not colors (colors become one channel).
> WHY: this single generalization dissolves both counterexample classes AND
> answers "should layering be shape-wise too?". The insight (user question →
> resolved in-session): shape-layering feels ambiguous only if it needs shape
> IDENTITY (fuzzy, inference-laden). Commit doesn't need identity — it needs a
> VERIFIABLE PARTITION OF TARGET CELLS. "Finish a part of some partition,
> purely" is exactly checkable for any partition:
>   - colors (BUILT — the current channel);
>   - target connected objects (exact cell sets, fixes class 2: each palette
>     box / bridge is its own part even when colors repeat — build FIRST, it's
>     easy and exact);
>   - symmetry regions / tiles / quadrants (fixes class 1: each TurnRight
>     deposit completes a quadrant — needs a region-partition detector in
>     inference.hpp, e.g. from tiling/symmetry facts, build SECOND);
>   - shape-as-guidance (focus on an object) can be a soft steering channel
>     later, separate from hard commit.
> Humans layer in many ways (color, object, region, rule) — but every way is a
> partition, and human verification is still cell-exact, same as the machine's.
> Focus generalizes the same way: focus = one PART (largest residual part of
> the active partition), rotation unchanged. Commit machinery (try_lock_layer,
> sweep, purity-against-schedule) is partition-agnostic already in spirit —
> the color arrays just become part arrays.
>
> ORDERING: build 1 before build 2 (peel is small and de-risks promotion for
> everything else); object-partition next; region detector after there's a
> puzzle in hand that needs it (probe-first precedent, as always).
>
> **LESSON 2026-07-06 — the valley, and focus seats.** A layer's staging ops
> can score worse-before-better (05a7bcf2 caps: fetch scores zero, MaskBy[8]
> scores NEGATIVE — pending reds sit on done-8 pillars — payoff only at the
> final MoveUntil). Rank-eviction kills such chains every level; no scoring
> weight fixes a valley. FIX (user: bias, never seeding): FOCUS SEATS — 64
> beam seats reserved for the best chains that FETCH the focus color,
> regardless of global rank (solver_executor, eviction block). Feels weird —
> protected chains look like junk for 3 levels — but it carried the exact
> staging (`InputFiltered[2] MoveRow[9] MaskBy[8]` = the hand solution's
> line, found verbatim) and the ladder then locked ALL FIVE COLORS = 900/900
> cells of example 1 in 8 min. Full stack that did it: vandal penalty (W=8,
> wrong paint on kept/done cells), W_FOCUS=16, sticky focus (never "none"
> while colors remain), focus seats, color-completion locks.
>
> **OPEN GAP 2026-07-06 — full lock that can't promote.** A lock explaining
> 900/900 of example 1 now hands itself to promotion (try_lock_layer), but if
> example 2 rejects it the failure is SILENT: no re-verify (promotion_count
> is 0), no drop, ladder idles complete-on-1. Needed: full-active lock that
> fails promotion → drop + rebuild under the wider set (the "my solution
> doesn't generalize" step, currently only triggered by successful promotion).
> Deferred cleanup: group the 8 loose focus_* fields into one focus_state
> struct with reset(). Code state: layer_fit() is the shared evaluator
> (commit + re-verify), MoveUntil renamed MoveToEdge, depth flat 32.
Date: 2026-07-02. Born from staring at 05a7bcf2's 8-line hand solution.

## The thesis (Shark's line, made the whole architecture)

> "Find an invariant rule to explain the variant, and make it invariant."
> — [[Dorr solving 05a7bcf2]]

ARC solving = explaining variants until everything is invariant. Not one
monolithic program found at once — **one variant at a time**:

1. **DIFF** — enumerate what varies (the residual). Diffs run in many
   directions (the inference direction taxonomy): input↔output per pair,
   across pairs, candidate↔target, layer↔layer.
2. **EXPLAIN one variant** — cheapest rule that accounts for it across ALL
   pairs (the three fates: constant / derivation / reject).
3. **FREEZE** — it's now an invariant; subtract it from the residual.
4. **RECURSE** on what still varies. Done when residual = 0.

Ordering rule: cheapest/least-entangled variant first (the ground/wall is
already invariant — just confirm it), most entangled last. Smallest energy
first, per the Dorr note.

## What it reframes

- **The scorer**: not "cells matching" but "variance remaining unexplained."
  The −12 wall exists because a productive prefix closes a whole variant
  DIMENSION and the cell-count scorer can't see it.
- **The search**: solve residuals, not whole programs. After a partial
  program locks correct cells, mask them from scoring and search the diff.

## Evidence: 05a7bcf2's hand solution IS this loop

Each line = one explained variant, frozen by a canvas op:
InputPalette (canvas) → yellow layer → red layer → red moved →
yellow→green → wall → background. Only expressible layered; one entangled
formula is hopeless. ARC-AGI-2 axis 1 (multi-rule entangled) is exactly
this shape.

## Generality guard

Layer by ROLE, not literal color — inference knows the roles
(in_every_input = ground, introduced = paint). "Solve the invariant layer
first, freeze, recurse" survives color-swap puzzles; hardcoded Filter[2]
doesn't. (Cost of naive color-layering: [[Arch Concepts P2 Future]].)

## Unifies (existing notes, one loop)

- Iterative Repair ([[Arch Concepts P2 Future]]) — the freeze/residual step
- Layering / Decomposition — variants often live one-per-role
- Coordinate descent ([[Search Strategies]]) — one branch at a time
- Diagnostic Backtracking — which step un-explained something
- Version-space intersection ([[Constraining Degrees Of Freedom By Invariants]])
- Figure/ground salience — ground = already-invariant; attend to variants
- LOOK/CONFIRM ([[Inference Notes]]) — explain on pair 1, confirm on rest

## First bites (smallest first)

1. **inference.hpp: `output_preserves_input` fact** (Identity-dims puzzles):
   per pair, fraction + mask of cells where input == output. High fraction
   ⇒ the first frozen layer is "copy input" — seed the beam with the
   Input-prefix and score only the changed cells. One detector + one
   consumer; also the first per-example-values fact (audit checklist).
2. **Residual scoring toggle**: score candidates only on cells the frozen
   prefix doesn't already explain.
3. **Freeze mechanic in the beam**: commit a locally-correct layer
   (canvas-op boundary), recurse on the diff — Iterative Repair made real.
