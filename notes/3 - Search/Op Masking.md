Op Masking (= Soft Filtering of Primitives Per Slot)
====================================================

Definition
----------
Op masking = at each generation step, prune or reorder the primitive list
so that primitives most likely to help on this puzzle/slot are tried first.

Hard mask: skip primitive entirely. Fast but risks removing the one op
  that would have solved the puzzle.
Soft mask: reorder by priority. Slower in worst case but no solution is
  unreachable; in practice high-priority primitives find candidates
  before the low-priority tail is reached, so cost is similar to hard mask.

Default to soft masking. Hard mask only when the primitive is provably
irrelevant (e.g. type mismatch, already done elsewhere).


Why It Matters
--------------
Branching factor at each generation step is roughly:
    primitive_count × |history_pool| × |slots_to_refine|

Cutting primitive_count is the highest-leverage lever because it
compounds multiplicatively across depth. Cutting it to a quarter turns
depth-8 search into orders-of-magnitude cheaper without changing
architecture. (Current count: the NODE_PRIMITIVES X-macro in
environment.hpp.)


Why Op Masking Beats Bidi (For This Codebase)
---------------------------------------------
1. No architectural rewrite. Bidi requires inverse semantics for every
   op, a backward execution path on GPU, and search-graph orchestration
   to merge forward/backward frontiers. Op masking is a reorder of the
   existing primitive iteration loop. Days vs months of work.

2. Bidi without neural guidance often LOSES. The Alford 2021 paper
   (the canonical bidi-for-ARC reference) showed bidi tied forward-only
   on ARC symmetry tasks; it only won on domains with strongly
   constrained inverses (24 Game, double-and-add). Without an RL policy
   choosing direction at each step, bidi multiplies branching across
   two directions instead of cutting it.

3. Op masking attacks the actual bottleneck. Huge search space here
   means too many candidates per step, not too many depth levels.
   Bidi cuts effective depth (N → N/2) but does nothing about the
   per-step branching factor. Masking attacks the per-step factor
   directly, which compounds across all depths regardless of search
   direction.

4. Composable. Op masking benefits both forward search AND any future
   bidi attempt. Time spent on masking is never wasted; time spent on
   bidi infrastructure is wasted if we later decide forward+masking
   suffices.

5. Incremental and measurable. Each mask (color, shape, histogram)
   can be added independently and benchmarked. Bidi is all-or-nothing
   architecturally — you can't half-build it.

6. No risk of pruning the solution. Soft masking only reorders;
   the search remains complete. Bidi adds complexity (backward states,
   meet detection) where bugs can silently cause solutions to be missed.


Mask Types Ranked By Generality


Mask Types Ranked By Generality
-------------------------------
(Generality = fraction of puzzles where the mask gives useful signal)

1. Color delta masking
   - Every puzzle has a target color set. If target colors are {X,Y,Z}
     and canvas has {X}, prioritize ops touching {Y,Z}, deprioritize
     ops touching unrelated colors.
   - Applies to 100% of puzzles. Strongest general signal.

2. Histogram delta masking
   - Like color delta but with magnitude/direction. If target has 5
     more pixels of color 4 than canvas, ops adding color 4 ranked up.
   - Strict superset of color delta but signal goes flat when colors
     are correct and only arrangement is wrong.

3. Dimension/shape masking
   - Every puzzle has a size_rule (Identity, Fixed, Scale, BoundingBox)
     already detected in detect_size_rule.
   - rule_type == Identity → deprioritize ScaleUp/Down, TileX/Y,
     DiagTile, TileByInputPattern, Tile, Fit (the 8 size-changing ops).
   - rule_type == Scale → prioritize those same ops.
   - Precompute once per puzzle, not per slot. Zero per-step cost.
   - Applies to 100% of puzzles.

4. Type-flow refinement
   - Current Objects/None binary split could be finer: single-object
     vs list, rectangular vs irregular. Needs new type metadata.
   - Incremental win over existing type_arity gating.

5. Position/translation masking
   - When canvas content is shape-correct but offset, prioritize
     coordinate-adjusting ops (Trim, Origin, AlignToCanvas, Shift,
     Move*, Fit). Detectable by checking if a translation of the
     canvas would match the target.

6. Domain-knowledge primitive grouping
   - "Looks like a tiling task → prioritize Tile family." Requires
     classifying the puzzle. High value when it fires, brittle.

7. Symmetry detection masking
   - If input/output share a symmetry axis, prioritize Flip/Turn ops.
   - Narrow but high-confidence when applicable.


Suggested Implementation Order
------------------------------
Start with dimension/shape masking:
  - Already computed (size_rule per puzzle in detect_size_rule)
  - Precomputable once per puzzle
  - Large effect (the 8 size-changing ops reordered for majority of puzzles)
  - Low risk: soft reorder, never eliminates an op

Then add color delta masking:
  - Per-slot precompute of target_colors vs canvas_colors
  - Reorder primitive_types iteration order in generate_phase_1
  - Histogram delta is the natural extension once color delta works

Stack from there. Each mask is independent and measurable in isolation.


Soft vs Hard Note
-----------------
A primitive being deprioritized still gets tried — just later. The
existing dry-yield exits (m_dry_yields >= 50, m_consec_dry_beam_entries
>= 2) and slot caps (MAX_CANDIDATES_PER_SLOT) cut the low-priority
tail naturally when high-priority ops are productive. If high-priority
ops are unproductive, we fall through to the rest of the list — no
solution becomes unreachable.
