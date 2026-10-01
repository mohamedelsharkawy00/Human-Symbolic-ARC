# MaskBy / ClipToColor Design
Date: 2026-06-29

> **STATUS 2026-07-01: BUILT.** `MaskBy[c]` AND its sibling `MaskOut[c]` are in
> the X-macro (environment.hpp:143-144), the GPU enum, and share one predicate-
> mask executor in src/cuda_solver.cu (~2315; `keep_match` flag picks AND vs
> ANDNOT). `c == 10` = "any non-transparent" as proposed. The rest of this note
> is the original design rationale.

## Problem
We keep needing "paint `last` only where the canvas already has color C" — a
**non-destructive masked union**. Today none of the ops do it:

- `Union` paints `last` everywhere it's non-transparent (no mask).
- `Intersection` keeps canvas colors but clips the **whole** canvas to `last`'s
  footprint — destructive, can't be appended to gate a single layer.
- `IntersectionL` / `UnionL` are list-folds in **local** (top-left-aligned)
  coords, not world-space masks.

First hit: 05a7bcf2 — a red column painted full-height that must show only on
the azure (color-8) rows. Worked around with
`Target[10] Filter[8] TileX[1] Intersection` (gate the whole canvas to azure
rows), but that only works because nothing else lives on the gap rows; it's not
a general per-layer mask.

## Proposed op
`MaskBy[c]` (a.k.a. `ClipToColor[c]`), arity 1, annotated by color:
- Keep each cell of `last` only where the **canvas** is currently color `c`
  (drop the rest), then it composes with a following `Union` as usual.
- Non-destructive: it filters `last`, never clears the canvas.
- `c == 10` (or a sentinel) could mean "any non-transparent" for a pure shape mask.

Net: `…buildLayer…, MaskBy[8], Union` = "this layer, but only over azure."

## Notes
- This is the world-space AND of a layer with a canvas color — the missing
  counterpart to `Difference` (canvas − layer) and `Intersection` (canvas ∩ layer
  but canvas-wide).
- Sibling to [[DifferenceL Design]] in spirit, but canvas-referencing, not a list fold.
