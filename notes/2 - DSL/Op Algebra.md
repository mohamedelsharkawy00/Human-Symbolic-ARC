# Op Algebra — completeness & color semantics
Date: 2026-06-30

The canonical reference for the set/compositing ops. When a puzzle makes you
juggle keep/remove/mask, find the cell here instead of inventing per-puzzle color
tricks.

## The two independent choices

Every binary op is really **two orthogonal decisions**, and the confusion comes
from them being baked together per-op:

1. **Footprint** — which cells survive: `OR` / `AND` / `ANDNOT`.
2. **Color source** — what color a surviving cell gets: **layer** / **canvas** /
   constant. ("Changing colors" is never its own op — it's `…, Recolor[c]` after.)

## Three families (by what they operate on)

| family | operands | OR | AND | ANDNOT |
|---|---|---|---|---|
| **Compose** (writes canvas) | layer ⊕ canvas | `Union` / `Underlay` | `Intersection` | `Difference` |
| **Select** (filters layer → layer) | layer, by predicate | — | `MaskBy[c]` | `MaskOut[c]` |
| **List-fold** (objects in `last` → 1) | objects within a list | `UnionL` | `IntersectionL` | `DifferenceL` *(NOT built as of 2026-07-01 — spec in [[DifferenceL Design]]; also a watch-item in 4 - To Do/TODO.txt)* |

## Compose, by footprint × color-source

This is the table that ends the "all over the place" feeling:

| footprint | color = **layer** | color = **canvas** |
|---|---|---|
| **OR** | `Union` (overlap→layer, "over") | `Underlay` (overlap→canvas, "under") |
| **AND** | *`MaskBy[10]` + `Union`* (idiom) | `Intersection` |
| **ANDNOT** | `MaskOut` (layer−canvas, as layer) | `Difference` (canvas−layer) |

Reading it:
- `Union`/`Underlay` are the OR row's two color-sources — the "over/under" knob is
  just *which side's color wins on the overlap*.
- **`Intersection` is the "under" flavor of AND** (overlap takes the *canvas*
  color). Its symmetric partner, **intersection-"over"** (overlap takes the
  *layer* color), is the idiom `MaskBy[10], Union` — expressible, not a new op.

## Precise color-wise definitions

- **`Union`** — cells = layer ∪ canvas; overlap colored by **layer** (later-wins).
- **`Underlay`** — cells = layer ∪ canvas; overlap colored by **canvas** (layer
  only fills transparent cells).
- **`Intersection`** — cells where **both** are non-transparent, colored by the
  **canvas**. The layer is a pure *stencil*; its colors are discarded. Outside the
  overlap → cleared. (Destructive: rewrites the canvas.)
- **`Difference`** — cells = canvas minus the layer's footprint, canvas colors;
  cleared where the layer covered.
- **`MaskBy[c]`** — layer cells where canvas == c (c=10 → canvas non-transparent),
  **layer** colors, produced as a layer (canvas untouched). Constructive AND.
- **`MaskOut[c]`** — layer cells where canvas != c (c=10 → canvas transparent),
  **layer** colors, as a layer. Constructive ANDNOT. `MaskOut[10]` = "keep the
  layer only in the gaps." See [[MaskBy Design]].
- **`Filter[c]` / `RemoveColor[c]`** — same as Mask*, but the predicate is the
  **layer's own** color, not the canvas's. (`Filter` keeps c, `RemoveColor` drops c.)

## The keep/remove 2×2 (selection)

| select by… | KEEP | REMOVE |
|---|---|---|
| the layer's own color ("what") | `Filter[c]` | `RemoveColor[c]` |
| the canvas underneath ("where") | `MaskBy[c]` | `MaskOut[c]` |

## Open cells

Both reachable as idioms; build only if one recurs (frequency discipline):

1. **Constructive OR** (canvas ∪ layer *as a layer*, canvas untouched) =
   `Union, Target[10]` (compose, then read the merged result back). A fused
   `Merge` op would be sugar for this pair — depth-compression only, not a gap.
2. **Constructive AND keeping *canvas* colors** — BUILT 2026-07-03 as
   `ReadUnder` ("what's on the canvas under this?"). First real demand:
   05a7bcf2's additive rewrite — it replaces the destructive
   clip-then-`Target[10]` staging idiom, keeping solutions Union-only so the
   layer-lock ladder can climb them. The algebra is now complete.

## Deferred redesign — color-source as a parameter

The algebra is functionally complete, but the *color-source* knob is exposed
inconsistently: an op-pair for OR (`Union`/`Underlay`), a different mechanism for
AND (`Intersection` vs `MaskBy`+`Union`). For true symmetry the principled fix is
**not** more ops — it's a unified compose family where color-source is an
annotation:

    Compose[footprint, color_source]
      footprint    ∈ {OR, AND, ANDNOT(c−l), ANDNOT(l−c)}
      color_source ∈ {layer, canvas, const c}

This collapses `Union`/`Underlay`/`Intersection`/`Difference`/(idioms) into one
annotated op with a uniform knob.

**Why deferred:** it's a redesign that touches every solution's import tokens and
the search's op vocabulary, and it trades the current flat, individually-named,
search-friendly ops for one high-arity-annotation op (bigger per-op branching).
Adopt only if the idiom-juggling (`MaskBy[10],Union` etc.) actually starts costing
real puzzles. Until then, **this note is the completeness map** and the idioms
cover the gaps.
