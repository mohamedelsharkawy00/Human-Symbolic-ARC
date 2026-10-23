LEARNED ORDERINGS — design note
================================

The DSL exposes train-output-derived color palettes in multiple natural orderings.
The loader does NOT pick "the right" ordering — it computes projections, and the
SEARCH composes them with input-side ops. Train output is the verifier; if the
wrong ordering is paired with the wrong input transform, the program fails on
train and gets pruned. Right pairing survives.

Each `OutputColorsBy<X>` terminal:
  - At puzzle load: for each train output, segment by color, sort the resulting
    regions by criterion <X>, read off the color sequence.
  - Verify the sequence is identical across all train outputs. If not consistent,
    the terminal is INVALID for this puzzle (returns empty).
  - When valid, the puzzle stores: an ordered int8 sequence of colors (the same
    shape as `puzzle.output_colors`, just sorted differently).
  - At execution: returns one 1x1 object per color, in the inferred order.

Mirror ops live on the input side: `SortBy<X>` over an object list. The search
pairs an input-side `SortBy<X>` with an output-side `OutputColorsBy<X>` and
unifies them via the binary `Zip[Recolor]` op.

CURRENTLY IMPLEMENTED
---------------------
- OutputColorsBySize (descending pixel count of output regions)
- SortBySize[desc | asc]
- Zip[Recolor]  (arity-2 binary op — first binary op in the DSL)

TO ADD LATER (when puzzles need them)
-------------------------------------
- OutputColorsByAppearance — first-pixel reading order (this is the existing
  `OutputColors`; could be renamed for symmetry).
- OutputColorsByPosition — sort output regions by centroid in reading order
  (top-to-bottom, then left-to-right). Mirrors a future `SortByPosition`.
- OutputColorsByValue — colors sorted ascending by value. Mostly degenerate
  (output is already a set of colors, sorting them by value is trivially
  consistent across train); useful only when paired with input-side ops that
  also sort by color value.
- OutputColorsByFrequency — sort by pixel count of each output color, descending.
  Different from BySize: BySize segments and sorts regions; ByFrequency totals
  pixels per color across the whole output. Distinct when a color has multiple
  disconnected regions of differing sizes.
- OutputColorsByBoundingBox — sort by bounding-box area (≠ pixel count for
  irregular shapes).

DEGENERACY GUARDS
-----------------
- ByValue is trivially consistent across train outputs (sort key is intrinsic
  to colors themselves). It carries no real signal alone; only useful when an
  input-side op also sorts by color value. The search will discover this
  naturally — wrong pairings lose against right ones — but worth noting that
  ByValue should NOT be the canonical fallback.
- If multiple orderings are consistent for a puzzle, that's fine. Search finds
  whichever yields the shortest program. No need to pick one in the loader.

EXTENSION PRINCIPLE
-------------------
For every input-side `SortBy<X>` we want to support, there should usually be a
matching `OutputColorsBy<X>` terminal. They come in pairs because the search
needs both sides of the pairing. Adding one without the other usually means
a particular ordering can be expressed on one side but not the other, which
makes the pair untraceable.
