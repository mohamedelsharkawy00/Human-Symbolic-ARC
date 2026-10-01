# Map Op Consolidation
Date: 2026-06-27

## Idea
The shape->color "map" ops should share one naming convention and one piece of
machinery. Adopt `<Key>To<Value>Map` and factor out the common parts.

Current ops in the family:
- `ShapeColorMap` — key = a glyph's shape (occupancy hash), value = the color the
  *largest target body* is recolored to. Single key->single target per scene.
- `ShapeToInteriorColorMap` — key = each object's own shape, value = the color its
  *enclosed interior* is filled with. Per-object. Added for 00dbd492.

Both hash via `arc::shape_occupancy_hash` and both read the same puzzle-level
table (`shape_color_keys/vals/count`). `infer()` populates it with the body-
recolor detector first, then the interior detector as a fallback when the first
rejects (a puzzle uses at most one).

## To do
1. Rename `ShapeColorMap` -> `ShapeToColorMap` for convention consistency. NOTE:
   - enum order must not shift existing ordinals (hash/search order depends on it,
     per Contest Hardening Plan) — keep its position, only change id + name string.
   - the name string is the import token: update `solutions/009d5c81.txt`
     (and any other file referencing "ShapeColorMap") in the same change, or
     import breaks.
2. Factor the shared occupancy-hash + table lookup out of the two device ops
   (`gpu_exec_shape_color_map`, `gpu_exec_shape_to_interior_color_map_warp`) and
   the two host detectors.
3. Once a third `…To…Map` lands, consider a single table that tags each entry with
   its application mode (body recolor vs interior fill) instead of the
   first-wins fallback in `infer()`.

## Related
- `OverlapCorner` general form (sequential/corner/stride generalization of
  AlignToCanvas, + `[dir]`/`stride` annotations) is documented inline in its op
  comment in `src/cuda_solver.cu`, not here.
