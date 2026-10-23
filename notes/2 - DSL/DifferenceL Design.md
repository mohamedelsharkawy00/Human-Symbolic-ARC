# DifferenceL — DSL op design (arbitrary-first n-ary difference)

- **Date:** 2026-05-24
- **Status:** SPEC — not yet implemented.
- **Context:** see `6 - Handoffs/Handoff Kaggle Port.md` (UPDATE 2026-05-24) for the set-op family.

## What it is

The list-fold sibling of `UnionL` / `IntersectL` (same family; but NOT the same
implementation — see the caveat under "DO NOT rewrite" below):

```
UnionL      = obj0 U obj1 U obj2 ...                 pixel set in ANY object
IntersectL  = obj0 & obj1 & obj2 ...                 pixel set in ALL objects
DifferenceL = (((obj0 - obj1) - obj2) - obj3) ...    LEFT-associative fold
```

Read AND compute it strictly left to right: subtract obj1 from obj0, then subtract
obj2 from THAT result, then obj3, and so on. ONE subtrahend per step, in list
order. This sequential fold IS the semantics — it is not shorthand for anything.

**DO NOT rewrite this as "obj0 minus the union of the rest".** In pure set theory
`(((A-B)-C)-D) = A - (B U C U D)` by De Morgan — BUT this DSL's per-step difference
may carry subtle behavior (bbox handling, dropped/empty objects, no
re-segmentation, color/position quirks) that the set-theoretic shortcut does NOT
capture. So compute the LITERAL fold: apply one subtraction per step and let each
step behave exactly as the DSL's difference behaves. Do not assume the
union-of-the-rest identity holds here.

**NOT the same implementation as `UnionL` / `IntersectL`.** Those two compute their
n-ary result in ONE all-at-once sweep (`gpu_exec_union_list` paints all objects into
one merged bbox; `gpu_exec_intersect_list` tests "set in ALL" per cell) — which is
valid ONLY because U and & are order-independent + idempotent, so all-at-once ==
fold. `DifferenceL` must NOT copy that style: because `-` is order-dependent, an
all-at-once sweep would BE the forbidden De Morgan shortcut. So:

- **SHARED with `UnionL`/`IntersectL`:** scaffolding only — list → 1 object, the
  count<=0 (empty) / count==1 (passthrough) guards, direct mask access, and
  sitting beside `gpu_exec_union_list`.
- **NOT SHARED:** the fold itself — `DifferenceL` runs a genuine step-by-step subtract.

## Ordering: arbitrary first (deliberate choice)

The minuend is simply `objects[0]` — whatever sits at index 0 in the list. This
op does NOT pick the first by size/position.

Why: only the FIRST object matters (the rest are just unioned together, which is
order-free). That single choice is delegated UPSTREAM — the program sets the
order before `DifferenceL` runs, e.g.:

```
... -> SortBySize[d] -> DifferenceL      (largest- or smallest-first by sort dir)
... -> Largest[..]   -> ... -> DifferenceL  (curate the list, then subtract)
```

So `DifferenceL` stays a dumb, predictable fold and the beam composes existing
ordering primitives in front of it when a specific minuend is wanted. No size/
position heuristic is baked into this op.

Consequence (vs `UnionL`/`IntersectL`): Union and Intersect give the SAME result for
any object order. `DifferenceL`'s result depends on WHICH object is first; the rest
are symmetric. That is expected here — ordering is an upstream concern.

## Semantics (exact)

- **Coordinate frame:** WORLD coords (same as binary Difference and `UnionL`). Overlap
  between obj0 and the others is judged by absolute grid position, NOT by
  top-left-aligned shape. (`IntersectL` is the odd sibling that aligns at origin;
  `DifferenceL` follows Difference/`UnionL`, not `IntersectL`.)
- **Output:** ONE object = obj0 after subtracting obj1, then obj2, ... IN ORDER. At
  each step, clear every pixel of the running result that coincides (world space)
  with a set pixel of the current subtrahend. Keeps obj0's colors and position.
- Does NOT touch the canvas. It is a list op like `UnionL`/`IntersectL`; only the
  canvas ops (Union/Intersection/Difference) write `ctx.output`.

Edge cases:

- `count <= 0` : empty result.
- `count == 1` : passthrough obj0 unchanged (nothing to subtract).
- obj0 fully erased : return obj0 with all-transparent mask (count 1), matching
  how the other list ops emit a possibly-empty single object. Downstream paint/
  score already tolerates transparent-only objects.
- Optional: tighten obj0's bbox to surviving pixels. Skip for v1 to mirror binary
  Difference, which does not re-tighten.

## Boolean context (why it's a primitive, not a composition)

This is the ANDNOT-fold. With `UnionL` (OR-fold) and `IntersectL` (AND-fold) it
rounds out the n-ary boolean set. It must be a PRIMITIVE: it cannot be composed
from `{UnionL, IntersectL, Invert}` because Invert is bbox-LOCAL, not a true grid
complement, so the identity `A - B = A & ~B` breaks (see HANDOFF UPDATE 2026-05-24).
Implemented directly in world coords, `DifferenceL` does true subtraction.

## Implementation (3 touchpoints, ~30 lines)

**1. `include/environment.hpp`** — `NODE_PRIMITIVES` X-macro, next to `UnionL` (~line 127):

```cpp
X(DifferenceL, 1, "DifferenceL", Objects, Objects, None, 0, 1)
```

(0-arg op, no annotation variants — identical schema slot to `UnionL`.)

**2. `include/cuda_dsl.cuh`** — `gpu_op_type` enum, next to `UnionL` (~line 143):

```cpp
DifferenceL,
```

**3. `src/cuda_solver.cu`:**

**a)** New device fn next to `gpu_exec_union_list` (uses direct mask access, the
established list-op pattern). Structured as the LITERAL left fold: the OUTER
loop subtracts ONE object per step, in order — `(((obj0 - obj1) - obj2)...)`.
Do NOT collapse the two loops into a single "covered by any other" pass;
keep each subtraction a discrete step so per-step DSL behavior stays exact.

```cpp
// Left fold of binary difference over the list, world coords:
//   running = obj0;  running = running - obj1;  running = running - obj2; ...
// obj0 = src.objects[0] (ordering is an upstream concern). One subtrahend
// per OUTER-loop step, in list order.
__device__ gpu_object_list& gpu_exec_difference_list(
        gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    if (src.count <= 0) return res;
    if (src.count == 1) { res.objects[0] = src.objects[0]; res.count = 1; return res; }

    res.objects[0] = src.objects[0];          // running result = obj0
    res.count = 1;
    gpu_fixed_object& obj = res.objects[0];

    for (int i = 1; i < src.count; ++i) {     // step i:  running = running - obj_i
        const auto& sub = src.objects[i];
        for (int r = 0; r < obj.height; ++r) {
            for (int c = 0; c < obj.width; ++c) {
                if (obj.mask[r][c] == GPU_TRANSPARENT) continue;
                int wx = obj.x0 + c, wy = obj.y0 + r;     // world coord
                int lr = wy - sub.y0, lc = wx - sub.x0;   // local in sub
                if (lr >= 0 && lr < sub.height && lc >= 0 && lc < sub.width
                    && sub.mask[lr][lc] != GPU_TRANSPARENT)
                    obj.mask[r][c] = GPU_TRANSPARENT;      // remove this pixel
            }
        }
        // If a per-step convention is added later (re-tighten bbox, drop if
        // empty, re-segment), do it HERE -- inside the per-subtrahend step --
        // so the fold stays faithful to (((obj0 - obj1) - obj2) - ...).
    }
    return res;
}
```

**b)** Dispatch case next to the `UnionL` case (in the lane-0 switch):

```cpp
case gpu_op_type::DifferenceL:  result = last ? &gpu_exec_difference_list(ctx, *last) : &gpu_acquire(ctx); break;
```

## Verify after adding

- Windows (MSVC) build still compiles (user does this manually in the IDE).
- The `kaggle/input/solver-source-code/` copy is generated by `scripts/kaggle_push.py`
  — do NOT hand-edit it; it regenerates on push.
- Run EVAL_MODE (kaggle) and watch train% AND eval% together: a new op widens the
  beam's branching factor, so confirm it HELPS rather than diluting search on a
  fixed timeout. Reject the op if eval% does not move.
