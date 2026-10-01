CUDA warp-parallelization lessons learned
===========================================

LESSON 1: Warp-parallelism must target PIXELS, not OBJECTS.
------------------------------------------------------------

We made this mistake twice:

1. First in transform (FlipH/V, TurnL/R) — tried striping across objects first, only
   moved to pixel-level striping after seeing the regression.

2. Again in tile — kept lane-0 serial for the merged-build (which iterates pixels
   across objects) and only striped the 900-byte mask COPY (which is mostly transparent
   padding, not real work).

Why this keeps being wrong:
- Typical ARC workloads have FEW objects (1-9) and MANY pixels per object (up to 900).
- Striping 5 objects across 32 lanes → 27 lanes idle.
- Striping 500 pixels across 32 lanes → all lanes useful, ~16 pixels each.

Rule of thumb:
- If the inner work has 100s-1000s of similar-work elements → parallelize the inner loop.
- If the outer loop has <32 items → do NOT stripe the outer loop. Stripe inside each item.
- Object count is almost always small. Pixel count is almost always large. Target pixels.

Corollary — the "big copy" trap:
- A 900-byte mask copy LOOKS like a fat pixel loop, but if most cells are transparent
  padding (fw × fh much smaller than 30 × 30), you're parallelizing meaningless zeros.
- Measure real work (fw × fh actual pixels), not struct size.

When in doubt: find the innermost loop that iterates over real pixel work, stripe THAT
across 32 lanes. Keep scattered writes (overlapping-cell races) on lane 0 if needed,
but don't let lane 0 own a loop that has hundreds of non-racing pixel iterations.


LESSON 2: Parallelization cost must be amortized across REAL work per call.
----------------------------------------------------------------------------

Third variant of the same mistake (Union regression):

- Tile emits many 900-byte tiles per call → real work dominates → warp wins.
- Union's per-call work is a small h*w merged region, often < 100 pixels spread
  across several objects. Each __syncwarp() between objects costs more than the
  pixels it guards.

Sync math (rough):
- __syncwarp() alone is cheap (~4 cycles), but it collapses divergence and forces
  register spills; real cost often 20-40 cycles in context.
- An object with 5 non-transparent pixels striped across 32 lanes = ~5 active
  lanes doing 1 write each, then 32 lanes all hit a sync. Net work: 5 writes.
  Net cost: 1 sync. Sync wins.
- A 900-byte mask copy striped across 32 lanes = 28 writes per lane. 900 writes
  of real work per sync. Work wins.

Before warp-parallelizing an op, estimate:
  typical_work_per_call / syncs_per_call = bytes_per_sync
If that number is small (< ~50), do NOT parallelize — the op is bound by sync
overhead, not arithmetic. Leave it on lane 0.

Good candidates: ops with large output masks (tile, scale, transform; maybe
skew, pattern, trim, fit, interior — check the estimate above per op).
Bad candidates: ops operating on small object lists with per-object work
(union, difference, intersection, filter on small inputs).

Also: op FREQUENCY (firings/sec) is a weaker signal than per-call work. Union
fires more than Tile but each call is tiny — frequency-count in logs rewards
"how often does this op appear in winning programs" not "how much GPU time it
consumes." Don't pick targets by frequency alone.
