> [2026-07-02] Entry point for all inference work is now
> `1 - Overall Design/Inference Master.md`. This file remains the full salience
> rebuild spec (recognizer code, hook site, measure plan) — Master §4 item 1.

# Salience / Anchoring — the one lever that reduced attempts

*Written 2026-06-16. The original salience code was reverted (no git history to
recover it), so the code below is a RECONSTRUCTION SPEC grounded in the real
architecture — with the two known flaws already fixed. Treat the file/function
references as integration points to confirm at build time, not verbatim diffs.*

---

## TL;DR

- The beam solver is a **coin flip**: a fixed-width beam (1024) ranked by a
  **blind scorer**. Because the score gradient does NOT point at the answer
  (the productive prefix scores BELOW junk — the "−12 wall"), the trajectory is
  an unanchored random walk. Any change to the *candidate set* reshuffles it.
- **Measured, one session (0520fde7, baseline 21.7M attempts / 47.7s / solved):**

  | change | what it touched | attempts | result |
  |---|---|---|---|
  | **salience (re-rank by recognition)** | **RANKING** | **9.3M (2.3x ↓)** | **solved** |
  | baseline | — | 21.7M | solved |
  | union-suffix | candidate set | 26.1M | no solve |
  | read-the-stream | candidate set | 34.5M | no solve |
  | force introduced-color | candidate set | 41M | no solve |
  | graded shape match | candidate set | 47M | no solve |
  | double-Union guard | candidate set | 48M | solved, slower |

  **Salience is the ONLY change that drove attempts DOWN.** It is also the only
  one that changed RANKING instead of the candidate set. That is not a
  coincidence — it is the whole thesis.

---

## Anchoring (why this is THE lever)

Picture the search as a ball on a landscape; height = score, the answer is a
deep valley.

- **Anchored:** the landscape slopes into the valley. The ball rolls toward the
  answer on its own; nudge it (perturb candidates) and it rolls back. Stable.
- **What we have (unanchored):** a flat plateau with random bumps, and the
  valley walls (productive prefixes) score LOWER than the plateau (the −12).
  The ball is shoved by whichever bump is tallest. Any nudge sends it anywhere.

Two ways to act:
- **Change which candidates exist** (force / prune / graft / wrap) = rearrange
  bumps on the plateau → reshuffle the random walk → attempts went UP, every time.
- **Change how candidates are ranked** (salience) = tilt the plateau into a
  slope → attempts went DOWN.

Anchoring = make the score gradient actually point at the solution. Salience is
a partial anchor: right axis (re-rank by recognition), wrong calibration.

---

## What salience IS

A **re-rank of the beam's refinement slots by a recognition signal** — not the
per-cell score, a separate question asked of each candidate:

> "Is the target *derivable* from this candidate's output — i.e. is the answer
> visible inside it, modulo a licensed recolor?"

The recognizer is `find_in` (already in `include/inference.hpp`): slide the
target over the candidate's output, look for a placement where the target's
figure lands consistently (one color map) on the candidate's figure. A strong
hit means "this candidate already contains the answer's structure" → it deserves
a refinement slot even if the cell-anchored scorer ranks it low.

The beam refines only the top ~30 slots per level (beam order decides
refinement). Salience re-orders those slots by recognition, so a structurally-
correct candidate gets refined before coincidental-shape junk.

---

## The two flaws (and the fixes — this is the rebuild)

### Flaw 1 — over-fires (~97% of the beam got the bonus)

`find_in` today matches only the **figure**: non-empty needle cells must map to
non-empty haystack cells (`include/inference.hpp`, the window loop). It ignores
the **ground**. So almost any placement that covers the figure scores high —
including wrong ones — and nearly the whole beam gets boosted, which is the same
as boosting nobody.

**Fix — figure AND ground under one placement (De Morgan).** The insight from
0520fde7: *"the 2s are the intersection of the 1s"* is identical to *"the 0s are
the union of the 0s."* A real match must explain BOTH the figure (the 2s) and
the ground (the 0s) at the same placement. So recognition is the agreement over
the *whole* window, figure and ground:

```cpp
// include/inference.hpp — sits beside find_in. Returns a recognition score in
// [0,1]: how well `needle` (the target) sits at (r0,c0) in `hay` (a candidate
// output), scoring FIGURE and GROUND together. empty = the see-through value
// (only -1/TRANSPARENT is assumption-free; a color is a per-puzzle hypothesis —
// carry it, same caveat as find_in).
struct recognition {
    int   row = 0, col = 0;
    float score = 0.0f;       // (figure_hit + ground_hit) / window_cells
    int8_t color_map[10];     // needle color -> hay color; -1 = unused
};

inline recognition recognize_at(const grid& needle, const grid& hay,
                                int r0, int c0, int8_t empty) {
    recognition best;
    int8_t map[10]; for (int i = 0; i < 10; ++i) map[i] = -1;
    int figure_hit = 0, ground_hit = 0, cells = 0;
    bool ok = true;
    for (int r = 0; ok && r < needle.height; ++r)
        for (int c = 0; ok && c < needle.width; ++c) {
            int nv = needle.data[r][c];
            int hv = hay.data[r0 + r][c0 + c];
            ++cells;
            if (nv == empty) {
                // GROUND: needle says "empty here" — reward iff hay is empty too.
                if (hv == empty) ++ground_hit;
            } else {
                // FIGURE: needle says "color here" — reward iff hay has a
                // consistently-mapped non-empty color.
                if (hv == empty) continue;          // figure missing -> no credit
                if (nv < 0 || nv > 9 || hv < 0 || hv > 9) { ok = false; break; }
                if (map[nv] == -1) map[nv] = (int8_t)hv;
                if (map[nv] != hv) { ok = false; break; }   // inconsistent recolor
                ++figure_hit;
            }
        }
    if (ok && cells > 0) {
        best.row = r0; best.col = c0;
        best.score = (float)(figure_hit + ground_hit) / (float)cells;
        for (int i = 0; i < 10; ++i) best.color_map[i] = map[i];
    }
    return best;   // score 0 if !ok
}

// Best placement of the target anywhere in the candidate output.
inline recognition recognize(const grid& target, const grid& candidate,
                             int8_t empty) {
    recognition best;
    for (int r0 = 0; r0 + target.height <= candidate.height; ++r0)
        for (int c0 = 0; c0 + target.width <= candidate.width; ++c0) {
            recognition m = recognize_at(target, candidate, r0, c0, empty);
            if (m.score > best.score) best = m;
        }
    return best;
}
```

The key line vs the old version: `if (nv == empty) { if (hv == empty) ++ground_hit; }`
— the ground is now scored, not skipped. A placement that nails the figure but
sits on the wrong ground no longer scores ~1.0, so the bonus stops being
universal. Calibrate with a threshold (only re-rank when `score` clears, say,
0.9) so it fires on the few real structural matches, not the many partial ones.

### Flaw 2 — recognizer too costly (2.3x fewer attempts, but wall-time flat)

`recognize` is O(candidate_cells × target_cells) per candidate, on the host, for
every beam member every level. Even at 2.3x fewer attempts the wall didn't move.

**Fixes (pick per budget):**
- **Gate before recognizing.** Cheap pre-checks kill most candidates: dims must
  fit (`target ≤ candidate`), and the candidate's color histogram must contain
  the target's non-empty colors (modulo the recolor map). Only survivors pay the
  sliding-window cost.
- **Recognize only the slots you'd refine.** Don't score the whole 1024 — score
  the ~30 the beam is about to refine, to *re-order just those*. That is the
  actual mechanism and bounds the cost to 30 recognitions/level.
- **Move it to the GPU** if it earns its place on host first. The score kernel
  (`src/cuda_solver.cu::dsl_score_kernel`) already segments the candidate grid;
  recognition is a sibling pass over the same `ctx.output`.

---

## Where it hooks (the re-rank injection)

The cell-anchored score is built in two places that must stay in sync (see the
session notes): the GPU kernel `dsl_score_kernel` (`src/cuda_solver.cu`, the
`total = … shape*60 + color*50 + position*80 − …` fold) drives the live beam,
and the host mirror `environment::analyze_output_score` drives logs/verify.

Salience does NOT change `total`. It adds a **separate ordering key** used where
the beam picks its ≤30 refinement slots per level (in `solver_executor` /
`solver_beam`, the slot-selection that reads beam order). Sketch:

```cpp
// Pseudocode at the refinement-slot selection (per level):
//   candidates: the scored beam members eligible to be refined.
//   target:     the per-example target grid (already on the puzzle).
for (auto& c : refine_candidates) {
    recognition rec = recognize(target, c.output, /*empty=*/TRANSPARENT);
    // Re-rank key: keep the scorer's total as primary signal, but let a
    // confident recognition promote a structurally-correct candidate. Tunable.
    c.rank_key = c.total + (rec.score >= REC_THRESHOLD ? REC_BONUS * rec.score : 0);
}
sort(refine_candidates by rank_key desc);
take top ~30;   // these get refined this level
```

Two tunables — `REC_THRESHOLD` (how confident before it counts; start ~0.9 to
dodge Flaw 1) and `REC_BONUS` (how much a real match outweighs the blind score).
Both behind an app-state toggle (mirror `pin_color_literals`) for clean A/B.

**Important — the empty/see-through value.** `recognize` needs an `empty`. Only
−1 (TRANSPARENT) is assumption-free. For figure/ground puzzles the natural ground
is the puzzle's `background_color` (`inf.colors.background_color`) — but passing a
color as `empty` is a per-puzzle hypothesis (same caveat as `find_in`). Safest:
recognize with `empty = TRANSPARENT` first; only fall back to the background
hypothesis when the candidate has no transparent cells (already painted a 0
ground), and record that the bonus depended on it.

---

## Build & measure plan

1. Add `recognize_at` / `recognize` beside `find_in` in `include/inference.hpp`
   (pure, no engine deps — unit-test it on 0520fde7's target vs a known-good and
   a known-junk candidate output; junk must score < REC_THRESHOLD).
2. Bridge the per-example target grid to the slot-selection site if not already
   reachable there.
3. Add the re-rank key at slot selection, behind a toggle, default OFF.
4. Measure on 0520fde7: attempts AND wall-time, vs baseline 21.7M / 47.7s.
   - Success = attempts down (expect ~9M) AND wall-time not worse (Flaw 2 fixed).
   - Watch over-fire rate: log what fraction of refine_candidates clear
     REC_THRESHOLD. The old version hit ~97%; figure+ground should be a few %.
5. Only then consider the GPU port and an eval-wide run.

---

## Why this is worth rebuilding (not just a re-run)

Every other lever this session moved attempts the WRONG way because it perturbed
a chaotic, unanchored beam. Salience is the one that anchored — it tilts ranking
toward "contains the answer's structure," which is what the blind scorer cannot
see. The 2.3x was real on a rough, over-firing, host-only prototype. Done right
(figure+ground so it stops over-firing; gated/slot-scoped so it's cheap), the
attempt cut should hold AND convert to wall-time — and, unlike the other levers,
it gets *more* stable under perturbation, because a real gradient pulls the
search back instead of letting it wander.

Related: `INFERENCE_NOTES.md` (the prose record + `find_in` EMPTY SEMANTICS),
and the propose() alternative (recognize the puzzle, emit the concept program,
skip the search entirely — the other door out of the −12 wall).
