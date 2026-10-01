> [2026-07-02] Entry point for all inference work is now
> `1 - Overall Design/Inference Master.md`. This file remains the detailed
> code walkthrough of include/inference.hpp.

# inference.hpp — the design story, pulled out of the comments
*(generated 2026-06-11 from the comment blocks in include/inference.hpp, so it can be read without wading through code)*

## What this file is

The "read the examples like a human" layer. A human does NOT solve ARC by blind
search over millions of programs — they first read the rules the examples hand
them for free (the **invariants**), and only then propose a transformation that
fits. inference.hpp is that front-end: given a puzzle, it reads invariants off
the train pairs so the beam solver doesn't have to discover everything by luck.

## The design contract

- **`infer()` is PURE**: puzzle in, an `inference` object out. It does not
  search, does not touch the GPU, and does not know the beam solver exists.
- **Dependency (deliberate, for now)**: inference.hpp is a *child* of
  environment.hpp (it includes it for the shared types: puzzle, example,
  size_rule). The end-goal is the reverse — extract the pure data types into a
  neutral root header both include, so inference depends on no engine logic.
  That root-flip is a separate, smoke-tested job. Until then the engine reads
  migrated invariants via a bridge onto puzzle fields (in puzzle_loader).
- **Growth is additive**: new invariant extractors append, each carrying an
  INSPIRED-BY block naming the puzzle that motivated it.

## The backbone: directions

Every invariant is a fact read by comparing grids along one of four
directions, **listed in human reading order** (the pair is read first, the
set-wide sweeps after — the enum order is this order, deliberately):

| direction | compares | meaning |
|---|---|---|
| `across_input_output` | input <-> output | the relation between **one pair's two grids** — read BOTH ways (find the output in the input AND the input in the output). The old name `in_to_out` implied one arrow; the walked trace of 0520fde7 showed the first look is inherently bidirectional, so the "to" is gone. |
| `across_inputs` | input <-> input | what is common to EVERY train input |
| `across_outputs` | output <-> output | what is common to EVERY train output |
| `across_inputs_outputs` | every grid | facts over ALL train grids, inputs and outputs together. Earned by an existing detector: `allowed_colors` is the union over every train grid (it had to call itself "across_inputs ∪ across_outputs" because this word didn't exist). |

**Naming rule (load-bearing grammar):** the plural S marks the scope.
`across_inputS` ranges over the whole train set; `across_input_output` (no S)
ranges over the two grids of a single pair.

## Two phases (learned from walking a real solve, not designed)

- **LOOK** — read ONE pair (pair 1 first) along a direction. Cheap, concrete,
  hypothesis-forming. A look may be wrong; that is its job.
- **CONFIRM** — replay the look's story on every remaining pair. A story that
  survives all pairs is an INVARIANT; one that dies is discarded — and HOW it
  dies matters: a counting failure can kill a whole hypothesis CLASS (e.g.
  "output sparser than every part" rules out selection entirely, not just one
  selection).

**The invariant rule:** a fact is an invariant ONLY if it holds in EVERY train
pair. Each extractor checks this internally and returns only consistent facts —
a fact seen in just one pair is a fluke and is silently dropped. (The existing
extractors fuse LOOK and CONFIRM in one loop; the phases are still there, just
inlined.)

## The channel loop (2026-06-11): directions became executable

The directions started as a taxonomy in comments; the color-presence channel
now RUNS through them. Key realization: a channel is a per-grid SUMMARY
function plus a FOLD, and the direction IS the fold:

- **Scope directions** (sets of grids — inputs / outputs / all) take two
  generic folds: `every` (present in EVERY grid of the scope, AND) and `any`
  (present in AT LEAST ONE, OR).
- **The relation direction** (`across_input_output`) compares each pair's two
  grids, then ANDs across pairs: `consumed` (in input, gone in output),
  `introduced` (absent in input, in output), `preserved` (in both).

`detect_color_presence` fills every (scope x fold) cell + all three relations
in one pass. This systematically EXPOSED four cells nobody had thought to
write: colors-in-every-output (the answer's mandatory palette),
colors-in-some-input, colors-in-every-grid, and `colors_preserved`. That's the
point of forcing the directions: the grid generates coverage instead of
documenting it.

Bespoke cells stay bespoke, visibly: `output_colors` (first-appearance order
is load-bearing — a fold can't keep it), `output_colors_by_size`,
`background_color` (needs frequency, not presence), `size_rule` (returns a
structured rule). The struct separates loop-filled cells from irregulars.
Old names survive as accessors: `allowed_colors()` = `color_any[all]`,
`colors_in_all_inputs()` = `color_every[inputs]`.

**Template for the next channel** (shape / count / position): write the
per-grid summary, reuse the same folds — all directions filled at once.

## The extractors (what's in the `inference` object)

Organized by CHANNEL, and within each channel by DIRECTION, so coverage is
intentional, not accidental. (The color-presence cells below are now produced
by the channel loop above; their semantics and safety notes are unchanged.)

### size · across_input_output — `detect_size_rule`
The first thing a human reads: how do output dims relate to input dims across
ALL pairs? That relationship eliminates most of the DSL before any
transformation is guessed (a reduction puzzle is never solved by tiling).

**Verified across the full corpus (1,147 puzzles: 633 train + 514 eval):**
- `Identity` / `Scale` / `Fixed` are ALWAYS trustworthy when returned.
- `BoundingBox` is the catch-all ("no clean formula — size is content-
  determined"). The only genuinely size-inconsistent puzzles all fall here:
  train 42f83767, 5587a8d0, 652646ff, a416fc5b, c803e39c. So:
  rule != BoundingBox => safe to gate on; == BoundingBox => trust content.
- Compares DIMENSIONS, not area (a transpose 12x16 -> 16x12 is correctly
  rejected from Identity).

*Inspired by 0520fde7* (7x3 -> 3x3 every pair: Scale 3/7, reduction-shaped).

### color · across_inputs — `detect_colors_in_all_inputs`
Colors present in EVERY train input — the invariant "ground" (background,
dividers, frames). Keeps the FULL set (`colors_in_all_inputs`) plus the
dominant one (`background_color`, most frequent). The engine's old version
kept only the background — lossy for inference (0520fde7 has {1,5} present in
every input, not just one).

### color · across_outputs — `detect_output_colors`
The union of colors across ALL train outputs, in first-appearance row-major /
train-pair ORDER. Order is load-bearing: the engine's OutputColors op consumes
this sequence. Includes 0.

### color · across_outputs by size — `detect_output_colors_by_size`
Train-output colors ranked by total pixel count (desc), background excluded.
Only produced when EVERY train output yields the SAME ranked sequence —
otherwise count=0. Depends on `background_color`, so it must run after
`detect_colors_in_all_inputs`. Feeds OutputColorsBySize.

### color · across_inputs_outputs — `detect_allowed_colors`
Every color appearing in ANY train grid. The only colors a color-LITERAL op
(AddColor/Recolor/RemoveColor) may emit — the search-pinning that collapses
the 10x literal fan-out. SAFE for literals (measured: every novel test-output
color is carried from the test input by a RELATION, never a literal — 0 lethal
cases on eval). Must NEVER be used to whitelist colors globally (that IS
lethal: ~-11% eval, measured).

## Pinning the literal fan-out: what helps, what doesn't (experiments removed, lessons kept, 2026-06-13)

Tried using color facts to pin/reorder the AddColor/Recolor literal fan-out and
speed convergence on the slow puzzle 0520fde7. BIAS and FORCE were both wired,
measured, and REVERTED (full revert — no code remains). The lessons are the
keepers:

**The real rule is below-cutoff vs in-beam, not union vs intersection.** The
beam is fixed-width (1024) and refills to full from the next-best candidate, so
pruning is never "remove a node" — it is "swap this node for whoever was rank
1025." Therefore:
- Pruning candidates that sit **below** the beam cutoff is a pure win: saves
  their generation+scoring cost, perturbs the trajectory not at all. This is
  why `allowed_color` (union of all train colors) gives the measured **3.4x** —
  it only ever drops colors NO example uses, which never held a beam seat.
- Pruning candidates that sit **in** the beam is a trajectory gamble: the seats
  refill with the next-best, and under a misranking scorer the replacements can
  steer worse. `guaranteed_introduced` = `in_every_output ∧ ¬in_any_input` is an
  intersection over REAL colors, so forcing it reached *into* the beam.

**Measured (0520fde7):**
- BIAS (try guaranteed colors first, set unchanged): did NOT converge.
  Reordering only breaks slot ties, and the blind scorer never gave the right
  partial program a seat to tie-break in the first place.
- FORCE (hard-skip non-guaranteed AddColor/Recolor): **WORSE** — ~41M attempts
  and never solved, vs baseline 21.7M and solved (root log
  `ARC/0520fde7.txt`). The winning program uses `AddColor[2]` and 2 IS the
  guaranteed color, so force never deleted the answer — it still broke the solve
  by displacing the productive 8-node lineage from the beam (search wandered to
  depth-13). Proof that fan-out width was never the bottleneck.

**Verdict:** inference is for CHEAP, SAFE, below-cutoff gating (allowed_color,
dimension rule) — keep doing that. Using inference to out-rank a blind scorer
mid-beam (bias / force / salience) keeps hitting the same wall: the scorer ranks
the productive prefix below junk. Fix RANKING, not the fan-out.

## Pre-fill the canvas with background (PARKED — TO BE CORPUS VERIFIED)

Idea: start the output canvas as the most-frequent color (`background_color`)
instead of empty/transparent. Then solutions stop having to lay their own ground
— the explicit ground op drops out: Manual loses `AddColor[0]` (10→8 with Origin
also gone), Before loses `Recolor[0]` (10→9), Auto already has none (8, no
change). Two real wins: (1) the concept-clean program (`Partition, IntersectL`)
ties the opaque exploit on LENGTH, removing the node-count tax that made search
prefer the `MoveColumn` trick; (2) deletes a 10x color-literal fan-out node, and
SAFELY (removes an op, not a beam candidate). Caveats: (a) does NOT sharpen
scorer discrimination — a pre-filled ground gives every candidate the same free
background-cell match, a ~uniform offset; (b) only valid where the puzzle's true
background IS the most-frequent color (a heuristic default like the dimension
rule — needs a corpus guard before trusting). VERIFY on the 1,147-puzzle corpus:
for how many is most-frequent == true background, and does pre-fill ever corrupt
a solve.

### color · across_input_output — `detect_colors_across_input_output`
Which colors does the transformation ALWAYS remove / ALWAYS add?
- `colors_consumed[c]`: in every pair, c is in the input and gone in the output
  (a divider stripped out, a marker consumed).
- `colors_introduced[c]`: in every pair, c is absent in the input and present
  in the output (the color the rule paints results in).
Computed by assuming true for all 10 colors and knocking a color out at the
first violating pair — one counterexample drops it.

*Inspired by 0520fde7* — color 5 (the divider) is in every input and never
survives into any output: a clean consumed color. Color 2 is introduced.

## find_in — the "first look" primitive

The first move a human makes on a pair: search each grid inside the other
(output-in-input AND input-in-output), regardless of which is bigger. A hit is
a placement of the NEEDLE inside the HAYSTACK where every non-empty needle
cell lands on a non-empty haystack cell under a CONSISTENT color mapping (so a
recolored copy still matches — 0520fde7's output 2s sitting on a panel's 1s).

The match deliberately reports its **RESIDUE** — non-empty haystack cells in
the window the needle did not cover. Residue is not noise; it is the next
question ("what killed those two cells?"). residue == 0 means a clean copy;
residue > 0 means the needle is a strict subset of that window — a part, or
(if later examples make the output sparser than every part) evidence it is a
PRODUCT instead.

**Empty semantics (the certainty boundary):** the only assumption-free value
for `empty` is TRANSPARENT (-1) — "no pixel here". **0 is a COLOR**: it often
plays background, but that is a per-puzzle ROLE. Passing a color as `empty` is
a HYPOTHESIS the caller is making ("this color is see-through here"), and any
fact derived from such a call must carry that hypothesis with it. No detector
currently calls find_in — the lift to a confirmed invariant (which story,
which transparency, checked across pairs) was removed as premature.

## Salience experiment (removed — record kept, 2026-06-11)

**Full rebuild spec with code: `3 - Search/Salience Anchoring.md`** — the
anchoring thesis, the figure+ground recognizer (the over-fire fix), the re-rank
hook, and the session evidence table (salience was the ONLY change that lowered
attempts). Read that to rebuild; the summary below is the original datum.

A beam re-rank using find_in as a candidate recognizer ("is the target visible
inside this candidate's output, modulo licensed recolor?") was built, measured
on 0520fde7, and removed as too complex/specific. The datum is real and kept:

- Re-ordering refinement slots by recognition cut attempts **21.7M -> 9.3M
  (2.3x)** with wall time unchanged.
- But the recognizer over-fired (~97% of beam entries got a bonus) because
  see-through matching skips ground cells — and ground carries half the
  evidence (De Morgan: "2s = intersection of 1s" == "0s = union of 0s", so a
  real match must explain figure AND ground).
- Any future recognizer should score both sets under one placement before it
  earns its way back in.

Background context from the same investigation: the literal scorer is
cell-anchored and color-literal, so `Input MoveColumn[3] Union` (the human
panel-overlay insight) was generated at attempt 67,872 (0.5s) but scored -12
and never earned a refinement slot; the solve then took 21.7M attempts to
rediscover it inside an 8-op program. **Recognition, not generation, is the
bottleneck.**

## propose() — deferred on purpose

Will turn the inference analysis into candidate DSL programs (vector<node*>).
node is an engine type, so proposal is engine-coupled — unlike analysis, which
is pure. Not needed until the first detector exists; when it returns, decide
whether it lives in this file or a separate propose.hpp.

## environment.hpp split (parked 2026-06-13)

environment.hpp is a god header. The plan that felt right: keep the NAME and
the world-types in place (arc_color, grid, dimension_rule, example — small and
beautiful) and PULL EVERYTHING ELSE OUT, so no includer breaks:
- DSL language definition (NODE_PRIMITIVES x-macro, node_type, node,
  flat_program, ~lines 81-310) -> its own dsl.hpp.
- Solver tuning constants (score weights, BEAM_WIDTH...) -> pipeline/solver.
- Algorithms (segment_objects_by_color_cpu, match_object_summaries,
  detect_target_objects, tree flatten/stringify/execute) -> pipeline/solver.
- Hardest, later: `puzzle` itself is half world half pipeline — train/test/rule
  is data, best_tree_slot/target_obj_slot/score state is solver scratch.
Each step independently buildable. Parked: a lot of surface, no urgency.

## Home-of-all-invariants (planned)

This file becomes the single home for all invariant reasoning. Some invariants
still live in environment.hpp because the ENGINE also consumes them. Migrating
them means rewiring those consumers — a deliberate, compile-and-smoke-tested
change, not an incidental move. (`detect_target_objects` deliberately STAYS in
the engine: it reads no direction — it's per-output decomposition for the
scorer, not an invariant.)

## Layout[k] silent clipping (failure record, 2026-07-10)

Found while solving 0a1d4ef5 ("read the arrangement of squares out of noise",
solved with `Input Foreground Foreground Foreground Split Layout[7]
ScaleDown[7] Union`).

Layout[k] MATERIALIZES the arrangement: it merges all objects into one painted
object sized (cols*k) x (rows*k). Grids cap at GPU_MAX_GRID (30), so the frame
is clamped — and any band whose slots start past the ceiling is amputated
without a trace. Concretely: the test pair had 4 rows on 10-paper = 40 tall;
rows 30-39 were scissored off and the output was a plausible-looking 3x3
missing its fourth row. Training pairs never exposed it (max 3 rows = exactly
30). Cost a full debugging round because the truncated result looks like a
banding bug, not a capacity bug.

The general shape of the failure: an op whose OUTPUT SIZE is data-dependent
(bands x k) can silently exceed the grid ceiling and degrade into a wrong but
plausible answer. Workaround that solved it: smaller k — cell size is
irrelevant to the read (clipped solid squares still vote their own color), so
any k with bands*k <= 30 is exact. Capacity at k=2 is 15 bands.

