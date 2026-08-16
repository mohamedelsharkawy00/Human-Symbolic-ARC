> [2026-07-02] Entry point for all inference work is now
> `1 - Overall Design/Inference Master.md`. P1 below remains the canonical
> statement of rules-vs-constants and the inference/search seam.

# DSL Design Principles

## P1. A rule is an op whose parameters come from the data, not a constant

*(the inference / annotation seam)*

Established 2026-05-24 while generalizing Tile (the `0a938d79` / "two pixels →
alternating tiled lines" puzzle). This is the lever on the eval-0% overfit
problem (see KAGGLE_PORT handoff): discovered programs score ~10% on training
they were tuned on and 0% on unseen eval/leaderboard. The reason is that most
programs encode CONSTANTS, not RULES.

- `TileY[4]` → a **CONSTANT.** Only correct when the answer is 4.
  Cannot survive a new input. Pure overfit.
- `Tile[3]` (infers the gap) → a **RULE.** "tile by whatever the marks say." Fires
  correctly on an input it has never seen, incl. the test input AND unseen eval
  puzzles of the same structure.

So the generalization axis is NOT "more modes / more annotations." It is:
**MOVE EACH PARAMETER FROM ANNOTATION (search picks a number) TO INFERENCE (the op
reads it from its own input at exec time).** The more parameters an op infers from
its own input, the more the discovered PROGRAM is a genuine rule rather than a
memorized constant — and a rule is what transfers to the test input and to eval.

Framing that makes it click: the op SEQUENCE is the induced rule; the per-example
data-inference inside each op is "applying that rule to this specific input." A
human induces the rule from the few train pairs, then applies it. The solver's
analog: one program that works on all train examples, whose ops re-derive their
parameters per example.

## The hard limit (where inference must STOP and search takes over)

You can only infer from a single input what that input alone determines.
Everything else must stay a separate, searchable op so REWARD (matching the
train OUTPUTS) disambiguates it — which is exactly how a human uses the train
pairs, not just the train inputs.

Worked example — the `0a938d79` periodic-lines class, parameter by parameter:

| parameter | determined by | belongs to |
| --- | --- | --- |
| seeds / marks | the input alone | INFERENCE (perceive) |
| axis of variation | the input alone | INFERENCE |
| lattice spacing (gap) | the input alone | INFERENCE |
| color cycle | the input alone | INFERENCE |
| "expand each mark to a full line" | only the OUTPUT shows | SEARCH + reward |
| forward vs bidirectional | only the OUTPUT shows | SEARCH + reward |
| period multiple / phase choice | only the OUTPUT shows | SEARCH + reward |

A bare two-pixel input does NOT tell you to make full rows — you only learn that
from the train OUTPUTS. So the line-fill stays a separate op (`TileX[1]`) chosen by
search; it is NOT a wart that it's separate, it's the correct seam.

**CONSEQUENCE:** do NOT fold everything into one mega-op. An over-general op that
hard-infers output-determined choices commits to one interpretation with no way
for reward to correct it. Infer the input-determined structure; leave the
output-determined choices compositional.

## Free bonus: inference-rich ops also fix depth-survivability

The search problem (handoff): correct hand-solutions are 10–13 ops deep, the
beam only finds ~4-op programs and prunes the deep correct ones before their last
op "clicks." An op that infers its parameters COLLAPSES depth:

```
RotateToPortrait -> TileX -> Tile -> RotateToTarget   (4 ops)
```

shrinks toward

```
TileX -> CompleteLattice                              (2 ops)
```

Shorter true programs survive the beam. "More human" and "findable by search"
pull the same direction.

## The heaviest / most-principled version (noted, not built)

Cleanest architecture separates PERCEPTION-that-emits-structure from
TRANSFORMS-that-consume-it: a "detect lattice/symmetry" op that outputs a
structured descriptor, and a "complete" op that fills it. The same descriptor
could then feed repair / recolor / crop-to-cell. The DSL's only currency today
is object lists, so this needs a new data type (a lattice/symmetry descriptor)
threaded through `gpu_exec_context` — a real architectural change, not an op add.

## Applied so far

- `Tile[3]` (`src/cuda_solver.cu`): reads spacing + color cycle from the marks
  (count * smallest-gap), instead of the constant 2*gap. Handles N marks /
  same-color / repeated colors. Line-fill (`TileX[1]`) and orientation
  (`RotateToPortrait`/`RotateToTarget`) deliberately kept as separate searchable ops.
- Open next steps (per the ladder): infer the AXIS (drop the rotate dance), then
  2D lattice. Both are "push more input-determined params into inference."
