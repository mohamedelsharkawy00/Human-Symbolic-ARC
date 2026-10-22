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

## P2. One word per act, every choice a name

Agreed 2026-10-21 (the user), before folding TileX, TileY and Tile[0] into one
word. Four known ideas, together:

- **Named arguments.** Each choice of a word is a name, and the name is its
  only form: `Tile[across,8]`, never one number standing for several choices.
  One list gives the code its names and the program its spelling, so the two
  cannot drift, and a reviewer reads one form.
- **Orthogonality.** A word's choices are independent of each other (which way,
  how far apart), so every mix of them is sayable without a new word. What it
  buys is **compositional generalization**: new mixes of known parts work
  without being taught (e619ca6e's copies run diagonally; no word says that
  today).
- **Equivariance.** A choice the picture shows is measured on each picture,
  never written as a number: "its own size" grows with the piece, "where it
  stands" moves with it. A fixed 3 does neither, and breaks on a test whose
  piece is bigger. This is P1, said for every choice.
- **Witness functions** (PROSE / FlashMeta, Polozov and Gulwani 2015). A choice
  only the answers show (which way the copies run, a step like 8) is read
  backward off the examples, from where the copies landed, not tried value by
  value. It is then written into the word, as a name or a count: the test input
  cannot show it, so the program has to carry it.

This replaces P1's consequence that output-determined choices stay separate
words: they become named choices of one word, read off the examples.

Words merge bottom up: twins first (one act split into names, like TileX and
TileY), then acts that differ only in step and range (the repeats, the moves),
then the walk (a repeat keeps every place, a move only the last), then the copy
(the walk, a pose, a color, a size).

### Applied: Tile (2026-10-21)

`Tile[steps: (a,b) (c,d), start: ..., end: ..., spread: lines|grid]` replaces
TileX, TileY, Tile[0-3] and DiagTile. The invariant: the same drawing laid
again, shifted, at every count of each step from `start` to `end`. A step is
always written as two pairs, rows down and columns right, `(0,0)` where there
is no second one. A step part is a count plus or minus up to two measures read
on each picture: `own width`, `own height`, `row gap` and `column gap` (the
fewest empty rows or columns between the piece's parts), and `color count`
(how many colors the picked pieces hold). The mark lattice (0a938d79) is its
lines made one drawing (UnionList), then tiled by `own height+row gap`: the
whole thing again, its gap after it. `lines` takes each step from the piece alone; `grid` has
every copy take every step again, which is a lattice (and, once `end` can stop
at the first thing in the way, a flood with step 1).
Copies are pieces; a copy is one step of the piece where a step is shorter.

The line's repeat asks `own width` (along rows) or `own height` (down columns)
first, but only on a cut into shapes: a whole color layer's own size is the
span of scattered cells, and repeated by it the layer
fit 62b74c02's pairs by coincidence and missed its test. Training: 267 right
(266), 40 wrong (39), gained 9b30e358, lost none.

Still to fold, each blocked by one missing thing: rays (a word upstream that
cuts each piece to its middle cell), moves (`start: last`, and whether each cell
stops on its own or the whole piece), trails and growth (`end` at the first
thing in the way, or meeting another walk), MoveToward, AlignTo, Origin and
AlignToCanvas (more relative numbers), ProjectOnSeeds (its seeds come from a
hidden list). Fractal stays its own word: its copies go only where its own
drawing is.
