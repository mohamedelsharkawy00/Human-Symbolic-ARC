> [2026-07-02] Entry point for all inference work is now `Inference Master.md`
> (same folder). This file remains the deep theory: the three fates, the
> invariance taxonomy, priors, and the ARGA/MADIL prior-art reading list.

# Constraining / Pruning Degrees of Freedom by Invariants

- **Date:** 2026-05-28
- **Status:** IDEA / SPEC — not implemented. Design discussion captured for later.
- **Related:** `SNAP_TO_ANCHORS_DESIGN.txt` (the all-annotated, zero-variant op is the
  clean instance of the "annotation path" described here).

## The core idea

A puzzle gives 2–4 training pairs. The SAME underlying rule applies to every
pair. Therefore anything that is a property of the RULE (not of a particular
input) is CONSTANT across all training examples. Every such constant that the
search currently treats as a free choice is a wasted degree of freedom.

Goal: infer the rule-constants ONCE — before, or early in, search — and freeze
them. Each frozen constant collapses a dimension of the search space. Less
freedom = smaller beam branching = the right program found sooner (or at all,
under a fixed timeout).

Restated: each training example is a CONSTRAINT. The set of programs consistent
with all examples is the INTERSECTION of the per-example constraint sets (a
"version space"). The more examples agree on a value, the tighter the
intersection. We want to compute that intersection up front and search only
inside it, instead of searching the full product space and rediscovering the
same constraints at every node.

## Why this matters here specifically

The beam's cost is roughly:

```
(#op choices)^depth   x   (#variant values)^(annotated slots)
```

The first factor is structural (which ops). The second is the one this idea
attacks: ops with a `value` field and a large VAR_COUNT multiply the space hard.
From `include/environment.hpp` `NODE_PRIMITIVES`, the high-VAR_COUNT offenders:

```
BidiColorMap  VAR_COUNT 45
TileX / TileY VAR_COUNT 30
Target        VAR_COUNT 10
Filter        VAR_COUNT 10
Recolor       VAR_COUNT 10
RemoveColor   VAR_COUNT 10
AddColor      VAR_COUNT 10
InteriorColor VAR_COUNT 11
ExtendOrthogonal / ExtendDiagonal VAR_COUNT 9
ScaleUp / ScaleDown               VAR_COUNT 9
MoveRow / MoveColumn / SlamRow / SlamColumn / Shift  VAR_COUNT 8
```

For most puzzles, the CORRECT variant of any such node is either (a) a single
constant shared by all examples, or (b) a value DERIVED from the input by a fixed
rule. It is almost never "freely re-chosen per example" — that would mean the
program is not a rule but a per-example fit, i.e. overfitting. So the live set of
variant values is usually tiny; pruning to it is a big win.

## The three fates of a `value` slot

Any node carrying a `value` parameter is, semantically, one of:

- **(a) CONSTANT OF THE RULE.** The same literal works for every example.
  e.g. `Recolor[ red->blue ]` in all pairs. The search picks it ONCE; freezing
  means: only consider the value(s) that scored on ALL examples.
- **(b) INPUT-DERIVED (ANNOTATION).** No single literal works, but a fixed FUNCTION
  of the input does. e.g. "Filter by the rarest color" → different concrete
  color per example, same rule. This is the existing `IS_ANNOTATED` column in
  the X-macro (`environment.hpp`); annotated ops compute their value at
  execution time from `ctx` instead of from search.
- **(c) FREE PER-EXAMPLE.** Different literal per example, no derivation. This is the
  BUG case — it is overfitting and should be rejected, not searched.

The inference layer's job: for each value slot in a candidate program, decide
which fate applies, and act on it.

- **(a):** intersect the per-example "values that work" sets. Search the
  intersection (often size 1).
- **(b):** if the intersection is empty but a known derivation rule reproduces the
  per-example values, convert the slot to an annotation.
- **(c):** if neither holds, treat as evidence the PROGRAM is wrong (penalize /
  prune the candidate), not as a slot to keep exploring.

Mechanically, (a) is "version-space intersection over literals"; (b) is "does any
registered annotation function explain the per-example values"; (c) is the
fallthrough.

## Categories of inferrable invariants (cheap → expensive)

### 1. Grid-level

- Output dimensions: fixed (always WxH)? same as input? input scaled by a
  constant ratio? cropped?
- Output palette: equals input palette? input palette minus background? input
  plus a fixed added color? a fixed constant palette?
- Background color: always 0? or the most-common input color?

**PRECEDENT ALREADY IN THE CODE: the SIZE RULE.** `gpu_rule_output_dims`
(`src/cuda_solver.cu:1793`) + `ctx.rule_type` / `ctx.rule_params` already infer the
output-dimension invariant once and let it size the canvas / drive
`RotateToTarget`. This is the prototype for everything in this note: infer an
invariant from train, store it on the context, let it constrain execution.
The work here is GENERALIZING that pattern to palette, counts, roles, etc.

### 2. Op-variant  (the big payoff — see *Why this matters*)

For each (op_type, slot) appearing in a candidate program, the set of variant
values that yield a nonzero / above-threshold intermediate on EVERY example.
Search only the intersection. Empty intersection → try annotation → else
penalize the program.

### 3. Object-set

- Object count preserved input→output? Output count a constant N? Output
  count = a function of the input (e.g. = number of distinct colors)?
- Pairing structure: N input objects map to N output objects? (SnapToAnchors:
  #bodies == #anchor-clusters == #stamps, constant across train.)
- "Every output contains exactly one object", "output is always a single
  framed region", etc.

### 4. Color-role

- "Color X always plays role Y." Weak as a literal (the literal color often
  varies), STRONG as a derivation: "the most frequent color is the body
  color", "the unique singleton color is the marker". The CONSTANT is the
  FUNCTION that picks the color, not the color itself.
- `0e206a2e` is the example: literal body color varies (3, 8, 8) but the rule
  "body color = most frequent non-bg" is invariant. The thing to freeze is the
  selector, not the value.

## How an inferred invariant is used (two modes)

**HARD CONSTRAINT (prune):** programs that violate the invariant are discarded before
scoring. Cheapest, most aggressive. e.g. wrong output dimensions → dead. Wrong
output palette → dead. Object count off → dead.

**ANNOTATION SOURCE (empower):** the invariant is fed into an op's value at execution
time, turning a passive constraint into an active capability. e.g. "the added
color is always C" → AddColor becomes annotated with C; "body color = most
frequent" → SnapToAnchors reads it from `ctx`. This is the `IS_ANNOTATED` path and
is strictly more expressive than pruning, because it lets ops DO the inferred
thing rather than merely be filtered by it.

Most invariants can serve either mode. Dimensions/palette/count are natural hard
constraints (they gate the whole program). Color-role selectors are natural
annotation sources (they parameterize specific ops).

## The failure mode (must be designed against)

With only 2–4 examples, MANY things line up by COINCIDENCE: both inputs happen to
contain a 5; both outputs happen to be 9x9; both have an even object count.
Freezing a coincidence as a HARD constraint KILLS the correct program for the
test input. This is the central risk and the reason this is not just "diff the
examples and lock everything that matches".

Mitigation: a PRIOR over which kinds of constants are plausibly RULES vs
accidents.

- **STRONG priors (safe to harden):** output dimensions, output palette
  relationship, object-count relationship, background. These are the kinds of
  things ARC rules are actually built from.
- **WEAK priors (use as soft score bonus, never a hard gate):** "both inputs
  contain color k", "both have <=3 objects", incidental coincidences.
- **Graded use:** hard-constrain on strong-prior invariants; only soft-prior the
  weak ones (a scoring nudge toward consistent programs, not a filter).

This is the inductive-bias question stated sharply: the DSL primitives themselves
(`OutputColors`, `InputPalette`, `Target`, `Seeds`) are ALREADY a curated prior —
they encode "these features matter". Extending the inference layer = deciding which
ADDITIONAL invariants are trustworthy enough to commit to, and at what strength
(hard gate / annotation / soft bonus).

A useful tie-breaker is description length / simplicity: prefer the explanation
where the SAME value or the SAME derivation accounts for all examples. A program
that needs a different literal per example pays for it; one constant or one
derivation is cheaper and should score better. (Connects to the MDL framing in
`1 - Overall Design/Priors And Robustness.md`.)

## Relationship to SnapToAnchors

SnapToAnchors is the degenerate-but-ideal case for this framework: it has ZERO
free variants. Every parameter it needs — body color, key-pixel set, anchor
positions, dihedral orientation — is DERIVED from the grid at execution time
(fate (b), the annotation path), not searched (fate (a)) and never free (fate
(c)). So there is nothing to prune at the op level; it is already maximally
constrained. That is exactly the shape this idea pushes the whole DSL toward:
ops whose parameters are inferred from the puzzle rather than explored by the
beam.

## Sketch of an implementation shape (for later — not today)

- A PRE-SEARCH PASS over the train pairs that emits a struct of inferred
  invariants (dims rule [exists], palette rule, bg, object-count rule, candidate
  color-role selectors). Store on the execution context next to `rule_type` /
  `rule_params` — reuse that machinery.
- HARD GATES wired into candidate admission: reject programs whose output
  dimensions / palette / object-count contradict strong-prior invariants before
  the expensive score.
- VARIANT INTERSECTION: per (op_type, slot), precompute the values surviving on
  all examples; restrict the beam's expansion of that slot to the survivors.
  Empty survivor set → attempt annotation → else down-weight the program.
- SOFT BONUS channel for weak-prior consistencies, folded into the existing
  scoring weights (`W_*` in `src/cuda_solver.cu`) rather than gating.
- Keep it CONSERVATIVE first: start with the strong-prior hard gates
  (dimensions, palette, count) where the win is largest and the coincidence risk
  lowest; add variant intersection next; leave color-role derivations and the
  soft channel for after the safe wins are banked.

## Open questions

- Threshold for "a value works on an example" (exact match? above some partial
  score?) — too strict empties intersections, too loose keeps junk.
- Where the variant-intersection runs: per candidate program (precise, costlier)
  vs once per op_type globally (cheap, coarse). Probably global negative pruning
  first ("this value never works on ANY example → drop everywhere"), per-program
  intersection later if it pays.
- How annotation conversion (fate (b)) is triggered: a registry of derivation
  functions tried when the literal intersection is empty.
- Interaction with observational equivalence / beam diversity (see
  `3 - Search/`) — pruning DOF should compound with those, but verify it does
  not collapse diversity prematurely.

---

# The cognitive framing: figure–ground / salience

*(merged 2026-07-01 from `3 - Search/Prior Art Invariants Salience.md`, noted 2026-06-07)*

**Figure–ground / salience.** Invariants = the GROUND (background, divider,
always-present palette) — factor it OUT. Variants = the SALIENT FIGURE (what
changes in→out) — that's what search should focus ON. The engine has no
figure-ground separation: it searches `AddColor[0..9]` giving the invariant
background color and the salient varying color EQUAL attention, and drowns.
Invariant-pinning = giving the engine bottom-up salience (subtract the ground,
search only the figure). Known lens: Gestalt psychology, visual attention,
saliency maps in vision ML. Use it as a DESIGN COMPASS, not a "secret nobody
found" (saliency is a whole CV subfield — stay calibrated).

## Prior art (this idea is published, but execution is open)

The shower-epiphany ("invariants crowd the search space; pinning them collapses
it") felt undiscussed in ARC. Checked: the *popular* discourse (Chollet's
framing, guides, blog posts) IS abstract and ignores this — but the *academic*
ARC literature has it under concrete names. The idea is published but NOT
dominant on the leaderboard, so the EXECUTION is still open. The edge isn't
owning the idea — it's combining it with a fast CUDA engine + corpus
verification, done better than the people who published it.

Papers / approaches to read:

1. **ARGA** (Abstract Reasoning with Graph Abstractions) — THE closest prior art.
   Derives constraints from the training examples to prune the search space;
   uses a Tabu list to avoid primitives with poor recent performance. This is
   literally the invariant-pinning idea, published. READ FIRST — stand on it
   instead of re-deriving it; see exactly how they prune and where this engine
   (much faster, GPU beam) can do it harder.
2. **MADIL** — MDL-based (Minimum Description Length) program synthesis for ARC.
   MDL is the formalization of "regularities/invariants should compress the
   search." arxiv.org/pdf/2505.01081
3. **Icecuber** (Kaggle'20 winner, 20.6%) — brute-force DAG over unary piece
   functions + greedy stacker. Beat everyone scoring <10% mostly via tighter
   representation/pruning (undocumented craft). Worth studying for the pruning.
4. **Generalized Planning for the ARC** — arxiv.org/pdf/2401.07426
   (constraint-driven pruning in a small program space with abstraction.)
5. **ARC-AGI 2025 research review** — lewish.io/posts/arc-agi-2025-research-review
   (landscape overview; find where constraint/invariant approaches sit.)

Calibrated takeaway: invariant-based search pruning in ARC IS a published
direction (ARGA, MADIL) the popular discourse ignores. Re-derived independently
from our own logs — which proves the instincts are research-grade — and the
published versions don't dominate, so execution is open. Cross-domain: this is
bread-and-butter in constraint satisfaction / SAT / superoptimization (reduce
the variable domain using what's provably fixed). Not a unicorn; a real,
earned, *grounded* edge.

Next concrete move (turn conviction into a number): on 0520fde7 (73M attempts,
depth 10): separate ground (invariant: the `5` divider, `0` background) from
figure (salient: color `2` introduced, the AND overlap). Count how much of the
73M-attempt search was wasted attending to the ground (searching AddColor/
RemoveColor/Recolor over invariant color indices). That number says whether
invariant-pinning is a 2x or 100x lever before building.

---

# The detection checklist: every level invariance can live at

*(merged 2026-07-01 from `3 - Search/Invariance Taxonomy.md`)*

Each puzzle is an equation: input → output. Detect what's invariant (unchanged
across all examples). Factor it out. Search only over what varies. Variants can
be at any level: color, shape, placement, pixels, objects. No color is
inherently "background" — only invariance matters.

Invariance detection methods:

- Pixel-level: cells that keep the same color input→output
- Color-level: colors whose pixel count / positions don't change
- Object-level: objects that appear identically in input and output
- Shape-level: shapes that persist (even if recolored or moved)
- Symmetry-level: symmetry type preserved across examples
- Grid-level: dimensions, aspect ratio unchanged
- Topology-level: connectivity, adjacency, containment preserved
- Count-level: number of objects / regions stays constant
- Position-level: relative placement / alignment unchanged
- Pattern-level: repeating tiling or periodicity preserved
- Color mapping: the color→color mapping is consistent across examples
- Border/frame: edge pattern stays the same
- Size-level: object dimensions stay the same
- Spacing: gaps / distances between objects preserved
- Layering: z-order / overlap relationships preserved
- Color palette: set of distinct colors used stays the same
- Orientation: rotation / reflection type consistent across examples
- Logic/rule: the transformation rule itself is invariant across examples
- Latent: hidden structure or encoding scheme consistent across examples
- Intent: the "goal" or purpose of the transform stays the same
- Conditional: the if/then branching logic is reused across examples
- Relational: relationships between objects (bigger-than, inside, adjacent) preserved
- Arithmetic: numeric relationships between properties preserved (e.g., count, ratio)
- Parity: even/odd properties unchanged
- Anchor/alignment: reference point or axis unchanged
- Compositionality: how sub-parts combine stays the same
- Traversal: direction or order of processing consistent
- Boundary: how edges/corners are treated stays the same
- Scale/granularity: resolution or level of detail preserved

---

# First implementation: color-level invariance

*(merged 2026-07-01 from `4 - To Do/Implementation Notes Todos.md`)*

A color that appears in every training input is invariant (the canvas, not the
paint). Remove it from both seed objects and palette before searching. This
cuts the object count roughly in half, halving beam search combinatorics.

- Detection: count presence across inputs. Present in all = invariant.
- Tiebreaker: when multiple colors are all-present, skip only the most frequent.
- Future options to explore:
  - Keep the less frequent (smaller pixel count = more likely object)
  - Keep all tied except the most frequent
  - Only skip the single most frequent among tied

(The mature version of this detection now lives in include/inference.hpp as
`detect_colors_in_all_inputs` — full always-present set + dominant
background_color; see `6 - Handoffs/Inference Notes.md` for what got built and
the measured pinning lessons.)
