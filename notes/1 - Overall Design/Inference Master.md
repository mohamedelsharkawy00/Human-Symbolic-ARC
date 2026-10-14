# INFERENCE — MASTER DOC (canonical, consolidated 2026-07-02)

> This is the single entry point for everything inference. It consolidates:
> `6 - Handoffs/Inference Notes.md` (code walkthrough), `1 - Overall Design/
> Constraining Degrees Of Freedom By Invariants.md` (the theory + taxonomy),
> `2 - DSL/Design Principles.md` (P1: rules vs constants), `3 - Search/Salience
> Anchoring.md` (the rebuild spec), `6 - Handoffs/Project Status 09-06-2026.md`
> (what shipped + measurements), `4 - To Do/Map Op Consolidation.md`, and the
> strategy framing in `5 - Competition/Research Synthesis.md` §5-6.
> Also (added on the 2026-07-02 second pass): `1 - Overall Design/Backward
> Deduction Design.md` (output-as-specification, witnesses) and `3 - Search/
> Op Masking.md` (soft per-puzzle op filtering — the cheap deployment of
> inference facts). Those files stay for depth; THIS file is the map and the
> current truth.
>
> **PRIORITY OVERRIDE (Home.md "NOW" block, standing directive):**
> (1) hand-write DSL solutions for eval tasks ONE AT A TIME, marking each
> expressible/inexpressible — the audit IS the verification, not
> procrastination; (2) the one deadline-feasible lever is OUTPUT-FIRST
> PRUNING (= Backward Deduction §5 option 1 = op masking seeded from the
> output type-signature). The audit feeds the lever. Roadmap §4 below is
> ordered accordingly.

## 0. Why inference is THE bet

The solver is stable and fast (~10% on the 633-task v1-train benchmark, ~20%
on v2 training per Home.md) but ~0 on unseen v2 eval — confirmed
capability-limited, not a pipeline bug, and expected: forward DSL beam search
is the Icecuber family, which ARC-AGI-2 was designed to defeat. The realistic
target is off-zero, not winning. Both measured diagnoses point the same way:
- The blind cell-anchored scorer ranks productive prefixes BELOW junk (the
  "−12 wall"); search is an unanchored random walk. Fixing RANKING, not the
  candidate set, is the only lever that ever reduced attempts (salience: 2.3x).
- Programs found by search encode CONSTANTS, not RULES (`TileY[4]` vs
  `Tile[3]`). Constants can't transfer to test/eval. Ops that infer their
  parameters from the input are what generalizes — and they're also shorter,
  which fixes depth-survivability for free.

Both are inference problems: **read the invariants first, make the engine use
them.** Recognition, not generation, is the bottleneck.

Prior art exists (ARGA = constraint-derived pruning, MADIL = MDL synthesis,
Icecuber's pruning craft) but doesn't dominate the leaderboard — the execution
is open. Our edge = this idea × a fast CUDA engine × corpus verification.
(Details + reading list: `Constraining Degrees Of Freedom By Invariants.md`.)

## 1. What exists in code TODAY (include/inference.hpp, verified 2026-08-07)

`infer()` is PURE: puzzle in → `inference` struct out. Runs once at puzzle
load (from puzzle_loader), bridged onto puzzle fields (engine doesn't read the
inference object directly yet — see §5.1). Note the 2026-06 rename: the old
`size_rule` struct is now `dimension_rule` (see DIFF_vs_baseline_2026-06-10);
older notes still say size_rule.

**File shape (reordered 2026-08-07).** Everything is one of four things:
*reading* (how to read a grid at all), *rule* (what a channel claims), *detector*
(the one pass that reads it), and `infer()` (the reading order). The file runs
vocabulary → eight channels, each contiguous as question / struct / private
machinery / detector → `struct inference` → `infer()` → `log_inference()`.

Shared readings, used by more than one channel: `color_tally`, `color_boxes`,
`walk_out` (the one flood fill), plus `NEIGHBOUR_ROW_8/COL_8/ROW_4/COL_4` and
`shape_occupancy_hash` in 0_core_types.hpp.

| channel | function | notes |
|---|---|---|
| size | `output_dimension_rule::infer_rule` | Identity/Scale/Fixed always trustworthy; BoundingBox = catch-all. Verified on all 1,147 puzzles. |
| color presence | `detect_color_presence` | the direction×fold channel loop: every (scope × every/any) cell + consumed/introduced/preserved relations |
| output colors | `detect_output_colors` | first-appearance ORDER is load-bearing (feeds OutputColors op) |
| output colors by size | `detect_output_colors_by_size` | must run after presence (needs background_color); feeds OutputColorsBySize |
| shape→color map | `detect_shape_color_map` (writes `shape_color_map_rule`) | glyph shape → target recolor (009d5c81); §3c |
| shape→interior map | `detect_shape_to_interior_color_map` | fallback when the first rejects (00dbd492); shares occupancy hash + table via `learn_shape_color`; §3c |
| unchanged cells | `detect_unchanged_cells` | Identity pairs only; how much of the input survives untouched |
| color diff | `detect_color_diff` | per pair per color: which cells each color gained and lost |
| color swaps | `detect_color_swaps` | first consumer of color diff; pins BidiColorMap's 45-way fan-out |
| object pairing | `detect_object_pairing`, `pick_split` | the edge level of correspondence; §3d, and the cuts in §3e |
| background | `elect_background` | ranked hypotheses per direction, elected by role; §3g |
| first-look primitive | `find_in` | needle-in-haystack under consistent color map; called by both edge collectors; what counts as "there" is §3f |

Inside the pairing channel the order is: `pick_split` tries every cut →
`detect_object_pairing` per cut → `segment_for_pairing` cuts both sides →
`collect_edges` (or `collect_motif_edges` for the three pattern cuts) →
`sit_on_top` for pieces still unpaired → `credit_landings` folds the far side.

Directions backbone (the taxonomy every detector fits): `across_input_output`
(one pair, read both ways) / `across_inputs` / `across_outputs` /
`across_inputs_outputs`. Plural S = whole-train scope. A fact is an invariant
ONLY if it survives every pair. Full prose: `6 - Handoffs/Inference Notes.md`.

Consumers wired into the engine:
- **Color-literal pinning** (`beam_solver::color_literal_allowed`, toggle
  `pin_color_literals`): AddColor/Recolor/RemoveColor literals restricted to
  allowed_colors. **Measured: 73M → 21.7M attempts (3.4x), 2m17s → 47s on
  0520fde7.** Corpus-safe: 0 cases where a test output needs an unseen literal.
- **Size rule** (`gpu_rule_output_dims` + ctx.rule_type/rule_params): sizes the
  canvas, drives RotateToTarget. This is the PROTOTYPE pattern: infer once,
  store on ctx, constrain execution.
- **Shape-map tables** (`shape_color_keys/vals/count` on the puzzle): feed the
  ShapeColorMap / ShapeToInteriorColorMap ops.

## 2. The rules that are PROVEN by measurement (do not relearn these)

1. **Below-cutoff pruning is safe and wins; in-beam interference loses.**
   Pruning candidates that never held a beam seat (allowed_colors) = pure 3.4x
   win. BIAS (reorder literals) did nothing. FORCE (hard-skip non-guaranteed
   literals) was ~2x WORSE (41M attempts, never solved) even though it never
   deleted the answer — it displaced the productive lineage. Fan-out width was
   never the bottleneck; RANKING is.
2. **Salience (re-rank refinement slots by recognition) is the only lever that
   cut attempts** (21.7M → 9.3M on 0520fde7). Every candidate-set change made
   things worse. It was removed for over-firing (~97% of beam) and host cost;
   the fixed rebuild spec (figure AND ground under one placement, gate + score
   only the ~30 refinement slots) is in `3 - Search/Salience Anchoring.md`.
3. **Strong vs weak priors.** Hard-gate only on strong-prior invariants
   (dimensions, palette relationship, object count, background). Weak
   coincidences ("both inputs contain a 5") get at most a soft score bonus.
   Never use allowed_colors as a global output whitelist (−11% eval, measured).
4. **The three fates of a value slot** (from Constraining DOF): (a) constant of
   the rule → intersect per-example working values; (b) input-derived → convert
   to annotation (IS_ANNOTATED path); (c) free per-example → the program is
   wrong, penalize it. This is the framework for op-variant pruning.
5. **Where inference must STOP** (Design Principles P1): infer only what the
   input alone determines (seeds, axis, spacing, color cycle). What only the
   OUTPUT shows (expand-to-line, direction, phase) stays a separate searchable
   op so reward disambiguates. Don't build mega-ops that hard-commit
   output-determined choices.

## 3. The invariance taxonomy (checklist when adding detectors)

Pixel / color / object / shape / symmetry / grid-dims / topology / count /
position / pattern-periodicity / color-mapping / border / size / spacing /
layering / palette / orientation / conditional / relational / arithmetic /
parity / anchor / compositionality / traversal / boundary / scale.
(Full list with one-liners: `Constraining Degrees Of Freedom By Invariants.md`.)

Template for a new channel: write the per-grid SUMMARY function, reuse the
generic folds (`every`/`any` per scope + consumed/introduced/preserved for the
pair relation) — all directions fill at once. Each extractor carries an
INSPIRED-BY block naming its motivating puzzle.

The human protocol these detectors mimic (from `3 - Search/Dorr solving
05a7bcf2.txt`): search the output's objects inside the input, least-redundant
key objects first; form the simplest hypothesis on example 1, abandon on
first counterexample and escalate complexity; order everything by
energy/compute, smallest first; on the test input, distrust any rule that
suddenly produces N/A or no change. "Find an invariant rule to explain the
variant, and make it invariant."

## 3a. Why nothing forces a channel to cover all four directions (decided 2026-07-26)

Every channel struct carries one member per direction, NA included. Nothing in
the code checks that. Adding a channel with three directions compiles fine —
catching it is the author's job.

The worked example this section used to point at, `dimension_facts`, was deleted
on 2026-07-26: every number in it was recomputable from the train pairs and only
two readers existed. `output_dimension_rule` replaced it — a chosen case plus the
numbers that case learned, with its own `infer_rule` reading them off the pairs.
Coverage is a property of the question, not a quota: that rule uses two of the
four directions on purpose, because output size is a question about outputs and
about pairs.

Four ways to force it were weighed and dropped:

- **Templates** (`facts_per_direction` + a declared-but-undefined `facts_for<D>`
  per channel). The only real guarantee in C++20: a missing direction is an
  incomplete type, so the channel cannot be built. Built, worked, reverted —
  the syntax is a permanent tax on every reader of the file, paid to catch a
  mistake made once per new channel.
- **`static_assert` on a `count` sentinel.** Only freezes the enum's length; it
  never compares the enum to a struct's members, so the two lists still drift.
  Also needs a uniform payload per direction, which the channels don't have.
- **Abstract base with pure virtuals.** Forces accessor *functions*, not stored
  data, and adds a vtable pointer — the facts get copied into plain `task`
  structs for GPU upload and must stay flat and trivially copyable.
- **X-macro.** Forces real named members with no templates, but must paste a
  type name per channel per direction, and the diagnostics are preprocessor
  soup.

What survives without any of them: each direction's facts sit in their own
member, so reading a pair-only fact off the inputs member is a compile error,
and `not_applicable` is empty so an NA cell cannot be read as data. Only
cross-channel completeness is unguarded.

The set of four directions is closed by construction (one pair, all inputs, all
outputs, all pooled), so the growth case this machinery guards against is not
expected to happen.

## 3b. Empty cells of the fact table (audit 2026-07-02)

The table has three axes: where you look (inputs / outputs / both / a pair),
what you notice (color, shape, count, position, ...), and how many examples are
squashed into one answer (true in every / true in at least one / the pair
verdict consumed-introduced-preserved). Each combination is one cell. Below are
the combinations nobody fills yet. Moved here from the header comment in
`include/inference.hpp`; tick off as they land.

- FOLD SYMMETRY
  - [ ] any-pair variants of consumed/introduced/preserved (relations are AND-folded only; no "consumed in at least one pair")
  - [ ] the 4th relation cell: absent-in-both (!in && !out)
- INPUT/OUTPUT MIRRORS (facts come in pairs — Learned Orderings rule)
  - [ ] input_colors (first-appearance) + input_colors_by_size
  - [ ] output-side background (bg is inputs-only today)
  - [ ] ordering axes: ByPosition / ByFrequency / ByBoundingBox (2 of 5 exist)
- PAIR-RELATION DIRECTION (must read BOTH ways)
  - [ ] output-in-input AND input-in-output containment (crop / embedding) — two `find_in` calls per pair; `find_in` has ZERO callers today
- PER-EXAMPLE VALUES
  - [ ] per-grid background (global bg = -1 when backgrounds differ per example -> shape-map detectors silently degrade)
  - [ ] keep per-example value vectors alongside folds (blocks fate (b): "which derivation explains the varying values")
- EMPTY CHANNELS
  - [ ] object (count rule / pairing / persistence)
  - [ ] position
  - [ ] count
  - [ ] topology
  - [ ] shape beyond the two ad-hoc maps
- APPLICATION COLUMNS (fact x {gate, annotate, reorder})
  - [ ] reorder/mask column is entirely empty (= the output-first pruning lever); consumed/introduced/preserved have NO consumer at all

## 3c. Shape->color map: what it learns, and the two upload paths (moved out of code 2026-07-26)

### What the channel answers

Shape+color, across_input_output: does a small glyph's SHAPE pick the color the
main object is repainted to?

It learns a shape->color lookup from the train pairs so the ShapeColorMap op can
apply it. The colors live only in the train OUTPUTS, so no device-side pixel read
can recover them — they have to be learned here.

Key = smallest-bbox color class, target = largest. The value is the color the
target wears in that pair's output.

Self-validating: if one shape ever maps to two colors, the whole hypothesis is
rejected (`valid = false`), and a rejected table makes the op a no-op.

Two detectors write the same storage — the body-recolor one, and
`detect_shape_to_interior_color_map` as a FALLBACK that infer() runs only after
the first rejects. So a puzzle's single learned table is whichever validated.

### The gotcha, which cost a debugging session — FIXED 2026-07-26

Any puzzle-level data an op reads used to be uploaded to the GPU by TWO
independent paths, and both had to be fed or the op silently no-opped on
whichever one was forgotten:

1. Beam search -> `build_example_upload()` (solver_executor.hpp)
2. Scratch path -> `execute_flat_program()` (dsl_executor.hpp) — the GUI
   hover/apply-solution and submission path.

The cause was structural, not carelessness. `execute_flat_program` took thirteen
parameters, nine of them puzzle fields unpacked by hand at all seventeen call
sites, so each new learned fact meant another parameter and another chance to
miss one.

The fix: all seventeen callers already had the `puzzle` in scope. The three entry
points (`execute_flat_program`, `execute_tree_via_flat`, `score_tree_output`) now
take `const puzzle&`, and both upload paths call one
`environment::fill_puzzle_level(ex_data, p)`.

So a new learned fact is now added in exactly one place, and the second path
cannot be forgotten because there is no second copy of the filling code. The two
device buffer sets (`d_cached_*` and `d_scratch_*`) are untouched — beam search
still keeps its upload cache across candidates, and single-shot GUI evaluation
still lands in the scratch slot.

## 3d. Object pairing: correspondence at the edge level (moved out of code 2026-08-07)

`object_pairing · across_input_output`. One edge = "this object appears there on
the other side". Read BOTH ways, because the two directions answer different
questions:

- forward (input → output) — where did this object GO?
- backward (output → input) — where did this object COME FROM?

The per-object edge count gives the relation class with no extra machinery:
forward 0 = deleted, 1 = persisted, many = copied; backward 0 = invented
(nothing in the input to build it from), else sourced.

And read at TWO LEVELS, in the order a human looks:

- **whole grid** — background removed, nothing cut up yet: "does the whole
  picture turn up over there?" This is the unit that TILES (007bbfb7 stamps the
  3x3 input wherever a cell is lit; 00576224 the same), and it exists whether or
  not the picture happens to be connected.
- **objects** — the same question asked of each piece, after segmentation.

Four passes in all: two levels x two directions. Two edge sources either way,
deduplicated:

1. **positional** — the glyph found verbatim on the other side (`find_in`):
   translation only, exact or uniformly recolored, covers embeddings inside
   bigger things.
2. **shape** — an OBJECT there with the same canonical shape hash:
   orientationless and positionless by construction, so turned/flipped/moved
   objects pair without any pose enumeration.

Facts only, evidence not verdicts: consumers fold and gate downstream.

### The set test — why matches are not judged one at a time

One match on its own says almost nothing. Plenty of shapes fit somewhere, and a
7-cell orange shape dropped inside a big orange blob was always going to fit
(007bbfb7 pair 4 — the TRUE stamps expect 1.45 accidents each, so judging them
one at a time throws the answer away).

A SET is different: if the exact matches together paint every cell that needed
painting, no gaps and nothing left over, that set is the reading whatever any
single match cost. Only when no set paints the picture do we fall back to
dropping weak matches one by one.

The matches are also not taken greediest-first — both one-at-a-time and greedy
pick locally and get poisoned by ties (00576224's middle band: a straddling
match and a mirrored tile look identical at the moment of choosing, and taking
the straddle first drags in two more pieces that overlap it). What tells a real
reading from a pile of coincidences is that the pieces FIT TOGETHER: every cell
that needs painting is painted exactly once, no overlaps, nothing outside. That
is a search, so it searches — every cell, fewest options first, backing out of
choices that lead nowhere.

### Anything less than the whole piece as itself must earn its place alone

Only the full identity claim gets to lean on the set test. 007bbfb7's stamps
each lose to chance and are right anyway, which is the whole reason pieces are
judged as a set. A recolor or a partly hidden find is weaker — it says the shape
is there wearing something else, or mostly not showing — and a weak claim that
was going to happen anyway says nothing.

Without this, a picture with no canvas cuts into 1x1 pieces that match every
cell under a free color map, and "every cell is a copy of a cell" tiles
perfectly (00576224 read that way twice over).

### Sitting on top: correspondence without identity

A piece nothing found still corresponds to something. Asking "does this piece
turn up on the other side, unchanged?" is a question about IDENTITY, and a piece
that CHANGED answers it nowhere — 025d127b slides the top of each outline one
cell right, so the whole shape is found nowhere and only its shattered fragments
are found anywhere. Since finer cutting can only ever explain more, the reading
that shattered the shape then wins every time.

So a piece left with no edge gets one more question, the one a person asks:
WHICH PIECE DO I SIT ON TOP OF THE MOST. That is correspondence without
identity, and it cannot flood — one edge per piece, only where the pieces
overlap at all, and only for pieces nothing else accounted for.

### Edge order

Edges come out in the order the pieces read, once the reading is settled. The
pass asks where each PIECE went, so the piece leads: down the input, and for one
piece that went several places, down the output after it. The search's own order
is by pose — every upright placement, then every mirrored one — which put
00576224's middle band last for no reason anyone reading it cares about.

The pieces are already in reading order — biggest first, then down the picture —
so following their index IS that order, with no second copy of the rule to keep
in step.

### Piece order: biggest first, then down the picture

The cut itself meets pieces row by row, so a small piece that starts high comes
out before a big one that starts lower (00576224's second pair reads small, big,
small). Size first is also the order the cover tries its options in, so the
readout and the search agree on what matters most. The glyphs ride along — they
are the pieces.

The same order governs the cover's own options per cell. The search takes the
first cover it reaches, so this order IS the preference — and a 1x1 piece can
tile anything, so trying small first hands every puzzle to one pixel repeated
(00d62c1b pair 3: a single green cell claimed 98 placements while a 43-pixel
shape was written off as gone).

### Some cells are simply new

A cell no EXACT piece can reach was not carried over from the other side — it
was made (00d62c1b fills enclosed interiors with yellow, and no green piece is
ever yellow). Demanding a cover of those cells forces the search to invent a
source, and a 1x1 piece will always volunteer, so the fabrications land on
whatever pixel comes first.

Only cells an exact piece could account for are REQUIRED; the rest are left
over, which is what `created_cells` has been counting correctly all along.

### Every piece accounts for itself before any piece is copied

The search offers pieces in two rounds: round one only pieces not yet credited
with a landing, round two the rest. Copying is a real relation but a stronger
claim, so it is what's left when going round once cannot explain the picture —
not the first thing reached because one piece happened to be tried early. The
greedy fallback uses the same two rounds, for the same reason.

### One of them is in front

Whoever paints the cell a piece is hidden at is in front of it, and nothing can
be both in front of and behind the same piece. Without this, 03560426's thin red
bar sits one row too high: it hides one cell either way, so the two placements
look identical until you ask which piece is on top and get both answers.

Checked as the piece goes down, so the search backs out of the contradiction
instead of returning it. Catches a stand-off between two; a ring of three or
more does not arise from ink drawn one piece at a time.

### Both ways round, and one list (changed 2026-08-11)

Where each input piece went, and where each output piece came from, are asked of
the same pieces and merged into one list. Every edge says which grid it starts
on. Where the two ends agree — an input piece found at an output piece, and that
output piece found back at the input one — only the stronger claim survives, so
nothing is counted twice.

One direction only was the earlier design, on the grounds that the other end is
the same fact read backwards. It is not. A picture can cut cleanly one way and
into rubble the other: 0607ce86's input is noise and its output is clean
repeated tiles, and no cut of the input ever finds them. Which way to read it
was then tried as a choice, and every score got it wrong — a total taken on one
picture against a total taken on the other is not a fair comparison. So the
choice was removed rather than improved: both are read.

Matches that were found and then thrown away are kept with their reason. Without
that a missing line is unexplainable: the log only ever showed what was kept, so
"it never matched" and "it matched and lost" look the same.

### Where the cover search stops, and what it cost to find out (2026-08-11)

`search_cover` is an exact cover, so it is exponential in the worst case and has
to be stopped. The stop was 20,000,000 nodes, a number chosen by taste, and on
1be83260 the app never finished loading the puzzle at all.

Instrumented, probe added and removed the same day:

- 378 cover searches on that puzzle. 377 finish in under 4,096 nodes.
- The 378th: 253 cells, 195 of them required, 570 candidate pieces. Eight
  million nodes in 80 seconds and still going, at 102,491 nodes a second.
- `deepest` pinned at 95 and never moved again. 76% of nodes were dead ends.

So it was not searching by then. It was re-permuting a prefix it had already
disproved. And the budget bounded a number nobody can feel: the cost lives
INSIDE a node — every node rescans all cells and for each of them all its
options — so 20M nodes is 195 seconds, not an instant.

Two things fixed it, and neither was the looking:

- The stop now comes from the instance itself, `pieces x cells`. For that search
  that is 144k nodes; for the other 377 it never binds. Never finishing became
  about 34 seconds.
- Every reading was being computed twice, once by the election and once by the
  pane rebuilding it for the table. `ranked_pairs` keeps what the election
  already worked out. 34 seconds became 8.6.

Worth knowing before touching it again: `gave_up` was never evidence. Against a
tree around 10^137 wide, 20M nodes is 10^-130 of it, so the verdict never meant
"I looked and found nothing" — which is why it correctly falls through to the
greedy reading exactly as `no_fit` does. Raising the budget was never going to
be the fix.

## 3e. The ways a picture can be cut, and how one is chosen (moved out of code 2026-08-07)

Which cut is right is not decidable before looking — it is a GUESS the
correspondence judges (`pick_split`), so every kind is a candidate, not a
setting.

| kind | what it reads |
|---|---|
| `connected_monochrome` | cells of one color that touch, CORNERS ALWAYS COUNT — two shapes meeting at a corner are one thing. The reading that said otherwise never once won on the puzzles we watch, and it tied with this one wherever nothing meets diagonally. |
| `connected_multicolor` | the same walk with the color question dropped: cells that touch are one piece whatever colors they wear. This is the DSL's Split[0], which two solved puzzles start from (05269061, 08ed6ac7) and nothing here could say. |
| `all_cells_of_one_color` | every cell of a color as one piece, joined up or not. |
| `minimal_repeating_set` | the SMALLEST block the picture itself repeats, found in the CONTENT: the shortest run of columns that repeats across, and of rows that repeats down. The block must fit twice, because this cut is CLAIMING the picture repeats and two whole copies is the least that shows it — so a lone copy with a stub is declined here, though the Pattern op still continues by it. One `repeat_periods` recipe in 0_core_types.hpp serves both, told apart by `must_fit_twice`. There is no largest block to ask for; the `repeating_motif_big` kind and the `biggest` flag it rode on were removed 2026-08-12. See [The smallest repeating block](The%20Smallest%20Repeating%20Block.md) and [A repeat has to fit twice](../4%20-%20To%20Do/A%20Repeat%20Has%20To%20Fit%20Twice.md). |
| `smallest_of_two_grids` | cut on a fixed step into equal blocks, one piece per block. The only kind that keeps ink of SEVERAL colors together, so the only one that can hold a tile (00576224's four-color 2x2). Available only when the sizes name a step — see `repeating_step_of`. |

All three pattern kinds — and only these — may hang off the edge (§3f), because a
block is a window onto a tiling that carries on and a shape is not. They are also
the three that route to `collect_motif_edges` instead of `collect_edges`.

**REPLACING THESE:** the DSL's `Split[0..2]` and `Colors` are the same readings
and are the single home of "what is an object" — see `cuda_dsl_pieces`. These go
when inference is wired to consume those instead.

**LATER:** cut along separator lines.

### How one is chosen

Try each and let the correspondence judge: FEWEST CELLS LEFT UNEXPLAINED first,
FEWEST PIECES as the tiebreak. Fewest pieces is what rules out the cutting that
explains everything by saying nothing — split a picture into single cells and of
course they tile it, but no split needs more pieces than that, so it loses to
any rival that also explains everything.

Judged over the WHOLE train set, one split for every pair. A transformation that
cuts the picture up differently from pair to pair would be a strange puzzle, and
holding it fixed is the stronger claim. Revisit if a puzzle needs it.

### Why the runners-up are kept

A split cannot tell that it read too coarsely. 00dbd492's two red squares merge
into one piece under one-piece-per-color, which explains everything with fewer
pieces and so wins — yet the squares get DIFFERENT interior fills. They behave
differently, and the correspondence has no way to know that. Only a consumer
that needs the finer reading finds out, by coming up empty.

So the order is kept and the next reading stays available; the trigger is that
consumer's own empty result, never a test inside the scoring. Re-reading under
`ranked[i]` is just another `detect_object_pairing` call — `infer()` is pure,
same argument as `background_election`.

What each ranked reading COST is kept too, so a readout can say why an order is
what it is instead of leaving it to be guessed at. Two readings can agree on
every number the ranking looks at (007bbfb7: one-per-color and a repeating step
both use 23 pieces and leave nothing over) and still be different cuts, so what
the cut ITSELF is has to be recorded as well.

### A cut that does not cut is not a reading

One piece means the motif is the whole picture, which is another way of saying
there isn't one — and one piece explains everything with fewer pieces than any
real reading can, so it wins the election for free (025d127b took it over three
actual shapes). So a pattern cut that yields fewer than two pieces is thrown
away whole.

Counted on what came OUT, not on how many blocks were tried: blank blocks are
not pieces. The step cut is exempt — its step is named by the sizes and answers
for itself.

### One block for both sides

The block is read off the INPUT, the same grid the tiling op takes its period
from. Measured separately the two sides disagree, because how big a block can be
depends on how much of the picture you are shown (017c7c7b's motif came out 2
rows on a 6-row input and 4 on a 9-row output), and pieces that are not the same
size cannot correspond.

### Cuts that were tried and dropped

**Every cell its own object** (`one_cell_each`, removed 2026-08-26). A single
cell is found wherever its color is found, so the cut answered every cell of
every grid and left nothing over anywhere — a perfect score, said by nobody, on
every puzzle at once. It only ever lost on price, and it won the moment a real
reading was a few percent dearer to say. What it was for, a grid whose cells
behave one at a time, the color cuts already say, and they say it without
claiming the whole grid.

**The biggest repeating patch** (`repeating_motif_big` and the `biggest` flag it
rode on, removed 2026-08-12). Periods behave like multiples: once a grid repeats
every p it repeats every multiple of p, so asking for the biggest is asking for
a greatest common multiple, which does not exist. It had an answer at all only
under an added rule — must divide the grid exactly — and what it then found was
always a whole number of the smallest one. A coarser view of the same fact,
bought with a caveat.

### A recolored match counts as explaining

For the left-over count that judges a cut, a recolor says this shape is there
wearing another color, which IS an account of those cells — the color itself is
another channel's business. Exact-only made every recolor puzzle tie at "nothing
explained", and then fewest-pieces handed the puzzle to whichever split found
nothing at all.

## 3f. find_in: what counts as "there" (moved out of code 2026-08-07)

**PARTLY HIDDEN COUNTS AS THERE.** A piece with something drawn on top of it is
still that piece (03560426 stacks pieces in a chain, so every one but the last
loses the cells the next one covers, and demanding the whole piece finds none of
them). Two separate questions per placement, in order of strength: does the
whole glyph fit under one color map, and failing that, is the glyph there AS
ITSELF with the rest behind other ink. Background showing through is never
hiding — it says the piece is simply not there.

**AND PARTLY OFF THE EDGE COUNTS AS THERE, WHERE THE READING IS A PATTERN.** A
tiling carries on past the frame, so a block may hang over the edge and the part
outside says nothing either way — the same silence as a cell behind another
piece, one step further out. Without it a 4-row block can never be looked for in
a 6-row grid at all (017c7c7b shows one and a half of its motif).

Shapes do NOT get this: a shape is a thing in a picture, not a window onto more
of it, and letting shapes hang off costs more than it buys (025d127b's three
shapes stopped fitting together).

## 3g. The background election (moved out of code 2026-08-07)

`Background · across_inputs` — the RANKED surviving background hypotheses,
judged by ROLE: background is the canvas figures sit on, the color that
dominates area AND touches every figure, over that side's own grids.

The two sides are elected INDEPENDENTLY — input canvas need not equal output
canvas (05269061: inputs sit on 0, outputs are a full repaint with none). Role
is answerable from one side's grids by themselves, no cross-side entanglement,
which is exactly why they elect separately. Candidates: any color present in
EVERY grid on the side, plus -1 ("no background").

The story judge (pairing residue) is gone: it was fooled by near-static
self-matches (00d62c1b), and role was doing the real work all along.

**LATER:** per-GRID background + the corpus-wide rule relating the two sides.

### Per-grid standout, and the corpus vote

Per-grid standout = the color with STRICTLY the most cells in this grid. No
margin — magic numbers struck too often; strict-most is parameter-free. A canvas
candidate is the standout in EVERY grid. A tie for top means no standout there
(00576224's equal colors → abstain). Separators and co-dominant rivals are meant
to be handled by OTHER facts later, not a tuned percentage here.

The corpus then VOTES: the color that is the canvas in the MOST grids is the
puzzle's background, extended to the grids where it's locally ambiguous. NOT
unanimity — one grid where a separator briefly out-areas the field (18286ef8:
orange is the field at 59/67% but dips to 36% under black in pair 1) must not
veto the verdict the clear grids give. Learn from the clear cases, extend to the
unclear.

**KNOWN GAP:** two-panel puzzles where the canvas is a MINORITY in MOST grids
(0520fde7) still can't be found by area — that needs the object story.

### The scored election

Sum weighted signals into a confidence per color. No single rule is a hard gate;
each is graded and votes. The top score elects IF it clears a confidence FLOOR,
else the direction abstains (-1) — the honest "ambiguous" answer. This replaced
the brittle unanimity/majority rules that kept trading one puzzle for another.
Weights and floor are FITTED to `tests/background_truth.txt`, not guessed. (No
separator term yet.)

### The output-side verdict

The canvas the output figures sit on. A winner that AGREES with the input canvas
is cross-side corroborated — the ground persisting through the transformation is
the strongest canvas evidence there is (412b6263: orange both sides, output dips
to 439).

Without input support the evidence is one-sided and must be dominance-CONFIDENT
(the same >= 500 bar as per_pair): an introduced canvas dominates its grids
(27f8ce4f: 555-666), a transformation artifact does not (25d8a9c8: the output IS
the answer stripes, no color passes 333). NOT the embeds gate — embeds is a
worst-grid number, and one grid whose figures majority-border something else
zeroes it (2f0c5170).

A canvas that PERSISTS comes first, before any confidence gate: if the input
canvas is still in every output, it stays the output canvas — a flood of answer
INK cannot dethrone it (506d28a5).

### Per-pair backgrounds must be confident

On a single grid the scored election's `standout_frac` saturates to 1000 (a
color trivially "wins" its one grid), so area is the only honest confidence
signal there. Take the winner only if it truly dominates its grid; else -1.

This is what separates a real per-pair puzzle (140c817e: each pair ~98%
dominant) from a BACKGROUNDLESS one (00576224 / 25d8a9c8: each grid coughs up a
weak ~25% coincidental color) — the latter must NOT read as per-pair.

Confident = dominates its grid absolutely (>= 500) OR beats the runner-up
DECISIVELY (>= 2x, the veto's own bar). A 48% field over a 14% runner-up on a
many-color grid is plainly the pair's ground (5034a0b5 pair 1), while
backgroundless near-ties (00576224: leader barely ahead) still abstain.

### The per-color background-ness prior

Rank in the measured order, most-background first. Measured by eye
(`Human Bias.md`, `scripts/color_pair_test.html`): background-ness ≈ inverse
brightness corrected by saturation.

```
0 black > 9 maroon > 5 grey > 6 magenta > 1 blue
        > 2 red > 7 orange > 3 green > 8 azure > 4 yellow
```

Used ONLY as the last tiebreak when area/embeds/story can't separate two canvas
candidates — the quieter color is the likelier canvas. A prior, not a verdict:
structure always overrides it.

### The fitted conventions

Every threshold in the election must be one of these, or a verdict needing none.
They are FITTED to `tests/background_truth.txt`, not derived — do not mint new
numbers alongside them.

| name | value | what it means |
|---|---|---|
| `BG_HALF` | 500 | an absolute claim (dominance, confidence, majority-of-cells) holds when it covers at least half, per-mille |
| `BG_DECISIVE` | 2 | one quantity beats another beyond co-dominance when it at least doubles it |
| `BG_NEAR_TIE` | 4/5 | two quantities are co-dominant when the smaller is within 20% of the larger. The one convention still lacking a verdict-form replacement |
| `BG_FLOOR` | 700 | the scored election elects only above this confidence, else abstains |

### The input-side verdict, strongest evidence first

The canvas the input figures sit on — what segmentation excludes. Walk
input-derived evidence strongest to weakest and take the first real canvas;
-1 means backgroundless or per-pair.

1. **UNDERNEATH.** A quiet color showing only as enclosed windows in every input
   is the ground showing through (1d0a4b61: black under the blue lattice). It
   beats the area winner by definition, since the pattern covering it is what
   wins the area.
2. **across_inputs.** The background is given in the input, undistorted by the
   transformation — unless a confident pair contradicts it (the veto).
3. **PER-PAIR.** Agreeing pairs name a global color; differing real pairs mean
   MODE=PER-PAIR.
4. **Pooled inputs+outputs.** Weaker, but still needs input agreement.

A weaker "global-most in_every" tier once lived below this. Deleted once its one
client (25094a63) accepted none — on a texture like 484b58aa it just elected the
loudest yarn in the weave.

There is deliberately NO one-answer merge of the two sides: collapsing them to
one color lies to somebody (2f0c5170: azure is the parsing ground of the input,
black the ground of the content/output). Ask the side you mean.

### The veto, and what rescues a color from it

A color DECISIVELY beaten inside some pair cannot be the global canvas: there it
is a figure sitting ON another ground (25e02866: red is pair 0's ground but the
figure on azure in pair 1 — its every-input presence is coincidence, not
canvas). Decisive = the local ground has at least twice the pretender's area in
that pair. A near-tie (06df4c85: fill 516 vs canvas 423) is co-dominance, not
contradiction.

Two things block a veto:

- **The global winner still HOLDS THE FIGURES locally** (embeds >= 500). A
  flooding panel can double the canvas's area without becoming the ground
  (57edb29d: azure panel on yellow wallpaper — the figures still sit in yellow).
- **CROSS-SIDE agreement.** When the output side independently elects the very
  color the veto targeted, the flip was ink, not ground (007bbfb7: chunky ink
  doubles black on the 9-cell inputs, but the 9x9 outputs re-elect black). A
  true veto survives — 25e02866's outputs do not re-elect the vetoed red.

### Per-pair majority, and per-pair mode

A strict MAJORITY of confident pair grounds names the canvas — learn from the
clear pairs, extend to the flipped ones (44f52bb0: black wins 4 of 6; red's two
wins are a chunky shape doubling the ground on a 9-cell grid, not a canvas
change). Unanimity is the special case. One pair's local win (00576224's
spurious 6) is still noise, and truly distinct pairs (140c817e: 9, 8, 7) have no
majority and stay per-pair.

PER-PAIR MODE is when the global election abstains yet EVERY pair has its own
real background that DIFFERS across pairs. Then the truth isn't "no background"
— it's a different canvas per pair (13f06aa5: red/blue/green). Constant WITHIN a
pair, varying ACROSS pairs. Detect-only for now; no consumer segments per-pair
yet.

Each pair elects over that pair's INPUT alone, NOT input+output pooled — the
transformation can flood the output with a new color (140c817e: each input is
98% one color, but the output fills with blue, so pooling ties and abstains).

A pair's ground may also be confirmed by HOLDING THE FIGURES (embeds >= 500),
the same local-canvas evidence the veto respects: 5b37cb25 pair 0's blue field
at 495 with a big co-dominant region still embeds the small figures. But holding
figures only counts for a color with BODY — one that is NOT a skeleton. A
separator skeleton is nearly all edge (458e3a53's yellow at gaps 925 embeds
every region it fences, but a fence is no table); a real field keeps interior
(5b37cb25's blue at ~540).

### Embeds: what it measures

Share of figure AREA embedded in a color, worst grid, per-mille. A figure is
embedded in whatever color forms the majority of its immediate outer
(4-adjacent) boundary. This is the local, layered reading — it handles a grey
FRAME (figures are embedded in the black canvas, not the frame) and interior
SPECKS (embedded in the shape around them, not the bottom canvas) that flat
touch/dominance/border tests all mis-call.

Area-weighted, not component count: on 00d62c1b the green structure shatters
into ~20 lattice fragments — counting them each as 1 lets the fragments outvote
the sea; weighting by their cells makes "most figure area sits on black" win
(black 588 vs green 412), matching the eye.

### Figure versus backdrop

A FIGURE is a component whose OUTER contour — the different-colored cells around
it reachable from the grid edge without crossing it, so interior specks don't
count — is majority-owned by ONE other component. That owner is what it is
embedded in.

Everything else is BACKDROP: border-touching components, and components ringed
by several owners. A framed canvas (5b37cb25) touches every frame piece plus its
own blobs, so no single owner, at ANY frame thickness. A verdict, not a
thickness threshold.

### The co-dominant band

Standout is decided among the SUBSTANTIAL colors — those co-dominant with the
leader — not by raw most-cells. A separator LATTICE (grey grid on 09629e4f, the
1-lattice on 06df4c85) inflates a foreground color's area past the true canvas.
Among the substantial colors the background-ness prior picks the winner: the
quieter color is the canvas. This is the swap experiment made law — black beats
grey even when the grey lattice has marginally more cells.

Leader-relative is the honest form. A largest-drop podium was tried
(2026-07-19) and the truth set rejected it: 412b6263 needs 62%-of-leader OFF the
podium while 06df4c85 needs 82% ON it, a boundary only a leader-relative band
can express. Gap-position sat black next to orange on 18286ef8 and the quiet
prior stole the separator grids.

### Underneath, and when a window is believable

A color is UNDERNEATH when in EVERY grid it is the quietest present and appears
ONLY as multiple enclosed windows — the ground showing through where the pattern
is erased. One enclosed blob is an object; many windows are the ground. A window
is believable as "underneath" only for quiet colors; loud enclosed patches are
figures.

And only when cut into a PATTERNED coverer — erased windows imply something was
erased (1d0a4b61: blue weave, gaps 789). On a SOLID field a quiet blob is an
object, not a window (2f0c5170: azure slab, gaps 96, black is the figure). The
coverer is the global area leader; if it is solid, no color may claim
underneath.

### What each score term is for

- **Area** leads. **Standout** corroborates. **Prior** is a light tiebreak.
- **Punched-through** = everywhere AND thin (the min of span and gaps),
  believable as ground only for quiet colors, so it is scaled by the prior
  (4e45f183: black mesh 882 × full prior = +176 lifts it past the floor; a loud
  mesh like 458e3a53's yellow gets a tenth of that).
- **Embeds** was DROPPED as a ranker: it can't tell a ground from a separator
  (figures sit on both), so it rewarded the wrong color half the time. It
  returns as a graded VOTE — the color most figure area sits in is canvas-like
  even when a flooding panel steals the area lead (57edb29d: yellow wallpaper,
  embeds 888 in its worst grid, area dips to 272). Better ground signals
  (enclosure/containment) come later.

## 4. Doubling-down roadmap (re-ordered 2026-07-02 per the Home.md directive)

0. **The eval expressibility audit (prerequisite, per Home.md).** Hand-write
   DSL solutions for eval tasks one at a time; write each verdict
   (expressible / inexpressible) to disk immediately. This decides whether the
   bottleneck on each task class is DSL expressivity or search/ranking — the
   two need different fixes, and everything below only helps
   expressible-but-unfound tasks.
1. **Output-first pruning / op masking** — the one deadline-feasible lever
   (Backward Deduction Design §5 option 1 + Op Masking note). Use the output
   type-signature (size relation via the existing size_rule, palette delta,
   is-subgrid) to SOFT-reorder the primitive list and seed the beam per
   puzzle. Soft mask only (reorder, never remove — FORCE already proved hard
   in-beam pruning loses, §2.1). Dimension/shape masking first: size_rule is
   already computed per puzzle, zero per-step cost. Then color-delta masking.
   Branching factor compounds multiplicatively across depth, and it attacks
   the per-step factor that bidi/depth tricks don't.
2. **Rebuild salience with the two fixes** — the one change proven to reduce
   attempts. Spec is complete and code-level in `3 - Search/Salience Anchoring.md`:
   `recognize_at`/`recognize` beside find_in, figure+ground scoring,
   REC_THRESHOLD ~0.9, re-rank only the ~30 refinement slots, behind a toggle.
   Measure on 0520fde7 vs baseline 21.7M/47.7s; over-fire rate must drop from
   ~97% to a few %.
3. **More annotation-source ops (fate b)** — the generalization lever. Pattern
   proven by Tile[3], SnapToAnchors (zero free variants — the ideal), the
   shape-maps, BidiColorMap-style findability. Next per Design Principles:
   infer the AXIS (drop the RotateToPortrait/RotateToTarget dance), then 2D
   lattice. Each one shortens true programs → beam survivability.
4. **Strong-prior hard gates at candidate admission** — reject wrong output
   dims / palette / object-count before scoring. Cheapest compute win;
   dimensions gate already exists as precedent.
5. **Op-variant intersection (fate a)** — per (op_type, slot), only expand
   variant values that work on every example. Start global-negative ("this
   value never works on ANY example → drop"), per-program later. Attacks the
   VAR_COUNT explosion (BidiColorMap 45, TileX/Y 30, color ops 10).
6. **New channels: shape / count / position** — struct has comment stubs; use
   the fold template. Then colors_consumed/introduced already exist — wire a
   consumer.
7. **propose()** — turn facts into candidate programs (engine-coupled;
   deliberately deferred until a detector needs it). This is the other door out
   of the −12 wall: recognize the puzzle class, emit the concept program, skip
   search.
8. **(long horizon) Witness/deduction front-end** — Backward Deduction Design
   option 2: a CPU pass that inverts tight-witness ops (rotate/crop/tile/
   recolor) backward from the output, leaving only the lossy residue to the
   GPU beam. Output enters as SPECIFICATION, not fitness — the only thing that
   changes the exponent. Prerequisite (not done): the per-op witness-tightness
   audit over cuda_solver.cu. DSL lever meanwhile: tight-witness ops deep in
   the tree, lossy ops (count/fill) outermost.

Before any of this: the quantification experiment from Constraining DOF — on
0520fde7, count how much of the 73M-attempt search attended to GROUND
(invariant colors) vs FIGURE. That number says whether the next pin is a 2x or
100x lever before building it.

## 4b. THE CURRENT BLOCKER: fan-out gates everything upstream (2026-07-20)

Everything we want next — unfusing the DSL, seeding pre-cut views, pushing
perception into inference — is blocked by the same thing: the per-level
BRANCHING FACTOR. Cost is roughly `(#op choices)^depth`. Unfusing an op makes
programs DEEPER (the exponent), so it is only affordable once branching is
small. Fan first, unfuse second. The order is forced, not chosen.

What does NOT reduce the fan (measured/decided today):
- RANKING (bg-first color literals): reorders the fan, does not shrink it.
  Kept — it is free and safe, but it is not the lever.
- PRUNING (hard-cut bg-semantic literals): shrinks the fan 10x and ran the
  13-puzzle re-export 2x faster — REJECTED anyway, it trades COMPLETENESS for
  speed and a lost puzzle outweighs any speedup. Steer, never kill.

What CAN reduce the fan without losing reachability:
1. ANNOTATION (fate (b) of the three fates): the op DERIVES its value from
   inference instead of enumerating it. 10 -> 1 with nothing unreachable,
   because the derivation IS the answer. SnapToAnchors is the existing ideal
   (zero free variants). Strictly more expressive than pruning.
2. EQUIVALENCE MERGE: two spellings provably producing the same grid collapse
   to one branch. Zero loss by construction (e.g. RemoveColor[bg] ==
   Foreground == filter-of-everything-else, once bg is known).
3. SEEDED VIEWS: perception (the seam) computed once in inference and handed
   to the beam as chain-starts, so segmentation costs no nodes. 30 seam-uses
   across 37 exported solutions — nearly every program pays for it today.

Chain: annotate/merge -> fan shrinks -> unfuse -> the DSL simplifies and
generalizes. Do not attempt the later links first.

## 4c. THE ACT BELONGS ON THE LINE (parked 2026-09-06)

A line is a SELECTION and what happened to it. That is the whole idea, and it
came out of watching the emitter give up on 009d5c81's neighbours.

WHAT THE EMITTER USED TO DO. It asked three questions of the whole reading -- is
there one turn, is there one move, was anything recolored -- and demanded every
edge agree. When two pieces turned differently it said nothing at all and wrote
"a turn that differs from piece to piece" into the missing-word list. So a puzzle
where two pieces do two things could never be said, however well it was read.

WHY THAT WAS THE WRONG AXIS. The edge already carries everything: `pose` says
which of the eight ways round, `row`/`col` against the piece's own `top`/`left`
say the displacement, and `recolored_to` says the colour it became when `exact`
is false. Each edge is already a line. Nothing needed computing; the aggregation
was throwing away what the reading had.

A LINE CAN SAY SEVERAL ACTS. One piece can turn AND move AND change colour. The
grammar gives each act its own sentence, so a three-act line is three sentences
that share the line's GET and SELECT. Only the last of them paints -- the rest
close with Withhold, which is exactly what that word was added for:

    CurrentObjects SelectColor[c] TurnRight      Withhold
    CurrentObjects SelectColor[c] MoveColumn[d]  Withhold
    CurrentObjects SelectColor[c] Recolor[r]     Union

THE SELECTION IS THE HARD PART, and it is a vocabulary limit, not a reading one.
Ten SELECT words exist and none of them can say "the pieces that turned right":
SelectAll, InvertSelection, SelectMostCommonColor, SelectObjectsWithHoles,
SelectColor, SelectObjectsContainingColor, SelectLargestObject,
SelectSmallestObject, SelectRow, SelectColumn. Colour is the only property a
piece knows about itself that a SELECT can also name, so the line key is keyed on
colour and a group that shares no colour cannot be spelled. That miss is worth
saying out loud -- naming the missing word is the valuable half of this reading.

BACKED BY EVERY PAIR, still. A line is kept only if the same
(colour, pose, down, across, recolor_to) appears in every pair. That keeps the
old guarantee -- one example's accident is not a rule -- while dropping the old
restriction that there be only ONE line. Disagreement stops being a failure and
becomes two lines.

STATUS: written, compiles, UNTESTED. 009d5c81 exercises none of it -- every
piece there has pose 0 and no displacement, so zero lines are built and its three
sentences come from the tied cut, color_swaps and shape_color_map instead. It
needs a puzzle whose pieces turn or move differently to prove anything, and the
first four candidates tried had stale inference logs (inference runs at load only
when the pane is already open), so nothing was learned from them.

## 5. Architecture debts (known, decided, not done)

1. **Engine reads inference via a bridge** onto puzzle fields, not the
   inference object. The god-header split it was waiting on is DONE
   (2026-07-26): shared types moved to `include/0_core_types.hpp`, and the
   include order is now core_types -> inference -> cuda_dsl -> environment, so
   inference no longer depends on the engine at all. The bridge itself remains.
2. **Scorer duplication**: the score formula lives in BOTH
   `dsl_score_kernel` (cuda_solver.cu) and `environment::analyze_output_score`
   — any salience/ranking work must keep them in sync (or not touch `total` at
   all, as the salience spec does).
3. **detect_target_objects stays in the engine** (per-output decomposition for
   the scorer, reads no direction — not an invariant). Deliberate.
4. **Map-op family cleanup**: rename ShapeColorMap → ShapeToColorMap (keep enum
   ordinal! update solutions/009d5c81.txt in same change), factor the shared
   occupancy-hash/table machinery. `4 - To Do/Map Op Consolidation.md`.

## 6. The discipline (applies doubly here)

- METHOD ORDER still rules: DSL expressibility → import-verified solutions →
  folds → benchmark → only then ranking/search. Inference work is exempt only
  where it's the ranking fix itself (salience).
- SHIP UGLY, THEN MEASURE (Research Synthesis §11): inference is exactly the
  kind of elegant theory that can eat months. Every item in §4 has a measurable
  gate (attempts, wall time, over-fire %, eval %). If a change has no number,
  it doesn't land.
- Every experiment on one puzzle (0520fde7 is the workhorse) must eventually be
  corpus-verified (1,147 puzzles) before it's trusted as a gate.
