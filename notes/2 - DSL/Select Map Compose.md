# Select → Map → Compose — the joint every op is made of

Date: 2026-07-20

## Current decision, 2026-10-03

Inference is the target now; search depth no longer justifies hidden steps.
The July 26 move of background removal and restoration into the engine is being
reversed. Colors and Split retain every real color. Removal is ordinary
`Colors SelectColor[c] DeleteSelection`, merged with UnionList only where a
single object is required. Output keeps its canvas frame; Trim names a crop.
Final fill is `Output SelectAll AddColor[c] Underlay`, with Trim before AddColor
for a BoundingBox output. The fixed-color rewrite comes before per-grid election.
SelectElectedBackground now elects from the current object layout, using the
same single-grid helper as the host. It adds whole-object marks and leaves
the stream untouched when no color is confident. Use `Colors` first to select
a color layer, then `DeleteSelection` to remove it. It reads no training pairs
or puzzle background field. Inference now compares the reading that
keeps all colors with one that cuts on each input's elected ground. Removal is
chosen only when that reading leaves fewer cells unexplained. Its replay uses
the same source for each pair, and its final fill is explicit:
`Input Colors SelectElectedBackground AddColor[11] Fit[1] Underlay`.

Fit stretches selected objects with nearest-neighbor sampling: Fit[0] targets
input dimensions, Fit[1] the output frame, and Fit[2] the painted output bounds
(including their offset). Unselected objects pass through. No painted bounds
means no selected result; it never invents a filled frame. Identity, Fixed and
Scale restoration use Fit[1]; BoundingBox uses Fit[2], preserving the final crop.
NotInferredYet cannot restore a frame. CPU and GPU share sizing and sampling.
Whole-input framed readings carry their own elected fill through the prefix.
Prefixes that discard ground (cut-out, layout, fractal) still need a faithful
intermediate frame before they can elect ground again; those inner readings
retain the fixed-color route, and dynamic outer cuts decline those frames.
Not every per-grid ground needs removing and restoring. SelectUnique
(2026-10-04) marks the one drawing with no twin, comparing frame size and every
cell, ground included, wherever the drawings lie. The odd pane is then kept
whole: `Input Panes SelectUnique Origin Union`. Reading and GPU share
`unique_drawing`.
The ground both ways (2026-10-04): a whole input is framed keeping its ground,
or ground first -- `Input Colors <ground> DeleteSelection UnionList SelectAll`
then the frame word, the ground filled only by the final fill. Frames that
delete the ground hand each grid's ground down to the reading inside them, so
nothing is elected from a grid only the host filled. The user has since asked
whether the ground should be deleted once for the whole reading instead; see
HANDOFF.
The ground owned once (2026-10-05) replaced all of that. `infer` asks the
election (3_background.hpp) for ground plans -- what is deleted first and what
is filled last, the two sides paired: kept, recolored, deleted, introduced,
every color kept, each input's own -- reads each plan's figures with no ground
in any reader (`read_figures`), and keeps the plan that leaves the fewest cells
unsaid. `say_the_plan` writes the plan's opening on every sentence that reads
the input, and its fill once, last. No detector takes a background; an empty
cell is nothing, and the matcher names empty like a color (a ray drawn where
nothing was, a figure and its ground trading places). VoteDenoise reads its
ground off the piece in hand (denoise_ground). The old reading modes, handed
grounds and ground-first frames are gone.
The historical session records below explain the old decisions, not the current
execution contract.

## Graveyard — deleted ops, DO NOT rebuild (2026-07-23/24/25)

Each was removed for a stated reason. If you feel the urge to add one back,
read the reason first — the need it seemed to fill is met another way.

- **InputWithoutBackground** — retired 2026-10-03. Source plus removal was
  fused. Its explicit composition is `Input Colors SelectColor[c]
  DeleteSelection UnionList` for an elected fixed color; a cut that follows
  can consume the surviving layers directly, without UnionList. Per-grid
  election is a separate next stage, not a hidden mode of these words.
- **Clear** — wiped the canvas to empty. DELETED: 0 solutions used it, and the
  only honest need ("start the output over") never arose because combines
  control paint order. A *list*-clear would be redundant with GET (a GET already
  replaces the list). Canvas-clear + list-clear both unnecessary.
- **Select[i]** (positional pick, keep object at list-index i) — DELETED:
  positional selection into an UNORDERED set is meaningless, and 0 solutions
  used it. LAW it established: **positional selection requires an explicit
  ordering context; ordering is a deliberate named act (SortBySize), never an
  accident of a producer.** Split is unordered BY CONTRACT — its scan-order is
  incidental (verified: nothing reads "the i-th of a Split"; the `objects[0]`
  reads are single-object results or caller-arranged templates). If positional
  select ever returns, it is legal only AFTER an ordering op, never after a bare
  producer.
- **InputPalette** — unfused to `Input Colors SelectInferredBackgroundColor
  DeleteSelection SortBySize[1]`. It was a fused GET (source + acts baked in).
- **Filter[c]** — unfused to `SelectColor[c] InvertSelection DeleteSelection`.
  It fused pick + strip + refit into one dropper.
- **RemoveColor[c]** — unfused to `SelectColor[c] DeleteSelection`. It was never
  a dropper (kept every object, erased the c-pixels) — just mis-filed as one.
  The mirror of AddColor; now a select+delete like Filter.
- **Largest[r]/Smallest[r]** → SelectLargestObject/SelectSmallestObject
  (lossless selectors; keep-only = `... InvertSelection DeleteSelection`).
- **Contains[c]** → SelectObjectsContainingColor[c] (same pattern).
- **Foreground** → SelectMostCommonColor (lossless frequency-ranked SYMBOLIC
  color selector — "most", resolved per list, ARGA-style). Old drop+resegment =
  `SelectMostCommonColor DeleteSelection Split[1]`. NOT the same as Largest
  (object-by-size) — it is color-by-total-area; the triple-chain in 0a1d4ef5
  peels three color layers, which object-dropping can't do.
- **TileByInputPattern[v]** — tiled the object list by a period measured on the
  RAW INPUT grid. DELETED 2026-08-18, and it was the wrong half of a pair: it
  reached past its own chain to measure a grid the chain had already moved on
  from (`Split … Recolor … TileByInputPattern` tiled by the untouched input),
  and it FUSED two acts — find the unit, then repeat it. Both halves already
  exist unfused: `Pattern` reads the unit off the object in hand, and `Tile[2]`
  repeats an object by its own size. LAW: an op measures the thing it was
  handed. Reaching for a grid outside the chain is how an op stops composing.
- **HolesToColorMap / OutputColors / OutputPaletteBySize / InputObjects /
  InputFiltered / Target** — earlier renames/unfusions (see session logs below).

## Kept, and why — ops that look drainable and are not

These are LIVE ops. They sit here rather than in the graveyard because the
question was asked and the answer was no; a reader who finds them under
"deleted, do not rebuild" learns the opposite of the truth.

- **MaskBy/MaskOut/ReadUnder** — NOT drained. They filter the object by the
  CANVAS beneath it (external predicate), so forcing select→delete would make a
  selection that goes STALE the moment the object moves (our marks travel with
  the object; a canvas-relative mark can't). Kept as canvas-reading ACTs (MAP).
  The mirror-dual of Intersection (there the canvas is the victim, here the
  object is).

## The shape

Every op that does something to objects is three decisions, not one:

| slot | question | who answers it today |
|---|---|---|
| **select** | which parts? | fused inside other ops; no vocabulary of its own |
| **map** | what do they become? | `Recolor[c]`, the `…To…Map` tables |
| **compose** | how does it land on the canvas? | `Union` / `Underlay` / `Intersection` / `Difference` |

Fuse two slots into one bracket and you pay their **product**, because one number
must cover every combination. Keep them apart and you pay their **sum**.

## Why this note exists

The same joint was written down three times, weeks apart, each time as a local
observation while fixing something else. Nobody put them side by side:

- **2026-06-27** — [[Map Op Consolidation]]: the map ops split into **key → value**,
  and the naming convention `<Key>To<Value>Map` says it out loud.
- **2026-06-30** — [[Op Algebra]] §"The two independent choices": *"every binary op
  is really two orthogonal decisions, and the confusion comes from them being
  baked together per-op"* — footprint × color-source.
- **2026-07-20** — tracing `HolesToColorMap`'s 50 variants: tens digit = hole
  count (select), ones digit = color (map). 5 × 10 = 50.

Three tickets, one disease. Expect a fourth rediscovery if this is not the home.

## Two verdicts already earned — do not re-litigate

1. **Fusing select+map costs a product.** `HolesToColorMap` = 50, `BidiColorMap`
   = 45. These are not 50 and 45 real choices; they are 5×10 and 10×9/2.
2. **Fusing compose's own two knobs was tried and REJECTED.** [[Op Algebra]] §"Deferred
   redesign" proposed `Compose[footprint, color_source]` and shelved it because it
   *"trades the current flat, individually-named, search-friendly ops for one
   high-arity-annotation op (bigger per-op branching)."* Merging grew the bracket.
   Splitting shrinks it — same principle, opposite direction, which is why the
   unfuse may pass where the merge failed.

## What blocked the split, and what unblocks it

The chain is unary: one live value. A **destructive filter** (`Contains`-style,
returns a sublist) throws the non-matching objects away, and there is nowhere to
hold them — so chaining four map entries collapses on the second. That is exactly
why `HolesToColorMap` hand-rolls `res = src` + `continue`: pass-through was the
only way to express a partition with one register.

Three escapes exist. Two are expensive:

- **Second register / partition** — needs an execution-model change. Not doing it.
- **Held chains (Hodel)** — his DSL is unary too; the only difference is that a
  *constant may be a program* (`mapply(f, objs)`), so his brackets each hold one
  decision. Same expressive power as ours, fewer spellings. Costs a new encoding:
  `gpu_flat_op {type, value}` has one byte and cannot name a subprogram. Note he
  has **160 primitives to our 73** — he is not fusing, he is shattering.

The third is cheap, and it is ARGA's (AAAI-23, in `concept_sources/arga_graph_dsl/`,
outside the repo — see `Home.md`):

> **A filter is a predicate on ONE object** — `filter(node, params) -> bool`
> (`ARCGraph.py:57`). It takes no list and returns no list.

Nothing is deleted, so nothing needs holding. The rest-of-the-list problem cannot
arise. Their whole DSL is ~11 ops: 6 filters (color, size, degree, neighbor-size,
neighbor-color, neighbor-degree) and ~5 transforms (update_color, move_node,
move_node_max, add_border, remove_node). Small *because* unfused.

Two details worth stealing verbatim:

- **`exclude: bool`** — every filter carries its own negation. One flag, not a
  second op. This is [[Op Algebra]]'s KEEP/REMOVE 2×2 as a bit.
- **Symbolic params resolve per grid** — `"most"`, `"least"`, `"max"`, `"min"`,
  `"same"`, `"odd"` are derived facts, not searched literals. We do this in
  exactly one place (`Recolor`'s `color >= 10` → rarest/most-common,
  `cuda_solver.cu:994`). ARGA does it everywhere. This is annotation living
  *inside* the select slot.

Caveat on ARGA's numbers: reported on a **160-task subset of ARC-v1**, not the
full set and not v2. The DSL-size lesson transfers; the score does not.

## Where each slot stands

- **compose** — complete. [[Op Algebra]] is the map; *"the algebra is now complete"*.
- **map** — partly annotated. `ShapeColorMap` and `ShapeToInteriorColorMap` learn
  their table host-side and cost **1 variant each**. `HolesToColorMap` has the
  same key→value shape but its table is *searched*, one entry per node, at 50-way
  fan. It is the family member that never got its detector.
- **select** — 8 lossless marking ops as of 2026-08-16. The remaining gap is
  narrower than "no vocabulary": there is no way to pick by a QUESTION about an
  object (fewest non-background cells, etc.), only by name and by size rank.

## Evidence: 0a2355a6

The only solution using `HolesToColorMap`:

    InputPalette Split HolesToColorMap[11] [23] [32] [44] Union

Four chained nodes spelling one map, one pair per node, because the op can only
say one pair at a time. Verified 2026-07-20 by counting enclosed holes per input
object against output color across all four training pairs:

    holes 1 -> 1    holes 2 -> 3    holes 3 -> 2    holes 4 -> 4

Clean and total, no conflicts. The map is **readable from the pairs** — fate (b),
input-derived, not a free choice. The search spent four fused nodes at 50-way fan
rediscovering a fact that falls out in one pass.

## BUILT 2026-07-20 — and what it cost

`SelectHoles[h]` (5 variants) + `Recolor` honouring a `selected` bit on
`gpu_fixed_object` (default 1, so every older op is unaffected).
`HolesToColorMap[50]` deleted, `gpu_object_hole_count()` is the one home for the
flood recipe, `solutions/0a2355a6.txt` migrated. Base fan 485 -> 440.

**It executes and it is no longer findable.** `import_sol` on 0a2355a6 returns
`solved=true` — the identity holds. But the search then ran 5+ minutes without
converging on a puzzle it used to solve. Kept anyway (2026-07-20): the op is
right, the finding is a search problem, and reverting would trade the general
form for a number.

Two causes, and the second is the one that bites:

1. **Depth is the exponent.** 7 nodes -> 11. Base fell 9%, depth rose 57%. Four
   extra levels at base ~440 is ~440^4 more space; the base saving is noise
   against it. This is [[Inference Master]] §4b landing exactly as written —
   *fan first, unfuse second* — ignored while quoting it.
2. **The productive prefix went invisible.** `SelectHoles[1]` alone changes no
   pixels, so the blind cell-anchored scorer gives it zero credit; the pair only
   pays off once `Recolor` lands adjacent and in order. The fused op at least
   moved pixels at every node. Unfusing deepened the "-12 wall" it has to climb.

The general lesson, which outlives this op: **an unfused chain needs either an
annotated value (so the search never traverses those nodes) or a scorer that can
see a half-finished pair.** Splitting select from map is right; doing it while
the value is still searched and the scorer is still blind is what broke it.

## The grammar — decided 2026-07-21

A solution line is a **sentence**. The GUI tabs already state its order
(gui.hpp:687, "pipeline-authoring order"):

    GET -> REDUCE -> SELECT -> TRANSLATE -> MAP -> GENERATE -> COMBINE

Collapsing the acting tabs (reduce/translate/map/generate) into one ACT, the
grammar is:

    NOW:    GET (SELECT* ACT+)* COMBINE       <- built, as a gate
    TARGET: (GET SELECT ACT COMBINE)*         <- the target as of 2026-08-16

Four roles only: GET (fetch), SELECT (mark, lossless), ACT (change objects —
segment/transform/reduce/drop all fold in here), COMBINE (paint).
Quantifiers: GET once; SELECT* (acting on everything needs no pick); ACT+ (a
sentence must DO something); COMBINE once, never chains.

KEY: a dropping "selector" (Filter/Largest/Contains) IS an act until made
lossless — it consumes objects. Only marking selectors are true SELECT, and
losslessness is HOW the grammar gets teeth: each conversion collapses that op's
ACT-fan (SelectHoles: 50->5) AND gives the grammar a real SELECT to constrain.

DONE 2026-08-16: the draining finished. All 8 selectors mark, none drop; every
one was checked in the kernel. Dropping is spelled out loud now, as
`Select… InvertSelection DeleteSelection`. SELECT is a populated class, so the
grammar has something to bite on — the precondition this note was waiting for.

Measured against the 37 solution files: every sentence ends in COMBINE (77/77),
and the SELECT->ACT cycle repeating inside one sentence is common and real
(0a2355a6 has four). The order holds ~85% unforced; the exceptions are legit
(second-pass select, canvas-continuation), not scrambles.

### Two stricter forms were tried and rejected on evidence

- **Linear tab order, no repeats** — forbids the `(SELECT ACT)*` cycle. Rejected:
  0a2355a6 genuinely needs four cycles in one sentence.
- **`(GET SELECT ACT COMBINE)*`** — one full sentence per layer, each ending in
  its own COMBINE. Attractive (every layer scorable, matches layer-locking) but
  **measured NOT behavior-preserving**: expanding 0a2355a6 into four sentences
  gave `solved=false` (import_sol, 2026-07-21). Swapping the per-sentence
  COMBINE from Union to Underlay ALSO failed — so the cause is NOT paint order.
  The mid-pipeline object list is load-bearing, and the four-sentence spelling
  lost it.

  CAUSE CORRECTED 2026-08-16 — the original read was "a COMBINE consumes the
  object list". The kernel says otherwise, and always did on the record: all
  four combines PASS THE LIST THROUGH unchanged (checked per exec function; see
  the verified kernel facts under the 2026-07-23 section, which already said
  this two days later and was never reconciled with this paragraph).
  What eats the list is the **GET**, not the paint. The four-beat form mandates
  a GET at every phrase start, and that fresh fetch discards the list COMBINE
  had just handed forward. The measurement stands; only the cause was wrong.

### Why B, on the three axes

- **search** — best: B changes nothing. The object list already flows through
  `(SELECT ACT)*` within a sentence; B just names the shape the search already
  makes. No new op, no new register, no new branching.
- **humanness** — good: one layer = one sentence, cycles inside = "for these do
  X, for those do Y."
- **generality** — lower ceiling than the alternative, and this is the conscious
  cost: B cannot resume a PAST sentence's object list across a COMBINE.

### The escape hatch (form A), not built

The original claim: `(GET SELECT ACT COMBINE)*` becomes lossless only with an
**object-list getter** AND the list **preserved across COMBINE** — a second
register, the execution-model change ruled out twice, priced at the candidate-set
growth that has lost every time (see the -12 wall / salience note).

PRICED WRONG — WITHDRAWN 2026-08-16. Half of that is already true: the list IS
preserved across COMBINE, in the kernel, today. So the register was priced
against a cause that does not exist. What the four-beat actually needs is
smaller: **GET optional at phrase start**, so a phrase may continue on the list
the COMBINE passed through. Today's legality rules already allow that shape —
SELECT and ACT after a COMBINE are legal. Nothing new to build.
UNVERIFIED: whether the 0a2355a6 four-sentence expansion still fails. That is a
re-run, not a debate — do it before trusting either version of this section.

## RESOLVED 2026-07-22: selection is the Excel model, at pixel depth

> SUPERSEDED 2026-07-25 by "REVERSED: selection is object-only" below. The
> Excel model and the boolean algebra (points 2-6) all survive; only point 1
> flipped — selection is per OBJECT again, not per pixel. Read the reversal
> section for why pixel depth was a wrong turn and why object-only is not a
> regression.

The old question "narrow or overwrite?" died when the second selector arrived —
the decided answer is neither: selectors ADD.

1. **Selection is per pixel**: `gpu_packed_bits` (1 bit/cell, same local
   indexing as the color mask) replaced the per-object `selected` int. One
   source of truth — "object selected" = any cell marked. Whole-object
   selectors mark all cells; SelectColor marks exactly the c-cells, which no
   per-object flag could say.
2. **Nothing selected at birth.** Fresh lists come out of `gpu_acquire` with
   every bit zeroed — the chokepoint owns the invariant, same as the old bug fix.
3. **Acts touch selected cells and NOTHING else, ever.** No selection = the act
   does nothing. There is NO fall-back to "everything" (tried a list-level
   flag with fallback; rejected: pure grammar beats convenience).
4. **Selectors are pure adders** (`|=`): the two real selectors plus SelectAll.
   Additive chains spell union: `SelectColor[1] SelectColor[2] Recolor[5]` =
   recolor the 1s and the 2s.
5. **The full boolean algebra is closed** (2026-07-23): OR is the additive
   selectors; **InvertSelection** is NOT. OR + NOT = everything, so "big AND
   red" is now spellable by De Morgan — no `&=`, no deselector needed. The
   parked "narrowing" question is dead.
6. **ClearSelection was deleted** — it was just `SelectAll InvertSelection`.
   The authoring loop is still Excel's: select additively → process → clear →
   repeat, the clear now spelled with those two.
7. **DeleteSelection** is the one delete verb: erase the marked cells, refit
   bboxes, drop emptied objects. `Filter[c]` (deleted) unfused to
   `SelectColor[c] InvertSelection DeleteSelection`.

### Naming convention — the name carries the SPECIES (2026-07-23)

A selector's name says which grain it marks, so a reader never has to guess:

- **`Select<CellProperty>`** marks matching CELLS. Cell-keyed.
  `SelectColor[c]`, `SelectInferredBackgroundColor`.
- **`SelectObjects<Property>`** marks WHOLE objects (every cell of the ones that
  match). Object-keyed — an object property can't point at part of an object.
  `SelectObjectsWithHoles[h]`, and the pattern the red-tab drains follow:
  SelectLargestObject, SelectObjectsContaining[c], SelectObjectAt[i]…

(SelectHoles was renamed to SelectObjectsWithHoles the day the convention was
set — the bare name had genuinely misread as cell-keyed.)

Migration debt, precisely stated: **only Recolor and DeleteSelection read the
bits today.** Every other act still touches everything because it never looks.
The drain = teach acts to read the bits, one act at a time.

## REVERSED 2026-07-25: selection is object-only, not pixel

The pixel-depth resolution above was reversed. Selection is per OBJECT again:
one `bool selected` per object, no `gpu_packed_bits`. This section is the whole
reasoning, so the decision is never re-argued from scratch.

### The bug pixel-depth dragged into every act

A cell-level mark forces every object-moving act to answer an unanswerable
question: "what does it mean to Shift/Recolor an object when only SOME of its
cells are marked?" Shift moves by the object's own width/height — a partial
mark has no coherent displacement. Any answer is arbitrary. Object-only kills
the question at the root: every act always sees whole objects, so the ambiguous
case cannot arrive. No per-act "is this mark whole or partial" rule needed.

### The expressiveness worry, and its answer

"But I need to edit INSIDE an object!" — Yes, sometimes. The answer is NOT to
mark cells. It is to **carve the piece into its own object first** (Split /
Colors), then select that whole object. Sub-object reach did not vanish — it
MOVED to the carving side, where it belongs: the thing you edit is a thing that
exists in the object list. "Recolor one arm" = split the arm out, select it,
recolor it.

### The one thing that genuinely died: "object = one color"

Under color/connectivity segmentation, an object was monochrome — color WAS
identity, and selection/recolor/count were trivial because "the object" and
"the color" were the same handle. `Partition` breaks that spell: it divides by
SPACE, not color, making a region that HOLDS many colors, whose identity is
spatial. Multi-color objects are first-class now. The monochrome simplicity is
gone — but it was always going to die, because ARC needs "that square", not
just "that color". Object-only did not kill it; `Partition` did, and `Partition`
had to exist.

### Granularity vs. what a selector may READ — the distinction that saves it

Two things got conflated; separate them and the crisis dissolves:

1. **Selection granularity** = what you can PICK. This is OBJECT, always. You
   cannot mark a stray cell. Acts touch whole selected objects.
2. **What a selector may LOOK AT to decide** = anything, including cell content.

A selector reading cells to DECIDE which whole object to pick is NOT pixel
selection. `SelectLargestObject` already reads every pixel (to count size);
`SelectObjectsWithHoles` reads interior topology; `SelectObjectsContainingColor`
scans cells. All look inside; all select whole objects. A content-query selector
("the object with the fewest non-background pixels") is the same species — reads
cells, outputs a per-object flag. **Reading inside is fine; MARKING cells is
what we banned.** So content queries keep object granularity intact; they do not
drag us back to pixels.

### The remaining real gap (named, not yet filled)

Carving is by COLOR (Colors) or CONNECTIVITY (Split). Neither carves by SPACE.
So a `Partition` tile — a multi-color region defined by space — cannot be split
INSIDE without re-fragmenting the by-space grouping Partition just made (Colors
after Partition is self-defeating). To ACT on part of a tile there is no carve
today. But to SELECT AMONG tiles by content (fewest non-bg pixels, etc.) needs
no carve — a content-query selector does it. 09629e4f is the first citizen:
"the square with the smallest count of non-color-0 pixels" is one object-level
property, and the old recipe faked it with erase-then-count (SelectColor[0]
DeleteSelection SelectSmallestObject) — a workaround for a missing primitive.

### What the flip cost, measured (theories die in the logs)

Object-only + the frame fix (below) broke ~20 curated solutions, then recovered
almost all. Net: the vocabulary got BETTER, not worse — e.g. 007bbfb7 the solver
re-found a SHORTER program (`Input Colors Fractal ...`) than the old
select-delete chain. The breakages were solutions leaning on the removed
capability or on an implicit tighten; each was a recipe fix (a carve, a Trim, a
corrected Underlay/Split), not a lost puzzle.

### The frame law: nothing tightens implicitly — Trim is the only shrinker

Segmentation (`Colors`, `Split`) used to TIGHTEN each object's bbox to its
pixels. That was an implicit shrink hiding inside a producer. LAW now: **a
producer never touches the frame; fitting the box to the pixels is a deliberate
act, and only `Trim` does it.** `Colors`/`Split` keep the SOURCE object's own
bbox as-is (each layer/blob inherits the parent's box), painting cells at their
place, rest transparent. Consequence: every recipe that relied on the old
auto-tighten now needs an explicit `Trim` where it consumed the glyph — this is
correct, not debt.

- WATCH: the ceiling-canvas trap. `Colors` once painted onto a 30x30 canvas and
  read the frame off THAT (=30x30), blowing every box to full grid. The fix was
  to iterate PER SOURCE OBJECT and copy the source object's own bbox — never
  derive the frame from a scratch canvas. (Cost a Fractal timeout before caught.)

### The selection law: EVERY op respects selection — obligation by role

The old migration debt ("only Recolor and DeleteSelection read the marks") was
too small a framing. The real law: **no op may silently ignore the selection
state it is handed.** But "respect" means a DIFFERENT obligation per role — it is
not one uniform rule:

- **GET / producer** — nothing is selected yet, so nothing to respect. (Fresh
  lists come out unselected; the `gpu_acquire` chokepoint owns that.)
- **Selector** — WRITES marks, does not read them (except InvertSelection, which
  reads to flip). A selector may look at cell content to decide WHICH whole
  object to mark (size, holes, color, non-bg count) — reading inside is fine;
  marking cells is what is banned.
- **Act** (transforms pixels: Recolor, Shift, flips, scales, …) — touches ONLY
  selected objects, and NEVER falls back to "everything" when nothing is
  selected. The bare use gets an explicit `SelectAll`.
- **Reducer / list-fold** (UnionList, IntersectionList, and kin) — must NOT
  DESTROY the marks. Fold PER selection group, so the selected/unselected
  distinction survives the merge as object identity (one merged object per
  group). This is what unlocked `... UnionList Partition`: the walls stay marked
  through the merge.
- **Structural op that reads selection as a ROLE** (Partition: selected = the
  divider walls) — strict, like an act: no selection means no walls.

Why per-role and not uniform: selection is object-level state, and an object's
relationship to it differs by what the op DOES to the object — transform it
(touch only marked), fold it (preserve marks), cut by it (marked = walls). The
one invariant across all: **the selection you were handed must change the
result; an op that ignores it is a bug.**

Migration status: acts still largely touch everything (they never look). Each is
a one-op fix — teach it to gate on `selected` (`if (!obj.selected) pass through;`).
Reducers DONE: UnionList and IntersectionList both fold per selection group, so
the mark survives as object identity.

PARKED — act migration is DEMAND-DRIVEN, not a block. An act only needs the gate
the day a puzzle wants it selective (transform SOME objects, not all). Gating all
~30 acts up front is speculative; do it one op at a time when a solution asks,
with the import_sol sweep as the safety net. Same discipline as draining the red
tab: need drives the order, not completeness.

### BANKED 2026-08-19 — what the grammar was worth, and that most of it is collected

The grammar's cut is measured, not estimated. Counting VARIANTS (what the
generator actually enumerates, not op names): 402 total — GET 3, SELECT 48,
ACT 347, COMBINE 4.

- mid-chain with nothing picked: **113 of 402 offered, 28%**
- mid-chain with a pick live:    **399 of 402 offered, 99%**

Almost all of that is Rule 0: 33 of the 52 acts do nothing before a pick, so
they are not offered. The cut lands at the start of every chain and again after
every fetch, carve or delete — and nowhere else. It compounds with depth,
roughly (113/402)^d, about a thousandth of the raw fan by depth 6.

MOST OF THE PAYMENT IS NOW COLLECTED — not all. The rules are live, the
lookahead gradient they needed is live (cuda_solver.cu, "score the canvas AS IF
the pending list were Union'd"), and the two parked follow-ons turned out to be
worth little:

- **force by construction** — pointless. The gate checks legality BEFORE
  create_unary_node, so no illegal candidate is ever built, hashed or scored.
  Construction would save the comparisons and nothing else.
- **sentence-at-a-time** — mostly moot. Its argument was that unlanded chains
  are unscorable, and lookahead already fixed that. What remains is depth
  accounting, and this is a beam over levels rather than iterative deepening,
  so that argument is weaker than it sounds.
- **algebraic pruning** — the one piece never built, and the part of the
  payment still outstanding. Local, needs no restructuring, each rule
  kernel-checkable: TurnRight x4 = nothing, FlipVertical x2 = nothing,
  TurnRight x3 = TurnLeft, the same combine twice, ScaleUp[k] ScaleDown[k].

Which is why the axis with room left is INFERENCE, not grammar. They multiply
rather than compete, and they cut differently:

| | grammar | inference |
|---|---|---|
| prunes by | shape — which op may follow which | fact — what value this op takes here |
| scope | every puzzle, identical | this puzzle only |
| measured | branching to 28% | HolesToColorMap 50 -> 1; colour literals 10 -> palette |
| overfits? | cannot, it is puzzle-independent | can, so it must steer and never kill |
| buys reach? | never | yes — a new fact can make a puzzle expressible |
| headroom | none, it is built | open-ended |

Grammar only makes the search faster. Inference can make it ABLE, and a puzzle
that cannot be expressed does not care how fast the search is.

Nearest concrete fan to pin, measured the same day: TileX and TileY search 30
periods EACH — 60 of 402 variants, 15% of the whole fan — guessing a period that
`repeat_periods` already computes. Same need TileByInputPattern was built for;
it was deleted for measuring the wrong grid, not for the idea being wrong.

### PARKED — priority frame: generality vs. speed (2026-07-25)

The parked work splits on WHAT it buys, and that decides which to unpark:

- **Act-migration = GENERALITY.** Widens what is EXPRESSIBLE ("transform some
  objects, not all"). Treats the 0% scoreboard — puzzles the solver cannot spell
  at any budget.
- **Grammar-force + algebraic pruning = SPEED.** Shrink the candidate set
  (generate-legal-by-construction; drop redundant spellings). Say nothing new —
  just find the same answer cheaper.

The trigger tells you which to reach for, per puzzle: a puzzle that TIMES OUT is
compute-bound -> speed. A puzzle with NO expressible program is
expressiveness-bound -> generality.

BUT the split is not clean — SPEED ALSO BUYS GENERALITY. Under a fixed time
budget, "cannot solve" and "times out before finding it" look IDENTICAL on the
scoreboard (both 0). So making search faster CONVERTS answers-you-can-already-
spell-but-couldn't-reach into real solves. And you usually can't tell the two
apart up front: a timeout may be "too slow" OR "no expressible program, so it
searched forever finding nothing." This argues SPEED FIRST — it helps the
compute-bound puzzles, can't hurt, and REVEALS the true expressiveness gap:
whatever is STILL 0 after search is fast is the set that genuinely needs new
vocabulary. Speed doesn't just help generality; it shows where generality is
actually needed.

### PARKED — the generator believes marks it cannot see (2026-08-19)

`set_generation_debt` carries `something_picked` through EVERY act, because when
it was written an act was an act. The kernel disagrees on four of them:

- **DeleteSelection** hands back the survivors, which are exactly the objects
  that were NOT picked. Nothing downstream is marked.
- **Split**, **Colors**, **Partition** carve new objects, and new objects carry
  no marks.

So after any of those the generator still thinks a pick is live, and Rule 0
offers all 21 pick-needing ops when not one of them can do anything. That is the
fan Rule 0 exists to remove, coming back exactly where `Split` and `Colors` sit —
which is the opening of most chains.

Costs candidates, never answers: every one of those is a longer spelling of a
program the search already has. So it is SPEED, and it waits with the rest of
the speed work.

Found by walking all 33 solutions against the same model and watching it be
wrong four times (the SelectAll sweep, 2026-08-19).

The fix in this project's own idiom: the fact is per-op, so it belongs in a
COLUMN on the op list beside ROLE — `sets` / `keeps` / `clears` — and a second
column for whether the op needs a pick at all, replacing the hand-written
`needs_a_pick`. A column cannot be forgotten; the macro will not compile without
it. What a column still cannot do is be WRONG-proof: the kernel is the truth and
the column is a claim. The version that cannot drift is a startup self-check —
run each op once over a list holding one marked and one unmarked object and
observe what comes back. `dsl_pieces_kernel` already exists to ask an op what it
made, so 67 assertions become 67 measurements.

### PARKED — the grammar, both forms (2026-07-25)

Two grammar goals sit parked, and BOTH now carry drift from the object-only flip:

- **Force NOW's grammar** `GET (SELECT* ACT+)* COMBINE` by CONSTRUCTION, not by a
  gate. Today the generator emits freely and a per-op legality check kills the
  bad ones (form B, the gate BUILT 2026-07-24). The clean version offers only
  legal next-ops, so no illegal candidate is ever built. See "the generator gate"
  section for what exists.
- **Reach TARGET** `(GET SELECT ACT COMBINE)+` — the pure four-beat sentence. Only
  reachable with form A's persistent register (an object list preserved across
  COMBINE), whose cost is the candidate-set explosion ruled out twice. Adopt only
  if a real corpus puzzle needs a late phrase to resume an early phrase's object
  list across a paint — unknown whether any does. See "escape hatch (form A)".

- DRIFT (2026-07-25): the gate's legality/role rules and the whole "decided
  2026-07-21" grammar section were written BEFORE the object-only flip, so the
  rules had to be RE-DERIVED against the object-only model before being forced.

  RE-DERIVED 2026-08-16. Two of the three worries are settled, one is real:
  - ROLES — settled, and by construction. Every op names its own role in a
    column of the op list (cuda_dsl.cuh), and `is_get_op`/`is_select_op`/
    `is_combine_op` read that column. No hand-kept list survives to drift, and
    an op cannot be added without a role.
  - PIXEL-ERA VOCABULARY — gone. All 8 selectors mark; nothing drops.
  - WHO PAYS THE SELECTION DEBT — **still wrong.** The rule clears the debt on
    the first ACT. That was right when reducers destroyed marks. They preserve
    them now, on purpose: UnionList folds per selection group so
    `UnionList Partition` keeps the walls marked. UnionList is an ACT, so it
    clears a debt it never read. Consequence: `SelectColor[c] UnionList Union`
    and `UnionList Union` paint the same pixels — one program, two spellings,
    exactly the fan the grammar exists to kill.
    The cause: paying was tied to BEING AN ACT; it belongs to READING THE MARKS.
    The catch, before rewriting it: act-migration is demand-driven, so today only
    Recolor, DeleteSelection and Partition read marks at all. Tie the debt to
    reading and it bites harder than the ops currently justify.

## Session log 2026-07-21 — panel purified, first two GETs unfused

All verified by import_sol, build clean throughout.

1. **Panel in grammar order**: GET · FUSED GET(red) · SELECT · SELECT SHAPED
   ACT(red) · REDUCE · TRANSLATE · MAP · GENERATE · COMBINE. Display order via
   `tab_display` (gui.hpp) — enum ordinals untouched. Red = migration debt made
   visible; the two red tabs are the worklist.
2. **`Target` → `Output`**, unfused and de-annotated: `Target[c]` ≡
   `Target[10] AddColor[c]` proven by execution (9/9 solutions), then variant
   dropped (11 → 1), renamed Output (enum, string, `gpu_exec_output`, comments,
   solutions). It reads the LIVE CANVAS, not the answer — the old name lied.
   Also renamed: comment-only `RotateToTarget`/`RotateToPortrait` → real op
   names; `GPU_TARGET_RAW` → `GPU_COLOR_ANY` (it is MaskBy's any-color wildcard).
3. **`InputObjects` DELETED** ≡ `Input Split[1]` — the mode digit matters:
   Split[0] merges colors (any_color=true), Split[1] splits per color, and
   InputObjects segmented per color. First try with [0] failed import; reading
   the flag found [1]. 8 solutions migrated (0b17323b fails at ORIGINAL too —
   pre-existing, like 1190bc91).
4. **`InputFiltered` DELETED** ≡ `Input Filter[c]`; `InputFiltered[10]`
   (keep-all) ≡ bare `Input`. 2 solutions migrated.
5. **`selected` uninitialized-memory bug — fixed at the cause.** The in-struct
   default never runs on device (objects live in reused pool memory), so any
   fresh-built object carried garbage marks and Recolor's guard skipped real
   objects NONDETERMINISTICALLY (same build: solved, then not — the signature
   that finally identified it, after "stale state" and "ordinal shift" theories
   both died). User's eye pinpointed it: "Filter[4] Recolor[3] — the Recolor
   didn't work." Fix: `gpu_acquire` marks all objects selected — the single
   chokepoint every list flows through; the 8 scattered per-builder inits were
   then REMOVED as redundant. One home for the invariant.

Op count 73 → 71. Base fan 485 → ~420. FUSED GET remaining: InputPalette,
OutputColors, OutputPaletteBySize — these point at CACHED puzzle-level lists
(computed at load, not derivable by one act op on Input), so they are the hard
kind: unfusing needs a new act token (e.g. `Palette`), not a rewrite.

## Session log 2026-07-22 — FUSED GET tab emptied and deleted, selection generalized

All verified by import_sol after each build (19 → 21 solutions green).

1. **OutputColors → InferredOutputColors**: never fused, atomic inferred fact
   (union of train-answer colors, first-appearance order) — renamed to say its
   provenance, moved to green GET, placed after Input/Output in the X-macro.
2. **InputPalette unfused and deleted**: ≡ `Input Colors DropInferredBackground
   SortBySize[1]`. New op DropInferredBackground (drops objects of the elected
   background color; red tab — it's a dropper). All palette machinery removed
   (palette_objs/palette_lists/LAST_IDX_PALETTE/host build). 19 solutions
   rewritten.
3. **08ed6ac7 was the canary, split was innocent**: `detect_output_colors_by_size`
   ranked BEFORE the background election ran, so bg was never excluded and the
   by-size palette began with 0 (rank shift). Fix: the detector call moved to
   right after the election in `infer()`. Ordering comments lie; logs decide.
4. **Dead-op rule**: an op whose backing inferred fact is absent can only no-op
   → verdict computed once in puzzle_loader right after inference
   (`puzzle::dead_ops`), consulted at all three generation offer points.
   Import/GUI unaffected. Any future inferred op adds one verdict line.
5. **OutputPaletteBySize → SizeRankColorMap**: it's the mapping family
   (rank → color, table inferred), joined ShapeColorMap/BidiColorMap in MAP.
   That emptied FUSED GET; the tab is deleted (enum, ui_action, click handler).
6. **Selection generalized to the Excel model** (see RESOLVED above): pixel
   bits, additive selectors, SelectAll + ClearSelection ops, acts strict.
   Bare Recolors in 6 solutions got explicit `SelectAll`.

Op count 71 → 74 (added DropInferredBackground, SelectColor, SelectAll,
ClearSelection; deleted InputPalette). Fan shrank anyway: dead ops leave the
pool per puzzle, and the fused brackets are gone.

## DECIDED 2026-07-23: the generator is typed GP, not a grammar

The long "grammar" agony (`(GET (SELECT* ACT+)* COMBINE)+`, and where do the
`*`/`+` loops go) dissolved once we saw it was never a grammar — it is a **type
discipline**. GET/SELECT/ACT/COMBINE are types (already what the ROLE column of
the op list says / the GUI tabs). A grammar *names* its loops and forces you to choose the
nesting; an automaton lets the loops *emerge* from which typed transitions are
legal. Generation wants emergence, so: one flat op stream, one append point, a
per-op legality condition. `SELECT*`, `ACT+`, phrase-repeat, and the multilayer
`+` all fall out of legal edges — nobody wrote a loop operator.

Kernel facts that ground the rules (verified, not assumed):
- Every COMBINE (Union/Underlay/Intersection/Difference) PASSES THE LIST THROUGH
  unchanged — the object list survives a combine (it is not consumed). So acting
  after a combine on the same list is real and used (9 `Union <act>` cases in the
  solutions), and a new GET after a combine starts a fresh phrase.

### Realization: obligation-carrying generation (form B)

Not "generate then check" (rejected — must generate-correct). The generator
carries DEBT as data and the debt STEERS which op it may emit next. This is the
existing incremental-depth (iterative-deepening) loop plus a debt the next
increment must respect. Two debts, both plain sentences:
- **"a selection is waiting to be acted on"** — a SELECT happened with no ACT
  after it. Paid by the FIRST act (one act clears the whole additive gesture;
  DeleteSelection counts — it is under ACT).
- **"work is not yet landed"** — a GET/SELECT/ACT touched the list since the last
  COMBINE. Paid by a COMBINE.

### The role rules — each op is legal when:

1. **GET** — it is the first op, or the last op was a COMBINE.
2. **SELECT** — at least one GET has happened.
3. **ACT** — at least one GET has happened.
4. **COMBINE** — at least one GET has happened, no selection is waiting to be
   acted on, and this combine is not the same kind as the one before it.
5. **STOP** — no selection is waiting to be acted on, and the last thing done to
   the list has been landed by a COMBINE.

Deliberately permissive: forbid only the proven-bad, leave the rest open, let
search + the examples judge the unseen-but-legal. Intended free space:
ACT-after-ACT (multi-act chains), SELECT-after-SELECT (additive marks, the one
wanted freedom), ACT/SELECT-after-COMBINE (work the passed-through list),
cross-type combine chains, many GETs (multilayer). One smell noted and passed:
SELECT-after-COMBINE is legal but unseen (0 cases) — first to revisit if it ever
generates garbage.

### The pruning rules — algebraic redundancy (never generate a longer spelling)

Separate tier: op-level, not role-level. Never emit a sequence provably equal to
a shorter one (term-rewriting normal-form idea; do NOT build full Knuth-Bendix
now — just bank the obvious wins). Classes:
- **Idempotent `XX=X`** (verified in kernel): Union, Underlay, Intersection,
  Difference, Trim, Fit. Same op twice in a row is wasted.
- **Involution `XX=id`**: FlipVertical, FlipHorizontal, Invert, InvertSelection —
  doubling returns to start.
- **Finite order `Xⁿ=id`**: TurnRight⁴=id (≥4 wasted), TurnRight³=TurnLeft (use
  the shorter spelling); TurnLeft likewise.
- **Inverse pairs `XY=id`**: TurnRight·TurnLeft; check ScaleUp[k]·ScaleDown[k],
  AddColor·RemoveColor.
TODO: sweep the whole op set for these the same way (kernel-proof each, never
guess). This is a CLASS to detect, not a fixed list.

## BUILT 2026-07-24: the generator gate (form B, first cut)

The 5 legality rules are live in the generator. One home each:
- **roles** — `is_get_op` / `is_select_op` / `is_act_op` / `is_combine_op` in
  cuda_dsl.cuh, beside `modifies_canvas`. Since 2026-08-16 these are not lists
  of their own: each reads the ROLE column of NODE_PRIMITIVES via `role_of`, and
  the GUI's tabs read the same column, so there is one statement of an op's role
  and no second copy to drift.
- **debt** — three fields on `node` (`sel_waiting`, `work_unlanded`,
  `last_combine`), set once by `set_generation_debt`, called from `update_hash`
  (the single node-finalize chokepoint — same one-home discipline that fixed the
  `selected` bug, so no manual `new node()` site can drift).
- **rules** — `legal_next(arg, type)` (the 5 role rules) and `legal_stop(tail)`
  in environment.hpp. Gated at all three generation offer points (phase_0 seed,
  phase_1 root-wrap, phase_1 history-wrap).

Verified: build clean, imports 34/36 (the 2 canaries), live search still
converges (0607ce86 from scratch, 0.76s). Not benchmarked — DSL work incomplete,
a rate comparison isn't meaningful yet.

HONEST LIMITATION — it is a GATE, not pure clean generation. The loop walks all
primitives and skips the illegal ones (generate-then-reject), instead of
iterating only the legal set. Behaviorally identical (no illegal program
survives), same speed (O(pairs) either way), so this is the correct first cut.
- The **fixed-base** path (locked root-wrap) already skips cleanly — one base,
  one allowed set, illegal primitives advanced past, never built.
- The **history-wrap** path is ops-outer × chains-inner, so legality is per
  (op, chain) pair — there is no single allowed set to hoist, and gate == clean
  there by nature.
TODO (someday, when this loop is touched anyway): **flip to chain-outer,
allowed-ops-inner** everywhere — compute each chain's debt once, iterate only its
legal continuations. Makes illegal candidates unrepresentable (true clean
generation). Decided NOT worth the surgery now: zero behavioral gain, marginal
speed, and the loop's resumable-index/yield machinery is load-bearing.

## Where we are in the improvement tree (recursion depth)

The stack, outermost first — each level opened to serve the one above and is
still open:

1. **Win the prize** — everything serves this.
2. **Fan-out is the blocker** ([[Inference Master]] §4b): shrink branching
   without losing completeness, then unfuse, then the DSL simplifies.
3. **select→map→compose grammar** (this note): the joint every op is made of;
   fusing two slots costs their product.
4. **Fix the ops so the grammar can exist**: pure GETs, lossless SELECTs,
   explicit ACTs. ← the current level. Sub-stack:
   4a. unfuse fused GETs — DONE 2026-07-22 (tab deleted; the "hard 3" resolved
       as: InputPalette unfused, the two Output* renamed as atomic
       inferred/map ops)
   4b. drain SELECT SHAPED ACT — DONE 2026-08-16, zero droppers left; all 8
       selectors mark, kernel-checked one by one
   4c. teach each act to read the selection bits (only Recolor does) — still
       open, and deliberately demand-driven
5. **(descended into and CLOSED)** `selected` bug — level 4's SelectHoles work
   planted it; 05a7bcf2 forced the descent; fixed at the chokepoint; redundancy
   swept. Deepest point reached this session, now popped.

Next after level 4: **implement the grammar in the generator** (the payoff that
re-fuses at search time — sentence-at-a-time emission, no dangling selector).
Everything unfused so far is FINDABILITY DEBT until that lands: solutions are
deeper now (SelectHoles: 0a2355a6 search no longer converges; the fused GETs
existed to fold depth). Deliberate, user-accepted ordering: "fix ops then
grammar." Do not panic at benchmark dips before the grammar generator exists.

## Order of work

1. **Selectors as predicates** (ARGA's form) — the missing slot, and the only one
   that makes a new property cost a *sum*. A predicate never returns a list, so it
   stays inside one register and one byte.
2. **`HolesToColorMap` gets its detector** — 50 → 1, joins its two siblings.
   Independent of 1; both reduce the same fan.
3. **Neither may kill completeness.** A learned map that finds a conflict, or a
   key unseen in training, must decline and leave the general path reachable.
   Steer, never kill — see [[Background Per-Side Split]] level 2.
