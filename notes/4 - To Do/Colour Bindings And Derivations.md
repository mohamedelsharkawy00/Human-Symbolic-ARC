# Colour Bindings And Derivations

Parked 2026-07-27, mid-reasoning. The user is going off to derive more facts;
this records where the chain got to so it can be picked up cold.

## How it started

`introduced` = a colour in every train output and no train input. It can only
arrive by an op that writes colour, so on those puzzles a program without one
cannot solve. Measured: 281 of 1120 puzzles have at least one.

The audit that came out of it (worth keeping regardless) — the eight ops that
can write a colour absent from their input:

    AddColor[c]              literal
    Recolor[c<10]            literal   (c=10/11 resolve from the object)
    InteriorColor[c>=0]      literal   (c<0 resolves from the object)
    SizeRankColorMap         output-derived
    ShapeColorMap            learned table
    ShapeToInteriorColorMap  learned table
    BidiColorMap             learned table
    Output                   the target grid itself

Everything else copies an existing colour or writes transparent. `Invert` was a
ninth until it was deleted — it wrote a hardcoded 1.

## Why the obvious uses died

- **Reject at acceptance**: pointless. `b.solved` is exact match, so a program
  that never introduces the colour is already rejected by the scorer.
- **Prune in the beam**: the predicate is not monotone. A chain lacking the
  paint gains it next level, so pruning cuts chains that were one op short.
- **Restrict the search space to paint-containing programs**: sound in principle
  (every solution has one) but risks beam starvation — today the beam finds the
  STRUCTURE first, scoring well without the paint, and only later gains it.
  Forbid paint-free programs and that discovery route is gone.
- **Append the paint in the closing frame**: only correct where the introduced
  colour IS the canvas. Measured: 45 of 281 (16%). The other 84% land the colour
  on particular cells, and a tail AddColor fills everything transparent.

The surviving safe versions are ranking, and generating the painted variant
ALONGSIDE the paint-free one (one or two extra candidates per program).

## Where the reasoning actually landed

Raw `consumed` / `introduced` say WHAT changed. What an op can act on is a fact
that says WHAT DECIDES it. Two of those already exist, each hardcoded as its own
detector plus its own op:

    shape     -> colour     ShapeColorMap
    size rank -> colour     SizeRankColorMap

The general form is one question asked of every object: **which property of this
object predicts the colour it wears in the output?** One detector family instead
of one detector per key. Candidate keys: shape hash, size rank, pixel count,
hole count, position, neighbour colour, containment.

## The split that matters

- **Lookup binding** — key -> a colour learned from train. Breaks when the test
  uses a colour train never showed.
- **Derivation** — the colour IS another object's colour: the enclosing shape's,
  the nearest marker's, the majority of its neighbours', the one it overlaps.
  Nothing memorised, so an unseen colour costs nothing, and it is more often the
  actual rule. ARC puzzles rarely mean "shape X is always blue"; they mean "each
  shape takes the colour of the thing it sits on".

Same detector shape either way: propose a relation, check it holds on every
train pair, keep it if it does. The difference is only what gets stored — a
table, or nothing.

## Also settled on the way

Colour and topology do NOT factor cleanly. The map ops are the coupling: a
structural property choosing a colour. Three classes, not two — topology,
colour literals, and bindings between them. Every colour literal that can become
a binding stops being a search dimension.

## Next step when resumed

List the candidate relations (enclosure, adjacency, nearest, overlap,
containment) and measure how many puzzles each one explains, before building any
detector.
