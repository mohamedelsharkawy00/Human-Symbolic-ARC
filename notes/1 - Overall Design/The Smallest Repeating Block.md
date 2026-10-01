# The smallest repeating block, and why there is no biggest to ask for

The home of the argument behind `repeat_periods` (0_core_types.hpp). The code
there is the recipe; this is why it is shaped that way.

Two readers share it, so they can never read a different period off the same
grid: the Pattern op reads the unit it continues with, and inference cuts a
picture by it (`split_kind::repeating_motif`).

## Smallest, and no threshold anywhere in it

A grid is a WINDOW on a pattern that carries on outside it. So a copy that
starts inside the frame and runs off the edge still counts — 017c7c7b shows a
4-row motif in a 6-row window.

Asking for the BIGGEST block instead needs a floor to mean anything: every
offset above the true period passes a trailing comparison too, so the biggest is
always one less than the size. And every floor you could pick either throws that
window away or makes the block depend on how much of the grid you were shown.

The smallest is a property of the pattern itself, so it needs no floor.

## There is no biggest, even in principle

Periods behave like multiples — once a grid repeats every 3 it repeats every 6,
every 9, and so on. So the largest is a greatest common multiple, which does not
exist.

It had an answer only under an added rule, *must divide the grid exactly*, and
what that found was then always a whole number of the smallest — the same
reading, coarser. Removed 2026-08-12, along with the `biggest` flag and the
`repeating_motif_big` cut it fed.

## Background is don't care, one way round only

A colored cell must be the cell the block puts there. A background cell where
the block has a color is forgiven. That way a motif standing on uneven ground
still has a block.

Forgiving it BOTH ways would let a block of pure background explain a whole
grid, by never being compared to anything.

Pass a value no cell can hold (-2) to compare raw, forgiving nothing.

## Fitting twice is the caller's question, not a setting

`must_fit_twice` is not a knob. The two callers are asking different things:

- **"Does this repeat?"** is a claim about the picture and needs two whole
  copies as evidence. Inference asks this. Why, and what it costs, is in
  [A repeat has to fit twice](../4%20-%20To%20Do/A%20Repeat%20Has%20To%20Fit%20Twice.md).
- **"What unit do I continue with?"** is a different question. The output
  already says it continues, so one whole copy and a matching stub names the
  unit. The Pattern op asks this.

017c7c7b pair 1 is 6 tall with a 4-tall motif: real, and invisible to the
stricter question.
