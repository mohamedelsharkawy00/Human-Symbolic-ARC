# The repeat search only sees rectangles

Parked 2026-08-12. Raised while working out what the grid version of a GCD is,
which turned out to be the smallest repeating tile.

## The picture

A shift that leaves a picture unchanged — move it k right and m down and it
looks the same — is the grid's version of a period. Those shifts are closed
under adding and under reversing, so they form a LATTICE, and the smallest
repeating tile is that lattice's fundamental cell. Its area is exactly how many
copies tile the picture. That is the real invariant: it does not depend on how
much of the picture you were shown.

## What we search for

`repeat_periods` (core_types.hpp) looks for a horizontal period and a vertical
period separately, so it only ever finds the AXIS-ALIGNED sublattice
`pw*Z x ph*Z`. Rectangles, nothing else.

## What that costs — and what it does NOT cost

It does not miss periodicity. Any full-rank lattice has finite index, so some
pure horizontal shift and some pure vertical shift are always in it — the
rectangle search will always find SOMETHING on a picture that repeats.

What it misses is MINIMALITY. When the true lattice is skewed, the rectangle it
finds is a whole number of true tiles, not one. Take a picture where each row is
the row above shifted one to the right:

    A B C A B C A B C
    C A B C A B C A B
    B C A B C A B C A
    A B C A B C A B C

The shift (1 right, 1 down) leaves it unchanged, so the true lattice has index
3 — three cells generate the whole picture. The rectangle search finds period 3
across and 3 down and reports a 3x3 tile: NINE cells, three times bigger than
the truth, and it reads as three different-looking rows rather than one shifted
thing.

Brickwork is the everyday case: every course offset by half a brick. Vertically
it takes two courses to line up again, so the rectangle is twice the brick.

## Why it matters here

The tile IS the object under the two motif cuts, so an oversized tile means
oversized objects, and the correspondence then matches nine cells where three
would do. It also inflates describe_cost for that cut — a bigger tile costs more
to say — so the cut can lose an election it should win.

## What a fix looks like

Search shift VECTORS rather than a pair of axis periods: for each candidate
(k, m), does shifting by it leave the picture unchanged, background forgiven the
same one way round it is now. Keep the two shortest independent ones found; the
tile is their fundamental cell. Cost is the same shape as today — a comparison
over the grid per candidate — but over W*H candidates instead of W+H.

Not urgent, and not known to bite any puzzle yet. The way to find out is to look
for pictures where a motif cut wins but its tile is suspiciously large, or where
the same picture reads as several rows that are each other shifted.
