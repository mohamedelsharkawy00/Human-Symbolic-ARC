# Split as properties, not kinds

Parked 2026-08-03. Raised while reading 017c7c7b's correspondence.
Matching built 2026-08-06, two of its three properties everywhere and the
third where it means anything; the cutting half is still parked.

## The observation

The ways of cutting a picture (`split_kind`, inference.hpp) are hand-picked
points in a small product space, and they don't say so:

| cut                   | same colour? | joined how?          | on a lattice?    |
| --------------------- | ------------ | -------------------- | ---------------- |
| touching, same colour | yes          | touching, corners    | no               |
| touching, any colour  | no           | touching, corners    | no               |
| whole colour          | yes          | anywhere in the grid | no               |
| one cell each         | n/a          | not at all           | no               |
| smallest block        | mixed        | n/a                  | from the picture |
| biggest block         | mixed        | n/a                  | from the picture |
| step block            | mixed        | n/a                  | from the sizes   |

Two properties do the work — **must a piece be one colour**, and **what counts
as joined** (nothing / sides / sides and corners / anywhere) — plus a third that
replaces joining altogether by cutting on a lattice.

## What got built

MATCHING is now properties, not a kind. `match_mode` (inference.hpp) carries
three independent freedoms — **moved**, **turned**, **recoloured** — and the
election runs the full power set of them, eight modes, against every cut. Seven
cuts times eight modes is 56 readings, all ranked, all in the GUI as a table:
cuts down the side, freedoms across the top, each cell saying what that pairing
left unexplained. Clicking a cell selects that reading.

Two things fell out of it immediately:

1. **017c7c7b** elects the biggest repeating block under **recolour only** —
   the blob recoloured where it already sits, which is what its solution does
   (`Split[1] SelectAll Recolor[2]`, with the tiling a separate step). Asking
   for the reading that STAYED PUT is what the old note wanted and could not
   have: matching always allowed a piece to move, so the in-place reading and
   the moved one arrived mixed together and neither could be asked for alone.
2. **00576224** elects the repeating step under **turned + recoloured**, and
   pair 1's middle band finally corresponds. It never did before, because
   matching at a fixed place tried one pose only. The band is the motif
   mirrored; on pair 0 that mirror happened to also be a recolour (four
   distinct colours, so the map was consistent) and slipped through, on pair 1
   colour 6 sits in two places, the map contradicts itself, and the whole band
   went unexplained. Same geometry, two verdicts — which is what a property
   being welded to a kind looks like from the outside.

One election fix came with it: `sit_on_top` edges no longer count as covering
output cells. They are not a find — they say a piece overlaps something — and
counting them let every reading with no freedoms at all claim it had explained
the picture.

## Where MOVED does not apply, and why that is not a hole

Turned and recoloured are asked of every reading. Moved is asked only of the
four shape cuts. On the three block cuts it is not asked, so each column with
moving on comes out an exact copy of the column beside it with moving off —
12 of the 56 readings saying nothing new.

That looks like a gap and is not one. A block cut says the whole picture is one
block over and over; every copy sits somewhere else, because that is what
repeating means. Asking whether a tiling was allowed to move is asking whether
a tiling is allowed to tile, and any answer to it is invented. The 12 twins are
a question that does not apply, not a reading that came out wrong.

What they do cost is twelve full pairing runs and twelve cells in the table
that look like a choice. Left as they are for now. If that cost ever matters,
the fix is to not run those cells and to draw them blank — the table would then
be saying which questions a cut can be asked, which is worth more than the runs
are worth saving.

## FOUND, NOT FIXED: the first match wins over the better one

Explaining the twins above turned this up, and it is a real fault, not a
question that fails to apply.

On 00576224's step row, TURNED on its own finds nothing that AS IS does not —
12 pieces used, 24 cells left over, the same numbers. It should find more. The
middle band of each output IS the input block mirrored left to right, same
colours:

    pair 0   block  7 9 / 4 3     band  9 7 / 3 4
    pair 1   block  8 6 / 6 4     band  6 8 / 4 6

`collect_motif_edges` takes `matches.hits.front()` — the FIRST hit — and only
then asks whether a recolour was allowed. The poses are tried in a fixed order
and a worse one comes first:

- pair 0: pose 0, no turn at all, matches under a colour swap (7 with 9, 4 with
  3). First hit, a recolour. Thrown out when recolour is off, and the exact
  mirror further down the list is never reached.
- pair 1: pose 2, rows flipped, gives 6 4 / 8 6, which matches under a swap
  (4 with 8). First again, a recolour again, shadowing the exact mirror again.

The cost is not only the twins. The ELECTED reading for 00576224 says twelve
exact and six recoloured. Those six are exact mirrors. The winning account of
the puzzle is worse than the evidence supports — it says recoloured where the
truth is turned.

The fix is to pick the best hit rather than the first: take an exact match if
any pose gives one, and fall back to a recoloured one only when recolour is
allowed. Not done yet.

## What is still parked, and why

The CUTS. They are still kinds, not properties.

The one hole this note used to name is closed: **any colour + touching** is
`connected_multicolor`, it cuts mixed-colour pieces on the eight-neighbour walk,
and it stands as its own row of the table. The old note said inference could
not produce that reading at all; it can, so that instruction is retired rather
than carried forward. Whether the two puzzles it was wanted for (05269061,
08ed6ac7) actually elect it is untested — the claim came from reading the DSL,
not from a run.

What the table is missing now is not a named point but two unasked questions:

1. **JOINED BY SIDES ONLY.** Every touching cut counts corners. Nothing offers
   the reading where two blobs meeting at a corner are two things, which is how
   a person reads a diagonal chain of squares.
2. **ANY COLOUR, ANYWHERE.** Whole-colour insists on one colour; drop that and
   it becomes all the ink as one piece, which is the reading a picture with one
   figure in it wants.

Cuts do not want a power set the way matching does. A cut is one way of
chopping the whole picture — you pick one and it applies everywhere — whereas
each piece is independently moved, turned, recoloured, so a person really does
read those as a combination. That asymmetry is why the table has a flat list
down one axis and a power set across the other.

The block cuts stay the odd ones out: their period is read per grid, so when
the transformation changes what repeats, the cut changes with it (017c7c7b's
three pairs came out 2, 4 and 3 rows). They are the only cuts whose definition
leans on the thing being measured.

## Next step when this is picked up

Take one of the two unasked questions above, add it as a row, and see whether
the election survives it on the reference puzzles. Not the whole product. The
two `static_assert`s on the name tables will insist the new row gets named
twice, long and short.
