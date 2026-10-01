# A piece is credited with cells it never carried

Parked 2026-08-26, while reading 0692e18c.

## The leak

`unexplained`, in `detect_correspondence`, marks a piece's WHOLE footprint
explained the moment anything speaks for it — it went somewhere, or something
arrived that named it. One edge, and every cell of that piece is accounted for.

That is fine when a piece is small. It is a hole when a cut makes one piece out
of half the grid: `EVERY CELL OF ONE COLOR, WHEREVER IT EXISTS` turns eighteen
scattered cells into a single object, and one meaningless edge launders all
eighteen.

## What it costs

0692e18c pair 0. Input is a 3x3 anti-diagonal of 6s, output is 9x9: the input
stamped at the three places the input itself is coloured, and each stamp is the
input INVERTED — its six empty cells, filled.

The elected reading is 41, `EVERY CELL OF ONE COLOR` with `[MOVE, HANG OFF]`,
117% of plain, **0 left over**. Its nine edges on pair 0 are all this:

    3x3 at 0,0 -> 3x3 at 6,-2, 2 off grid
    3x3 at 0,0 -> 3x3 at 7,0,  1 off grid
    9x9 at 0,0 -> 9x9 at -2,-5, 16 off grid

Glyphs parked mostly off the edge, showing one or two cells. They physically
carry two cells and are credited with seventy.

`MADE OF ONE CELL` was the same disease by a different route — every cell finds
a cell, nothing is ever left over — and it is deleted (2026-08-26). Hanging off
the edge is the other door, and the room behind both is this rule.

## What is actually true about that puzzle

The right cut is already there and is the CHEAPEST reading in the whole table:
reading 80, `A MOTIF THE SIZE OF THE SMALLER GRID`, **95% of plain** — the only
reading under 100%. Its pieces are exactly right:

    pair 0 in    3x3 at 0,0  px 3
    pair 0 out   3x3 at 6,0  px 6
    pair 0 out   3x3 at 3,3  px 6
    pair 0 out   3x3 at 0,6  px 6

One input block, three output blocks, sitting exactly where the input's three
cells say they should. It draws nothing, and for a true reason: the input's lit
cells are (0,2),(1,1),(2,0) and each block's lit cells are the other six.
**Zero cells in common.** Not a weak match — the strongest possible mismatch.

So it leaves everything over, prices at 0.95 + 70/70 = 1.95, and loses to the
junk reading's 1.17 + 0.

## Two facts at two resolutions

Read the 9x9 output at one value per 3x3 block — "is this block occupied?" — and
you get the input back, exactly:

    ..X        ..6
    .X.   =    .6.
    X..        6..

Nothing added, nothing lost, no negation. **The output at block resolution IS
the input.** That is the correspondence. What fills each block — the input with
filled and empty swapped — is the op. Placement here, content there.

## Three ways out, none of them plumbing

1. **NEGATE as a freedom.** One new column; pair 0 becomes three edges, the
   input block copied to three places, inverted. Says the whole puzzle in one
   sentence and reuses everything. Against it: every existing freedom preserves
   the cell count, and the complement of a small shape is a BIG shape that fits
   loosely in many places — a new way in for exactly the junk the cross-pair
   sieve was built to stop.

2. **Correspondence by place.** Every edge in the file today is justified by
   cells agreeing somewhere: `find_in` compares cells, the shape-hash source
   compares cells up to pose, `sit_on_top` needs overlapping ground,
   `collect_motif_edges` calls `find_in`. There is no such thing as "these
   correspond because of WHERE they are" — which is what a person uses here.
   We half-believe it already: `down 0 across 0` is correspondence by place and
   is exempt from every check.
   Against it: a place-claim can never be wrong, and a claim that cannot fail is
   not evidence — the same rule that `landed_on_something` exists for. It would
   hand every same-size puzzle a free block-to-block reading at 0 left over.
   **It could only be safe if a place-line claims NO cells:** it says where, it
   draws, and every cell it does not actually carry stays left over for the op.
   That distinction does not exist in the file, which is also why reading 41
   wins.

3. **Cut the two grids in the ratio their sizes state.** The size rule already
   says `Scale`, 3/1 both ways, and `repeating_step_of` computes the same 3x3 —
   then `detect_correspondence` cuts BOTH grids on that one block, so the input
   is a single piece and the output is nine. Cut the input at 1x1 and the output
   at 3x3 and both come apart into nine, and input cell (0,2) and output block
   (0,2) become the SAME PLACE.
   On its own this buys little: `collect_motif_edges` would compare a 1x1 cell
   against a 3x3 block by testing the block's top-left cell only, and draw three
   lines that are right by coincidence. It is the enabler for (2), not a fix.

Correspondence never sees the size rule at all — `pick_split(train, background)`
takes the pairs and the background, and the rule is computed in the same
`infer()` and never passed down.

## Why the leak has to go first

Even with (1) or (2), reading 80 has to BEAT reading 41, and 41 scores a perfect
zero for free. Close the leak and the junk edges buy what they actually carry —
two cells of seventy — and 41 loses on price to 80 without anything else
changing.

## Corrections, 2026-08-28

**The reading numbers in this note are stale.** `MADE OF ONE CELL` was deleted
(commit 1fea52f), so there are five cuts and eighty readings, not six and
ninety-six. Every index above a cut boundary shifted down by sixteen:

    reading 80  ->  reading 64     A MOTIF THE SIZE OF THE SMALLER GRID [AS IS]
    reading 41  ->  reading 41     unchanged, below the boundary

And the pane now counts from zero, so a number read off the screen is the same
number the log prints. It used to be one higher.

**The leak itself is closed** as of the uncommitted work of 2026-08-28 — see
`Where The Elections Stand.md`. A piece is no longer credited with cells no edge
touched. 0692e18c now elects reading 9 with 55 left over instead of a reading
claiming a perfect zero, so the free zero this note is about is gone.

**What it did NOT fix:** the right reading still draws nothing, because the
output blocks are the input's exact complement and no freedom says so. The three
ways out at the top of this note all still stand.
