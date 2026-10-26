# The Pane Family — what it solves, and what is left

Written 2026-09-21, the session that built pane_split, pane_mask, pane_pick,
the Panes op and say_panes. Everything here is measured on the 1000 training
puzzles. Nothing here was searched for: every program is read off the facts
and handed over whole.

    116  inputs where a split is elected
     37  of those where a rule also holds and reproduces the held-out test
     31  of those the reading can say as a program, all of them right

## What this is worth, stated carefully

MAX_SEARCH_DEPTH is 32. 6 of the 31 programs are longer than that, so the
search could not have found those at any budget. The rest are reachable in
principle. 0520fde7 is the only one of the 31 that had a saved solution before
this session -- 4.59M attempts and 17.8s -- and it now costs zero attempts.

WHAT IS NOT MEASURED: whether the search, given time, would find the other 24.
No benchmark has been run against them. Until one is, +31 solved without search
is the honest number and +30 net new is a guess.

## The 31 it solves

    id        cut              shape         ops   past the depth cap
    --------------------------------------------------------------
    0520fde7  wall v 5 2       AND             7
    0c9aba6e  wall h 7 2       NOR            11
    195ba7dc  wall v 2 2       OR              7
    1b2d62fb  wall v 1 2       AND             7
    281123b4  wall v 3 4       PRIORITY       42   yes
    2dee498d  bare v - 3       OR              7
    31d5ba1a  bare h - 2       XOR            13
    3428a4f5  wall h 4 2       XOR            13
    34b99a2b  wall v 4 2       XOR            13
    3d31c5b3  bare h - 4       PRIORITY       42   yes
    506d28a5  wall h 4 2       OR              7
    5d2a5c43  wall v 1 2       OR              7
    6430c8c4  wall h 4 2       NOR            11
    66f2d22f  bare v - 2       NOR            11
    6a11f6da  bare h - 3       PRIORITY       33   yes
    75b8110e  bare both - 2x2  PRIORITY       42   yes
    94f9d214  bare h - 2       NOR            11
    99b1bc43  wall h 4 2       XOR            13
    a68b268e  wall both 1 2x2  AS THEY COME   11
    ce4f8723  wall h 4 2       OR              7
    cf98881b  wall v 2 3       AS THEY COME   11
    d19f7514  bare h - 2       OR              7
    dae9d2b5  bare v - 2       OR              7
    e133d23d  wall v 4 2       OR              7
    e345f17b  bare v - 2       NOR            11
    e3497940  wall v 5 2       AS THEY COME   14
    e98196ab  wall h 5 2       AS THEY COME   11
    e99362f0  wall both 4 2x2  PRIORITY       42   yes
    ea9794b1  bare both - 2x2  PRIORITY       42   yes
    f2829549  wall v 1 2       NOR            11
    fafffa47  bare h - 2       NOR            11

By shape: AND 2, AS THEY COME 4, NOR 7, OR 8, PRIORITY 6, XOR 4

## The 6 it reads but cannot say

The rule holds and reproduces the test answer. The language has no word for it.

    0b148d64  blank both 0 2x2 pick: the odd palette
    5bd6f4ac  bare both - 3x3  pick: the one in that place (place 2)
    662c240a  bare h - 3       pick: the one that breaks the fold (fold 1)
    88a62173  blank both 0 2x2 mask: a table that is neither a shape nor an order
    be03b35f  blank both 0 2x2 mask: one pane decides, across a lattice, read through a pose
    d47aa2ff  wall v 5 2       mask: a different colour each side, and a copy where they meet

Two want a selector nothing offers: the pane whose palette is unlike the others,
the pane that folds unlike the others. One wants a place in a lattice, where a
band names a row or a column but two bands name a cross and not a pane. Three
are mask tables that are neither a boolean shape nor an order -- a different
colour per side, or a copy mixed in with paints.

## The 79 with a cut and no rule

These are the headroom, and they are a bigger pile than everything above. The
split is elected and believed; nothing yet says what the panes come to. Worth
reading before any new op is proposed -- three times this session an op looked
necessary and the words were already there.

    09629e4f  0962bcdd  103eff5b  15113be4  1acc24af  1da012fc
    1e5d6875  20fb2937  234bbc79  25c199f5  2685904e  272f95fa
    278e5215  29623171  2ccd9fef  337b420f  351d6448  363442ee
    36d67576  3906de3d  3bdb4ada  40f6cd08  412b6263  4c177718
    4e45f183  516b51b7  5614dbcf  5833af48  5adee1b2  5ecac7f7
    60b61512  639f5a19  6773b310  68b67ca3  692cd3b6  694f12f3
    6a1e5592  6d0160f0  72ca375d  73182012  759f3fd3  770cc55f
    7c008303  84ba50d3  8597cfd7  85b81ff1  8d510a79  8e5a5113
    8efcae92  93b4f4b3  94be5b80  9772c176  995c5fa3  9a4bb226
    a416fc5b  a61ba2ce  a64e4611  ae4f1146  af24b4cc  b0f4d537
    b20f7c8b  b230c067  b4a43f3b  b7999b51  bbb1b8b6  c803e39c
    c9680e90  cbded52d  d4c90558  d94c3b52  dc2aa30b  dc46ea44
    e3fe1151  e4888269  e509e548  e84fef15  ed74f2f2  fe45cba4
    fea12743

## UNEXPLAINED: a band names a pane in a row and not in a lattice

Left open on purpose rather than buried. After Panes, on a 3x3 lattice:

    Input Panes SelectColumn[2] InvertSelection DeleteSelection Origin Union
        -> draws nothing at all

    Input Panes SelectRow[0] DeleteSelection Origin Union
        -> draws nothing at all

The same two selectors work on a ROW of panes: 2dee498d is solved by exactly
    Input Panes SelectColumn[0] InvertSelection DeleteSelection Origin Union

So SelectColumn[2] appears to mark no object on a lattice, and DeleteSelection
on a marked band appears to take everything rather than the band. One of those
two readings is wrong and neither was chased down. gpu_layout_bands is the
grouping both selectors go through and is where to start.

It cost 5bd6f4ac, whose whole rule is "the pane in place 2", and it is why
say_pane_pick declines a lattice and says "a place in a lattice of panes" is a
word it does not have. That message is a guess at the cause, not a finding --
if the behaviour above is a bug rather than a rule, the word exists already and
the decline is wrong.

An unexplained result is usually hiding something, and this one is sitting
underneath a selector every other reading in the file relies on.

## What to do next

0. Find out what the band selectors do on a lattice (above). It is the only
   thing on this page nobody understands.
1. Benchmark the search against the 24 reachable ones, so the number above
   stops being a guess.
2. Read the 79. The next tasks are in there, not in new op ideas.
3. Run the eval set. Everything here is measured on training only, so the
   split's false-positive rate outside it is unknown.

## Three things that cost a wrong answer

Kept because each was invisible until it was traced.

ORIGIN. Panes are cut where they lie. Without it they never stack and a union
hands back one pane, which looks like a rule that nearly works.

A GET OPENING EVERY SENTENCE. A sentence starting with Colors carves what the
sentence before it left instead of the input. It parses, runs, and draws
something plausible. 0 of 6 pairs became 3 of 6 when each got its own source.
sentence_with_no_source refuses such a program now.

THE GROUND FOUND, NOT ASSUMED. A table saying "8 where no pane has anything, 0
everywhere else" reads equally as NOR painting 8 and as OR painting 0. Only the
first is a program: painting the ground is not the same as leaving it. Taking
the empty pattern for the ground cost every NOR puzzle its sentence.

## What the reading says outside the panes

Added 2026-09-22, measured the same way and with the same tool: every training
puzzle read, the programs it spells run on the train pairs, and where one fits
them all, asked for the held-out test answer. Nothing searched for.

     976  puzzles the reading speaks for at all
      77  programs that fit every train pair
      73  of those right on the held-out test
       4  wrong, none of them panes

The panes are tallied apart and still read 31 fit, 31 right, 0 wrong.

### The 4 that fit train and miss test

Worse than a near miss. try_what_inference_said stops the search the moment a
suggestion fits every train pair, so one of these does not come second to
something better -- it ships.

    id        what the reading says
    ------------------------------------------------------------
    00d62c1b  Split[0] Trim BidiColorMap[3] ShapeToInteriorColorMap
    44d8ac46  Split[0] Trim BidiColorMap[1] ShapeToInteriorColorMap
    7b6016b9  Split[0] Trim ShapeToInteriorColorMap
    a5313dff  Split[0] Trim BidiColorMap[0] ShapeToInteriorColorMap

All four are the interior fill, and three of them also say a colour swap. The
table ShapeToInteriorColorMap carries is READ OFF the train pairs, so it fits
them by construction and fitting says nothing at all about it -- a test grid
holding a shape the table never saw has no honest answer to give. That is where
to look first, and it is a reading of the four, not a finding: nobody has yet
put them side by side cell for cell.

The three swaps are not the ones a swap gate reaches. A swap is only said now
where the elected reading allowed a colour to change, and all three of these
are elected [Color], so the gate has no opinion about them. It was 05f2a901
that the gate was for, where [Move] matched every piece exactly and 0<->2 held
only because a shape that moves away leaves ground where it stood.

### The pane pick, respelled

The two programs quoted under UNEXPLAINED above are spelled as they were on
2026-09-21. A paint lands only what is marked now, and DeleteSelection hands
back the survivors with nothing marked at all, so both of them need the hand
said again after the delete -- Origin is an act, and an act on nothing picked
leaves the pane where it was cut:

    Input Panes SelectColumn[0] InvertSelection DeleteSelection SelectAll Origin Union

That is what say_pane_pick emits today and what solves 2dee498d. The lattice
question underneath is NOT answered by it: both spellings drew nothing back
when a paint still ignored the mark, so the cause there is still whatever it
was. It has not been re-asked since.

## The table said only as its operators draw it, 2026-10-26

AGENTS 2.5 again: a rule is replayed with the operators' own arithmetic
before it is said. The mask was replayed by its table alone, and say_panes
then picked the shape by checking the table against itself. Now the mask
reads the shape off the table and keeps it only where the shape's operators
redraw every output, drawn as each op draws (pane_mask_rule::read_how_said,
draw_said); IntersectionList's fold is one helper for the op and the replay
(intersected_cell); say_panes only spells what the mask kept. The priority
sentences carry the pose's turn as the others do (no puzzle needs it yet).

d47aa2ff, among the 6 above, had since been said wrong: "as they come"
compared a one-pane pattern's paint with itself, so "paint 2" read as "copy
the pane", and the program kept the panes' own colors. Replayed, that fails;
its readout says NO OPERATOR FOR what the panes come to again. Training
otherwise unchanged: the 32 pane-mask puzzles solved, programs identical.
