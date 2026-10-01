# One block for a pair is chosen, not computed

Parked 2026-08-12, while working out what a GCD is on a grid.

## The setup

The motif cuts make objects out of repeating patches, so they need a patch SIZE.
A pair has two grids, they may repeat on different sizes, and both sides must be
cut on the SAME size — objects of different sizes cannot correspond, so nothing
can say "this 2x2 became that 3x3".

`block_of` picks that size. Today it measures the period on the input, measures
it on the output, and keeps whichever of the two describes its OWN grid more
cheaply.

## Why that is a heuristic

It picks a size that suits one grid and imposes it on the other. Take an input
repeating every 2 and an output repeating every 3:

    input    1 2 1 2 1 2 1 2 1 2 1 2      repeats every 2
    output   1 2 3 1 2 3 1 2 3 1 2 3      repeats every 3

The input describes itself very cheaply at 2, so `block_of` picks 2 — and then
cuts the output at 2, where nothing repeats at all:

    output at 2   1 2 | 3 1 | 2 3 | 1 2 | 3 1 | 2 3     all different

Six objects, no two alike, nothing to correspond with the input's identical
pair. The size that works for both is 6, and the current rule cannot reach it,
because it only ever chooses BETWEEN the two measured periods.

## The exact answer is the LCM

Shifts that leave a picture unchanged form a lattice, and the smallest repeating
tile is that lattice's fundamental cell — the grid's version of a GCD. Shifts
that preserve BOTH pictures are the intersection of the two lattices, which in
one dimension is `lcm(a, b)`. So the shared block is the LCM of the two grids'
periods, not a choice between them.

## How often it bites

Measured over the 1000 training puzzles, counting only pairs where BOTH grids
genuinely repeat — at least two whole copies — and neither period divides the
other:

    10 pairs.

    5207a7b5   input period 3 | output 4 | lcm 12
    e40b9e2f   input period 4 | output 5 | lcm 20
    e9614598   input period 4 | output 5 | lcm 20
    72207abc   input period 2 | output 7 | lcm 14
    4938f0c2   input period 11 | output 15 | lcm 165
    28bf18c6, 310f3251, 834ec97d, db3e9e38, e9ac8c9e

A first count said 229, which was wrong: the filter accepted a period of 13 on a
14-wide grid. That is the fits-twice degeneracy, fixed the same day in
`repeat_periods` — see A Repeat Has To Fit Twice.

## What a fix looks like

Compute both periods, take `lcm` per axis, and use that if it still fits the
grids — an LCM can exceed the picture, in which case the pair has no shared
period and the motif cut should decline rather than invent one. Cheap: two
periods are already measured, the LCM is one line.

Ten pairs is thin, so this is worth doing when one of those ten matters, or
alongside any other work in `block_of`.
