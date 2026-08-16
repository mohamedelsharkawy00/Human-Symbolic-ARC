# Step blocks are not known to be copies

Parked 2026-08-12, while marking copies of one motif in the pane.

## What was done

Objects that are copies of one thing are drawn with a cyan box, so a motif seen
five times reads as one thing five times rather than five things that resemble
each other. Two ways an object qualifies:

- it shares a canonical shape hash and cell count with another object on the
  same grid — whole copies, whatever cut made them; or
- the cut is `repeating_motif`, in which case EVERY block is a copy, because
  cutting on a verified period is what makes them so. This is what catches the
  stub at the far edge: the motif cut short, whose shape hash therefore does not
  match its whole brothers.

## What was deliberately not done

`repeating_step` was left out of the second rule. Its block size comes from the
grids' SIZE RATIO — output three times the input, so the block is one input —
and nothing has checked that the picture actually repeats on it. The blocks are
equal rectangles, which is not the same as copies. Marking them all cyan would
state something no one verified.

## Why it needs revisiting

The result is an asymmetry with no principle behind it. Two cuts both carve a
picture into a grid of equal blocks; on one of them every block is called a
copy, on the other none is, and the difference is only whether a period was
checked beforehand.

Both readings are wrong at the edges:

- Under `repeating_step`, blocks that ARE identical get no cyan, though the
  shape-hash rule will catch the whole ones. Only a truncated stub of a genuine
  repeat is missed.
- Under `repeating_motif`, background forgiveness means a block can be called a
  copy on thin evidence — `repeat_periods` forgives a cell of canvas where the
  block has ink, one way round, so a nearly empty block passes.

## The fix worth considering

Ask the question directly instead of inferring it from the cut: a block is a
copy when it MATCHES another block where the two overlap. That is one comparison
per pair of blocks, it handles the truncated stub without a special case, it
works the same for both cuts, and it stops depending on which cut happened to
verify a period first.

The obstacle is where it lives. `is_copy` is currently decided in the loader
from `object_info` alone, which carries a shape hash and a cell count but no
cells. The comparison needs the glyphs, which exist only inside
`detect_correspondence`. So the honest fix computes it there and carries it out
on the object, rather than reconstructing it downstream.
