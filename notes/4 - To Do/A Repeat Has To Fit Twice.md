# A repeat has to fit twice, and what that costs

Changed 2026-08-12 in `repeat_periods` (core_types.hpp). Raised by eye on
017c7c7b: the motif is plainly the full 3-wide row by 2 tall, and the code was
returning 2x2.

## The bug

`row_holds(p)` compares every cell from column p onward against the cell p to
its left. Past halfway there is almost nothing left of the row to compare, so a
long period passes on a handful of cells and means nothing.

A 3-wide row of `010` "repeats every 2" on the strength of column 2 alone —
one comparison, 1.5 copies. That false period then becomes the motif and drives
the whole reading.

This is the SAME degeneracy that made a biggest-period question meaningless (see
Inference Master 3e and the comment at `repeat_periods`). It was only ever
attributed to `biggest`; it bites `smallest` too, wherever the true period is
more than half the size. It also silently wrecked a survey of the training set
during the same session — 229 puzzles looked like they had conflicting periods,
and after the fix ten did.

## The rule now

A period must fit at least twice: `p * 2 <= size`. If none does, the picture does
not repeat on that axis and the period is the whole size. Two whole copies is
the least you can see and still call something a repeat, so the rule is not a
threshold anyone picked — it is what the word means.

## What it costs — the off-the-edge case

A real repeat can show fewer than two whole copies. Five wide with period three,
`A B C A B`, is genuinely 3-periodic: one whole copy and a stub that matches.
The new rule rejects it and reports no repeat.

That collides with `may_hang_off`, which exists precisely so a copy running off
the edge still counts as an object. The two now disagree about a partial copy:
the cut is allowed to keep one, but the period search will not propose a period
that only ever produces one.

So the trade is false positives for false negatives:

- WITHOUT the rule, a false period becomes the motif and drives a whole reading
  that is wrong from the first step. 017c7c7b.
- WITH it, a thin-but-real repeat is missed, the motif cut declines, and some
  other cut wins the election instead.

Declining is the safer failure. A cut that abstains loses an election; a cut
that lies wins one it should not.

## The sharper rule, if this bites

What makes a period evidence is HOW MANY CELLS IT PREDICTED CORRECTLY, not how
many whole copies fit. `p = 2` on a 3-wide grid predicts one column; `p = 3` on
a 5-wide grid predicts two. The second is better evidence despite fitting fewer
times.

Counting predicted cells would keep `A B C A B` and still reject `010`. It needs
a floor on how many cells count as enough, which two-whole-copies does not — so
it is the better rule only once something actually needs it.

## How to tell whether it bites

The motif cuts declining shows up as `cut into 0` for those rows in
`logs/inference/<id>.txt`. Look for puzzles where a motif row reads 0 pieces but
the picture visibly repeats with a fat stub at one edge.
