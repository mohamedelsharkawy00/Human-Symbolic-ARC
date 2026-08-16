# VoteDenoise Needs The Background — FIXED 2026-07-26

Found 2026-07-26, while normalizing the solution corpus to the inference skeleton.

## What happened

Every exported solution was rewritten to start with
`SelectInferredBackgroundColor DeleteSelection` after its segmentation op.
Fourteen took it. One did not: **0607ce86**, whose whole program is

    Input VoteDenoise Underlay

Inserting the removal makes it fail, with or without a flatten (`UnionList`) and
with or without a background restore at the tail.

## Why

Not because the op mishandles transparency — it already skips transparent cells
both in period detection and in the vote.

The cause is the margin accounting that CHOOSES the period
(`gpu_exec_vote_denoise`, cuda_solver.cu ~2054):

    if (a == bg && b == bg) continue;        // both background - ignore
    if (a != bg && b != bg) { ++both; ... }  // both structure - agree or disagree
    else ++dis;                              // structure vs background - DISAGREE

That last line is the whole discriminator, and the op's own header says why: a
true period only loses margin to noise, a false one bleeds it everywhere.

Delete the background and those cells are transparent, which the loop SKIPS
instead of counting as disagreement. False periods stop bleeding margin, the 70%
gate lets them through, and the wrong period wins.

So the background is not scenery here. It is the negative evidence the period
search is built on. This is also why restoring it at the tail does not help —
the damage happens during period selection, long before the tail.

## The fix

One line was not enough. Patching only the margin test still failed, because the
in-pattern line test and the majority vote read the mask too, and both treated
transparent as a value of its own — the template counted it in `counts[10]`, and
the vote counted it not at all.

What worked is normalising at every read, nine in total:

    // A cell emptied by a background removal upstream is still the ground:
    // read it as bg, so the op behaves the same whether the background was
    // deleted or left in place.
    auto ground_at = [&](int r, int c) -> int8_t {
        int8_t v = s.mask[r][c];
        return v == GPU_TRANSPARENT ? bg : v;
    };

The lesson generalises past this op: a fix that treats one reader of a value is
wrong when several readers share the assumption. Normalise where the value is
READ, not where the symptom shows.

Check the same question for any other op whose decision depends on the ground
being PRESENT as evidence rather than as a canvas.

## Current state

0607ce86 now carries the skeleton form and is verified solving:

    Input Colors SelectInferredBackgroundColor DeleteSelection UnionList VoteDenoise Underlay

The whole corpus was re-run after the op change: 33 solutions, all passing.
Two files were deleted along the way because they never imported as solved,
before and independently of any change made here — 0b17323b and 1190bc91.
