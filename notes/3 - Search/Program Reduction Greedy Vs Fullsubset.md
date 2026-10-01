FULL-SUBSET vs GREEDY LEAVE-ONE-OUT REDUCTION — REDUNDANT (evidence-backed)
==========================================================================
Date: 2026-06-18

WHAT
----
reduce_best_tree() (include/solver_executor.hpp) reduces each new best program
by trying to drop ops and keeping the result if it strictly improves the summed
train score (node-count penalty included) without regressing the solved set.

Two versions were implemented:
  1. GREEDY: drop the single best-improving op, repeat until stable. O(N)/round.
  2. FULL-SUBSET: enumerate all 2^(N-1) drop combinations, pick the best.
     Motivation: catch COUPLED removals (op + its inverse, or any subset that
     only helps when removed together) that greedy-single misses.
     Bounded because MAX_SEARCH_DEPTH=15 caps program length (worst 2^14).

CURRENT STATE: reverted to GREEDY. Full-subset removed from code.

EVIDENCE (from logs/ after a training run)
------------------------------------------
1. The reducer DOES fire. "REDUCED" lines appear across many puzzles:
     0520fde7 x10 (10->9 each), 0c786b71 x5 (6->5), 0607ce86 x4,
     00576224/017c7c7b/025d127b/03560426 x1 (5->4, 4->3, 5->4 ...).
2. Reduction sizes (greedy version): 20x dropped 1 op (N->N-1), 3x dropped
   2 ops (N->N-2). So MULTI-OP reductions DO happen -- but greedy achieves them
   by removing one op per round over multiple rounds (sequential singles), NOT
   in one coupled step.
   => Full-subset redundancy still holds: the only thing full-subset can do that
      iterated greedy cannot is a COUPLED pair (A and B each hurt alone, help
      together). Since greedy already produced every 2-op reduction seen, those
      were sequential singles, and no coupled-only case appeared. Full-subset's
      extra 2^N cost bought nothing observable.

NEW CONCERN: the whole reduction feature may be MARGINAL for solve RATE
----------------------------------------------------------------------
All reductions were small (1 node off already-short 4-10 op programs). That
matches the "no observable difference" report: trimming 5->4 tidies the answer
but doesn't change WHICH puzzles solve. So:
  - These puzzles were NOT actually stuck in a no-op tar pit (the original
    motivation). The reducer is cleaning near-minimal solutions, not rescuing
    bloated-prefix searches.
  - reduce_best_tree (greedy OR full) is therefore unproven as a solve-rate
    lever. It is cheap and harmless, but don't assume it helps the score.

STILL NOT MEASURED
------------------
  - A timed A/B: solve rate + wall time WITH vs WITHOUT reduce_best_tree on a
    fixed task set. Only this proves whether the feature helps, hurts, or is
    inert. The "REDUCED fires" evidence shows it RUNS, not that it WINS.

THE COUPLED-REMOVAL CASE greedy CANNOT do (known gap, deliberately parked)
--------------------------------------------------------------------------
Greedy only takes a step if removing ONE op strictly improves. So it misses the
case where removing op A alone is WORSE and op B alone is WORSE, but removing
A AND B together is BETTER (mutually-dependent ops that are net-redundant as a
pair). Only full-subset (or pair-aware search) finds that.

This case is REAL but was NEVER OBSERVED: across ~23 reductions in the logs, zero
coupled-only removals surfaced (and full-subset showed no behaviour difference).
Decision: accept the small-sample evidence, keep cheap greedy, do NOT chase it.
Revisit ONLY if a stuck puzzle is later traced to a missed coupled removal -- at
that point add a detection probe (after greedy, test pairs/triples and LOG any
coupled win, without acting) to prove the case occurs before paying for it.

NEXT IF REVISITED
-----------------
  - A/B reduce_best_tree on/off; if inert for solve rate, consider removing it.
  - Coupled-removal detection probe (see above) before reinstating full-subset.
