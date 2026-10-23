Selection

## Why This Matters for the Beam Diversity Fix

When we removed math nodes (DeltaX, Add, Multiply, etc.), two things happened:

  1. best_by_type lost ~8 guaranteed slots (no DeltaX type in the beam anymore).
     Those slots went to the general fill, where popular types like SlamRow grabbed
     them. Result: less diversity, slower convergence on some puzzles.

  2. The replacement candidates (grid operations) are more expensive to evaluate
     than integer math was. Slightly lower throughput.

The two-pass fill fix (step 4b) addresses #1 by ensuring diversity in the first pass
while still filling the beam completely in the second pass. This gave us 4x speedup
on puzzle 09629e4f compared to the old code with math nodes.


## Key Constants (environment.hpp)

  BEAM_WIDTH = 128         How many candidates survive each level.
  MAX_SEARCH_DEPTH         Max tree depth for candidate programs.
  MAX_STAGNANT_LEVELS      Levels without improvement before diversification.
  MAX_HISTORY_POOL         Max remembered sub-expressions.
  MAX_VISITED_NODES        Dedup cache size for candidates.
