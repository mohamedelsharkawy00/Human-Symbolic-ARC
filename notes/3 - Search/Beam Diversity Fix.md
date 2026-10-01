# Beam Diversity Fix for Node Removal Regressions
Date: 2026-03-23

## Problem
Removing node types (e.g. DeltaX, Add, Multiply, etc.) causes 3x slowdowns on some puzzles even though those nodes aren't used in the solution.

## Root Cause
commit_level_champion() reserves one beam slot per node_type via best_by_type. When math nodes existed, they held seats in the beam as harmless filler. Removing them frees those seats, which get grabbed by the general fill step — flooding the beam with extra copies of popular-but-wrong candidates (e.g. 5 SlamRows instead of 1). This kills diversity and starves the actual solution path.

## Fix (v2 — two-pass fill)
Use a two-pass approach in the general fill step of commit_level_champion():
  Pass 1: Fill with a per-type cap of 2 to ensure diversity.
  Pass 2: If beam still has empty slots, fill the rest with best-scoring candidates regardless of type.

This ensures diversity without underfilling the beam. A hard cap alone (v1) caused the beam to
underfill on deeper puzzles, APS collapsed from ~50K to ~2K, and the solver stalled.

## Code Change (solver_executor.hpp, general fill loop in commit_level_champion)
Before:
    for (auto &cand : level_candidates) {
      if (next_beam.size() >= BEAM_WIDTH) break;
      if (included.count(cand.n->hash)) continue;
      next_beam.push_back({environment::copy(cand.n), cand.score, cand.grid_hash});
      included.insert(cand.n->hash);
    }

After:
    unordered_map<node_type, int> type_counts;
    for (auto &cand : level_candidates) {
      if (next_beam.size() >= BEAM_WIDTH) break;
      if (included.count(cand.n->hash)) continue;
      if (type_counts[cand.n->type]++ >= 2) continue;
      next_beam.push_back({environment::copy(cand.n), cand.score, cand.grid_hash});
      included.insert(cand.n->hash);
    }
    if (next_beam.size() < BEAM_WIDTH) {
      for (auto &cand : level_candidates) {
        if (next_beam.size() >= BEAM_WIDTH) break;
        if (included.count(cand.n->hash)) continue;
        next_beam.push_back({environment::copy(cand.n), cand.score, cand.grid_hash});
        included.insert(cand.n->hash);
      }
    }

## Results
- puzzle 045e512c: Modified 1444 faster than both original 1443 and original 1444.
- puzzle 09629e4f: Modified 1444 solved in 2:05 vs 1443's 8:38 (4x faster).
  APS stayed healthy (~50-66K) throughout. No cliff.
