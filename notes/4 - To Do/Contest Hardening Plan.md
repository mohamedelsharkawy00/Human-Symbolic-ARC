# Contest Hardening Plan
Written: 2026-03-24, refreshed 2026-08-15
Final submission deadline: 2026-11-02 (~2.5 months left)

## Priority Order
1. Reliability first, coverage second.
   Coverage without reliability is building on sand.

## Immediate (this week)
- Build the regression benchmark (~20 puzzles, before/after on every DSL change)
- Write down solver invariants:
    - Beam must always fill to BEAM_WIDTH (128)
    - Every node_type gets at least one reserved slot (best_by_type)
    - APS must never collapse (flag if drops below threshold)
    - No candidate should be generated for disabled/dead node types
    - History pool must not exceed MAX_HISTORY_POOL
- Paper review commit_level_champion() — understand every line, no surprises

## Before any DSL change
- Run the benchmark before AND after
- Check: any puzzle regress >20% in solve time?
- Check: APS stays healthy throughout the run?
- Check: beam is filling properly?

## Known failure modes (found so far)
1. Removing node types shifts enum ordinals → changes hash/search order
2. Removing node types kills beam diversity → popular types hog freed slots
3. Hard type caps can underfill the beam → APS collapses
4. Cheap-to-evaluate filler nodes (math ops) being replaced by expensive grid
   ops reduces throughput even at same beam size

## Contest day checklist
- Solver must not crash
- APS must stay above minimum threshold
- No silent regressions — every solved puzzle must still solve
- Have a known-good fallback build tested on full puzzle set
- Our eval set is NOT Kaggle's on six tasks, and the repo ships no puzzles at
  all — `scripts/fetch_data.py` rebuilds them. Read the data section in
  `5 - Competition/Contest Rules.md` before uploading.
