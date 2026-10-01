# Beam Order Shuffle Hardening
Date: 2026-06-27

## Status 2026-07-14
The predicted bug happened: on 007bbfb7, ranking ties were broken by container
iteration order (a `static thread_local unordered_map`'s bucket count survives
puzzle switches), so a fresh start (50s, 11M attempts) and a switched-to start
(10s, 2.8M) were repeatably different searches. Fixed by `ranks_higher` in
solver_beam.hpp: every ranking site now totally orders by score, then tree
hash — no tie left for order to break. Remaining work here is only the seeded
shuffle stress test below, which now has a deterministic baseline to diff
against.

## Idea
Harden the solver against hidden ordering dependencies by shuffling the beam
order (and/or candidate enumeration order) and confirming every puzzle still
solves in a timely manner.

## Why
The solver's correctness should not depend on the *order* candidates happen to
be enumerated, hashed, or seated in the beam. Right now there are known cases
where order matters (see Beam Diversity Fix, and "Removing node types shifts
enum ordinals → changes hash/search order" in the Contest Hardening Plan). If a
solution path only survives because of an incidental tie-break order, that's
fragile — a future DSL change that perturbs ordering could silently break it.

## Test
1. Add a deterministic shuffle (seeded RNG) to the beam / candidate order at the
   start of each run.
2. Run the regression benchmark across several seeds.
3. Pass criteria:
   - Every puzzle that solved before still solves.
   - No puzzle regresses >20% in solve time (same bar as the Contest Hardening Plan).
   - APS stays above the minimum threshold throughout each run.
4. If a puzzle only solves under one ordering, that's a robustness bug — the
   beam fill / diversity logic isn't doing its job and needs investigating.

## Notes
- Keep the shuffle behind a flag/seed so it's reproducible and can be disabled
  for the known-good contest build.
- This is a stress test, not a shipping feature: the goal is to *find* order
  dependence, then fix the underlying fragility, not to ship a randomized solver.
