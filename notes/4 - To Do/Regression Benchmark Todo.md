# Regression Benchmark Needed
Date: 2026-03-24

## Problem
DSL changes (like removing node types) can cause non-obvious regressions in solver
convergence speed. These are invisible without testing — the solver still works, just
slower. We spent a full day debugging a side effect of unarization that a 5-minute
benchmark would have caught immediately.

## What We Need
A before/after benchmark that runs automatically after any DSL change:
  - ~20 representative puzzles (mix of easy, medium, hard)
  - Track: solve time, APS, total candidates, final score
  - Compare against a baseline and flag any regressions (e.g. >20% slower)

## Candidate Puzzles to Include
  - 045e512c (caught the beam diversity regression)
  - 09629e4f (caught the APS collapse from beam underfill)
  - Add ~18 more that cover different solution patterns (tiling, projection,
    filtering, multi-step composition, etc.)

## When
Soon. Before the next round of DSL changes.
