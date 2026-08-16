Overview

# How the ARC Beam Solver Works (Plain English)
Date: 2026-03-24

Think of the solver as trying to write a program that transforms input grids into output
grids. It doesn't know what program to write, so it searches by building programs
piece-by-piece and keeping the best ones.


## The Big Picture

The solver has a "beam" — a list of 128 candidate programs. Each level, it tries to
improve every candidate by swapping out parts of it, then picks the best 128 to keep.
It repeats this until one candidate perfectly matches all outputs.


## Step by Step

### 1. Setup (initialize_terminals + rebuild_skeleton_beam)

The solver starts by creating building blocks:

  TERMINALS: The raw ingredients.
    - Input (the puzzle input grid)
    - Target (the expected output — only used during scoring)
    - Seeds, Palette (derived from the puzzle)
    - Constants (-10 to 10, plus color values)

  SKELETON BEAM: Simple 1-step programs to seed the search.
    - One for each primitive: Split(Input), Overlay(Input, Input), Filter[0](Input), etc.
    - For nodes with variants (Shift, Scale, etc.), it creates one per variant value.
    - Also includes Largest[0](Split(Input)) and similar useful starters.

  HISTORY POOL: A memory of every useful sub-expression seen so far.
    Starts with just the terminals.


### 2. The Main Loop (step function)

The solver alternates between two modes:

  EVALUATE: Send candidates to the GPU/scorer to get a score for each one.
            Score = how close this program's output is to the expected output.

  GENERATE: For each beam member, try improving it by swapping sub-expressions.

The main loop runs: generate candidates → score them → pick the best → repeat.

