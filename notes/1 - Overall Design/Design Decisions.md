Design Decisions & Conversation Log

Hindsight vs. Layering
- Question: Difference between Vanilla & SOTA Hindsight RL?
    - Answer: Vanilla (HER) learns from State Outcomes (results). SOTA
      (Structure-Aware) learns from Program Structures (logic).
    - Decision: We use SOTA to free/reuse logic fragments (ASTs) rather than
      just caching grids.

- Question: Which implementation first: Layering or Hindsight?
    - Answer: Layered Solving (Decomposition). It requires less core-rewriting
      and immediately reduces search breadth.
    - Strategy: Combine both. Layering handles "Breadth" (Context), Hindsight
      handles "Depth" (Complexity).

The "God Solver" Architecture
- Question: "How far are we from Human Solving?" & "Is it Node Count or Depth?"
    - Answer: Pixel-solvers are ~1% human efficiency. The gap is Object
      Perception.
    - Insight: It is technically Node Count/Depth, but the cause is the
      "Translation Tax". Describing "Gravity" in pixel-logic takes 50 nodes; in
      Canvas-logic it takes 2.
    - The "Canvas" method isn't just a different style; it is mathematically
      superior because it aligns the Solver's Search Space with the Puzzle's
      Problem Space (Human Priors).

- Final Agreed Architecture (The "God Solver"):
    1. Pre-Processing: Hardcoded Computer Vision (Connected Components) to give
       pixels "Object Awareness" (Phase 1).
    2. Decomposition: Split Grid into Layers (Color/Object) to solve
       independently.
    3. Core Solver: Imperative "Cursor" Bots (Select -> Move -> Paint/Mutate)
       instead of Functional Expressions.

- Question: "Can't we just use a Better DSL instead of a whole new Canvas
  Architecture?"
    - Answer: YES. You can simulate "Canvas" logic inside the current Tree
      solver using "God Nodes" (e.g., MoveObject(Red, Down)).
    - The Catch: It is computationally expensive if not cached (re-calculating
      object physics for every pixel).
    - Resolution: We can stick to the current engine if we implement
      Caching/Memoization for these heavy high-level nodes. This is the "Cached
      DSL" approach: safer and faster to implement than a rewriting the engine.

