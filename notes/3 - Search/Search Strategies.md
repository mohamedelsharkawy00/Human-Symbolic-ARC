Search Strategy & Symbolic TTT
- Question: "Is Hindsight/Induced DSL just Test-Time Training (TTT)?"
    - Answer: YES. It is Symbolic TTT.
    - Optimization: Instead of updating Neural Weights (Slow, requires
      Gradients), we update the Vocabulary (Fast, Discrete).
    - Advantage: SOTA (Un-DSL) relies on Retrieval (Did I see this code
      before?). Symbolic TTT relies on Adaptation (I just learned this rule, now
      I can use it).
    - The Innovation: Real-Time Symbolic TTT. Doing "DreamCoder-style" learning
      instantly by freezing search trees (Hindsight) instead of offline
      sleep/wake cycles.

Flat vs. Hierarchical DSL
- Question: "Why not just put all 1,000 Clustering strategies into the main DSL
  and skip Meta-Search?"
    - Answer: Combinatorial Explosion (The "Flat vs Deep" Trap).
    - Hierarchical DSL:
        - Phase 1 (See): Check 1,000 Perception rules. Lock the best one (Cost:
          1,000).
        - Phase 2 (Act): Check 100 Action rules for 2 steps (Cost: 10,000).
        - Total: 11,000 Checks.
    - Flat DSL: 1,100 * 1,100 * 1,100 = 1.3 Billion Checks. (TIMEOUT)
    - Insight: Intelligence is not just a bag of functions. It is a Pipeline
      (Perception -> Action). Enforcing this pipeline is the most valuable Human
      Prior you can use.

Coordinate Descent
- Concept: The "Spotlight" Architecture creates 3 distinct branches: Condition,
  Then, Else.
- Coordinate Descent (Round-Robin / Focused Mutation):
    - Lock 2 branches and optimize 1. Cycle through phases:
        1. The Splitter: Mutate only Condition.
        2. The Solver A: Mutate only Then.
        3. The Solver B: Mutate only Else.
    - Pros: Optimizes complex logic linearly (3 * N^2) instead of exponentially
      (N^6).

The "Amnesiac Leaf" Bottleneck (SOLVED)
- Discovery: Leaf nodes (Constants, Row, Col, Neighbors) were explicitly forced
  to 0.0 Freeze during reinforcement.
- The "1-Pixel Off" Trap: The solver found the correct logic "Skeleton" but
  re-randomized its "Muscles" (specific coordinates) every generation.
- Fix: Allow leaves to accumulate Freeze stability.
- Observed Impact: Stalled puzzles collapsed into perfect solutions in <15
  seconds.

Refactoring & Abstraction Strategy
- Goal: Move from 'Memoryless Random Search' to 'Accumulating Knowledge Search'.
- Refactoring (Cleanup): A post-process pass that simplifies the Best Tree
  without changing its output behavior (e.g. Add(X, 0) -> X).
    - Benefit: Prevents 'Bloat' from protecting bad logic. Makes patterns
      visible for the Abstraction phase.
- Abstraction (Hindsight Vocabulary): Identify frequent/stable subtrees and
  promote them to 'Learned Primitives'.
    - Mechanism: If 'BitAnd(InputUp, InputLeft)' appears in 50% of top solutions
      -> Create new terminal 'OP_SYM_01' and inject into mutation pool.
    - Benefit: 'Ratcheting'. The solver solves Depth 3. We save it. The next
      search uses it as Depth 1. Effective depth doubles.

Search Strategy Pivot (Genetic -> Beam)
- Question: 'Why use Random Search + Refactoring? Why not just use Beam Search
  from the start to avoid bloat?'
    - Answer: You are correct. Genetic Algorithms create mess and require
      cleanup. Systematic (Beam) Search refuses to create mess in the first
      place.
- Concept: Beam Search = Search + Refactoring Integrated.
    - If Beam Search generates 'Sub(Add(X, 5), 5)', it evaluates it.
    - It notices 'Output == Output(X)'.
    - It checks its memory: 'Do I already have a program for Output(X)? Yes, X.'
    - 'Is X simpler? Yes.'
    - Action: Discard Complex Version.
- Result: The Search Frontier ONLY contains the simplest way to express any
  given logic. Bloat is mathematically impossible if implemented correctly with
  a Visited Set.
- Decision: Abandon Genetic Algorithm patching. Rewrite Engine to Breadth-First
  Beam Search.
References

- ARC DSLs (Gemini Share): https://gemini.google.com/share/f9a457480c8a
- Exploring the Gestalt Principles of Design:
  https://www.toptal.com/designers/ui/gestalt-principles-of-design
- Gemini - ARC Solver Discussion: https://gemini.google.com/app/e6d285050401d89d
- Gestalt - Google Search:
  https://www.google.com/search?num=10&sca_esv=6183b4817dd2e014&sxsrf=ANbL-n5bppcHj9qOHDCjYJ0wP2-9YU3yuQ:1769189365979&udm=2&fbs=ADc_l-aN0CWEZBOHjofHoaMMDiKpaEWjvZ2Py1XXV8d8KvlI3jljrY5CkLlk8Dq3IvwBz-R5R-93bnJN-gfJetFY0A5MB-rerWVUzYIWeP5MmTxfBURsID9Ypb5UUoCkcK3WmfB3LrjVZ5p8ot95XhaAAFETS9zY8cY1Q8cIjPuAg_KFWvnJXK_ORlBbA82cROYt2uTBwsklprfF9MYEWuB6Gf09_Pi8qQ&q=Gestalt&sa=X&ved=2ahUKEwiEisz9l6KSAxUBK_sDHQQBLZEQtKgLegQIEBAB&biw=1272&bih=588&dpr=1.5&aic=0#sv=CAMSVhoyKhBlLWFxenAxOGxBRF9kRXpNMg5hcXpwMThsQURfZEV6TToOV2gyMXBCZzl4eFF4a00gBCocCgZtb3NhaWMSEGUtYXF6cDE4bEFEX2RFek0YADABGAcgnKOSjg0wAkoKCAEQAhgCIAIoAg
