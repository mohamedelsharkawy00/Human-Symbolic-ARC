Gemini Optimizations (Canvas/Imperative)
- Why Canvas/Imperative is critical (The "Translation Tax"):
    - Pixel/Functional approach forces re-simulation of physics inside every
      pixel's formula (e.g. gravity = checking all pixels above).
    - This explodes Node Count/Depth not because the logic is "hard", but
      because the language is wrong.
    - Canvas mode aligns with the Problem Class (Human Priors: Agents, Objects,
      Physics), drastically reducing Tree Depth.
    - Result: A "Move Red Right" task drops from ~15 specialized pixel-logic
      nodes to ~2 command nodes.

- Transition to Imperative / Mutation-based architecture:
    - Move from "expression-per-pixel" (out[y][x] = f(in...)) to a sequence of
      instructions that mutate the grid state.
    - Concept: Treat the grid as a world where we apply rules like "If pixel is
      blue, move it 2 squares right" or "Fill enclosed areas with red".
    - Goal: Better handling of objects, gravity, and spatial mechanics that are
      hard to describe with pure functional math.
    - Benefit: Solves the "fallback/return 0" problem; rules only apply when
      their conditions are met, otherwise the grid remains unchanged.

- Example of Final Solution Form (Imperative/Canvas):
    - Instead of a Tree: Function(x,y) -> Color
    - It is a List of Rules: [Rule1, Rule2, Rule3...]
    - Example Solution for "Move Blue Down":
        1. Select(Color::Blue) -> Returns Mask of blue pixels
        2. Move(Selection, Direction::Down) -> Mutates grid
        3. Loop(Until::Collision) -> Repeats step 2
    - Data Structure: linear vector of Structs { OpCode, Argument1, Argument2 }
      (Bytecode) instead of pointer-based Tree.

Efficiency Levels (The Evolution of Solvers)
- Level 1: Pure Random (Blind)
    - Method: Roll dice, check result.
    - Pros: Easy to code.
    - Cons: 99.9% waste testing garbage (e.g. InputUp 500 times).
    - Status: Deprecated.

- Level 2: Random + Hashing (Baseline)
    - Method: Roll dice, check hash. If duplicate, discard without testing.
    - Pros: Fast, no memory overhead, highly parallel.
    - Cons: High "Generation" count (CPU waste creating duplications), low
      "Evaluation" count.
    - Status: CURRENT IMPLEMENTATION.

- Level 3: Grammar-Guided (Constraint)
    - Method: Restrict dice rolls. Never generate semantic nonsense (e.g.
      Rotate(Color)).
    - Pros: "Generated" count overlaps closely with "Evaluated". Much higher
      efficiency.
    - Cons: Requires complex grammar rules maintenance.
    - Status: Planned Upgrade.

- Level 4: Neural/Probabilistic (SOTA)
    - Method: Use AI to predict the best dice roll.
    - Pros: Near-zero waste.
    - Cons: Massive compute per step (inference).
    - Status: Future Goal.

