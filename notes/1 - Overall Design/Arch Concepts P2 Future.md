Future Architecture Concepts
- TensorGP (GPU/TPU Accelerated Genetic Programming)
    - Goal: Increase APS from 1.5M to 500M+.
    - Concept:
        - Flatten Trees into linear instruction arrays (Bytecode).
        - Batch execute 1M+ trees on GPU using CUDA or OpenCL kernels.
        - Treat execution as "Virtual Machine" steps on massive parallel data.
        - Eliminates branching overhead by using mask-based execution.

- Neural-Symbolic Solver (Differentiable Programming)
    - Goal: Replace "Random Search" with "Gradient Descent".
    - Concept:
        - Represent program instructions as continuous probability distributions
          ("Soft" Logic).
        - Execute on TPU/GPU as Matrix Operations.
        - Use Backpropagation to "Train" the program to fit the input/output
          pairs.
        - Eliminates combinatorial explosion by following the gradient.

- Decomposition by Color (Layering)
    - Note: Is there any benefit to splitting the problem: outputting each color
      on its own, then layering them?
    - Goal: Simplify complex "image construction" puzzles by breaking them into
      10 independent boolean problems.
    - Concept:
        - Instead of finding one formula for the whole image, find 10 separate
          trees, one for each color (0-9).
        - Each tree solves a binary classification problem: "Should pixel (x,y)
          be Color N?".
        - Composition: Recombine the layers using a "Painter's Algorithm" (or
          priority order).
    - Pros: Reduces search space drastically for multi-object scenes;
      trivializes simple "draw a box and a line" tasks.
    - Cons: Breaks "Color Generalization" (harder to solve "swap red and blue"
      or "color + 1"); increases runtime by 10x if naive.

- Iterative Repair (Gradient Boosting / Residual Learning)
    - Goal: Solve the "General" problem while keeping the benefits of
      decomposition.
    - Concept:
        - Pass 1: Find the best "Base Tree" that explains most of the image (or
          the background). Lock it.
        - Pass 2: Calculate the Error Grid (pixels where Base Tree != Expected
          Output).
        - Optimization: Search for a new tree specifically targeting the Error
          Pixels (e.g., "If condition, draw Red, else Transparent").
        - Overlay: Composite the new tree on top of the Base Tree. Repeat until
          error is 0.
    - Pros: Preserves generality (can solve "Color + 1" in one pass); handles
      complex composition naturally; efficient use of compute.
