DSL Combinations & Observational Equivalence

The Idea
Running and saving all DSL combinations on disk so that when a certain behavior or "addon" is needed, it can be quickly retrieved without searching.

The Strategy: Observational Equivalence
Blindly generating all DSL combinations (ASTs) of depth 1, 2, 3, etc., would result in an exponential explosion (O(B^D) where B is the number of primitives and D is the depth), quickly consuming all disk space.

However, a highly efficient DSL Lookup Dictionary can be built using Program Synthesis (Bottom-Up Enumerative Synthesis with Observational Equivalence).

Many DSL combinations do the exact same thing (e.g., Rotate 360 does the same as Identity). We only need to save one combination for each unique behavior.

1. Create Canonical Test Inputs: Generate a small set of representative inputs (e.g., a 5x5 grid with a dot, a grid with a hollow square, a grid with noise).
2. Bottom-Up Generation: Write a script to generate ASTs starting from depth 1, then depth 2, etc.
3. Behavior Hashing: Run each AST on the canonical inputs and hash the resulting outputs to create a "Behavior Signature".
4. Pruning & Saving: 
   - If this Behavior Signature has been seen before, throw the new AST away (it's redundant).
   - If the signature is new, save the AST to disk.
5. The Dictionary (Addon Library): We end up with a file on disk mapping Behavior Signature -> Smallest DSL String.

How the Solver Uses It
When the solver is trying to solve an ARC puzzle and needs a specific transformation for a sub-component (like "make this shape move right and become blue"):
- It temporarily applies the goal transformation on the canonical inputs to get a target Behavior Signature.
- It looks up that signature in the on-disk dictionary.
- If it exists, it instantly grabs the pre-compiled DSL block ("addon") and plugs it into the solver. No searching required.

Proposed Implementation Plan
1. analysis/dsl_dictionary_builder.cpp: Create an isolated, standalone C++ program to do this generation.
2. Canonical Grids: Define a hardcoded set of 5-10 small diverse test grids.
3. Serialization: Save the results into an optimized binary format (dsl_addons.bin) or a structured file loading it at startup into a fast std::unordered_map.
4. Integration: Modify the main solver (Beam / Tournament) to optionally query this library before trying to mutate ASTs blindly.

What Are Canonical Grids?
"Canonical" in this context means a standard, fixed set of test grids.

To figure out what a piece of DSL code does, we need to test it. If we test two different DSL combinations on the exact same set of grids and they produce the exact same output every time, we consider them "observationally equivalent" (they do the same thing). 

The "canonical grids" are just those standardized test dummies. 

For example, we might define 5 canonical grids:
1. A flat solid grid (all blue)
2. A single dot (one red dot in the middle)
3. A hollow box (a 3x3 yellow outline)
4. Noise (random colored pixels)
5. A diagonal line

When we generate a new DSL combination like Rotate90(Rotate90(Grid)), we run it on those 5 canonical grids. The 5 output grids are combined into a hash (a "behavior signature"). 
If we later generate Rotate180(Grid), we run it on those same 5 canonical grids. It will produce the exact same 5 output grids, resulting in the same hash. 

Because the hashes match, we instantly know Rotate180(Grid) and Rotate90(Rotate90(Grid)) are identical in behavior, so we only save the shorter one.

Using ARC Training Grids as Canonical Grids
Using the actual ARC training grids (the inputs from the JSON files) as canonical grids is a very intuitive approach because they represent the exact type of data the DSL will be working with. 

However, there is a major performance catch if applied naively.

The Problem: Too Many Grids
There are roughly 400 puzzles in the training set, each with about 3-4 examples. That’s around 1,500 grids.
If we generate an AST (like Rotate90), we would have to run it on all 1,500 grids to create its Behavior Signature.
- If we generate 10 million ASTs, we have to perform 15 billion grid operations just to build the dictionary. It becomes computationally way too slow.

The Solution: A Carefully Chosen Subset
Instead of using all 1,500 ARC training grids, we can hand-pick 5 to 10 ARC training grids that represent different "types" of puzzles.

For example, we extract 6 grids from the ARC dataset:
1. A flat/solid grid (e.g., from a puzzle about recoloring lines)
2. A 3x3 tiny grid (e.g., from a pattern completion puzzle)
3. A 30x30 huge grid with noise (e.g., from a denoising puzzle)
4. A grid with a single colored pixel (e.g., from a pathfinding puzzle)
5. A grid with distinct geometric objects (e.g., squares and crosses)
6. A grid with a hollow box/border

By using just 5 to 10 highly diverse ARC grids instead of 1,500, we get the best of both worlds:
- Speed: Evaluating a new AST takes microseconds instead of milliseconds.
- Accuracy: Because the 5-10 ARC grids are so completely different from each other, almost no two distinct DSL behaviors will accidentally produce the exact same outputs across all of them (avoiding hash collisions).

What Do State-of-the-Art ARC Solvers Use?
The latest top-tier ARC solvers generally take one of two approaches for canonical grids, and the first one is actually the most dominant:

1. Dynamic "On-The-Fly" Canonical Grids (Puzzle-Specific)
Instead of building a massive global dictionary of DSL combinations beforehand, top solvers (like those based on DreamCoder or the Hodel solver) build the dictionary on the fly for the specific puzzle they are solving right now.

- What they use: If a puzzle has 3 training pairs, they use exactly those 3 Train_In grids as the canonical grids for that session.
- Why this is brilliant: It aggressively prunes things that don't matter for this exact puzzle. For example, if the current puzzle only has solid red squares, then InsideColor(Red) and BorderColor(Red) might do the exact same thing on this specific puzzle. The solver will realize they are "observationally equivalent for this puzzle" and instantly throw one of them away.
- The downside: You have to rebuild the subset dictionary for every new puzzle, which adds to the solver's startup time per puzzle.

2. Global "Symmetry-Breaking" Synthetic Grids
If an architecture wants a global, pre-computed dictionary (like our addon library), they usually don't use real ARC examples. Real ARC examples have too much symmetry (e.g., big solid boxes), which causes false-positive hash collisions (thinking two operations are identical globally when they aren't).

Instead, they generate 2 to 4 "Synthetic Symmetry-Breakers":
1. The Coordinate Gradient: A grid where every pixel's color is completely unique or sequential (e.g., Color = (x*width + y) % 10). This entirely breaks all geometric symmetries. If Rotate90 and Identity hash the same here, the math is broken.
2. The "Single Pixel" Grid: A grid of all black with exactly one red dot at a non-center coordinate (e.g., x=1, y=2). This tests translation/movement (MoveUp, MoveLeft).
3. Pure Noise: A grid filled with completely random std::rand() % 10 colors. 

Conclusion
For a global, instant-lookup addon dictionary, Synthetic Symmetry-Breaker Grids are the safest choice. They guarantee that absolutely zero false equivalences slip through, avoiding the accidental deletion of valid DSL combinations.


ARC: It’s like trying to write a 500-line magical spell that brings a pile of sand to life and asks it to solve a riddle. :D