LET/VAR IMPLEMENTATION ANALYSIS

Q&A: THE LET/VAR "EXPLOSION"

1. What exactly is the "noise"?
The "noise" refers to Structural Aliasing—thousands of ways to write the same logic that all look different to the solver but do the exact same thing. This dilutes the beam search (fixed width, e.g., 1024) with variations of the same 3-4 strategies, pushing unique strategies or the actual solution off the bottom of the beam.

Types of noise observed:
- Wrapping Noise: Let(Constant(5), Input) vs just Input.
- Identity Noise: Let(Row, Var) vs just Row.
- Terminal Substitution: Having two ways to get a value (e.g., Constant(2) and Var) effectively halves the useful beam width.

2. Why does it explode the search space?
In a standard DSL, you pick a primitive and fill its slots. With Let, you create a scoped binding.
- Combinatorial Explosion: For any valid expression E, the solver can now generate Let(val=E, body=Var).
- Search Depth Inflation: Let adds structural weight. If Let is treated as 5 nodes, a program using it looks "heavier" and is more likely to be pruned by the node_count penalty, even if it's logically simpler.

3. How do other DSLs handle this "correctly"?
Most systems that successfully use variables use one of three methods:
- E-Graphs (Equality Saturation): Collapse all equivalent expressions into a single "e-class" node, so 100 variations of the same result only take 1 slot in the search.
- Library Induction (DreamCoder): Instead of active search variables, they "compress" frequently used code chunks into new, atomic primitives between search sessions.
- De Bruijn Indices: Use numeric stack-relative indices (e.g., Var(0)) instead of names to make structural equivalence detection easier.

