# The Unary Dilemma
Date: 2026-03-15

## The Core Concept
The "Unary Revolution" involves converting binary primitives that accept scalar arguments (like Recolor, Shift, Scale) into a set of unary primitives with baked-in values.

## The Trade-off: Search Speed vs. Mathematical Flexibility

### 1. The Benefit: Speed & Density
- **Binary**: Finding `Scale(Obj, Constant(3))` requires exploring a 2-node depth.
- **Unary**: `ScaleUp_3(Obj)` is found at depth 1. 
- **Pruning**: Baked nodes are unique types, making hashing and pruning significantly more efficient.

### 2. The Cost: The "Variable Size" Problem
- How do we handle puzzles where the scale factor or shift offset changes per example based on other object dimensions?
- **Example**: "Scale object A to match the width of object B."

## The Proposed Solution: Semantic over Arithmetic
Instead of keeping arithmetic primitives (`Add`, `Subtract`, `Multiply`) to let the solver "calculate" values, we shift to high-level **Semantic Primitives**:
- **Maximize**: Stretch/Scale to fill the output grid.
- **FitTo(A, B)**: Resize A to match the bounding box of B.
- **CenterIn(A, B)**: Move A to the center of B.

## Conclusion
ARC is primarily a test of **Geometry and Recognition**, not arithmetic. By replacing "A + B" with "Double" or "Fill", we align the solver's "thought process" with human intuition.

DSL Hardening Notes & Proposed Primitives
Audit Findings (Brittle/Rigid Primitives)
- Shift (Rigid): Only moves by exactly one bounding-box step. Causes the solver to chain multiple shifts to move objects by small pixel amounts, leading to overfitting.
- Filter/Sort (Blind): Currently only support "Color" and "Area". Need to support spatial properties like Width and Height.
- Scale (Integer-only): Limited to doubling/tripling size. Cannot handle non-integer resizing (e.g., 2x2 to 3x3).

Enhanced Filter/Sort Properties
- Width
- Height