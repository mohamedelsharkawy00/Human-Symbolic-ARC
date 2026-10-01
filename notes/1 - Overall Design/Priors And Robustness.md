# Priors and Robustness

## General Intelligence vs. Priors

**Question:** "But isn't ARC supposed to be General? Are humans specific?"

- **Answer:** Humans are NOT "Blank Slate General"; we are Specialized
  Priors-Machines (hardwired for Objects, Gravity, Continuity).
- ARC is explicitly a test of Human-Like General Intelligence, which
  requires these priors (Core Knowledge).
- **Conclusion:** Adding specialized nodes (Gravity, Objectness) is not
  "cheating" or making it less general. It is installing the "Human
  Operating System" required to even understand the questions. A solver
  without these priors is a Blank Slate, which mathematically cannot learn
  complex tasks efficiently.

## Handling Missing DSL Operators

**Question:** "What if we miss a DSL operator? The fallback is a cliff (Hard
Failure)."

- **Answer:** This is the Code of Brittleness. If you need Rotate45 and don't
  have it, constructing it from atoms takes Depth 15 (Failure).
- **Goal:** 100% ARC. We cannot accept Depth 15. The search must stay in the 1–5
  range.
- **Solution:**
    1. **Hindsight RL is the safety net:** it "learns" the missing operator by
       caching the Depth 15 tree once found, turning it into a Depth 1 node
       for future use.
    2. **Powerful Primitives:** Instead of narrow operators (MoveRight), use
       powerful math primitives (AffineTransform) that can generate any
       spatial transformation in 1 step. This ensures the "Fallback" is
       still shallow.

## Meta-Clustering & Object Detection

**Question:** "What if hardcoded Object Detection misses a clustering rule...?"

- **Answer:** Meta-Clustering. Instead of only hardcoding "touching=object", we
  allow the solver to search for the Object Definition.
- **Math:** Searching for "What defines an object" (Depth 3) is infinitely
  cheaper than searching for "How to move every pixel" (Depth 15).
- **Strategy:** Use Hardcoded (Color/Touching) for speed (95% coverage). If that
  fails, run Meta-Clustering search to deduce the custom object rule.
- **V3 Implementation:** Strategy is a Program (Meta-Search). Benefit: The Main
  Solver (Physics/Movement) sits on top of this API and doesn't change when
  we upgrade the Clustering Engine.
