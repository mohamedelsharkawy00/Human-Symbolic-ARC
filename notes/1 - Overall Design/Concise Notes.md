# Concise Notes

## Algorithmic Optimizations

- **Bidirectional Search:** Search forward from input and backward from target output to meet in the middle (or run as two distinct complete passes), dramatically reducing search space.
- **Type Refinement:** Use abstract properties (e.g., shape, size, color constraints) to analytically prune invalid operations without evaluating them.
- **Problem Decomposition:** Split tasks into finding *what* (content/color) and *where* (layout/spatial), running two shallow searches instead of one deep search.
- **Library Learning:** Mine common subtrees from solved tasks and promote them as reusable compound primitives to speed up future solutions.
- **Color-Biased Constants:** Prioritize new colors (found in example 1 output but not input) during operations like Recolor to speed up discovery.
- **Structural Diffing:** Initialize the search by computing the explicit delta/difference (structural, pixel, object, or property changes) between input and target output grids.
- **Example Ordering:** Always start with the example that has the most information (or total pixels).

## Engineering Goals

- **Codebase:** Standard, modern, DRY, native, general, readable, and maintainable.
- **Performance:** Low latency, deterministic, no jitter, hardware-clock locked.
- **Precision:** Maintain absolute precision; consistently capture all detail and explicitly instruct AI agents to do the same.

## Open Questions

- **Hardware Acceleration:** Should we move the engine to GPU / DirectX?
- **Dependencies:** Should we introduce new libraries to shrink the codebase and boost performance?

## Backup Ideas / Alternatives

- "How Shark Solves Arc".
- Improvment of current DSL approach.
- Refinement of existing stuff.

## Others

- **Shark:** btw also: arc and chess have alot/some stuff in common, the grid nature of each imposes such
- **Dorr:** then see how chess engines work?
- **Shark:** All hail the lord Dorr

> When the sky falls,
> All hell breaks loose,
> And everything goes up in flames:
> - Fix DSL first, then the solver
