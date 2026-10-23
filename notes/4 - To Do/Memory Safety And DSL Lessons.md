Trauma 1: Heap Corruption


# Memory Safety and Debugging Lessons

## 1. Prevent Heap Corruption (Buffer Overruns)
- **Problem:** Writing past fixed boundaries (e.g., `data[30][30]`) corrupts adjacent heap memory, like reference counters or pointers.
- **Solution:** Always clamp calculated dimensions and indices. Never assume a calculation stays within bounds.
  ```cpp
  width = min(calculated_width, MAX_GRID_DIM);
  ```

## 2. Unify Memory Management Patterns
- **Problem:** Mixing direct `delete` with custom reference counting (`remove_grid()`) causes double-frees or use-after-free crashes.
- **Solution:** Use **RAII** (`std::unique_ptr` or `std::shared_ptr`) if possible. If manual ref-counting is required, enforce a strict policy where ONLY the ref-counting utility is allowed to touch the memory.

## 3. Fast Debugging with Sanity Checks
- **Solution:** Wrap raw array access in helper functions with `assert` or use `std::vector::at()` in debug builds.
- **Solution:** Enable **Address Sanitizer (ASan)** with `/fsanitize=address`. It catches overruns at the exact moment they occur.

## 4. Boundary "Gates"
- Treat constructors and dimension setting as "gates." Ensure an object cannot even be created in an invalid state (e.g., dimensions > 30).

## 5. Prefer Standard Algorithms
- Use `std::ranges` or `std::transform` to reduce the risk of index typos that occur in nested `for` loops.

---

# C++23 Modernization Strategies

## 1. Automatic Lifetime Management
- Use `std::shared_ptr<grid>` (or `std::unique_ptr`) instead of manual reference counting.
- Eliminates the risk of dangling pointers, double-frees, and use-after-free crashes.
- Shared pointers correctly handle concurrent destruction in multi-threaded solvers.

## 2. Safe Multidimensional Access (`std::mdspan`)
- Avoid raw pointer arithmetic or manual 2D array indexing.
- `std::mdspan` provides a safe, standard way to view a flat or 2D buffer with defined boundaries.

## 3. The "Safety Wrapper" Pattern
- Wrap hyper-critical performance buffers in a small access struct.
- Use `assert()` in Debug builds to catch out-of-bounds access immediately.
- Use `std::span` to pass buffers around instead of raw pointers and separate size variables.


# DSL Design Philosophy: Eliminating Latent Steps

## 1. The "Flat Landscape" Problem
- **Issue:** Primitives like `Interior` or `Components` were designed to preserve original colors. When identifying a "hole" (color 0), the resulting grid looks identical to the input or an empty grid.
- **Consequence:** Beam solvers cannot "score" the progress. The discovery is pruned before it can be used by a `Recolor` or `Overlay` primitive.

## 2. Feature Maps over Latent Markers
- **Solution:** Segmentation primitives should output a **Feature Map** (Occupancy Mask with color 1) by default.
- **Benefit:** Identifying a correct shape/region immediately yields a non-zero score gradient. The solver "sees" it found the right thing, allowing the search to proceed to the next refinement step (recoloring).
- **Rule:** If a primitive doesn't yield a visible/scorable change, it must be a parameter to a visible primitive (e.g. `Filter`) rather than a standalone node.
