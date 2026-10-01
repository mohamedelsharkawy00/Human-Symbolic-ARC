========================================================================
ARC SOLVER — CUDA TRANSITION STATUS (2026-04-05)
========================================================================

GOAL: Full CUDA solver. Bottom-up: fixed CPU DSL → replace legacy → CUDA kernel.

========================================================================
COMPLETED
========================================================================

PHASE 1 — Fixed-size CPU DSL executor:
  - fixed_dsl.hpp: all 33 active DSL ops, no heap, no pointers, GPU-portable types
  - Validated 100% match (10,400+ checks) against old heap-based executor
  - No APS regression

PHASE 2 — Replace legacy, remove dead code:
  - execute_candidate() in solver_executor.hpp uses fixed DSL directly
  - dsl_executor.hpp gutted to ~30 line bridge (routes execute_flat_program
    through fixed DSL). Only used by secondary paths:
      score_tree_output chain, test eval in worker()
  - Removed from environment.hpp: input_to_object, flood_fill,
    segment_objects, segment_by_color, input_bbox_size (~155 lines)

========================================================================
WHAT REMAINS AND WHY
========================================================================

environment.hpp still has:
  - grid, object, node structs — used by beam search tree generation
  - Tree ops (stringify, parse, simplify, flatten, copy, remove, score)
  - detect_invariant_colors, detect_size_rule
  - nearest_manhattan — kept as reference for commented-out Nearest DSL op
  - 5 commented DSL ops in NODE_PRIMITIVES (all arity-2):
      Nearest(L83), Align(L84), Fill(L98), Project(L106), Connect(L107)
    Binary generation phases (3-6) are disabled in solver_executor.hpp L587

========================================================================
NEXT: PORT DSL EXECUTOR TO CUDA KERNEL
========================================================================

fixed_dsl.hpp is already GPU-portable by design. Main challenges:

1. MEMORY: fixed_exec_context is ~100MB (128 pools × 900 objects × 909B).
   Real programs use <10 pools. Size down for GPU.

2. INTEGRATION TARGET: Replace current worker() flow:
     CPU: execute_candidate → grids → pack → GPU: score → CPU: pick winner
   With:
     CPU: pack programs+inputs → GPU: execute+score → CPU: read winner

3. cuda_dsl_host_batch in cuda_dsl.cuh already defines the batch format.

========================================================================
GOTCHAS
========================================================================

ENUM SYNC: gpu_op_type (cuda_dsl.cuh) SKIPS the 5 commented-out node_types.
  Adding/uncommenting an op requires updating: both enums, convert_program_to_gpu_ops
  switch, AND fixed_eval_program switch.

INCLUDE ORDER: fixed_dsl.hpp uses environment.hpp types but doesn't include it.
  Relies on solver_beam.hpp including environment.hpp first.
  solver_beam.hpp bottom: fixed_dsl.hpp → dsl_executor.hpp → solver_executor.hpp

BUILD: Must use tmp_build.bat (or scripts/build.bat) — calls vcvars64.bat first.
  From Claude Code: cmd //c "C:/Users/nitro/Desktop/ARC/tmp_build.bat"
  Direct cmake from bash fails (no MSVC include paths).
  MSVC 2025 (VS 18), C++20, Ninja, CUDA toolkit.

APS: No hot-path instrumentation. Sampled validation only.

POOL OVERFLOW: acquire() clamps at pool[127], safe because max program
  depth is 8 and real programs use <10 slots.
