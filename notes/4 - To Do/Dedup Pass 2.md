# Dedup Pass 2 — remaining duplication found 08-07-2026

Pass 1 (done): all CPU/GPU agree-by-comment contracts now have one home in
cuda_dsl.cuh — op list, scoring weights, move deltas, domain constants
(grid size / transparent / color count / max ops), shape occupancy hash.

Pass 2 findings, worst first, not yet fixed:

1. GUI buffer helpers duplicated verbatim.
   pad_buffer, upload_grid, allocate_empty_grid, update_display_buffer exist
   twice: gui.hpp ~1251-1361 (Windows) and gui_stub.hpp 17-54 (headless
   Kaggle). Not display code — they define the grid memory layout the solver
   consumes, so drift means the Kaggle build behaves differently from the
   local build. Fix: move the four into one shared header; gui_stub keeps
   only the load_font_atlas no-op.

2. GPU upload recipe written twice.
   dsl_executor.hpp 55-119 (single-example scratch path) restates
   solver_executor.hpp build_example_upload ~1094-1129 (main path) nearly
   line for line: OutputColors 1x1-object list, by-size palette, shape-color
   table, background derivation, rule params, target CC summary copy.
   Already bit once (score-bar mismatch, see comment at dsl_executor.hpp ~81).
   Fix: extract shared "fill puzzle-level fields" + "fill one example slot"
   helpers both paths call.

3. Grid flood fill written three times on the CPU.
   Same 8-connected queue walk in fixed_dsl.hpp fixed_flood_fill,
   environment.hpp segment_objects_by_color_cpu, inference.hpp
   detect_shape_to_interior_color_map; they differ only in background rule
   and per-component output. Fix: one helper taking an is-background
   predicate and a per-component callback. Device gpu_flood_fill stays —
   different storage and warp mechanics, not a copy.

4. CPU/GPU logic twins (bigger jobs, judge separately).
   - inference.hpp interior detection mirrors gpu_interior_paint_warp.
   - detect_shape_color_map key-smallest/target-largest selection mirrors
     gpu_exec_shape_color_map.
   Learn-side and apply-side genuinely differ in shape; unifying is a
   project, not a cleanup.
   Where a detector REPLAYS an op rather than learning a table for it, it is
   a cleanup after all: see "1 - Overall Design/One Body Two Compilers.md"
   (2026-09-22). VoteDenoise went first; project_on_seeds' host stamping of
   gpu_exec_project_on_seeds is the next candidate.

Checked and clean: the exec/score kernel split in cuda_solver.cu is a true
split, not a copy — no scoring duplication.
