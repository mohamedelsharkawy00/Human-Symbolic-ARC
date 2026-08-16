ARC CUDA solver — session handout for next Claude
===================================================

You are continuing warp-parallelization work on src/cuda_solver.cu. One block =
one warp (32 threads) = one job. Goal: raise APS (attempts per second).


WHERE WE ARE
============

Current APS (fresh-kernel window, no deep-search phase):
  median 71k / p75 117k / p90 154k

Up from ~43k baseline. Peak measured ~73k before; current wins came from
dead-code removal + reverting O2 back to nvcc default O3.

CUDA compile flag: nvcc default -O3 (we briefly tried -O2 for faster builds,
reverted — runtime cost wasn't worth it).


WHAT'S WARP-PARALLELIZED (working, shipped)
===========================================

- gpu_exec_transform_warp (TurnR/TurnL/FlipH/FlipV) — base template, all others
  follow its shape. A+B+D1+E+F applied (all-lane acquire, stripe metadata
  clear, selective mask clear, divide-free pixel advance).
- gpu_exec_scale_up_warp / gpu_exec_scale_down_warp
- gpu_exec_tile_warp — handles TileX, TileY, TileByInputPattern.
  Key detail: outer-serialize objects, inner-stripe pixels.
  __syncwarp() between objects preserves "later wins" on overlaps.

Dispatcher at gpu_eval_program_warp() routes these ops to warp path before
the lane-0 fallback switch.


WHAT'S NOT WARP-PARALLELIZED (still lane-0-only in kernel)
==========================================================

Trim, Skew, Pattern, Fit, Interior, Union, Intersection, Difference, Bridge,
Partition, Recolor, Filter, Shift, Move*, Slam*, Invert, RemoveColor,
DiagTile, Input, Target, Split, Colors, Foreground, Largest, Smallest,
Seeds, Palette.

Seeds/Palette are no-ops (just index updates), handled separately.


LESSONS LEARNED (critical — do not repeat these mistakes)
=========================================================

LESSON 1 (pixels-not-objects) and LESSON 2 (amortize sync across real work per
call) now live canonically in "3 - Search/CUDA Lessons.md" — the fuller copy.
They were duplicated here; read them there. Summary: stripe PIXELS not OBJECTS
(object count is tiny, pixel count large); and only warp-parallelize an op when
typical_work_per_call / syncs_per_call is large (> ~50 bytes/sync), else the
__syncwarp cost dominates — leave it on lane 0.


PROFILING RESULTS (last nsys + vtune + ncu session)
===================================================

Tool setup: scripts/profile.bat (builds + VTune CPU hotspots + Nsight
Compute kernel deep-dive + Python analysis). Needs admin (auto-elevates).
Also nsys_command.txt for lighter nsys profile+stats pair.

Nsys 30s capture:
  Kernel time:    11.2 s (37% of wall clock)
  cudaMemcpy API: 15.3 s CPU-wall (98.6% of CPU-side API time)
  Actual GPU memcpy time: 0.15 s (103 ms H→D + 51 ms D→H)
  12800 kernel launches, avg 876 µs, median 684 µs, max 7.3 ms.

Interpretation: CPU isn't waiting on memcpy transfers. It's blocked in
the CUDA driver waiting for kernels. No streams, no async copies.

VTune CPU hotspots:
  nvcudart_hybrid64.dll  46.7%  <- driver spin wait
  nvcuda64.dll            5.9%  <- also driver
  memcpy_avx_ermsb        5.0%  <- host-side memcpy
  create_unary_node       2.1%  <- beam search program construction
  copy_and_replace_recur  2.0%  <- same
  Atomic_integral::--    ~5%    <- refcount churn (3 entries)
  flat_hash_set::contains 1.8%
  memory_resource::alloc  1.5%

Total CUDA driver time ~53%. ARC code ~22%. Rest is OS/sync.

NCU kernel deep-dive (3 launches):
  regs=255 register/thread  <- HARD CAP, compiler spilling
  occupancy: 2.08%, 14.73%, 2.09%
  SM utilization: 0.16%, 16.69%, 0.16%

Interpretation: kernel is register-starved. At 255 regs × 32 threads =
8160 regs per warp → max 8 warps per SM (65536 / 8160). After alignment
overhead, effectively 2 warps per SM. Two warps can't hide memory latency,
so SM sits mostly idle. 4050 has 20 SMs, you're using maybe 4-8 warps
total across the whole GPU.

Register pressure source: the giant switch in gpu_eval_program_warp
inlines every op implementation into one function. Compiler keeps all
possibly-live locals across all ops.

__launch_bounds__ attempted previously — did not help, compiler just
spilled more to local memory.


PTX INSPECTION NOTES
====================

Command: nvcc --ptx -O3 -arch=sm_89 -Iinclude -std=c++20 src/cuda_solver.cu
         -o tile_warp.ptx

After dead-code removal: 27563 lines / 905 KB (was 41358 / 1450 KB).

Branch patterns: 1710 predicated branches (@p bra). ZERO predicated stores
(@p st.). nvcc compiles our `if (v == TRANSPARENT) continue` to real
branches, not predicated stores. This is correct behavior on modern GPUs —
predicated stores would waste bandwidth on transparent/OOB lanes. Don't
try to force predication unless profiling proves divergence is the
bottleneck.

Many `if (x == -1) y = z` style patterns DO become setp + selp (predicated
select, no branch) — those are cheap and fine.


PRIORITIZED SPEEDUP TARGETS (by APS gain over current baseline)
================================================================

1. CPU↔GPU pipelining (streams + cudaMemcpyAsync + pinned host memory +
   double-buffered device buffers)
   Estimate: +50-100%
   Effort: medium (~200 lines around cuda_dsl_exec_and_score and its
   scratch twin in src/cuda_solver.cu, ~line 1850-1957)
   Risk: low. Standard CUDA pattern.
   Watch: batch size may need to halve if VRAM is tight (currently contexts
   eat ~4 GB of 6 GB; double-buffering needs 2x).

2. Split giant kernel into smaller per-phase / per-op kernels
   Estimate: +400-900% (if it cracks register pressure)
   Effort: high. Needs dispatch refactor, possibly one launch per op.
   Risk: medium-high. Launch overhead (17 µs) could eat gains if too granular.
   This is the real ceiling. 255 regs → 2% occupancy → 0.16% SM util.

3. Warp-parallelize remaining ops with large per-call work
   (Skew, Pattern, Trim, Fit, Interior — check LESSON 2 per op)
   Estimate: +5-15% each, +30-50% cumulative
   Effort: low per op. Pattern is established.
   Risk: low if we A/B each.

4. Bitmask mask representation (uint32_t occupancy[30] + lazy color array)
   Estimate: +20-50% (uncertain)
   Effort: very high. Touches every DSL op.
   Risk: high. Dual-path bugs. Doesn't directly fix register pressure.

5. Beam-solver CPU hotspots (create_unary_node, copy_and_replace_recursive,
   hash set, allocator churn)
   Estimate: +5-15% cumulative
   Effort: low-medium per function.
   Defer until pipelining lands.

6. Eliminate branches in hot pixel loops (predicated stores via sink-byte)
   Estimate: -5% to +5%
   Effort: low.
   Risk: likely regression. Skip unless divergence proven to dominate.


RECOMMENDED NEXT ACTION
=======================

Start with #1 (pipelining). Biggest provable win, bounded scope, low risk.

After it lands and re-profile:
  - If kernel time drops to match CPU-prep time (streams fully hiding) →
    move to #3 (more warp-parallel ops) or #2 (kernel split) depending on
    occupancy numbers.
  - If kernel time doesn't drop → register pressure is still the cap →
    #2 (kernel split) or a focused #4 (bitmask PoC on one op).


KEY FILES / LINE NUMBERS
========================

src/cuda_solver.cu (line numbers drift — grep by name if stale):
  gpu_exec_transform_warp             (reference implementation, cleanest) ~721
  gpu_exec_scale_up_warp              ~941
  gpu_exec_scale_down_warp            ~1013
  gpu_exec_tile_warp                  ~1191
  gpu_eval_program_warp (dispatcher)  ~1372
  warp-handler chain (is_transform, ScaleUp, ScaleDown, TileX/Y,
    TileByInputPattern, Seeds, Palette) starts ~1389
  lane-0 fallback switch (remaining non-warp ops) ~1438
  dsl_exec_and_score_kernel_v2 (the single kernel) ~1637
  cuda_dsl_exec_and_score (main batch entry, all sync memcpy) ~1855
  cuda_dsl_exec_and_score_scratch (twin for scratch path) ~1925

include/cuda_dsl.cuh:
  gpu_exec_context, gpu_fixed_object (14 meta bytes + 900 mask bytes = 914)
  GPU_KERNEL_MAX_OBJECTS=64, GPU_MAX_GRID=30, GPU_TRANSPARENT=-1

CMakeLists.txt:
  nvcc on default -O3 (do not re-add -O2 — tried, reverted, not worth it)
  sm_75 + sm_89 targeted.

Logs: logs/007bbfb7.txt (APS stats — use awk pipeline below)
  awk '/\[APS\]/ {gsub(",","",$4); aps=$4+0; if(aps>1000) print aps}' LOG \
    | sort -n \
    | awk 'BEGIN{c=0} {a[c++]=$1; s+=$1} \
           END{print "mean:",s/c; print "median:",a[int(c/2)]; \
               print "p75:",a[int(c*0.75)]; print "p90:",a[int(c*0.90)]}'
  Note: a freshly-started log has high APS (shallow programs); deep-search
  phases drag the average. Compare windows (first 2000 vs last 2000 samples)
  rather than lifetime stats.


DEAD CODE REMOVED THIS SESSION (do not re-add)
===============================================

- Serial gpu_exec_scale_up / gpu_exec_scale_down (warp versions supersede)
- Serial gpu_exec_tile / gpu_exec_tile_by_input_pattern
- Experimental gpu_exec_union_warp (regressed, reverted)
- 11 unreachable fallback switch cases

PTX size dropped 33%, predicated branches dropped 18%.


USER PREFERENCES OBSERVED
=========================

- Prefers fixes over reverts: "prefer fix than revert, take another look"
- Wants changes described before committing: asks for descriptions of new
  code to "lend a human eye"
- Likes ELI5 explanations of GPU concepts
- Likes prioritized suggestion lists with impact estimates
- Wants durable memory captured in root .txt files when new lessons arise
- Does not want you to blindly parallelize without LESSON 2 check
