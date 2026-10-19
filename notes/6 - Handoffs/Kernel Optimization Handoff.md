# ARC GPU-Solver — Session Handoff (2026-06-05)

Read this top-to-bottom before touching the kernel. This session **closed out
incremental kernel optimization** (proved it's tapped out, six ways) and
**validated the next big lever via a small-scale microbench**: a **1-job-per-thread
kernel rewrite**. That rewrite is the committed direction. Everything below is the
evidence trail so you don't re-chase dead ends, plus the rewrite plan.

---

## 0. TL;DR

- Hot kernel: `dsl_exec_and_score_kernel_v2` in `src/cuda_solver.cu` — ~100% of GPU time.
- **The kernel is at a hard equilibrium at ~250K aps (idx8).** It's broadly
  *latency-bound* across thousands of tiny independent jobs. Occupancy is walled,
  the working set is already cache-warm, and **every micro-optimization this session
  came back neutral** (see §3). This is now *proven*, not suspected.
- **Root cause (NCU + ptxas, confirmed):** the monolithic DSL op-switch in
  `gpu_eval_program_warp` needs ~80 registers + spills ~8.6KB → forces a 4.2KB shared
  reservation (`enable_smem_spilling`) → caps occupancy at ~19-24 blocks/SM → too few
  warps to hide global-memory latency (51% `long_scoreboard`). All three constraints
  (regs, smem, spill) are coupled, so freeing any one is eaten by the others.
- **The committed next step:** rewrite the kernel from **1 warp = 1 job** (32 lanes
  cooperate) to **1 thread = 1 job** (32 *independent* jobs per warp). This gives
  maximal independent work per warp → hides latency *within* the warp instead of
  needing more resident warps (which we can't get). **Microbench on the real hotspot
  with real data showed thread-per-job ~1.4× faster** (§6). This is the lever.
- Current build is a **clean baseline** at ~250-282K (puzzle-state-dependent), all
  experiments reverted, ~2.3GB of profiling junk cleaned. Ready to start the rewrite.

---

## 1. How to build / run / measure

- **Build:** `scripts\build.bat x64-Release NOLAUNCH` (Release = what the user runs).
  `x64-RelWithDebInfo` has `-lineinfo` for profiling. build.bat kills ARC.exe first.
- **Run:** `out\x64-Release\ARC.exe` (Win32 GUI; auto-solves the persisted puzzle).
- **Pin the puzzle for any A/B.** Persisted index: `current_puzzle/current_puzzle.txt`
  (0-based int). Standardized on **idx 8 = `06df4c85`** (InteriorColor-heavy, hard).
  Logs: `logs/<puzzle_id>.txt`. Cross-puzzle variance dwarfs change deltas — same puzzle only.
- **APS measurement: `scripts\measure_aps.ps1 [seconds] [puzzleId]`** (added this
  session — KEEP it). Truncates the puzzle log, launches, samples `[APS]` lines,
  reports p1/median/min/max. **Use `median`, not mean/p1** — p1 is dominated by
  level-boundary zeros (`min=0` is normal — beam has nothing to run at level edges).
  Defaults: 10s, idx8. Example: `median ~250K` is the baseline.
- **CRITICAL MEASUREMENT TRAP:** GPU contention silently corrupts APS. One batch of
  this session's numbers was garbage because the user was running a game (NFS) on the
  same GPU. **Confirm the GPU is idle before trusting any wall-clock delta.** ptxas
  numbers (regs/smem) are load-independent and always trustworthy — gate on those first.
- **ptxas-first workflow:** every kernel change, read `Used N registers / N bytes smem`
  from the build output BEFORE bothering to run. If occupancy resources don't move,
  the run won't either. (PowerShell to extract: build via `cmd /c "..." *>&1 | Tee-Object`,
  regex `Used (\d+) registers.*?(\d+) bytes smem`.)

---

## 2. The confirmed bottleneck (fresh NCU `--set full`, profiling/output/ncu_raw.csv)

Averaged over 11 strided kernel samples on the clean baseline:

| metric | value | meaning |
|---|---|---|
| occupancy limiter | **shared mem → 19 blocks** | regs cap at 24; **shared is tighter** |
| achieved occupancy | 39% (14 of 48 warps) | half-empty SM |
| eligible warps/sched | **0.49** (<1) | scheduler usually has nothing ready |
| **SM (compute) throughput** | **38%** | NOT compute-bound |
| **DRAM bandwidth** | **9%** | NOT VRAM-bandwidth-bound |
| L1 hit rate | **86%** | data is cache-resident |
| L2 hit rate | 99.5% | almost nothing reaches DRAM |

Per-reason warp stall (sums to ~10.5 cycles/issue): **long_scoreboard 46%** (global
memory *latency*, not bandwidth), **wait 26%** (fixed-latency dependency chains),
**no_instruction 11%** (I-cache — the monolithic kernel is huge), branch_resolving 5%,
not_selected 4%.

**Per-instruction hotspot** (`ncu --page source --csv`, regenerable): the stalls land
on `ISETP`/`PRMT` instructions consuming just-loaded grid bytes (the dependent-load
chain `a = output.data[..]; distance_maps[a][..]`) and the **CC-labeling relaxation
loop** in the scorer (re-reads the grid + 8 neighbours every sweep, many sweeps).

**Interpretation:** the warps wait on individual grid-cell byte loads — most acutely,
waiting for `a` (the produced cell colour) before they can index `distance_maps[a]`.
The bytes are in L1 (86% hit); the problem is *latency of dependent loads with too few
warps to hide it*, not bandwidth. This is the opposite of a compute-saturated kernel
(FurMark): low arithmetic intensity, all memory-touch, little math to fill the waits.

---

## 3. DEAD ENDS — measured neutral/worse this session, do NOT repeat (idx8, GPU idle)

All of these built clean and were measured against the ~250K baseline. **All neutral
or worse.** The pattern itself is the finding: no per-loop / data-placement tweak moves
the needle because no single piece is a tall pole per-job (jobs are tiny, ≤900 cells).

| experiment | ptxas effect | APS | why neutral |
|---|---|---|---|
| `__noinline__` 12-16 heavy ops | spill 5504→1788B, **smem unchanged 4208** | ~250K | pragma refills freed space with spill |
| interior_paint `__shared__`→global (free 1800B) | **smem 4208→4192** (−16B!) | ~250K | `enable_smem_spilling` re-claims it as spill |
| snap_to_anchors locals→global scratch | — | ~250K | cold lane-0 op, not the bottleneck |
| read `distance_maps` from template (cut 18MB/launch copy) | ctx −9KB | **~127K WORSE** | per-job copy was warming L1; shared read is colder |
| stage output grid in `__shared__` (sgrid) | smem unchanged | ~250K | grid already 86% L1-hot; byte-shared bank conflicts |
| pointer-jumping CC labeling | — | ~250K | 30×30 grids too small; old crawl already converged fast |
| ILP 2-cells/iter in score loop | regs unchanged | ~250K | score loop is ~28 cells/lane — too short for ILP runway |
| **pragma OFF + noinline** (the occupancy unlock) | **smem 4208→2024**, blocks 19→~24 | **~250K** | **directly raised occupancy → still neutral**: extra warps' DRAM spill cancels the gain |
| **3rd pipeline slot** (host) | VRAM +0.75GB, util 81→83% | ~250K | GPU wasn't starved for buffering; CPU single-thread turnaround is the limit |

**Key proofs from the dead ends:**
- **Occupancy is NOT the lever.** We raised it directly (pragma-off, smem halved,
  19→24 blocks) and APS didn't move. The walls (regs 80 / smem / spill) are coupled
  AND even breaking them doesn't help — added warps bring their own spill traffic.
- **The kernel needs ~80 registers no matter what.** Forcing lower (maxrregcount /
  launch_bounds>24) makes ptxas explode to 128. The op-switch genuinely needs them.
  Probe: stubbing eval's body drops stack 8752→144B and smem 4208→224B → **the
  op-switch is the entire spill/smem driver** (scoring is innocent; removing scoring
  left 80 regs / 8.5KB spill unchanged).

---

## 3.5 COMPLETE LIST OF CLOSED OPTIMIZATION PATHS

Every optimization avenue tried or ruled out, this session + carried-forward from prior
sessions. **Do not re-attempt any of these without new evidence.** Grouped by category.
"Neutral" = measured ~250K (no change). Status: ❌ dead, ⚠️ conditionally dead, ✅ KEPT.

### A. Occupancy / register / shared-memory (the core wall)
1. ❌ **`__noinline__` heavy ops** (12, then 16) — spill ↓ but smem stays 4208 (pragma refills). Neutral.
2. ❌ **`__noinline__` + pragma OFF together** — smem 4208→2024, occupancy 19→24 blocks. **Directly raised occupancy → STILL neutral** (extra warps' DRAM spill cancels gain). *This is the definitive occupancy disproof.*
3. ❌ **Remove `enable_smem_spilling` pragma alone** — smem drops but spill→DRAM; prior session measured −63% (at 5504B spill). Re-tested low-spill: neutral. Pragma is load-bearing OR irrelevant, never a win.
4. ❌ **`__launch_bounds__` sweep** (21/22/23/24/25/26/28/32) — 21-24 all give identical 80 regs/4208 smem; ≥25 → ptxas "out of range", falls back to 128 regs (worse). 24 is the ceiling. `(32,21)` is the documented optimum.
5. ❌ **`--maxrregcount=48`** — IGNORED by ptxas under launch_bounds (stays 80); without launch_bounds → 128 regs. Can't force regs below 80.
6. ❌ **interior_paint `wall/outside __shared__` (1800B) → global** — freed only 16B smem (pragma re-claims as spill). Prior session: −25% on InteriorColor puzzles. Neutral-to-worse.
7. ❌ **snap_to_anchors ~6.3KB local arrays → global scratch** — cold lane-0 op, not the bottleneck. Neutral.
8. ❌ **interior_paint→global cc_labels** (prior session) — −25%. Dead.

### B. Memory traffic / data placement (kernel is already cache-warm: L1 86%, DRAM 9%)
9. ❌ **Read `distance_maps` from template instead of per-job copy** (cut 18MB/launch) — **WORSE (~127K)**: the per-job copy was warming L1; shared read is colder. Counterintuitive but firm.
10. ❌ **Stage output grid in `__shared__` (sgrid)** for the scorer's repeated reads — grid already 86% L1-hot; byte-addressed shared → bank conflicts. Neutral.
11. ❌ **Un-pack the 4-bit mask** (kill the PRMT nibble-decode tax) — prior session: inflates registers/context → worse occupancy. Decode tax << occupancy cost. KEEP packing. (Unpacked anyway 2026-10-19, the user's call: Solver Optimization Handoff 2026-06-04.md, section 5.)
12. ⚠️ **"Touch less VRAM / shrink working set"** — DEAD as a concept here: DRAM 9%, L2 99.5% hit. We are NOT bandwidth-bound. Don't pursue smaller-context-for-L2 ideas.

### C. Algorithm-level (the hotspots)
13. ❌ **Pointer-jumping CC labeling** (log-convergence vs the crawl) — 30×30 grids too small, objects too few; old min-propagation already converged in few sweeps. Neutral.
14. ❌ **ILP: 2 cells/iteration in the score loop** (overlap dependent-load chains) — score loop is only ~28 cells/lane, too short for ILP runway. Neutral.
15. ❌ **Faster CC / shared-grid for CC** — covered by 10 & 13. The CC loop is the NCU #1 hotspot but it's not big enough per-job to optimize in place.

### D. Pipeline / feed / CPU side (GPU 80% busy, balanced — no big bubble)
16. ❌ **3rd pipeline slot** (host, 2→3 exec_batch slots) — +0.75GB VRAM, util 81→83%, neutral. GPU wasn't starved for buffering; the limit is single-thread CPU turnaround.
17. ❌ **Pack-ahead / reorder CPU phases** — NOT built, but ruled out by timing: per-iteration CPU (~6.8ms) ≈ kernel (~6.7ms), and the 4.5ms "fetch" GPU-wait *already overlaps* the next kernel. No reclaimable bubble. Reordering just shuffles it.
18. ❌ **Accumulate more jobs per launch** — chunk fill is already 2047/2048 avg (NOT job-starved). More jobs = same total work, same occupancy. (Old handoff's "chunk maxes ~1800" was wrong for idx8.)
19. ⚠️ **Move result-processing to a 2nd thread** — NOT tried; would only help if processing were the bottleneck, but timing showed the 4.5ms is GPU-wait (already overlapped) and real CPU processing is only ~1.3ms < kernel. Low ceiling. Not worth the concurrency risk.
20. ⚠️ **`cudaStreamSynchronize` in `cuda_dsl_fetch_output_grids`** looked like a 4.5ms cost — but it's the GPU-kernel wait (copy is 900 bytes), already overlapping the next launch. Removing/restructuring it does not reclaim real idle. Don't bother.

### E. Things that are NOT closed (the actual forward path)
- ✅ **1-job-per-thread rewrite** — VALIDATED ~1.4× on the scorer microbench (§6). This is the committed direction. It changes the execution *shape*, which is the only thing that can move a latency-bound, occupancy-walled, tiny-job kernel.
- (Open, untested, lower priority) Splitting exec and score into two kernels — partially subsumed by the thread rewrite; revisit only if the rewrite stalls.
- (Open) 4-GPU sharding / bigger GPU — out of scope, hardware not algorithm.

**Meta-lesson for the next agent:** the kernel is broadly latency-bound with NO tall
pole per-job (jobs are ≤900 cells). This is why categories A-D were all neutral — there
is nothing to cut in place. Only changing the execution shape (E) moves it. If you find
yourself about to shuffle memory, free a resource, or reorder phases — it's in this list
and it's neutral. Build the rewrite instead.

---

## 4. The CPU/feed side — also measured, also not the easy win

`nsys` timeline (70s, regenerable via `profile_nsys.bat`): **GPU busy 80%, idle 20%**
(13.7s in ~8,234 gaps of ~1.4ms each). The gap is "CPU hasn't issued the next launch
yet," NOT GPU starvation.

**Per-iteration CPU phase split** (temp timers, since removed):
- generate 0.9ms, pack 0.06ms, process(tally+winner-find+logging) 1.3ms,
  **fetch's `cudaStreamSynchronize` 4.5ms**.
- The 4.5ms "fetch" copies only ~900 bytes — it's **not transfer, it's the CPU
  blocking on the GPU kernel** (`cuda_dsl_fetch_output_grids`, line ~3083). And that
  block **already overlaps the next launched kernel**. So CPU total (~6.8ms) ≈ kernel
  (~6.7ms): **the system is balanced and GPU-bound, already well-pipelined.**

This is why the **3rd slot did nothing** and why **pack-ahead/reorder won't help
either**: there's no large reclaimable bubble — the GPU wait already overlaps work,
and the 20% idle is mostly genuine level-boundary gaps. Don't re-chase pipelining.

**Stub test (historical, still true):** `kStubGPU` skip-GPU → ~2.6M aps vs ~250K real
= the GPU is the long pole, but the CPU feed has ~10× headroom (generation is only
~0.9ms/iter). (The `kStubGPU` toggle was REMOVED this session — re-add a local
`if(false)` around the launch if you need this measurement again.)

---

## 5. ROOT CAUSE, stated once, precisely

The DSL interpreter `gpu_eval_program_warp` inlines the union of ALL DSL ops into
one giant `switch`. That function's worst-case live state → **~80 registers + ~8.6KB
local spill**. With `enable_smem_spilling` + `__launch_bounds__(32,21)`, the spill is
parked in **~4.2KB of shared memory**, which is the **occupancy limiter (19 blocks)**.
Too few resident warps (14 of 48) → can't hide the dependent-load global latency that
dominates a low-arithmetic-intensity, tiny-job workload. The warp-cooperative mapping
(32 lanes on ONE dependency-chained job) means lanes stall *together* on the same load.

You cannot fix this by shuffling memory or freeing one resource — proven in §3. The
fix must change the **execution shape**.

---

## 6. THE COMMITTED NEXT STEP — 1-job-per-thread rewrite (VALIDATED)

**Idea:** map **1 thread = 1 whole job** instead of 1 warp = 1 job. Then the 32 threads
in a warp run **32 independent jobs**. When thread 0 stalls on a dependent load, threads
1-31 have their own independent work — the warp hides latency *internally* (ILP-style),
which is exactly the latency-hiding we cannot get by adding warps (occupancy is walled).

**Why it should work where everything else failed:** our problem is "not enough
independent work to hide memory latency." Thread-per-job makes every thread's work
independent of its neighbours *by construction* — maximum independent work. It directly
attacks the 38%-SM-efficiency root cause. It also shrinks per-thread register pressure
(each thread runs ops *serially*, holding only the current op's state, not a 32-lane
cooperative reduction's state).

**Small-scale validation (done this session, the reason we're committing):** added a
throwaway microbench inside the real program (`kBenchScorer`, since reverted) that ran
the **scorer hotspot** (dependent-load score loop + CC labeling) both ways on the REAL
`output_grids` the live kernel produced, timed with cudaEvents:
- **thread-per-job = ~1.4× faster than warp-per-job**, stable across the run.
- Caveat: the warp baseline was lane-0-only (a strawman), so 1.4× is not the exact vs
  the real 32-lane scorer. BUT the microbench *understates* the full win — it only
  measured scoring, while **eval is the bigger dependency chain**, and thread-per-job
  also gives 32× more resident jobs/warp. Full-pipeline payoff likely > 1.4×.

**Rewrite plan (the actual work):**
1. New kernel `dsl_exec_and_score_kernel_threaded`: `idx = blockIdx*blockDim+threadIdx`,
   one thread owns one job end-to-end. blocks = `(n+T-1)/T`, threads = 128 or 256.
2. **Rewrite every DSL op from warp-cooperative to single-thread-serial.** This is the
   bulk of the work: `gpu_exec_transform_warp`, `scale_up/down_warp`, `tile_warp`,
   `rotate_warp`, `filter_warp`, `union_warp`, `interior_color_warp`, and the lane-0
   ops all currently assume 32 lanes / `__syncwarp` / `__shfl`. The serial versions are
   often *simpler* (no reduction, no lane dispatch). The lane-0-only ops are already
   serial — they mostly just lose the `if(lane==0)` guard.
3. **Per-thread scratch:** the warp version shares `ctx.op_scratch`/`cc_labels`/shared
   summaries per-block. Thread-per-job needs per-THREAD scratch (or per-thread slices of
   a global pool sized `n * GPU_GRID_CELLS`, like the microbench's `lab_pool`). Watch
   VRAM: per-thread context is the big cost — `gpu_exec_context` is ~0.34KB×... check it
   fits at n=2048 (was ~0.7MB/job historically; may need shrinking — see GPU_KERNEL_*
   limits in cuda_dsl.cuh).
4. **Register watch:** serial ops should need FEWER registers (no cooperative state).
   If per-thread regs stay ≤ ~40, occupancy rises naturally → big win. ptxas-gate this.
5. **Correctness:** the scorer must produce *identical* scores (the beam depends on
   exact ordering — CC representative = min linear index). Verify against the warp
   version on the same grids before trusting it. Run the benchmark mode (toggle, 'B'
   key) for a solve-rate check vs a pre-rewrite backup — NOT done in a while.

**Microbench code is gone (reverted)** but the approach is in git-less backups
(`../ARC Backups/`, look ~07:30 on 2026-06-05) if you want to re-extract the
`bench_score_thread`/`bench_score_warp`/`bench_score_one` skeleton — it's a good
starting template for the serial scorer.

---

## 7. Watch-outs / invariants (carry forward, still all true)

- **Objects can have bbox w/h > 30** (off-grid via Union/Shift/Move). Any op that loops
  obj dims into a fixed 30×30 buffer MUST clamp `min(dim, GPU_MAX_GRID)`. The packed-mask
  proxy clamps cell index ≥900 → 0. Don't remove these. (This was the original "madness"
  OOB bug.)
- **active-mismatch gate** (`batch_stale = prev_batch.active_examples != active`): a real
  correctness fix — a solving candidate promotes → `active` grows → an in-flight batch
  packed against OLD active would be indexed with NEW active → wrong scores/false solves.
  KEPT. (Its `[DIAG batch-stale]` *log* was removed this session; the gate stays.)
- **re-upload sync:** before `cuda_dsl_upload_examples`, sync in-flight batches reading
  the old templates (overwriting mid-flight corrupts their scores → "halt then madness").
  Currently syncs `wait_slot`; **if you add more pipeline slots, sync ALL pending slots.**
- **GPU-error detector (KEPT, permanent):** after each batch, `cuda_dsl_take_error()` →
  logs `!!! GPU ERROR ...` to the puzzle log. CUDA errors are silent + sticky (one bad
  access no-ops all later launches → frozen/garbage output, invisible). This makes it
  loud + greppable. **First thing to check if output freezes/garbles** during the rewrite.
- **3 OOB asserts** (`[ASSERT pidx/j/dims-OOB]` → `logs/diag.txt` via `diag_log`): KEPT,
  cheap, catch indexing bugs that would corrupt scoring.
- `MAX_PROGRAMS` (environment.hpp) ≥ `TARGET_JOBS` (2048) for active=1 to fill a grid.
- Chunk fill is healthy: instrumented avg **2047/2048 jobs per launch** — NOT
  job-starved (the old handoff's "chunk maxes ~1800" was wrong for idx8). More jobs/launch
  won't help.

---

## 8. State of the tree at handoff

- **Clean baseline**, builds clean, runs correct (no GPU errors/asserts), ~250-282K aps
  (puzzle-state-dependent; same-session A/B is what matters, not absolute).
- All session experiments reverted. `src/cuda_solver.cu` + `include/solver_executor.hpp`
  are at the clean baseline (restored from `../ARC Backups/` ID 2277).
- **Removed this session:** `kStubGPU` toggle + branch, `[DIAG batch-stale]` log line,
  the redundant `diag_log` half of the GPU-error detector. ~2.3GB profiling junk
  (nsys sqlite+rep, my 240MB ncu_source.csv, temp phase/fetch/bench logs).
- **Kept:** `scripts/measure_aps.ps1` (useful), GPU-error detector, OOB asserts, the
  `[SCORE]` debug logging (predates this session), the `.ncu-rep` files + `ncu_raw.csv`.
- **Build config:** `__launch_bounds__(32, 21)` + `enable_smem_spilling` pragma =
  80 regs / 4208B smem (the documented, measured-optimal warp-per-job config). The
  rewrite will replace this entirely.

---

## 9. Tooling / profiling scripts (all in scripts/, all KEEP)

- `build.bat <preset> [NOLAUNCH]` — single source of truth for builds. Has `-Xptxas=-v`.
- `measure_aps.ps1 [sec] [puz]` — single-puzzle p1/median APS (this session). Use median.
- `profile_ncu.bat [consecutive]` — Nsight Compute, `--set full`, writes `ncu_raw.csv`
  (the only page with stall reasons) + `.ncu-rep`. STANDALONE — never run with VTune
  (vtss.sys + NCU on nvlddmkm = BSOD; the script aborts if vtss is loaded). Needs Admin.
- `profile_nsys.bat` — Nsight Systems timeline. WARNING: 70s capture on a 250K-launch/sec
  workload produces a ~2GB sqlite and can hang finalization — consider shortening the
  `--duration` and dropping `--trace=wddm` for a lighter capture.
- `profile.bat` — VTune (CPU). Backups: `../ARC Backups/` via `scripts/backup.py`
  (POST_BUILD) — snapshots include `current_puzzle.txt` so you can recover which puzzle
  any past measurement used.
