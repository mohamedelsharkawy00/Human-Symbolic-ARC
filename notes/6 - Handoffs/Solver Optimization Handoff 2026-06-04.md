# ARC GPU-Solver Optimization — Session Handoff (2026-06-04)

Detailed handoff for the next agent/session. Read this top-to-bottom before touching
the kernel. The cheap wins are done; the next tier is a real refactor (Section 6) —
the previous session proved every shortcut to it backfires, and *why*.

---

## 0. TL;DR
- Hot kernel: `dsl_exec_and_score_kernel_v2` in `src/cuda_solver.cu` — ~100% of GPU time.
- This session: **fed the GPU** (chunk accumulation), **raised occupancy** (`__launch_bounds__(32,21)`),
  **fixed a latent kernel OOB** (the "madness"), **coalesced local arrays**. Net: large, stable throughput gain.
- **Current bottleneck (NCU-confirmed): occupancy, register-spill-capped via shared memory.**
  long_scoreboard (memory latency) is 42% of stall cycles and isn't hidden because there
  are only ~0.49 *eligible* warps/scheduler.
- **The only real unlock left is reducing `gpu_eval_program_warp`'s register pressure**
  (it inlines ALL DSL ops → ~255-reg union → forced spill → shared cap → occupancy
  ceiling). That's Section 6. Everything else is a tweak.

---

## 1. How to build / run / measure (DO THIS RIGHT or numbers lie)

- **Build:** `scripts\build.bat x64-Release NOLAUNCH` (Release is what the user runs).
  `x64-RelWithDebInfo` has `-lineinfo` for profiling/sanitizer. build.bat kills ARC.exe first.
- **Run:** `out\x64-Release\ARC.exe` (Win32 GUI; auto-solves the persisted puzzle on launch).
- **CRITICAL — puzzle pinning.** aps is *puzzle-specific*; cross-puzzle variance (e.g. 243K vs 870K)
  **dwarfs** change deltas (+14%). The persisted index is `current_puzzle/current_puzzle.txt`
  (0-based, plain int). Logs are named `logs/<puzzle_id>.txt`. **Pin the same puzzle for any A/B.**
  This session standardized on **idx 8 = `06df4c85`** (a hard, InteriorColor-heavy puzzle).
  Backups in `../ARC Backups/` snapshot `current_puzzle.txt` per build — use them to learn which
  puzzle an old measurement used (that's how we discovered earlier A/Bs were cross-puzzle).
- **aps measurement (clean delta, avoids log-append issues):** launch, sleep 12s, read
  `Attempts:` from `logs/<id>.txt`, sleep 30s, read again, `aps=(a2-a1)/30`. Build.bat
  kills the GUI, so close any open instance first.
- **Better than single-puzzle aps: benchmark mode** (`toggle_benchmark`) → runs N puzzles,
  reports `benchmark_total_aps` + train/test solved. Use it for a *corpus* number and to
  catch solve-rate regressions. We have NOT run a solve-rate check since the kernel changes —
  **do this** (the OOB clamps truncate >30 objects; should be fine but verify).

---

## 2. What's in the current build (all kept, all real)

| change | where | effect |
|---|---|---|
| **Chunk accumulation** | `solver_executor.hpp` worker, `TARGET_JOBS=2048` | grid 1-wave → 8.5-wave; the structural win that *unlocked* occupancy |
| **`__launch_bounds__(32,21)`** | `cuda_solver.cu` kernel sig | 255→80 regs, 16.7%→~44% occ; +58% on idx8 |
| **`enable_smem_spilling` pragma** | kernel body, first stmt | spills → fast shared not DRAM; **load-bearing** (see §5) |
| **Local arrays → global `op_scratch`** | slam_row/slam_column/skew + autocrop in-place | killed uncoalesced L1TEX local loads (NCU "1.3/32 bytes"); +14% on idx8 |
| **Kernel OOB clamps** | slam/slam/interior_paint + packed-mask proxy | THE "madness" fix (§4) |
| **CUDA error detector** | `cuda_dsl_exec_and_score` launch + worker | permanent; logs `!!! GPU ERROR` to puzzle log |
| **active-mismatch gate** | worker result proc (`batch_stale`) | skips a batch whose `active` changed mid-flight |
| **re-upload sync** | worker, before `cuda_dsl_upload_examples` | waits in-flight batch before overwriting templates |
| **OOM check** | `resize_exec_buffers` | aborts loud instead of corrupting context |
| 4-bit packed mask | `gpu_packed_mask` in `cuda_dsl.cuh` | half mask traffic; KEEP (helps long_scoreboard) |

Approx idx8 (06df4c85) ladder, same puzzle: (32,12) no-local 243K → (32,21) no-local 383K
(+58%) → (32,21)+local 438K (+80% total). Occupancy gain is **puzzle-dependent (+16% to +58%)**.

---

## 3. The confirmed bottleneck (NCU `--set full`, see `profiling/output/ncu_raw.csv`)

Scheduler: **No Eligible 64.7%**, **Eligible Warps/Scheduler 0.49** (<1 ⇒ scheduler usually
idle), Achieved/Theoretical warps/SM **14.2/19**, Warp Cycles/Issued **11.4**.

Per-reason warp stall (cycles between issues, sums to ~11.4):
| reason | % | meaning / source |
|---|---|---|
| **long_scoreboard** | **42%** | L1TEX memory latency (global: masks/output/cc_labels; + local). Hidden by occupancy. |
| **wait** | 24% | fixed-latency dependency chains — incl. the 4-bit mask nibble decode + index math |
| **no_instruction** | 11% | I-cache misses — the monolithic all-ops-inlined kernel is huge |
| branch_resolving | 4.5% | divergence (lane-0 serial sections) |
| not_selected | 3.3% | **low ⇒ occupancy-starved, not throughput-saturated** |
| short_scoreboard / throttles | ~9% | smem-spill waits + pipe throttle |

**Pull these any time without the GUI:** `profile_ncu.bat` now also writes `ncu_raw.csv`
(`ncu -i <rep> --csv --page raw`). The default `--csv` page OMITS stall reasons — use raw.
Parse columns matching `warps_issue_stalled_*_per_issue_active`.

---

## 4. The "madness" bug (root-caused & fixed — context so you don't re-chase it)
Symptom looked like a display/cross-puzzle leak + score spam + a "short pause." It was a
**kernel out-of-bounds → sticky CUDA error → every later launch no-ops → frozen output_grids
→ identical stale scores**. Root: objects whose **bbox w/h > 30** (Union/Shift/Move push
objects off-grid) fed into `slam_row`/`slam_column`/`interior_paint`, which looped those dims
into fixed 30×30 local/shared arrays → OOB. Fix: `min(obj.width/height, GPU_MAX_GRID)` at those
sites + a clamp in the `gpu_packed_mask` proxy `operator[]`. **Lesson:** there was ZERO CUDA
error checking — a sticky fault was invisible for a whole session. The `capture_cuda_err`
detector now makes it loud (`grep 'GPU ERROR' logs/<id>.txt`). compute-sanitizer was too slow
to reach the fault under the GUI; the synchronous-launch + per-batch error capture + log the
chunk's program op-types is what pinned it.

---

## 5. DEAD ENDS — measured, do NOT repeat (idx8, same-puzzle)
- **Un-pack the mask** (remove 4-bit packing to kill the ~15% decode tax): would inflate
  registers/context → **worse occupancy** (the dominant lever). NCU `--set full` confirmed
  occupancy >> the decode tax. Don't.
- **interior_paint `wall`/`outside` → global cc_labels**: freed only 16 B shared (compiler
  overlays it with the spill area) and cost **-25%** (global access in the relaxation sweeps).
- **Remove `enable_smem_spilling`**: shared dropped 4208→2024 B (more blocks possible) but aps
  **-63%** — spills went to DRAM, long_scoreboard exploded. The pragma is load-bearing.
- **`__noinline__` on all ops + aggressive `__launch_bounds__`** (a *prior* session): regressed,
  but partly on a µs/ms-misread measurement — see the gpu-kernel-bottleneck memo. Re-test
  cleanly if attempting Section 6, don't trust the old verdict blindly.

**Why the occupancy lever is blocked:** the shared cap *is* the register-spill area. Freeing it
costs DRAM spills (catastrophic) — proven above. NCU's "52% occupancy upside" is an upper bound
that ignores the spill cost. The cheap occupancy is fully extracted at `(32,21)+smem_spilling`.

---

## 6. THE NEXT BIG LEVER (this is the project — the user wants it)
**Reduce `gpu_eval_program_warp`'s register pressure so the spill area shrinks.** It currently
inlines the union of ALL DSL ops → the kernel needs ~255 regs (capped to 80 by launch_bounds,
the rest spilled). Less spill ⇒ smaller shared ⇒ **more warps** (hides long_scoreboard #1) ⇒
fewer local-spill loads (#1) ⇒ and a smaller kernel helps **no_instruction (#3, 11%)** too.
One refactor hits the top 3 reasons at once. This is the only thing that moves occupancy past
the current ceiling.

Approaches, roughly increasing effort/payoff:
1. **`__noinline__` the heaviest/rarest ops** so `eval` is a lean dispatcher and each op holds
   only its own register set (not the union). Earlier attempt regressed — but on a tainted
   measurement and *before* accumulation fed the GPU. Re-test cleanly per §1. Watch: ptxas
   per-function regs/spill, and whether the dispatcher stays low-reg.
2. **Split exec and score into two kernels.** `eval` writes `output_grids` to global (already
   does), a second kernel reads them + scores. Each kernel has a smaller live-register set →
   each can hit higher occupancy independently. Cost: extra launch + global round-trip (cheap;
   the data's already there). This is the cleanest structural fix.
3. **Shrink per-op live state.** The big per-thread scratch is mostly gone (local arrays →
   `op_scratch`/in-place), but eval still carries lots of live pool pointers/indices. Auditing
   the switch for reducible live values is tedious but real.

Measurement plan for §6: pin idx8, baseline = current build, A/B each approach with `--set full`
→ read `ncu_raw.csv` for **Achieved Active Warps/SM**, **No Eligible %**, **long_scoreboard %**,
and wall-clock aps. Also run a **stub test** (`kStubGPU=true` in the worker) to confirm whether
the GPU is still the long pole vs CPU generation (stub-off aps vs stub-on aps; this session the
GPU was the long pole — kernel time, not generation).

---

## 7. Smaller / orthogonal levers (if §6 is too big right now)
- **Finish local-array → global**: `snap_to_anchors` (`label[30][30]` + `qr/qc[900]`) and the
  several `gpu_fixed_grid temp` (~904 B) locals still sit on the stack (more uncoalesced L1TEX
  local loads, NCU #2 / 45.79% upper bound). Same pattern as the `op_scratch` fix already in
  slam/skew. Lane-0 ops only ⇒ a single global scratch is race-free (dispatch is `if(lane==0)`).
- **`wait` (24%)** is partly the nibble-decode dependency chain — hard to cut without un-packing
  (which hurts occupancy). Lower priority than §6.
- **Generation/feeding**: `TARGET_JOBS=2048` is already generation-bound (the beam level runs out
  ~1800 candidates before filling more); higher buys nothing. VRAM peak was only ~1.5 GB of 6 GB.
  If §6 raises GPU throughput a lot, re-check whether generation becomes the new ceiling (stub test).

---

## 8. Tooling / diagnostics added this session (keep)
- `capture_cuda_err` (cuda_solver.cu) + `cuda_dsl_take_error` (cuda_dsl.cuh) + worker check →
  `!!! GPU ERROR` in the puzzle log on any sticky fault. **First thing to check if "magic
  madness" returns.** CUDA does NOT check errors by default; one `cudaGetLastError()` after a
  launch is a catch-all (sticky errors surface there from any of the 64 unchecked cuda calls).
- `diag_log` (global_app_state.hpp) → `logs/diag.txt` (truncated per run). Currently used by the
  GPU-error path and a leftover `[DIAG batch-stale]` line (harmless; the *gate* is the real fix —
  strip the log call if you want).
- `profile_ncu.bat`: `--set full` + dual CSV export (summary + `ncu_raw.csv` with stall reasons).
- `kStubGPU` toggle in the worker for the GPU-vs-CPU ceiling test.

---

## 9. Watch-outs / invariants
- **Objects can have bbox w/h > 30** (off-grid via Union/Shift/Move). Any op that loops obj dims
  into a fixed 30×30 buffer MUST clamp `min(dim, GPU_MAX_GRID)`. The packed-mask proxy clamps
  cell index ≥900 → 0 as a safety net. Don't remove these.
- **Lane-0-only ops** (dispatch `if(lane==0)` ~line 2380): safe to share one global scratch
  (`ctx.op_scratch`, int8[900]). Warp ops (transform/scale/tile/rotate/filter/union/interior)
  are all-lane — they use byte-striped writes / `cc_labels` etc., NOT op_scratch.
- **`ctx.cc_labels`** (int16[900]) is scorer-only, idle during op execution. **`ctx.op_scratch`**
  (int8[900]) is lane-0 op scratch. Don't alias them carelessly across overlapping lifetimes.
- Packed mask is 30×30 only; bbox>30 cells can't be represented (clamping truncates — correct,
  since the mask never held them).
- `MAX_PROGRAMS` (environment.hpp) must be ≥ `TARGET_JOBS` for active=1 to fill a full grid.

---

## 10. ARCHIVED MEMORY (folded in here because the agent-memory store is being wiped)

### 10a. Build / profiling script structure (was `arc-profiling-setup.md`)
- **`scripts/build.bat` is the single source of truth.** Usage: `build.bat <preset> [NOLAUNCH]`.
  NOLAUNCH builds without starting ARC.exe (profilers launch it themselves). All three
  profilers delegate their build via `call build.bat x64-RelWithDebInfo NOLAUNCH`:
  - **profile.bat** — Intel VTune (CPU hotspots).
  - **profile_nsys.bat** — Nsight Systems (timeline; summaries only).
  - **profile_ncu.bat** — Nsight Compute (single-kernel deep dive). **MUST stay standalone /
    VTune-free**: VTune's `vtss.sys` + NCU instrumenting `nvlddmkm.sys` = **BSOD** (the script
    detects a loaded vtss driver and aborts — reboot to clear). Defaults to **STRIDED** sampling
    (every ~1000th launch via `--kernel-id` invocation regex `.*000$`, `--launch-count 10`)
    because the UAC relaunch drops forwarded args; pass `consecutive` for back-to-back (10 after
    10000 warmup). `--set full` (scheduler + per-reason warp stalls), `--import-source yes` paired
    with `-lineinfo` (in CMakeLists CUDA RelWithDebInfo). NCU **replays each kernel ~17x** → only
    sample a handful, never all ~5000. Exports `ncu_<mode>.csv` (summary) + **`ncu_raw.csv`**
    (raw page — the only one with stall reasons). Needs Admin (HW counters); self-elevates.
- RelWithDebInfo = profiling preset; **x64-Release = normal dev build** (what the user runs).
- Backups: `C:\Users\nitro\Desktop\ARC Backups\` via `scripts/backup.py` (POST_BUILD) — snapshots
  include `current_puzzle.txt`, so you can recover which puzzle any past measurement used.

### 10b. Bottleneck history, chronological (was `gpu-kernel-bottleneck.md`)
`dsl_exec_and_score_kernel_v2` is ~100% of GPU time. RTX 4050 Laptop, 6 GB, 20 SMs. The story
evolved as instrumentation improved — later entries CORRECT earlier ones:

1. **(2026-06-03, since SUPERSEDED) "latency-bound serial launch loop."** VTune: 61% CPU in the
   CUDA runtime (worker blocked on GPU), own generation only 15.8%. NCU: SM Busy 7-11%, occ ~10%,
   Waves/SM ~1.4. Concluded GPU starved, not saturated. → led to accumulation (correct fix).
2. **Confirmed non-levers (still true):** 4-bit packing is perf-neutral on traffic (decode tax ≈
   traffic saved) but KEEP it (helps occupancy/context). Batch-cap increase alone is inert (the
   generator never fills a chunk from one `generate_branch`, capped ~8 beam entries) — the fix was
   *accumulation across pumps*, not a bigger cap.
3. **(2026-06-03) Scorer rewrite:** end-of-kernel CC segmentation was a serial lane-0 queue-BFS
   (19% of stalls, 31 idle lanes) → replaced with warp-parallel monotonic min-label propagation,
   summaries into `__shared__`. Dropped cc_visited+cc_qr+cc_qc (~4.6 KB) → single `cc_labels[900]`.
   **Still needs solve-rate verification** (same scoring-path class that broke puzzle 00d62 during packing).
4. **RESOLVED (2026-06-03) — bottleneck was block-starvation (GPU-bound).** Stub test decisive:
   `kStubGPU` (skip GPU) → ~2.6M aps vs ~80K aps real = **~32x GPU-bound, but CPU has huge feed
   headroom.** Each launch was only ~222 blocks (~1 wave/240) → GPU couldn't hide its own latency.
   Fix: accumulate candidates into a full multi-wave grid before launching. `MAX_PROGRAMS` 512→2048;
   worker sizes `chunk_cap = TARGET_JOBS(2048)/active` and re-pumps `pump_generate` until the chunk
   fills or a **beam-level boundary (must NOT cross levels)**. Result roughly tripled attempts/run.
5. **Occupancy IS the bottleneck once the grid is full (2026-06-04, +58%).** NCU `--set full`:
   #1 Theoretical Occupancy (register+smem limited → too few warps to hide L1TEX latency); #2 Long
   Scoreboard. Clean same-puzzle A/B (TARGET_JOBS=2048, 30s aps delta): `__launch_bounds__(32,12)`
   =243K, (32,16)=343K(+41%), **(32,21)=383K(+58%)**, (32,24)=374K(plateau) → settled **(32,21)**
   ≈80 regs/~44% occ. `enable_smem_spilling` makes the cap a win (spills → fast shared not DRAM).
   New limiter: **shared memory** (4208 B/block) caps blocks at ~19-23. DO NOT un-pack the mask.
6. **(2026-06-04, THIS session) occupancy ceiling proven — see §5.** The shared cap *is* the
   spill area; both ways to free it backfired (interior_paint→global -25%; remove smem_spilling
   -63%). Only unlock = reduce eval's register need (§6). Also added local-array→`op_scratch`
   coalescing (+14%, killed the uncoalesced local loads NCU flagged as 45.79% upper bound).
7. **The "madness" = kernel OOB (FIXED 2026-06-04) — see §4.** Objects bbox>30 → fixed 30×30
   array OOB → sticky CUDA fault → frozen output_grids → stale-score spam. Fixed with dim clamps +
   the permanent `capture_cuda_err` detector. Earlier theories (display leak, active-mismatch,
   re-upload race) were wrong/secondary — though the **active-mismatch gate** (`batch_stale =
   prev_batch.active_examples != active`, skip winner/update_best when stale) IS a real correctness
   fix kept in the worker (a solving candidate promotes → `active` grows → an in-flight batch packed
   against the OLD active would be indexed with the NEW active → wrong scores/false solves).
8. **TARGET_JOBS bound:** by VRAM (2048 ≈ 1.7 GB ×2 slots on 6 GB = tight; observed peak only
   ~1.5 GB though) and worker responsiveness. Generation-bound at 2048 (chunk maxes ~1800). Lives
   at `solver_executor.hpp const int TARGET_JOBS`.

### 10c. Still-open / unverified (carry forward)
- **Solve-rate verification** of the scorer rewrite + accumulation + OOB clamps — NOT done. Run
  benchmark mode and compare train/test solved vs a pre-change backup before trusting correctness.
- Finish local-array→global (`snap_to_anchors`, `gpu_fixed_grid temp`) — §7.
- Leftover `[DIAG batch-stale]` diag_log line and `kStubGPU` toggle can be cleaned up.
- Pipeline bubbles (accumulation is serial before launch; GPU idle during it + at level
  boundaries) — a 3rd slot / stream overlap is the post-occupancy lever if §6 lands.
