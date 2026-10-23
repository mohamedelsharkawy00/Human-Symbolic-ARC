HANDOFF - 2026-05-17  (single self-contained doc)
=================================================

This file folds in everything still relevant from the prior
HANDOFF (2026-05-14) and MEMORY_AUDIT.txt. Those two files have been
deleted - this is now the only handoff. Sections:

  PART 1  Performance ground rules
  PART 2  The bad_alloc crash: full root cause (was MEMORY_AUDIT.txt)
  PART 3  Allocation-hardening pass landed THIS session (the fix)
  PART 4  Worst-case analysis after the fix
  PART 5  Verification still owed
  PART 6  node_pool bump arena (optimization, not crash fix)
  PART 7  Secondary issues (known, not addressed)
  PART 8  STREAM A: hover preview / tile semantics / puzzle 0a938d79
  PART 9  Build + key files index


=============================================================
PART 1 - PERFORMANCE IS PARAMOUNT (read first)
=============================================================
Search/solver app. Hot paths run millions of times/sec. Every change
must be net-neutral or net-faster:
  - No per-frame allocs on the search hot path. Pre-reserve.
  - No virtual calls / std::function / type erasure in inner loops.
    Templates and direct calls only.
  - No exceptions on the hot path. bad_alloc is the one fatal
    exception (fatal condition, not control flow).
  - Benchmark mode (APS = attempts/sec) IS the regression test. If APS
    drops, the change is wrong even if the diff looks clean.
  - /OPT:ICF folds identical-codegen template instantiations in the
    PDB - the crash dump's "vector<bbox>" was actually vector<champion>
    (both 24B). Bit us once; remember when reading dumps.
If a refactor is "cleaner but slower," stop and ask. User has stated
this explicitly and repeatedly.


=============================================================
PART 2 - THE bad_alloc CRASH: FULL ROOT CAUSE
(folded from MEMORY_AUDIT.txt, 2026-05-14 investigation)
=============================================================
Repeated 0xC0000409 (uncaught std::bad_alloc -> terminate -> abort)
during benchmark runs, ~10 min in on hard puzzles.

DIAGNOSIS METHOD
  - WER local dumps for ARC.exe -> dumps/ (full memory, ~8GB each).
  - WinDbg/cdb from C:\Tools\WinDbg\ (Store-sandboxed WinDbg can't be
    driven headless; standalone binaries were extracted there).
  - RelWithDebInfo build (out/x64-RelWithDebInfo/ARC.pdb) - the Release
    build wasn't emitting PDBs; this is why benchmark/profile runs must
    use RelWithDebInfo.

FAILING STACK (ARC.exe.16836.dmp)
  abort <- terminate <- FindHandler (no catch) <- _CxxThrowException
  <- __scrt_throw_std_bad_alloc <- operator new
  <- std::vector<...>::_Emplace_reallocate
  <- arc::beam_solver::merge_slot_candidates +0x1b6
  <- arc::beam_solver::pump_generate +0x2a8 <- worker

REGISTER STATE AT THROW
  vector size      = 587,683
  new capacity     = 882,036  (1.5x growth)
  element size     = 24 B  (champion = node* + float + uint64_t)
  contiguous block requested ~= 21 MB
  default NT heap  = 386 MB committed, 17% fragmented, 336 uncommitted
                     ranges -> could not carve 21 MB.

WHAT BLEW UP / WHY
  level_candidates (vector<champion>) accumulated across ALL slots in a
  level with no size cap, hit ~588k. The vector itself (~14 MB) was the
  VICTIM. The CAUSE was node_pool: a pmr::unsynchronized_pool_resource
  (free-list, see environment.hpp:258-265) that never returns memory to
  the OS mid-run. It had grown near the 8 GB job-object cap (set in
  enforce_memory_limit(), os_utilities.hpp:94) and fragmented the
  address space, so the 21 MB vector realloc couldn't find a contiguous
  block.

CRITICAL CORRECTION ESTABLISHED 2026-05-17
  node_pool DOES fully reset between puzzles: node_pool.release() at
  solver_executor.hpp:1272 (puzzle change) and :1283 (solver reset),
  called AFTER reload_current_puzzle() clears all node*-holding
  containers. So this was a SINGLE hard puzzle ratcheting node_pool to
  multi-GB on its own, NOT a benchmark-wide cumulative leak. The fix
  therefore only had to bound the single-puzzle peak, not chase a leak.

ORIGINAL UNBOUNDED-CONTAINER CATALOG (from the audit)
  No size cap, only "reset at boundary":
    1. current_slot_candidates  vector<champion>  (per-slot)
    2. level_candidates         vector<champion>  <- BLEW UP (per-level)
    3. pending_evaluation       queue<node*>
    4. visited                  flat_hash_set_u64
    5. visited_grids            flat_hash_set_u64
  Already capped (left alone): history_pool (MAX_HISTORY_POOL=1000),
  beam (BEAM_WIDTH=1024), global_cache (MAX_CACHE_ENTRIES flush).

ALREADY-LANDED BEFORE THIS SESSION (from the audit, kept)
  - puzzle::best_tree manual release before puzzles.clear() in
    load_single_puzzle (the per-puzzle cross-boundary leak). NOTE: this
    code path was replaced this session by the m_puzzle single-object
    refactor, which handles the same release - see PART 3.
  - cleanup() in solver_beam.hpp IS called from
    reload_current_puzzle() on both change_puzzle and reset_solver
    paths (an earlier hypothesis that it was skipped was wrong).

WHY AUDIT-AND-PATCH LOSES (the audit's thesis, still true)
  Raw node* everywhere + manual environment::remove + permissive
  push_back + opt-in cleanup() => every new container is one missed
  cap/clear from the next crash. The structural fix (bounded containers
  by type + per-puzzle bump arena, dropping ref counts) makes the bug
  class impossible by construction. This session implemented the
  "bound every growing container" half by analysis (PART 3); the bump
  arena half remains open (PART 6).


=============================================================
PART 3 - ALLOCATION-HARDENING PASS LANDED THIS SESSION
=============================================================
Method: walk the program in execution order from main(), and at every
container that grows after init either bound it or confirm it's already
bounded. User's rule: everything that grows after init gets a cap + an
eviction strategy, justified by ANALYSIS in an in-code comment (entry
size x reachable count), never "discover it at runtime."

The crash is CLOSED AND EMPIRICALLY VALIDATED - a full 633-puzzle
benchmark ran to completion with zero crashes. See PART 5 for the run
data and the additional GUI / memory-limit / pinned_vector fixes that
were also in the validated binary.

INIT / LOAD PATH (pre-reserve, kill startup realloc churn)
  include/puzzle_loader.hpp
    - gpu_buffer.reserve(32 MB) at top of init_gui_buffers. Budget
      walked: ~17 MB worst case (font solid-sheets 5.5MB + bg 2.5MB +
      8MB object preview slots + puzzle data); 32MB ~2x headroom.
    - list_puzzle_files: files.reserve(2048) (ARC sets ~1147 total).
    - File read: was istreambuf_iterator (char-by-char, repeated
      realloc). Now stat size, hard cap MAX_PUZZLE_FILE_BYTES = 1 MB,
      single content.resize + read. One exact alloc, no growth.
    - parse_grid: REWRITTEN zero-alloc. Was substr() + per-cell
      std::string + stoi. Now iterates the source string in place over
      [start,end] with an integer accumulator. No heap at all. Negative
      handling more robust than the old stoi("-") path.
    - apply_solution: LEAK FIXED. Did task.best_tree = parse(...)
      without releasing a prior best_tree -> re-import / import-onto-
      solved leaked the whole prior subtree. Now removes prior first.
    - compute_distance_maps: confirmed already zero-alloc (static
      thread_local BFS queue + clear()); added a comment so nobody
      "localizes" it and reintroduces per-call alloc.
  include/gui.hpp
    - load_font_atlas: cps.reserve(128), colors.reserve(64).
  include/solver_beam.hpp
    - preview_grids.reserve(32).

m_puzzles VECTOR -> SINGLE m_puzzle (refactor)
  Was vector<puzzle> that only ever held [0] (comment literally said
  "We only hold the active puzzle"). The type lied.
    - load_single_puzzle: vector<puzzle>& -> puzzle&, returns bool.
      Releases prior best_tree, resets via out = puzzle{}.
    - beam_solver: puzzle m_puzzle; bool m_puzzle_loaded;
    - All 17 call sites converted (empty()->!m_puzzle_loaded,
      [0]->m_puzzle). Removed unused get_puzzles(). The old manual
      best_tree release loop is gone (single-object path handles it -
      this supersedes the audit's puzzles.clear() leak patch).

WORKER HOT LOOP (solver_executor.hpp, per-iteration result block)
  Per-iteration heap churn -> reused static thread_local with
  clear()/assign() at point of use. All physically bounded (proven):
    - chunk (flat_candidate_chunk) + original_nodes: bound 256
      (GPU_CHUNK_SIZE, enforced by the fill while-guard).
    - program_totals / program_all_solved: bound 2560
      (GPU_CHUNK_SIZE * MAX_TRAIN_EXAMPLES). vector<bool>->vector<char>
      (bool specialization can't be a clean reusable buffer; both read
      sites work unchanged via implicit char<->bool).
    - winner_res / winner_breakdowns: bound ~10 (MAX_TRAIN_EXAMPLES).
      winner_res.train uses assign() not resize() so trailing
      [active, train.size()) slots are freshly defaulted, not stale.

SEARCH CAPS (the actual crash fix)
  include/solver_beam.hpp
    - MAX_CANDIDATES_PER_SLOT: 1,000,000 -> 200,000. THE dominant
      live-tree term: current_slot_candidates holds every candidate of
      a slot live simultaneously (also borrowed by pending_evaluation)
      until merge clears it. Enforced via the EXISTING back-pressure
      check at solver_executor.hpp:685 - no new code path, no
      ownership risk.
    - MAX_VISITED_NODES: 10,000,000 -> 1,000,000 (~160MB->~16MB; with
      the 200k slot cap a slot can't insert near 1M distinct hashes
      anyway, so 10M was dead ceiling).
    - MAX_CACHE_ENTRIES: 10,000,000 -> 1,000,000 (~400MB->~40MB;
      cache_entry 16B padded, ~40B/entry w/ map overhead; realistic
      recurrence working set well under 1M).
  include/solver_executor.hpp
    - level_candidates: HARD CAP 200,000 at end of
      merge_slot_candidates (after the existing type-bucket dedup).
      Over cap: nth_element top-200k by score (O(n)),
      environment::remove the rest, resize. EvictLowestScore -
      semantically near-free because level_candidates is a funnel that
      distills to <=BEAM_WIDTH (1024) anyway.
    - merge_slot_candidates scratch (best_idx_by_type, kept,
      keep_mask): static thread_local + safe clear()/assign() at top.
      Bounds STRUCTURAL (best_idx_by_type keyed by uint16_t -> hard
      65,536; kept <= that; keep_mask <= level cap) so no eviction
      policy needed. level_candidates = std::move(kept) changed to
      .swap(kept) so the thread_local kept keeps its buffer (move
      would steal it -> realloc next slot). Double-free analysis done:
      post-swap kept holds already-removed/transferred pointers;
      next-call clear() only drops pointers (no remove). Safe - kept
      is touched only in this function.
    - commit_level_champion scratch (best_by_type, next_beam): same
      treatment. best_by_type structurally <=65,536 and already evicts
      within-bucket; next_beam already evicts to BEAM_WIDTH via
      partial_sort. beam = next_beam -> beam.swap(next_beam)
      (release_candidates(beam) already ran, so next_beam ends holding
      stale pointers harmlessly dropped by next clear()).

PRODUCER CHAIN - WALKED, NOTHING TO DO
  generate_branch -> process_current_phase -> execute_current_phase ->
  generate_phase_{prune,0,1}: all confirmed zero growing-after-init
  containers. Control loops over existing structures (beam,
  history_pool/history_by_type already capped) + one create_unary_node
  (one node_pool alloc) per candidate, bounded by
  MAX_CANDIDATES_PER_SLOT at the caller. No edits needed.

pending_evaluation / visited_grids: now transitively bounded - filled
in lockstep with current_slot_candidates (<=200k) / cleared per level
(<= one level's candidate count). No direct cap added; bounded by the
upstream caps above.


=============================================================
PART 4 - RESULTING WORST-CASE (BY ANALYSIS)
=============================================================
node = ~40B, avg tree = 8 nodes (~320B/tree, user-confirmed),
champion = 24B. RSS = Resident Set Size = physical RAM the process
holds; a free-list pool keeps freed pages resident so RSS tracks the
high-water (why Task Manager showed 6 GB).

  Peak simultaneously-live nodes:
    current_slot_candidates  200k * 320B   ~64 MB
    level_candidates         200k * 320B   ~64 MB
    best_by_type/next_beam   <=65,536      ~21 MB
    beam / history_pool      ~2k           ~0.6 MB
    Peak live                              ~150 MB

  node_pool high-water = peak live * pool overshoot (geometric chunk
  doubling; node is fixed-size so no size-class fragmentation) ~x2-3
                                                            ~= ~450 MB

  Non-pool resident (counts vs 8GB job cap):
    CUDA context + driver                  ~0.5-1.5 GB (fixed, DOMINANT)
    visited 1M*16B + cache 1M*40B          ~56 MB
    pinned / exec batches / gpu_buffer     ~tens of MB

  TOTAL WORST-CASE RSS ~= ~1.5 GB, dominated by the fixed CUDA context,
  NOT node_pool. Well under the 8 GB job cap. node_pool fully resets
  per puzzle via release(), so no benchmark-wide ratchet. High-water
  ratchets WITHIN a puzzle (bounded ~450 MB) and FULLY RESETS BETWEEN
  puzzles.

Conclusion: caps + the existing per-puzzle release() close the crash
by construction.


=============================================================
PART 5 - VALIDATION RESULTS + EXTRA FIXES IN THE SAME BINARY
=============================================================

FULL-RUN VALIDATION (the crash is empirically closed)
  First-ever benchmark to run to completion. Binary:
  out/x64-RelWithDebInfo/ARC.exe (built ~06:33, process started
  06:35:01, PID 23028).

    ========== BENCHMARK SUMMARY ==========
    Puzzles Run:    633
    Train Solved:   65 / 633 (10.3%)
    Test Solved:    58 / 633 (9.2%)
    Bench Time:     01:35:32
    Avg APS:        138675.61
    ========================================

  Zero crashes across all 633 puzzles / 95 minutes. Every prior run
  died with bad_alloc partway. The memory hardening (PART 3) is
  validated by construction AND empirically. Process was still alive
  afterward at ~8.3 GB commit / ~0.34 GB resident - i.e. sitting ABOVE
  the old hardcoded 8 GB cap, which under the old code was exactly the
  crash state. The new dual-limit design (below) correctly does not
  kill it because resident is tiny; the 8 GB was almost all CUDA
  committed-but-never-resident reservation.

  TIMEZONE GOTCHA (cost this session ~an hour of confusion - read it):
  The benchmark FILENAME timestamp is formatted via
  current_zone()->to_local with {:%I-%M-%S_%p} (main.cpp ~92). It is
  offset by 1 hour from the process StartTime / file mtime. The run
  named "...05-35-11_AM" was the process that started 06:35:01 wall
  clock. Do NOT use the filename time to reason about which binary
  produced a run - use the process StartTime or file mtime. The
  filename clock is an hour behind.

ADDITIONAL FIXES ALSO IN THE VALIDATED BINARY
  These landed after the PART 3 pass, prompted by two further crash
  dumps (2904, 18184) from interim manual launches, and were all in
  the 06:33 binary that completed the 633-puzzle run:

  A. GUI swapchain fix (crash dump 2904: AV in nvwgf2umx on alt-tab).
     Root cause: feeding a 0x0 / minimized-window extent into
     vkCreateSwapchainKHR (invalid imageExtent) crashed the NV driver.
     - include/vulkan_context.hpp: stores hwnd; adds minimized()
       (IsIconic + zero-client-rect), needs_recreate(), client_extent();
       recreate_swap_chain bails on zero extent; begin_frame sets a
       framebuffer_resized flag on VK_ERROR_OUT_OF_DATE_KHR; end_frame
       sets it on OUT_OF_DATE/SUBOPTIMAL from vkQueuePresentKHR.
     - src/main.cpp: while minimized -> skip render AND recreate
       entirely (DWM keeps the last frame so the taskbar hover preview
       is preserved); recreate driven by the flag using GetClientRect
       (NOT GetSystemMetrics, which returns full-screen size even when
       minimized); removed the blind needs_resize re-arm.
     Reference-correct per Vulkan WSI guidance.

  B. pinned_vector::resize null-safety (crash dump 18184: 8-byte write
     to null in pack_cuda_dsl_exec_batch, solver_executor.hpp:1011,
     worker thread - matched-PDB confirmed). Root cause: cudaMallocHost
     returns null on failure and was unchecked; old code freed the old
     buffer FIRST and set m_capacity even on failure, so one transient
     pinned-memory failure destroyed the buffer permanently and every
     later resize skipped realloc -> null deref.
     include/cuda_dsl.cuh: alloc BEFORE free; m_capacity set only on
     success; on failure print a [FATAL] stderr line and std::abort()
     (pinned exhaustion is genuinely fatal for the GPU batch - fail
     loud at the real site, not a null-deref three frames away).

  C. Dual memory-limit redesign (the 8 GB hardcoded cap was a
     COMMIT cap and false-killed on CUDA's ~5 GB committed-but-not-
     resident reservations - see machine numbers: 24 GB RAM, system
     commit 10.9/29 GB, ARC ~7.4 GB commit vs ~2.4 GB resident).
     include/os_utilities.hpp enforce_memory_limit() now sets BOTH:
       - WORKING-SET MAX (JOB_OBJECT_LIMIT_WORKINGSET, default 12 GB):
         the real "never slow my machine" control. Caps physical RAM
         ARC holds resident; on breach Windows trims ARC's own pages,
         the rest of the machine never gets starved. Env: ARC_WS_MAX_GB.
       - COMMIT CAP (JOB_OBJECT_LIMIT_PROCESS_MEMORY, default 15 GB):
         runaway guillotine only, set high so it never false-trips on
         CUDA reservations. Env: ARC_COMMIT_GB.
     Both env-overridable (0 disables that limit); signature unchanged
     so main.cpp:25 call site is untouched. WS (12) <= commit (15),
     correct ordering.

THE ONE REMAINING OPEN ITEM - SOLVE-QUALITY A/B
  Stability is DONE. The only unproven thing: whether the caps
  (MAX_CANDIDATES_PER_SLOT 1M->200k, level_candidates 200k
  EvictLowestScore) lowered the SOLVE RATE vs uncapped. We have no
  completed pre-cap baseline (every prior run crashed before
  finishing), so 65/58 (10.3%/9.2%) cannot be diffed against old
  behaviour. 10%/9% is plausible-normal for this solver on ARC-train
  and the tail shows clean 10s timeouts (not cap starvation), but it
  is UNBENCHMARKED against uncapped. This 633-run is now THE baseline
  for all future comparisons. If solve rate ever needs defending,
  raise MAX_CANDIDATES_PER_SLOT / the level cap and re-run to compare
  against these exact numbers.

OPTIONAL FOLLOW-UP CHECKS (not blocking)
  - memory_tracer -> dumps/memory_trace.csv during a run to confirm
    per-puzzle RSS plateau (analysis says ~1-2 GB; the live process
    showed 0.34 GB resident, consistent).
  - Spot-check previously-solved puzzles still solve. (0a938d79 still
    unsolved for unrelated reasons - PART 8 - expected.)
  Use RelWithDebInfo for any benchmark/profile/dump run, and ensure
  the PDB matches the exact binary (a stale PDB on a newer binary
  produces FICTIONAL stacks - this misled the analysis twice this
  session; always check `lmvm ARC` shows the matching timestamp).


=============================================================
PART 6 - NODE_POOL BUMP ARENA (OPTIMIZATION, NOT CRASH FIX)
=============================================================
Crash is closed by caps, so this is purely PERFORMANCE/cleanliness,
NOT required for stability. One remaining high-value lever. Hottest
ops run millions/sec; expected gains:
  - Alloc: free-list size-class walk -> single pointer bump (dominant;
    create_unary_node, copy_and_replace).
  - Dealloc + ref_count: gone. No per-node free-list insert, no
    inc/dec/branch on every copy/remove. ref_count (4B/node) removed.
  - Tree-walk cache locality: bump-adjacent nodes contiguous, so
    count_nodes / flatten_unary_program / update_hash /
    predict_replaced_hash become near-linear scans not pointer chases.
  - Per-puzzle teardown: O(N) -> O(1) (arena.head = base).
Not "just pointers -> prealloc": prealloc is the mechanism; the payoff
is removing free-list traversal + ref-count arithmetic + pointer-chase
misses from the three hottest ops.

CONSTRAINTS / RISKS:
  - Magnitude UNMEASURED (5%? 30%?). Land behind before/after APS. If
    APS doesn't improve, don't merge.
  - Mid-puzzle reset is NOT safe with raw node* + no generation tag
    (live pointers in beam/level_candidates -> use-after-free; the
    exact bug class PART 2 warns about). Mid-puzzle reset needs
    generation tagging + validate-before-deref. Otherwise reset only
    at the puzzle boundary (the existing release() site).
  - "Just free the free-list" is NOT possible: pmr has no trim API;
    free slots are scattered holes inside chunks pinned by live
    survivors (beam/history_pool). Can't return whole pages without
    compaction, and compaction needs no-raw-node* = the rewrite.

Recommended order:
  1. Bump arena, reset at puzzle boundary only (drop-in for the
     release() sites). Keep ref_count initially to de-risk.
  2. Once stable, delete ref_count + environment::remove site by site.
  3. Only then consider generation-tagged mid-puzzle reset IF
     profiling shows within-puzzle high-water still a problem
     (analysis says it won't be - ~450 MB).


=============================================================
PART 7 - SECONDARY ISSUES (known, NOT addressed this session)
=============================================================
Small, unrelated to bad_alloc, worth a future sweep:
  - bucket_cutoff[2][MAX_SEARCH_DEPTH+1] not zeroed in cleanup() -
    stale values bleed across puzzles. Logic bug, not memory.
  - m_solver_state.current_max_depth set during attempts
    (~solver_executor.hpp:1283 area), never reset on
    reload_current_puzzle.
  - GUI preview_cache.clear() lives in gui.hpp (~556) and only runs
    when the GUI ticks - unreliable during minimized benchmark runs.
    Should move into the solver reload path.
Left out to keep this pass focused on the allocation walk.


=============================================================
PART 8 - STREAM A: HOVER PREVIEW / TILE / PUZZLE 0a938d79
(folded from HANDOFF 2026-05-14; NOT touched this session)
=============================================================
Shares no code paths with the memory work. State as of 2026-05-14,
unchanged since:

BBOX OVERLAY (LANDED, stable)
  preview_payload carries object_bboxes parallel to object_lists.
  include/global_app_state.hpp:52-57
    struct preview_payload {
      struct bbox { int x0,y0,x1,y1,canvas_w,canvas_h; };
      vector<grid> accumulators;
      vector<vector<grid>> object_lists;
      vector<vector<bbox>> object_bboxes;
    };
  include/gui.hpp ~572/~595/~628-630: object_bboxes sized/reserved/
  pushed alongside object_lists in the
  cuda_dsl_fetch_last_object_list_scratch block.
  include/gui.hpp ~754-785: object-strip render loop draws 4 thin rect
  outlines via solid_sheets[UI_COLOR_LINE] at world-bbox coords inside
  each mini-tile. Thickness max(1, lay.scale/2), clamped to
  min(bw,bh). Lookup quirk: render path is a separate scope from the
  `payload` local alias, so it does state.preview_cache.find(h) to get
  object_bboxes - keep in mind if refactoring.

TILEX / TILEY (LANDED, current behavior)
  src/cuda_solver.cu gpu_exec_tile_warp ~1417:
   - Emits ONE output object PER source object (res->count=src_count),
     NOT a merged blob (the big 2026-05-14 change).
   - Each output spans the full canvas on the tiled axis, preserves
     the source's own bbox on the fixed axis.
   - Modulo period = the [N] annotation, NOT fw/fh (previously fw/fh,
     [N] ignored - fixed).
   - TileX: ow=ctx.output.width, oh=sh, tile (c%per) on X.
     TileY: ow=sw, oh=ctx.output.height, tile (r%per) on Y.
   - [N] range 1..30 (environment.hpp:113-114, VAR_COUNT=30,
     value=idx+1).
  Merged-bbox scratch removed; min/fw/fh kept only for the
  passthrough degenerate check (fw<=0||fh<=0).

TILE (REVERTED to original; world-position experiments backed out)
  src/cuda_solver.cu gpu_exec_tile ~1576: anchor (0,0), period from
  s.width/s.height, samples s.mask[r%s.height][c%s.width], untiled
  axis takes r/c directly. All 3 variants share:
    sr = tile_y ? (r % s.height) : r;
    sc = tile_x ? (c % s.width)  : c;
  CAVEAT (still present): for Tile[1]/Tile[2] the untiled axis indexes
  the source mask directly with r or c, so a small source object only
  paints rows [0..s.height) on that axis, not at its world position
  ("Tile anchors at x0=0,y0=0" limitation). World-position experiments
  (per-pixel period, combined-bbox period) were tried and the user
  said "scratch that, back to original" - so it was reverted.

PUZZLE 0a938d79 - STILL NOT SOLVED
  Current attempt solutions/0a938d79.txt:
    Input -> Union -> InputPalette -> RotateToPortrait -> TileX[1]
          -> UnionL -> Tile[2] -> RotateToTarget -> Union
  First training example (22x9 portrait): red(2) at (5,0), green(3) at
  (7,8). Expected output rows 5,7,9,...,21 alternate full-rows of
  red/green with one blank row between, period 4: [red,blank,green,
  blank]. Missing primitive: produce an alternating colored-line/blank
  pattern continuing across the canvas starting from arbitrarily-placed
  source pixels. None of the tile primitives express this cleanly.
  Ideas: (a) a new op taking N evenly-spaced colored lines and
  continuing the arithmetic sequence (period = 2*gap when N=2);
  (b) explicit repeated Translate + Union to lay periodic copies.
  Reference: solutions/017c7c7b.txt uses Tile[0].

SIDE ISSUES (environment, not ARC code; from 2026-05-14)
  - OBS Game Capture hook (C:\ProgramData\obs-studio-hook\
    graphics-hook64.dll) crashed ARC on alt-tab swap-chain recreate.
    Blocked by replacing that folder with a same-named file. Display/
    Window Capture still work; only Game Capture disabled. Re-enable:
    delete that file.
  - BSOD 0xD1 nvlddmkm.sys on alt-tab between ARC benchmark and
    Overwatch - NVIDIA driver bug, dual-GPU-app contention on the 6GB
    VRAM laptop. Not actionable from ARC. Don't run two GPU-heavy apps
    at once on this driver.

INFRA SET UP (keep)
  - WER local dumps for ARC.exe -> dumps/ARC.exe.<pid>.dmp (full mem,
    DumpType=2, DumpCount=10 rotating). Registry config + disable
    steps: 4 - To Do/WER Dumps Enabled.md.
  - WinDbg standalone -> C:\Tools\WinDbg\cdb.exe (headless-drivable).
  - Memory tracer: PowerShell sampler, WorkingSet/PrivateMemory @5s ->
    dumps/memory_trace.csv. Re-run during benchmark to compare slopes
    (this is the PART 5 verification tool).


=============================================================
PART 9 - BUILD + KEY FILES INDEX
=============================================================
Build: .\scripts\build.bat  (vswhere.exe warning harmless).
Use the RelWithDebInfo config for any benchmark/profile/dump work
(Release wasn't producing PDBs; out/x64-RelWithDebInfo/ARC.pdb).

Memory / hardening (this session):
  include/puzzle_loader.hpp    reserves, 1MB file cap, zero-alloc
                               parse_grid, apply_solution leak fix,
                               load_single_puzzle -> puzzle& + bool
  include/gui.hpp              load_font_atlas reserves; Stream A
                               bbox overlay (~572/595/628/754-785)
  include/solver_beam.hpp      MAX_* cap constants, m_puzzle (single),
                               cleanup(), flat_candidate_chunk
  include/solver_executor.hpp  worker hot-loop thread_locals,
                               level_candidates cap, merge/commit
                               scratch, node_pool.release (1272/1283),
                               generator chain
  include/environment.hpp      node (~40B, ref_count), node_pool (pmr
                               pool 258-265), puzzle, BEAM_WIDTH,
                               enforce_memory_limit ref
  include/os_utilities.hpp     enforce_memory_limit (8 GB job cap)

Stream A:
  include/global_app_state.hpp preview_payload struct (52-57)
  src/cuda_solver.cu           gpu_exec_tile_warp ~1417 (TileX/Y),
                               gpu_exec_tile ~1576 (Tile, reverted)
  include/environment.hpp      113-114 [N] annotation range
  solutions/0a938d79.txt       unsolved attempt
  solutions/017c7c7b.txt       uses Tile[0]

Verification / infra:
  dumps/memory_trace.csv       re-run target (PART 5)
  dumps/ARC.exe.*.dmp          original crash dumps
  4 - To Do/WER Dumps Enabled.md  dump folder + registry config
