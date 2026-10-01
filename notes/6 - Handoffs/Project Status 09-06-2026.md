PROJECT STATUS HANDOFF - 2026-06-09
====================================

CURRENT CODE STATE
------------------
Restored to clean baseline backup ID 2338 (2026-06-08 10:20 PM) on the files
that had been edited during a debugging session. This baseline contains all the
real work (inference layer + color-literal pinning) and NONE of the abandoned
switch-bug-hunt edits. Files restored from 2338:
  include/solver_executor.hpp, solver_beam.hpp, global_app_state.hpp,
  cuda_dsl.cuh, environment.hpp, puzzle_loader.hpp, inference.hpp,
  src/cuda_solver.cu
Action needed: rebuild via scripts\build.bat so the binary matches this source.


WHAT WE BUILT (confirmed, in the codebase)
------------------------------------------
1. + 2. Inference layer (include/inference.hpp) — built this session.
   [CONSOLIDATED 2026-07-01: the full, newer description (directions backbone,
   channel loop, every extractor, the pinning lessons) is canonical in
   "6 - Handoffs/Inference Notes.md" (generated 2026-06-11 from the code
   comments). This doc originally duplicated an early snapshot of it, incl. the
   pre-rename 3-direction enum; read Inference Notes.md instead.]
   Kept here, this doc's unique facts: infer() runs once at puzzle load from
   puzzle_loader and is bridged onto puzzle fields; detect_size_rule was
   verified across 1147 puzzles (633 train + 514 eval), drift < 1.2% per class;
   detect_target_objects deliberately STAYS in the engine (per-output
   decomposition for the scorer, reads no direction, so not an invariant).

3. Color-literal pinning (the measured win).
   - global_app_state: atomic<bool> pin_color_literals (toggle).
   - beam_solver::color_literal_allowed(type, value): restricts AddColor /
     Recolor / RemoveColor literals to the puzzle's allowed_colors (union of all
     train colors). Applied at the candidate-generation emission sites.
   - Rationale verified on corpus: every novel test-OUTPUT color is present in
     the test INPUT (carried by a relation, not a literal) - 0 cases where a
     literal would be needed for an unseen color. So pinning literals to train
     colors is safe; do NOT use allowed_colors as a global color whitelist
     (that loses ~11% of eval - measured).
   - Measured result on 0520fde7: attempts 73,017,975 -> 21,725,445 (~3.4x
     fewer), time 2m17s -> 47s, still solves train+test.


WHAT WE TRIED FOR THE SWITCH BUG (all reverted, did NOT fix it)
---------------------------------------------------------------
The bug: switching to a puzzle in the running GUI makes it solve much slower
(e.g. 0520fde7 ~50s fresh vs >2min after a switch). A fresh process launch is
always fast. Deterministic (not random). The bug PREDATES all of today's edits.

Attempts that were tried and removed (none fixed it):
  - Reset bucket_cutoff in cleanup().
  - Reset/hoist tl_current_slot (GPU two-slot pipeline index) on switch.
  - Reorder: drain in-flight GPU batches before reload_current_puzzle.
  - "Always start fresh on switch" (clear best_tree/is_solved/active on reload).
  - cuda_dsl_reset_for_new_puzzle: zero reused exec/scratch device buffers on
    switch + zero example-template tail.
  - Deterministic tie-break (grid_hash) in the 4 beam-selection sites.
  - gpu_acquire clearing object masks (this DID fix a separate uninitialized
    GPU read, but slowed the app and did not fix the switch bug; reverted).


CONFIRMED FACTS ABOUT THE SWITCH BUG (no speculation)
-----------------------------------------------------
  - Deterministic: same fresh run is reliably fast; same switched run is
    reliably slow. Not random.
  - Only a fresh PROCESS start is fast. In-process reset does not cure it.
  - compute-sanitizer initcheck: found 100 uninitialized GLOBAL-memory reads in
    gpu_eval_program_warp. These are IDENTICAL on a fresh run vs a switched run
    (100 either way), so they are pre-existing and NOT the switch bug. They are
    a separate finding. (initcheck is blind to whatever the switch changes.)
  - Per-level beam dump diff (fast vs slow run of 0520fde7):
      level 1: 3 slots == 3 slots (identical beam)
      level 2: 3 slots == 3 slots; top champion identical
               (Input MoveRow[-1] Union, score 30.88)
      level 3: 5 slots (fast) vs 6 slots (slow)  <- first divergence
    At level 2 the beam's top entry matched, but the many score=0.00 tied
    candidates were a DIFFERENT SET between fast and slow.
  - A deterministic tie-break did NOT fix it -> the candidate SET genuinely
    differs (not merely ordering). So something in candidate GENERATION between
    level 2 and level 3 produces a different set after a switch.

RULED OUT as the cause (verified, not guessed):
  - active_example_count carryover (diag showed it resets to 1 on every switch).
  - GPU example-template staleness / d_examples_uploaded (scores identical early).
  - best_tree beam injection (disabling it changed nothing).
  - Solution-file auto-import (only happens on manual Import key, not on switch).
  - All beam_solver members are reset in cleanup() (audited the full member list).
  - Node hashing is content-based, not pointer-based (no address nondeterminism).
  - Unstable tie-breaking (the grid_hash tie-break fix did not resolve it).


SHELVED / TO DO LATER
---------------------
1. Switch bug, next concrete step:
   Capture the GENERATED candidate set (not the final beam) at the level 2 ->
   level 3 transition, fresh vs switched, and diff the sets. The first program
   present in one set but not the other identifies what generation read
   differently after a switch. Tools available and installed:
     - compute-sanitizer (CUDA) at:
       "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/compute-sanitizer.bat"
       tools used: --tool initcheck (global mem). Not yet tried: memcheck.
     - Time Travel Debugging standalone: ttd.exe on PATH
       (C:/Users/nitro/AppData/Local/Microsoft/WindowsApps/ttd.exe).
       Needs an ADMIN terminal to record. Recording a CUDA app is ~100x+ slower
       and traces are huge; record only a 2-3s window (ring buffer: ttd.exe
       -acceptEula -ring -out ttd_traces -launch out\x64-Release\ARC.exe).
   Note: the switch bug does NOT block the Kaggle path (headless loads each task
   in a fresh process), so it is a dev-GUI convenience bug, not a submission
   blocker.

2. Separate pre-existing finding to fix later (independent of switch bug):
   gpu_eval_program_warp reads uninitialized global memory (100 reads, found by
   initcheck, present on fresh runs too). gpu_acquire (src/cuda_solver.cu) resets
   only count, not the object masks of reused pool slots; readers iterate the
   full width*height box. A correct fix must clear masks for objects actually
   used WITHOUT clearing all 32 slots per acquire (that regressed APS ~100x).
   Likely cheapest: clear each object's mask when an op writes it, only for the
   used objects.

3. Inference layer next channels (the struct already has comment stubs):
   - in_to_out color: colors_consumed, colors_introduced.
   - shape / position / count channels.

4. Inference architecture (decided, not yet done):
   Goal is for the engine to read invariants from the inference object directly
   instead of via the bridge onto puzzle fields. Blocked on extracting the
   shared data types out of environment.hpp so inference is not forced to depend
   on the whole engine. Do as a dedicated, build-and-smoke-tested change.

5. propose() (candidate proposer) is intentionally NOT implemented yet. It will
   emit node* programs from the inference analysis (engine-coupled). Add it with
   the first detector.


VERIFICATION TOOLS PROVEN USEFUL THIS SESSION
---------------------------------------------
  - scripts\build.bat is the only reliable build path (sets MSVC env + Ninja).
    Invoking cmake directly from a plain shell fails (cmake not on PATH).
  - compute-sanitizer initcheck reliably finds GPU uninitialized global reads.
  - Per-level beam dump (temporary log, removed) was what localized the
    divergence to level 3. If re-added: key it on the level-commit event only,
    gate it to one puzzle id, and write to logs/ (do not tie its reset to
    cuda_dsl_invalidate_examples, which fires on promotion mid-solve).
