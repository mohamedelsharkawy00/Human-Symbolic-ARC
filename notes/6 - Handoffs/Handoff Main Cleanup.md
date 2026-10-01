================================================================
HANDOFF: main.cpp #ifdef cleanup
================================================================
Status: PARKED. Solid base state, builds green on Windows.
        Big remaining refactor scoped but not started.

----------------------------------------------------------------
WHY THIS EXISTS
----------------------------------------------------------------
The Windows (MSVC/GUI) and Kaggle (gcc/headless) builds share one
source tree but have diverged in ugly ways: main.cpp is dotted
with #ifdef _MSC_VER blocks doing platform-detection ceremony
around code that should "just work" on both. The cleanup goal
is: the solver is the product, the GUI is an addon. main() reads
top-to-bottom without conditionals on the happy path.


----------------------------------------------------------------
ALREADY DONE (verified, building clean)
----------------------------------------------------------------
1. fmt unification.
   - Was: std::format on MSVC, fmtlib on gcc, via #ifdef
     branches in 3 headers.
   - Now: every formatting call goes through the vendored
     fmtlib (include/third_party/fmt/). No std::format
     anywhere in src/ or include/ (except inside fmt itself).
   - Call sites qualify as `fmt::format(...)` (necessary
     because `using namespace std;` + transitive <format>
     made bare `format(` ambiguous on MSVC).
   - FMT_HEADER_ONLY defined ONCE per build system:
       * MSVC: CMakeLists.txt add_compile_definitions
       * Kaggle: kaggle/src/arc_build/build.py -DFMT_HEADER_ONLY
   - /utf-8 added to MSVC CXX flags (fmt requirement).

2. Entry-point fork removed.
   - Was: #ifdef _MSC_VER -> WinMain / #else -> main
   - Now: single `int main()`. The GUI block inside main()
     fetches the Win32 module handle itself via
     GetModuleHandle(nullptr); SW_SHOW replaces the unused
     n_show_cmd.
   - CMake: target_link_options /ENTRY:mainCRTStartup so the
     windows-subsystem exe uses main as its entry symbol (no
     console window, no WinMain).

3. /analyze commented out (CMakeLists.txt).
   - Was producing noisy false positives inside fmt headers
     (C6239, C6385). Re-enable later by uncommenting the
     add_compile_options line.

4. Misc tiny cleanups.
   - Dead include comments stripped.
   - `int main(int argc, char** argv)` -> `int main()`
     (args were unused).
   - Verbose comments above guards trimmed.


----------------------------------------------------------------
WHAT THE BUILD ACTUALLY LOOKS LIKE NOW
----------------------------------------------------------------
- Windows local: scripts/build.bat
    -> sets up VsDevCmd, kills running ARC.exe, configures
       via cmake preset x64-Release (Ninja), builds, launches.
    -> Output: out/x64-Release/ARC.exe
- Kaggle: kaggle/src/arc_build/build.py
    -> globs source from /kaggle/input/**/src/main.cpp
    -> nvcc -std=c++20 -arch=sm_75 -O2 -DFMT_HEADER_ONLY ...
    -> Output: /kaggle/working/arc_solver
- Both pass through the same main() now.


----------------------------------------------------------------
REMAINING #ifdef _MSC_VER BLOCKS IN main.cpp
----------------------------------------------------------------
Each one gates code that genuinely can't compile elsewhere.
Numbered by priority for future cleanup.

A. (KEEP) enforce_memory_limit()
   - Win32 job object that caps process RAM, dev-machine
     seatbelt against runaway allocations.
   - DECISION: keep. It's load-bearing safety code, not
     ceremony. Kaggle has container-level limits.
   - Possible future polish: make it a no-op off-Windows
     so the CALL SITE loses its #ifdef (the impl keeps it).
     Not worth doing right now.

B. (KEEP / SEATBELT) GetModuleFileNameW chdir-climb
   - Walks up from the .exe path to find the project root
     (the dir containing kaggle/input) and chdir's there so
     relative paths like puzzles/training/ resolve.
   - Exists because out/x64-Release/ARC.exe is 3 levels
     deep, but data lives at project root. Double-clicking
     the exe in Explorer works because of this.
   - SUPERSEDED by the big refactor below (C). Once the
     solver locates the challenges JSON itself, this whole
     block disappears.

C. (BIG REFACTOR — PLANNED, NOT STARTED) Per-task file split
   - Today: both builds read puzzles/training/*.json (one
     file per task). Windows from CWD; Kaggle from
     /kaggle/working/puzzles/training (build.py splits the
     big challenges JSON into per-task files before run).
   - Code site: include/solver_beam.hpp ~line 467
     #ifdef _MSC_VER
         m_puzzle_files = list_puzzle_files();
     #else
         m_puzzle_files = list_puzzle_files(
             "/kaggle/working/puzzles/training");
     #endif
   - Want: solver reads the big arc-agi_*_challenges.json
     DIRECTLY, on both builds. build.py becomes pure
     compile+run (no glob, no split, no staging).
   - There's already a load_task_from_challenges() in
     include/puzzle_loader.hpp ready for this.

D. (FOLLOWS FROM C) GUI block in main.cpp
   - Window/Vulkan setup is genuinely Win32-only. Stays
     #ifdef'd. But the chdir block (B) and the file-list
     split (C) both vanish, so the file gets much cleaner
     anyway.

E. (LOW PRIORITY) Headless startup config #ifndef _MSC_VER
   - ARC_PUZZLE_LIMIT / ARC_TIMEOUT / ARC_SHARD_INDEX /
     ARC_SHARD_TOTAL env-var reads + benchmarking auto-on.
   - Could become unconditional (Windows would just ignore
     unset vars), but it pairs with the GUI toggling those
     interactively on Windows. Leave for now.

F. (LOW PRIORITY) Benchmark timestamp report
   - strftime + fmt::format guarded by #ifdef _MSC_VER
     because the GUI writes a human-readable
     benchmarks/benchmark_<ts>.txt; Kaggle uses
     submission.json. Could go either way.


----------------------------------------------------------------
THE BIG REFACTOR (C) — DESIGN, READY TO EXECUTE
----------------------------------------------------------------
Goal:
  * Solver locates arc-agi_*_challenges.json itself.
  * Loads tasks directly via load_task_from_challenges
    (already exists in include/puzzle_loader.hpp).
  * No per-task splitting in build.py. No chdir-climb in
    main.cpp. No #ifdef file-list split in solver_beam.hpp.
  * Windows double-click survives (solver finds data by
    exe-relative search, not CWD).

Discovery order in the solver (identical on both platforms):
  1. ARC_CHALLENGES_JSON env var (escape hatch / override).
  2. Recursive search under /kaggle/input
     (covers Kaggle; root is fixed, file path varies by slug).
  3. Recursive search under <exe_dir>/../../../kaggle/input
     or by walking up from <exe_dir> looking for kaggle/input
     (covers Windows double-click from any CWD).
  4. Fail loudly.

What changes:
  * include/puzzle_loader.hpp:
      - Add `find_challenges_json()` that does the search.
      - Add the loop that loads ALL tasks from the big JSON
        into the puzzle list (load_task_from_challenges
        already handles one task at a time).
  * include/solver_beam.hpp ~467:
      - Replace the #ifdef block with one unconditional call
        that uses the new big-JSON path.
  * src/main.cpp:
      - DELETE the chdir-climb #ifdef block (B above).
  * kaggle/src/arc_build/build.py:
      - DELETE the "split big JSON into per-task files"
        section (lines ~88-95).
      - DELETE the os.chdir("/kaggle/working") line if no
        longer needed (verify other writes still go to
        /kaggle/working — submission_<n>.json paths).
      - The glob for source root + the nvcc compile stays.
      - Keep EVAL_MODE flag handling.

Kaggle env-var verification — OPEN QUESTION:
  - We confirmed via web docs that Kaggle sets
    KAGGLE_KERNEL_RUN_TYPE (Interactive/Batch) and mounts
    data at fixed /kaggle/input. No Kaggle-provided env
    var points at a specific file (that's why build.py
    globs).
  - MCP tried to fetch the env_probe notebook's saved
    output but token lacks kernels.get permission.
  - If paranoid before refactoring: re-run the env probe
    at kaggle/src/diagnostics/env_probe/arc-competition.ipynb
    and check the ENVIRONMENT VARIABLES + KAGGLE INPUT
    DATA FILES sections. Not strictly needed — the design
    works on documented facts.

Risk to mind during execution:
  - Kaggle's score run is the thing that must not silently
    break. The big JSON path replaces the per-task file
    path; load_task_from_challenges exists but may not be
    exercised by the main solver loop yet. Test with
    EVAL_MODE=True to confirm scores match the old
    per-task-file path before declaring done.
  - submission write paths in main.cpp end-of-loop:
    verify they still land in /kaggle/working with the
    sharded suffix; that section may have its own #ifdef
    that the refactor doesn't touch but should sanity-check.


----------------------------------------------------------------
KEY FACTS / GOTCHAS LEARNED ALONG THE WAY
----------------------------------------------------------------
- The codebase wraps everything in `namespace arc { ... }`
  and uses `using namespace arc;` + `using namespace std;`
  in main.cpp. The `using namespace std` is what caused the
  fmt::format / std::format ambiguity. Don't remove the
  arc namespace — it's collision insurance worth keeping.

- fmt is vendored at include/third_party/fmt/ (header-only,
  fmt 11.0.2). No fmt/chrono.h is present, which is why
  the benchmark timestamp uses strftime + a plain
  fmt::format, not fmt's chrono spec.

- Kaggle data layout (verified):
    /kaggle/input/competitions/arc-prize-2026-arc-agi-2/
      contains the 6 ARC JSONs (arc-agi_*_challenges.json,
      arc-agi_*_solutions.json).
    /kaggle/input/solver-source-code/   (flat, no nesting)
      is the source dataset slug.
  Both vary by deployment quirks; build.py globs to be safe.

- `add_executable(ARC WIN32 ...)` makes a /subsystem:windows
  exe (no console). Pairs with /ENTRY:mainCRTStartup so
  plain main() works as the entry. Don't switch to console
  subsystem — would pop a console window.

- VsDevCmd.bat needs to run before any cl.exe invocation;
  build.bat handles it automatically. If invoking cmake
  directly from a fresh shell, expect "Cannot open include
  file: 'algorithm'" errors — that's the missing INCLUDE
  env, not a real code bug.

- /analyze (when re-enabled) produces 3 known false
  positives in fmt headers: C6239 in base.h, two C6385 in
  format-inl.h. Plus a real C6262 stack-size warning in
  gui.hpp(623) that pre-dates all this.

- The handoff document HANDOFF_KAGGLE_PORT.txt at project
  root has more detailed Kaggle-port history if needed.


----------------------------------------------------------------
QUICK RESUME CHECKLIST (for future-you)
----------------------------------------------------------------
[ ] Read this file top to bottom.
[ ] Open src/main.cpp and locate the remaining #ifdef
    blocks; map them to A-F above.
[ ] Decide whether to do refactor C now or keep parking.
[ ] If executing C:
      1. Add find_challenges_json() and a load-all-tasks
         function to include/puzzle_loader.hpp.
      2. Replace the #ifdef block in solver_beam.hpp ~467.
      3. Delete chdir block in main.cpp.
      4. Delete split section in build.py.
      5. Test Windows build (build.bat).
      6. Test Kaggle (push dataset, run notebook with
         EVAL_MODE=True, confirm score matches prior
         baseline).
[ ] Re-enable /analyze in CMakeLists.txt if the noise is
    gone (it should be — the fmt noise is unchanged, but
    you can suppress with #pragma push/disable around
    the fmt include if it bothers you).
