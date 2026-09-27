# 1. Context
1. High-performance symbolic C++20 solver for the ARC-AGI-2 2026 Kaggle contest.
2. Source: https://arcprize.org/competitions/2026/arc-agi-2
3. Tries to be how a human solves ARC, mostly how I solve it, ported to a computer program. Every part of it should be something a person does when they look at a puzzle, written down so a machine can do it too.
4. It stays symbolic because a clear mechanism lets you spot what to improve at first sight, unlike the black box methods that are popular.

# 2. Getting Started
1. Clone it: `git clone https://github.com/mohamedelsharkawy00/human-symbolic-arc`
2. Needs Visual Studio with the C++ tools (CMake and Ninja come with it), the CUDA toolkit, the Vulkan SDK, and Python. Built against Visual Studio 18, CUDA 13.2, Vulkan 1.4.335.
3. The repo ships no puzzles, the competition forbids redistributing them, so `scripts\fetch_data.py` rebuilds the whole set from the public ARC-AGI-2 repository into `kaggle\input\competitions\arc-prize-2026-arc-agi-2\`. The build does this for you the first time they are missing.
4. Build with `scripts\build.bat`, it sets up the MSVC developer command line and Ninja for you, so every build runs in the same environment.
5. Run it: `out\x64-Release\ARC.exe`, it walks up from itself to find the repo, so it works from anywhere.

To measure build time, run `scripts\profile_build.bat` (Release), or pass
`x64-RelWithDebInfo` or `x64-Debug`. Each run creates a fresh directory under
`out/build-profile-*`, including fresh shaders, without launching ARC. It writes
`report.txt`, `report.json`, and logs for setup, configuration, and compilation.
Compiler, linker, and backup timings are recorded separately; parallel job times
overlap and should not be added together. Installed tools, existing puzzle data,
and OS disk caches are reused. The normal post-build backup still runs. This
measures the clean-build workflow, not the existing-output rename/cleanup path.

For CUDA compiler internals, run `scripts\profile_build.bat --cuda` (CUDA 12.8+).
This adds NVCC phase timings and a device compiler timeline, summarized in
`cuda-report.txt` and `cuda-report.json`. When installed, CUDA Compile Time
Advisor also writes `cuda-advisor.txt`. Load the `timings/cuda-*.trace.json`
file in `edge://tracing` or `chrome://tracing` to explore nested compiler events.
Tracing adds overhead; compare runs using the same profiling mode. To regenerate
only the CUDA analysis, run `python scripts/summarize_cuda_build.py out/build-profile-...`.

CMake builds leave the CUDA pieces extraction API out by default, because
nothing calls it yet and compiling it costs about a third of a clean build. Turn
it back on with `cmake --preset x64-Release -DARC_ENABLE_CUDA_PIECES=ON`. Direct
NVCC builds, which define nothing, keep the API.

# 3. Layout
1. `include\core` is the solver. Give it a puzzle, call `step()` until it is solved or out of time, read the best program off the puzzle. It has no thread, window, sound or settings file of its own.
2. `include\gui` is the window: the puzzle list, pause, settings, sounds, and the pictures of what the search is doing. It never runs the solver itself (section 4).
3. The GUI depends on the solver and never the other way, and the build enforces both halves. The solver is compiled with only `include\core` on its include path, so a core file that reached into `include\gui` would not build. `ARC.exe` links the solver's headers but not its GPU code, so GUI code that reached for the GPU would not link.
4. Two programs, one solver. `ARC.exe` (`src\gui_main.cpp`) is the GUI, for analysis on Windows. `arc_headless.exe` (`src\kaggle_main.cpp`) is the solver on its own: every search the GUI shows, and the Kaggle run. Both come out of every build.
5. Headers are numbered in the order they are meant to be read, core first. The number says where a file sits in the story, not what it may include: `8_vocabulary.hpp` needs only `0_core.hpp`, but its words name facts established by `1_readings.hpp` through `7_infer.hpp`.

# 4. One puzzle, one process
1. Every search runs in a fresh `arc_headless` process, one per puzzle. A solver that has worked through one puzzle carries its memory and state into the next, and freeing it by hand never cleaned up as thoroughly as ending the process does; puzzles that ran after another went measurably worse than the same puzzle run first. The GUI creates no CUDA context at all.
2. In the GUI, a search process is started once the window has stayed on a puzzle for a moment (stepping through puzzles starts nothing) and waits paused, so SPACE starts it instantly. Pause and resume are sent to it. Another puzzle, a reset or a dataset change kills it; the next search starts clean. If the GUI dies, its search dies with it.
3. The process streams what it paints and where it stands, and the window replays it into the same view as before. Hovering a program in the footer asks the process what it paints, since only it can run one. Import is sent to it too. Its stderr goes to `logs\search_process.txt`.
4. `arc_headless --task ID --datasets MASK --timeout SECONDS --out FILE` solves one puzzle and writes what it found to `FILE`: its line of the submission and how the search went. `MASK` is the dataset bits to look `ID` up in (1 train, 2 eval, 4 test, 8 custom). `--serve` is the GUI's form: it streams on stdout and takes commands on stdin (`include\core\22_search_stream.hpp` has both). `--logs` writes `logs\<id>.txt` and the inference readout, `--verbose` puts debug lines in them too, `--max-attempts N` and `--start-examples N` are the experiment knobs.
5. Everything that runs many puzzles, the Kaggle run and the GUI's benchmark, starts one of these per puzzle. A puzzle whose process dies still gets a line in the submission, its test inputs echoed as answers, since every task must be there.

# 5. Benchmark
1. The BENCHMARK button in the GUI runs the checked datasets from their first puzzle, each for up to TIMEOUT seconds, stopping after LIMIT puzzles.
2. The search on screen is killed, and each puzzle runs in its own `arc_headless.exe`, one at a time, shown live the same way. Stepping puzzles, pause, reset and import do nothing until the run ends; afterwards a fresh search starts, paused, on the last puzzle it ran.
3. Results go to `benchmarks\benchmark_<time>.txt`, summary on top, and the predictions to `kaggle\working\submission.json`. Both are written after every puzzle, so a run stopped early keeps what it finished. Switching BENCHMARK off, or closing the app, kills the puzzle that is running.

# 6. Kaggle
1. `scripts\kaggle_push.py` uploads `include\core`, `src\kaggle_main.cpp` and `src\cuda_solver.cu` as a Kaggle dataset; nothing from `include\gui` goes.
2. The notebook in `kaggle\src\arc_build\` (generated from `build.py` by `scripts\build_to_ipynb.py`) compiles it on Kaggle's GPU and starts one `arc_headless` per GPU. Each solves its share of the puzzles, one process per puzzle, and the notebook merges their files into `submission.json`.
3. Run the same thing locally before pushing: `arc_headless.exe` with `ARC_DATASET` set to test, eval, train or custom, and optionally `ARC_TIMEOUT` (seconds per puzzle, default 120), `ARC_PUZZLE_LIMIT`, and `ARC_SHARD_INDEX` / `ARC_SHARD_TOTAL`. It writes `kaggle\working\submission.json`, as the Kaggle run does.

# 7. Logs
1. Nothing in `logs/` accumulates. Each file is the current one of its kind, cleared by whatever starts the next.

# 8. Licence
1. MIT No Attribution, see `LICENSE`. Take it and do anything, no strings.
