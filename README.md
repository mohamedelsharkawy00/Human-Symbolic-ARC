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

# 3. Build and check
1. Build: `scripts\build.bat x64-Release NOLAUNCH`, never any other way.
2. Score and readouts, one run: `scripts\said.bat [--as <label>] [--solved [label]] [id...]` builds arc_said and reads every training puzzle, or the ids named, into `out\x64-Release\`, or `out\<label>\` when a run is named to compare.
3. What it writes: `said.txt`, a line per puzzle whose program fits the training pairs (id, SOLVED or WRONG, program); `tally.txt`, the counts, also on screen; and `logs\inference\<id>.txt`, each puzzle's readout (background, readings, fired rules, sentences).
4. A label is a new name, never a build folder (x64-Release), which readouts.py takes for the unnamed run; a full run writes about 225 MB, so delete old labels.
5. It reads a puzzle on every CPU thread and takes about 9 minutes on the 20-thread laptop (558 s the last run), most of it the readings whose words may lay cells wrong (it was 255 s before them, and 30 s before the passes after the first).
6. `ARC_THREADS=1` reads one at a time (the profiler does, under the label `profile`).
7. `ARC_DATASET=eval` reads the evaluation set instead (training by default).
8. Run a program by hand: `out\x64-Release\arc_spell.exe <id> "<program>"` shows what it drew beside what was wanted, pair by pair; `--test` after the program also shows each test, whether it fits the pairs or not.
9. Tests: none besides the score and the readouts. `tests/puzzles` holds small made-up puzzles to view in the app (the CUS dataset button).

# 4. Layout
1. `include\core` is the solver. Give it a puzzle, call `step()` until it is solved or out of time, read the best program off the puzzle. It has no thread, window, sound or settings file of its own.
2. `include\gui` is the window: the puzzle list, pause, settings, sounds, and the pictures of what the search is doing. It never runs the solver itself (section 5).
3. The GUI depends on the solver and never the other way, and the build enforces both halves. The solver is compiled with only `include\core` on its include path, so a core file that reached into `include\gui` would not build. `ARC.exe` links the solver's headers but not its GPU code, so GUI code that reached for the GPU would not link.
4. Two programs, one solver. `ARC.exe` (`src\gui_main.cpp`) is the GUI, for analysis on Windows. `arc_headless.exe` (`src\kaggle_main.cpp`) is the solver on its own: every search the GUI shows, and the Kaggle run. Both come out of every build.
5. Headers are numbered in the order they are meant to be read, core first. The number says where a file sits in the story, not what it may include: `8_vocabulary.hpp` needs only `0_core.hpp`, but its words name facts established by `1_readings.hpp` through `7_infer.hpp`.

# 5. One puzzle, one process
1. Every search runs in a fresh `arc_headless` process, one per puzzle. A solver that has worked through one puzzle carries its memory and state into the next, and freeing it by hand never cleaned up as thoroughly as ending the process does; puzzles that ran after another went measurably worse than the same puzzle run first. The GUI creates no CUDA context at all.
2. In the GUI, a search process is started once the window has stayed on a puzzle for a moment (stepping through puzzles starts nothing) and waits paused, so SPACE starts it instantly. Pause and resume are sent to it. Another puzzle, a reset or a dataset change kills it; the next search starts clean. If the GUI dies, its search dies with it.
3. The process streams what it paints and where it stands, and the window replays it into the same view as before. Hovering a program in the footer asks the process what it paints, since only it can run one. Import is sent to it too. Its stderr goes to `logs\search_process.txt`.
4. `arc_headless --task ID --datasets MASK --timeout SECONDS --out FILE` solves one puzzle and writes what it found to `FILE`: its line of the submission and how the search went. `MASK` is the dataset bits to look `ID` up in (1 train, 2 eval, 4 test, 8 custom). `--serve` is the GUI's form: it streams on stdout and takes commands on stdin (`include\core\22_search_stream.hpp` has both). `--logs` writes `logs\<id>.txt` and the inference readout, `--verbose` puts debug lines in them too, `--max-attempts N` and `--start-examples N` are the experiment knobs.
5. Everything that runs many puzzles, the Kaggle run and the GUI's benchmark, starts one of these per puzzle. A puzzle whose process dies still gets a line in the submission, its test inputs echoed as answers, since every task must be there.

# 6. Benchmark
1. The BENCHMARK button in the GUI runs the checked datasets from their first puzzle, each for up to TIMEOUT seconds, stopping after LIMIT puzzles.
2. The search on screen is killed, and each puzzle runs in its own `arc_headless.exe`, one at a time, shown live the same way. Stepping puzzles, pause, reset and import do nothing until the run ends; afterwards a fresh search starts, paused, on the last puzzle it ran.
3. Results go to `benchmarks\benchmark_<time>.txt`, summary on top, and the predictions to `kaggle\working\submission.json`. Both are written after every puzzle, so a run stopped early keeps what it finished. Switching BENCHMARK off, or closing the app, kills the puzzle that is running.

# 7. Kaggle
1. `scripts\kaggle_push.py` uploads `include\core`, `src\kaggle_main.cpp`, `src\cuda_solver.cu` and `scripts\said.cpp` with `in_parallel.hpp` as a Kaggle dataset, and `commit.txt`, the commit they came from; nothing from `include\gui` goes.
2. The notebook in `kaggle\src\arc_build\` (generated from `build.py` by `scripts\build_to_ipynb.py`) compiles it on Kaggle's GPU and starts one `arc_headless` per GPU. Each solves its share of the puzzles, one process per puzzle, and the notebook merges their files into `submission.json`. Told to read training or evaluation (`ARC_DATASET`), it builds `arc_said` instead: a scored run (below).
3. Kaggle uses inference only: `ARC_INFERENCE_ONLY=1`. It emits programs, accepts the first that exactly reproduces every training output, and predicts the test outputs. No search runs. Run the same path locally with `arc_headless.exe`, `ARC_INFERENCE_ONLY=1`, and `ARC_DATASET` set to test, eval, train or custom. Optional controls are `ARC_PUZZLE_LIMIT` and `ARC_SHARD_INDEX` / `ARC_SHARD_TOTAL`. It writes `kaggle\working\submission.json`. One task can be run with `arc_headless --task ID --datasets MASK --infer-only --out FILE`. The older search path is still available when inference-only is not set.

One dataset (`solver-source-code`) and one notebook (`arc-build`) serve
everything, pushed from any folder: a run builds whatever was pushed last, and
its `commit.txt` says which commit. Before a full Kaggle run, upload the current
source and run the smoke check:

```text
python scripts/kaggle_push.py -m "submission preparation"
python scripts/kaggle_push.py --smoke
kaggle kernels status mohamedelsharkawy0/arc-build
kaggle kernels output mohamedelsharkawy0/arc-build -p out/kaggle-smoke-result
```

This builds on the assigned GPU, runs inference on three test tasks,
and checks task IDs, prediction counts and grid format. The output
is `smoke_predictions.json`, not a full submission or a score measurement.
To run the notebook as the submission, use `python scripts/kaggle_push.py --kernel`,
last before submitting: the browser shows the newest version, and a `--smoke` or
`--run` version reads what it was told. Check the accelerator and time budget
before a full run. Starting a notebook does not submit it to the competition.

A scored run reads training or evaluation on Kaggle's GPU, for a machine with
none:

```text
python scripts/kaggle_push.py --run train [id...]
kaggle kernels status mohamedelsharkawy0/arc-build
kaggle kernels output mohamedelsharkawy0/arc-build -p out/<label>
tar -xzf out/<label>/said.tar.gz -C out/<label>
```

`out\<label>\` then holds what `scripts\said.bat --as <label>` writes, and
`commit.txt`, so `scripts\readouts.py` compares it the same way.

## Kaggle output and scoring (checked 2026-10-03)

The [competition evaluation page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/evaluation)
requires `submission.json`, with every challenge task ID. Each task maps to a
list with one entry per test input, in the same order. Each entry has both
`attempt_1` and `attempt_2`, each holding a full grid. Grids are rectangular
integer arrays, colors 0-9, dimensions 1-30 (the
[ARC format](https://github.com/arcprize/ARC-AGI-2/blob/main/readme.md)).

The Kaggle score counts test outputs: an output earns one point if either
attempt exactly matches its answer, including dimensions and every cell.
Divide correct outputs by all test outputs. This differs from `arc_said`'s
whole-puzzle tally, which requires every test output in a puzzle to match.
Fitting the demonstration pairs alone earns no points.

Our writer repeats its one prediction in both attempt slots. This meets the
format but gives no second distinct guess. With no accepted program it echoes
the test input; that is a valid grid, not evidence of a solved puzzle.

The user-requested local evaluation on 2026-10-03 used commit `1dcfc8b`,
inference only, with test answers withheld until scoring: 120 tasks, 167 test
outputs, **0 right / 167 wrong (0%)**. One program fit its demonstration pairs
but failed its test; 119 tasks had no accepted program. All entries validated,
with zero child failures and zero search. Saved locally in `out/eval-one-shot/`
(`run.log`, `kaggle/working/submission.json`, `score.py`, `score.json`).
No solver rules were changed using evaluation feedback.

# 8. Logs
1. Nothing in `logs/` accumulates. Each file is the current one of its kind, cleared by whatever starts the next.

# 9. Licence
1. MIT No Attribution, see `LICENSE`. Take it and do anything, no strings.
