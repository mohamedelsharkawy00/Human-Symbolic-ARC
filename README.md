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
6. Two builds: the GUI one above for analysis on Windows, and a headless one for Kaggle that runs the solver with no window, no renderer, and no command port.
7. The headless one is what produces a submission. `scripts\kaggle_push.py` copies the headless source up as a Kaggle dataset, then the notebook in `kaggle\src\arc_build\` compiles it on Kaggle's GPU and runs it.
8. Predictions are written to `kaggle\working\submission.json`, locally and on Kaggle alike.

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

# 3. Logs
1. Nothing in `logs/` accumulates. Each file is the current one of its kind, cleared by whatever starts the next.

# 4. Licence
1. MIT No Attribution, see `LICENSE`. Take it and do anything, no strings.
