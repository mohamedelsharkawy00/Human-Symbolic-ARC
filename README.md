# 1. Context
1. High-performance symbolic C++20 solver for the ARC-AGI-2 2026 Kaggle contest.
2. Source: https://arcprize.org/competitions/2026/arc-agi-2
3. Tries to follow the human process of solving ARC most closely, and stays symbolic because a clear mechanism lets you spot what to improve at first sight, unlike the black box methods that are popular.

# 2. Getting Started
1. Clone it: `git clone https://github.com/mohamedelsharkawy00/human-symbolic-arc`
2. Needs Visual Studio with the C++ tools (CMake and Ninja come with it), the CUDA toolkit, the Vulkan SDK, and Python. Built against Visual Studio 18, CUDA 13.2, Vulkan 1.4.335.
3. The repo ships no puzzles, the competition forbids redistributing them, so `scripts\fetch_data.py` rebuilds the whole set from the public ARC-AGI-2 repository into `kaggle\input\competitions\arc-prize-2026-arc-agi-2\`. The build does this for you the first time they are missing.
4. Build with `scripts\build.bat`, it sets up the MSVC developer command line and Ninja for you, so every build runs in the same environment.
5. Run it: `out\x64-Release\ARC.exe`, it walks up from itself to find the repo, so it works from anywhere.
6. Two builds: the GUI one above for analysis on Windows, and a headless one for Kaggle that runs the solver with no window, no renderer, and no command port.
7. The headless one is what produces a submission. `scripts\kaggle_push.py` copies the headless source up as a Kaggle dataset, then the notebook in `kaggle\src\arc_build\` compiles it on Kaggle's GPU and runs it.
8. Predictions are written to `kaggle\working\submission.json`, locally and on Kaggle alike.

# 3. Licence
1. MIT No Attribution, see `LICENSE`. Take it and do anything, no strings.
2. `include\third_party\fmt` is someone else's, MIT, and keeps its own notice.
