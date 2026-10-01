# Cloud prep

Written 2026-10-01, in the first cloud session, for the user and for the next
agent. Every claim below was tried on the cloud box: Ubuntu 24.04, 4 cores,
15 GB of memory, no GPU.

## Short version

1. The cloud box has no GPU. After small repo changes the readouts run there
   (tested). Scoring with `arc_said` can't run there.
2. Decide where the right/wrong numbers come from. Recommended: the PC, driven
   from the Claude app with Remote Control.
3. Then an agent makes the repo changes, the user checks the Windows build, and
   it all goes to main.

## What the cloud box has

- **Works:** Linux with g++ 13, clang, CMake 3.28, Ninja and Python 3. Already
  installed; nothing to add.
- **Works:** the repo. Each session clones `main` from GitHub and works on its
  own `claude/...` branch.
- **Works:** the readouts (`arc_probe`). Once CMake is fixed it builds with g++
  in about 40 s. All 1000 training puzzles take about 2 min 20 s and about
  220 MB.
- **Works:** `scripts/readouts.py`, as is.
- **Works:** `git clone` of arcprize/ARC-AGI-2. This is how the puzzles can get
  in: all 1000 training and 120 evaluation tasks.
- **Missing:** GPU, CUDA, Vulkan. `arc_said` and `arc_spell` stop with "no GPU".
  The CopyTo checks also run programs on the GPU. So there are no right/wrong
  numbers in the cloud.
- **Blocked:** the GitHub zip download and the GitHub API. The network policy
  refuses them (403), so `scripts/fetch_data.py` fails as it is.
- **Fails:** CMake as it is. It stops at "Compiler requires the CUDA toolkit".
  After that come `find_package(Vulkan REQUIRED)`, and the `/analyze` flags,
  which sit outside `if(MSVC)` so g++ would get them.
- **Careful:** `.gitattributes` says `* -text`, so files are stored byte for
  byte. A `.sh` file saved with Windows line endings breaks in bash.
- **Note:** `CLAUDE_CODE_REMOTE=true` is set in every cloud session. A start-up
  script can check it and do nothing on the PC.
- **Note:** pypi.org, kaggle.com and the Ubuntu and NVIDIA package servers are
  reachable too. That only matters for option D, or for compiling the GPU half
  later.

## Decide first: where the numbers come from

The rule is numbers after every change, and the cloud box can't make them. Pick
one.

- **A. Score on the PC.** The cloud agent reads, designs and runs readouts, and
  says each change is unscored. The user pulls its branch and runs `arc_said`.
  Work: none.
- **B. Remote Control (recommended).** Run `claude remote-control` in the repo
  folder on the PC, or use the Claude Desktop app. The session shows up in the
  Claude app, so it can be driven from a phone or the web while it runs on the
  PC's GPU. Full loop, numbers included. Work: none. The PC has to stay on.
- **C. A CPU runner for programs.** Lets `arc_said` run without a GPU. The
  program runner is about 4,300 lines of GPU code where 32 lanes work together
  (43 sync points). Two runners can drift apart, and the rule is that the host
  replays the same arithmetic the GPU runs. Work: big. Not now.
- **D. Score on Kaggle's GPU.** kaggle.com is reachable from the cloud. Needs a
  new notebook that builds `arc_said` and runs it on the training set, plus a
  Kaggle token stored in the cloud environment. `kaggle_push.py` only pushes
  from a folder named `ARC`, and the cloud folder is `human-symbolic-arc`.
  Work: medium. Slow per check, and it uses Kaggle GPU hours.

The repo prep below is for sessions that run in the cloud (A, C, D). B doesn't
need it, and it does no harm there.

## On the PC (the user)

In order:

1. Push everything to main. Cloud sessions start from `main` on GitHub;
   untracked files and `out\` don't come along.
2. Hand the next section to an agent: a local one, or a cloud one (the CMake
   part is already tested on Linux).
3. Check the Windows build still works:

   ```
   scripts\build.bat x64-Release NOLAUNCH
   out\x64-Release\arc_said.exe > said.txt
   ```

   Same tally as now: 146 right, 1 wrong. CopyTo 20/20.
4. Commit and push to main.
5. Start a new cloud session and check it. The agent should find the puzzles in
   `kaggle/input` and `arc_probe` built in `out/x64-Release`.

As needed:

- **Option B:** run `claude remote-control` in the repo folder. The session
  shows up in the Claude app.
- **Option D:** store a Kaggle token in the cloud environment: the cloud
  environment menu in the session's title bar, then Edit, then an environment
  variable. Tell the agent the variable's name. Never paste the token into a
  chat.
- **Optional:** let the cloud compile the GPU half (compile only; nothing can
  run). NVIDIA's package server is reachable, so the environment's setup script
  (same menu, then Edit) can install nvcc (`cuda-nvcc-13-x`, matching the PC).
  Not tried yet.
- **Bringing cloud work home:** the cloud agent pushes to its own `claude/...`
  branch when the user says so. Then `git fetch` and merge it, or open a PR on
  GitHub, and score it on the PC.

Nothing else in the cloud settings needs to change. The network already allows
`git clone`, and the start-up hook lives in the repo.

## In the repo (an agent)

- **CMakeLists.txt builds without CUDA (tested).** Replace
  `project(ARC LANGUAGES CXX CUDA)` with:

  ```cmake
  project(ARC LANGUAGES CXX)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
  endif()
  ```

  Then move the `/analyze` `add_compile_options` into `if(MSVC)`; put
  `find_package(Vulkan REQUIRED)`, the shaders, `app.rc` and the `ARC` window
  target under `if(WIN32)`; and put `arc_gpu`, `arc_headless`, `arc_spell`,
  `arc_said` and `arc_copy_to_tests` under `if(CMAKE_CUDA_COMPILER)`. Keep
  `arc_probe` outside: it needs only `arc_core`. Nice to have: the
  `WIN32`/`UNICODE` definitions only under `if(WIN32)`, and `-Xcompiler /W4`
  only with MSVC. On the PC nothing changes: CUDA is found, so every target
  builds as today.
- **Shell scripts stay on LF.** Add `*.sh text eol=lf` to `.gitattributes`
  after `* -text`. That rule stores CRLF as is, and bash fails on CRLF.
- **`scripts/build.sh` and `scripts/probe.sh`.** Linux twins of the `.bat`
  files, using the same presets. Binaries land in `out/x64-Release` with no
  `.exe`. Call them with `bash`, since Windows doesn't record the run bit.
- **`fetch_data.py` falls back to git clone (tested).** When the zip download
  fails, clone `https://github.com/arcprize/ARC-AGI-2` with `--depth 1` into a
  temp folder and read `data/training` and `data/evaluation` from it. Take the
  stamp from `git log -1 -- data` instead of the GitHub API, which is blocked
  too.
- **A start-up hook.** Use the `session-start-hook` skill. In
  `.claude/settings.json`, SessionStart runs `bash scripts/cloud_setup.sh`. The
  script stops at once unless `CLAUDE_CODE_REMOTE` is `true`, so it does nothing
  on the PC. It fetches the puzzles if they're missing, then configures and
  builds `arc_probe` (about a minute). Give the hook a few minutes of timeout.
- **A said baseline in git.** Copy `out\probe\said_baseline.txt` to
  `baselines/said.txt`. It's small and holds no puzzles. Without it the cloud
  can't tell what's solved, and `readouts.py` needs it for `missing` and
  `explained`. Refresh it each time the PC scores.
- **AGENTS.md gets the cloud note below.**
- **HANDOFF.md says where the cloud's baselines come from:**
  `baselines/said.txt`, plus a probe of the untouched checkout made at the start
  of each session.

Leave alone: the window app, the profiling scripts and the Kaggle push stay
PC-only.

## Note for every cloud agent

Paste this into AGENTS.md as its own section.

```markdown
## In the cloud
- No GPU and no CUDA here. arc_said, arc_spell and the CopyTo checks
  can't run, and src/cuda_solver.cu can't even be compiled. So there
  are no right/wrong numbers: say a change is unscored, and the user
  scores it on their PC.
- The start-up hook fetches the puzzles and builds arc_probe. If it
  didn't, run `python3 scripts/fetch_data.py`.
- Before the first change, probe the untouched checkout:
  `bash scripts/probe.sh baseline` (about 2.5 minutes, 220 MB).
- After a change: `bash scripts/probe.sh after`, then
  `python3 scripts/readouts.py missing after baselines/said.txt` and
  `python3 scripts/readouts.py explained baseline after baselines/said.txt baselines/said.txt`.
  The second still uses the old solved list, so a puzzle the change
  newly solves shows up in it too. Flag it for the PC check; don't
  call it a loss.
- Paths use /, and binaries have no .exe.
- Never commit kaggle/input (the puzzles can't be shared) or out/.
- Each probe label takes about 220 MB. Delete old ones.
- Push only when the user says, to the session's own branch. The box
  is wiped when idle, and unpushed work is lost.
```
