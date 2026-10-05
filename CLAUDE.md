@AGENTS.md

Read HANDOFF.md before working.

For Linux cloud sessions, run `python3 scripts/cloud_setup.py` from the repo root.
This builds `out/cloud/arc_probe` with a C++20 compiler and fetches puzzle data
only when missing. It needs neither CUDA nor Vulkan. Use `--build-only` if data
is supplied separately.

Run `out/cloud/arc_probe out/probe/cloud <puzzle-id>...` from the repo root to
write inference readouts, or omit the IDs for all training puzzles.
Use `python3 scripts/readouts.py` as described in AGENTS.md to compare readouts.

This CPU target does not execute emitted programs or measure prediction accuracy.
Full scoring still runs on the local Windows CUDA build or a GPU runner.
Report unrun checks plainly; do not claim that readouts prove a score.

Kaggle preparation and the private smoke run are documented in README.md.
