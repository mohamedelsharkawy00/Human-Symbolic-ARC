Profiling

Quick Start
Run from the project root (requires Administrator for ETW tracing):
scripts\profile.bat
This will:
1. Build a RelWithDebInfo configuration
2. Launch ARC under VTune's hardware-event CPU sampler
3. Export CPU stacks, source lines, and thread data to CSV
4. Run analyze.py to produce a hotspot report

Output
All outputs are written to profiling/output/:
File: vtune_results/           - Raw VTune data (reopen in VTune GUI)
File: cpu_stacks_vtune.csv     - Function-level hotspots
File: source_lines.csv         - Source-line-level hotspots
File: threads.csv              - Per-thread CPU time
File: hotspots.txt             - Human-readable hotspot summary

Architecture (as of April 2026)
The hot path is fully on GPU (CUDA):
- CPU generates candidate programs (beam search tree expansion)
- CPU packs programs into flat op arrays (~10KB per batch)
- GPU executes all programs and scores them in one kernel launch
- CPU picks the winner from GPU results

Key areas to profile:
- GPU kernel time (dsl_exec_and_score_kernel_v2)
- CPU beam search generation (step, generate_phase_*, collect_flat_candidate_chunk)
- CUDA memcpy overhead (H2D programs, D2H scores/grids)
- Queue stalls (GPU idle waiting for CPU to generate candidates)

Standalone Analysis
Re-analyze an existing CSV without re-profiling:
python profiling\analyze.py

Opening in VTune GUI
To explore the trace interactively:
vtune-gui profiling\output\vtune_results
