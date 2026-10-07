CPU profiling of dataset inference and DSL replay

Run scripts\profile.bat from the project root.
It builds RelWithDebInfo, runs arc_said through the full training dataset
under the label "profile" (its readouts land in out\profile, as every
arc_said run's do), one puzzle at a time (ARC_THREADS=1, so profiles compare
with earlier ones),
with VTune user-mode CPU sampling and call stacks, then exports function,
source-line and thread reports. No Administrator prompt is needed.

Each run is saved in profiling/output/cpu-<timestamp>/:
  said.txt              emitted programs and verdicts
  score.txt             dataset score tally
  collection.log        VTune diagnostics
  vtune_results/        raw profile, including call stacks
  cpu_stacks_vtune.csv   function CPU time
  source_lines.csv      source-line CPU time
  threads.csv           per-thread CPU time
  callstacks.csv        caller stacks for the hot functions
  summary.txt           elapsed and CPU time
  hotspots.txt          summary

Re-analyze: python profiling\analyze.py profiling\output\cpu-<timestamp>
Open raw results: vtune-gui profiling\output\cpu-<timestamp>\vtune_results
The sampler attaches after CUDA startup to avoid a driver startup stall.
The score covers all puzzles; the profile excludes startup and attachment time.
Sampling adds overhead; profile elapsed time is not an unprofiled benchmark.
Functional areas are name-based summaries; use raw stacks for caller costs.

GPU profiling remains in scripts\profile_nsys.bat and profile_ncu.bat.
Those scripts still target the GUI, not the dataset runner.

Collection and finalization are separate: a shutdown control-connection error
can leave usable samples. Success requires a completed score and readable report.
