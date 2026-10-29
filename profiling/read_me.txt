CPU profiling of dataset inference and DSL replay

Run scripts\profile.bat [id...] from the project root.
It builds RelWithDebInfo, runs arc_headless --said through the puzzles named, or the
full training dataset where none are,
under the label "profile" (its readouts land in out\profile, as every
scored run's do), one puzzle at a time (ARC_THREADS=1, so profiles compare
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
The sampler attaches once the puzzles are loaded.
The score covers all puzzles; the profile excludes startup and attachment time.
Sampling adds overhead; profile elapsed time is not an unprofiled benchmark.
Functional areas are name-based summaries; use raw stacks for caller costs.

Collection and finalization are separate: a shutdown control-connection error
can leave usable samples. Success requires a completed score and readable report.
