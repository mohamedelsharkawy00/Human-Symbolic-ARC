# Working on this repo

## What it is
A searchless ARC-AGI-2 solver in C++20 and CUDA. Inference reads the training
pairs, names the rule as a line of words, and says it as a DSL program. The
reading matches pieces between input and output ("everything in ARC is this
matches that"), writes every change down cell by cell, and each word crosses off
the cells it explains. Whatever no word claims is listed as the word the
language lacks.

## Rules
- Inference is the target, not search. Nothing in the reading searches.
- The training set is the measure for now, not the evaluation set.
- A word is said only after it is replayed on every training pair on the host,
  with the arithmetic the GPU op runs (a shared `ARC_HOST_DEVICE` helper in
  `8b_op_math.hpp` the op calls), and seen to change something in every pair.
- New words are compositions of existing ops where possible. Add an op only when
  none exists.
- No binary ops, no hidden state in ops, prefer the DSL over preprocessing.
- You have priority over the old code: change it when it is wrong, and say so.
- Finish one stage completely and consistently before starting the next.
- Every concept has one owner. If the same handling starts spreading into more
  places, stop and move it to its owner instead of patching each place (DRY,
  separation of concerns).

## Working with the user
- Short, plain words, no jargon. They often reply "TLDR": keep answers to a few
  lines.
- Think with them first on design choices. Agreed work, just do.
- Don't ask them to commit. They say when.
- After every change, give the numbers: right, wrong, gained, lost.

## Build and check
- Build: `scripts\build.bat x64-Release NOLAUNCH`, never any other way.
- Score and readouts, one run: `scripts\said.bat [--as <label>] [id...]`
  builds arc_said and reads every training puzzle, or the ids named, into
  `out\x64-Release\`, or `out\<label>\` when a run is named to compare: `said.txt`, a line per puzzle whose program fits the
  training pairs (id, SOLVED or WRONG, program); `tally.txt`, the counts, also
  on screen; and `logs\inference\<id>.txt`, each puzzle's readout
  (background, readings, fired rules, sentences). A label is a new name,
  never a build folder (x64-Release, cloud), which readouts.py takes for
  the unnamed run; a full run writes about 225 MB,
  so delete old labels. It reads a puzzle on every CPU thread and takes about
  30 s on the 20-thread laptop (twice that when it runs hot); `ARC_THREADS=1`
  reads one at a time (the profiler does, under the label `profile`).
  `ARC_DATASET=eval` reads the evaluation set instead (training by default).
- Run a program by hand: `out\x64-Release\arc_spell.exe <id> "<program>"`
  shows what it drew beside what was wanted, pair by pair.
- Tests: none besides the score and the readouts. `tests/puzzles` holds small
  made-up puzzles to view in the app (the CUS dataset button).
- Compare two runs with `scripts\readouts.py`: `solved <before> <after>` (lost
  and gained), `explained <before> <after>` (readings that explain every cell
  yet are unsolved; must not grow), `missing <label>` (which missing words
  block the unsolved puzzles).
- A change is good when no solved puzzle is lost, the wrong count does not go
  up, and no new "explained but unsolved" appears.

## Linux cloud
No GPU there. Run from the repo root.
- Setup: `python3 scripts/cloud_setup.py` builds `out/cloud/arc_said` without
  a GPU, with a C++20 compiler (no CUDA, no Vulkan), and fetches the puzzles
  only when they are missing; `--build-only` skips that when they come another
  way.
- Readouts: `out/cloud/arc_said [--as <label>] [id...]` writes
  `out/cloud/logs/inference` (or `out/<label>/...`) and no score; no ids reads every training
  puzzle. Compare runs with `python3 scripts/readouts.py` as above.
- No score: nothing there runs the programs. Right and wrong come from the
  Windows CUDA build or a GPU runner. Say plainly which checks did not run,
  and never claim readouts prove a score.

## Where things are
- `include/core`, numbered in reading order. Inference lives in
  `3_background.hpp` (background election and the ground plan: the one owner
  of the background), `6_correspondence.hpp` (readings, fired rules, the line,
  missing words, ranking the cuts), `7_infer.hpp` (replays `*_holds` on three
  templates, `paint_holds`, `slides_hold` and `drops_hold`; picks,
  frames, `carry_the_hand`, `finish_the_line`, `says_the_most`,
  `read_figures`, `read_the_cut`, `infer`),
  `9_sentences.hpp` (the line spelled as programs, `say_the_plan`) and
  `10_log.hpp` (readouts).
- The background: `infer` asks the election for plans (delete what first,
  fill what last), reads each plan's figures with `read_figures` (no detector
  sees a background; empty cells are nothing), keeps the plan that explains
  most, and `say_the_plan` writes its delete and fill once.
- Every choice the reading makes (the ground plan, the cut, whether the
  line opens with the matching's acts or the words say every change) has one owner,
  `says_the_most`: each candidate is read to the end of its words, in the
  order given, and the one leaving the fewest cells unsaid is kept; the order
  only breaks a tie, and once one leaves none no later one is read. Each
  choice is made just above the first word that reads it.
- `include/core/8_vocabulary.hpp`: the ops. Its top comment lists the words the
  reading cannot say yet and how to unlock one.
- `include/core/8b_op_math.hpp`: what each op does to the cells, called by the
  GPU op and by its replay in `7_infer.hpp`. `1_readings.hpp` is only how to
  read a grid (histogram, boxes, objects, shapes).
- `src/cuda_solver.cu`: the GPU ops.
- `README.md`: Kaggle preparation and the private smoke run.

## Now
Read `HANDOFF.md` before working: where things stand and the next problem.
