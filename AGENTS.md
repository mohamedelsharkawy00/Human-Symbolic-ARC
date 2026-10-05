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
  with the arithmetic the GPU op runs (a shared `ARC_HOST_DEVICE` helper the op
  calls), and seen to change something in every pair.
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
- Build: `scripts\build.bat x64-Release NOLAUNCH`
- Score: `out\x64-Release\arc_said.exe > said.txt` prints a line per puzzle
  whose program fits the training pairs (id, SOLVED or WRONG, program) and a
  tally on stderr. `arc_said <id>` runs one puzzle. A full run takes 2 to 6
  minutes, depending on the machine.
- Run a program by hand: `out\x64-Release\arc_spell.exe <id> "<program>"`
  shows what it drew beside what was wanted, pair by pair.
- Tests: none besides the score and the readouts. `tests/puzzles` holds small
  made-up puzzles to view in the app (the CUS dataset button).
- Readouts: `scripts\probe.bat <label> [id...]` writes the inference readout of
  each puzzle to `out\probe\<label>\logs\inference\<id>.txt`: background,
  readings, fired rules, sentences.
- Compare runs with `scripts\readouts.py`: `solved` (lost and gained),
  `explained` (readings that explain every cell yet are unsolved; must not
  grow), `missing` (which missing words block the unsolved puzzles).
- A change is good when no solved puzzle is lost, the wrong count does not go
  up, and no new "explained but unsolved" appears.

## Where things are
- `include/core`, numbered in reading order. Inference lives in
  `3_background.hpp` (background election and the ground plan: the one owner
  of the background), `6_correspondence.hpp` (readings, fired rules, the line,
  missing words, choosing the cut), `7_infer.hpp` (replays `*_holds`, picks,
  frames, `carry_the_hand`, `finish_the_line`, `read_figures`, `infer`),
  `9_sentences.hpp` (the line spelled as programs, `say_the_plan`) and
  `10_log.hpp` (readouts).
- The background: `infer` asks the election for plans (delete what first,
  fill what last), reads each plan's figures with `read_figures` (no detector
  sees a background; empty cells are nothing), keeps the plan that explains
  most, and `say_the_plan` writes its delete and fill once.
- `include/core/8_vocabulary.hpp`: the ops. Its top comment lists the words the
  reading cannot say yet and how to unlock one.
- `src/cuda_solver.cu`: the GPU ops.

## Now
Read `HANDOFF.md`: where things stand and the next problem.
