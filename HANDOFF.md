# Handoff, 2026-10-03

## Where it stands
- Training: 147 fit, 146 right, 1 wrong (44d8ac46). CopyTo 20/20. One reading
  explains every cell and is still unsolved.
- Baselines on this machine (`out\` is not in git): `out/kaggle-after-said.txt`
  (arc_said output) and `out/probe/kaggle-after/` (readouts).
- Recently: TileX and TileY unlocked, the repeat counted from the piece; a pick
  of every color but the most common; a paint laid over a piece explains the
  cells it covers; choosing the cut prefers one with at most 32 pieces per input
  (what a GPU list holds); rays and repeats laid under pieces (Underlay) or only
  onto them (MaskBy[10]).
- Still locked: the list at the top of `include/core/8_vocabulary.hpp`.

## Kaggle and cloud preparation
- Current source passed the private `mohamedelsharkawy0/arc-smoke-check` notebook
  on Kaggle's T4 / CUDA 12.8: three tasks, valid predictions, no child failures.
  This is a plumbing check, not a score or competition submission.
- Version 34 finished: 240 valid entries, zero child failures, 62 tasks fit
  their training pairs; 18,239.9 seconds solving plus 108.9 seconds compiling.
  It mistakenly used search. The user wants ONLY one-shot inference tested.
  Its files are in `out/kaggle-v34-result/`; do not treat it as an inference run.
- Version 35 was accidentally cancelled; the identical inference-only notebook
  was restarted as version 36 on 2026-10-03. Last checked: queued for L4.
  The user then requested T4: version 37 uses `NvidiaTeslaT4` and completed.
  The user reports stopping the older L4 run themselves.
  It sets `ARC_INFERENCE_ONLY=1`: each child uses `--infer-only`,
  emits programs, replays each on every training pair, accepts the first exact
  fit (the same selection as arc_said), predicts, and exits. No beam solver
  is constructed and no search loop or timeout runs in this path.
  Version 37: 93.0 seconds inference plus 108.3 seconds compilation; 240 valid
  entries, 57 training fits, zero search steps and zero child failures.
  Output is in `out/kaggle-v37-result/`.
  Submitted this exact version to `arc-prize-2026-arc-agi-2` on 2026-10-03,
  submission reference 56790276, output `submission.json`. Last checked:
  COMPLETE, public score 0.00, no error reported. Hidden run logs are not
  exposed; the zero's cause is not established.
  Claude cloud work remains deferred.
- Zero-score audit found a loader bug: training outputs before inputs were
  paired incorrectly, and spaces before field colons prevented loading.
  The loader now reads each example within its own object, independent of
  field order and whitespace. Three known solves failed with the old loader
  after harmless JSON reformatting; all three now fit in every tested layout.
  `arc_puzzle_loader_tests`: 17 checks pass; CopyTo: 20 checks pass.
  Full notebooks now reject child failures, just as smoke checks already did.
  This is a confirmed bug, not proof of the hidden zero's cause.
  Full audit: 146 right, 1 wrong, 0 gained, 0 lost. All 1000 predictions
  match the prior run with test answers entirely withheld; 147 fits, zero
  child failures, zero search. All 1000 inference summaries match; explained
  but unsolved stays 1, no new cases. Evidence: `out/submission-audit/` and
  `out/probe/submission-audit/`.
  T4 version 38 COMPLETE: 107.7s compile, 95.1s inference, 240 valid entries,
  57 fits, zero child failures/search. Predictions exactly match version 37.
  Outputs: `out/kaggle-v38-result/`. Submission was rejected: daily team
  allowance (1) already used. Kaggle said try tomorrow UTC (18 hours).
  Next: submit exact notebook version 38 after the allowance resets, then
  check its hidden score. Version 38 has NOT been submitted successfully.
- The new inference-only runner passed all 1000 training tasks locally:
  exactly the same 147 accepted programs and verdicts as arc_said (146 right,
  1 wrong, 0 gained, 0 lost), zero search steps and zero child failures.
  All 1000 submission entries validate; CopyTo 20/20. Results:
  `out/inference-train.log`, `out/inference-train-submission.json`,
  `out/inference-only-said.txt`. Use a fixed executable for these checks;
  rebuilding arc_headless while it launches children interrupts the run.
- `python scripts/kaggle_push.py --smoke` runs the separate short notebook;
  upload source first with `python scripts/kaggle_push.py -m "message"`.
  The notebook now compiles for its assigned GPU and rejects invalid output.
- Port fixes: explicit equality for `seen_act` avoids a GCC 11 compiler crash;
  shared-memory spilling is enabled only on CUDA 13+. No rule changes.
- Rechecked: 146 right, 1 wrong, 0 gained, 0 lost; CopyTo 20/20;
  explained-but-unsolved stays 1, no new cases. All 1000 inference summaries
  match before/after. Files are `out/kaggle-original-said.txt` (before),
  `out/kaggle-after-said.txt` (after), and
  `out/probe/kaggle-before/`, `out/probe/kaggle-after/` (readouts).
- Linux CPU setup: `python3 scripts/cloud_setup.py`, then
  `out/cloud/arc_probe out/probe/cloud [id...]` from the repo root.
  `CLAUDE.md` imports AGENTS.md. This builds readouts, not GPU replay or scores.
  Tested on Kaggle's CPU Linux runner (GCC 11.4): three inference summaries
  match Windows (`00576224`, `44d8ac46`, `140c817e`). The actual Claude cloud
  session has not been started; its checkout must contain the setup files.

## Next: a background that changes from pair to pair
57 training puzzles have a different background in each pair, and 55 of them
are unsolved. Example 140c817e: the background is 9, 8 and 7 in its three pairs,
and 8 in the test. Each dot grows full rays across, becomes 2, and its diagonal
neighbors become 3.

What happens now:
- The election sees it. `background_election::is_per_pair()` in
  `3_background.hpp` flags it, and `across_input_output` holds each pair's own
  background, elected from that pair's input alone
  (`confident_background(elect_side_background({ &input }))`). The readout says
  `MODE=PER-PAIR`.
- Nothing uses it. `input_elected()` returns -1, no background, so `infer()` in
  `7_infer.hpp` cuts with none: the whole background becomes one piece, the rays
  read as that piece being recolored, and no word fits.
- The GPU has one background per puzzle too. `fill_puzzle_level` in
  `13_environment.hpp` sets `ex.background_color` (and the output's) from the
  election, and the kernel copies it into `ctx.background_color` for every
  example.

Two ways to fix it. Decide with the user first.
1. A stand-in color. Pick a color no training grid or test input uses. For the
   reading, recolor each pair's own background to it, so the election finds one
   background and everything after works unchanged. To run a program, recolor
   the example the same way (input and wanted output, in `fill_example_slot`,
   `18_dsl_executor.hpp`) and the drawn grid back (`execute_flat_program`).
   Small and contained, but it is preprocessing, which the project avoids.
2. Each grid its own background. The GPU half is small: make the background per
   example in the upload (`cuda_dsl_example_data`, and the kernel's
   `tmpl.background_color`). The host half is big: the cut
   (`prepare_segmentation` in the correspondence, the cuts in `replay_facts`)
   and every rule that compares cells with `ground_in` or `ground_out` must use
   the pair's own color.

Either way, a test input's background is the same per-grid election run on it,
falling back to its most common color. Every place a program runs must agree:
`environment::execute_tree_via_flat` and `execute_flat_program` (arc_said,
arc_spell, the tests, `predict_test` in `21_submission.hpp`), and the beam
search's upload (`fill_example_slot` in `19_solver_executor.hpp`).

Keep 4c4377d9 and ea786f4a: both have a per-pair background and are solved
today.

## After that
- Recolors: 143 unsolved puzzles lack only a recolor word. Most take the color
  from another piece (a key, a marker, a partner), so they are compositions of
  existing words (ColorFrom's partners, picks, frames), not one-property rules.
- Cells drawn on the ground: 83 unsolved puzzles lack only that.
- `python scripts\readouts.py missing <label> <said>` gives the current list.
