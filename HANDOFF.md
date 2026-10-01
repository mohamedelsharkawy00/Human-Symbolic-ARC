# Handoff, 2026-10-01

## Where it stands
- Training: 147 fit, 146 right, 1 wrong (44d8ac46). CopyTo 20/20. One reading
  explains every cell and is still unsolved.
- Baselines on this machine (`out\` is not in git): `out\probe\said_baseline.txt`
  (arc_said output) and `out\probe\baseline\` (readouts).
- Recently: TileX and TileY unlocked, the repeat counted from the piece; a pick
  of every color but the most common; a paint laid over a piece explains the
  cells it covers; choosing the cut prefers one with at most 32 pieces per input
  (what a GPU list holds); rays and repeats laid under pieces (Underlay) or only
  onto them (MaskBy[10]).
- Still locked: the list at the top of `include/core/8_vocabulary.hpp`.

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
