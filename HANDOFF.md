# Handoff, 2026-10-06

Read AGENTS.md first: the rules, how to build and check, where things are.
This file holds only what is current; past stages are in git history
(`git log`, and this file's own history).

## Where it stands
- Training: 170 right, 0 wrong (fd4faba). Current run: out/order3-said.txt
  and out/probe/order3/ (out\ is not in git).
- solutions/ holds the inferred program of every solved puzzle. Left as they
  were, not solved now: the manual 05a7bcf2, 10fcaaa3, 32597951, f25fbde4,
  and 44d8ac46 (an old inferred one that no longer fits). solutions/shelf/
  holds puzzles outside training.
- A full score has taken 34 to 82 s on the same code: the laptop's clock
  moves with heat, so one run's seconds say little. Compare two builds by
  interleaving them (old, new, old, new) and taking each one's fastest run.
  The current build has not been timed that way yet. scripts\profile.bat profiles arc_said (VTune,
  attached after CUDA starts).
- What blocks the unsolved (`readouts.py missing order3 out/order3-said.txt`,
  named / the only word missing): copies 435 / 59, a recolor 397 / 122, where
  pieces went 225 / 42, turns 215 / 14, cells drawn on the ground 147 / 89,
  shape changes 115 / 10.

## Next, in the user's order
1. DRY: the rest of the replays (`*_holds`, 7_infer.hpp) onto templates, as
   paint_holds (rays, trails, copies, repeats, the turned figure) and
   slides_hold (align, step, edge, shift, toward) now are. About 15 left:
   color from a partner, picked recolor, removal, mirror, outline, fills,
   boxes, bridges, size rank. Open, the user decides: replay_facts keeps the
   pieces it made ready and carry_the_hand (the one door that changes them)
   wipes them, instead of every try remaking them (handed_pieces already
   makes them once a pick for the paints).
2. Merge the score and the readouts (the user: "confusing and inconsistent").
   Today arc_said (scripts/said.cpp) scores: normal build, run by hand, needs
   the GPU. arc_probe (scripts/probe.cpp) writes the readouts: built and run
   by scripts\probe.bat, no GPU, the only one the Linux cloud builds.
   scripts/readouts.py reads both. Wanted: one run that scores and writes the
   readouts, built and run one way, outputs in one place, the readout written
   by one owner; a GPU-free readout-only build must stay. Agree the name, the
   one command and where outputs go with the user first, then update
   AGENTS.md.
3. 18286ef8, parked (the user agreed). Sayable, checked with arc_spell (fits,
   test right): `SelectColor[9] MoveToward[14]`, then `SelectColor[6]
   Recolor[9]`, then the 5s' box `AddColor[5] Underlay`. Not found; it needs
   the move to claim the input cell its landing covers (it ties ColorFrom's
   "the 9 takes the 5 around it" at 6 cells), "6 becomes 9" sayable after the
   move (by color it is only an act, and the acts open the line), and the box
   fill asked of a line that moves pieces.
4. Speed: read the cuts and plans of one puzzle in parallel (low priority).

Smaller, open:
- Recolors are the stage after: 122 unsolved lack only a recolor, mostly a
  color taken from another piece (compositions of ColorFrom's partners,
  picks, frames). Work one puzzle at a time with the user.
- Under the plan "delete 0, fill 0" a spurious one-cell Shape Picks Fill is
  said before the rounds, and carry_the_hand cannot carry a learned table,
  so no word that needs the hand is asked after it (c35c1b4c showed it).
- The freedoms inside a cut are still chosen by the matching, not the words;
  do it if the data asks. FitInto (would_fit answered with the place) when a
  puzzle needs it; WouldFit's gap is "for now" (the user: not general).

## How the reading works now (decisions to keep)
- The ground is owned once (3_background.hpp): a plan deletes its ground
  first and fills last; every detector sees figures and empty cells only. On
  a same-size puzzle the fill color is ground on the inputs too
  (fill_is_ground_too); on a size change it can be a figure the rule reads
  (1b2d62fb's 0s).
- Every choice is says_the_most's: the plan, the cut, and whether the line
  opens with the matching's acts or leaves them to the words.
- The line (finish_the_line): each kind of word offers its best on the hand
  the line leaves, the biggest offer is said, then all are asked again; each
  kind says at most 8 words, since a reading lists at most 127 rules (int8_t
  indices; a 120-word line crashed the run). The paints follow the line,
  each in a sentence of its own (fired_rule::paints_after_the_line).
- A recolor or a move answers to any cell changed in place, however the
  matching read it (left_over_cell::answers_to). The cell a moved piece
  leaves may show anything but the piece itself.
- A replay checks everything its op draws: the trail redraws its body; the
  turned figure is the whole drawing, never a part (parts fit by accident).
- MoveToward[value]: value / PARTNER_KINDS is how far (a step, its own size,
  until it touches), value % PARTNER_KINDS the partner (each partner_by way,
  then each color). move_toward (1_readings.hpp) is the GPU op's and the
  replay's one arithmetic; the GPU turn and flip ops read through
  pose_source (pose_of), as the replays do.

## Kaggle and cloud
- Only one-shot inference is tested on Kaggle (ARC_INFERENCE_ONLY=1, no
  search). Submission 56790276 (notebook v37) scored 0.00; v38, with the
  loader fix, was never submitted. A local evaluation of 120 tasks scored 0.
  Source upload: `python scripts/kaggle_push.py -m "..."`; smoke run with
  `--smoke`. Scoring rules and links are in README.md.
- Linux cloud: `python3 scripts/cloud_setup.py` builds the readouts only (no
  GPU); never run in an actual Claude cloud session. A CPU executor for
  scoring there was judged not worth it (the user).
