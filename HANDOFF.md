# Handoff, 2026-10-07

Read AGENTS.md first: the rules, how to build and check, where things are.
This file holds only what is current; past stages are in git history
(`git log`, and this file's own history).

## Where it stands
- Training: 172 right, 0 wrong (the replays made DRY and the score and
  readouts merged, not yet committed; 170 at 32d371d). Current run:
  out/merged/; out/base/ is 32d371d (out\ is not in git).
- solutions/ holds the inferred program of every solved puzzle. Left as they
  were, not solved now: the manual 05a7bcf2, 10fcaaa3, 32597951, f25fbde4,
  and 44d8ac46 (an old inferred one that no longer fits). solutions/shelf/
  holds puzzles outside training.
- A full score has taken 34 to 82 s on the same code: the laptop's clock
  moves with heat, so one run's seconds say little. Compare two builds by
  interleaving them (old, new, old, new) and taking each one's fastest run.
  The current build has not been timed that way yet; one run of it, the
  score and the readouts together, took 82 s. scripts\profile.bat profiles
  arc_said (VTune, attached after CUDA starts).
- What blocks the unsolved (`readouts.py missing merged`,
  named / the only word missing): copies 439 / 60, a recolor 398 / 120, where
  pieces went 227 / 43, turns 214 / 14, cells drawn on the ground 145 / 87,
  shape changes 116 / 8.

## Next, in the user's order
1. 18286ef8, parked (the user agreed). Sayable, checked with arc_spell (fits,
   test right): `SelectColor[9] MoveToward[14]`, then `SelectColor[6]
   Recolor[9]`, then the 5s' box `AddColor[5] Underlay`. Not found; it needs
   the move to claim the input cell its landing covers (it ties ColorFrom's
   "the 9 takes the 5 around it" at 6 cells), "6 becomes 9" sayable after the
   move (by color it is only an act, and the acts open the line), and the box
   fill asked of a line that moves pieces.
2. Speed: read the cuts and plans of one puzzle in parallel (low priority).

Smaller, open:
- Recolors are the stage after: 120 unsolved lack only a recolor, mostly a
  color taken from another piece (compositions of ColorFrom's partners,
  picks, frames). Work one puzzle at a time with the user.
- carry_the_hand cannot carry a learned table, so no word that needs the
  hand is asked after a Shape Picks Fill. The spurious tiny fills that set
  this off are gone with the at-work rule: 20 readouts named Shape Picks
  Fill, now 10, the real ones (6 of them solved); 6d75e8bb is solved by it.
  The readouts show only the elected plan, so c35c1b4c's case (under "delete
  0, fill 0") is unchecked.
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
- Every replay (`*_holds`, 7_infer.hpp) sits on one of three templates:
  paint_holds (recolors through recolors_hold, mirror, rays, trails, copies,
  repeats, the turned figure, boxes, bars, both fills), slides_hold (the
  moves) and drops_hold (removal, outline). Each is handed the pieces its
  pick takes, made once a pick by handed_pieces; replay_facts keeps none.
- At work, asked the same way of every word in every pair (replay_facts):
  at some cell it acts on, the output shows what its picked pieces did not
  show there before. A copy laid on its own likeness is still at work
  (4938f0c2's last pair is already whole).
- paint_holds owns how a paint lies: over, under (first to reach a cell
  keeps it, as Underlay) or onto, on one canvas of what is down -- the cut's
  pieces and every paint said before, laid after each is said -- and in its
  own colors, one color, or its own else one. A paint is never no color.
- Size rank and the fill by shape replay on the hand, so they are asked
  only while the hand is known (not after a word carry_the_hand cannot
  carry).
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
