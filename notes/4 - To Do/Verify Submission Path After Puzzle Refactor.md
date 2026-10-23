# Verify Submission Path After Puzzle Refactor

Parked 2026-07-26.

## What changed

`submission.hpp:70` calls `environment::score_tree_output`, whose signature lost
nine unpacked puzzle fields in favour of `const puzzle&`. See §3c of
`notes/1 - Overall Design/Inference Master.md`.

## Why it is still open

The GUI hover and import-solutions paths were checked and are correct. The
submission path is the third caller of the same entry point and nothing has
exercised it since the change.

## What a failure would look like

The tree runs, but a puzzle-level table reads as absent, so an op that depends on
it no-ops. The submitted grid is then a near-miss rather than a crash, which is
the failure mode that hides.

## Also unverified at the time of parking

Beam search solving end to end, and the enum renumbering: `NotInferredYet` was
added as the first case, moving Identity/Fixed/Scale/BoundingBox from 0-3 to 1-4.
The kernel compares by name now, so it should hold, but nothing has run.
A Scale puzzle solving correctly proves both at once.
