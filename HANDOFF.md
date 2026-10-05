# Handoff, 2026-10-05

## Completed: the ground owned once (central background, 2026-10-05)
- Agreed with the user: detectors see only objects. Ground cells are EMPTY
  (-1, nothing); empty is not a ground. Only the plan knows the background
  color: it deletes it first and fills it last. A detector that needs the
  color reads it from the plan. The user's earlier rule ("delete the ground,
  do what is needed, fill it back after"; both ways, keep whichever explains
  more) now lives in one place instead of 23 detectors and the frames.
- How it reads: `ground_plans(election)` (3_background.hpp) names the plans,
  each a pair (delete what first, fill what last); `figures_only` empties
  those cells; `read_figures` (7_infer.hpp) reads with no ground anywhere,
  and frames read their inside the same way (TRANSPARENT in and out, so no
  frame fills the ground in the middle); `infer` keeps the plan that leaves
  the fewest cells unsaid (a tie keeps the earlier plan; it stops at the
  first plan that leaves none); `say_the_plan` (9_sentences.hpp) writes the
  delete once at the start and the fill once at the end.
- Plans: 2.5 per puzzle on average. Chosen in 1000 puzzles: delete c, fill c
  765; keep everything 115; each input's own both ways 94; own, then fill a
  color 14; c, then fill another color 8; own, fill nothing 2; keep, then
  fill a new color 2. "Delete a color, fill nothing" was never chosen (0 of
  87) and is no longer offered.
- Deleted: the old modes (automatic/per_input/handed/fixed_color), the
  ground parameters of 25 detectors, ground-first and handed frames, the
  duality patch, the inference's cut/fill sources, `--central`, every
  `!is_color(ground)` bail-out, the hidden `has_output_ground()` bail-out
  (it blocked align/move/to-edge/shift/remove/outline replays), and the
  GPU's `ctx.background_color` (VoteDenoise uses `denoise_ground`; no hidden
  state). The search side (13_environment, 14_puzzle_loader, 16_solver_beam,
  the GUI) reads the one election, `inf.ground_election`.
- Empty as a value: the recolor table names empty (5_matching: slot 10 is
  empty, UNSET is -2; empty matches the ground only where the searched grid
  has empty cells, so raw grids read as before); match chance counted over
  the whole grid; VoteDenoise votes empty in its own slot (`vote_slot`);
  panes count empty lines as blank and a mask answer may leave cells empty.
  A swap may not trade a color that an earlier word paints (9565186b had
  claimed every cell while Recolor[5] then BidiColorMap[1,5] undid it).
- Numbers: 152 right, 0 wrong. Against the duality baseline (150 right, 1
  wrong): lost 0, gained 54d9e175 and be94b721; 44d8ac46, the 1 wrong, no
  longer fits. Explained-but-unsolved: 0. CopyTo 20/20; background checks
  14,000/14,000. A full score takes 581 s on this machine (plans are read one
  after another; about 5 minutes before this stage). ASan audit of the final
  reading: all 1000 puzzles, every plan, written and said, 0 errors (105
  minutes under the sanitizer). Its readouts match out/probe/cleanup/ in
  every reading; 87 differ only in the plan list (the dropped plan).
- Evidence: out/speed-said.txt (identical to out/cleanup3-said.txt),
  out/cleanup-tests.log, out/probe/cleanup/, out/final-audit.log and
  out/figures-audit/ (the audit's readouts).
- Removed afterward (user: too much; only the test puzzles stay): the ASan
  audit and the background checks (never in git), the CopyTo checks, the
  loader checks, and the truth lists (background, pane split, pane rule,
  pane programs), with their CMake targets and out\ scripts. The ones in
  git are in its history (HEAD 1a97d77). Older notes may still name them.
  What remains in tests/ is tests/puzzles (made-up puzzles for the app).
- Per-pair grounds are no longer the blocker: 57 puzzles are per-pair, 5
  solved (as before); the 52 unsolved name a recolor 28 times, where each
  copy was drawn 24, where each piece went 19.
- Open, not agreed:
  - solutions/ is not re-exported: 47 saved programs are unchanged, 103 are
    now said in other words, 54d9e175 and be94b721 have no file. Files for
    05a7bcf2, 10fcaaa3, 11852cab, 32597951, f25fbde4 (manual) and 44d8ac46
    are not inferred now.
  - f45f5ca7 was solved by an interim build, not now. The old reading never
    solved it, so it is not a loss; worth a look.
  - Speed: the plans could be read in parallel.

## Completed: the ground both ways through a frame (2026-10-04)
- User's rule: delete the ground, do what is needed, fill it back after; and
  keep both ways ("duality"), whichever explains more. Frames had broken it:
  they filled the ground back in the MIDDLE on the host, so per-pair frames
  were switched off, and a whole input was always scaled/voted with its ground.
- Whole-input frames (ScaleUp, ScaleDown, Tile[0], poses, Pattern) are also
  asked ground first: `Input Colors <ground> DeleteSelection UnionList SelectAll
  <frame>`, the figures alone framed, the ground filled only at the end by the
  existing final fill. Asked LAST in choose_piece_frame, so it wins only on a
  strictly larger share; a tie keeps the ground or the cut-out found earlier
  (asked first, it stole ties and rewrote 1cf80156, 7468f01a and 68b67ca3).
- Frames that delete the ground (cut-out, layout, fractal, ground-first) no
  longer switch off for per-pair grounds. The reading inside is HANDED each
  framed grid's ground and fill from its input (background_source::handed,
  infer's `handed` mode): no re-election from a host-filled grid, no ground
  words of its own; the outer program fills last. Fixed grounds unchanged.
- e57337a4 is now inferred: `Input Colors SelectElectedBackground
  DeleteSelection UnionList SelectAll ScaleDown[5] Colors Withhold
  CurrentObjects SelectAll Trim Union Input Colors SelectElectedBackground
  AddColor[11] Fit[1] Underlay`.
- Build passed; CopyTo 20/20, background checks 13,999/13,999: e57337a4
  end to end (including a ground no training input shows) and, on the first
  input of every training puzzle, each ground-first whole-input frame as the
  host draws it against the GPU program (7,401 new checks).
- Full score: 151 fit, 150 right, 1 wrong (44d8ac46), gained e57337a4, lost 0.
  The other 150 programs are unchanged. All 1000 readouts: explained-but-
  unsolved stays 1, none new; 8 readouts changed (e57337a4; handed cut-outs
  now hold in 20818e16, 25e02866, 7c9b52a0, 83eb0a57, c8cbb738; ground-first
  Pattern in 8e2edd66, e3f79277), all still lacking other words.
  Saved the inferred program to solutions/e57337a4.txt (156 active files).
  Evidence: out/duality-build.log, out/duality-tests.log, out/duality-said.txt,
  out/duality-score.txt, out/probe/duality/.
- OPEN, raised by the user: background handling is one central concept, yet
  it is threaded through 23 detectors (ground_in/ground_out parameters, fixed
  grounds only, so per-pair puzzles get none of them), the frames, replays and
  spelling -- and this stage added one more patch. Proposed next stage, not
  yet agreed: delete the ground ONCE from every input and output, read the
  figures-only puzzle with no notion of ground, fill ONCE at the end; read
  the untouched puzzle too and keep whichever explains more. Detectors would
  lose their ground parameters and today's frame patch would go. Needs an
  audit of how every detector treats empty cells; stage it with full checks.

## Completed: the odd pane, SelectUnique (2026-10-04)
- SelectUnique marks the one drawing in the list with no twin: same frame
  size and same cells, transparent ones included, wherever each one lies.
  Fewer than three objects, or not exactly one drawing without a twin, adds
  no mark; earlier marks stay. One ARC_HOST_DEVICE helper, unique_drawing in
  1_readings.hpp, serves both the GPU op and the reading (the pane pick's
  OddOneOut), so they cannot disagree.
- The odd pane is now said as `Input Panes SelectUnique Origin Union`: the
  whole pane, ground included, so nothing is removed or restored. That is how
  2dc579da, whose ground changes from pair to pair, is solved without any
  per-pair election. User approved the new op.
- Codex wrote and built it, then ran out of usage; Claude Code ran the checks
  and finished the stage. Codex's thread: 01a1016e-4a89-78a2-a0aa-381ce2e73a11
  under ~/.codex/sessions/2026/10/03/.
- Build passed; CopyTo 20/20, background checks 6,592/6,592 (new: SelectUnique
  cases, both odd-pane puzzles replayed, 2dc579da with a ground color it never
  shows in training).
- Full score: 150 fit, 149 right, 1 wrong (44d8ac46), gained 2dc579da and
  88a62173, lost 0. The other 148 programs are unchanged. All 1000 readouts:
  only those two changed; explained-but-unsolved stays 1, with none new.
  Saved both solutions (155 active files) and added both to
  tests/pane_program_truth.txt.
  Evidence: out/odd-pane-build.log, out/odd-pane-tests.log,
  out/odd-pane-said.txt, out/odd-pane-score.txt, out/probe/odd-pane/.

## Completed: Fit and all output-size policies (2026-10-04)
- Fit[0] stretches selected objects to input dimensions (the old default).
  Fit[1] uses the live output frame; Fit[2] uses painted output bounds and their
  offset. Sampling and target bounds are ARC_HOST_DEVICE helpers shared with
  host replay. Empty target bounds discard selected objects without selecting
  or painting surviving unselected objects.
- Identity, Fixed and Scale use Fit[1] for elected-background restoration;
  BoundingBox uses Fit[2], so filling does not expand the final crop.
  NotInferredYet records a missing frame rather than assuming input dimensions.
  output_dimension_rule::canvas_size_for is shared by host replay, the fixed
  upload context and CUDA. Invalid dimensions/ratios yield no frame, including
  zero denominators, oversize results and ratios that round to zero.
- Inference can compare local-ground readings for every inferred size policy.
  Replayed standalone explanations count as complete in that comparison;
  otherwise a partial background cut displaced 4c4377d9's complete block rule.
  Whole-input framed readings carry their elected fill through their prefix,
  targeting the outer frame. Fill is included before candidate program pricing.
- Sparse intermediate frames remain a separate boundary (see Next below).
- Build passed; CopyTo 20/20, background/size checks 6,570/6,570. Tests cover all
  size cases, nearest-neighbor grow/shrink, painted-bound offsets and empty
  paint, emitted fill against host replay, a framed fill, and every training
  pair of 140c817e, ea786f4a, 4c4377d9 and 23b5c85d.
- Full score: 148 fit, 147 right, 1 wrong (44d8ac46), gained 0, lost 0.
  All 1000 readouts checked: explained-but-unsolved remains 1, with none new.
  Refreshed accepted programs in place: 09629e4f spells Fit[0], 140c817e spells
  Fit[1]. All other accepted programs and verdicts are unchanged. 153 active
  saved solutions remain; no export backups. Whitespace check passed.
  Evidence: out/size-fit-build.log, out/size-fit-tests.log,
  out/size-fit-said.txt, out/size-fit-score.txt, out/probe/size-fit/.

## Completed: same-size per-pair inference (2026-10-04)
- background_source names a fixed color/no ground or an input-local election.
  Correspondence resolves it per input and keeps that color in each prepared
  pair; replay_facts carries each pair's output ground for every general word.
- infer compares the usual reading with an input-local-ground reading on
  same-size PER-PAIR puzzles. It chooses the latter only if it leaves strictly
  fewer cells unexplained. A tie keeps the existing all-color explanation.
  The alternative uses the general cut/word replay; older detectors that own
  fixed-color cuts remain in the usual reading, with their matching emitters.
- The chosen cut emits Colors SelectElectedBackground DeleteSelection before
  segmentation. Final restoration is Input Colors SelectElectedBackground
  AddColor[11] Underlay. Neither selection nor fill consults training at runtime.
- 140c817e now fits training and passes its test. An extra replay replaces its
  input/output ground with color 6, absent from its training inputs, and passes
  with the same program. ea786f4a and 4c4377d9 retain their programs.
- Resized puzzles, including their inner framed readings, retain the old route.
  Frame composition splices the inner cut but not its fill; enabling dynamic
  ground there without explicit restoration lost 23b5c85d. The boundary is now
  enforced by background_reading::fixed_color and covered by a regression check.
- Build passed; CopyTo 20/20 and background/inference contracts 6,507/6,507.
  Full score: 148 fit, 147 right, 1 wrong, gained 140c817e, lost none.
  All 1000 readouts checked: only the 30 chosen per-pair readings changed.
  Explained-but-unsolved stays 1, with no new cases.
  All 147 previous accepted programs/verdicts are unchanged. Saved the new
  solution to solutions/140c817e.txt (153 active files, no export backups).
  Evidence: out/per-pair-build.log, out/per-pair-tests.log,
  out/per-pair-said.txt, out/per-pair-score.txt, out/probe/per-pair/.

## Completed: shared election and SelectElectedBackground (2026-10-04)
- The selector elects from the current object layout, preserving its frame.
  It adds whole-object marks like SelectColor; abstention leaves marks alone.
  Explicit removal: Input Colors SelectElectedBackground DeleteSelection.
  A selected color layer can fill its frame with AddColor[11] then Underlay.
  No training pair, input snapshot, or puzzle background field is read by it.
- 3_background.hpp now shares ARC_HOST_DEVICE measurement, candidate scoring,
  ranking and confidence between host and GPU. Object/contour scratch is
  bounded by the 900-cell grid, with no device allocation. Across-pair choices
  remain on the host. Host per-pair election calls elect_grid_background.
- Inference does NOT emit the new selector yet. Per-pair cuts and explicit
  restoration are the next stage; do not remove ground globally just because
  the puzzle is PER-PAIR. Preserve 4c4377d9 and ea786f4a.
- Full score: 146 right, 1 wrong, 0 gained, 0 lost. All 147 accepted programs
  and verdicts, and all 1000 readouts, match the relative-AddColor baseline.
  Explained-but-unsolved remains 1, with no new cases.
- Build passed; CopyTo 20/20 and background/election checks 6,499/6,499.
- Corpus contracts compare the bounded election to the pooled host path and
  replay the GPU selector on all 3,232 training inputs. Also check abstention,
  additive marks, whole-object selection, changed paint, transparent margins,
  translated frames, deletion and relative fill.
- Evidence: out/elected-build.log, out/elected-tests.log,
  out/elected-said.txt, out/elected-score.txt, out/probe/elected-background/.

## Completed extension: relative AddColor (2026-10-04)
- User approved giving AddColor the same color arguments as Recolor:
  0..9 are literals, 10 least common, 11 most common, measured separately
  on each selected object. Ties pick the lowest color; an empty object has
  no relative color and stays empty. Resolve before changing any cells.
- The GPU uses gpu_obj_resolve_color, which calls the existing shared
  ARC_HOST_DEVICE object_paint_color helper. No second color-choice recipe.
- This is vocabulary support only. No emitter uses relative AddColor yet;
  per-pair background reading and SelectElectedBackground remain next.
- Build passed; CopyTo 20/20 and background/fill contracts 22/22. Full score:
  146 right, 1 wrong, 0 gained, 0 lost. All 147 programs and verdicts exactly
  match the explicit-background baseline. No solution files needed rewriting.
  Evidence: out/addcolor-build.log, out/addcolor-tests.log, out/addcolor-said.txt.
  All 1000 readouts checked in out/probe/addcolor-relative: explained-but-unsolved
  stays 1, no new cases.

## Completed: explicit background steps (2026-10-03)
- User chose consistency over search depth. The existing programs have been
  rewritten without changing their drawings. Per-pair election is next.
- Colors and Split now keep all real colors. Explicit `Colors SelectColor[c]
  DeleteSelection` removes a named ground; `UnionList` is used when the next
  operation needs the single framed input instead of separate color layers.
- InputWithoutBackground is retired: it fused a source and background removal.
- Output now preserves the whole canvas frame, including transparent margins
  and an entirely empty canvas. User explicitly approved this change.
- The kernel no longer fills output gaps automatically. Programs finish with
  `Output SelectAll AddColor[c] Underlay` when an output ground is elected.
  BoundingBox programs insert Trim before AddColor, so the fill stays inside
  the drawn bounds and cannot enlarge the final crop. No new operation needed.
- Emitter cleanup removes SelectAll immediately before Colors/Split and repeated
  ground removal only while the written operations prove that color absent.
  Recolor or any unrecognised act drops that proof; no runtime state is added.
- Background election itself has NOT changed. Fixed literals preserve the old
  readings; per-pair selection is the next stage. Keep ea786f4a's ground:
  its existing correct program deliberately uses a cut that keeps all colors.
- The fresh pre-change export is `out/solutions-export/said.txt`; its 147
  programs fit training, 146 right / 1 wrong. The user requested overwrite,
  not export backups. The 35 temporary export backups were deleted.
- A separate pre-existing full-build snapshot hook remains in scripts/backup.py.

- Verified full training run after the rewrite: **146 right, 1 wrong, 0 gained,
  0 lost** (147 fits). CopyTo 20/20; explicit-background contracts 16/16.
  All 1000 readouts checked: explained-but-unsolved stays 1, no new cases.
  Every accepted training drawing is still exact. All 146 correct test results
  are unchanged; 44d8ac46's wrong test prediction was also compared byte-for-byte
  with the old submission and is unchanged.
- Current score/programs: `out/explicit-background-said.txt`; checks:
  `out/explicit-background-tests.log`; readouts: `out/probe/explicit-background/`.
  `arc_background_tests` was the focused GPU contract target (Colors/Split keep
  ground, explicit delete/fill, Output frame, bbox fill, safe emitter cleanup);
  removed 2026-10-05.
- All 147 inferred programs overwrite `solutions/<id>.txt`. Five older manual
  training solutions also migrated and replayed successfully; their results
  are in `out/explicit-background-manual/`. User approved moving dbff022c, which
  is absent from training, into `solutions/shelf/`. Active folder: 152 files.
  The 35 pane reference programs in tests/pane_program_truth.txt were refreshed.

## Where it stands
- Training: 152 fit, 152 right, 0 wrong. No reading explains every cell and
  is still unsolved. Checks: the score (arc_said) and the readouts; no test
  programs remain.
- Current baseline (`out\` is not in git): `out/speed-said.txt` and
  `out/probe/cleanup/`. Before the central background: `out/duality-said.txt`
  and `out/probe/duality/`. Before the explicit-background rewrite:
  `out/solutions-export/said.txt` and `out/probe/kaggle-after/`.
- What blocks the unsolved (`readouts.py missing cleanup out/speed-said.txt`,
  named / the only word missing): a recolor 419 / 141; where each copy was
  drawn 318 / 44; where each piece went 203 / 38; the cells drawn on the
  ground 185 / 88; how each piece turned 171 / 14; how a piece changed its
  shape 109 / 12.
- Before October 3: TileX and TileY unlocked, the repeat counted from the piece; a pick
  of every color but the most common; a paint laid over a piece explains the
  cells it covers; choosing the cut prefers one with at most 32 pieces per input
  (what a GPU list holds); rays and repeats laid under pieces (Underlay) or only
  onto them (MaskBy[10]).
- Still locked: the list at the top of `include/core/8_vocabulary.hpp`.

## Decisions and open questions from 2026-10-03
- Today's code/preparation changes were pushed to main as `1dcfc8b`; Kaggle
  scoring rules and the local evaluation baseline followed as `f6d59cf`.
  The 240 visible test tasks were checked: all are identical to training
  tasks. The 57 visible fits are not a measure of unseen-task performance.
- The next proposed solver work is the per-pair background problem below.
  The explicit-ops decisions above supersede the earlier proposal. Keep
  training as the development measure; today's evaluation was explicitly requested by the
  user, not a change to the standing rule or permission to tune on evaluation.
- Claude cloud preparation is in main, but an actual Claude cloud session
  has not been configured or verified. Connect this repo on main and set its
  environment setup script to `python3 scripts/cloud_setup.py`. It needs a
  C++20 compiler and network access to fetch public puzzle data when absent.
  Ready scope: CPU inference edits and readouts. Program execution, accuracy
  scoring, and CopyTo still need the local GPU or Kaggle.
- We considered a CPU executor to make a $100 cloud-credit offer more useful.
  The user decided it is not worth doing now. No CPU executor/prototype was
  implemented; do not start one as an assumed prerequisite for cloud work.
  Correction to the initial assessment: substantial CPU/GPU logic IS shared
  through ARC_HOST_DEVICE helpers in `1_readings.hpp` and `7_infer.hpp`:
  segmentation, matching partners, placement, rays, repeats, snapping, etc.
  The missing layer is full program execution: object selection/list state,
  canvas composition, and CPU equivalents for CUDA-specific parallel loops.
  `17_fixed_dsl.hpp` prepares data, while `18_dsl_executor.hpp` invokes CUDA.
  The interpreter mentions 76 operation types; accepted training programs
  use 67. These counts do NOT imply 67 separate algorithm rewrites.
  The quoted 2-4 working days / 2-4 hour prototype were rough estimates only,
  not measured work or a reliable dollar estimate.
- Latest user question before this handoff update: what the "project
  coordinator limit" in Claude resets. We verified what a coordinator does
  (directs worker threads), but NOT that specific limit's meaning or reset
  behavior. Asked for its exact wording/screenshot and reset tooltip; none
  received yet. Do not claim an ordinary usage reset clears every limit.
  Official reset guidance says an offer restores the session OR weekly
  allowance it specifies, not credit balances. The user's $100 offer terms
  and eligibility are also unverified. References:
  https://support.claude.com/en/articles/17007452-what-is-a-limit-reset
  https://claude.com/blog/projects-redesigned
  https://code.claude.com/docs/en/cloud-environments#setup-scripts
- No solver changes followed the completed audit/evaluation. Last verified
  training counts remain 146 right, 1 wrong, 0 gained, 0 lost; CopyTo 20/20,
  loader checks 17/17, explained-but-unsolved 1. Later work was analysis/docs.

## Kaggle and cloud preparation
- User requested a local evaluation and a fresh check of Kaggle's output rules
  on 2026-10-03. Commit `1dcfc8b`, inference only: **0/167 test outputs**,
  0/120 whole tasks. One task fit its demonstration pairs but failed its test;
  119 had no accepted program. All 120 entries valid, zero process failures,
  zero search, test answers withheld from the solver until external scoring.
  Evidence: `out/eval-one-shot/score.json`, `score.py`, `run.log`, and
  `kaggle/working/submission.json` inside that directory. This is the first
  recorded evaluation baseline here; gained/lost are not available.
  Training remains 146 right, 1 wrong, 0 gained, 0 lost; no solver edits.
  This supports a coverage/generalization problem, although the hidden zero's
  exact cause remains unverified. Fixing the JSON loader did not fix local eval.
  Kaggle counts exact matches per test OUTPUT, accepting either of two attempts;
  `arc_said` counts whole tasks. Both attempt keys, every ID, and ordered test
  entries are required. Our identical attempts are valid but offer no second
  guess. Official links and details are now in README.md's scoring section.
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
  allowance (1) already used. On October 3 Kaggle said to retry October 4 UTC
  (about 18 hours at that time; not a live countdown).
  Pending: exact notebook version 38 may be submitted after reset to test the
  loader fix on hidden data. It has NOT been submitted successfully. The later
  local evaluation still scored zero, so do not promise this retry improves
  the score; the proposed development priority is solver coverage below.
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

## Next: faithful intermediate frames
SUPERSEDED (2026-10-05) by the central background: frames now read with no
ground, so none fills the ground in the middle and nothing elects from a
host-filled image. Kept below for its history and its regressions.

Every output-size policy now has an explicit final-fill composition. A remaining
boundary is intermediate ground: cut-out, laid-out and fractal host frames can
restore ground that their DSL object stream does not contain. Their inner
readings stay fixed-color, and per-grid outer cuts decline those frames. Before
opening that route, make the intermediate fill an explicit composition and
replay the exact stream the next election sees. Do not elect from a host-only
filled image. Keep 23b5c85d, 4c4377d9 and ea786f4a as regressions.

Do not add another election, hidden fill, or removal mode to segmentation ops.
Continue to read current objects, use whole-object selection, and keep fills,
cuts and frame changes explicit in the program.

Agreed order (2026-10-04): finish this per-pair stage first, then recolors.
Work one example puzzle at a time with the user, as with 2dc579da, and first
try keeping the whole piece with its ground before removing and restoring it.
57 training puzzles are PER-PAIR: 5 solved, 52 not; 20 of those change size.
Most also lack other words (a recolor, where pieces went), so per-pair ground
alone will not unlock them all. Since the duality stage, cut-out, layout and
fractal frames run on per-pair grounds with the ground handed down, and no
frame fills ground in the middle; the central-background proposal above may
replace that plumbing.
Update 2026-10-05: the central background replaced it. Each input's own
ground is now a plan like any other, and the 52 unsolved per-pair puzzles
lack other words (above), so recolors are the natural next stage. Not yet
agreed with the user.

## After that
- Recolors: 141 unsolved puzzles lack only a recolor word. Most take the color
  from another piece (a key, a marker, a partner), so they are compositions of
  existing words (ColorFrom's partners, picks, frames), not one-property rules.
- Cells drawn on the ground: 88 unsolved puzzles lack only that.
- `python scripts\readouts.py missing <label> <said>` gives the current list.
