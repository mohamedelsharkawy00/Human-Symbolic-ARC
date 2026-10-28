# Opus 5.5: background membership

Handoff, 2026-10-28. Read README.md, AGENTS.md and the local HANDOFF.md first.
This note concerns the background redesign only. The code and this note are
committed together after bfab5c3; use git history for the exact commit.

## What the user agreed

Keep the grid whole; let grouping decide which cells belong. Recognizing
background should add knowledge rather than erase cells before other readers
can use them. Background and exclusion from an object are different facts.

The user approved implementation repeatedly, then explicitly asked to implement
it, said continue, and requested this handoff and a push to main. Continue the
agreed work without asking for the same approval again. New design choices still
need thought with the user under AGENTS.md. Keep replies short.

The aim is fewer special paths through stronger shared ideas. Merely renaming
the old paths or passing more copies around does not complete the redesign.

## Completed and verified

1. `include/core/1_readings.hpp`: `each_object` accepts an `includes(row, col)`
   predicate. It checks membership once per colored cell before grouping.
   Transparent cells remain excluded without calling the predicate. Its other
   overload includes all colored cells. The unused background-color argument
   was removed from every caller, including operators in `src/ops.cpp`.
2. `include/core/4_objects.hpp`: `each_shape_piece` marks cells already belonging
   to shapes and groups the remaining cells from the same intact source grid.
   It no longer copies a grid and erases those cells. Overlapping shapes and
   their order remain unchanged.
3. `include/core/9_rules.hpp`: ranked layout selects cells by its existing color
   choice while grouping the intact input, instead of building a filtered grid.
4. `inference::before_ground`, the stored vector of input copies, is removed.
   `read_with_ground` lends its complete pairs through `read_figures`,
   `read_the_cut` and `finish_the_line` as `whole_pairs`. `rooms_fenced` and
   `shaped_rooms` read those pairs directly. No reference is stored in the
   returned inference or its cache.

These are prerequisites, not the main behavior change. The source diff is
57 added / 63 removed lines, excluding this note. No new operator or rule.

## What is still true today

- `figures_only` in `3_background.hpp` still replaces background colors with
  TRANSPARENT in copied pairs. Most readers still receive those filtered pairs.
- Empty cells retain the background's spatial outline. The earlier claim that
  removal loses all shape/position information was too broad: what is lost in
  that view is the original color and the distinction between excluded cells
  and actual absence.
- `whole_pairs` is the source of the current reading. A later pass gets that
  pass's picture; a framed reading gets its framed pairs. It is not necessarily
  the original puzzle input. Framing currently operates on filtered figures.
- Shape learning already reads the complete pairs before ground removal.
  Background shapes still take `shaped_rooms` / `ROOM_SHAPE` in `11_replays`
  and `12_line`, with special spelling in `16_sentences`.
- `said_reading::before_ground` in `16_sentences.hpp` still exists. It stores
  sentence indices, not input copies. Do not confuse it with the removed field.
- Walls still have the separate `walls_apart` plan and `set_the_walls_apart`.
  No election, ranking, pass limit or rule preference changed.

## Remaining work, in order

1. Settle one representation of background membership separate from cell
   colors. Keep one source of truth for colors and explicit object membership;
   the exact representation is not implemented or settled by this handoff.
2. Pass complete grids and membership into every grouping, including color
   layers, connected groups, repeating blocks and the learned shape cut.
   Preserve object order, overlap, frames and tile boundaries.
3. Update matching and changed-cell accounting. Excluded cells still exist;
   changing a background cell must remain visible as evidence. Audit
   `6_correspondence.hpp`, including `summarize_pair`, `find_blanks`,
   `write_down_what_changed`, cell counts and claims.
4. Update readers to distinguish background, cells outside an object, and
   actual empty space. Audit direct TRANSPARENT tests rather than replacing
   them mechanically: different tests answer different questions.
5. Let background shapes use ordinary shape grouping and rules. Share the
   existing shape owner; do not add another background-only detector.
6. Apply the same distinction to walls and rooms. Preserve their real roles
   while removing the need to erase them globally to see other objects.
7. Keep replay and emitted programs consistent. Express selection in the DSL;
   do not add hidden operator state. Preserve the rule that each claimed rule
   is replayed on every pair with the operators' own arithmetic.
8. Carry the distinction through crops, frames, later passes, output sizing,
   background filling and cached readings. A context change must not reuse
   facts from a different selection or grid.
9. Delete replaced machinery: `figures_only`, `shaped_rooms` / `ROOM_SHAPE`, and
   special before-background sentence handling when their jobs have genuinely
   moved to the shared owners. Review `foreground_input`, `say_the_plan`,
   `restore_background`, `plan_to_read` and wall handling together with callers.
   Background filling itself can still be a real output operation.
10. Complete and measure each stage before starting another. Remove obsolete
    code and update the relevant notes. No promise of score gains from these
    structural changes has been established.

Start with item 1 and its path into grouping. Do not remove `figures_only`
alone and send every background cell through today's matching unchanged.

## Evidence and checks

Both stages passed a full local Release build without compiler warnings using
`scripts\build.bat x64-Release NOLAUNCH`. No GUI was launched.

| Stage | Training right / wrong | Evaluation right / wrong | Gained / lost |
| --- | --- | --- | --- |
| Before: shape_from_* | 278 / 31 | 3 / 1 | baseline |
| Membership: membership_* | 278 / 31 | 3 / 1 | 0 / 0 on both sets |
| Complete pairs: whole_source_* | 278 / 31 | 3 / 1 | 0 / 0 on both sets |

Each stage's programs and all 1,000 training plus 120 evaluation readouts were
byte-identical to the preceding stage. Explained-but-unsolved remained 0 on
both sets. Wrong counts fitted programs with wrong test answers, not no-fit
puzzles. These checks establish preserved behavior, not completion of the
redesign. No speed improvement was measured.

Current local baselines: `out/whole_source_full`, `out/whole_source_eval`.
Evidence: `logs/whole_source_comparison.json`,
`logs/whole_source_{full,eval}_{solved,explained}.txt`,
`logs/whole_source_build.txt`; corresponding `membership_*` files cover stage 1.
Runs and logs are local, not shipped by this commit.

Use `scripts/readouts.py solved <before> <after>` and `explained <before> <after>`.
The acceptance rule is no solved puzzle lost and no new explained-but-unsolved.
Builds and diagnostics go to logs/. Start score runs hidden and return rather
than waiting; report their results when checked. Never launch the GUI without
the user's acceptance. Do not push to Kaggle as part of this handoff.

The local `HANDOFF.md` remains ignored and must not be added or pushed.
