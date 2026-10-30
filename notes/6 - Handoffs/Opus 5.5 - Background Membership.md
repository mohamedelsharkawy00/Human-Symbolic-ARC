# Opus 5.5: background membership

Handoff, 2026-10-28. Read README.md and the local HANDOFF.md first.
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

5. Item 1 settled (the user: "decent, aim for elegance"): `cell_set`
   (`0_core.hpp`), one bit a cell at its `cell_index`, a place off the largest
   grid in no set. `figure_pair` (`3_background.hpp`): a pair whole, and which
   cells of each grid are figures. `plan_figures` builds them from the plan in
   one place; `figures_only` now only empties what is no figure, for the
   readers that do not ask the figures yet. Walls are found by grouping the
   whole input over its figures and set apart by leaving the figures;
   `rooms_fenced` and `shaped_rooms` ask the figures instead of an emptied
   cell; `cell_marks` and the ray lengths use `cell_set`. The pairs that
   `whole_pairs` lent are now `figure_pair`s.
6. The shape cut's places over selected cells (the user: "go with the shared
   shape cut over selected cells"): `each_place_it_fits` takes the cells it may
   land on, `each_shape_place` (`4_objects.hpp`) is the places the shape cut
   takes, called by `each_shape_piece` (the operator's `Split [4]`) and by
   `shaped_rooms` over the input's cells that are no figure. `shaped_rooms`
   has no search of its own.

## Two questions kept apart, and the two gaps fixed at their owners (2026-10-28)

1. Do the emptied pairs (`figures_only`) match what the emitted program sees?
   Established, by a reader audit (`2_size`, `5`-`7` and `13` read whole;
   `8`-`12` by their empty and ground tests and the functions around them):
   each input is what the program holds once the plan's opening sentence has
   run (`Input Colors SelectColor [g] DeleteSelection UnionList`, the walls too
   where set apart), each output what its sentences must draw before the fill.
   The operators read an empty cell as the ground as well (removal empties,
   negatives trade a figure with empty, fills and rays land on empty, the
   denoise vote counts empty). A replayed rule fed whole grids would read
   something its operator never sees. Only Partition's rooms and `Split [4]`'s
   ground places open on the whole input; their replays read `figure_pair`.
2. Does removing the ground before the rules are read lose no evidence? Not
   established; the evidence says it loses some.
   - Probe, temporary and removed: every plan the election allows read, the
     elected first, so a tie keeps it (`out/all_plans_{full,eval}`,
     `logs/all_plans_*`). Training gained 2, lost 0: 94133066 under fill 0
     and 868de0fa with nothing deleted. New wrong 5 (321b1fc6 575b1a71
     e73095fd e88171ec f823c43c) and explained but unsolved 0 -> 1
     (6cdd2623), all with nothing deleted. Evaluation identical, 3 right, 1
     wrong.
   - 94133066: the elected plan fills 1, the color of the block it snaps, so
     its outputs are read without the block's body and `redraws_every_output`
     refuses the snap (`[snap_to_anchors] holds 0`). The snap program with the
     elected fill (`... SnapToAnchors Union Output SelectAll Trim AddColor [1]
     Underlay`) fits 3/3 and is right on its test (`arc_spell`).
   - 868de0fa: each hollow takes the color its shape names (odd squares 7,
     even 2). With nothing deleted the hollows are pieces and the reading
     says it (`SelectPartner [the piece around it] InvertSelection
     ShapeColorMap [0]`); with 0 deleted they are empty cells no cut groups,
     and 122 cells stay unsaid.
   - Another plan left fewer cells unsaid in 100 training puzzles (2 solved,
     7 wrong, 91 no fit; nothing deleted 56, each input's own ground 35) and
     22 evaluation puzzles (all no fit; each input's own 15, nothing deleted
     7; 3a25b0d8 590 -> 42, d8e07eb2 1202 -> 407, dd6b8c4b 35 -> 4).
   - The plan shape behind 94133066, a fill in a color the inputs show as a
     figure: 9 training puzzles (2 solved), none in evaluation.
   - 332efdb3: every input cell is ground, so nothing is left to cut; a saved
     program that grows from the input as one piece fits and is right, but no
     plan's reading finds one today (29 unsaid either way).
   - f823c43c: not this question; `VoteDenoise` fails with the ground and
     without it.
   - History: of the 23 puzzles solved only under another plan on 10-18, 20
     are solved under the elected plan today, by words that read the empty
     cells (notes/6 - Handoffs/Handoff 2026-10-20.md).
   - Not established: whether readers aware of the ground, beside the emptied
     pairs, would find these without reading a second plan.
3. 94133066 fixed at the shared replay owner (the user: "put the final-fill
   comparison at the shared replay owner"): `redraws_the_output`
   (`7_facts.hpp`) judges a drawing as the program ends with it, the plan's
   own cells laid -- every painted cell in the output's color, every empty one
   a cell the plan lays (no figure: the fill under, set-apart walls over), at
   the output's size. The fill's color is the plan's, elected per input with
   the operator's own `confident_background(elect_grid_background(...))`; the
   drawing is closed as `draw_on_canvas` closes it, the box the fill's Trim
   and Fit find. `redraws_every_output` and the three rules that judged with a
   copy of their own (edge slide, copy-to, the diagonal) ask it; the whole-
   output detectors and `prepare_figures` take the `figure_pair`s.
   `fill_check_*` against `whole_source_*`: training 279 right (278), 31
   wrong, gained 94133066, lost 0, explained but unsolved 0 -> 0, one readout
   changed (its own); evaluation 3 / 1, identical. Not covered: frames
   (`choose_piece_frame`) and the line's paints judge against the emptied
   outputs still (7bb29440 is a frame).
4. 868de0fa through existing picks after Partition (the user: "use existing
   picks after Partition... derive the group and its name from the
   evidence; don't try every pick"). Partition's rooms already are the
   ground's regions as objects, and `rooms_table` read the right table off
   them; the paint failed only because "any other key -> 2" also repainted
   the room around the walls, which the outputs keep as ground (diagnostic,
   `logs/room_table_diag_868de0fa.txt`). Now `rooms_a_pick_names`
   (`11_replays.hpp`) reads the rooms each output repaints, hands the rooms
   as Partition hands them back (walls deleted, trimmed) to `pick_for_group`,
   which names exactly that group in every pair, and the table is read off and
   replayed over those rooms only (`ROOM_NAMED`, the rule's `rooms` pick).
   868de0fa's name came out "all but the largest": `... Partition SelectColor
   [0] InvertSelection DeleteSelection SelectAll Trim SelectAll
   InvertSelection SelectLargestObject [0] InvertSelection ShapeColorMap [0]
   Union`. No new operator or operator value (`14_vocabulary.hpp` and
   `src/ops.cpp` untouched); inside the reading, one more room-paint choice
   (`ROOM_NAMED`) and the rule's `rooms` pick. `named_rooms_*` against `fill_check_*`:
   training 280 right (279), 31 wrong, gained 868de0fa, lost 0, explained but
   unsolved 0 -> 0, one readout changed (its own); evaluation 3 / 1, programs
   identical, one readout changed (71e489b6, unsolved: the table over its
   repainted rooms says 8 more cells, 53 -> 48 unsaid). The every-room table
   and the named one share one spelling branch, checked byte-identical on the
   43 puzzles whose readouts name Partition.
5. Frames and paints judged against the emptied outputs (the user: "separate
   missing evidence from an incorrect replay check"). A judge calls a cell
   wrong that the program's end shows right where the color laid is the
   plan's fill, on a cell the fill lays. Measured over all 1,120 puzzles by a
   temporary diagnostic, removed; programs and readouts byte-identical to
   `named_rooms_*` (`logs/fill_diag_{train,eval}.txt`,
   `logs/fill_shadow_{train,eval}.txt`).
   - Such cells: 389 training and 61 evaluation puzzles. Mostly the recolor
     rule, which tries every color 0-9 and so the fill's too; then key swaps,
     rays, bars, rooms, boxes, fills. So "a drawing lays only colors read off
     the figures and the emptied outputs" is false for that rule.
   - Verdicts it changes: `lay_paint` and rays, none. Strict walks
     (`walk_paint`), no visible paint or recolor: only recolors to the ground
     color, the pieces vanishing into the fill, which a removal says (9,987
     walks in 173 training puzzles, 36 of them solved; 905 in 17 evaluation,
     1 solved). Frames (`choose_piece_frame`): 2 puzzles, 1b2d62fb (solved)
     and a8c38be5, each to another frame that misses too (shares 0.53 ->
     0.60, 0.64 -> 0.65). The framed reading: 2f0c5170's window, its 0 body
     read as gone (159 of 190 cells left over; 25 of 324 with 8 filled), its
     copy's place still unsaid either way.
   - The probe's 116 gaps: 21 hold hidden vanishing recolors (16 with no
     removal on the elected line; first written 15, a miscount); no other
     check error. The rest is missing
     evidence, the other plan's figures: the ground kept as pieces (56 + 7
     kept "delete none"; 9f236235's empty lattice blocks, 471 -> 38), or each
     input's own ground (35 + 15; 3a25b0d8 deletes 1, pair 0's figure
     outline, 590 -> 42). Of the gaps, 19 are frames: 8 held only under
     another plan, 11 a different frame.
   - Withdrawn: "a fill-aware judge would let `Recolor [g]` stand beside
     `DeleteSelection`". A recolor claims only cells waiting on a recolor
     (`recolor_claims`, `answers_to(a_recolor)`), and a vanished cell is a
     gone cell, so a recolor to the ground, judged right, claims nothing.
6. The 16 missing removals, and ground-colored paints (the user: "replay
   should judge what the program actually does; preferring removal over
   recoloring belongs in rule selection"). A second temporary diagnostic,
   removed; programs and readouts byte-identical to `named_rooms_*`
   (`logs/ground_diag_{train,eval}.txt`).
   - 182e5d0f, read first: each 5 at the end of a 3-path moves to the path's
     start, the path vanishing but for its first cell. Its recolors to the
     ground hold only tolerantly (some cells wrong); the removal rule's group
     is the 5s that vanish, unnamed (a staying 5 differs only by where it
     sits). A move along a path is missing, not a removal.
   - All 16, round by round: 2 hold only tolerantly (182e5d0f, 456873bc).
     In most of the rest a recolor to the ground holds strictly on a part of
     the removal rule's group, a part a pick names, while the whole group has
     none (6d0160f0 in all 161 rounds, dd6b8c4b's 9s gathered into the ring,
     a move). The removal rule says only the whole derived group, named.
     Every one of the 16 lacks where each piece went (7) or where each copy
     was drawn (13).
   - Ground-colored paints, judged as the program ends (a cell the fill lays
     shows its color; before is the line's picture): in 207 training and 27
     evaluation puzzles one holds, and in 174 and 26 a ray in the ground's
     color erases cells of pieces it did not pick, where the output shows
     the ground. Laid over, such a ray claims those cells (`covered_claims`),
     so the reading would change there. Whether that gains or loses is not
     measured: it needs the reading run with the check corrected.
7. One cell test and one picture for the judges (the user: "my aim is
   elegance"; trace every judge and caller, "confirm the canvas is complete
   wherever paints use it", "leave merging moves and removals into the canvas
   for later").
   - One test: `comes_out` (`3_background.hpp`), whether a cell left showing a
     color, or nothing, comes out as the output once the plan lays its own
     cells; `redraws_the_output` asks it of every cell. Every judge asks it
     instead of the emptied output: the paints (`lay_paint`, `tally_walk`,
     `walk_paint`, the rays' `lay_every_length`), moves (`slides_show`),
     removals (`drops_hold`, `pieces_to_remove`), hand changes
     (`hand_act_holds`, `hand_change`), a pair finished (`read_pair`), and the
     line's accounting (`line_shows`, the end's claims check, `laid_by`). A
     one-color paint reads its color off the whole output (`lay_paint`,
     `inside_holds`), so it can be the ground's. Still on the emptied outputs:
     the readers that find evidence (`pieces_recolored`, `colored_right`,
     `offsets_to_ask`, the room tables, the shape rule, `grown_walk`'s
     refusal, `paint_colors`) and the frames.
   - One picture: a paint's before is what is down where it lands, not its
     picked pieces: the line's canvas for the line's paints (`turned_holds`
     and `inside_holds` now take it), the hand for acts replayed as paints
     (`paint_holds` lays on `replay_facts::pieces_down` where no canvas is
     named). Moves and removals still draw the hand themselves.
   - Deleted: `partner_pieces::shown`, `paint_walk::cell::was`,
     `ray_fan::cell::was`, the free `pieces_down`, `landing_on`'s nothing
     down, `occupancies` and its replay key (a paint laid over is replayed once
     per canvas now), the room readers' `pairs` parameter. `replay_facts`
     holds the pairs whole in every constructor (`choose_piece_frame` passes
     them).
   - The canvas copies the line's drawing whole (it kept cells of pieces a
     still line's size rank dropped). A temporary counter, removed (rebuilt
     warning-free, 10 puzzles byte-identical): in all 1,120 puzzles no round
     asked paints on a canvas the line did not draw, and no dropped piece's
     cell was kept.
   - `one_picture_*` against `named_rooms_*`: training 281 right (280), 42
     wrong (31), gained 62b74c02 and ba26e723, lost 31aa019c, explained but
     unsolved 0 -> 0; 13 new fits (1 right), 1 fit lost (cbded52d, was
     wrong), 225 readouts changed. Evaluation 3 / 1, programs identical, 41
     readouts changed. Reading time 8,266 -> 4,793 s on training, cause not
     checked. `logs/one_picture_*`.
   - 31aa019c, lost: a ray in the ground's color, laid only onto what is down,
     erases 24 cells of the pieces that vanish and is said first; a paint that
     reads what is down freezes the line, so the removal ("Cells At Least 2 is
     removed", 59 cells) is never asked, and a second pass removes the rest,
     wrong on its test. The replay judges the ray right; preferring the
     removal is selection's.
8. The regression fixed at the shared selection owner (the user: "fix only
   the regression on 31aa019c ... without a puzzle-specific exception"):
   `weigh_paint` (`12_line.hpp`), which already says what any paint may
   account for, now also says a paint accounts only for cells where the
   output holds a figure. A cell the output leaves to the plan went; what
   takes a piece away says it -- a removal, a move, a change of shape -- as a
   recolor to the ground already claims nothing (`recolor_claims`). Only a
   paint in the plan's own color is touched: no other comes out right there.
   The replay is unchanged. 31aa019c alone: solved, program byte-identical to
   `named_rooms_*`. `figure_claims_*` against `named_rooms_*`: training 281
   right (280), 34 wrong (31), gained 62b74c02, lost 0, explained but
   unsolved 0 -> 0; 5 new fits (1 right), 1 fit lost (cbded52d, was wrong).
   Evaluation 3 / 1, programs identical. Against `one_picture_*`: 31aa019c
   solved again; ba26e723 wrong again, as in `named_rooms_*` (its solve there
   rested on such a claim, in a later pass; not traced); 8 wrong fits gone.

## What is still true today

- `figures_only` in `3_background.hpp` still hands most readers copies with
  every cell that is no figure emptied, made from the plan's figure sets.
- Empty cells retain the background's spatial outline. The earlier claim that
  removal loses all shape/position information was too broad: what is lost in
  that view is the original color and the distinction between excluded cells
  and actual absence.
- The `figure_pair`s are the source of the current reading. A later pass gets
  that pass's picture; a framed reading gets its framed pairs. It is not
  necessarily the original puzzle input. Framing currently operates on filtered
  figures.
- Shape learning already reads the complete pairs before ground removal.
  Background shapes still take `shaped_rooms` / `ROOM_SHAPE` in `11_replays`
  and `12_line`, with special spelling in `16_sentences`.
- `said_reading::before_ground` in `16_sentences.hpp` still exists. It stores
  sentence indices, not input copies. Do not confuse it with the removed field.
- Walls (2026-10-30): no separate plan any more. They stay in, marked
  (`partner_pieces::wall`, `is_wall`): the partner ways by likeness pass them by.
  No election, ranking, pass limit or rule preference changed.

## Remaining work, in order

The user agreed (2026-10-28) to move one reader at a time, its grouping and
its empty-cell tests together, so no function is touched twice: the matching
(`5_matching`, `6_correspondence`), then the other readers file by file, all
expected identical; then background shapes (walls done 2026-10-30) through the shared
owners, the first step that can change the score; then the deletions. The
items below are what each reader's move has to cover.

1. DONE (item 5 above): one representation of membership, separate from
   cell colors.
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

Next: the user's decision on items 7 and 8. Do not remove `figures_only`
alone and send every background cell through today's matching unchanged.

## Evidence and checks

Both stages passed a full local Release build without compiler warnings using
`scripts\build.bat x64-Release NOLAUNCH`. No GUI was launched.

| Stage | Training right / wrong | Evaluation right / wrong | Gained / lost |
| --- | --- | --- | --- |
| Before: shape_from_* | 278 / 31 | 3 / 1 | baseline |
| Membership: membership_* | 278 / 31 | 3 / 1 | 0 / 0 on both sets |
| Complete pairs: whole_source_* | 278 / 31 | 3 / 1 | 0 / 0 on both sets |
| Cell sets: cell_set_* | 278 / 31 | 3 / 1 | 0 / 0 on both sets |
| Shape places: shape_place_* | 278 / 31 | 3 / 1 | 0 / 0 on both sets |
| Fill-aware check: fill_check_* | 279 / 31 | 3 / 1 | +94133066 / 0 |
| Named rooms: named_rooms_* | 280 / 31 | 3 / 1 | +868de0fa / 0 |
| One test, one picture: one_picture_* | 281 / 42 | 3 / 1 | +62b74c02 +ba26e723 / -31aa019c |
| Paints claim figures only: figure_claims_* | 281 / 34 | 3 / 1 | +62b74c02 / 0 |

Up to the shape places, each stage's programs and all 1,000 training plus 120
evaluation readouts were byte-identical to the preceding stage; the stages
after changed only what their items say. Explained-but-unsolved remained 0 on
both sets. Wrong counts fitted programs with wrong test answers, not no-fit
puzzles. The identical stages establish preserved behavior, not completion of
the redesign; no speed improvement was measured for them.

Current local baselines: `out/figure_claims_full`, `out/figure_claims_eval`
(the user, 2026-10-28): 281 right / 34 wrong on training, 3 / 1 on
evaluation; against `named_rooms_*` gained 1, lost 0, explained but unsolved
0. Before them `one_picture_*`, `named_rooms_*`, `fill_check_*` and
`whole_source_*` (the cell-set and shape-place runs were identical to
`whole_source_*` and deleted once compared). The probe's runs,
`out/all_plans_{full,eval}`, are kept.
Evidence for items 7 and 8: `logs/one_picture_*` and `logs/figure_claims_*`
(builds, runs, solved and explained comparisons, `*_comparison.json`).
Evidence: `logs/whole_source_comparison.json`,
`logs/whole_source_{full,eval}_{solved,explained}.txt`,
`logs/whole_source_build.txt`; corresponding `membership_*` files cover stage 1,
`cell_set_*` files the cell sets and `shape_place_*` files the shape places.
Runs and logs are local, not shipped by this commit.

Use `scripts/readouts.py solved <before> <after>` and `explained <before> <after>`.
The acceptance rule is no solved puzzle lost and no new explained-but-unsolved.
Builds and diagnostics go to logs/. Start score runs hidden and return rather
than waiting; report their results when checked. Never launch the GUI without
the user's acceptance. Do not push to Kaggle as part of this handoff.

The local `HANDOFF.md` remains ignored and must not be added or pushed.
