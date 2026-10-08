# Handoff, 2026-10-07

Read AGENTS.md first: the rules, how to build and check, where things are.
This file holds only what is current; past stages are in git history
(`git log`, and this file's own history).

## Where it stands
- Training: 199 right, 0 wrong (not yet committed; 198 with the first so
  many by size, out/ranksall/; 197 with the side touch, out/touchsides/; 196
  with the one-pair keys, out/shownkey/; 195 with the filled frame and
  symmetry, out/symmetry/; 188 with the table's "any other", out/pairs/; 187
  at 7740de8, out/tile/, 172 at a201ff6, out/merged/, 170 at 32d371d,
  out/base/). Current run: out/everypair/ (out\ is not in git).
- solutions/ holds the inferred program of every solved puzzle. Left as they
  were: the manual 05a7bcf2, 10fcaaa3 (now inferred too), 32597951,
  f25fbde4, and 44d8ac46 (an old inferred one that no longer fits).
  solutions/shelf/ holds puzzles outside training.
- A full score has taken 34 to 82 s on the same code: the laptop's clock
  moves with heat, so one run's seconds say little. Compare two builds by
  interleaving them (old, new, old, new) and taking each one's fastest run.
  Three runs back to back, before the last two changes: 43.5, 59.1 and
  48.8 s (one run with the paints asked biggest first: 77 s, not compared
  with the old build, the user was busy; 57 s before; 28.7 to 34.2 s before
  the cut by sides and lists of 900; 137 s with lists of 900 and no reading
  limit). scripts\profile.bat
  profiles arc_said (VTune, attached after CUDA starts).
- What blocks the unsolved (`readouts.py missing everypair`,
  named / the only word missing): copies 485 / 73, a recolor 383 / 71, turns
  254 / 26, where pieces went 223 / 44, cells drawn on the ground 143 / 67,
  shape changes 116 / 7. The line now counts what it draws as said, so other
  plans and cuts are elected for many unsolved puzzles than before (a recolor
  was the only missing word in 117).

## Next, in the user's order
1. 18286ef8, parked (the user agreed). Sayable, checked with arc_spell (fits,
   test right): `SelectColor[9] MoveToward[15]`, then `SelectColor[6]
   Recolor[9]`, then the 5s' box `AddColor[5] Underlay`. Not found; it needs
   the move to claim the input cell its landing covers (it ties ColorFrom's
   "the 9 takes the 5 around it" at 6 cells), "6 becomes 9" sayable after the
   move (by color it is only an act, and the acts open the line), and the box
   fill asked of a line that moves pieces.
2. Speed: read the cuts and plans of one puzzle in parallel (low priority).

Smaller, open:
- Recolors, the stage now: 71 unsolved lack only a recolor, mostly a color
  taken from another piece (compositions of ColorFrom's partners, picks,
  frames). Work one puzzle at a time with the user, the fewest pixels first
  (all grids, test answers too): 25d8a9c8, 44f52bb0, 27a28665, b7256dcd,
  9caf5b84 and d23f8c26 done; next e3fe1151 (200: a 7 cross cuts four panes,
  and each pane's stray 7 takes the color it lacks, so all four hold the
  same colors -- a new op, no composition says it), fc754716 (294).
- ef26cbf6, open: "ColorFrom[0], the nearest" says it (checked in Python,
  test right) with the walls picked along with the shapes, so they are no
  shape's partner -- but every dot is as near a wall as the next, and a piece
  whose partners disagree is one ColorFrom's word refuses: let through, it
  made d23f8c26 fit and go wrong on its test (its "6 becomes 0" is a pairs'
  coincidence).
- Four-connected before eight until eight is needed (the user): the partners
  ask the touch at a side before the touch at a corner. Asked of the cuts too
  (the cut by sides read just ahead of the cut by corners), it lost
  4938f0c2, 64a7c07e and 9172f3a0, two of them wrong, so the cut by sides
  still comes last.
- The filled frame also solved 3194b014 and 445eab21 (the largest piece's
  color), b9b7f026 (the piece with a hole), and two that look lucky:
  642d658d (the color layer with two holes) and d9fac9be (the third column
  band's color).
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
  each in a sentence of its own (fired_rule::paints_after_the_line), and
  compete the same way, the biggest first, then all asked again (e729b7be:
  a repeat asked first took cells by coincidence ahead of the half turn that
  draws everything). A paint is weighed only where it explains a cell in
  every pair (91714a58's one-cell rays finished a line of coincidences).
- What the line draws is said (Drawn By The Line, list_claims): after the
  words, every left-over cell the hand they leave shows as the output shows
  it is claimed, however the matching paired the pieces (d23f8c26's middle
  column, paired with pieces the words remove). Only of a hand the words
  carried, on pairs that lie over their outputs.
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
- The cut by sides alone (four_connected_monochrome, Split[3]) is read after
  every other cut, only where it cuts differently from the corner cut and
  into no more pieces than the reading reads.
- A tiled input is copies of the input's pieces: its program cuts the input
  before Tile (frame_the_sentences), Split keeps each piece in the frame it
  was cut from, so Tile lays its copies the input's size apart, and the
  reading through the frame cuts a grid laid out of copies one copy at a time
  (grid::tile_width, each_object). 10fcaaa3: two 3s from two copies touch.
- A list holds 900 pieces, one for every cell of the largest grid
  (MAX_OBJECTS_IN_A_LIST): only search ran thousands of programs at once.
  The GPU copies only the pieces in use and keeps a ring of 8 lists; host
  uploads sit on the heap. The reading reads a cut only where no input cuts
  into more than 128 pieces (MOST_PIECES_READ, TEMPORARY: reading hundreds of
  pieces cost minutes); search's big batches would not fit the GPU now.
- Partner ways are asked surest first (partner_ways_asked, 1_readings.hpp):
  around it and inside it, then lined up in its rows or columns, then the
  nearest, then the touch at a side (touching_sides, four-connected), then
  the touch at a side or a corner (touching), then the rest. A tie goes to
  the first asked. touching_sides came last in partner_by, so ColorFrom and
  WouldFit kept their numbers; AlignTo, MirrorAround and MoveToward count
  the ways, so their numbers moved, and the stored solutions were renumbered.
- ColorFrom leaves a picked piece with no partner as it is, as its op does,
  and the piece has to come out as it was (LEFT_AS_IT_IS, recolors_hold);
  one whose partners disagree it refuses (d23f8c26).
- Bridge hands back its bars as one drawing, a cell to the first bar that
  crosses it, so a list never runs out of room for them.
- ShapeColorMap[kind]: every picked piece takes the color its own key names
  in the learned table: [0] its shape, the same turned or mirrored either way
  (canonical_shape_hash), [1] its symmetry, which turns and flips leave its
  drawing as it was, colors included (symmetry_of). shape_key_of
  (1_readings.hpp) is the one key of reading and op, shape_table the one
  lookup; the table also has a color for every key it does not name. The
  table is read off the pairs by read_off (7_infer.hpp), for the line's shape
  word and the filled frame alike: it names the keys seen in two pairs at
  least (b230c067's twins are new in every pair), and a key seen in one pair
  only where its color is the table's alone to say -- no other key comes out
  in it and its input shows it nowhere (27a28665's plus, 6; ef26cbf6's and
  b7256dcd's shapes take their markers' colors, and naming them made both
  wrong). The keys seen in one pair and not named may all come out in one
  color, seen in two pairs, which is then the color of every other key
  (25d8a9c8: a full row 5, any other shape 0).
  detect_body_recolor reads a key naming a figure's color, said in three
  words: the key takes its shape's color, the figure the key's (ColorFrom),
  the key goes.
- The filled frame (piece_frame_rule::filled), asked where every output is
  one color and the size rule gives the canvas: the pieces a pick takes,
  each in one color -- its own, or the one its key names (ShapeColorMap) --
  filled to its box and stretched over the canvas, AddColor[11] Fit[1] as a
  plan lays its fill (filled_input). Asked after every other frame, its own
  color before a table; a table only where it is at work in every pair.
  44f52bb0: the whole picture (no ground deleted, Split[0]), 7 where no turn
  or flip leaves it as it was, else 1.
- Every word picks by cell count too: SelectAtLeastCells[n] (2..10, its
  count paints_at_least, shared with the op), fewer through InvertSelection;
  a count is asked only where it splits some pair's pieces. And by the first
  so many by size, largest or smallest first, or every piece but those
  (largest_ranks and kin): one SelectLargestObject or SelectSmallestObject a
  rank, asked only where every pair's order parts there with no tie
  (9caf5b84: every color but the two most common becomes 7). And by the row
  or column so many lines past the middle of all the pieces paint, or every
  piece but those (SelectRowFromMiddle, SelectColumnFromMiddle, new ops; the
  middle by line_from_middle, none where the span has two middle lines):
  d23f8c26 keeps its middle column, cut by sides since a diagonal 8 crosses
  it.
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
