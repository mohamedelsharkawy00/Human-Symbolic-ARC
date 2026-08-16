# Solution Skeleton From Inference — DONE 2026-07-26

Started as a claim: the input background is used for segmentation then removal,
and the output background is laid down last. Both turned out to be true across
the whole corpus, and both now live in the engine instead of in programs.

## What the corpus said

Leftmost token is first, and first is the bottom layer.

Every solution that could be verified opens with segmentation followed by
`SelectInferredBackgroundColor DeleteSelection`, and closes by underlaying the
output background. Getting there took 15 rewrites, all verified solving.

Three of my rewriting mistakes are worth remembering, because each looked like a
counterexample to the claim and was not:

1. Inserting `Colors` where the pipeline was grid-level, which converts it to
   object-level. Fix: `UnionList` to flatten back.
2. Flattening with `Union` where `UnionList` was needed.
3. Deleting the background without restoring it at the tail.

## What moved into the engine

- `gpu_exec_colors` skips the elected input background.
- `gpu_exec_split` erases it from the perception canvas before segmenting.
- After eval, and after the BoundingBox crop, every still-transparent cell
  becomes the elected output background (`output_background_color`, bridged from
  `p.bg.output_elected` through `fill_puzzle_level`).

The closing frame runs AFTER eval, not before. Painting the canvas at birth was
tried first and broke 14 puzzles: transparency is how every op says "nothing
here", so a pre-filled canvas changes what Trim, crop and mask logic see.

## What it bought

    tokens across the corpus   536 -> 393   (27% was frame, not puzzle)
    mean chain length         12.5 -> 9.1

Cost is roughly (op choices)^depth, so this is ~3.4 levels of depth back on
every puzzle — the budget Inference Master §4b says unfusing was waiting on.

The fan shrank too: `SelectInferredBackgroundColor` was deleted from the
primitive list on the same day — nothing needs it once segmentation stops
handing the ground back. `AddColor` and `Underlay` stay, being general ops used
mid-program.

## The one apparent counterexample, which wasn't

0520fde7 kept the background as its SUBJECT (`Split[1] SelectColor[0]
InvertSelection DeleteSelection`), so erasing the ground left it nothing. But
reachability was not lost: with the ground gone the solver re-solved it in ~18s
with a 5-token program where the old one needed 17.

    Input Colors SizeRankColorMap Underlay FlipHorizontal Intersection

## Asymmetry worth keeping in mind

`Colors` drops a colour layer and every other object is byte-identical.
`Split` erases cells, which changes the geometry of what remains — regions that
touched through ground cells can split, and shapes that enclosed ground gain
holes. The rule is "the background is never wanted as a colour layer", not
"the background is never wanted".

## Left open

Nothing on the head or tail. The next fan question is whether any other op is
only ever used to undo something inference already knows.
