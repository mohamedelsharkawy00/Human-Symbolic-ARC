# Connectivity is two questions, and only one of them is answered out loud

Parked 2026-08-12. Raised while writing the cut labels for the correspondence
pane: the label said CORNERS TOO, and the obvious next question was whether
everything in inference agrees about corners. It does not, and it probably
should not — but nothing says so.

## What is actually there

**Grouping is always 8-connected, and there is no alternative.** `walk_out`
(`1_readings.hpp`) hardcodes `for (int dir = 0; dir < 8; ++dir)`, and it
is the only walker in the family. Four callers use it and all inherit corners:

- `correspondence.hpp` — the cuts, so every object in the pane
- `infer.hpp` — the blob measure
- `background.hpp`, twice — the blob pass and the outside flood

**Two measures count neighbours 4-connected instead**, both in
`background.hpp`, and they are not groupings — they ask "who is beside me":

- `gaps` — the share of a colour's cells that have a differently coloured
  neighbour. A ground showing through is thin, so this is a fraying measure.
- the outer contour vote — which blob the cells around this one belong to.

## What is fine and what is missing

The split is defensible, and probably right. Grouping asks *is this one thing*,
where a diagonal join is usually what a person sees. Adjacency asks *what is
beside this*, where a diagonal is a corner touch and not a border — a wall with
a diagonal gap should not leak.

So this is not a bug to fix. It is a rule that exists in the code and nowhere in
words, and the evidence is that the two uses are documented unevenly:

- The contour vote **says so**: "edges only, not corners", right at the loop.
- `gaps` says nothing. Someone changing it to `walk_out`'s 8 for consistency
  would be doing the reasonable thing and would silently move every background
  score that leans on `gaps`.

## What to do

Say the rule once, where both can see it — next to `NEIGHBOUR_ROW_4/8` in
`0_core_types.hpp` is the natural home, since that is where the choice is offered:

> Grouping is 8. Adjacency is 4. A diagonal join makes one thing out of two;
> a diagonal touch is not a border.

Then either every 4-connected loop is an adjacency question and needs no further
comment, or it is not and has to justify itself.

## How to find out whether it matters

Flip `gaps` to 8-connected and reload the training set. `logs/inference/*.txt`
carries the background election per puzzle, so a diff over the `[background]`
sections says exactly which puzzles the choice is load-bearing for. If none
move, the rule is free and only the words are missing. If some move, the note
above is understating it and `gaps` deserves a name for what it measures.

Not urgent. Nothing is known to be wrong; this is a rule with no written form,
which is the state things are in just before someone "tidies" them.
