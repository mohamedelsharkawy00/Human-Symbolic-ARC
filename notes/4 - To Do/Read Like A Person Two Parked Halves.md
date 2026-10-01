# Read like a person: the two parked halves

Parked 2026-08-10. Raised while asking why a person reads a pair in about a
hundred steps and the cover search cannot do it in twenty million.

## Where this came from

The arithmetic kills the "brains are just massively parallel" answer outright.
The cover instance on `1be83260` has a search tree around 10^137 wide. Against
that, ~10^11 neurons leaves 10^126 to go, and a neuron fires on the order of
milliseconds while a person sees the structure of a picture in a few hundred.
That is about a hundred sequential steps end to end. Whatever a person is
doing, there is no room in it for a search.

Four things fill those hundred steps:

1. They never build the candidate list.
2. They do not demand exactness.
3. They see the coarse thing first.
4. They settle rather than try.

## What was built

**3 — see the coarse thing first.** This is the whole look-first election in
`pick_split`: every cut priced in bits as a share of saying the grid plainly,
scored on both grids, before any matching happens. It also produced the
both-ways-at-once reading, which removed the direction question entirely.

**2 — do not demand exactness.** Half done. `search_cover` no longer *pays* for
an exhaustive proof: the stop comes from the instance's own size
(`pieces x cells`) rather than a magic twenty million. But the demand itself is
untouched — it still asks for a perfect partition and gives up sooner, rather
than asking a weaker question in the first place.

## What is parked

### 1 — never build the candidate list

`gather_matches` generates every placement of every piece at every pose, then
`build_cover` asks which subset tiles. On `1be83260` that is 570 candidate
placements over 253 cells. A person looks at that grid and sees three pieces.

The blowup is manufactured before the search starts. Whatever prunes it has to
be a property of the picture, not of the search — the search cannot tell that
its candidate list was silly.

Open question, and the reason this is parked rather than started: what makes a
placement worth proposing at all? "It fits" is what we use now and it is far too
weak. Shape-sharing between the two grids is too strong — it would kill every
denoise puzzle, where the whole point is that the input's copy is corrupted.

### 4 — settle rather than try

`search_cover` is still a backtracking tree. Measured on `1be83260`: 102,491
nodes/sec, `deepest` pinned at 95 and never moving again, 76% of nodes dead
ends. It is not hunting for a cover, it is re-permuting a prefix it already
disproved.

A person does not try combinations; everything votes at once and the picture
relaxes into one reading. One pass, not a tree. The min-remaining-values
heuristic already computes the information a relaxation would need — options per
cell — and throws it away at every node.

Note the ordering: 4 only matters if 1 is not done. Nine hundred candidate
placements make a tree that needs replacing; a dozen do not.

## Why neither is urgent

Both are about cost, and cost is no longer the binding problem. Every reading is
matched, all 114 puzzles load in 16.7s, and `1be83260` — which never finished at
all before this work — is 8.6s and the only one over half a second.

Start these when speed or correctness demands it. Correctness is the more likely
trigger of the two, and it cannot be judged yet: there is no hand-labelled truth
for which cut and which direction a puzzle should be read with.
