# Correspondence first

**TL;DR:** With four weeks left, fold the standalone detectors into the
existing correspondence system instead of building a new layer. First, find out
why correspondence misses puzzles the detectors already solve. Then integrate
one detector family and start combining explanations, with copy + recolor as
the likely first test. If fresh puzzles show no gains by the end of week 2,
stop the architecture work.

Decision from the discussion on 2026-09-28, with four weeks remaining.

## Direction

Broaden correspondence first, make its evidence reliable, then compose the
supported explanations. Extend the existing explanation system rather than
building a second one alongside it.

The longer-term ambition is a broadly expressive symbolic transformation
language, with inference that prefers explanations natural to human-authored
ARC puzzles. Expressiveness and preference are separate concerns: supporting
more transformations also introduces more explanations that fit the examples.
The current operations already generalize, but their reach is limited by
assumptions that remain coupled and by inference paths that cannot cooperate.

For the remaining four weeks, the immediate opportunity is to compose existing
capabilities and bring transformation detectors into correspondence as readings.
Covering every possible transformation is a longer-term ambition.

## Current barriers

- Correspondence defines its readings as segmentation times allowed matching
  transformations. It already owns competing readings, matches, explained and
  unexplained cells, and an election.
- Other detectors produce separate answers outside that election. Several
  require their own transformation to reproduce the entire output, so a valid
  partial explanation can be rejected before composition is possible.
- The sentence emitter often returns after the first successful detector.
  Branch order therefore determines which explanation is expressed.

Removing early returns alone would not solve composition. Nor would merely
registering every detector as another reading: the underlying representation,
matching, and evidence accounting must support what those detectors explain.

## First step: audit the gap

Start with puzzles that standalone detectors already solve. For each, determine
why correspondence cannot express the same explanation:

1. Is the required object representation or grouping missing?
2. Is the transformation unsupported by matching?
3. Is valid evidence discarded during matching or election?
4. Does the evidence exist but lack a correct DSL expression?

These are different failures and should receive different fixes. The audit
should record the successful detector, correspondence's reading and unexplained
observations, the failure category, and the smallest required change.

Use the first day to map these gaps. Choose one detector family to integrate
through the existing correspondence machinery, based on the findings.

## Integration before composition

Preserve the chosen family's successful solutions while making its explanation
visible in correspondence's shared accounting. Reuse existing perception,
transformation helpers, matching evidence, and DSL operations wherever possible.
Keep the implementation DRY without adding abstractions solely for anticipated
future cases.

A contributed reading needs to retain what it acts on, the relationship it
establishes, the observations it explains, what remains unexplained, and any
dependencies on other readings. Some transformations involve groups, regions,
or generated cells; they should not be forced into ordinary one-to-one object
edges. Background and size facts can remain shared context.

Composition follows once object identities, coordinate frames, and coverage
agree. Distinguish competing readings, independent effects, and dependent steps.
Do not double-count explained cells or assume that overlapping effects commute.
Validate the complete emitted program against every training output.

Copying plus recoloring is a possible first composition test, subject to the
audit. Success means existing capabilities explain their combination without a
new detector dedicated to that combination.

## Four-week boundary

The order is:

**Broaden correspondence -> reliable shared evidence -> composition.**

This replaces the earlier proposal to start with a separate explanation layer
or immediately expand relational argument inference. Improving the existing
matching types alone is insufficient if other detectors remain outside the
reading system.

Keep the current solver working throughout. Measure fresh-puzzle gains and
regressions, reserve evaluation puzzles from design decisions, and leave the
last week for stabilization, runtime, and submission checks. If the work yields
no fresh-puzzle gains by the end of week two, stop architectural expansion and
prioritize demonstrated gaps.

## Starting points in the code

- `include/core/5_matching.hpp`: matching permissions and correspondence edges.
- `include/core/6_correspondence.hpp`: readings, evidence accounting, and election.
- `include/core/7_infer.hpp`: standalone detectors and inference ordering.
- `include/core/9_sentences.hpp`: expression of readings and early-return branches.

Baseline after adding CopyTo: across the 1,000 training tasks, inference-generated
programs fit 104 tasks' training examples and solve 103 held-out tests, with one
existing held-out failure. The change added `363442ee` and `88a10436`, preserved
all previous results, and passed 20 focused CopyTo checks.

## Response after measuring (2026-09-29)

**TL;DR:** Correspondence now keeps the shared evidence this plan asked for,
and that evidence shows that combining existing capabilities is not where the
gains are yet: three measurements found 2, 4 and 0 more tasks. What unsolved
tasks lack is the reason for a change, something elsewhere in the grid that
decides a piece's new color or place. The next step is to read that reason off
the examples: the relational argument inference this plan deferred.

### What changed

- A reading is now a list of the rules that fired. Each names the pieces and
  cells it accounts for, or what the language has no word for.
- The two block segmentations no longer compete; their matches are evidence for
  the rules that claim them.
- One price chooses what is said, whether the elected reading's own line or a
  standalone detector: complete first, then fewest unexplained cells, then
  fewest operations. Branch order no longer decides.
- Accounting is per piece. An act explains only the pieces it covers, and a
  moved, copied, turned or recolored piece that nothing says stays unexplained.
  Unsolved tasks reported as fully explained fell from 84 to 3.
- Training: 106 tasks fit and 105 tests are right, with the one existing
  failure (was 104 and 103). The additions are `0d3d703e` and `7468f01a`.

### What the measurements showed

Measured on unsolved training tasks with scratch tools outside the solver. The
first two ran candidate programs to check them; the third only read evidence.

| Question | Tasks finished |
|---|---|
| Chain two existing explanations | 2 of 894 |
| Add one operation to the elected reading's program | 4 of 894 |
| Let the 15 detectors that must redraw a whole output explain part of it | 0 of 638 same-size tasks |

Most of those detectors reached their checks in about 560 of the 638 tasks.
Some detector was right on every cell it changed in only 26 tasks, and none
covered everything the elected reading left unexplained.

What the elected readings cannot say, across the 638 same-size unsolved tasks:
a recolor in 298 (the only missing word in 109, and the only thing unexplained
in 89), where each copy was drawn in 239, cells drawn on the ground in 146, how
each piece turned in 135, where each piece went in 113, and how a piece changed
its shape in 80.

### What that means

The recolor words exist: one color becomes another, two colors swap, a shape or
a size picks the color. In these tasks they do not hold, because something else
in the grid decides the change:

- `0becf7df`: a 2×2 key names the swaps, and differs per example.
- `0d87d2a6`: a line joins two border marks, and boxes it crosses turn blue.
- `103eff5b`: a small template colors its enlarged copy.
- `12eac192`: shapes of fewer than three cells turn green.
- `13f06aa5`: each shape points at an edge; a dotted line runs to it, and the
  edge takes the pointer's color.

The barrier is less that capabilities cannot cooperate than that explanations
lack the reason for a change. By this plan's own boundary, that demonstrated
gap comes first.

### Revised next step

Read the reason off the evidence. For each changed piece, look for what in the
same input carries its new value: a key cell, a touching piece, a mark in its
row or column, its own size. Keep a relation only if it holds for every piece it
selects in every example, the test the existing acts already pass, so inference
stays searchless. Each relation joins the reading's list as a rule, and
composition remains the frame: `0d87d2a6` is two such rules, one drawing the
line and one recoloring what it crosses.

First, count which relations explain the most of the 89 same-size tasks whose
only unexplained part is a recolor.
