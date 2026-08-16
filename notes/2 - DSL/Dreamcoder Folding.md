# DSL Folding (DreamCoder-style) — Priority Notes

Date started: 2026-05-12

## Idea
Promote frequently-co-occurring op sequences from curated solutions into
new DSL primitives. Each fold drops 1 op + 1 depth level per use, which
compounds when the search has to rediscover the same idiom on every puzzle.

Project notes already advocate this under "Hindsight Vocabulary" / "Symbolic
TTT" (see notes/3_search/search_strategies.txt). This file is the playbook.

## When to fold an idiom

A bigram (or longer chain) is a primitive candidate iff:

1. **Lockstep**: one op deterministically follows the other in EVERY observed
   use (e.g. Target -> AddColor was 3/3 → folded into Target[c]). Lockstep
   means folding is lossless — nothing useful is lost.

2. **High frequency**: the bigram appears in many solutions. The bigger the
   coverage, the bigger the search-time payoff. AddColor -> Union appeared in
   7/11 solutions before any folding.

3. **Atomic concept**: there's a one-sentence human description for what the
   chain does ("set up the canvas", "decompose the input"). If the answer
   needs "and then... and also..." it isn't atomic yet.

Strength tiers (in order):
- (1) + (3): lockstep AND atomic → fold immediately, even with low coverage
- (2) + (3): frequent AND atomic, non-lockstep → fold as a coexisting op
- only (2): frequent but mechanically composed → skip; not a real concept

## Method (data-driven)

1. Aggregate solutions in solutions/*.txt.
2. Strip annotation brackets (Filter[5] → Filter) for op identity.
3. Compute:
   - Unigram frequency: which ops are most used.
   - For each op: distribution of (predecessor, successor). A lockstep is
     an op whose successor distribution is a single key.
   - Bigrams/trigrams by frequency AND distinct-solution coverage.
4. Inspect any pattern that's (lockstep ∧ ≥2 samples) or (high coverage).
5. Ask the atomic-concept question for each candidate.

## Folds done so far

- **Target -> AddColor[c] → Target[c]** (lockstep 3/3). Implemented:
  X-macro IS_ANNOTATED=1, VAR_COUNT=10. Migrated 3 solution files.
- **Input -> Split → InputObjects** (high coverage, atomic). Implemented as
  a new chain-start primitive. Migrated 2 solution files.

## Candidate folds queued (data so far)

- **Target[c] -> Union → TargetUnion[c]** (or similar). 3/3 lockstep after the
  Target fold. Saves another op + depth level on the 3 affected solutions.
- **Recolor -> AddColor → Tint[c]** (2/2 lockstep, n=2).
- **Partition -> RemoveColor[0] → GroupByColor** (2/2 lockstep, n=2).
- **AddColor -> Union → Paint[c]** (7/8 = 88% lockstep, n=8).

## Risks / things to watch

- **Lockstep with n=1 is noise**, not evidence. Wait for n≥2 before folding.
- **Folding without migrating solutions** breaks importability. Always
  rewrite affected .txt files in the same edit.
- **Coexisting ops bloat the search branching factor**. InputObjects added
  another chain-start; if the new primitive is rarely useful, the cost
  isn't paid back. Measure APS impact before keeping.
- **Annotated chain-starts** (Target[c]) need special handling in
  seed_starter_nodes — must generate one node per variant. Easy to forget.
- **Folding too aggressively** ends up with hundreds of niche primitives.
  Hodel's solver had ~70; Icecuber's similar. Keep the bar at "saves depth
  on multiple real puzzles" — not "saves depth in theory."

## Workflow: data → fold → measure → keep or revert

1. Run the n-gram analysis on solutions/ (`scripts/` candidate).
2. Pick the strongest candidate by the tier rules above.
3. Implement: X-macro, gpu enum, gpu executor, dispatch case, starter list
   if applicable.
4. Migrate solution .txt files.
5. Build, benchmark a representative puzzle subset.
6. If APS regressed or no puzzle solves faster → revert. Otherwise keep.
7. Repeat — fold again with the new data (now compressed).

The key insight: folding is iterative. Each fold reveals new bigrams that
become viable candidates after the previous compression. Run the analysis
again after each fold.
