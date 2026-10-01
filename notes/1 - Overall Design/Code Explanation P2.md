Generation

### 3. Generating Candidates (the phases)

Each beam member is a tree. The solver picks a "slot" — a position in the tree —
and tries replacing whatever is there with something new.

Example: if the beam has Overlay(Split(Input), Input), the solver might try
replacing "Split(Input)" with "Filter[3](Input)" to get Overlay(Filter[3](Input), Input).

It generates replacements in phases:

  Phase 0 (prune):    Try each child of the current subtree as a replacement
                      (simplification — drop a wrapper).

  Phase 1 (history):  Try sub-expressions from the history pool
                      (reuse stuff that worked before).

  Phase 2 (unary):    Try wrapping with a 1-arg primitive: Split(x), Trim(x), etc.

  Phases 3-6 (binary): Try 2-arg primitives with various second arguments:
                        terminals, history pool entries, etc.

Each generated candidate gets scored. The solver processes candidates in batches,
yielding back to the scorer periodically.


### 4. Picking Winners (commit_level_champion)

After all slots are processed, the solver has a big pile of scored candidates
(level_candidates). Now it picks the next beam:

  a) BEST PER TYPE: For each node_type (Overlay, Fill, Split, etc.), keep the
     single highest-scoring candidate. This guarantees diversity — at least one
     Overlay, one Fill, one Split, etc. survive.

  b) DIVERSITY FILL (our new two-pass fix):
     Pass 1: Fill remaining beam slots sorted by score, but cap each type at 2
             extra entries. This prevents one hot type from grabbing all slots.
     Pass 2: If there are still empty slots, fill them with the best remaining
             candidates regardless of type. The beam always reaches 128.

  c) MONOTONICITY: Before all this, the current beam members are injected into
     level_candidates. This means a beam member can survive even if no improvement
     was found — the search never forgets its best programs.


### 5. Anti-Stagnation

  SLOT STAGNATION: If several consecutive slots don't improve the level best,
                   stop early and commit. Don't waste time on unpromising slots.

  LEVEL STAGNATION: If the beam's top score hasn't improved for several levels,
                    drop the bottom half of the beam and inject history pool entries.
                    This breaks out of local optima by reintroducing old building blocks.


### 6. History Pool (update_history_from_beam)

After each level, the solver decomposes every beam member into sub-trees and adds
new ones to the history pool. This means if the beam found Overlay(Fill(X), Input),
then Fill(X), X, and Overlay(Fill(X), Input) all become available as building blocks
for future levels.

The pool is capped (MAX_HISTORY_POOL) — oldest entries get evicted.
