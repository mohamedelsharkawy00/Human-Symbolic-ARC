# THE ORDER. READ THIS BEFORE TOUCHING THE SEARCH OR SCORING.

You keep forgetting the order. The order is the whole point. If you do it
out of order, you waste days tweaking a search whose DSL can't express
the answer in the first place.

## THE ORDER

1. **Solve training puzzles by hand with the current DSL.**
   Write the solution string. If you can't write it in <30 ops, the DSL
   is the bottleneck — not the search, not the score. Go to step 2.
   If you CAN write it, save it to solutions/<puzzle_id>.txt.

2. **Verify the solution by importing it.**
   The solver has an Import button. If the imported tree scores `solved=true`
   on all train examples, the DSL can express it. If it doesn't, the DSL
   semantics don't match your mental model — fix that first.

3. **Look at the solutions you have. Find folds.**
   Run the n-gram analysis on solutions/. See lockstep transitions,
   frequent bigrams. Fold the strongest candidate into a new primitive.
   (See notes/2_dsl/dreamcoder_folding.txt for the playbook.)

4. **Migrate every affected solution file BEFORE rebuilding.**
   Folding changes the DSL. Old solution strings won't parse correctly.
   Run a regex over solutions/ as part of the same change.

5. **Build. Benchmark a representative puzzle subset.**
   Did the fold help? APS stable? Any solve regressions? Keep or revert.

6. **Only now think about scoring or search tweaks.**
   With a DSL that can express your solutions in fewer/shallower nodes,
   the score function and beam dynamics have a real chance of finding them.
   Score-tweaking on a DSL that can't even reach the answer is pure waste.

## THE MISTAKES YOU MAKE

- **Tweaking the score before checking the DSL.** You spend hours
  calibrating weights. The real problem is the solution is 8 levels deep
  past MAX_SEARCH_DEPTH and no score is going to dig it out.
- **Adding terms instead of primitives.** Each score term has cross-
  interaction risk and per-pixel cost. A primitive is cheap, local, and
  measurably reduces depth.
- **Skipping the manual-solution step.** "I'll just run the search and
  see what it finds." It finds local maxima. Without curated solutions
  you can't even tell if the maxima are good.
- **Folding without data.** "This pattern feels common." Run the n-grams.
  If it's not in the data, it's not common.

## CHECKLIST BEFORE TOUCHING SCORE WEIGHTS

  [ ] At least one curated solution per puzzle class you care about
  [ ] All curated solutions import as solved=true
  [ ] n-gram analysis run; no obvious lockstep folds remain
  [ ] Search depth bound > the deepest curated solution's depth
  [ ] Search APS is healthy (else the search itself is broken first)

If any item is unchecked, do that BEFORE touching weights.

## THE MENTAL MODEL

DSL design = how *expressible* your solutions are.
Score = how *findable* expressible solutions are.
Search = how *fast* findable solutions are found.

You can only optimize a layer once the layer above it is good enough.
Don't optimize search on an unfindable solution. Don't optimize score
on an unexpressible solution. Don't optimize DSL without solutions to
test against.

Curated solutions are the ground truth. Everything else is downstream.
