# Symbolic TTT Techniques

Date: 2026-07-16
Source: external Claude conversation on test-time training without neural nets.

## Principle
- TTT strictly means updating weights, so the purist answer is no.
- The principle — adapt the solver to each task at test time using its demo
  pairs — is fully achievable symbolically, into constraints and search priors
  instead of weights.

## 1. Invariant mining (biggest immediate pruner)
- For each demo pair, compute properties: output size = input size? Color
  histogram preserved? Object count preserved?
- Every property that holds across all demo pairs becomes a hard constraint
  that prunes the program space before searching.
- Example: if output is always the input cropped to the largest object, 99% of
  programs are eliminated before search starts.
- This is "learning from the demos at test time", just into constraints.

## 2. Per-task primitive re-weighting
- Run a cheap first pass, observe which primitives produce partial credit
  (right output size, right palette, pixel overlap improving).
- Re-weight the search distribution toward those primitives, restart deeper.
- Search-space analogue of gradient steps: cheap evaluations -> update beliefs
  (weights over primitives) -> search again.
- Multi-armed bandit or simple softmax-over-scores both work.
- Distinct from Learned Orderings (offline, across tasks); this is per-task at
  test time.

## 3. Iterative program repair
- Do not discard near-miss programs. If a candidate gets 3 of 4 demo pairs
  right, freeze it and search only over edits to it: swap one primitive,
  insert one step, adjust one argument.
- Hill-climbing initialized by search — the symbolic version of "a few
  fine-tuning steps": adapt an almost-right solution instead of restarting.

## 4. Library learning (already covered elsewhere)
- Promote recurring sub-programs into new named primitives; the DSL grows and
  compresses, future searches get shallower.
- Symbolic counterpart of pretraining, with per-task search as the fine-tune.
- Already the plan here: see "Hindsight Vocabulary" in Search Strategies.md and
  the playbook in notes/2 - DSL/Dreamcoder Folding.md.

## Suggested order
- Invariant mining (1) -> weighted beam search (2) -> repair loop on
  near-misses (3), with library learning (4) as the ambitious layer if time
  allows before November.
