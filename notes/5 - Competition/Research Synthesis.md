# Research Synthesis — ARC-AGI-2 landscape + strategy thread (2026-06-20)

> Durable context lives HERE, on disk — not in the memory system (user preference).
> Companion files: nvarc_reference/ , concept_sources/ , concept_sources/arc_agi2_chollet_axes.md

> =========================================================================================
> ## TOP PRIORITY — READ THIS FIRST: SHIP UGLY, THEN MEASURE  (full version in §11.2)
> The single biggest risk to THIS project is NOT a technical gap — it is elegant theory becoming a
> substitute for building. The exact traits that make this engine good (depth, craft, first-principles,
> owning the whole stack) tip directly into over-engineering and theory-as-procrastination — the gift
> and the trap are the SAME muscle. That is the documented personal failure mode here, so this rule
> outranks everything else in this file.
> - Real deadline: Nov 2, 2026. ~Daily submissions. The game is ITERATION CADENCE, not completeness.
> - The dumbest WORKING version of any idea below teaches more than another month of design notes.
> - Do NOT perfect the DSL. Measure, don't polish.
> - Litmus test: if you're reading/writing theory instead of shipping something measurable today, STOP
>   and ship the ugly version. The notes are the map; the submission queue is the territory.
> =========================================================================================

> ## THE OPENINGS — why this is winnable (Chollet's "regrettable choices" 😈)
> The benchmark was hardened against the OLD game (brute-force search). It left three doors open:
> 1. He killed search-as-SOLVER ("less brute-forcible") — but search-as-GENERATOR + VERIFIER is wide
>    open, and it's exactly the verified executor the neural winners lacked.
>    -> our DSL becomes a data factory + referee, NOT a solver.
> 2. He bounded ARC to FOUR Core Knowledge priors (1911.01547 §II.1.3) — finite, enumerable, programmable.
>    -> the atom set is CLOSED; "atoms done" is a finite checklist (op-coverage, §6), not an open hunt.
> 3. The winners open-sourced everything (NVARC repo; a 7M TRM that fits a P100).
>    -> the blueprint and the model are already on disk; we hold the expensive half (a fast VERIFIED
>    inductive engine).
> NET: the front door (search-as-solver) is bolted; GENERATION + VERIFICATION + a FINITE VOCABULARY are
> left wide open. Walk in. Keep eyes on the prize. 😈
> =========================================================================================

## 1. What the dead "nvarc-2025" Kaggle link was
A fork of NVARC's open-source ARC Prize **2025 winning** solution (CPMP = Jean-François Puget
+ Ivan Sorokin, NVIDIA KGMoN). People keep re-running it → new public highs (the 32.64% fork).
Go to source instead: https://github.com/1ytic/NVARC

2025 ARC-AGI-2 private podium:
- 1 NVARC 24.03% — synthetic data + test-time training + Qwen3-4B + TRM
- 2 ARChitects 16.53% — 2D masked diffusion + recursive self-refinement
- 3 MindsAI 12.64% — test-time fine-tuning, augmentation ensembles, tokenizer dropout
Humans ~75%. All winners are NEURAL refinement loops. NVARC has NO DSL — transforms are
one-off LLM Python, kept when >=8/20 implementations agree (consensus verification).

## 2. Local research artifacts (already downloaded)
- nvarc_reference/ — SDG prompts (summary/mix/generate_puzzle_output) + scripts.
  Augmentation (build_datasets.py) = 8 dihedral × random color permutation, applied to
  input+output (their color_mapping is in-place → cascades/collapses colors; use a LUT).
  New concepts come from `mix` (combine 2 tasks), NOT augmentation.
- concept_sources/ — Hodel arc-dsl (160 ops), BARC (163 seeds + common.py + 144 ranked
  concept tags), ConceptARC (16 groups), arc_agi2_chollet_axes.md.
LICENSE: BARC/Hodel/ConceptARC are MIT/Apache/research — reference only, never paste into
the CC0/MIT-0 submission. Full rules constraints (with section numbers, and how to re-read
the JS-only Kaggle page) now live in `Contest Rules.md`.

## 3. The four ARC-AGI-2 target axes (Chollet, arXiv:2505.11831) — see arc_agi2_chollet_axes.md
1 multi-rule (entangled) 898e7135 | 2 multi-step (iterate-until-done) cbebaa4b |
3 contextual (grid-derived branch) b5ca7ac4 | 4 in-context symbol (legend/lookup) e3721c99.
All four example tasks are in OUR eval set (the 120 we score 0 on).

## 4. Core Knowledge priors — Chollet, "On the Measure of Intelligence", arXiv:1911.01547 §II.1.3
FOUR systems, closed by design (ARC assumes ONLY these, no acquired/language/cultural knowledge):
1 objectness + elementary physics (cohesion, persistence, contact)
2 agentness + goal-directedness
3 natural numbers + elementary arithmetic
4 elementary geometry + TOPOLOGY (distance, orientation, in/out relationships)
Source behind it: Spelke & Kinzler "Core Knowledge".
Topology is NOT eval-only — it's enclosure/containment/connectivity. We solved a topology task:
00d62c1b via `InteriorColor[4]`. ConceptARC has an `InsideOutside` group.

## 5. The intellectual through-line (the part that was only in chat)

VOCABULARY vs POLICY (the spine):
- A DSL programs priors as VOCABULARY = what's possible. ✅ hand-codable; we did it.
- Humans also have a prior over what's RELEVANT here = a distribution over programs given the
  grid (perception that makes bindings salient + "a designer meant one clean rule"). This is the
  POLICY. In synthesis terms: DSL = hypothesis SPACE; the missing piece = the SEARCH POLICY.

AUTHORED vs FITTED params:
- Neural is just code (a trained net is matmuls you can transcribe to C++). The boundary was
  never code-vs-not-code; it's where the PARAMETERS come from: your head (introspection-bounded)
  vs fitted to data. The policy/perception ("popping") is below introspection → can't be authored,
  only fit. Training = a COMPILER from examples to weights (like trusting LLVM over hand-assembly;
  the output is still code).
- Caveat proven by Viola–Jones (below): authorable ≠ computable; learnable-cheap ≠ authorable.

THE BINDING WALL (why search loses on AGI-2):
- Binding = a variable whose value comes out of the puzzle (let x = extract(grid) in op[x]).
  Needed because one program must fit examples that differ in a recoverable value.
- General binding = searching over extraction FUNCTIONS, with the value INVISIBLE (only its
  downstream effect shows) → exponential AND weak pruning. Worse than depth.
- AGI-2 axis 4 makes the binding NOVEL PER TASK → defeats fixed menus AND learned libraries.
- Our DSL's scalar-purge / let-removal (DAG memoization) replaced common-subexpression reuse but
  NOT value-binding. Semantic prims (FitTo/CenterIn) are a CLOSED MENU of geometric bindings.

GENERATE vs SOLVE ASYMMETRY (why the pivot works):
- Solving a binding task = invert an unknown extractor from 3 examples (impossible-hard).
- Generating one = CHOOSE the extractor, emit consistent examples (a for-loop). Zero search.
- So: don't search the binding space — SAMPLE it to make data; a trained model amortizes the
  search into weights once, offline. The unsearchable curse becomes a sampling distribution.

SYMBOLIC TTT (test-time training has a pure-symbolic translation):
- Symbolic search already adapts per task (adapt the CHOICE). TTT adapts the WEIGHTS (the CHOOSER).
- Symbolic analog = closed loop: inference.hpp facts → set search knobs Φ (enabled ops, ordering,
  constant biases, constraints, task-macro library) → structural-diff = the "loss" → search a bit,
  measure which ops cut the residual on demos → upweight/prune/promote-macros → re-search. Iterate.
  "weights"=Φ+macros, "optimizer"=solve-measure-rebias, "training set"=the 3 demos.
- Ceiling: adapts priorities + composes macros from EXISTING primitives; cannot invent a primitive
  (binding wall persists). Adapts the chooser's PRIORITIES, not its VOCABULARY.

VIOLA–JONES = the thesis in one shipped system:
- Real-time face detector = hand-designed Haar features + a LEARNED weighting (AdaBoost picks ~6000
  of 180k features + thresholds from labeled faces) + cascade. Hand-features + learned selector.
- Pure-rule detectors (80s-90s) worked frontal/lab, collapsed in the wild → deep nets cleared the
  tail. Same head/tail shape as ARC. The irreducible learned piece is small: a weighting over
  hand-features. This IS "Viola–Jones for ARC" = our target architecture.

INDUCTION vs TRANSDUCTION (BARC's thesis — explore this, see Open threads):
- Induction = synthesize a program (our DSL). Transduction = directly predict the output grid (TRM).
  BARC showed they solve DIFFERENT tasks and ensemble well. Likely the real coverage unlock.

## 6. Strategic conclusion + roadmap
Our CUDA/C++ DSL solver (op count lives in the NODE_PRIMITIVES X-macro, environment.hpp; beam, max depth ~8): ~9% on the 633-task benchmark, 0/120 on
AGI-2 public eval (by design — AGI-2 is "less brute-forcible"). The architecture (unarized,
scalar-purged tree-of-unary-ops) is optimal for fast AGI-1-style search and a hard ceiling for
AGI-2's loops/conditionals/bindings.

Pivot — change the DSL's JOB, not its design:
1. Keep DSL as fast shallow solver + correct-by-construction VERIFIER/GENERATOR (its real leverage;
   it's the verified executor NVARC paid an LLM to approximate).
2. Atoms are FINITE (four priors). Finish op coverage toward Hodel∩BARC-frequency, then STOP —
   after "atoms done" every gain is navigation, not vocabulary. [NEXT BRICK: 33-vs-160 op checklist]
3. Generate deep, concept-diverse, human-shaped (designer-style) task families — one generator per
   Chollet axis (generation is easy where solving is impossible). Augment (D4 × color, via LUT).
4. Learn the small fitted piece: either a POLICY HEAD over inference.hpp's facts (Viola–Jones style,
   neural-guided search) or a depth-amortized transducer (TRM, 7M params, fits P100/12h, no external
   API). TRM's ~16 recursion steps = learned fixed-count depth (the fold our DSL can't express).
5. Consider induction(DSL) + transduction(TRM) ensemble for coverage.

## 7. Honest limits
- Symbolic TTT can't invent primitives; the novel-binding tail needs the learned piece.
- The learned piece is SMALL (limited human stuff = low-complexity prior = why TRM is 7M) but not
  zero, and not hand-authorable (it's the sub-introspection saliency/weighting).

## 8. Open threads / things we have NOT explored yet
- INDUCTION+TRANSDUCTION ensemble (BARC paper "Combining Induction and Transduction") — they fail on
  different tasks; ensembling is plausibly the biggest single coverage win.
- THE EFFICIENCY FRONTIER — leaderboard is 2D (cost/task × score; ~$0.42/task, P100/12h). Symbolic
  search is CHEAP per task; TTT is expensive. We might win the efficiency axis even at lower raw score.
- 2-GUESS scoring favors candidate DIVERSITY — symbolic search yields a ranked candidate list for free.
- VERIFICATION ASYMMETRY (NP-like): verifying a program is cheap, finding it is hard. Our executor is
  the verifier → why generation is our superpower.
- Object-centric / relational PERCEPTION front-end (slot-attention style) to make bindings salient
  the way human vision does — shrinks the binding space for both DSL and a learned model.
- DreamCoder wake-sleep LIBRARY LEARNING to grow the op set automatically (offline base "weights").
- Test-time augmentation + VOTING at inference as a cheap score boost.
- ARC-AGI-3 = a different game (interactive/agentic: exploration, planning, memory) — field is moving.

## 9. Addendum — efficiency, NP, induction/transduction, ensemble (added later in session)

EFFICIENCY FRONTIER — there are TWO distinct "ARC-AGI-2" leaderboards:
- PUBLIC (arcprize.org/leaderboard): any model + scaffolding, 2-D cost/task × score. Early-2026 points:
  Gemini 3.1 Pro + code-evolution ~95.1% @ $8.71/task (imbue); Jeremy Berman ~80% @ $8.42 (~25x cheaper
  than prior leaders); Gemini 3 Flash 61.4% @ $2.42. Humans ~$17/task.
- KAGGLE (our $700K prize): P100, 12h, NO internet, NO external APIs, ~$0.42/task HARD CAP. The public
  95%/$8.71 runs are ILLEGAL here ($8.71 ≈ 20x our whole per-task budget; frontier APIs banned). Our game
  is a FIXED POINT at ~$0.42/task; at that cost the public frontier collapses to the small-model/NVARC
  region (~24-33%). In Kaggle, efficiency is a GATE not a scored axis — fit under it, then ranked by score.
  Our cheapness = HEADROOM = optionality (afford ensembling + TTA-voting + some TTT within budget where a
  heavy neural pipeline is compute-bound). Sources: arcprize.org/leaderboard, sanj.dev, imbue, madebynathan.

ARC IS NP-LIKE (the verification asymmetry, #3 explained):
- Finding a program consistent with the 3 demos = hard search; CHECKING a candidate = trivial (run on demos).
  Payoffs: (1) generation is FREE (construct a program + run = guaranteed-correct pairs, no search);
  (2) propose-and-verify (let a learned/expensive part propose, filter with the cheap symbolic executor);
  (3) verify candidates for the 2-guess submission.
- SUBTLETY: consistent-with-demos != correct-on-test (we have no test answer to check). Many programs pass
  the 3 demos; only one generalizes. The verifier shrinks the haystack to demo-consistent; picking the
  generalizer still needs priors/Occam = the policy. Owning a fast verifier = owning the lever; our executor
  IS that oracle. This is the deep reason "generate is easy, solve is hard."

INDUCTION vs TRANSDUCTION — citation: Li, Hu, ..., Kevin Ellis, arXiv:2411.02272 "Combining Induction and
Transduction for Abstract Reasoning" (PDF: cs.cornell.edu/~ellisk/documents/arc_induction_vs_transduction.pdf).
- Induction = infer an explicit PROGRAM, run it on the test (our DSL; code-LLMs; Gemini code-evolution).
  Self-verifying.
- Transduction = directly predict the test OUTPUT grid, no rule (TRM, answer-emitting LLMs). NOT self-verifying.
- Finding: SAME data + SAME architecture -> solve DIFFERENT tasks. Induction = precise computation + composing
  concepts (axes 1-2); transduction = fuzzier perceptual concepts (axes 3-4 = our 0/120 tail). Ensemble ~= human.
- We already own the HARDER half (a fast VERIFIED inductive solver). Adding a TRM = additive coverage on exactly
  the tasks induction structurally can't reach.

ENSEMBLE SKELETON (per test task):
1. Induction (DSL search, budgeted) -> candidate programs, each verified pass/fail on the demos.
2. Transduction (TRM + augment + vote) -> candidate output grid(s). Soft-check: run TRM on the demo INPUTS;
   if it reproduces the demo OUTPUTS, trust it more.
3. Arbitrate (verifier = referee): an induction program passing ALL demos = high-precision guess;
   TRM top-vote = coverage guess.
4. Two guesses = one per engine (they fail on different tasks -> free diversity; the 2-guess rule rewards this).

## 10. Sub-introspection vs Gödel / halting / Rice (the closing riff)

Theme of the whole thread = the POLICY/perception ("popping") is below introspection: a system produces a
judgment but can't author/read the mechanism that produced it (true for humans AND for an LLM judging its own
output). Question: is that the halting problem / Gödel incompleteness, or not?

VERDICT: it RHYMES with them but ISN'T them — and the difference is decisive, in our favor.
- Gödel (can't prove all truths about itself / own consistency), Halting/Turing (can't decide if arbitrary
  programs halt), RICE'S THEOREM (can't decide any non-trivial SEMANTIC property of a program from its code —
  the closest cousin: "what does my own policy compute?") = NECESSARY, unconditional limits. Common engine:
  diagonalization / self-reference. No wiring, no instrument escapes them. Ever.
- Sub-introspection (the "popping", activation-blindness) = CONTINGENT, architectural inaccessibility. The
  meta/verbal layer just isn't WIRED to read the object layer; not a paradox. BREAKABLE: mechanistic
  interpretability reads a net's internals; fMRI/electrodes read the brain. Unexposed != undecidable.

THE DECISIVE TEST: undecidable means NO workaround (that's the definition). Sub-introspection HAS an obvious
workaround — look from the OUTSIDE (fit a model of the policy from data, probe it, instrument it = Viola-Jones
learning the weighting, training-as-compiler, interp tools). The existence of the workaround PROVES it's not
in the Gödel class. If it were really halting, the whole pivot (learn the policy you can't author) would be
forbidden by theorem. It isn't forbidden — it's available — therefore not Gödel.

THE TRUE FLOOR (where it genuinely IS Gödel-family): COMPLETE, certified, self-contained self-knowledge —
a system perfectly modeling AND trusting all of itself (Rice + diagonalization + Löb's theorem for the
self-trust part). That is impossible. But we never needed total self-knowledge — only a good-enough EXTERNAL
model of our own policy, which no theorem blocks.

UPSHOT: the "can't author the policy" issue isn't a wall — it's a window we're standing on the wrong side of.
Gödel says some windows are painted shut forever; this one opens from outside. This is the formal license for
the entire strategy: don't introspect/author the policy — observe and fit it externally.

## 11. Working discipline (for whoever reads this later, incl. future-me)

1. THESE NOTES ARE A HYPOTHESIS, NOT GOSPEL. This synthesis was produced in ONE session by an
   assistant that was confidently WRONG several times (face detector, the prior count, "data-dependent
   control"). Apply this doc's own lesson — don't trust the clean story, VERIFY FROM OUTSIDE.
   Cheapest test before committing to the pivot: actually check whether a transducer catches the
   tasks our search FAILS. If it doesn't, the induction+transduction thesis is wrong for our data —
   better to learn that in a week than a quarter.
2. SHIP UGLY, THEN MEASURE. The biggest risk is elegant theory becoming a substitute for building.
   Real deadline (Nov 2, 2026), ~daily submissions: the game is ITERATION CADENCE, not theoretical
   completeness. The dumbest working version of any idea here teaches more than another month of design
   notes. Do NOT perfect the DSL. Measure, don't polish.
