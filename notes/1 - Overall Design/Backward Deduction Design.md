# Backward Deduction vs Forward Beam — Why Output-First Is Algorithmically Faster

Design note. Captures the architecture argument behind moving (part of) the
solver from forward GPU beam search toward backward, output-first deduction.
Current solver = forward beam (`solver_beam.hpp` + `cuda_solver.cu`). This note is
about what's **MISSING** and why it matters. **Nothing here is implemented yet.**

## 1. The human method this formalizes

How a human (the user) actually solves ARC, in order:

1. **LOOK AT THE OUTPUT FIRST.** Read its type signature before anything else:
    - output size vs input size: same / smaller / larger / fixed-constant
        - same → recolor / overlay / local cellular rule
        - smaller → crop / extract region / count-summary / downscale
        - larger → tile / mirror / scale / pattern completion
        - fixed (e.g. always 3x3, 1x1) → counting / classification
    - color delta: which colors added/removed; is output a sub-palette of input?
      (colors in input but not output were probably scaffolding/distractors)
    - is output a contiguous sub-grid of input? → extraction; skip generative ops
2. **FIND CORRESPONDENCE:** which part of output came from which part of input.
   Kinds: cell↔cell (positional/geometric), object↔object (segment+match),
   region↔whole (extract / summarize), feature↔property (semantic, NO
   spatial map — e.g. "#red objects" → color of a 3x3). Noticing the ABSENCE
   of a spatial map and switching to semantic is itself a step.
3. **INTERSECT ACROSS EXAMPLES:** example 1 yields MANY candidate rules (most are
   coincidences). Example 2+ is a FILTER, not new data to map from scratch:
   keep only rules that also explain it. Survivors are usually ~1.
    - intersection EMPTY → wrong correspondence TYPE; backtrack to step 2,
      re-segment (cell→object, positional→semantic).
    - intersection >1 → underdetermined; go to step 4.
4. **TIEBREAK BY GENERALITY:** among survivors pick the MOST general / most abstract
   rule, NOT the shortest description. "Reflect across vertical axis" beats
   "shift right 1, wrap edge, recolor blue→red" even though the latter is a
   shorter token string. Proxy for generality = FEWEST special cases / FEWEST
   hardcoded magic constants / RELATIONAL over ABSOLUTE references. A symmetry
   has zero tunable constants; a hand-shift has one. Generality only breaks
   ties AMONG survivors; it NEVER overrides a failed example (step 3 is hard).

This loop "is usually enough" because step 1 prunes so hard that steps 2–4 look
at a handful of candidates, not millions. The work is front-loaded into reading
the output well.

### Step 2 UNBLOCKED 2026-07-25 — correspondence needed selection first

Step 2 (FIND CORRESPONDENCE) is the origin of the whole 2026-07 select/map/
compose + object-only saga, and it was buried under it. The thread, closed:

- Correspondence is the claim **"which input objects map to which output fate"**
  (object↔object, with an ARITY: 1→1, 1→many, many→1 — the "kinds" on line 22).
- To STATE that claim you must NAME an object set — and naming an object set IS a
  selection. So correspondence cannot be expressed until selection is a clean,
  object-level primitive. Selection is the LANGUAGE correspondence is written in.
- That requirement is what forced: unfuse select-from-act → object-only selection
  → the frame/Partition/reducer fixes (see [[Select Map Compose]], REVERSED
  2026-07-25). The saga was not a detour from correspondence; it was BUILDING THE
  LANGUAGE correspondence had to be spoken in.
- The one checkable signal for a correct correspondence, WITHOUT program
  lookahead: **does the marked object set map to ONE consistent output fate**
  (all become color X / all vanish / all stay) while the unmarked ones don't? If
  so the grouping is real, and its arity is the shape of that fate. This is the
  inference-side fact search should be HANDED, not rediscover per candidate (the
  speed move that started task-zero).
- STATUS: still unbuilt (this note is design). But the blocker (clean object-
  level selection) is now GONE, so step 2 is buildable. A ~10-row arity truth set
  was walked in the origin session (propose/correct, like the background walk) —
  its rows live in that session's transcript, not yet a committed truth file.

Both forks spring from ONE root — making selection a first-class op (which
correspondence demanded). Unfusing select from act created two problems at once:

1. **What does a selection MEAN?** → the granularity fight → object-only selection.
2. **How does a selection EARN its place in search?** → grammar. An unfused
   selection has NO score on its own (marking objects doesn't touch the canvas,
   so the scorer sees nothing). So a selection must be FOLLOWED BY an action to
   matter — and "select is always followed by act" IS a grammar rule. Generalize
   it and you get "each layer is a sentence": `GET (SELECT ACT)* COMBINE`. Grammar
   was not a tangent from correspondence — it is the SCORING consequence of the
   same unfusing correspondence required. (See [[Select Map Compose]] for the
   grammar's state and drift.)

## 2. What this is, in the literature

The loop above is, almost exactly:

- **VERSION-SPACE LEARNING** (Mitchell, candidate-elimination): maintain the SET of
  all hypotheses consistent with examples, as two boundaries S (most specific)
  and G (most general). Each example squeezes S/G together. Collapse to empty =
  wrong representation = backtrack. "Most general first" = reading off the G
  boundary. (Textbook ancestor; conceptual frame, not literal algorithm — ARC's
  hypothesis language is richer & noisier than classic candidate-elimination.)
- **PROSE / FlashMeta** (Microsoft), the mature deductive-synthesis framework:
  WITNESS FUNCTIONS = the inverse semantics of a DSL operator. Given a desired
  OUTPUT for an operator, a witness deduces a SPEC (constraint) on the operator's
  ARGUMENTS. The meta-algorithm (D4, "data-driven domain-specific deduction")
  propagates output constraints down to subexpressions via witnesses =
  "deductive backpropagation." This is the executable form of "look at output
  first." PROSE keeps the surviving program set as a compact version-space DAG
  and has a programmable RANKING layer (we'd point it at "most general").

**NOT SOTA-as-a-framework to adopt:**

- PROSE was DISCONTINUED by Microsoft as of 2025-10-14. Non-commercial license,
  .NET/C# NuGet (`Microsoft.ProgramSynthesis`), CPU, string/table domain. NO GPU,
  NO grid domain. Adopting it would mean throwing away our CUDA executor (the
  actual engine). So: **STEAL THE IDEA (witnesses), do NOT adopt the package.**
- There is NO off-the-shelf witness/version-space engine for ARC. On ARC,
  "SOTA" is split between LLM approaches and bespoke search solvers (Icecuber
  2020 winner, Hodel arc-dsl) — all FORWARD search like ours. ARGA
  (arXiv:2210.09880) = graph-abstraction DSL + constraint acquisition + tabu
  search, still forward. ILP-for-ARC (arXiv:2405.06399) = inductive, logic-based.
  NB color collision: "Vector Symbolic Algebras for ARC" (arXiv:2511.08747,
  Nov 2025) is NEURAL EMBEDDINGS — different thing from VERSION-space. Don't
  let the name send us down the wrong path.

Conclusion: the witness/version-space architecture is something we **BUILD** (in
CUDA C++ ourselves), not something we adopt.

## 3. The key argument: "but forward search also uses the output (the score)"

Objection: the beam's reward/score function compares candidate output to the
target output, so forward search IS guided by the output. Why isn't that the
same speed as backward deduction?

Answer: BOTH use the output. The difference is WHERE the output enters and in
what ROLE — and that one difference is an exponential.

- **FORWARD BEAM (current):** generate a candidate program FIRST (blind to output,
  ops picked from the grammar), RUN it on GPU, THEN the output grades it ("0.73").
  Output enters AFTER construction, as a FILTER/GRADE on something already built.
  → output is a **FITNESS FUNCTION**: tells you HOW GOOD a guess is.
- **BACKWARD DEDUCTION:** the output enters BEFORE construction and tells you WHICH
  OP to even consider. You don't generate `rotate` and check — the output BEING
  a rotation of the input is what PROPOSES `rotate`. Every other primitive is
  NEVER instantiated, never run on GPU, never hashed.
  → output is a **SPECIFICATION**: tells you WHAT the guess must be.

**Why SPECIFICATION beats FITNESS exponentially:**

- A SCORE is a SCALAR. It ranks candidates but does NOT tell you what to build
  next; you still enumerate the next op from the WHOLE grammar and score them
  all. Score prunes the frontier you ALREADY EXPANDED; it does nothing to stop
  you expanding the wrong frontier.
- A WITNESS returns a SET (the version space): it EXCLUDES whole branches BEFORE
  they are ever built. Scoring prunes AFTER expansion; witnesses prune BEFORE
  expansion.
- Cost: at branching factor b, depth d, forward explores ~b^d nodes that the
  score trims (O(b) build+execute work per level to reject). Deduction explores
  only the (often tiny) witness-permitted set. The win is ALGORITHMIC — it
  CHANGES THE EXPONENT by never instantiating most of the tree — NOT a constant
  factor. No amount of GPU throughput buys this; GPU only makes per-candidate
  build-then-score cheap, but you still pay b^d candidates.

**A scalar score is lossy about *why*:**

- "0.73" doesn't say WHICH cells are wrong or WHAT op fixes them — it compresses
  a whole-grid comparison into one float. So the beam has no gradient toward the
  answer; it must try ALL next-ops and re-score. The output's information is
  bottlenecked through one number.
- A witness reads the STRUCTURE of the mismatch ("these cells are color 4 that
  should be 7") and NAMES the op. It uses the full output, not a projection onto R.
- DECEPTIVE-SCORE / DEPTH-SURVIVABILITY: a correct program whose PREFIX scores
  poorly gets pruned before it can "click." This failure mode EXISTS ONLY IN
  FORWARD SEARCH. (See HANDOFF BidiColorMap note: we had to hand-verify that
  "each prefix is also locally best-scoring" so the beam wouldn't prune the
  correct deep program. With witnesses that check is unnecessary — a partial
  deduction is CONSISTENT or not, never "looks bad now, good later.")

**The sharp version:**

Forward search uses the output as a FITNESS FUNCTION (how good is this guess);
backward deduction uses it as a SPECIFICATION (what must the guess be). Fitness
ranks an exponential pile you still have to build; specification builds the small
pile directly. A grade is O(scalar) of information; a constraint is O(structure).

## 4. Where the advantage vanishes (be honest — this is why we keep the beam)

"Constraint not inverse": a witness need not return THE input; it returns a SPEC
(constraint) on the input. Tightness varies by operator:

- `rotate90` → input == `rotate270(O)`. Equality, single point. ("inverse")
- `concat` → exists split k: top==O[:k], bottom==O[k:]. Finite disjunction;
  still enumerable.
- `recolor` → b in colors(O); input == O with b-cells set to some a. Partially
  symbolic; pins b, leaves a constrained.
- `count` / `majority` / `fill` (LOSSY) → "any grid with count==5". Says ALMOST
  NOTHING; infinite set. DON'T expand it — CARRY it as a symbolic
  predicate and let it be CHECKED later (discharged by forward
  evaluation handing it a concrete grid, or by cross-example
  conjunction: count==5 & count==3 & count==8 constrains the RULE
  ("output encodes the count") even though no single one constrains
  the grid).

So the deduction speed advantage is FULL for invertible ops, SHRINKS TO ZERO for
lossy ops. Where a witness returns a huge set, it constructs nothing and you are
BACK TO SEARCH. The beam is exactly the right tool AT THE LOSSY JOINTS where
deduction degenerates into search anyway.

**Why "they just naturally meet" (no stop-rule needed):**

- Backward front tightens op by op until it hits a lossy op, where the
  constraint goes loose and STOPS narrowing — it naturally STALLS there.
- Forward front produces concrete facts that get CONSUMED by exactly those loose
  constraints (count==5 is loose until handed an actual grid → then a check).
- Neither side is told where to stop. The seam forms WHERE deduction runs out
  and evidence takes over = AT THE LOSSY OPERATOR, every time. The lossy ops
  are the JOINTS of the structure; the meeting point is SELF-LOCATING (the
  fixpoint of backward-propagate + forward-evaluate run to convergence). No
  materialize-vs-carry policy to tune: a loose constraint discharges the moment
  a concrete value reaches it, dictated by program structure, not a heuristic.

**DSL DESIGN LEVER** that follows: put TIGHT-witness ops DEEP in the program tree
(deduce through them) and LOSSY ops at the TOP (check them, don't invert them).
A `count` should be the OUTERMOST op — applied last forward, discharged by
checking backward. A lossy op buried mid-tree stalls deduction.

## 5. Where this leaves our code (current = forward GPU beam, see HANDOFF)

`beam_solver` (`solver_beam.hpp`) is a FORWARD program-synthesis beam search:
generate program trees → execute on CUDA (`cuda_solver.cu`) → score vs target →
keep top ~1024 → refine slot/level, hill-climb. This is the GENERATE-FORWARD-AND-
SCORE / search-first approach — the OPPOSITE control flow from sections 1–3.
Symptom already logged in HANDOFF: hand solutions run ~10–13 nodes for tasks a
human does in 2–3 mental steps ("structural DSL tax"). That depth is the cost of
forward search composing primitives blindly; backward deduction collapses it
because it skips the dead compositions.

You CANNOT bolt witnesses onto a forward GPU beam — opposite control flows.
Options (not yet decided):

1. **CHEAP / fits Kaggle deadline / keeps GPU as the engine: OUTPUT-FIRST
   PRUNING.** Use the output type-signature (size relation, palette delta,
   is-subgrid) to MASK the op set and SEED the beam PER PUZZLE. Captures step 1
   (highest-value single idea) without changing the engine. (cf. existing
   design-only `3 - Search/Op Masking.md`.)
2. **RIGHT / big build:** a separate CPU witness/deduction FRONT-END that runs
   BEFORE the beam. Invertible ops (rotate/crop/tile/recolor — most of our
   DSL) deduced backward on CPU; only the lossy/compositional residue falls to
   the GPU beam. This is "two fronts meet" made real and directly attacks the
   node-depth complaint. The GPU beam is NOT thrown away — it is the forward/
   verification half that deduction FALLS BACK TO at the lossy joints.
3. **PHASED:** ship (1) now, prototype (2) as the next-gen engine.

**RECOMMENDATION: (1) immediately, (2) as the real bet.**

OPEN PREREQUISITE before committing (1) vs (2): audit which of our SPECIFIC GPU
ops have clean/tight witnesses vs lossy ones. Feasibility of (2) is a week vs a
quarter depending on that ratio. Need to read `cuda_solver.cu` (op semantics) +
the op_masking / DIFFERENCEL design notes to build the per-op witness-tightness
table. NOT yet done.
