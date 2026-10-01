# Human Work Landscape — every dataset, study, and solver that records or imitates humans

> Swept 2026-08-15. Companion to `Research Synthesis.md` (§5 vocabulary-vs-policy, §10 fit-the-policy-from-outside).
> Everything below is THIRD-PARTY prior art — reference only, never paste into the CC0/MIT-0 submission.
> Same rule as concept_sources and nvarc_reference.

## Why this note exists

`Research Synthesis.md` §5 says the missing piece is the POLICY — the human prior over which
program is relevant here — and §10 argues it cannot be authored from introspection, only fitted
from OUTSIDE. This note is the outside view: the field has been recording humans solving ARC
since 2021, at pixel-edit resolution, with written explanations attached. That data is the
observable trace of the policy we said we could not author.

The repo covers the SOLVER landscape well (NVARC, BARC, Hodel, Icecuber, DreamCoder, MDL,
graph DSLs). It had nothing on the HUMAN landscape. That gap is now closed here.

---

## 1. Human action traces — the search order itself, step by step

### H-ARC — the big one
- Over 1700 participants, 15,500+ solution attempts.
- Full ARC-1 coverage: all 400 training + 400 evaluation tasks. Largest human evaluation to date.
- Records step-by-step behavioural action traces from the solving interface (pixel-level edits),
  PLUS a natural-language description of the rule each person inferred.
- Paper: https://arxiv.org/pdf/2409.01374
- Journal version (Nature Scientific Data, 2025): https://www.nature.com/articles/s41597-025-05687-1
- Code/data: https://github.com/le-gris/h-arc
- Browser viewer: https://arc-visualizations.github.io/

### ARCTraj — object-level trajectories
- ~10,000 trajectories, annotated with task identifiers, timestamps, and success labels.
- 400 ARC-1 training tasks. Collected through the O2ARC web interface (see §5).
- Difference from H-ARC: records temporally ordered OBJECT-level actions rather than raw pixel
  edits — closer to "what did they grab and move", which is the level our DSL speaks.
- Paper: https://arxiv.org/html/2511.11079v1

### Johnson, Vong, Lake, Gureckis 2021 — the original study
- "Fast and flexible: Human program induction in abstract reasoning tasks", CogSci 2021.
- 40 ARC tasks, average human accuracy 84%. Action sequences recorded during solving.
- Includes an ERROR analysis — what people get wrong, not only what they get right.
- Paper: https://arxiv.org/abs/2103.05823

### Misalignments in human trajectories
- "Addressing and Visualizing Misalignments in Human Task-Solving Trajectories".
- About where human traces DISAGREE with each other on the same task, and why.
- Matters because it bounds how much a single trace can be trusted as ground truth.
- Paper: https://arxiv.org/pdf/2409.14191

### DARC — recursive decomposition
- Humans breaking tasks into sub-steps; a decomposition annotation set.
- Site: https://iprc-dip.github.io/DARC/

## 2. Human language — the rule stated in words

### LARC (Language-annotated / Language-complete ARC)
- Humans instructing OTHER humans to solve tasks through natural language only, no demonstration.
- Successful instructions exist for 88% of the ARC-1 tasks.
- Annotated with a taxonomy of 17 concept categories — including procedure, spatial reference,
  validation, loop, and object detection.
- The paper's own finding: these instructions resemble programs but constantly reach for
  primitives and communicative moves that are NOT directly executable. That gap is a direct
  measurement of what a hand-built DSL is missing.
- Paper: https://arxiv.org/abs/2106.07824
- Data: https://github.com/samacqua/LARC
- CogSci version: https://escholarship.org/uc/item/3r6807ct

### MC-LARC (Shin & Lee, 2024)
- Reformulates ARC into multiple-choice questions.
- CAUTION flagged in the literature: the multiple-choice format introduces shortcut pathways
  that the open-ended original does not have. Weaker evidence than LARC proper.

### Core knowledge objects in reasoning and language use
- Ecanow, Wong, Acquaviva, Pu, Kryven, Tenenbaum (2021).
- What objects and relations humans NAME when describing these tasks — the vocabulary side of
  core knowledge, measured rather than assumed.
- Paper: https://escholarship.org/uc/item/90f1p4kf

## 3. Human performance — how well, and how differently, people do

### ARC-AGI-2 official human testing — THE ONLY HUMAN DATA ON OUR ACTUAL TARGET SET
- 400 people tested across 1,417 unique tasks.
- Each task attempted by roughly 9-10 participants on average (range 2-10).
- Calibration bar: every task must be solved by at least two independent non-expert testers
  within two attempts each — deliberately matching the AI pass@2 rule.
- Result: 100% of ARC-AGI-2 tasks were solved by at least two non-expert humans from the public.
- Aggregated by task, 75% of human attempts succeeded; across all attempted test pairs, 66%
  succeeded. Panels averaged about 60%.
- Participants also filled a survey on demographics, problem-solving habits, and cognitive state.
- Dataset: https://huggingface.co/datasets/arcprize/arc_agi_2_human_testing
- Report: https://arxiv.org/pdf/2505.11831

### Comparing Humans, GPT-4, and GPT-4V (Mitchell, Palmarini, Moskvichev, 2023)
- Direct human-vs-machine comparison on ConceptARC.
- Paper: https://arxiv.org/pdf/2311.09247

### Do LLMs solve ARC visual analogies like people do? (Opiełka, Rosenbusch, Vijverberg, Stevenson, 2024)
- Compares children, adults, and LLMs on the same items.
- Children and adults beat most LLMs. The error analysis is the valuable part: LLMs fall back on
  COPYING part of the analogy — the same fallback young children use, and the one adults do not.
  Adults use abstract rule-based reasoning instead.
- Reads as a direct measurement of the policy gap: shallow combinatorial moves versus rule-first.
- Paper: https://arxiv.org/abs/2403.09734

### ChildPlay (adjacent)
- Child-cognition-style evaluation of models beyond text. Context, not ARC data.
- Paper: https://arxiv.org/abs/2407.11068v3

## 4. Solvers that already try to be human

### Object-centric Decision Transformer — closest existing work to our thesis
- "Unraveling the ARC Puzzle: Mimicking Human Solutions with Object-Centric Decision Transformer".
- Imitation learning over human O2ARC traces, plus a Push-and-Pull clustering method for object
  detection. It explicitly learns the human ACTION SEQUENCE, not the answer.
- This is the neural version of the thing we want symbolically. Read it for what it learned about
  human action order, not for its architecture.
- Paper: https://arxiv.org/abs/2306.08204

### Neurodiversity-inspired solver (Ainooson, Sanyal, Michelson, Yang, Kunda, 2023)
- Visual imagery plus program synthesis, framed around how differently-wired humans reason.
- Paper: https://arxiv.org/abs/2302.09425

### PQA: Perceptual Question Answering
- Human perception framing for abstract visual tasks. Adjacent, older.
- Paper: https://arxiv.org/abs/2104.03589

## 5. Platforms — where the traces come from, and how to collect our own

- **O2ARC 3.0** — the web platform that produced the ARCTraj and Decision-Transformer traces.
  Solve and CREATE tasks. IJCAI-24 demo. https://o2arc.com/
- **ARCLE** — ARC restated as an interactive reinforcement-learning environment, with action and
  observation spaces shaped like the human interface (O2ARCEnv, ARCEnv, RawARCEnv). CoLLAs-24.
  https://ar5iv.labs.arxiv.org/html/2407.20806
- **Lab42 playground / editor** — interactive environment for exploring tasks by hand.
  https://arc-editor.lab42.global/
- **ARC-TGI (2026)** — human-validated task generators with reasoning-chain templates.
  https://arxiv.org/html/2603.05099v1

## 6. Where the field stands on human-likeness

- Living survey (2026): "The ARC of Progress towards AGI". https://arxiv.org/html/2603.13372v1
  Families it recognises: program synthesis, neuro-symbolic hybrid, pure neural/inductive,
  test-time adaptation. Its headline numbers: pure transduction caps around 40%, pure induction
  at most 79%, hybrids reach 80-94%.
  CAVEAT: those are PUBLIC-leaderboard numbers (any model, any cost). Per `Research Synthesis.md`
  §9 they are not Kaggle-legal — our track is capped near $0.42/task with no internet.
- THE FINDING THAT MATTERS: the survey has no category for cognitively-inspired, perception-first,
  object-centric-by-design, or human-trace-grounded architectures. Progress there is described as
  computational scaling rather than alignment with how people actually reason.
  The human-aligned niche is EMPTY, not crowded. That is the positioning for this project.
- ARC Prize 2025 Technical Report: https://arxiv.org/html/2601.10904v1

## 7. Honest gaps in this sweep

1. **ARC-2 has no action traces.** Every rich trace dataset above (H-ARC, ARCTraj, LARC, O2ARC,
   DARC) is built on ARC-**1**. For ARC-AGI-2 we have outcomes, timings, and demographics only.
   Any policy fitted from human traces is fitted on the EASIER benchmark and transferred.
2. **Eye-tracking is unverified.** The ARC Prize resources page links an eye-tracking video
   (https://www.youtube.com/watch?v=nsSuHWKz8d8) but the sweep found no corresponding paper or
   released dataset. Treat as a lead, not a source.
3. This is a link sweep, not a read. Nothing here has been downloaded, inspected, or checked
   against its own claims. Apply `Research Synthesis.md` §11.1 — verify from outside before
   committing to anything in this note.

## 8. The one cheap test, if any of this gets used

Same shape as the induction-vs-transduction test in `Research Synthesis.md` §11.1: take the tasks
our search FAILS, pull the human traces for those same tasks out of H-ARC, and look at what people
did first. If human first-moves are predictable from `inference.hpp` facts we already compute,
the policy head has a training signal and a supervision target. If they are not, this whole
direction is wrong and a week has bought that answer.
