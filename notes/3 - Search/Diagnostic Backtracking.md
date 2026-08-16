Diagnostic Backtracking
=======================

Core Idea
---------
Beam search asks: "is this candidate good?"
Diagnostic backtracking asks: "which step in this candidate broke it?"

The first gives you a binary keep/drop.
The second gives you actionable repair information.
Same compute, much more signal.


Why It Matters
--------------
Standard beam search prunes by final score and keeps top-K survivors.
It does NOT diagnose failures. A beam can lose a correct prefix because
the wrong continuation dragged the chain's score down — even if the
prefix itself was on the right path.

Humans don't restart from scratch and don't run backward search when
they hit a dead end. They:
  1. Recognize the dead end ("output doesn't match")
  2. Locate the suspect step ("shape wrong → Largest was probably the
     wrong choice")
  3. Backtrack to THAT step (keep the working prefix)
  4. Try a focused alternative (swap just the suspect op)

This is credit assignment over the chain — figuring out which op
caused the failure, not just whether the whole program failed.


Why It's Better Than The Alternatives
-------------------------------------
- Cheap: scores are already computed on GPU. Intermediate scoring is
  a small kernel extension.
- Local: no architectural rewrite. Diagnosis is per-candidate, retry
  uses the existing slot mechanism.
- Composable: stacks with masking, DSL improvements, and any future
  bidi work. Time invested is never wasted.
- Sidesteps the destructive-ops problem: destruction is fine if you
  can localize WHERE in the chain the destruction went wrong.
- Aligned with human solving — usually a sign of the right abstraction.


Implementation Sketch
---------------------
1. GPU kernel changes (cuda_solver.cu, dsl_exec_and_score_kernel_v2)
   - After each op in gpu_eval_program_warp, compute a quick
     canvas-vs-target score using existing data.
   - Store as score_per_step[GPU_MAX_OPS] alongside final score.
   - Cost: roughly N× the current single end-of-chain score, where
     N is program length. Real cost; needs measurement.

2. Host-side diagnosis (solver_executor.hpp)
   - When a candidate underperforms, scan score_per_step for the
     largest score drop — the suspect step.
   - Tag the candidate's tree with the suspect path.

3. Slot prioritization (prepare_next_level_slots)
   - Currently caps at 30 slots uniformly.
   - Instead: weight slots by suspect-score, putting most-likely-broken
     paths first. Spend retry budget where it's most likely to help.

4. Targeted retry (generate_phase_1)
   - When refining a suspect slot, bias toward primitives that fix
     the kind of damage that step caused.
   - E.g., if the score drop was in histogram error, prioritize ops
     that adjust color counts. If the drop was in dimension penalty,
     prioritize size-changing ops.


Cost / Risk
-----------
Cost: per-step scoring on GPU is ~10-50× the current single
end-of-chain score, depending on program length. Real GPU cost.

Risk: low. The diagnosis is advisory — it biases slot ordering,
doesn't eliminate slots. If the diagnosis is wrong, you fall through
to normal slot iteration. No solution becomes unreachable.

Worth prototyping after cheaper masking work shows its limits.
The idea is durable; not urgent.


Connection To Other Ideas
-------------------------
- Subsumes "intermediate scoring" — same data, but used for backtrack
  guidance rather than as a beam selection criterion. Better framing:
  intermediate scores tell you WHERE to retry, not whether.

- Subsumes "Union as checkpoint" — natural checkpoints are wherever
  the score-per-step actually changes. No need to special-case Union.

- Complements op masking — masking biases initial choice; diagnostic
  backtracking biases retry choice when the initial pick fails.

- Complements bidi (if ever pursued) — knowing which step broke a
  forward chain tells you where the backward search should target.
