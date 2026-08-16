TO DO

Make sure arity 2 is killed
Cuda
Start submiting to competetion
Kaggle uses g++ 11.4.0 on Ubuntu 22.04.
To match exactly, your best bet is WSL with Ubuntu 22.04:
Do you have WSL already installed?
If so, what distro?
We can set up Ubuntu 22.04 in WSL with g++ 11.4.0, CUDA 12.8, and everything
matching Kaggle's environment exactly. That way your local builds will behave
identically to Kaggle.


MERGED FROM notes/to_do.txt

log other / all events

[BEFORE CONTEST SUBMISSION] evaluate_test_after_solve (solver_executor.hpp
~1274): when test has no ground-truth output (output.width==0), code skips tree
eval but still calls update_example_views with an empty grid -> blanks BEST cell.
Fix: run execute_tree_via_flat in the else branch so the prediction shows. Dead
path today (all loaded puzzles have ground truth) -- only matters if running on
unlabeled/submission-mode puzzles.

spurious "SIMPLIFIED SOLUTION FOUND: N nodes (was N)" log lines
(solver_executor.hpp ~961, the else-if branch). Fires whenever
promotion_count==0 && !is_test_solve, with no p.is_solved gate and no actual
node-count drop check. Cosmetic only -- fix by gating on p.is_solved &&
node_count < p.best_node_count.

[BEFORE CONTEST SUBMISSION] audit all writes to active_example_count. Current
sites in solver_executor.hpp: 1033 (reset to 1, safe), 1045 (jump to full size
on solved-puzzle reload, fixed), 1140 (best_tree promotion, fixed), 1368
(try_promote_examples, covered by callers). Any new path that grows
active_example_count must call cuda_dsl_invalidate_examples() before the next
batch eval, otherwise the GPU's main template cache will only have the old prefix
and the kernel will score against stale slots -- reporting solved=true for
candidates that don't actually solve. Symptom: false TASK SOLVED on training
set, GUI shows mismatched grids. Bug pattern: lazy upload + single bool
"uploaded" flag. Structural fix would be tracking an int "uploaded_count" and
re-uploading deltas instead of relying on callers to invalidate.

allow model to fix correctness from cpu basline backup
all gpu stuff be logged so models can read it
consolidate score weights into one header (currently duplicated in
environment.hpp, cuda_dsl.cuh, cuda_solver.cu -- silent drift risk)
consolidate benchmark in 1 file

restore dropped binary ops as unary-with-baked-context (pattern:
ProjectOnSeeds for Project)
  Fill (needed by 007bbfb7, 017c7c7b) -- 2nd operand is usually Input/Palette
  Nearest (needed by 045e512c-style, 05f2a901) -- 2nd operand is usually
  Palette/Input
  Align (needed by 05f2a901) -- 2nd operand is usually Seeds/Palette
  Connect -- check what 2nd operand pattern looks like in checkpoint solutions
  08ed6ac7's deep Overlay chain has DERIVED 2nd operands -> needs real binary
  dispatch refactor, not the bake-context trick

log beam contents
fix tree prefix or bracket styles

lookahead scoring: scorer currently only sees ctx.output, so transform-only
chains (Filter, Split, AlignToCanvas, ...) are invisible to the gradient --
search can only navigate via printer ops (Union/Intersection/Difference).
Symptom: Input -> Union scores +61 while Seeds -> Union scores -43, even though
Seeds is the right path; beam chases Input and inflates depth (05f2a901 ended up
depth 13 vs old depth 4).

score the hypothetical canvas-if-we-Unioned-the-current-list-right-now alongside
the real canvas. Gives transforms a real gradient.

then bias by precision: penalize hypothetical paints whose pixels are
wrong-colored, so Seeds (few correct) beats Input (many correct + many wrong).

one extra list-walk per node eval; should massively shorten typical solutions.


The biggest win is APS consistency. CV is 0.86 (similar variance) but the floor
matters more -- no more 0-APS dead seconds because the solver isn't grinding the
visited-set's DRAM stalls.


Meta: the right question is upstream of the question being asked
You ask "why is X slow?" The actionable answer is usually upstream -- "X is slow
because Y produces too many of them, and Y produces too many because Z's design
was conservative."

Pattern: profile data is local; the fix is structural. The hot function is
rarely the function to fix. It's the function called by the function that calls
the function that's hot.

You saw this with visited.contains -- 27% in the profile, but the fix wasn't in
contains, it was in the generator that produced too many candidates to check. Fix
the upstream and the downstream stops being hot.

The deeper move: when something is hot, ask "who's calling this so much?" Not
"how do I make this call cheaper?"

suppport -1 variants without hacks in


we need gui to show the mid tree results in tree view or smth.

IDEA: output-inferred delta op. Instead of hardcoding a grow/expand rule, infer
the delta (what changed from input to output) from training pairs, expressed as
offsets relative to each object's center. Then stamp that delta at new seed
locations. The search finds the right source expression; the output match
validates it. No fill rule parameter needed -- the rule is implicit in the
example pair. Open question: does scale-invariance matter (same rule applied to
objects of different sizes)?

IDEA (from 09c534e7, 2026-07-10): weigh damage in the free-search score. The
lock path already scores per layer (commit only on zero-wrong-cells prefixes,
then score the residual), but the free path scores the aggregate canvas, so
patch-style chains (paint wrong, repair later: Input Union ... Outline
Difference Target[1]) can outrun clean layerings. Possible knob: penalize
overwriting already-correct cells more than the flat pixel score does, so
patch-style only wins when genuinely shorter, not merely greedier. Risk:
patch-style freedom may be carrying puzzles with no clean layering -- needs
benchmark evidence before touching weights.

REVISIT (lock ladder can grow but never shrink). Observed (2026-07-10): "i
mostly see deep locking on new examples and no proper refreeing".
Example puzzle: 0a1d4ef5.

Why, for a fresh reader: a lock (solver_beam.hpp ~223) is a committed prefix
every candidate must build on. It has only two exit doors, and neither fires
in the observed case:
  1. Full puzzle reset.
  2. Promotion re-verify (solver_executor.hpp try_promote_examples ~2094):
     after new examples activate, the lock is re-checked -- but the bar is
     only "still paints zero wrong cells on the widened set". That is a
     necessary condition, not evidence the layer is right: an overfit layer
     usually passes on new examples simply because painting nothing wrong is
     easy (worst case it paints less there). It survives, candidates are
     still forced to build on it, and the ladder deepens around a
     decomposition the wider set never endorsed.
Meanwhile the only stall response is focus rotation (solver_beam.hpp ~237):
after 8 stagnant levels it changes which COLOR it chases, and after a full
circle it drops the focus. It never touches the ladder. So stagnation with a
bad lock has no path back: deep locking, no refreeing.

Fix sketch: at the moment the focus circle completes with no progress
anywhere (where the code currently just drops the focus), peel the TOP layer
off the ladder instead, re-run layer_fit on the remainder, resume. Repeated
stalls peel deeper -- the ladder becomes something the search can back out of
one commitment at a time, instead of a ratchet. Same peel applies at
promotion re-verify failure, which today drops the WHOLE ladder even if only
the top layer overfit.

DSL missing: diagonal line drawing. ProjectOnSeeds marches axis-aligned only. No
op handles 45-degree lines -- neither Bridge (fills bounding box between
same-color objects on same axis) nor ProjectOnSeeds (stamps a whole shape,
marches to grid edge unconditionally) covers diagonal single-pixel lines. Also
missing: ray-until-obstacle (stop at another object, not just wall) and
gravity/slide.
