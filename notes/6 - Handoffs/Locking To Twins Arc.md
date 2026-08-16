# Locking To Twins Arc (2026-07-11 → 07-13)

Two days of chasing non-convergence ended in one root cause and one small fix.
Everything else built along the way is now the safety net around it. This is
the full chronicle — every problem, its evidence, its fix, and what was tried
and deleted — so nothing has to be re-derived.

## The symptom

1. 007bbfb7 (solved pre-locking in 1.5s / 256K attempts, 4-node solution
   `Input RemoveColor[0] Fractal AddColor[0] Union`) burned 25M attempts over
   90+ seconds, frozen, never converging.
2. 009d5c81 (known solution `Target[0] Union InputPalette ShapeColorMap
   RemoveColor[1] Union`) swung between 32 seconds and 29+ minutes on
   identical code — a lottery decided in the first second of each run.

## Chapter 1: the lock deadlock (007bbfb7)

1. Evidence: log showed PROMOTED to 2/5 at 1.24s, then LAYER LOCKED at 1.25s
   with colors [0 4 6] done — then 25M attempts of nothing. The locked prefix
   solved examples 0-1 by coincidence (palette-fractal bbox accident, see
   Open Items) and could never solve example 2.
2. The state was terminal BY CONSTRUCTION: all colors of the active set were
   done, so no new color could finish → no new lock could commit
   (try_lock_layer requires finishing a new color). Every candidate carried
   the frozen prefix, which failed example 2 → promotion always returned 0.
   The overfit re-verify only ran AFTER a successful promotion → nothing
   could ever drop the lock.
3. Fix: locks are vetted against the FULL train set at commit time
   (layer_fit over p.train.size(), not the active set). Principle: a lock is
   permanent — the active set has no authority to certify permanence. The
   practice quiz cannot sign the diploma.
4. Deleted as a consequence: the post-promotion overfit re-verify (widening
   the active set can no longer invalidate a lock that passed all pairs).
5. Result: 007bbfb7 back from never to 3.3s (later 1.38s with coverage
   ranking, later 6.3s pure flat — see Chapter 7).

## Chapter 2: lean locks and the reduction cliff

1. Problem A — junk frozen into prefixes: lock 1 froze as
   `Target[0] FlipVertical Union` with a provably output-neutral FlipVertical
   inside (flip of a uniform canvas snapshot). Every junk op in a lock steals
   depth budget from all later layers.
2. Problem B — greedy leave-one-out reduction cannot remove ops that only
   cancel jointly (`FlipHorizontal Fractal FlipHorizontal`): removing either
   flip alone breaks the tree, so both survive. (A full-subset variant was
   tried long ago and shelved as "no observable difference" — the flip pair
   was the live counterexample that unshelved it.)
3. Fix A: leave-k-out reduction with a depth dial. Hot path (every new best)
   stays k=1. Lock commits and full-train solves run full depth — every k,
   generic combination walk, restart from k=1 after any hit, cost-guarded at
   ~20k combinations per sweep.
4. Problem C — the cliff, found by instrumentation ([LOCK-REDUCE] /
   [LOCK-PREFIX] probes, still in the code): score-judged deep reduction on
   `Target[0] FlipVertical Union` returned just `Target[0]` — it deleted the
   DEPOSIT, because unpainted cells score leniently (provisional paint), so a
   blank canvas scored no worse. The gutted tree failed lock qualification
   (not canvas-rooted) and the all-or-nothing fallback committed the RAW
   tree, junk flip included. One illegal deletion discarded all legal ones.
5. Fix C: lock-bound (deep) reduction judges deletions by CERTIFICATION —
   same cells explained, same colors finished (layer_fit), beam score
   excluded entirely. Removing the flip passes (288 cells stay), removing
   the deposit fails (288 cells vanish). No fallback cliff possible.
6. Aside from the same session: the finale reduce also runs deep now, so the
   stored solution is the shortest train-equivalent SUBSET of the found tree.
   Shortest subset ≠ shortest equivalent program — crossing between spellings
   (e.g. stamp-then-repair vs clean-then-stamp on 007bbfb7) needs rewrite
   moves, not deletions. Documented in notes/3 - Search/Substitution
   Reduction.md; not built.

## Chapter 3: correct-but-greedy locks (009d5c81)

1. Evidence: lock froze `InputFiltered[0] AddColor[7] Union` at 0.4s — paint
   every shape 7. TRUE on all five pairs, so full-train vetting rightly
   passed it. But the real rule is "color each shape by its shape"
   (ShapeColorMap), and the cheap layer flattened the evidence: the beam then
   had to discover shape discrimination just for the leftover color-2 cells,
   on top of a frozen prefix. It saturated instead: stagnant counters cycling,
   every candidate visited, focus rotated through all colors and gave up.
2. Insight: vetting catches WRONG locks, not wrong DECOMPOSITIONS. A layer
   can be correct on every pair and still strand the remainder. Humans never
   hit this because they never ask "what is the cheapest way to finish color
   7" — they adopt the hypothesis with the least unexplained residue.
3. Fix (kept): coverage-ranked commits. The boundary sweep probes ALL
   qualifying survivors (probe mode on try_lock_layer: qualification +
   coverage without freezing) and commits the one explaining the most cells;
   ties → more new colors finished → fewer nodes. First-qualifier-wins is
   gone.
4. Probe redundancy fix (kept): probe verdicts depend only on (tree hash,
   lock state), so failed probes are cached in a hash set (probe_failed) and
   skipped until a commit or drop changes the lock state. Beam survivors
   were re-paying the full-train CPU eval at every boundary.
5. Limitation discovered immediately: coverage ranking only arbitrates
   between qualifiers present at the SAME boundary. The cheap layer wins by
   arriving earlier (0.4s) — the substantive layer does not exist in an
   infant beam. The race is across time, not within a boundary.

## Chapter 4: the ladder exit — built, shrunk, settled

1. Trigger (kept): two full focus circles without a single lock commit.
   Event-based, no timers — the circle is the system's own verdict ("every
   unfinished color got its turn and none advanced"). Any commit resets the
   count; commits finish new colors so at most 10 can ever land — no churn
   case exists. The existing FOCUS_ROTATE_LEVELS drives circle length.
2. First attempt — drop + opener ban by exact hash: the same 2-layer chain
   re-locked 55ms after the drop (a superset with a different hash walked
   past the ban).
3. Second attempt — ban by spine containment: then NOTHING could relock,
   because the entire post-drop population were descendants of the banned
   prefix. The beam was both saturated and collectively banned.
4. Third attempt — failure memory (record {colors done, colors stranded} per
   drop; veto any lock recreating that done-set with zero progress on its
   stranded colors): principled, terminating (finite states), and correct —
   but each cycle cost minutes and the cheap-opener space was large.
5. What survived (one strike): on the second circle, drop_ladder() does a
   COLD RESTART — beam, history pool, depth buckets, grid dedup, and score
   cache all wiped (anything kept recolonizes the trap: measured 25 minutes
   of "flat" search still rooted in the dropped prefix via history-pool
   fragments, plus the focus bonus warping scores), reseeds starters, and
   turns locking OFF for the rest of the puzzle. Layering gets one honest
   attempt; when it proves confused, flat search takes over for good.
6. Deleted: opener bans, containment walk, failure memory, re-arm logic —
   built 2026-07-11, observed to never pay off (every rebuilt ladder was
   another cheap trap), removed 2026-07-12.
7. Exit latency measured end-to-end on a live stall: strike at 17:47, flat
   promoted 2/5 by 22:50. Correct but expensive — circle length scales with
   level time, which grows as trees deepen. This priced the next chapter.

## Chapter 5: flat-first (the human order)

1. Inversion: locking starts UNARMED (locking_armed = false); it arms at the
   first level-stagnation event (existing stagnant_levels >= 5 signal — no
   new constants). The focus machinery is also dormant until armed, so the
   initial flat phase is truly flat.
2. Rationale: a human grasps the transformation whole first ("it is a
   fractal stamp") and decomposes only when the whole will not fit in one
   glance. Decomposition is the fallback, not the opening move. Flat search
   is the machine's "grasp it whole"; locking is its "break it into steps".
3. Effect: the opener race is deleted, not mitigated — by the time locks may
   commit, the beam is deep enough that substantive layers exist to
   out-cover cheap ones. Measured immediately: 009d5c81 solved in 1:12 (flat
   30s → armed → substantive ladder → solve) vs 29+ minutes the run before.

## Chapter 6: the root cause — scorer blindness to object chains

1. The question that broke it open: "we have the solution, search for it and
   the close ones in the logs." Found: `InputPalette ShapeColorMap` was
   generated at t=0.16s of every run — and scored -148, because nothing had
   been deposited and the scorer judges only the canvas. Its Union completion
   was worth 922 correct cells. 3M log mentions of ShapeColorMap: born
   correct, scored as garbage, pruned, every run.
2. So the beam hill-climbed in canvas space while the real work happens in
   object space. Whole solution families died one op before their deposit.
   Deposit-early families (paint-everything layers) ruled the beam because
   only they scored — that fed the cheap locks, which fed the ladders, which
   fed the exits. The entire pathology chain traces to this.
3. Also explains why locking looked so necessary: its de-facto job had
   become "preserve partial progress the scorer keeps murdering."

## Chapter 7: the fix — explicit deposit twins

1. First attempt (implicit, reverted): the exec kernel painted the pending
   object list into transparent cells at end-of-program ("implicit final
   underlay"). It worked — 009d5c81 in 60s/32s — but bent semantics:
   a. Programs could legally end mid-air (a solution shipped whose final
      deposit was the preview: `... Difference InputPalette ShapeColorMap`).
   b. Phantom locks: `Target[0] Intersection` certified 865 cells that
      existed only in its preview — a lock is a mid-program freeze, and only
      canvas paint survives into continuations, so the certified paint was
      not in the frozen state at all. Patched with a preview-off toggle for
      layer_fit, then the whole mechanism was replaced.
2. Final form (explicit): the generator emits, alongside every candidate NOT
   already ending in a canvas op, its sibling with a real `Underlay`
   appended (add_candidate_if_new, solver_executor.hpp). One ordinary tree,
   ordinary semantics, competing in the same batch.
3. Why Underlay and not all five canvas ops: the twin's job is survival, not
   completion enumeration. Underlay is the one PURE reveal — fills
   transparent cells, never damages existing paint, so twin score is a
   faithful "canvas + what pending would add", never less than the bare
   chain. Union can overpaint good canvas (noisy signal); Intersection/
   Difference are different transformations, not previews; Clear reveals
   nothing. Once the family survives via the twin, the other deposits are
   one ordinary generation step away. Emitting all five would multiply
   candidate volume ~5x for completions the search reaches a level later.
4. Cost: up to 2x candidate volume on object-ended chains (batch capacity is
   enforced at drain time, so no overflow). Benchmark-set numbers unmeasured.
5. Results, both sentinels, pure flat, zero locking invoked:
   a. 009d5c81: 3.94s — canonical solution, explicitly deposited:
      `InputPalette ShapeColorMap RemoveColor[1] Underlay Target[0] Underlay`.
   b. 007bbfb7: 6.28s — THE canonical 5-op form:
      `Input RemoveColor[0] Fractal AddColor[0] Underlay`.
6. Reading: the twin is one op of exact lookahead at every deposit point.
   The beam hill-climbs in object space, the walk to the answer is almost
   greedy, and locking retreats to its true niche — puzzles genuinely too
   deep for one flat pass — which is exactly where flat-first now puts it.

## Supporting changes shipped in the same arc

1. `modifies_canvas(node_type)` in cuda_dsl.cuh — single source of truth for
   the canvas ops ("printer-ops"): Union, Underlay, Intersection, Difference,
   Clear. Kernel-verified by grepping ctx.output writes per exec function.
   Locks may root at any of them (paint-then-carve layers are lockable; a
   `Target[0] Intersection`-opened run solved in 32s during the implicit
   era). Used at lock entry, boundary sweep, and (while it existed) the
   reduction root-guard — the guard died with the implicit preview, since
   deleting a deposit now visibly loses paint and certification rejects it
   naturally.
2. GUI op palette: COMBINE tab derives from modifies_canvas (was hand-listed
   and over-included four list ops); MaskBy/MaskOut → REDUCE,
   UnionL/IntersectionL → SELECT.
3. Diagnostics kept in the logs: [LOCK-REDUCE] (reduction in/out per lock
   commit) and [LOCK-PREFIX] (exact frozen chain). Cheap, and they solved
   Chapter 2 — leave them in.

## Postscript (2026-07-13): 05a7bcf2 and depth-saturation arming

1. 05a7bcf2 — the 7-layer ladder puzzle, solved by this solver's locking era
   (stored solution is its own output) — exposed the arming signal's blind
   spot: flat search crawled 16 minutes with honest cells STILL improving
   (mismatches 170 → 58), so no stall-shaped signal could ever fire.
   Progressing is not converging.
2. Evidence hunt per suggested signal: score-stall never fires (crawl resets
   it); cell-stall never fires (cells kept improving — the suggested fix was
   disproven by its own evidence); gain-per-second collapsed ~50x but any
   cutoff on it is a magic number; the beam's trees were at the 32-op
   MAX_SEARCH_DEPTH cap — the one pure event.
3. Fix shipped: locking also arms on depth saturation (any beam tree at the
   cap). Flat's whole-glance phase is over by construction at the ceiling.
4. Scaffolding layers, measured: the ladder's first rung
   `InputPalette NormalizeOrientation[1] Underlay` was generated and twinned
   at t=0.2s and scored -1092 — honestly, because its value is instrumental
   (a substrate for later ReadUnder layers), not paint-correctness. No paint
   scorer can price it; only the lock ladder (purity: wrong paint allowed on
   undone colors) can build through it. This is why layering exists.

## Postscript (2026-07-14): the alias experiment — why the twin is NOT redundant

1. Attempt: replace the twin with score-aliasing — evaluate every bare chain
   WITH an Underlay appended (one program, no twin tree), keep the bare chain
   in the beam, materialize the Underlay only when a tree wins something.
   Halves candidate volume; looked like pure win.
2. It broke 009d5c81 (stuck at beam top 382/440 across three runs) via a
   chain of two bugs and one structural flaw:
   a. Bug: the winner's score was cached under the MATERIALIZED tree's hash
      while the beam holds the bare tree — bare-chain winners scored zero.
      Fixed (cache under the bare hash), didn't cure the stall.
   b. Bug: bare chain and explicit-deposit form now paint identical grids,
      so grid-dedup culled one. Fixed (dedup keyed by deposit state too),
      didn't cure the stall.
   c. Structural: beam seats are won per GPU batch — only each batch's top
      scorer gets its score recorded and can enter the beam. Under aliasing,
      bare X and explicit "X Underlay" are the SAME program with the SAME
      score; a tie never beats the incumbent and the bare form packs first,
      so the deposited tree stopped winning seats.
3. Why that kills convergence: the two forms are different IDENTITIES with
   different futures. Extending bare X keeps morphing the object, nothing
   committed; extending "X Underlay" builds a SECOND layer on committed
   paint. Multi-layer programs can only grow from the deposited identity —
   the solution "InputPalette ShapeColorMap RemoveColor[1] Underlay
   Target[0] Underlay" needs the mid-chain deposit as a growth point. Starve
   that identity of beam seats and multi-layer solutions are unreachable.
4. The twin's 2x cost is therefore two different things: duplicate COMPUTE
   (the GPU runs near-identical programs — genuine waste) and duplicate
   IDENTITY (two record-book entries — load-bearing). The alias deleted
   both; only the first was safe to delete.
5. Reverted to explicit twins (code-identical to the known-good build);
   009d5c81 back to solved, 5.2s.
6. The CORRECT optimization, if the 2x ever matters: evaluate the deposited
   program once and credit the score to BOTH identities — two cache entries,
   two candidates, one GPU run. Kills the duplicate compute, keeps the
   duplicate identity. Requires the winner-recording site to credit both
   hashes and the twin to skip evaluation (pack-skip via cache). Not built —
   attack the measured 48% GPU idle first, it's the bigger lever and has no
   structural risk.

## Open items

1. Fractal ignores an object's bbox offset (x0/y0): the block-stencil and
   stamp are bbox-local and anchored at the output origin, so cropped
   objects (e.g. InputPalette's per-color segmentation, whose bbox shrinks
   when a color misses rows/columns) stamp shifted-to-origin. Invisible for
   full-grid objects; wrong for offset ones. This was the coincidence that
   fed Chapter 1's bad lock (bbox = full grid on the two active examples,
   cropped on example 2). Fix direction: grid-space block mapping + reading
   the mask through y0/x0 — needs a frame size, which the object alone does
   not carry (ctx.input dims are right for input-derived objects).
2. Substitution reduction (notes/3 - Search/Substitution Reduction.md):
   deletion-only reduction has a floor — shortest subset, not shortest
   equivalent program. Layer folding is the table-free first move if ever
   needed.
3. Benchmark-set run: sentinels are the fastest they have ever been, but
   twins double candidate volume on object-ended chains and flat-first
   changes when locking engages — corpus-wide effect unmeasured. Run before
   trusting the arc beyond two puzzles.
4. The perception gap stands: the machine still proposes from syntax and
   learns from failure; a human proposes from evidence (shape↔color
   covariation seen before painting). The twin fix makes enumeration see
   honestly; inference — measuring input↔output covariations up front and
   generating/ranking layer candidates from them — is the planned next
   frontier (user: "the inference will be added later"). Related reading:
   notes/3 - Search/Salience Anchoring.md, notes/1 - Overall Design/
   Constraining Degrees Of Freedom By Invariants.md.
