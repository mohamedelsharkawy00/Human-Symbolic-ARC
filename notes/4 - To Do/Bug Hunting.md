Bug-hunting tasks. Goal: surface silent/elusive bugs (the kind that only show up under specific workloads, like the cuda_dsl_examples_uploaded false-positive that needed a mid-puzzle promotion to trigger). Pattern to fight: "lazy state + plausible-but-wrong fallback values + delayed downstream symptom."


1. Grep for lazy-init / cache-validity flags
   Find every: static bool *_uploaded, *_initialized, *_dirty, *_cached, *_valid, etc.
   For each one, ask: if the underlying data GREW after the flag was set, would callers still see "good"? If yes → same shape as the d_examples_uploaded bug, fix.
   Concrete fix: replace bool with int (uploaded_count, valid_size) and have callers compare against their current need.
   Suspects to inspect first: anything in cuda_solver.cu and any *_cache static in solver_executor.hpp / fixed_dsl.hpp.
   Effort: ~half a day. Finite list.


2. Poison-fill all device buffers at allocation
   Change every cudaMalloc(&dev_buf, n) for read-as-input GPU buffers (templates, contexts, programs, distance maps, output grids) to cudaMalloc + cudaMemset to a sentinel pattern (0xCD, or for color buffers, value 99 — outside the valid 0..9 range).
   Effect: any kernel that reads uninitialized device memory produces visibly wrong output (extreme negative scores, color-99 cells in GUI), instead of silently scoring against zeroes that sometimes look plausible.
   Permanent canary. One-time change.


3. Add invariant asserts at GPU/host boundary
   Before launching the main exec_and_score kernel: assert(d_uploaded_count >= active_example_count). (This requires task #1 — the bool needs to become a count first.)
   Before reading d_example_templates[ei] inside the kernel: assert ei < d_uploaded_count (or pass uploaded_count as a kernel parameter and check on entry).
   Cheap, catches the NEXT version of this bug at the moment it would fire — not three layers downstream in a screenshot.


4. Adversarial puzzle regression suite
   Pick ~10 puzzles known to need many promotions (mid-puzzle active growth) and ~10 known to solve at active=1.
   Run the full solver against them, diff outputs/timings against a known-good baseline.
   Anything that used-to-solve but now doesn't (or vice versa) is a regression candidate.
   Should be runnable as a single command, ideally as a CI/nightly step.


5. Reusable divergence-check pattern
   Anywhere two paths SHOULD agree, add an opt-in cross-check:
     - CPU re-eval vs GPU result (we had this; covered by main vs scratch in this codebase since CPU went away)
     - Main batch path vs scratch path on the same (program, example)
     - Cached score vs fresh score
   The d_examples_uploaded bug was found because main and scratch disagreed; the disagreement was the loud signal in an otherwise silent failure. Generalize.
   Implementation: a debug build flag that, on every winner candidate, re-evaluates via the alternate path and warns on disagreement. Off by default for performance, easy to enable when chasing a smell.


6. Confirm the promotion-solve path fires (fix of 2026-07-16, unexercised)
   The "export says Timeout though test looked solved" bug (0b17323b's old export: Train Timeout, Test N/A, elapsed 1.18s) was the KNOWN HOLE: a layer-lock promotion that fills the train set never marked the task solved, so the solver ground to timeout and the export label (which only means "solved flag was false") lied.
   Fix shipped at the single chokepoint: try_promote_examples parks the verified tree (solve_found_by_promotion in search_state, auto-cleared on puzzle switch) when a promotion fills the train set; the worker loop commits it as best, marks solved, and logs "!!!! SOLVE FOUND BY PROMOTION: marking task solved !!!!". Subsumed the worker loop's old narrower safety net.
   Verified so far: clean build, normal solve+export regression (05269061), 0692e18c fresh-solves ~1:39 and exports truthfully — but that run went through update_best_solution's own finale, so the new consumer block has never fired in a log. No known puzzle triggers the lock-promotion route on demand (05a7bcf2 not solvable yet; 0b17323b's 1.18s solve was a fluke).
   To close: after any long benchmark, grep logs/ for "SOLVE FOUND BY PROMOTION". Present once = path proven. Until then treat the consumer block in the worker loop (solver_executor.hpp, "A promotion filled the train set") as unexercised code.

WHAT NOT TO DO
   Don't rewrite everything defensively. Don't add input validation in every function. Bloat creates new bugs and hides old ones.
   Targeted poison + targeted asserts at the few boundaries actually distrusted = enough.
   Pick boundaries: GPU/host data transfers, persistent caches, lazy-init paths. Inside a single function, trust your own code.

7. Settings never save while minimized (found 2026-07-19)
   The once-a-second settings saver lives in the rendering tail of the main
   loop, which is skipped while the window is minimized — so toggles made
   through the command port persist only via the exit flush. If the app dies
   without a clean exit, minimized-session changes are lost.
   Fix idea: move the save check above the rendering tail (it is cheap and
   change-detecting), so it runs every loop iteration regardless of
   visibility.
   Related lockstep hazard: ui_action (enum) and ui_action_names (words) are
   two parallel lists kept in step by hand; a missing name shifted every later
   command off by one (toggle_bg, fixed). Any addition must touch both — or
   better, generate one from the other someday.

8. App stops answering after ~81 puzzle loads over the command port (found 2026-08-07)
   Seen while walking puzzles to collect logs/inference dumps as a regression
   oracle for the inference.hpp DRY pass. Both runs walked from first_puzzle,
   both wrote exactly 81 dumps, then stopped. The process stays alive with
   Responding = false; the port still ACCEPTS a connection but never writes the
   reply line, so the client times out on read rather than being refused.
   NOT the poll rate. Run 1 polled state in a tight loop (a fresh connection
   every few ms); run 2 slept 200 ms between polls. Same stopping point, so
   connection churn and ephemeral-port exhaustion are ruled out as the cause —
   both runs died at the same COUNT, not after the same elapsed time.
   Both runs were --minimized, which is where this most likely meets item 7
   above: the rendering tail is skipped while minimized, so anything the app
   only services from that tail stops being serviced. A HYPOTHESIS, not a
   verdict — nobody has run the same walk with the window visible, which is the
   one experiment that separates "minimized path" from "puzzle-load leak".
   To close: walk 200 puzzles un-minimized. If it sails past 81, the fault is in
   what minimizing skips and item 7 is the same root. If it still stops near 81,
   look for something accumulating per puzzle load — the load path allocates a
   fresh inference object and rewrites a dump file each time.
   Not urgent for headless/Kaggle (no GUI, no command port), only for the walk
   tooling an agent drives.


9. Sweep for counts packed into a signed byte
   FOUND AND FIXED IN inference.hpp 2026-08-07; the class was never looked for
   anywhere else. object_pairing_facts::edge stored `hidden` and `outside` as
   int8_t. Both count CELLS of a piece, and a piece can run to MAX_GRID_DIM^2 =
   900 — so anything past 127 wrapped. Live logs showed "-92 off grid" and
   "-83 hidden": a negative count of cells, printed for who knows how long,
   because nobody reads that column closely.
   The nasty half was not the display. `landing_shows` reads `hidden` to decide
   whether a piece actually paints a cell, so a value that wrapped to exactly
   0 (256, 512, 768) reads as "nothing covers me" and the piece claims cells it
   is behind — which changes the cover, which changes the split election. A
   wrapped 0 is indistinguishable from a real 0 in the logs, so whether that
   ever fired is not knowable retrospectively. Only that it could.
   TO DO: the same shape almost certainly lives elsewhere. Grep environment.hpp,
   solver_beam.hpp, solver_executor.hpp and the kernels for int8_t/char fields
   and ask of each: is this a COUNT or an INDEX, and what bounds it? Coordinates
   and color ids are fine (0..29, -1..9). Anything counting cells, pixels,
   matches or nodes is a candidate.
   Cheap detector, no rebuild needed: impossible values show up in the dumps.
   Sweep logs/inference/*.txt for negative counts (px, edges, hidden, off grid,
   left_over, carried_cells, created_cells), for the background table's
   embeds/dom/span/gaps above 1000, and for min_kept outside 0..1 (which would
   catch a nan). That sweep is what proved the `hidden` overflow was real rather
   than theoretical, on a puzzle outside the 20-puzzle sample.
   WATCH OUT when widening a field: puzzle_loader fills the GUI readout structs
   with POSITIONAL aggregate initialisers, so reordering members silently
   reassigns values. Widening caught it only because the types stopped matching;
   two same-width fields would have compiled and lied.
