VERIFYING INFERENCE CHANGES — the harness lies if you let it
=============================================================

Written 2026-08-07, after a session where two "20/20 byte-identical" results
turned out to be comparing stale files against themselves.

THE TRAP
--------
`logs/inference/<id>.txt` is NOT written every time a puzzle loads. The facts
are read only when someone wants them — see the comment at the gate in
`solver_executor.hpp`:

    if (m_puzzle_loaded && !m_puzzle.inferred &&
        (inference_view != INFERENCE_OFF || !is_paused))

So inference runs, and the log is written, only when

  - the inference pane is OPEN (`inference_view != 0`), OR
  - the solver is UNPAUSED.

Drive the app minimized + paused + pane off — the obvious way for an agent to
walk puzzles without stealing the screen — and NOTHING is recomputed. Every log
on disk stays whatever it was. A diff then compares a file against itself and
reports a clean pass.

`settings/settings.txt` carries `inference_view` across launches, so whether the
harness works at all depends on how the GUI was left days ago. That is why the
same script passed honestly in the morning and vacuously an hour later.

SECOND TRAP: THE JUMP LANDS BEFORE THE FACTS DO
-----------------------------------------------
`puzzle <id>` lands asynchronously. `state` reports the new id as soon as the
jump is applied — inference is still running behind it, and on a heavy puzzle
that is seconds, not milliseconds. Polling `state` until the id matches and then
copying the log grabs the PREVIOUS puzzle's file.

THIRD TRAP: THE FIRST PUZZLE AFTER OPENING THE PANE
---------------------------------------------------
`cycle_inference` does not just flip a flag — it satisfies the gate for the
puzzle already loaded, so the app starts inferring immediately. The first
`puzzle <id>` sent right after collides with that, and its log is not rewritten.
Happened three times out of three; every later puzzle in the same walk is fine.

HOW TO DO IT RIGHT
------------------
1. Open the pane once at the start: send `cycle_inference` (0 -> 1). Restore
   `inference_view 0` in settings afterwards if that is how you found it.
2. Per puzzle: record the log's mtime, send `puzzle <id>`, then wait for the
   MTIME TO CHANGE. Not for the id. Give it a generous deadline and report the
   ones that never move rather than copying them.
3. Never run the solver just to make inference happen. Opening the pane is the
   cheap half of that gate and does not start a search.
4. Walk the list TWICE. The second pass skips whatever it already captured, so
   it costs one puzzle's time and picks up the one the pane-open race ate. A
   walk that reports 19/20 is not a failure to investigate — it is this.
5. A FRESH OUTPUT DIRECTORY EVERY RUN. Because step 4's skip is what makes the
   second pass cheap, pointing a walk at a directory that already has captures
   makes it skip EVERYTHING and report a confident 20/20 having done nothing.
   Reusing a name from earlier in the same session is the easy way to do this:
   the diff then compares the new build against hours-old files and "fails" for
   reasons that have nothing to do with the change. Name directories after the
   change plus a timestamp, never after their role ("final", "after").

PROVE THE CAPTURE IS REAL, DON'T ASSUME IT
------------------------------------------
A comparison that cannot fail is worthless. Before trusting a byte-identical
result, check that the captured files actually came from the build you think:

  - Add or change ONE line of the log in the build under test (a new field, a
    renamed heading) and confirm every captured file carries it. That marker
    separates "the change altered nothing" from "nothing was regenerated".
  - In this session `claimed_by` played that role: 20/20 of the good capture had
    it, 0/20 of the pre-change capture did, and the two stale captures inherited
    it from files still sitting on disk.

Also filter BOTH sides of a diff identically. Piping one side through grep and
comparing it to a raw file makes every line differ (CRLF), which looks like a
catastrophic regression and wastes a round of panic.

THE PRINCIPLE
-------------
A green result from a harness that was never wired up is worse than a red one:
red gets investigated, green gets believed. Any verification loop needs a way to
FAIL LOUDLY when it is not actually running. Build the marker first, then trust
the diff.
