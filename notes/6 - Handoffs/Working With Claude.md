WORKING WITH CLAUDE — speedup notes
====================================

Lessons from the 08ed6ac7 compression session (16 → 5 nodes). Took days; could
have been hours. Most time went into conversation, not code.

WHAT WAS SLOW
-------------
- Re-deriving the same idea from many angles (the "two lists meeting" problem
  came up 6+ times before landing on a fused unary op).
- Claude offering 3-4 alternatives when only 1 was viable — had to read all
  to reject all-but-one.
- Cold starts after multi-day gaps. No persistent context between sessions.
- Philosophy mixed with implementation. Both were valuable, but interleaved
  they slowed each other down.

WHAT TO DO NEXT TIME
--------------------

1. State hard constraints UP FRONT, not mid-conversation. The big ones discovered
   this session:
   - No binary ops. Ever.
   - No hidden / ambient state in ops.
   - Prefer DSL over preprocessing/archetype layer.
   - Prefer general/composable over fused/specialized (but accept fused unary
     when the alternative is binary).
   - Terminals can know about puzzle context; ops cannot reach into globals.
   Telling Claude these up front kills 80% of the option space immediately.

2. Write a one-paragraph "WHAT I AM BUILDING AND WHY" before opening any
   conversation about implementation. Not a plan, not a spec — just the goal
   in your own words. If you can't write it, you're not ready to build.

3. Separate philosophy sessions from implementation sessions. When you want to
   think about "how do humans solve ARC", make that its own conversation,
   produce a notes file, then start a fresh implementation conversation that
   references it.

4. Decide the constraint BEFORE exploring options. Don't ask "should this be
   per-example or puzzle-level?" after we've started designing it. Decide
   first, build second.

5. Keep a running WHAT_I_AM_DOING.md. One paragraph. Current goal + last
   decision + next step. Update as you go. When you come back 3 days later,
   read that file and you're in flow in 30 seconds.

6. Ask for the SMALLEST viable change explicitly. "Just add what this puzzle
   needs" → 30 minutes. "How should we generalise this?" → hours. Pre-narrow
   the scope.

OPENING-MESSAGE TEMPLATE
------------------------
Paste this at the start of an implementation session:

  GOAL: <one sentence — what success looks like>
  CONSTRAINTS: <hard rules I won't violate>
  NON-GOALS: <things I'm explicitly not doing now>
  HUNCH: <my current best guess at the approach>
  QUESTION FOR CLAUDE: <the actual decision I want help with>

Claude will match the frame instead of opening the option space. Should
roughly 2-3x throughput.

WHAT WASN'T WASTED
------------------
- The circling on "where do the two lists meet" surfaced the no-binary-ops
  constraint as durable architectural knowledge.
- The philosophy thread (perception vs search, archetypes vs DSL, lifting
  search into preprocessing) reframed the project. Worth keeping.
- notes/2_dsl/learned_orderings.txt means the next ordering-family op is a
  short job, not another multi-day discovery.

THE EXPENSIVE PART IS THE FRAME, NOT THE CODE
---------------------------------------------
First op of a family: days, because the architectural decisions are new.
Subsequent ops in the same family: ~1 hour, because the frame already exists.
When something feels slow, ask: "am I building, or am I figuring out the
frame?" If it's frame work, that's normal — but separate it from build work.
