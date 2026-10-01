# THE ORDER. READ THIS BEFORE TOUCHING ANYTHING.

You keep forgetting the order. The order is the whole point. Out of order you
waste days tuning a search whose grammar cannot reach the answer, or a score
whose DSL cannot say it.

## THE UNIT IS ONE PUZZLE

Walk the training set in order and finish each one before starting the next.
A puzzle is done when the solver finds it BY ITSELF and every puzzle before it
still does. 33 of 1000 as of 2026-08-29.

## THE ORDER, INSIDE ONE PUZZLE

1. **Read the puzzle by hand.** Say what happens in plain words, no ops yet.
   If you cannot say it, nothing downstream can.

2. **Write it with the ops you have.** If it needs more than 30 ops, the DSL
   is the bottleneck and nothing else is. Save it to solutions/<id>.txt.

3. **Import it.** If it does not score solved on every train pair, the ops do
   not mean what you think they mean. Fix that before anything else.

4. **Ask what inference read.** It decides what search is ALLOWED to build —
   dead ops, allowed colours, colour swaps, and the two backgrounds. A wrong
   reading shuts the door before search ever knocks on it, and the search
   looks broken when it is obedient.

5. **Ask what the grammar allows.** `legal_next` decides which op may follow
   which. A solution the grammar forbids is unreachable at every score.

6. **Only now, score and search.** With the answer expressible, permitted and
   reachable, ranking has something real to do.

## THE REGRESSION SET IS WHAT YOU ALREADY DID

There is no separate benchmark set to curate. The puzzles already finished ARE
the regression set, and it grows by one each time. A fix that solves puzzle N
and breaks any puzzle before it is not a fix.

## THE MENTAL MODEL

    Inference  = what the puzzle is ALLOWED to say.
    Grammar    = which chains can EXIST at all.
    DSL        = whether the answer is EXPRESSIBLE.
    Score      = whether an expressible answer is FINDABLE.
    Search     = how FAST a findable answer is found.

Each layer is only worth optimizing once the layer above it is good enough.

## THE MISTAKES YOU MAKE

- **Fixing the puzzle instead of the cause.** One puzzle at a time rewards a
  special case, and a special case is a regression waiting for puzzle N+1.
  Prefer what holds on the whole set over what holds on this one.
- **Tweaking the score before checking the layers above it.** Hours of weight
  calibration on an answer that inference forbade or the grammar cannot reach.
- **Adding terms instead of primitives.** Each score term carries cross-
  interaction risk and per-pixel cost. A primitive is cheap, local, and
  measurably reduces depth.
- **Skipping the hand solution.** "Just run it and see." It finds local
  maxima, and without a curated answer you cannot tell a good one from a bad.

## GROUND TRUTH

Curated solutions are the ground truth. Everything else is downstream.
