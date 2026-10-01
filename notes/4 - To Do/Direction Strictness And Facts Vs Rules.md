# Direction Strictness, And Facts vs Rules

Parked 2026-07-26.

## The two open questions

1. Must every channel cover all four reading directions?
2. When is a channel called facts, and when is it called a rule?

## What today settled

`output_dimension_rule` uses two of the four directions:

| direction | used by |
|---|---|
| across_input_output | Identity, Scale — the per-pair ratio |
| across_outputs | Fixed — the one size all outputs share |
| across_inputs | nothing |
| across_inputs_outputs | nothing |

That is not an oversight. Output size is a question about outputs and about
pairs. The input side matters only as the thing a ratio is applied to, and Scale
already carries the ratio.

So a channel covering all four is not the target. Coverage is a property of the
question, not a quota.

## What that costs

Nothing enforces coverage, and nothing can — C++20 cannot bind a struct's
members to an enum's entries without templates or macros, both rejected. The
author checks it. See §3a in `notes/1 - Overall Design/Inference Master.md`.

## Stale note to fix when this is picked up

§3a and §3b of the Inference Master use `dimension_facts` as the worked example.
That struct was deleted on 2026-07-26: every number in it was recomputable from
the train pairs, and only two readers existed. The pattern needs a live example
or the sections need rewriting.

> UPDATE 2026-07-26: the naming line was applied. `shape_color_map_facts` ->
> `shape_color_map_rule` and `color_swap_facts` -> `color_swap_rule`, since both
> pick one validated answer. `color_facts` stays facts. `background_election`
> is still the mixed case: it holds the ranked readings AND the election.

## The naming question

`output_dimension_rule` carries a chosen case plus the numbers that case learned,
and it names itself a rule. `color_facts` and the rest carry readings with no
choice made, and name themselves facts.

The dividing line to test: a rule has picked one answer, facts have not.
If that holds everywhere, the two words are already correct and the question
closes. If some channel picks an answer and still calls itself facts, it is
mis-named.
