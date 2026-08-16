# Comment Capitalization Sweep

Parked 2026-07-26.

## What

Every comment that starts a sentence begins with a capital letter.

## Done

`include/inference.hpp` — 71 lines.

## Left

`include/core_types.hpp`, `include/environment.hpp`, and the rest of `include/` and `src/`.

## The rule applied

Capitalize a comment line only where a sentence begins:

1. The line above is not a comment, or
2. the comment line above ended with `.`, `!`, or `?`.

Continuation lines of a multi-line comment stay lowercase.

## The exception

Leave the line alone when its first word is a name — a file, a field, a function.
Capitalizing it would misspell the thing it points at.

Left lowercase in inference.hpp for that reason:

`inference.hpp`, `color_diff`, `color_swap`, `object_pairing`, `across_inputs`,
`in_every`, `infer()`, `background_color`, `by_size`.
