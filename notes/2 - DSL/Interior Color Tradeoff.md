InteriorColor — known trade-off (2026-04-27)

Current rule:
  1. wall = every non-transparent cell.
  2. outside = 4-conn flood from grid edges through non-wall (transparent) cells.
  3. paint cells where !wall && !outside with fill_value.

Trade-off:
A non-transparent cell that sits *inside* an enclosure is treated as wall by
step 1, so step 3 won't paint it. It keeps its original color instead of being
filled. Example — green wall, single red pixel inside, InteriorColor[4]:
    . . . . .          . . . . .
    . G G G .          . G G G .
    . G . G .   -->    . G 4 G .
    . G R G .          . G R G .   ← red stays red, not yellow
    . G G G .          . G G G .

We can't tell a "true wall pixel" apart from "inside noise" with the current
definition (both are non-transparent). To paint inside-noise too, we'd need a
smarter wall definition that excludes interior cells. Candidates:
  - most-common non-transparent color = wall
  - largest connected component = wall
  - color of cells adjacent to the outside flood = wall

Not relevant for 00d62c1b (no interior noise). Revisit if a puzzle needs it.
