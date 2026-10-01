# Structural Ground: Enclosure (the signal the eye uses that area cannot fake)

## The observation (2026-07-19, user and Claude independently)

On 42a15761 both of us read BLACK as the ground at only 27% area, against a red
that holds 72% and every standout. Same on 1d0a4b61 (black ~10% under a dense
blue lattice). The eye is not counting cells — it is reading structure.

## The principle

The ground is the color that ENCLOSES the figures and SHOWS THROUGH their
holes — the layer that stays put while things change on it. Area is only a
proxy that happens to agree on most puzzles; on figure-dense puzzles the proxy
inverts (the figure material out-areas its own ground) and the election
follows the proxy off a cliff.

The four signals, all area-independent:

1. STABILITY — the ground is the layer the transformation never touches
   (42a15761: tiles flip hollow/solid, the black mesh is identical in input
   and output). Uses the PAIRS, not one grid — likely the strongest signal.
2. ENCLOSURE — the ground is the majority color around each figure's outer
   boundary (the good idea inside the evicted embeds, minus the worst-grid
   measuring).
3. HOLES — the ground is what shows through openings INSIDE figures (black in
   42a15761's hollow tiles, black in 1d0a4b61's erased-pattern windows).
4. CONNECTEDNESS — the ground is one component spanning the grid while the
   figures shatter into many.

## Background is a LENS, not a hidden fact

Electing a background is choosing what the objects are. 42a15761 through each
lens:
- black as ground -> the objects are red tiles, program = "flip tiles between
  solid and hollow".
- red as ground -> the objects are the black things (the mesh becomes one big
  lattice figure, the hole-marks become movable dots), program = "move the
  black dots to different cells".

Both are legitimate programs for the same transformation — and the "wrong"
lens can even be the CHEAPER one (fewer dots than tiles). So an ambiguous
election is not fatal: it picks which program family the search explores
first. This is exactly what the ranked hypothesis list is for (see the step-2
plan on background_election in inference.hpp): the beam searches under
ranked[0], and the solver's own stall event — depth saturation without a
solve — switches it to ranked[1], i.e. to the other lens.

## Poster-child puzzles (truth file carries them as x|area-winner for now)

- 1d0a4b61  true ground 0, area elects 1 (shows through the erased windows)
- 42a15761  true ground 0, area elects 2 (shows through mesh + hollow tiles)
- 4e45f183  true ground 0, elects -1 (shows through BETWEEN the panels — the
  "separator mesh" is really the table the tiles are laid on; WRONG-marked)

## When to build

After the 1k-puzzle walk has collected enough minority-ground cases to fit
against — the truth file rows marked with this note are the test set.

## Nested ground (2f0c5170): figure/ground is RECURSIVE

The black patches on 2f0c5170's azure field are PANELS: objects at the grid
level, but inside each panel black is the LOCAL canvas with figures on it —
and the output is one panel's interior, which is why the output canvas is
black. Ground at one level is figure at the level above. The underneath
verdict must distinguish "window through to what is BENEATH this grid's
pattern" (1d0a4b61) from "viewport into ANOTHER world with its own ground"
(2f0c5170); today the patterned-coverer gate separates them, but the honest
model is recursive layers (the old Foreground-on-itself idea).

## Texture ground and the separator promotion (458e3a53, 2026-07-19)

Two extensions the color vocabulary cannot express yet:

- PERIODIC TEXTURE as ground: the canvas is the repeating pattern itself;
  figure = whatever BREAKS the repetition (the solid panels). This is the
  stability signal generalized from time (unchanged input->output) to space
  (unchanged period-to-period): ground = the predictable part.
- SEPARATOR vs GROUND is decided by QUIETNESS, not structure (two user
  verdicts, 2026-07-19): yellow mesh on 458e3a53 = separator, not background;
  black mesh on 4e45f183, same structural role (spans, connects, encloses,
  never moves) = BACKGROUND, the empty space showing through. So
  connectedness/enclosure/stability cannot distinguish the two roles — both
  score high on all three. The discriminator is the quietness prior: a loud
  lattice reads as drawn INK (separator), a maximally quiet lattice reads as
  exposed CANVAS (ground) — for quiet colors the two roles collapse into one.
  4e45f183 is the WRONG-marked test row for this.
