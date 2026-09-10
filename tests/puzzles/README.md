Custom ARC puzzles
=================

Enable CUS beside TRN / EVL / TST in the app. Disable the other datasets to
browse only these fixtures. Selection is saved between launches.

Add one ARC JSON file per puzzle in this directory, with `train` and `test`
arrays and inline `input` / `output` grids. The filename becomes its ID.
Files load alphabetically, ahead of competition puzzles when both are enabled.
Toggle CUS off and on to reload files (enable another dataset first if needed).

- `connectivity_stress_30`: corner gaps admit an 8-way outside flood. Red teeth
  amplify interior blue contour votes until they outnumber exterior black votes.
- `connectivity_control_30`: sealed corners; 4-way and 8-way floods agree.

Both are 30x30; their output recolors blue interiors yellow. Regenerate and
verify their standalone flood checks with `python tests/connectivity_stress.py`.
