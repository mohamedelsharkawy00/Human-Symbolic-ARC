# Copying to objects

`CopyTo[anchor]` makes one copy of every selected object at every unselected
object. The selected objects supply the drawings; the unselected objects supply
destination frames. The result contains only the copies, all selected, in source
order then destination order. It preserves colors, transparent cells and size.

The argument names matching anchors of the source and destination frames:

| | Left | Center | Right |
| --- | --- | --- | --- |
| Top | 0 | 1 | 2 |
| Middle | 3 | 4 | 5 |
| Bottom | 6 | 7 | 8 |

An even span uses its earlier middle cell. `Trim` decides whether an object's
frame is its tight bounding box. `Recolor`, `Union` and `Underlay` remain separate
choices. CopyTo neither reads the input again nor paints the canvas. No selection
is a no-op; no destinations produces no copies. A product exceeding the object
list capacity produces no copies rather than silently losing destinations.

For `363442ee`, `Panes` removes the divider from the working layer, `Split[0]`
finds the figures, and `Trim` tightens their frames. Select the largest figure
and copy it to the centers of the others. Paint the input first to retain the
original figure and divider. The program is in `solutions/363442ee.txt`.

The corresponding inference rule checks this relationship across every training
pair, first on connected figures and then inside an already-elected pane split.
It currently recognizes a unique largest source figure. The primitive accepts
arbitrary selections, including multiple source figures. CPU and GPU share the
anchor arithmetic; the rule spells ordinary DSL operations.

Build `arc_copy_to_tests` explicitly and run it from the repository root for the
GPU contract and inference-transfer checks. `arc_said 363442ee` checks the learned
program against the puzzle's training and held-out outputs.
