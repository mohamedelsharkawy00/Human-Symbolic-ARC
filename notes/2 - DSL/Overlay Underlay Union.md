Overlay vs Underlay Union — proposed variant (2026-05-30)

Status: SHIPPED, but in a DIFFERENT SHAPE than proposed (updated 2026-07-01).
The capability landed as a STANDALONE `Underlay` op (environment.hpp:145,
gpu_exec_underlay_warp in src/cuda_solver.cu ~733), NOT as the `Union[1]`
annotation variant this note designs. The motivation/payoff analysis below
(10fcaaa3, layering order, bg-fill-last caveat) still applies; the
Union[0]/Union[1] compatibility discussion is moot.

Motivation
----------
Union paints src onto the canvas (ctx.output) and src wins every pixel:
    if (paint != TRANSPARENT) canvas[gr][gc] = paint;   // overlay
This is fine when the last layer painted should be on top. It hurts when an
earlier layer must stay on top of a later one.

Concrete case — 10fcaaa3 (tile 2x2 + diagonal-8 rays from each colored cell):
the rays and the colored cells collide after tiling. e.g. train ex2, a 6 at
(2,1) throws a down-right ray onto (3,2), which is itself a 6. The expected
output keeps both as 6, i.e. CELLS must win over RAYS. With overlay-only Union
the rays (painted last, because they are derived FROM the cells) overwrite the
cells, and an overwritten cell can't be recovered from the canvas — only by
re-sourcing from Input.

So the working program needs three layers and a re-stamp:
    Input Tile[0] Union                                  # bg(0) + cells
    RemoveColor[0] Split[1] ExtendDiagonal[1] Recolor[8] Union   # rays (cover cells)
    Input Tile[0] RemoveColor[0] Union                   # re-stamp cells on top
~13 ops, and the trailing re-stamp is load-bearing, not redundant.

Proposed op
-----------
Make Union annotated with 2 variants:
    Union  / Union[0] = overlay  (current: src wins)
    Union[1]          = underlay (src painted BEHIND: deposit only where the
                        canvas is still transparent, so existing pixels win)

Underlay body is the overlay body plus one guard:
    if (paint == TRANSPARENT) continue;
    if (underlay && canvas[gr][gc] != TRANSPARENT) continue;   // keep existing
    canvas[gr][gc] = paint;

Payoff
------
With underlay the layering inverts and the re-stamp disappears: paint cells
first, drop rays behind them, fill background last.
    Input Tile[0] RemoveColor[0] Union            # cells only, transparent bg
    Split[1] ExtendDiagonal[1] Recolor[8] Union[1] # rays underlay (don't cover cells)
    Target[0] Union                                # fill transparent bg with 0
~10 ops, no Input re-source. NOTE underlay requires the background to still be
transparent when the rays are painted — that's why bg-fill (Target[0]) is moved
to the very end. If bg(0) is laid down first, every cell is non-transparent and
underlay paints nowhere.

Compatibility
-------------
Bare "Union" in existing solution files parses to value 0 = overlay (the parser
defaults a missing [n] to 0), so behavior is preserved. Printing changes to
Union[0] for normal unions (round-trips fine). Intersection/Difference unaffected.

Related: Target[10] raw-canvas read (lets a branch re-read ctx.output mid-pipeline
with transparents preserved); Split[1] pixelize (one 1x1 object per cell).
