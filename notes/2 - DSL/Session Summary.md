SESSION SUMMARY - COLOR-0 BACKGROUND RESTORE

CURRENT STATE:

DSL is unary-only: every op reads ctx.last_pool_idx and writes a new pool slot.
There are no binary ops, and there is no way to hold onto two separately
computed intermediates and merge them later.

FillByOutput exists as a unary specialization of the checkpoint's binary
Fill(f1, f2). It tiles f1 clipped to ctx.output's non-transparent silhouette.
The silhouette source is training-only because it reads the target grid.

Transparency (-1) is a real sentinel, not a color. analyze_output_score compares
raw ints, so a mask cell left transparent where the target has color 0 counts as
a mismatch. On 007bbfb7 train, that gives 18, 36, or 54 mismatches after
importing the target solution.

The GUI shader used to clamp -1 to 0 silently as a safety clamp for out-of-range
palette indices. This is now fixed. Alpha blending is on, fragColor is vec4, and
transparent cells emit alpha=0. Bugs where the DSL leaves cells transparent are
now visually obvious.

[SCORE] formula logging now fires from apply_solution too on the import path.
It is tagged phase=import_train or phase=import_test, so imported-solution runs
can be filtered from normal beam output.

THE GAP:

The imported solution for 007bbfb7 is functionally close to right:

FillByOutput(Input(Union(ScaleUp[3](RemoveColor[0](Input())))))

The final output has transparent cells where the target wants color 0.
RemoveColor[0] strips the background early in the chain, nothing reintroduces
it, and FillByOutput only writes non-transparent cells from its source. Score
reads mismatches=18..54 per train example even though on an alpha-blended GUI
you would see "pattern on see-through" instead of "pattern on black."

No existing unary op refills transparent mask cells with a specific color.

Recolor[c]: sets obj.color metadata, but does not touch mask cells.
Filter[c]: keeps objects whose obj.color == c, but does not alter masks.
Union, Split, Colors, Foreground: rearrange or segment, but do not fill
transparent cells.
FillByOutput: only paints non-transparent source cells into the output
silhouette.

OPTION A - AddBackground[c]:

New unary op. It takes last and returns a copy where every transparent cell
inside each object's bounding box is replaced with color c. Non-transparent
cells are unchanged.

Semantics:

__device__ gpu_object_list& gpu_exec_add_background(
    gpu_exec_context& ctx, const gpu_object_list& src, int8_t c) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        gpu_fixed_object& out = res.objects[res.count++];
        out = src.objects[i];
        for (int r = 0; r < out.height; ++r)
            for (int c2 = 0; c2 < out.width; ++c2)
                if (out.mask[r][c2] == GPU_TRANSPARENT)
                    out.mask[r][c2] = c;
    }
    return res;
}

Usage for 007bbfb7:

AddBackground[0](FillByOutput(Input(Union(ScaleUp[3](RemoveColor[0](Input()))))))

Natural composition pattern:

RemoveColor[X] -> work on the pattern -> AddBackground[X] to restore.

Schema:

Arity 1, parameterized by color 0..9, for 10 variants. This matches the Recolor,
Filter, and RemoveColor shape in NODE_PRIMITIVES.

X(AddBackground, 1, "AddBackground", 7, Objects, Objects, None, 1, 10) in
environment.hpp.

Add AddBackground to the gpu_op_type enum in cuda_dsl.cuh, in the same position
as in NODE_PRIMITIVES. fixed_dsl.hpp casts node_type to gpu_op_type directly.

Add a case in the lane-0 fallback switch in gpu_eval_program_warp in
src/cuda_solver.cu, around line 1438. Grep for case gpu_op_type::RemoveColor if
the line has drifted.

Warp-parallel note:

The per-pixel loop over whole bounding boxes has no races and no inter-object
sync. It is a good candidate per LESSON 2 because real work is sum(fw * fh) per
call, which could be hundreds of cells across several objects. Start lane-0 to
ship; promote to warp-parallel with the Transform/Tile template if profiling says
it matters.

Effort: about 40 lines across 3 files. One commit.

OPTION B - REGISTER SLOTS PLUS SLOT-AWARE COMBINING OPS:

This is a more general capability. It lets the DSL save an intermediate result
to a numbered slot and combine it with last later. This unlocks more than
background restoration: extract background -> transform -> overlay back, or
build pattern in one branch plus silhouette in another -> combine.

Dispatch stays unary. Every op reads last or a slot and writes last. There are
no tree-shaped programs and no operand references in gpu_flat_op.

New ops:

Save[n]: copies last into slot n. last passes through unchanged.
Load[n]: replaces last with slot n.
OverlayWith[n]: paints last on top of slot n, with later-wins overlap behavior.
UnionWith[n]: opaque merge of last with slot n.
DifferenceWith[n]: subtracts slot n from last.

Start with Save / Load plus one combining op. Add more only if the beam uses
them.

Usage examples:

007bbfb7 equivalent to AddBackground[0](...):

Input -> Save[0] -> RemoveColor[0] -> ScaleUp[3] -> FillByOutput -> OverlayWith[0]

Extract background, recolor, put back:

Input -> Colors -> Filter[0] -> Recolor[5] -> Save[0] -> Input -> RemoveColor[0] -> OverlayWith[0]

What changes:

gpu_exec_context gains gpu_object_list slots[NUM_SLOTS]. Start with 2-4 slots.
Each is about 58KB in the current object-list layout, so check the VRAM budget
before picking.

gpu_flat_op::value, already int8_t, is reused as the slot index.

gpu_eval_program_warp: Save[n] copies pool[last_pool_idx] to slots[n] and passes
through. Load[n] and the With[n] ops read slots[n] as one operand and last as the
other.

Add enum entries, NODE_PRIMITIVES entries, and switch cases in the dispatcher.

Beam search needs no structural change because these are just more unary ops to
enumerate. The node penalty discourages gratuitous Save / Load pairs.

Costs and risks:

Slot count is a hyperparameter. Too few blocks useful programs; too many balloons
beam-search branching. Start with 2.

With[n] ops need the same warp-parallel treatment as existing mask writers. Per
LESSON 2, only warp-parallelize if per-call work is large.

The beam has to learn "save now, use later", which is a longer causal distance
than any current op. It may need a heuristic bump, such as always emitting Save[n]
as a free continuation on promising prefixes, if pure enumeration is too slow.

Effort: about 300-500 lines, one commit. It does not change the flat-op
serialization format beyond adding enum values.

RECOMMENDATION:

Ship Option A first. It closes the specific 007bbfb7 regression with about 40
lines and zero search-learning risk.

Option B is the right move if or when a second puzzle class shows up that
genuinely needs a held-over side branch, not just background restoration. At that
point the slot-machine generality pays off. Doing B without a second concrete use
case risks adding a search-space bomb the beam cannot currently navigate.

OTHER NOTES DROPPED FROM THE OLD SUMMARY:

Earlier speculation about "beam starves target prefix", "boost W_RARITY_MATCH",
or widening MAX_CANDIDATES_PER_SLOT to 1M was premised on "scoring is fine,
search cannot find it." That diagnosis was wrong. Scoring was counting
transparent-vs-0 as mismatches, and the solution tree itself does not produce 0s
in the first place. AddBackground[c] fixes the generation side; no beam, weight,
or candidate-budget tuning is needed for this specific puzzle.
