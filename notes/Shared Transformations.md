# Shared transformations before GPU inference

Prefer sharing transformation logic between inference and execution before deciding whether inference should run on the GPU.

These address different problems:

- Shared logic removes duplicated algorithms and keeps inference's predicted drawings consistent with execution.
- GPU inference may improve speed, but moving inference there does not automatically remove duplication. Its benefit needs profiling.

`7_infer.hpp` replays operations such as sliding pieces, chaining corners, and stamping templates. GPU execution implements those behaviors separately. This appears to be a substantial remaining DRY opportunity; its potential line reduction has not been measured.

Follow the existing `vote_denoise_cells` pattern: extract an operation's shared arithmetic into a host/device helper, placed only as far upstream as its first use needs. Inference verifies the resulting drawing; execution uses the same transformation. Keep caller-specific selection, validation, and storage handling with their callers.

Start with one duplicated operation, perhaps sliding pieces. Check that inference and execution preserve their behavior before extending the pattern. Avoid building a general framework ahead of actual uses.

Then profile inference's runtime and its contribution to total solving time. Inference currently uses variable-sized collections, branching, and early exits, so moving it wholesale to the GPU would require substantial redesign. If profiling identifies worthwhile parallel work, consider moving that part rather than assuming all inference belongs there.

The sequence is: share one operation, measure inference, then choose where it runs. Shared host/device logic remains useful whichever execution location wins.

## First extraction: SlamRow and SlamColumn

Implemented `slam_cells` in `7_infer.hpp`, immediately before its first inference use. It owns the row/column scan, movement direction, and collision with the edge or preceding cell. Both GPU operators call it through one `gpu_exec_slam` adapter. Inference uses the same helper to draw and record whether anything moved.

The callers still own selection, the GPU's mask-dimension clamp and scratch storage, and inference's rejection of overlapping pieces. The helper reads original cells and writes into separate cleared storage. Production code decreased by 51 lines.

Validation:

- `tests/core_transforms.cpp` checks movement, colors (including black), packed cells, both axes and directions, zero movement, empty/full shapes, and 30x30 boundaries. Its optional CUDA mode also verifies selected and unselected execution: 40 GPU cases passed against both the old and new libraries.
- Inference reports were byte-for-byte identical for all six local puzzle fixtures and the first 50 entries in the local training dataset.
- Existing core vocabulary tests and the `arc_headless`, `arc_spell`, and `arc_said` builds passed.

An initial CPU timing sample, after compilation finished, measured only `infer(train)` on those 56 puzzles. Three runs totaled 1.639, 1.645, and 1.642 seconds. Median per-puzzle time was approximately 11 ms; the slowest was approximately 274 ms. This is a small local sample, not a full-dataset or search benchmark, and does not establish a speed improvement from the refactor.

Next, measure inference's share of total solving time before deciding about GPU inference. The next extraction below shares `AlignToCanvas` displacement arithmetic without moving inference to the GPU.


## Second extraction: AlignToCanvas

`shift_to_touch(moving, target)` now owns the displacement calculation used by `slide_to_touch_rule` and GPU execution. It is an ordinary host/device function, with no function templates. Horizontal-axis ties and left/up distance ties are explicit in one place.

The existing `color_box` became `cell_box`: the same inclusive bounds now describe color extents, moving objects, and painted canvas content. It starts empty and owns expansion through `include_cell`, so callers no longer supply sentinel coordinates or repeat four boundary updates. Color-box construction, inference, and GPU canvas measurement use that operation.

The combined production diff for the slam and alignment extractions removes 58 net lines (7 more than the first extraction). The structural benefit is one alignment formula and one box-expansion operation, rather than a large line-count reduction.

Tests are consolidated in `tests/core_transforms.cpp`. All 55 GPU cases passed against the pre-alignment and post-alignment libraries: 40 slam cases and 15 alignment cases, including selection, empty canvas, overlapping boxes, and ties. CPU transformation and inference checks passed; all 56 inference reports remained byte-for-byte identical. The three solver consumer builds passed.

### When the templates earn their place

The slam helper uses two template parameters for its read and write callbacks. The movement loop is ordinary code; the compiler supplies the callback types. Both callers read the same cells, but inference writes directly onto its grid while GPU execution writes a mask. Those callbacks avoid an extra temporary object and painting pass in inference.

Keep that small adapter where it makes the data flow simpler. Prefer ordinary functions for shared arithmetic, as alignment demonstrates. Generality should follow actual callers, not require every shared operation to use a template.


## Checkpoint and next work

The completed scope is shared slam movement, shared alignment displacement, and reusable cell-box bounds, with the regression tests described above. Inference remains on the CPU. No further transformation changes were made after the alignment extraction.

The next candidate reviewed was `OverlapCorner`. Both implementations order pieces left-to-right, with top-to-bottom ties, then place each piece's top-left at the previous piece's bottom-right. However, inference uses `stable_sort` while GPU execution uses a selection sort. Equal-position pieces may therefore have different relative order, which can affect the color painted at an overlap. Establish and test the intended tie behavior before unifying the ordering; do not silently treat these implementations as equivalent.

`ProjectOnSeeds` is another candidate, but its duplicated logic has not yet been reviewed in this continuation.

Still unmeasured: inference's share of total solving time and the benefit of moving any part of inference to the GPU. The local timing sample above does not answer either question. Continue extracting only where shared logic makes the actual callers easier to understand.

## Later: one name for "takes an argument"

`is_parameterized(type)` in `20_environment.hpp` only returns `is_annotated(type)`: two names for one question, whether an op's argument range holds more than one value. `24_solver_executor.hpp` asks it as `is_parameterized` (2 calls); `20_environment.hpp` and `32_gui.hpp` ask it as `is_annotated`. Keep one name, move the callers over, and delete the other. The range itself now comes from `row_of` in `14_vocabulary.hpp`, so the survivor could live there beside `takes_value`.

## Later: other issues seen while cleaning files 0-12

Noted during the 2026-09-24 cleanup and not yet fixed.

Duplication:

- The puzzle-level fields (shape table, output colors by size, both backgrounds, size rule) are declared in three structs, `gpu_exec_context`, `cuda_dsl_example_data` and `gpu_example_template` in `cuda_solver.cu`, and copied field by field between them.
- `gpu_object_summary` repeats `object_summary`. It stays separate only because the kernel keeps it in `__shared__` arrays, which cannot run `object_summary`'s default values.
- `gpu_painted_box` repeats `cell_box`.
- `20_environment.hpp` still builds its own `type_name` and `primitive_types` tables from `NODE_PRIMITIVES`.
- `paint_piece` and the stamping loop in `project_on_seeds_rule` are the same "paint an object at an offset" loop; one fails at the edge, the other clips.
- `5_matching.hpp` walks a posed footprint onto the target grid in three separate loops: `landing_shows`, `landed_on_non_background`, and the chance-odds loop in `gather_matches`.

Names and magic values:

- `rule_params[6]` and `pane_boxes[..][4]` are raw arrays filled by position; a small struct would name each slot.
- `GPU_MAX_OBJECTS` sizes a host list, not anything on the GPU.
- `find_objects_by_color` returns same-color groups as summaries; it does not search by color.
- `repeat_periods` takes a magic `-2` for "every cell counts", returns through out-parameters, and has a bare `bool` flag.
- Bare `true`/`false` arguments hide their meaning at call sites, such as `find_in(..., false, y, x, true, ...)` and `mark_painted(..., false, true)`.

Dead or always-constant code:

- `gpu_score_terms::color` is always 0, because the kernel sets it so, yet it is still summed and printed in every `[SCORE]` line.
- The `!in_every` check in `score_candidates` and the `!is_color` check in `label_objects` never fire.
- `word_for_segmentation` has a `case` that returns what `default` already returns.
- `select_segmentation` only calls `correspondence_election::infer`.
- The comment above the score weights in `12_dsl.cuh` says `13_environment.hpp` uses them for a CPU breakdown; that breakdown no longer exists.

Style and structure:

- The op vocabulary (`node_type`, `op_role`, `NODE_PRIMITIVES`) sits in the global namespace, while everything else is in `arc`.
- Every header has `using namespace std;`.

Performance:

- `fall_back_to_greedy` draws both footprints inside its sort comparator, again on every comparison.

Process:

- The byte-for-byte check of all 1,360 puzzles' inference output lived only in a scratchpad. Nothing in the repo compares inference output across the dataset; it is worth adding under `scripts/` or `tests/`.
