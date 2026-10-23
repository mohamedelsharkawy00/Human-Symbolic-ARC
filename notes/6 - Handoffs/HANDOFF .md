# Handoff: object layering and the shared segmenter — 2026-09-08

For a model starting cold on this repo. Two commits landed: `9b8255c`, `71e2530`, both on `main`, tree clean.

The user checked both by hand — app runs, puzzles solve. **No benchmark was run at any point in this session.** Treat every claim about speed or solve rate below as unverified.

---

## Working here

- **Build:** `scripts\build.bat x64-Release NOLAUNCH` from PowerShell. **Without `NOLAUNCH` it launches the GUI app** on success, which you rarely want. There is no test suite.
- **`cmake` is not on PATH.** The script sets up the MSVC environment itself. Calling cmake directly fails on `winsock2.h`.
- **Every source file is CRLF.** Read and write *bytes* when scripting edits, or you will silently convert a file to LF and produce a diff of thousands of lines.
- **Kernel stats:** ninja skips unchanged files. To see ptxas numbers, touch `src/cuda_solver.cu` first, then build and grep for `stack frame` and `Used .* registers`.
- **Comment style.** Short, plain, lead with *why*. No ruler lines. Don't restate the code, don't narrate what a change removed, don't explain that something ordinary is ordinary. Section banners are a bare `// OBJECTS · the related groups of cells a grid breaks into` — no `─────` rules (other files still have them; they're not wanted).

---

## Where things are

### `include/0_core.hpp` — renamed this session from `0_core_types.hpp`

The ARC data format and nothing else: `arc_color` + `COLOR_COUNT`, `MAX_GRID_DIM`, `grid`, `example`. 28 lines. `example` is now just `grid input, output;`.

Git recorded the rename as delete + create — the file changed too much for similarity detection — so `git log --follow` may not trace through it.

### `include/1_readings.hpp` — where nearly everything landed

Ordered deliberately: vocabulary, then objects (the main event), then smaller readings, then poses and shape hashing.

| what | ~line |
|---|---|
| `neighbour_row_8` / `neighbour_col_8` — derived, host+device | 27 |
| `NEIGHBOUR_ROW_8` / `COL_8` tables, built *from* those functions | 36 |
| `packed_grid` — 4-bit cell store, moved up from the GPU header | 73 |
| `walk_out` — the one flood fill, host+device | 120 |
| `object_summary` — colour, box, tally; plain bytes, GPU-shippable | 163 |
| `object` — summary + `packed_grid cells`, box-local | 184 |
| `object_walk_scratch` — ~6.3 KB, caller-owned | 229 |
| `each_object` — **the** segmenter; templated on grid type, host+device | 240 |
| `find_objects` / `find_objects_by_color` | 295 / 315 |

`each_object(grid, background, any_color, scratch, take)` is the single home of "what counts as one object". `take` gets `(object_summary, const int16_t* cells, int n)`.

**The cells pointer is scratch the next group overwrites — a taker must copy what it keeps.** That contract is what lets the summary path allocate nothing, and it is not enforced by anything.

### `src/cuda_solver.cu`

`gpu_flood_fill` is **deleted**. `gpu_segment_objects` (~177) is a five-line wrapper calling `arc::each_object` with `gpu_take_object` (~156) — a functor, not a lambda, because nvcc won't carry a lambda into device code without `--expt-extended-lambda` and this build doesn't pass it. A struct needs no flag.

The functor writes the full-grid frame (`x0=y0=0`, box = whole grid), matching old behaviour exactly: segmentation never tightens, Trim is the only shrinker.

---

## The one behaviour change

`find_objects_by_color` used to treat **black (0) as background, always**. It now takes the elected background, and both callers pass `p.inf.background.output_elected()` — `environment.hpp` (`detect_target_objects`) and `dsl_executor.hpp` (~56).

This was a real bug. ARC puzzles can put black figures on a coloured ground, and the CC-match scoring term was blind to them. `environment.hpp:733` was already doing the right thing for the GPU upload, so the segmenter was the odd one out.

**If solve quality regresses and you're bisecting, look here first.** Everything else this session was structural.

---

## Measured cost of the kernel swap

Both kernels gained **+4,496 bytes of stack frame** — `dsl_pieces_kernel` 9,088 → 13,584, the exec/score path 9,744 → 14,240.

Registers (255, 77, 80) and shared memory (1,800 / 4,240 B) did **not** move, and 255 is the ceiling — so **no warps were lost, occupancy is unchanged.** The cost is local-memory footprint and traffic. Never measured in wall-clock.

---

## What's left, in the order I'd take it

1. **Two segmenters remain** in `include/fixed_dsl.hpp` — `fixed_segment_objects` (~101) and `fixed_segment_by_color` (~121), building seed objects at ~301. The last places that answer "what is an object" independently. Same treatment the kernel got, at lower risk: they run at upload time, not inside a kernel.

2. **Recover part of the 4,496 bytes** with a streaming callback — hand cells to `take` as the walk finds them instead of buffering into `object_walk_scratch::cells` (1,800 B). The kernel writes each cell straight into its mask anyway. Only worth it if a profile says local memory matters.

3. **`object_info` in `3_correspondence.hpp` (~163)** declares the same six fields as `object_summary` under different names (`left`/`top`/`width`/`height` vs `x0`/`y0`/`w`/`h`). Embedding `object_summary` means touching **~157** field references in the most intricate file here. Mechanical, large, unverifiable without a benchmark.

4. **Small and independent:**
   - `pairs_overlay` and `direction` still sit in `1_readings.hpp` but ask about a whole train set — which that file's own header says it doesn't do. Move to their callers.
   - `split_kind`'s comment (`3_correspondence.hpp` ~48) says its members complete "AN OBJECT IS …". The user's better framing: an object *is* a related group of cells, and the cuts vary what *related* means — so they complete "TWO CELLS ARE RELATED WHEN …".
   - The "912B per object" figure at `cuda_dsl.cuh:372` looks **stale** now the mask is packed (450 B). `GPU_KERNEL_MAX_OBJECTS = 32` was chosen from it, so the cap may be tighter than it needs to be.
   - `notes/` still references the old filename `0_core_types.hpp` in ~7 files.
   - `scripts/build.bat` prints `'vswhere.exe' is not recognized` and falls through to a hard-coded VS path. Harmless now, fragile if VS moves.

---

## Decisions worth not re-litigating

- **An object is a related group of cells.** That definition is stable; what varies per puzzle is what *related* means — same colour touching, any colour touching, same colour anywhere, one period of a repeat. Those are `split_kind`. The user is firm on this and it's a good framing.
- **`object_summary` is a separate struct on purpose**, not fields inside `object`. It has to go where cells cannot: the scorer keeps an array of them in every `puzzle` and ships them to the GPU. A type owning a vector or a mask can't go there cheaply.
- **`packed_grid` over a cell list** — chosen because it holds no addresses, so the same bytes mean the same thing on host and device. A cell list is smaller for sparse objects but can't cross to the GPU as-is.
- **Layering rule.** "Core holds only what core itself uses" was tried and is **wrong** — it has no fixed point and would eventually empty core entirely. The rule that stops: a definition belongs at the lowest layer where its *meaning* is established. The user calls moving something to a lower layer **"moving up / upstream"**.

---

## Mistakes I made — don't repeat them

- Rewrote every file LF and had to restore CRLF. Read and write bytes.
- Reported a CPU/GPU mask-convention bug that didn't exist. `gpu_flood_fill` sets `x0=y0=0`, so its "grid coordinates" *are* box-local — the box is just the whole grid. Both sides agreed all along. Verify before claiming.
- Predicted the kernel swap would cost occupancy. It didn't; registers were already at the ceiling. Measure GPU limits, don't reason about them.
- Wrote comments narrating what had just moved and what used to be there. The user cut them repeatedly. Comments serve the reader, not the diff.
- Put `// namespace arc` back on a closing brace in a 28-line file. It earns its place only in long ones.
- Ran builds without `NOLAUNCH` for most of the session, silently starting the app each time.
- Used substring anchors when scripting edits that matched a second, unintended site. Anchor on enough context to be unique, and assert the match count.
