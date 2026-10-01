# One Body, Two Compilers (2026-09-22)

Found while teaching inference to say VoteDenoise for 0607ce86.

## The fight this settles

Inference runs on the CPU. The ops run on the GPU. Whenever inference needs to
know what an op would do, there are two copies of the same arithmetic, and two
copies drift. The answer we kept reaching for was to move more onto the GPU, so
that there would be only one copy. That trades a drift problem for a much bigger
one: inference is branchy, vector-heavy and runs once per puzzle, which is
everything a kernel is bad at.

The other answer is to move neither side. Move the ARITHMETIC out of both, into
a header each compiler reads.

## The shape

    template <typename At, typename Put>
    ARC_HOST_DEVICE inline bool vote_denoise_cells(int w, int h, int8_t bg,
                                                   At ground_at, Put put);

- `ARC_HOST_DEVICE` (0_core.hpp) is `__host__ __device__` under nvcc and nothing
  under MSVC. `repeat_periods` already lived this way; this is the same move.
- The body knows no storage. It is handed a size, the facts it needs, a reader
  and a writer. The kernel passes lambdas over `gpu_fixed_object::mask`; the
  detector passes lambdas over `grid::data`. Reading and writing stay apart, so
  the vote never reads its own output.
- Lambdas written inside a `__device__` function are device lambdas, so nvcc
  builds a device instance of the template from the kernel's call and MSVC
  builds a host instance from inference's. One source, two compilations,
  nothing to keep in step. The template inlines, so the callbacks cost nothing.

What stays in each caller is only what genuinely differs: the kernel's pick
check and its fallback background, the detector's comparison with the output.

## What the body may use

Only what both compilers accept on both sides:

- fixed arrays sized by `MAX_GRID_DIM` / `COLOR_COUNT`, zero-initialised with
  `= {}` -- no `vector`, no `string`, no `memset`;
- `ARC_HOST_DEVICE` helpers such as `is_color`, and `constexpr` constants
  (nvcc runs with `--expt-relaxed-constexpr`);
- a stated contract instead of defensive branches. Here: `ground_at` returns a
  color, because both callers read a hole as the ground. Once that was said,
  the extra transparent bucket and every `is_color` guard in the vote were dead
  and came out.

## When it does not apply

- Warp-parallel kernels. A body striped across 32 lanes (`*_warp`, see CUDA
  Lessons) is written for the lanes, not for a loop; share the per-cell rule
  instead, not the loop around it.
- Learn-side against apply-side. When inference LEARNS a table and the op
  APPLIES it, the two are different computations that share a fact, not twins
  (Dedup Pass 2, item 4). Share the fact's definition (a hash, a key) and
  leave the two computations apart.

## How to know it held

The refactor moved the kernel's body verbatim, so the kernel's behaviour must
not change. It did not: the stored 0607ce86 program still fits 3/3 through the
refactored kernel, and arc_said over all 1000 training puzzles changed in one
line only -- 0607ce86 now solved by the reading on its own.

## Worth doing next

The twins in Dedup Pass 2 item 4 were judged "a project, not a cleanup". Where a
detector re-runs an op to check it (periodic_denoise, project_on_seeds'
stamping), this pattern makes it a cleanup: find the op's per-cell rule, lift it
into a header, hand it the two storages.
