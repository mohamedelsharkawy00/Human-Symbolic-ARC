// This file defines the data used to execute and score solver programs.
// Operators are combined into sentences, and those sentences together
// form a candidate program for the solver to try.
//
// The earlier chapters describe what happens in the training examples.
// To solve a puzzle, we need a program that can make those changes.
// Its operators can find objects, move them, change their colors,
// and draw the result.
//
// 8_vocabulary.hpp defines the operators; here we define the data sent to the GPU,
// and the results returned so the solver can judge each candidate program.
#pragma once
// Single home of the grid/color limits the GPU_* names below alias, and of
// output_dimension_rule, whose cases the kernel switches on.
#include "7_infer.hpp"
#include "8_vocabulary.hpp"
#include "11_text_format.hpp"  // to_text, for gpu_score_terms::say
#include <cstddef>
#include <cstdint>
#include <cstdio>   // fprintf / stderr in pinned_vector::resize failure path
#include <cstdlib>  // std::abort
#include <cstring>  // memset
#include <string>
#include <utility>
#include <vector>

// Thin wrappers over cudaMallocHost / cudaFreeHost. Defined in cuda_solver.cu so
// this header stays CUDA-free for non-CUDA translation units.
void* pinned_buffer_alloc(size_t bytes);
void  pinned_buffer_free(void* ptr);
// Said once, by main, after the last thing that could still want the GPU.
void  cuda_runtime_shutting_down();

// Drop-in replacement for std::vector<T> that uses pinned host memory so
// cudaMemcpyAsync can issue real DMA transfers instead of silently staging
// through an internal bounce buffer. Only the subset of vector API actually
// used by cuda_dsl_exec_batch is implemented.
template <typename T>
class pinned_vector {
    T*     m_data = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;

public:
    pinned_vector() = default;
    ~pinned_vector() { if (m_data) pinned_buffer_free(m_data); }

    pinned_vector(const pinned_vector&) = delete;
    pinned_vector& operator=(const pinned_vector&) = delete;

    pinned_vector(pinned_vector&& o) noexcept
        : m_data(o.m_data), m_size(o.m_size), m_capacity(o.m_capacity) {
        o.m_data = nullptr; o.m_size = 0; o.m_capacity = 0;
    }
    pinned_vector& operator=(pinned_vector&& o) noexcept {
        if (this != &o) {
            if (m_data) pinned_buffer_free(m_data);
            m_data = o.m_data; m_size = o.m_size; m_capacity = o.m_capacity;
            o.m_data = nullptr; o.m_size = 0; o.m_capacity = 0;
        }
        return *this;
    }

    void resize(size_t n) {
        if (n > m_capacity) {
            // Alloc BEFORE freeing the old buffer. The previous code freed
            // first, so a transient pinned-memory failure (cudaMallocHost
            // returns null — e.g. the job commit cap was hit) destroyed the
            // working buffer; worse, m_capacity was set even on failure, so
            // every later resize saw "big enough", skipped realloc, and
            // m_data stayed null forever -> 8-byte write to null in
            // pack_cuda_dsl_exec_batch (the diagnosed AV).
            T* fresh = static_cast<T*>(pinned_buffer_alloc(n * sizeof(T)));
            if (!fresh) {
                // Pinned host memory exhausted. This is fatal — the GPU
                // batch cannot run without it, and silently continuing
                // null-derefs later. Fail loud and immediate instead.
                fprintf(stderr,
                        "[FATAL] pinned_vector: cudaMallocHost failed for "
                        "%zu bytes (pinned memory / commit cap exhausted). "
                        "Old buffer preserved; aborting.\n",
                        n * sizeof(T));
                fflush(stderr);
                std::abort();
            }
            if (m_data) pinned_buffer_free(m_data);
            m_data = fresh;
            m_capacity = n;   // only on success
        }
        m_size = n;
    }

    T*       data()       { return m_data; }
    const T* data() const { return m_data; }
    size_t   size() const { return m_size; }

    T&       operator[](size_t i)       { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
};

// --- Domain constants: single home; 13_environment.hpp derives its names from these ---
static constexpr int GPU_MAX_GRID = arc::MAX_GRID_DIM;
static constexpr int GPU_GRID_CELLS = arc::MAX_GRID_CELLS;
// How long one program may be. The batch stores a program's length and its node
// count in an int8_t apiece (resize_exec_buffers, cuda_solver.cu), so 127 is as
// far as this may go before those two widen -- and it is the ONLY thing binding
// program length now that the object pool is a ring (gpu_acquire) rather than
// one slot per producing op. A written program is what spends this; the search
// stops at MAX_SEARCH_DEPTH, far below.
static constexpr int GPU_MAX_OPS = 127;
static constexpr int GPU_TRANSPARENT = arc::TRANSPARENT;
// A grid cut both ways leaves at most this many panes worth carrying.
static constexpr int GPU_MAX_PANES = 32;
static constexpr int GPU_MAX_COLORS = arc::COLOR_COUNT;

// The packed cell store lives upstream in 1_readings.hpp, where both sides can
// reach it: it holds no addresses, so the same bytes mean the same thing on the
// host and on the device. This name is what the DSL calls it, and there is only
// one — CPU and GPU objects hold their cells in exactly these bytes, so handing
// an object to the device is a copy, never a conversion.
using gpu_packed_mask = arc::object_cells;
// --- Scoring weights & penalties: single source of truth for all scoring ---
// Used by the kernel (dsl_exec_and_score_kernel_v2) and by the CPU-side
// debug/analysis breakdown in 13_environment.hpp.
static constexpr float GPU_SCORE_W_EXACT_MATCH = 1.0f;
static constexpr float GPU_SCORE_W_HISTOGRAM = 1.0f;
static constexpr float GPU_SCORE_W_DISTANCE_MAP = 0.1f;
static constexpr float GPU_SCORE_W_RARITY_MATCH = 0.5f;
static constexpr float GPU_SCORE_W_NODE_PENALTY = 1.0f;
// CC-match reward per component whose box and cell count match one of the
// target's, whatever its color. Calibrated so 1 matched CC clears the
// trivial-baseline trap (~-61) that the per-pixel terms create on
// gridline-heavy puzzles.
static constexpr float GPU_SCORE_W_SHAPE_MATCH = 60.0f;
// Continuous position reward per shape-matched CC: scales from full (centers
// coincide) down to 0 (centers max_dim apart). Smooth gradient toward the
// correct cell, no threshold cliff.
static constexpr float GPU_SCORE_W_POSITION_MATCH = 80.0f;

// `selected` lives before `mask` so gpu_warp_copy_header (which copies
// everything ahead of the mask) carries it.
// MSVC C4324: intentional padding; see object_cells in 1_readings.hpp.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
struct gpu_fixed_object {
    int16_t x0 = 0, y0 = 0;
    int16_t x1 = 0, y1 = 0;
    int16_t width = 0, height = 0;
    bool selected = false;
    gpu_packed_mask mask;

    ARC_HOST_DEVICE void clear() {
        x0 = y0 = x1 = y1 = width = height = 0;
        selected = false;
        mask.clear();
    }
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

struct gpu_flat_op {
    node_type type;
    int8_t value;
};

// =====================================================================
// GPU-side execution types (reduced limits for VRAM)
// =====================================================================

// The per-job gpu_exec_context is dominated by pool[POOL_SIZE] *
// gpu_object_list(MAX_OBJECTS * 912B): with a list as long as a grid has cells
// (MAX_OBJECTS_IN_A_LIST), about 0.8 MB a list.
//
// POOL_SIZE must cover the object-lists that are LIVE AT ONCE, which is two --
// the list an op reads and the list it writes -- not the number a program
// acquires over its whole length, because gpu_acquire wraps (cuda_solver.cu).
// It had to cover the whole program while that allocator clamped instead of
// wrapping: past POOL_SIZE producing ops every op wrote the slot it was reading
// and the scores came back quietly wrong. 8 is kept, four times the two and the
// floor 18_dsl_executor.hpp asks, so an op that ever held a third list is still
// safe; this header is GPU-only and must not pull in 13_environment.hpp, so the
// value stays a literal.
static constexpr int GPU_KERNEL_MAX_OBJECTS = arc::MAX_OBJECTS_IN_A_LIST;
static constexpr int GPU_KERNEL_POOL_SIZE   = 8;

// `selected` is the SELECT slot at object depth: one flag per object, not per
// cell. Selection works like Excel: nothing is selected at birth, selectors
// ADD whole objects, InvertSelection flips membership, SelectAll marks all.
// Acts touch ONLY selected objects — no selection means acts do nothing, never
// "everything". Sub-object reach comes from carving the piece into its own
// object first (Split/Colors), then selecting it — not from marking cells.
// The flag is always valid: host objects default it false, device pool objects
// clear it at the gpu_acquire chokepoint.

struct gpu_object_list {
    gpu_fixed_object objects[GPU_KERNEL_MAX_OBJECTS];
    int count = 0;
};

// The box the paint actually fills, which is not always the box the object
// declares -- Colors hands back layers carrying the whole source bbox.
struct gpu_painted_box { int x0, y0, width, height; };

inline ARC_HOST_DEVICE gpu_painted_box gpu_box_of_paint(const gpu_fixed_object& obj) {
    arc::cell_box box;
    for (int row = 0; row < obj.height; ++row)
        for (int col = 0; col < obj.width; ++col)
            if (obj.mask[row][col] != GPU_TRANSPARENT) box.include_cell(row, col);
    return {box.col0, box.row0, box.width(), box.height()};
}

// The glyph's key, read off the paint: the learned table's key of the `kind`
// shape_key_of names -- its shape, the same whichever way round the glyph
// lies, or its symmetry. Inference learns its keys from tight color boxes, so
// applying must key a tight box too or a padded object keys a different glyph
// and every lookup misses. Untrimmed input is therefore accepted rather than
// required to arrive trimmed.
inline ARC_HOST_DEVICE uint32_t gpu_shape_key_of_paint(const gpu_fixed_object& obj, int kind) {
    const gpu_painted_box box = gpu_box_of_paint(obj);
    if (box.width <= 0 || box.height <= 0) return 0;
    return arc::shape_key_of(kind, box.width, box.height, [&](int row, int col) {
        return (int8_t)obj.mask[box.y0 + row][box.x0 + col];
    });
}

// Cells packed exactly like an object's, so a grid and an object hold the same
// bytes and moving between them is a copy. Lanes painting the canvas in
// parallel must go through gpu_paint_cell, never the plain cell proxy: two
// side-by-side cells share a byte.
struct gpu_fixed_grid {
    // Named `data` because that is what a grid is expected to answer to: the
    // shared object walk in 1_readings.hpp and gpu_object_view both index a
    // grid as data[row][col], and a packed store indexes exactly that way.
    gpu_packed_mask data;
    int16_t width = 0, height = 0;

    ARC_HOST_DEVICE void clear_cells() { data.clear(); }
    ARC_HOST_DEVICE void clear() { data.clear(); width = height = 0; }
};

// --- What the DSL thinks the pieces of a grid are, read back to the CPU ---
// The ops are the single home of "what is an object", so nothing on the CPU
// segments a grid itself: it asks for the pieces and gets these. Sent back as
// an OWNER MAP rather than a stack of masks — a gpu_object_list carries a full
// 30x30 mask per object and would be ~800 KB, where one byte per cell plus a
// box per piece is under 1.5 KB. Ops whose pieces overlap cannot be read this
// way; Split and Colors cut a grid into disjoint pieces, so they can.
static constexpr int DSL_MAX_PIECES = 64;

struct dsl_pieces {
    struct piece {
        int8_t x0 = 0, y0 = 0, w = 0, h = 0;
        // -1 when the piece carries several colors (Split[0] does)
        int8_t color = -1;
        int16_t cells = 0;
    };
    piece pieces[DSL_MAX_PIECES];
    int count = 0;
    // Which piece owns each cell, -1 = none. Glyphs are rebuilt from this.
    int8_t owner[GPU_MAX_GRID][GPU_MAX_GRID];
};

// Run ONE op over each grid and hand back the pieces it made. `variant` is the
// op's annotation (Split's depth: 0 object with colors mixed, 1 object of one
// color, 2 single pixels); ignored by ops that take none. The op itself is
// untouched — this runs it through the same evaluator the solver uses.
#if !defined(ARC_ENABLE_CUDA_PIECES) || ARC_ENABLE_CUDA_PIECES
void cuda_dsl_pieces(const gpu_fixed_grid* grids, int grid_count,
                     node_type op, int8_t variant, int8_t background,
                     dsl_pieces* out);
#endif

// Per-job execution context living in device global memory.
// ~2MB per job with current limits.
struct gpu_exec_context {
    gpu_fixed_grid input;
    gpu_fixed_grid output;
    // The size rule, stashed once at kernel start so ops can recover the *target*
    // output dims at any time. RestoreOrientation needs the target orientation but
    // can't read it off `output` — a prior NormalizeOrientation may have transposed
    // `output` mid-program. The rule (input data, never derived/cached state) is
    // the single source of truth; gpu_rule_output_dims() recomputes from it.
    int8_t  rule_type = 0;
    int16_t rule_params[6] = {0, 0, 0, 0, 0, 0};
    // Did NormalizeOrientation flip this example? RestoreOrientation reads it
    // to undo exactly what was done (works on square grids, unlike target dims).
    int8_t orientation_flipped = 0;
    gpu_object_list seed_objs;
    // WHERE THIS EXAMPLE'S PANES LIE, as top/left/width/height. Geometry, not
    // cells: Panes clips whatever the chain holds to these boxes, so what counts
    // as figure stays the caller's words (Colors, SelectColor, DeleteSelection)
    // rather than something this op decides on everyone's behalf.
    int8_t pane_boxes[GPU_MAX_PANES][4];
    int8_t pane_count;
    // puzzle-level: train output colors ranked by total pixel count (desc); count below
    int8_t output_colors_by_size[GPU_MAX_COLORS];
    int8_t output_colors_by_size_count;
    // Learned shape->color lookup for the ShapeColorMap op (puzzle-level broadcast).
    arc::shape_table shape_colors;
    int8_t distance_maps[GPU_MAX_COLORS][GPU_MAX_GRID][GPU_MAX_GRID];

    gpu_object_list pool[GPU_KERNEL_POOL_SIZE];
    int pool_head;
    int current_objects_idx;  // warp-visible: which pool slot (or sentinel) holds current result
    // Sentinels for results outside the pool:
    static constexpr int CURRENT_OBJECTS_NONE    = -1;
    static constexpr int CURRENT_OBJECTS_SEEDS   = -2;

    // Per-cell label store for the CC-match scorer's warp-parallel connected-
    // component labeling (monotonic min-label propagation). Lives here in global
    // memory (already-allocated, accessed once per kernel) rather than the stack.
    // Replaces the old serial-BFS workspace (cc_visited + cc_qr + cc_qc, ~4.6KB);
    // the labeling no longer needs a visited mask or an explicit queue, so this is
    // ~1.8KB and the segmentation is now spread across all 32 lanes instead of
    // running on lane 0 alone.
    int16_t cc_labels[GPU_GRID_CELLS];

    // Lane-0 op scratch grid (global). Replaces per-thread local `int8_t nm[30][30]`
    // in slam/skew/etc. Per-thread LOCAL arrays are interleaved across 32 threads, so
    // a lane-0-only array reads ~1.3 of every 32 bytes/sector (NCU "L1TEX local load
    // access pattern"). A contiguous global buffer accessed by lane 0 is coalesced.
    // Safe to share across ops: the lane-0 op switch runs one op at a time.
    int8_t op_scratch[GPU_GRID_CELLS];
    // Lane-0 scratch for ExtendTo grown in step (grow_in_step): the piece that
    // took each cell, the turn it took it in, and the cells still to grow from.
    int16_t grow_scratch[3][GPU_GRID_CELLS];
};
// Ops overlay a packed mask on op_scratch, so it must have room for one.
static_assert(sizeof(gpu_exec_context::op_scratch) >= sizeof(gpu_packed_mask),
              "op_scratch must hold a packed mask");

// Per-example data uploaded once per puzzle (not per batch).
static constexpr int MAX_UPLOAD_EXAMPLES = 16;
// Max distinct target CC summaries we carry per example (matches host MAX_TARGET_OBJECTS).
static constexpr int GPU_MAX_TARGET_OBJECTS = 16;

// Compact CC summary: bbox and pixel count, the fields of object_summary. Its
// own type because the kernel keeps these in __shared__ memory, which cannot
// run object_summary's default values.
struct gpu_object_summary {
    int8_t w;
    int8_t h;
    int8_t x0;
    int8_t y0;
    int16_t count;
};

struct cuda_dsl_example_data {
    int count = 0;
    gpu_packed_mask input_grids[MAX_UPLOAD_EXAMPLES];
    int8_t input_dims[MAX_UPLOAD_EXAMPLES * 2];
    gpu_packed_mask target_grids[MAX_UPLOAD_EXAMPLES];
    int8_t target_dims[MAX_UPLOAD_EXAMPLES * 2];
    int8_t distance_maps[MAX_UPLOAD_EXAMPLES * GPU_MAX_COLORS * GPU_GRID_CELLS];
    gpu_object_list seed_lists[MAX_UPLOAD_EXAMPLES];
    int8_t pane_boxes[MAX_UPLOAD_EXAMPLES][GPU_MAX_PANES][4];
    int8_t pane_counts[MAX_UPLOAD_EXAMPLES];
    int8_t output_colors_by_size[GPU_MAX_COLORS] = {};
    int8_t output_colors_by_size_count = 0;
    // Learned shape->color lookup for the ShapeColorMap op (puzzle-level).
    arc::shape_table shape_colors;
    int8_t rule_type = 0;
    int16_t rule_params[6] = {};
    // Per-example target CC summaries, precomputed host-side.
    gpu_object_summary target_objs[MAX_UPLOAD_EXAMPLES * GPU_MAX_TARGET_OBJECTS] = {};
    int8_t target_obj_counts[MAX_UPLOAD_EXAMPLES] = {};
};

// What the score was made of, straight from the kernel that made it. A host
// mirror of the formula drifted the moment a term was added on one side only,
// and it did: focus, vandal and lookahead were GPU-only, so every [SCORE] line
// ever printed was a partial sum that could not equal the number the beam ranks
// by. One home: the kernel computes these, so the kernel reports them.
struct gpu_score_terms {
    float vandal = 0, exact = 0, histogram = 0, distance = 0;
    float rarity = 0, shape = 0, color = 0, position = 0;
    float node_penalty = 0, dimension_penalty = 0;
    float hand = 0;
    float total = 0;

    // ONE HOME for the readout, because two of them drifted the moment a term
    // was added to one side: the import line silently lost `hand` and a reader
    // had to recover it from total minus sum. Everything that prints a score
    // asks here, so the printed sum is always the whole formula.
    float parts_sum() const {
        return vandal + exact + histogram + distance + rarity + shape +
               color + position + hand + node_penalty + dimension_penalty;
    }
    std::string say(const std::string& tree, int example, const char* phase = nullptr) const {
        return arc::to_text(
            "[SCORE] %s%s%stree=\"%s\" ex=%d vandal=%.4f exact=%.4f histogram=%.4f "
            "distance=%.4f rarity=%.4f shape=%.4f color=%.4f position=%.4f "
            "hand=%.4f node_penalty=%.4f dim_penalty=%.4f sum=%.4f total=%.4f",
            phase ? "phase=" : "", phase ? phase : "", phase ? " " : "", tree.c_str(), example,
            vandal, exact, histogram, distance, rarity, shape, color, position,
            hand, node_penalty, dimension_penalty, parts_sum(), total);
    }
};

// Host-side batch for combined exec+score.
// One job = one (program, example) pair.
// Only contains per-batch data (programs + results). No per-example data.
struct cuda_dsl_exec_batch {
    int job_count = 0;
    int active_examples = 0;

    // Per-job program ops: [job_count * GPU_MAX_OPS]
    pinned_vector<gpu_flat_op> programs;
    pinned_vector<int8_t>      program_lengths;
    pinned_vector<int8_t>      node_counts;

    // Which candidate index each job belongs to
    pinned_vector<int16_t> program_indices;

    // Results (filled by kernel)
    pinned_vector<float>  scores;
    pinned_vector<int8_t> solved;
    pinned_vector<int8_t> output_grids;     // [job_count * GRID_CELLS]
    pinned_vector<int8_t> output_dims_result; // [job_count * 2]
    // What each job PAINTED, as one number. Two candidates that paint the same
    // canvas are the same answer however differently they were spelled, and the
    // grid itself is far too wide to bring back for every job.
    pinned_vector<uint64_t> output_hashes;  // [job_count]
    pinned_vector<gpu_score_terms> score_terms;  // [job_count]

    void resize(int count) {
        job_count = count;
        programs.resize(count * GPU_MAX_OPS);
        program_lengths.resize(count);
        node_counts.resize(count);
        program_indices.resize(count);
        scores.resize(count);
        solved.resize(count);
        output_grids.resize(count * GPU_GRID_CELLS);
        output_dims_result.resize(count * 2);
        output_hashes.resize(count);
        score_terms.resize(count);
    }
};

// Upload per-example data to GPU. Call once per puzzle (or after invalidation).
void cuda_dsl_upload_examples(const cuda_dsl_example_data& examples);

// Execute flat programs on GPU and score results. Copies back only scores+solved.
// NOTE: does NOT block for completion. Caller must call cuda_dsl_wait_for_exec(slot)
// before reading batch.scores / batch.solved.
void cuda_dsl_exec_and_score(cuda_dsl_exec_batch& batch, int slot);

// Block until the cuda_dsl_exec_and_score(slot) call has finished copying results.
// Call this immediately before reading batch.scores / batch.solved on the CPU.
void cuda_dsl_wait_for_exec(int slot);
// DIAG: returns true and fills `out` if a CUDA error was captured since last call.
bool cuda_dsl_take_error(char* out, int outsz);

// Fetch output grids for a specific winner from GPU. Call after picking winner.
void cuda_dsl_fetch_output_grids(cuda_dsl_exec_batch& batch, int winner_idx, int active_examples, int slot);
// The same copy, issued and left to travel: the caller reads it one harvest
// later, after the wait_for_exec on that slot has ordered it, and so never
// stands still for a picture. It lands in its OWN buffer rather than the
// batch's, because the batch is repacked in between and can reallocate under a
// DMA. cuda_dsl_winner_grids/dims hand back where it landed: `active` grids laid
// end to end, first example first.
void cuda_dsl_fetch_output_grids_async(int winner_idx, int active_examples, int slot);
const int8_t* cuda_dsl_winner_grids(int slot);
const int8_t* cuda_dsl_winner_dims(int slot);
// Reads and zeroes the fetch's own split: nanoseconds in the copies, and in
// the stream sync that follows them.
void cuda_dsl_take_fetch_ns(long long* copy_ns, long long* sync_ns);

// Call when puzzle changes to force re-upload of per-example data.
void cuda_dsl_invalidate_examples();

// Returns true if per-example data is already cached on device.
bool cuda_dsl_examples_uploaded();

// --- Scratch path: isolated device slot for 1-off bridge calls. ---
// These do not touch the main example cache used by the beam search,
// so firing them mid-beam does not invalidate the training upload.
void cuda_dsl_upload_scratch_example(const cuda_dsl_example_data& example);
void cuda_dsl_exec_and_score_scratch(cuda_dsl_exec_batch& batch);
void cuda_dsl_fetch_output_grids_scratch(cuda_dsl_exec_batch& batch, int winner_idx, int active_examples);

// Fetch the object list produced by the last op of a single scratch job
// (i.e. the list pointed to by ctx.current_objects_idx). Used by the GUI hover
// preview. Returns false if the job produced no object list (CURRENT_OBJECTS_NONE).
// out_canvas_w/h (optional) receive the job's working output-canvas dims after
// execution, which reflect any rotate ops (e.g. NormalizeOrientation makes the
// canvas portrait) so the preview can size the tile background to match.
bool cuda_dsl_fetch_last_object_list_scratch(int job_idx, gpu_object_list& out,
                                             int* out_canvas_w = nullptr,
                                             int* out_canvas_h = nullptr);
