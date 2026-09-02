#pragma once
// Single home of the grid/color limits the GPU_* names below alias, and of
// output_dimension_rule, whose cases the kernel switches on.
#include "inference/4_infer.hpp"

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdio>   // fprintf / stderr in pinned_vector::resize failure path
#include <cstdlib>  // std::abort
#include <cstring>  // memset
#include <utility>

// Thin wrappers over cudaMallocHost / cudaFreeHost. Defined in cuda_solver.cu so
// this header stays CUDA-free for non-CUDA translation units.
void* pinned_buffer_alloc(size_t bytes);
void  pinned_buffer_free(void* ptr);

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

// --- Domain constants: single home; environment.hpp derives its names from these ---
static constexpr int GPU_MAX_GRID = arc::MAX_GRID_DIM;
static constexpr int GPU_GRID_CELLS = GPU_MAX_GRID * GPU_MAX_GRID;
static constexpr int GPU_MAX_OPS = 64;
static constexpr int GPU_MAX_OBJECTS = 900;
static constexpr int GPU_TRANSPARENT = arc::TRANSPARENT;
static constexpr int GPU_MAX_COLORS = arc::COLOR_COUNT;
// --- Scoring weights & penalties: single source of truth for all scoring ---
// Used by the kernel (dsl_exec_and_score_kernel_v2) and by the CPU-side
// debug/analysis breakdown in environment.hpp.
static constexpr float GPU_SCORE_W_EXACT_MATCH = 1.0f;
static constexpr float GPU_SCORE_W_HISTOGRAM = 1.0f;
static constexpr float GPU_SCORE_W_DISTANCE_MAP = 0.1f;
static constexpr float GPU_SCORE_W_RARITY_MATCH = 0.5f;
static constexpr float GPU_SCORE_W_NODE_PENALTY = 1.0f;
// CC-match rewards: shape-only and shape+color, scored independently so a
// wrong-color-right-shape program gets partial credit without ever tying
// the true solution. Calibrated so 1 matched CC clears the trivial-baseline
// trap (~-61) that the per-pixel terms create on gridline-heavy puzzles.
static constexpr float GPU_SCORE_W_SHAPE_MATCH = 60.0f;
static constexpr float GPU_SCORE_W_COLOR_MATCH = 50.0f;
// Continuous position reward per shape-matched CC: scales from full (centers
// coincide) down to 0 (centers max_dim apart). Smooth gradient toward the
// correct cell, no threshold cliff.
static constexpr float GPU_SCORE_W_POSITION_MATCH = 80.0f;

// Balanced ternary basis for MoveRow/MoveColumn/Slam variants: covers [-30,30] in ≤4 chains.
// Single home of the values: environment.hpp (CPU array) and cuda_solver.cu
// (device array) both initialize from this, so they cannot drift.
#define GPU_MOVE_DELTAS_INIT { 1, -1, 3, -3, 9, -9, 27, -27 }

// --- Fixed-size object representation (CPU + GPU compatible) ---
// No heap, no pointers, trivially copyable. Mirrors environment.hpp object/object_list
// but with inline mask and fixed-size arrays.

struct fixed_object {
    int16_t x0 = 0, y0 = 0;      // bounding box top-left (world coords)
    int16_t x1 = 0, y1 = 0;      // bounding box bottom-right (world coords)
    int16_t width = 0, height = 0;
    int8_t  mask[GPU_MAX_GRID][GPU_MAX_GRID]; // pixel colors or GPU_TRANSPARENT(-1)

    void clear() {
        x0 = y0 = x1 = y1 = width = height = 0;
        memset(mask, GPU_TRANSPARENT, sizeof(mask));
    }
};

struct fixed_object_list {
    fixed_object objects[GPU_MAX_OBJECTS];
    int count = 0;

    void clear() { count = 0; }

    bool push(const fixed_object& obj) {
        if (count >= GPU_MAX_OBJECTS) return false;
        objects[count++] = obj;
        return true;
    }
};

// What an op does to the layer, in pipeline-authoring order (a single solution
// line reads left-to-right through these): bring a source in (GET) -> cut it
// down (REDUCE) -> pick objects (SELECT) -> move it (TRANSLATE) -> recolor or
// reason (MAP) -> grow it (GENERATE) -> merge into the canvas (COMBINE).
// The generator's four sentence roles fold out of this: the middle four are ACT.
// Order is frozen — the GUI indexes its tab rows by these ordinals.
enum class op_role { get, reduce, select, translate, map, generate, combine, count };

// --- Op vocabulary: the single home of the DSL op list ---
// CPU (search, GUI, serializer) and GPU (kernel dispatch) both derive from
// this one macro, so the two sides cannot drift apart.
// Every op is unary (one Objects input, one Objects output); nodes form a chain.
// ROLE is a column here, not a list elsewhere, so a new op cannot be added
// without saying what it does — the macro will not compile without it.
// THE ORDER IS THE SEARCH ORDER, simplest first: fewest variants, then fewest
// lines of code. Ties: turns before flips, rights before lefts, inputs before
// outputs, canvas as Union/Underlay/Intersection/Difference, then fewer letters.
#define NODE_PRIMITIVES /* Schema: X(ID, NAME, IS_ANNOTATED, VAR_COUNT, ROLE) */ \
  X(SelectAll, "SelectAll", 0, 1, select)                                        \
  X(InvertSelection, "InvertSelection", 0, 1, select)                            \
  X(TurnRight, "TurnRight", 0, 1, translate)                                     \
  X(TurnLeft, "TurnLeft", 0, 1, translate)                                       \
  X(FlipHorizontal, "FlipHorizontal", 0, 1, translate)                           \
  X(FlipVertical, "FlipVertical", 0, 1, translate)                               \
  X(Input, "Input", 0, 1, get)                                                   \
  X(Output, "Output", 0, 1, get)                                                   X(CurrentObjects, "CurrentObjects", 0, 1, get)                                 \
  X(Origin, "Origin", 0, 1, translate)                                           \
  X(Union, "Union", 0, 1, combine)                                               \
  /* DeleteSelection consumes the marks, it does not make them: an act. */       \
  X(DeleteSelection, "DeleteSelection", 0, 1, reduce)                            \
  /* The canvas-readers judge by what lies beneath, not by the object: acts. */  \
  X(ReadUnder, "ReadUnder", 0, 1, map)                                           \
  X(Outline, "Outline", 0, 1, reduce)                                            \
  X(SelectMostCommonColor, "SelectMostCommonColor", 0, 1, select)                \
  /* Colors extracts the palette as objects — a producing act, not a source. */  \
  X(Colors, "Colors", 0, 1, map)                                                 \
  X(Trim, "Trim", 0, 1, reduce)                                                  \
  X(OverlapCorner, "OverlapCorner", 0, 1, map)                                   \
  X(Underlay, "Underlay", 0, 1, combine)                                         \
  X(Fit, "Fit", 0, 1, reduce)                                                    \
  X(Intersection, "Intersection", 0, 1, combine)                                 \
  X(Fractal, "Fractal", 0, 1, generate)                                          \
  X(Difference, "Difference", 0, 1, combine)                                     \
  X(UnionList, "UnionList", 0, 1, reduce)                                        \
  X(AlignToCanvas, "AlignToCanvas", 0, 1, translate)                             \
  X(IntersectionList, "IntersectionList", 0, 1, reduce)                          \
  X(SizeRankColorMap, "SizeRankColorMap", 0, 1, map)                             \
  X(Pattern, "Pattern", 0, 1, generate)                                          \
  X(ProjectOnSeeds, "ProjectOnSeeds", 0, 1, map)                                 \
  X(Bridge, "Bridge", 0, 1, generate)                                            \
  X(Partition, "Partition", 0, 1, generate)                                      \
  X(RestoreOrientation, "RestoreOrientation", 0, 1, translate)                   \
  X(ShapeColorMap, "ShapeColorMap", 0, 1, map)                                   \
  X(Seeds, "Seeds", 0, 1, get)                                                   \
  X(VoteDenoise, "VoteDenoise", 0, 1, map)                                       \
  X(ShapeToInteriorColorMap, "ShapeToInteriorColorMap", 0, 1, map)               \
  X(SnapToAnchors, "SnapToAnchors", 0, 1, map)                                   \
  X(SortBySize, "SortBySize", 1, 2, map)                                         \
  X(DiagTile, "DiagTile", 1, 2, generate)                                        \
  X(NormalizeOrientation, "NormalizeOrientation", 1, 2, translate)               \
  X(Split, "Split", 1, 3, generate)                                              \
  X(Tile, "Tile", 1, 4, generate)                                                \
  X(SelectObjectsWithHoles, "SelectObjectsWithHoles", 1, 5, select)              \
  X(MoveRow, "MoveRow", 1, 8, translate)                                         \
  X(MoveColumn, "MoveColumn", 1, 8, translate)                                   \
  X(Shift, "Shift", 1, 8, translate)                                             \
  X(MoveToEdge, "MoveToEdge", 1, 8, translate)                                   \
  X(SlamRow, "SlamRow", 1, 8, translate)                                         \
  X(SlamColumn, "SlamColumn", 1, 8, translate)                                   \
  X(ExtendTo, "ExtendTo", 1, 8, generate)                                        \
  X(ExtendDiagonal, "ExtendDiagonal", 1, 9, generate)                            \
  X(ExtendOrthogonal, "ExtendOrthogonal", 1, 9, generate)                        \
  X(ScaleUp, "ScaleUp", 1, 9, translate)                                         \
  X(ScaleDown, "ScaleDown", 1, 9, translate)                                     \
  X(Layout, "Layout", 1, 9, translate)                                           \
  X(SelectColor, "SelectColor", 1, 10, select)                                   \
  X(SelectObjectsContainingColor, "SelectObjectsContainingColor", 1, 10, select) \
  X(SelectLargestObject, "SelectLargestObject", 1, 10, select)                   \
  X(SelectSmallestObject, "SelectSmallestObject", 1, 10, select)                 \
  X(AddColor, "AddColor", 1, 10, generate)                                       \
  X(MaskBy, "MaskBy", 1, 11, map)                                                \
  X(MaskOut, "MaskOut", 1, 11, map)                                              \
  X(InteriorColor, "InteriorColor", 1, 11, map)                                  \
  X(Recolor, "Recolor", 1, 12, map)                                              \
  X(TileX, "TileX", 1, 30, generate)                                             \
  X(TileY, "TileY", 1, 30, generate)                                             \
  X(BidiColorMap, "BidiColorMap", 1, 45, map)                                    \
  X(Skew, "Skew", 1, 47, translate)

enum class node_type : uint8_t {
#define X(id, n, is_ann, var, role) id,
    NODE_PRIMITIVES
#undef X
    type_count
};

// Every op's role, straight off the ROLE column. The one thing anyone reads to
// ask what an op does — nobody keeps a second list.
// A SWITCH, not a table: the kernel calls this (via is_get_op, in
// gpu_eval_program_warp), and a table would be host memory the device cannot
// reach. Relaxed-constexpr compiles that silently and it faults at runtime, so
// this must stay pure comparison with nothing to dereference.
inline constexpr op_role role_of(node_type t) {
    switch (t) {
#define X(id, n, is_ann, var, role) case node_type::id: return op_role::role;
    NODE_PRIMITIVES
#undef X
    default: return op_role::map;
    }
}

struct gpu_flat_op {
    node_type type;
    int8_t value;
};

// The canvas ops ("printer-ops"): the only five that write ctx.output —
// Union/Underlay paint, Intersection/Difference erase. Everything else works on
// the pending object list. Kernel-verified 2026-07-12 (grep for ctx.output
// writes per exec function). Locks may only root at one of these: a lock
// certifies canvas cells, and only canvas paint survives into the continuations
// that build on the frozen prefix.
inline constexpr bool modifies_canvas(node_type t) {
    return t == node_type::Union || t == node_type::Underlay ||
           t == node_type::Intersection || t == node_type::Difference;
}

// --- Generation roles (the sentence: GET SELECT ACT COMBINE) --------------
// The four roles the generator's legality rules speak in, read off the ROLE
// column. GET = pure source (chain start). SELECT = marks objects, drops
// nothing. COMBINE = lands the list on the canvas. ACT = the four middle roles
// folded together — everything else that does work.
inline constexpr bool is_get_op(node_type t)     { return role_of(t) == op_role::get; }
inline constexpr bool is_select_op(node_type t)  { return role_of(t) == op_role::select; }
inline constexpr bool is_combine_op(node_type t) { return role_of(t) == op_role::combine; }
inline constexpr bool is_act_op(node_type t) {
    return !is_get_op(t) && !is_select_op(t) && !is_combine_op(t);
}

// Ops that hand back objects OTHER than the ones they were given, so no mark
// can cross them. A mark is attached to an object's identity: a carve turns one
// object into several, and the several are new; DeleteSelection keeps exactly
// the ones that were NOT picked. Everything else passes its objects forward, so
// the marks ride along with them.
// The same property decides whether an op can be selection-gated at all -- every
// gated act is one-object-in, one-object-out. Kernel is the truth, this is the
// claim; the drift-proof version is a column plus a startup self-check.
inline constexpr bool hands_back_new_objects(node_type t) {
    return t == node_type::Split || t == node_type::Colors ||
           t == node_type::Partition || t == node_type::DeleteSelection;
}

// Ops that do NOTHING when nothing is picked, so offering one before a pick is
// pure waste: it spells a program the search already has, one op longer. The
// generator refuses them until something is marked (legal_next).
// Each entry is kernel-checked, never guessed — Recolor and Shift pass unpicked
// objects through untouched, DeleteSelection deletes nothing, Partition cuts
// nothing without walls. Grows as more acts are taught to read the marks.
// NOT here: UnionList / IntersectionList, which do real work either way (with
// no marks they merge everything into one).
inline constexpr bool needs_a_pick(node_type t) {
    return t == node_type::Recolor || t == node_type::Shift ||
           t == node_type::DeleteSelection || t == node_type::Partition ||
           t == node_type::TurnRight || t == node_type::TurnLeft ||
           t == node_type::FlipHorizontal || t == node_type::FlipVertical ||
           t == node_type::MoveRow || t == node_type::MoveColumn ||
           t == node_type::SlamRow || t == node_type::SlamColumn ||
           t == node_type::MoveToEdge || t == node_type::Origin ||
           t == node_type::AlignToCanvas || t == node_type::Skew ||
           t == node_type::ScaleUp || t == node_type::ScaleDown ||
           t == node_type::Trim || t == node_type::Fit ||
           t == node_type::Outline ||
           t == node_type::InteriorColor || t == node_type::BidiColorMap ||
           t == node_type::VoteDenoise || t == node_type::ShapeToInteriorColorMap ||
           t == node_type::SortBySize ||
           t == node_type::TileX || t == node_type::TileY ||
           t == node_type::Fractal || t == node_type::AddColor ||
           t == node_type::Pattern || t == node_type::ExtendOrthogonal ||
           t == node_type::ExtendDiagonal;
}

// =====================================================================
// GPU-side execution types (reduced limits for VRAM)
// =====================================================================

// Context-shrink experiment: the per-job gpu_exec_context is dominated by
// pool[POOL_SIZE] * gpu_object_list(MAX_OBJECTS * 912B). Halving objects keeps
// the context small for L2 reuse. MAX_OBJECTS=32 truncates scenes with >32
// objects (ops already cap).
//
// POOL_SIZE must cover the live object-lists a program acquires. gpu_acquire is
// called per producing op, and some ops grab 2 slots (a pass buffer + publish),
// so worst case ~2 * MAX_SEARCH_DEPTH. At 24 slots, programs past ~14 ops hit
// the clamp in gpu_acquire (cuda_solver.cu) and silently overwrite slot N-1 ->
// corrupted scores for deep candidates (which the search DOES generate, up to
// MAX_SEARCH_DEPTH). Sized to 2*MAX_SEARCH_DEPTH so no deep program clamps; this
// header is GPU-only and must not pull in environment.hpp, so the value is a
// literal — keep it >= 2*MAX_SEARCH_DEPTH (dsl_executor.hpp static_asserts it).
static constexpr int GPU_KERNEL_MAX_OBJECTS = 32;
static constexpr int GPU_KERNEL_POOL_SIZE   = 64;

// __host__ __device__ under nvcc, nothing under the host compiler (this header
// is also included by MSVC-compiled .hpp files, which don't know those keywords).
#ifdef __CUDACC__
  #define GPU_MASK_HD __host__ __device__
#else
  #define GPU_MASK_HD
#endif

// DIAG: the proxy clamps an out-of-bounds packed-mask cell access to cell 0 so it
// can't fault. If clamping makes the kernel illegal-access vanish, the bug is a
// mask OOB. (printf here blows the ptxas budget under launch_bounds, so no log.)
#define GPU_MASK_OOB(r, c, i) ((void)0)

// 4-bit packed 30x30 cell mask: 2 cells/byte (450B vs 900B). Value v in [-1,9]
// (GPU_TRANSPARENT..colors) stored as nibble (v+1) in [0,10]. Proxies expose
// mask[r][c] read/write so lane-0 ops and all reads are unchanged. WARNING:
// the proxy write is a byte read-modify-write, so it is ONLY safe single-
// threaded; warp ops that write cells in parallel must write whole bytes via
// .data[] directly (two cells/byte) to avoid the shared-byte race.
struct gpu_packed_mask {
    static constexpr int CELLS = GPU_MAX_GRID * GPU_MAX_GRID;
    uint8_t data[(CELLS + 1) / 2];

    struct cref {
        uint8_t* b; int sh;
        GPU_MASK_HD operator int8_t() const { return (int8_t)(((*b) >> sh) & 0xF) - 1; }
        GPU_MASK_HD void operator=(int8_t v) {
            uint8_t n = (uint8_t)((int)v + 1) & 0xF;
            *b = (uint8_t)(((*b) & ~(0xF << sh)) | (n << sh));
        }
    };
    struct ccref {
        const uint8_t* b; int sh;
        GPU_MASK_HD operator int8_t() const { return (int8_t)(((*b) >> sh) & 0xF) - 1; }
    };
    struct row  { uint8_t* base; int r;
        GPU_MASK_HD cref  operator[](int c) const { int i = r * GPU_MAX_GRID + c;
            if (i < 0 || i >= CELLS) { GPU_MASK_OOB(r, c, i); i = 0; }   // DIAG: clamp OOB to prevent the fault
            return cref { base + (i >> 1), (i & 1) * 4 }; } };
    struct crow { const uint8_t* base; int r;
        GPU_MASK_HD ccref operator[](int c) const { int i = r * GPU_MAX_GRID + c;
            if (i < 0 || i >= CELLS) { GPU_MASK_OOB(r, c, i); i = 0; }   // DIAG: clamp OOB to prevent the fault
            return ccref{ base + (i >> 1), (i & 1) * 4 }; } };

    GPU_MASK_HD row  operator[](int r)       { return row { data, r }; }
    GPU_MASK_HD crow operator[](int r) const { return crow{ data, r }; }
    GPU_MASK_HD void clear() { for (int i = 0; i < (int)sizeof(data); ++i) data[i] = 0; }  // all transparent
    // Pack value v into cell (r,c) — for use inside a single byte by one lane.
    GPU_MASK_HD static uint8_t enc(int8_t v) { return (uint8_t)((int)v + 1) & 0xF; }
};

// Pack an unpacked GPU_MAX_GRID^2 int8 grid into a packed mask, cell by cell.
GPU_MASK_HD inline void gpu_mask_from_grid(gpu_packed_mask& m, const int8_t g[][GPU_MAX_GRID]) {
    for (int r = 0; r < GPU_MAX_GRID; ++r)
        for (int c = 0; c < GPU_MAX_GRID; ++c)
            m[r][c] = g[r][c];
}

// `selected` is the SELECT slot at object depth: one flag per object, not per
// cell. Selection works like Excel: nothing is selected at birth, selectors
// ADD whole objects, InvertSelection flips membership, SelectAll marks all.
// Acts touch ONLY selected objects — no selection means acts do nothing, never
// "everything". Sub-object reach comes from carving the piece into its own
// object first (Split/Colors), then selecting it — not from marking cells.
// The flag is always valid: host objects default it false, device pool objects
// clear it at the gpu_acquire chokepoint.

// `selected` lives before `mask` so gpu_warp_copy_header (which copies
// everything ahead of the mask) carries it.
struct gpu_fixed_object {
    int16_t x0 = 0, y0 = 0;
    int16_t x1 = 0, y1 = 0;
    int16_t width = 0, height = 0;
    bool selected = false;
    gpu_packed_mask mask;
};

struct gpu_object_list {
    gpu_fixed_object objects[GPU_KERNEL_MAX_OBJECTS];
    int count = 0;
};

struct gpu_fixed_grid {
    int8_t data[GPU_MAX_GRID][GPU_MAX_GRID];
    int16_t width = 0, height = 0;
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
// op's annotation (Split's depth: 0 blob with colors mixed, 1 blob of one
// color, 2 single pixels); ignored by ops that take none. The op itself is
// untouched — this runs it through the same evaluator the solver uses.
void cuda_dsl_pieces(const gpu_fixed_grid* grids, int grid_count,
                     node_type op, int8_t variant, int8_t background,
                     dsl_pieces* out);

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
    int8_t output_colors_by_size[10];     // puzzle-level: train output colors ranked by total pixel count (desc); count below
    int8_t output_colors_by_size_count;
    // puzzle-level: dominant across-inputs color (inference background_color); -1 = unknown
    int8_t background_color = -1;
    // puzzle-level: the canvas the OUTPUT figures sit on; the kernel starts the
    // canvas in this color, so no program has to paint it back. -1 = none elected
    int8_t output_background_color = -1;
    // Learned shape->color lookup for the ShapeColorMap op (puzzle-level broadcast).
    uint32_t shape_color_keys[16];
    int8_t   shape_color_vals[16];
    int8_t   shape_color_count;
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
};

// Per-example data uploaded once per puzzle (not per batch).
static constexpr int MAX_UPLOAD_EXAMPLES = 16;
// Max distinct target CC summaries we carry per example (matches host MAX_TARGET_OBJECTS).
static constexpr int GPU_MAX_TARGET_OBJECTS = 16;

// Compact CC summary: color, bbox dims, pixel count. Matches host
// object_summary layout but kept in this header so device code can include it.
struct gpu_object_summary {
    int8_t color;
    int8_t w;
    int8_t h;
    int8_t x0;
    int8_t y0;
    int16_t count;
};

struct cuda_dsl_example_data {
    int count = 0;
    int8_t input_grids[MAX_UPLOAD_EXAMPLES * GPU_GRID_CELLS];
    int8_t input_dims[MAX_UPLOAD_EXAMPLES * 2];
    int8_t target_grids[MAX_UPLOAD_EXAMPLES * GPU_GRID_CELLS];
    int8_t target_dims[MAX_UPLOAD_EXAMPLES * 2];
    int8_t distance_maps[MAX_UPLOAD_EXAMPLES * GPU_MAX_COLORS * GPU_GRID_CELLS];
    gpu_object_list seed_lists[MAX_UPLOAD_EXAMPLES];
    int8_t output_colors_by_size[10] = {};
    int8_t output_colors_by_size_count = 0;
    // Learned shape->color lookup for the ShapeColorMap op (puzzle-level).
    uint32_t shape_color_keys[16] = {};
    int8_t   shape_color_vals[16] = {};
    int8_t   shape_color_count = 0;
    // puzzle-level: dominant across-inputs color (inference background_color); -1 = unknown
    int8_t background_color = -1;
    // puzzle-level: the canvas the OUTPUT figures sit on. -1 = none elected
    int8_t output_background_color = -1;
    int8_t rule_type = 0;
    int16_t rule_params[6] = {};
    // Per-example target CC summaries, precomputed host-side.
    gpu_object_summary target_objs[MAX_UPLOAD_EXAMPLES * GPU_MAX_TARGET_OBJECTS] = {};
    int8_t target_obj_counts[MAX_UPLOAD_EXAMPLES] = {};
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

// Layered solving: the one unfinished color the scorer should favor (-1 = none).
void cuda_dsl_set_focus_color(int color);
// Colors locked as complete (bitmask); wrong paint on them scores as vandalism.
void cuda_dsl_set_done_colors(int mask);

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
