#include "12_dsl.cuh"
#include "20_cuda_solver.cuh"
#include <chrono>
#include <atomic>
#include <cstring>
#include <cstdio>

void* pinned_buffer_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

// The pinned buffers live in thread_local batches, so their destructors run as
// threads unwind at process exit -- after the CUDA runtime has already torn
// itself down, where every free fails with "initialization error". The pages go
// back to the operating system regardless, so the last thing main does is say
// the runtime is gone and the frees stop asking.
static std::atomic<bool> cuda_runtime_alive{true};

void cuda_runtime_shutting_down() { cuda_runtime_alive.store(false); }

void pinned_buffer_free(void* ptr) {
    if (ptr && cuda_runtime_alive.load()) cudaFreeHost(ptr);
}

gpu_memory read_gpu_memory() {
    gpu_memory status;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total_bytes == 0)
        return status;
    status.free_mb = free_bytes / (1024ull * 1024ull);
    status.total_mb = total_bytes / (1024ull * 1024ull);
    status.used_mb = status.total_mb - status.free_mb;
    status.used_percent = (int)((status.used_mb * 100ull) / status.total_mb);
    return status;
}

bool cuda_init() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return false;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);

    // Identify the physical GPU this process bound to. With CUDA_VISIBLE_DEVICES
    // each process sees its assigned card as device 0, but pciBusID is the unique
    // physical identifier -> distinct bus IDs across shards prove 4 separate GPUs.
    fprintf(stderr, "[GPU] %s  bus %04x:%02x:%02x  visible_count=%d\n",
            prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, device_count);

    FILE* log = fopen("logs/cuda.txt", "w");
    if (log) {
        fprintf(log, "[CUDA] %s - %d SMs, %.0f MHz, %.1f GB VRAM\n",
                prop.name, prop.multiProcessorCount,
                clockKHz / 1000.0,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        fclose(log);
    }
    return true;
}

// =====================================================================
// GPU DSL Execution + Scoring Kernel
// =====================================================================

// Scoring weights live in 12_dsl.cuh (GPU_SCORE_W_*, single home).

// --- Device helpers ---

#define GPU_DEV_FOR_EACH_PIXEL(obj, r_var, c_var, v_var) \
    for (int r_var = 0; r_var < (obj).height; ++r_var) \
        for (int c_var = 0; c_var < (obj).width; ++c_var) \
            if (int8_t v_var = (obj).mask[r_var][c_var]; v_var != GPU_TRANSPARENT)

__device__ int8_t gpu_get_pixel(const gpu_fixed_object& obj, int r, int c) {
    return obj.mask[r][c];
}

// Write one cell of a packed store from a lane that may be racing another lane
// on the SAME BYTE. Two side-by-side cells share a byte, so the plain proxy
// write is a read-modify-write that can lose the neighbour's nibble. The device
// has no byte-wide atomic, so this compare-and-swaps the 32-bit word the byte
// sits in; arc::cells is aligned and padded to keep that word inside the
// store. Only for lane-striped painting -- serial code should use the proxy,
// which costs nothing.
//
// This loop is what packing the canvas costs: one puzzle went 8.6 -> 9.0,
// about 4.7% slower, measured 2026-09-09. Taken deliberately, for one cell
// type across host and device instead of two and a conversion between them.
// If it is ever wanted back, it is contained: only the union paint, the
// overlay paint, the intersection clear and the canvas transpose call this.
__device__ inline void gpu_paint_cell(arc::cells& m, int r, int c, int8_t value) {
    const int index = r * GPU_MAX_GRID + c;
    if (index < 0 || index >= arc::MAX_GRID_CELLS) return;
    const int shift = arc::cells::shift_in_word(index);
    unsigned int* word = reinterpret_cast<unsigned int*>(m.bytes) + index / arc::cells::CELLS_PER_BYTE / arc::cells::WORD_BYTES;
    const unsigned int keep = ~((unsigned int)arc::cells::CELL_MASK << shift);
    const unsigned int put = (unsigned int)arc::cells::enc(value) << shift;
    unsigned int old = *word, assumed;
    do {
        assumed = old;
        old = atomicCAS(word, assumed, (assumed & keep) | put);
    } while (assumed != old);
}

__device__ int gpu_count_pixels(const gpu_fixed_object& obj) {
    int n = 0;
    GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) { (void)r; (void)c; (void)v; ++n; }
    return n;
}

__device__ gpu_object_list& gpu_acquire(gpu_exec_context& ctx) {
    // THE POOL IS A RING, NOT A RUN. Two lists are live at a time -- the one an
    // op reads and the one it writes -- so a slot handed out a whole lap ago is
    // long dead and taking it back is free. Clamping handed out the LAST slot
    // for ever once a program passed POOL_SIZE producing ops, and that slot is
    // the source the very next op reads: it overwrote what it was about to read,
    // with no crash and no word, so a long program came back quietly wrong.
    // How long a program may be is GPU_MAX_OPS' question, never this allocator's.
    if (ctx.pool_head >= GPU_KERNEL_POOL_SIZE) ctx.pool_head = 0;
    auto& slot = ctx.pool[ctx.pool_head++];
    slot.count = 0;
    // Fresh lists have NOTHING selected (Excel model), and the flag is the
    // only truth — acts touch selected objects and nothing else, ever. In-struct
    // initializers never run in reused pool memory, so the invariant lives
    // here, the single chokepoint every list flows through.
    for (int i = 0; i < GPU_KERNEL_MAX_OBJECTS; ++i) slot.objects[i].selected = false;
    return slot;
}

// Resolve current_objects_idx to actual object list pointer (readable by any lane)
__device__ gpu_object_list* gpu_current_objects(gpu_exec_context& ctx) {
    int idx = ctx.current_objects_idx;
    if (idx == gpu_exec_context::CURRENT_OBJECTS_SEEDS)         return &ctx.seed_objs;
    if (idx >= 0 && idx < GPU_KERNEL_POOL_SIZE)           return &ctx.pool[idx];
    return nullptr;
}

// Lane-0 helper: acquire a slot and publish its index
__device__ gpu_object_list& gpu_acquire_and_publish(gpu_exec_context& ctx) {
    auto& slot = gpu_acquire(ctx);
    ctx.current_objects_idx = ctx.pool_head - 1;
    return slot;
}

// Stripe-copy only the USED objects of a list (count * sizeof(object)), never
// the empty room behind them, which a list as long as a grid has cells mostly
// is. Unused dst slots stay stale but are never read (ops are bounded by
// count). `lanes` lanes cooperate, all 32 of a warp at init; lane 0 writes the
// count.
__device__ inline void gpu_copy_used_objs(const gpu_object_list& src,
                                          gpu_object_list& dst, int lane, int lanes = 32) {
    int n = src.count;
    if (n < 0) n = 0;
    if (n > GPU_KERNEL_MAX_OBJECTS) n = GPU_KERNEL_MAX_OBJECTS;
    const int8_t* s = reinterpret_cast<const int8_t*>(src.objects);
    int8_t* d = reinterpret_cast<int8_t*>(dst.objects);
    int used = n * (int)sizeof(gpu_fixed_object);
    for (int i = lane; i < used; i += lanes) d[i] = s[i];
    if (lane == 0) dst.count = n;
}

// A LIST AN OP HANDS ON AS IT CAME, copied by the one lane that runs the op.
__device__ inline void gpu_copy_list(gpu_object_list& to, const gpu_object_list& from) {
    gpu_copy_used_objs(from, to, 0, 1);
}

// --- Paint objects to grid ---
__device__ void gpu_paint_objects(const gpu_object_list& src, int w, int h, gpu_fixed_grid& out) {
    out.width = (int16_t)w;
    out.height = (int16_t)h;
    out.clear_cells();
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int gr = obj.y0 + r, gc = obj.x0 + c;
            if (gr >= 0 && gr < GPU_MAX_GRID && gc >= 0 && gc < GPU_MAX_GRID)
                out.data[gr][gc] = gpu_get_pixel(obj, r, c);
        }
    }
}

// Segmentation is not the kernel's own any more: arc::each_object in
// 1_readings.hpp is the single home of "what counts as one object", and both
// sides call it, so inference and execution cannot come to different answers.
// What stays here is only the shape the answer is written into.
//
// A functor rather than a lambda: nvcc will not take a lambda into device code
// without --expt-extended-lambda, and a struct needs no build flag.
struct gpu_take_object {
    gpu_object_list* out;
    const gpu_fixed_grid* g;
    // The frame every piece keeps: the frame of what it was cut from.
    int x0, y0, width, height;

    __device__ void operator()(const arc::object_summary&, const int16_t* cells, int n) const {
        if (out->count >= GPU_KERNEL_MAX_OBJECTS) return;
        gpu_fixed_object& obj = out->objects[out->count++];
        // Keep the frame it was cut from; segmentation never tightens (Trim is
        // the only shrinker). The object's cells sit where they lie in that
        // frame, rest transparent.
        obj.x0 = (int16_t)x0; obj.y0 = (int16_t)y0;
        obj.x1 = (int16_t)(x0 + width - 1); obj.y1 = (int16_t)(y0 + height - 1);
        obj.width = (int16_t)width; obj.height = (int16_t)height;
        obj.mask.clear();
        for (int i = 0; i < n; ++i) {
            const int r = arc::cell_row(cells[i]), c = arc::cell_col(cells[i]);
            obj.mask[r - y0][c - x0] = g->data[r][c];
        }
    }
};

// The pieces of `g`, each in the frame of what they were cut from (x0, y0,
// width, height, which holds every painted cell of g).
__device__ void gpu_segment_objects(const gpu_fixed_grid& g, gpu_object_list& out, int8_t bg_color,
                                    arc::grouping how, int x0, int y0, int width, int height) {
    out.count = 0;
    if (g.width <= 0 || g.height <= 0 || width <= 0 || height <= 0) return;
    arc::object_walk_scratch scratch;
    arc::each_object(g, (int)bg_color, how, scratch, gpu_take_object{ &out, &g, x0, y0, width, height });
}

__device__ int8_t gpu_majority_vote(const arc::cells& mask, int r, int c, int factor, int oh, int ow) {
    return arc::majority_vote(mask, r, c, factor, oh, ow);
}

// =====================================================================
// DSL Ops (device)
// =====================================================================

// Scan an object's mask for the first non-transparent pixel color.
// Returns GPU_TRANSPARENT if the mask is empty.
__device__ inline int8_t gpu_obj_first_color(const gpu_fixed_object& o) {
    for (int r = 0; r < o.height; ++r)
        for (int c = 0; c < o.width; ++c)
            if (o.mask[r][c] != GPU_TRANSPARENT) return o.mask[r][c];
    return GPU_TRANSPARENT;
}

// Shared color-source rule for painting ops: resolve an annotation value to a
// per-object color. 0..9 = that literal; 10 = the object's rarest color;
// 11 = its most common. Frequency ties break toward the lower color number.
// Returns GPU_TRANSPARENT if the mask is empty.
__device__ inline int8_t gpu_obj_resolve_color(const gpu_fixed_object& o, int8_t value) {
    return arc::object_paint_color(o.height, o.width, value,
        [&](int row, int col) { return (int8_t)o.mask[row][col]; });
}

// One folding step of FNV-1a, so the canvas and the hand are stirred into the
// same number the same way.
__device__ inline uint64_t gpu_fold(uint64_t hash, uint64_t value) {
    return (hash ^ value) * 1099511628211ULL;
}

// A source has no frame to inherit, so it takes the one its own content makes:
// the box around every cell that is not transparent. This is not the tightening
// the frame law reserves for Trim — there is no earlier box being discarded,
// only a box being chosen for the first time. A grid with nothing on it makes no
// object at all, because nothing painted is nothing to hand on.
template <typename grid_type>
__device__ inline void gpu_frame_to_content(gpu_object_list& res, const grid_type& g) {
    int min_x = g.width, min_y = g.height, max_x = -1, max_y = -1;
    for (int row = 0; row < g.height; ++row)
        for (int col = 0; col < g.width; ++col) {
            if (g.data[row][col] == GPU_TRANSPARENT) continue;
            if (col < min_x) min_x = col;
            if (col > max_x) max_x = col;
            if (row < min_y) min_y = row;
            if (row > max_y) max_y = row;
        }
    if (max_x < 0) { res.count = 0; return; }
    gpu_fixed_object& obj = res.objects[0];
    obj.x0 = (int16_t)min_x; obj.y0 = (int16_t)min_y;
    obj.x1 = (int16_t)max_x; obj.y1 = (int16_t)max_y;
    obj.width  = (int16_t)(max_x - min_x + 1);
    obj.height = (int16_t)(max_y - min_y + 1);
    obj.mask.clear();
    for (int row = min_y; row <= max_y; ++row)
        for (int col = min_x; col <= max_x; ++col)
            obj.mask[row - min_y][col - min_x] = g.data[row][col];
    res.count = 1;
}

__device__ gpu_object_list& gpu_exec_input(gpu_exec_context& ctx) {
    auto& res = gpu_acquire(ctx);
    gpu_frame_to_content(res, ctx.input);
    return res;
}

// Color-argument sentinel meaning "any non-transparent color" (value 10, one past
// the 0..9 real colors). Used by MaskBy[10]/MaskOut[10] to match on the canvas
// being painted at all, regardless of which color.
static constexpr int8_t GPU_COLOR_ANY = GPU_MAX_COLORS;

__device__ gpu_object_list& gpu_exec_output(gpu_exec_context& ctx) {
    // Target: output-shaped snapshot of the live canvas (ctx.output, accumulated
    // via Union/Intersection/Difference), read back AS-IS with transparents kept,
    // so downstream ops see the actual painted result mid-pipeline. This is a pure
    // source now — the old Target[c] color-fill is written explicitly as
    // `Target AddColor[c]`, so Target carries no annotation.
    auto& res = gpu_acquire(ctx);
    // Output is a snapshot of the WHOLE canvas, including its empty frame.
    // Trim is the only word that reduces this frame to its painted bounds.
    if (ctx.output.width <= 0 || ctx.output.height <= 0) return res;
    auto& obj = res.objects[0];
    obj.x0 = obj.y0 = 0;
    obj.width = ctx.output.width; obj.height = ctx.output.height;
    obj.x1 = obj.width - 1; obj.y1 = obj.height - 1;
    obj.mask = ctx.output.data;
    res.count = 1;
    return res;
}

__device__ gpu_object_list& gpu_exec_seeds(gpu_exec_context& ctx) { return ctx.seed_objs; }

// Split[depth]: segmentation depth, coarse to fine. [0] = connected object,
// colors mixed. [1] = connected object of one color. [2] = single pixels.
// [3], finer than [1] and coarser than [2], came last: one color's cells
// joined by sides alone, a corner joining nothing.
__device__ gpu_object_list& gpu_exec_split(gpu_exec_context& ctx, const gpu_object_list& src, int8_t mode) {
    auto& res = gpu_acquire(ctx);
    gpu_fixed_grid temp;
    // Size the perception canvas to the full 30x30 ceiling, not the input dims:
    // the stream may have grown past the input bbox (Tile/Shift/Extend...), and
    // input-sizing would silently crop those pixels before segmentation.
    gpu_paint_objects(src, GPU_MAX_GRID, GPU_MAX_GRID, temp);
    if (mode == 2) {
        // Split[2] = pixelize: one 1x1 object per non-transparent cell (no
        // connectivity merge). A list holds one object per cell, so all fit.
        for (int r = 0; r < GPU_MAX_GRID && res.count < GPU_KERNEL_MAX_OBJECTS; ++r)
            for (int c = 0; c < GPU_MAX_GRID && res.count < GPU_KERNEL_MAX_OBJECTS; ++c) {
                int8_t v = temp.data[r][c];
                if (v == GPU_TRANSPARENT) continue;
                gpu_fixed_object& obj = res.objects[res.count++];
                obj.x0 = obj.x1 = (int16_t)c;
                obj.y0 = obj.y1 = (int16_t)r;
                obj.width = obj.height = 1;
                obj.mask.clear();
                obj.mask[0][0] = v;
            }
        return res;
    }
    // EACH PIECE KEEPS THE FRAME OF WHAT IT WAS CUT FROM: the frame the objects
    // it was handed span, as UnionList's one object spans it -- the whole input
    // where the input was cut -- the way Colors' layers keep theirs. So Tile
    // after it lays each piece's copies the input's size apart, a copy apiece
    // (10fcaaa3: cut, then tiled, the copies never join).
    int x0 = GPU_MAX_GRID, y0 = GPU_MAX_GRID, x1 = -1, y1 = -1;
    for (int i = 0; i < src.count; ++i) {
        const auto& o = src.objects[i];
        if (o.width <= 0 || o.height <= 0) continue;
        x0 = min(x0, (int)o.x0); y0 = min(y0, (int)o.y0);
        x1 = max(x1, (int)o.x1); y1 = max(y1, (int)o.y1);
    }
    x0 = max(x0, 0); y0 = max(y0, 0);
    x1 = min(x1, GPU_MAX_GRID - 1); y1 = min(y1, GPU_MAX_GRID - 1);
    gpu_segment_objects(temp, res, GPU_TRANSPARENT,
                        mode == 0 ? arc::grouping::touching_any_color
                        : mode == 3 ? arc::grouping::touching_sides_same_color
                                    : arc::grouping::touching_same_color,
                        x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    return res;
}

// Colors splits each source object into ALL its color layers, unselected.
// Every layer keeps the source frame. Removal and trimming are separate words.
__device__ gpu_object_list& gpu_exec_colors(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count; ++i) {
        const auto& sobj = src.objects[i];
        int w = sobj.width, h = sobj.height;
        for (int color = 0; color < GPU_MAX_COLORS; ++color) {
            bool found = false;
            for (int r = 0; r < h && !found; ++r)
                for (int c = 0; c < w; ++c)
                    if (sobj.mask[r][c] == color) { found = true; break; }
            if (!found) continue;
            if (res.count >= GPU_KERNEL_MAX_OBJECTS) return res;
            gpu_fixed_object& obj = res.objects[res.count];
            obj.x0 = sobj.x0; obj.y0 = sobj.y0; obj.x1 = sobj.x1; obj.y1 = sobj.y1;
            obj.width = sobj.width; obj.height = sobj.height;
            // A carve CLEARS, and that is what makes a carve useful: the pick
            // that follows it is how a sentence narrows to one layer. Handing
            // the mark down instead leaves every layer marked, and since
            // selectors only ever ADD, the next pick can no longer say "just
            // this one" -- measured on 05a7bcf2, where it recoloured all three.
            obj.selected = false;
            obj.mask.clear();
            for (int r = 0; r < h; ++r)
                for (int c = 0; c < w; ++c)
                    if (sobj.mask[r][c] == color) obj.mask[r][c] = (int8_t)color;
            res.count++;
        }
    }
    return res;
}

// SelectMostCommonColor: mark the cells of the most frequent color across the
// whole list (a frequency-ranked, symbolic color selector — "most", resolved
// per list). Lossless — the old Foreground (drop that color + resegment) is now
// `SelectMostCommonColor DeleteSelection Split[1]`.
__device__ gpu_object_list& gpu_exec_select_most_common_color(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    // The count the reading replays (color_tally, 8b_op_math.hpp).
    arc::color_tally freq;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c) freq.add((int8_t)obj.mask[r][c]);
    }
    const int top = freq.most();
    if (top < 0) return res;  // nothing painted -> nothing to mark
    // Mark every object that contains the most-common color. After a carve
    // (Colors/Split) each object is single-color, so this is exactly "the
    // top-color object(s)".
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height && !obj.selected; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] == top) { obj.selected = true; break; }
    }
    return res;
}

// SortBySize: reorder the PICKED objects by pixel count, among the slots they
// already occupy. Everything unpicked stays exactly where it is, so the act
// touches only what was chosen -- the same law the other acts follow, said for
// an ordering: a sort of some is a sort of their own places.
// value=0 -> descending (biggest first), value=1 -> ascending (smallest first).
// Nothing picked means nothing is reordered.
__device__ gpu_object_list& gpu_exec_sort_by_size(gpu_exec_context& ctx, const gpu_object_list& src, int desc) {
    auto& res = gpu_acquire(ctx);
    if (src.count <= 0) return res;
    int n = src.count > GPU_KERNEL_MAX_OBJECTS ? GPU_KERNEL_MAX_OBJECTS : src.count;
    // The whole list rides through in place first; the sort only moves the picked.
    for (int i = 0; i < n; ++i) res.objects[i] = src.objects[i];
    res.count = n;

    int slots[GPU_KERNEL_MAX_OBJECTS], counts[GPU_KERNEL_MAX_OBJECTS], indices[GPU_KERNEL_MAX_OBJECTS];
    int m = 0;
    for (int i = 0; i < n; ++i)
        if (src.objects[i].selected) {
            slots[m] = i; indices[m] = i;
            counts[m] = gpu_count_pixels(src.objects[i]);
            ++m;
        }
    if (m == 0) return res;

    for (int i = 1; i < m; ++i) {
        int j = i;
        while (j > 0 && (desc ? counts[j] > counts[j-1] : counts[j] < counts[j-1])) {
            int tc = counts[j]; counts[j] = counts[j-1]; counts[j-1] = tc;
            int ti = indices[j]; indices[j] = indices[j-1]; indices[j-1] = ti;
            j--;
        }
    }
    // k-th picked slot receives the k-th object in sorted order.
    for (int k = 0; k < m; ++k) res.objects[slots[k]] = src.objects[indices[k]];
    return res;
}

// SizeRankColorMap: recolor each object in src using puzzle-level by-size palette
// at the corresponding rank. If palette is empty/invalid (count=0), returns empty.
// If src has more objects than palette entries, the trailing objects are dropped.
__device__ gpu_object_list& gpu_exec_size_rank_color_map(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    int pal_n = ctx.output_colors_by_size_count;
    if (pal_n <= 0) return res;
    int n = src.count < pal_n ? src.count : pal_n;
    if (n > GPU_KERNEL_MAX_OBJECTS) n = GPU_KERNEL_MAX_OBJECTS;
    for (int i = 0; i < n; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        int8_t color = ctx.output_colors_by_size[i];
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] != GPU_TRANSPARENT) obj.mask[r][c] = color;
        res.count++;
    }
    return res;
}

// SelectObjectBySize[rank]: mark every cell of the rank-th object by pixel count
// (largest=true -> biggest first, else smallest first). Lossless — the list is
// unchanged, only the rank-th object's cells get selected. Keeping only it is
// `SelectObjectBySize[r] InvertSelection DeleteSelection`, same as Filter.
// Add a mark to the one object with no twin, as `how` reads twins
// (unique_piece); keep every object and prior mark.
__device__ gpu_object_list& gpu_exec_select_unique(gpu_exec_context& ctx, const gpu_object_list& src, int8_t how) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    const int found = arc::unique_piece(how, src.count,
        [&](int i) { return src.objects[i].width; },
        [&](int i) { return src.objects[i].height; },
        [&](int i, int r, int c) { return (int8_t)src.objects[i].mask[r][c]; });
    if (found >= 0) res.objects[found].selected = true;
    return res;
}

__device__ gpu_object_list& gpu_exec_select_object_by_size(gpu_exec_context& ctx, const gpu_object_list& src, int rank, bool largest) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    // Past the ranks, every object with the most -- or the fewest -- of a count (count_of).
    if (rank >= arc::SIZE_RANKS) {
        const int kind = rank - arc::SIZE_RANKS;
        arc::each_extreme(res.count, largest, [&](int i) {
            const gpu_fixed_object& obj = res.objects[i];
            return arc::count_of(kind, min((int)obj.height, GPU_MAX_GRID), min((int)obj.width, GPU_MAX_GRID),
                                 [&](int row, int col) { return (int)obj.mask[row][col]; });
        }, [&](int i) { res.objects[i].selected = true; });
        return res;
    }
    if (rank < 0 || rank >= res.count) return res;
    int counts[GPU_KERNEL_MAX_OBJECTS], indices[GPU_KERNEL_MAX_OBJECTS];
    for (int i = 0; i < res.count; ++i) {
        counts[i] = gpu_count_pixels(res.objects[i]);
        indices[i] = i;
    }
    for (int i = 1; i < res.count; ++i) {
        int j = i;
        while (j > 0 && (largest ? counts[j] > counts[j-1] : counts[j] < counts[j-1])) {
            int tc = counts[j]; counts[j] = counts[j-1]; counts[j-1] = tc;
            int ti = indices[j]; indices[j] = indices[j-1]; indices[j-1] = ti;
            j--;
        }
    }
    res.objects[indices[rank]].selected = true;
    return res;
}

// s clipped to rows [r0, r1) and columns [c0, c1), and there to the cells
// `keep(row, col)` holds, in that frame; false when nothing is left.
template <class Keep>
__device__ inline bool gpu_clip_to(const gpu_fixed_object& s, int r0, int c0, int r1, int c1, Keep keep,
                                   gpu_fixed_object& obj) {
    obj.x0 = (int16_t)c0; obj.y0 = (int16_t)r0;
    obj.x1 = (int16_t)(c1 - 1); obj.y1 = (int16_t)(r1 - 1);
    obj.width = (int16_t)(c1 - c0); obj.height = (int16_t)(r1 - r0);
    obj.selected = false;
    obj.mask.clear();
    bool any = false;
    GPU_DEV_FOR_EACH_PIXEL(s, sr, sc, v) {
        const int gr = s.y0 + sr, gc = s.x0 + sc;
        if (gr < r0 || gr >= r1 || gc < c0 || gc >= c1 || !keep(gr, gc)) continue;
        obj.mask[gr - r0][gc - c0] = v;
        any = true;
    }
    return any;
}
// s clipped to rows [r0, r1) and columns [c0, c1); false when nothing is left.
__device__ inline bool gpu_clip_to_box(const gpu_fixed_object& s, int r0, int c0, int r1, int c1,
                                       gpu_fixed_object& obj) {
    return gpu_clip_to(s, r0, c0, r1, c1, [](int, int) { return true; }, obj);
}

// Panes: Partition's act with the cut already decided. Partition reads the
// walls off a selection; the elected split read them off every training pair at
// once, so here the boxes simply come up with the example. What is cut is
// whatever the chain holds -- explicitly remove the ground first and the panes come
// back as figures, hand it Colors and they come back a layer at a time.
__device__ gpu_object_list& gpu_exec_panes(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int b = 0; b < ctx.pane_count; ++b) {
        const int r0 = ctx.pane_boxes[b][0], c0 = ctx.pane_boxes[b][1];
        const int w  = ctx.pane_boxes[b][2], h  = ctx.pane_boxes[b][3];
        if (w <= 0 || h <= 0) continue;
        const int r1 = r0 + h, c1 = c0 + w;
        for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
            const auto& s = src.objects[i];
            gpu_fixed_object obj;
            gpu_clip_to_box(s, r0, c0, r1, c1, obj);
            // EVERY PANE COMES BACK, even one the source never reached -- empty,
            // but there. Partition drops such a tile because its tiles are
            // whatever the walls happen to leave; panes are a fixed set that gets
            // stacked, so a missing one would silently shift every later pane
            // into the wrong place in the list.
            res.objects[res.count++] = obj;
        }
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_intersection(gpu_exec_context& ctx, const gpu_object_list& src) {
    // canvas := canvas ∩ src — clear canvas pixels not covered by any src object.
    // Mirrors Union (canvas ∪ src) and Difference (canvas − src): write-only on canvas, pass src through.
    bool covered[GPU_MAX_GRID][GPU_MAX_GRID];
    memset(covered, 0, sizeof(covered));
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        // Only what was picked lands. A paint that ignored the mark would undo
        // the pick the sentence just made -- which is why, until now, the only
        // way to print one layer alone was to delete every other one first.
        if (!obj.selected) continue;
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int gr = obj.y0 + r, gc = obj.x0 + c;
            if (gr >= 0 && gr < ctx.output.height && gc >= 0 && gc < ctx.output.width
                && gpu_get_pixel(obj, r, c) != GPU_TRANSPARENT) {
                covered[gr][gc] = true;
            }
        }
    }
    for (int r = 0; r < ctx.output.height; ++r)
        for (int c = 0; c < ctx.output.width; ++c)
            if (!covered[r][c]) ctx.output.data[r][c] = GPU_TRANSPARENT;

    auto& pass = gpu_acquire(ctx);
    for (int i = 0; i < src.count && pass.count < GPU_KERNEL_MAX_OBJECTS; ++i)
        pass.objects[pass.count++] = src.objects[i];
    return pass;
}

// Copy one object verbatim, mask and marks included. The one home for "this
// object rides through untouched", which every act needs for what wasn't picked.
__device__ inline void gpu_warp_copy_object(gpu_fixed_object& dobj, const gpu_fixed_object& sobj, int lane) {
    constexpr int obj_bytes = (int)sizeof(gpu_fixed_object);
    int8_t* d = reinterpret_cast<int8_t*>(&dobj);
    const int8_t* s = reinterpret_cast<const int8_t*>(&sobj);
    for (int b = lane; b < obj_bytes; b += 32) d[b] = s[b];
}

// Copy the first `n` objects verbatim and set the output count.
__device__ inline void gpu_warp_copy_objects(gpu_object_list* res, const gpu_object_list* src, int n, int lane) {
    for (int i = 0; i < n; ++i) gpu_warp_copy_object(res->objects[i], src->objects[i], lane);
    if (lane == 0) res->count = n;
    __syncwarp();
}

// Calls cell(row, col, v) for each painted, on-canvas cell of the picked objects
// among src's first n, one object at a time so later objects win shared cells.
template <class Cell>
__device__ inline void gpu_each_picked_cell_warp(const gpu_exec_context& ctx, const gpu_object_list& src,
                                                 int n, int lane, Cell cell) {
    const int oh = ctx.output.height, ow = ctx.output.width;
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj = src.objects[i];
        if (!obj.selected) continue;
        const int pw = obj.width, total = pw * obj.height;
        for (int p = lane; p < total; p += 32) {
            const int r = p / pw, c = p - r * pw;
            const int8_t v = obj.mask[r][c];
            if (v == GPU_TRANSPARENT) continue;
            const int gr = obj.y0 + r, gc = obj.x0 + c;
            if (gr >= 0 && gr < oh && gc >= 0 && gc < ow) cell(gr, gc, v);
        }
        __syncwarp();
    }
}

// Warp-parallel Intersection: canvas := canvas ∩ src (clear canvas cells not
// covered by any src object), then pass src through. Was lane-0 serial with a
// 3.6KB `covered[30][30]` local array. Here `covered` lives in per-job global
// scratch (ctx.op_scratch, 900B, unused during ops); the 32 lanes stripe both
// the mark phase (src pixels -> covered, all writes set 1 so conflicts benign)
// and the clear phase (output cells). The canvas is packed, so two side-by-side
// cells share a byte and the clear goes through gpu_paint_cell rather than the
// plain proxy. Mirrors gpu_exec_union_warp.
__device__ void gpu_exec_intersection_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    int oh = ctx.output.height, ow = ctx.output.width;
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);

    uint8_t* covered = reinterpret_cast<uint8_t*>(ctx.op_scratch);  // 900B per-job global
    for (int p = lane; p < GPU_GRID_CELLS; p += 32) covered[p] = 0;
    __syncwarp();

    gpu_each_picked_cell_warp(ctx, *src, n, lane, [&](int r, int c, int8_t) {
        covered[r * GPU_MAX_GRID + c] = 1;
    });

    // Clear phase: drop canvas cells not covered by any src object.
    for (int p = lane; p < oh * ow; p += 32) {
        int r = p / ow, c = p - r * ow;
        if (!covered[r * GPU_MAX_GRID + c]) gpu_paint_cell(ctx.output.data, r, c, GPU_TRANSPARENT);
    }
    __syncwarp();

    gpu_warp_copy_objects(&gpu_acquire_and_publish(ctx), src, n, lane);
}

// Warp-parallel union: composite src's pixels onto the canvas, then pass src
// through UNCHANGED (canvas := canvas ∪ src). The per-pixel paint — the measured
// ~27% hotspot, previously lane-0 serial — is striped across the 32 lanes:
// pixels of one object map to disjoint canvas cells (no write conflict), and
// overlapping objects resolve later-wins because the warp syncs after each one.
// Without that sync, later-wins held only when both objects put the cell on the
// same lane, which untrimmed full-grid frames happen to do; a trimmed dot and
// the square around it put one cell on two lanes, and the CAS retry let the
// earlier object land last (00dbd492 lost its markers under the fill). Union
// must NOT collapse src into one object (that's UnionList); keeping src intact lets
// a downstream per-object op still see each object. Mirrors gpu_exec_*_warp.
__device__ void gpu_exec_union_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    const int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    gpu_each_picked_cell_warp(ctx, *src, n, lane, [&](int r, int c, int8_t v) {
        gpu_paint_cell(ctx.output.data, r, c, v);
    });
    gpu_warp_copy_objects(&gpu_acquire_and_publish(ctx), src, n, lane);
}

// Withhold: closes the sentence and paints nothing. The hand rides through so
// the next sentence can act on it, and the canvas is left exactly as it was --
// which is the whole point: a sentence that only prepares should read as having
// done nothing, not as having damaged the picture.
__device__ void gpu_exec_withhold_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    gpu_warp_copy_objects(&gpu_acquire_and_publish(ctx), src, n, lane);
}

// Underlay: like Union but paints each pixel ONLY into transparent canvas cells
// (paint-behind). One-op, non-destructive z-order -- "put this under what's
// already there" instead of reordering the chain. The only change from
// gpu_exec_union_warp is the per-cell guard: store iff the canvas cell is still
// transparent. src is passed through unchanged, same as Union.
__device__ void gpu_exec_underlay_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    const int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    gpu_each_picked_cell_warp(ctx, *src, n, lane, [&](int r, int c, int8_t v) {
        if (ctx.output.data[r][c] == GPU_TRANSPARENT) gpu_paint_cell(ctx.output.data, r, c, v);
    });
    gpu_warp_copy_objects(&gpu_acquire_and_publish(ctx), src, n, lane);
}

__device__ gpu_object_list& gpu_exec_difference(gpu_exec_context& ctx, const gpu_object_list& src) {
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        // Only what was picked lands. A paint that ignored the mark would undo
        // the pick the sentence just made -- which is why, until now, the only
        // way to print one layer alone was to delete every other one first.
        if (!obj.selected) continue;
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int gr = obj.y0 + r, gc = obj.x0 + c;
            if (gr >= 0 && gr < ctx.output.height && gc >= 0 && gc < ctx.output.width) {
                int8_t sv = gpu_get_pixel(obj, r, c);
                if (sv != GPU_TRANSPARENT) ctx.output.data[gr][gc] = GPU_TRANSPARENT;
            }
        }
    }
    auto& pass = gpu_acquire(ctx);
    for (int i = 0; i < src.count && pass.count < GPU_KERNEL_MAX_OBJECTS; ++i)
        pass.objects[pass.count++] = src.objects[i];
    return pass;
}

// Wall = every non-transparent cell. Outside = 4-connected flood from grid
// edges through non-wall (transparent) cells. Interior = transparent cells
// unreachable from the edge -> painted `fill_value`. Warp-parallel: wall-build
// and interior-paint stripe across 32 lanes; the flood replaces the serial
// queue-BFS with monotonic LABEL PROPAGATION in shared memory — each sweep
// marks any transparent cell touching the edge or an `outside` neighbour, until
// a sweep changes nothing (warp __any_sync). `outside` only flips 0->1, so the
// intra-sweep cross-lane reads are benign; it converges in <= mh*mw sweeps.
__device__ void gpu_interior_paint_warp(const gpu_fixed_object& obj_src,
                                        gpu_fixed_object& out, int8_t fill_value, int lane) {
    int mw = min((int)obj_src.width, GPU_MAX_GRID), mh = min((int)obj_src.height, GPU_MAX_GRID);  // clamp: bbox can exceed the 30x30 wall/outside shared arrays
    {   // Copy obj_src -> out (striped); interior is then painted onto it.
        const int8_t* s = reinterpret_cast<const int8_t*>(&obj_src);
        int8_t* d = reinterpret_cast<int8_t*>(&out);
        for (int b = lane; b < (int)sizeof(gpu_fixed_object); b += 32) d[b] = s[b];
    }
    __syncwarp();
    if (mw <= 0 || mh <= 0) return;

    // wall/outside in __shared__ (fast for the relaxation sweeps). Tested moving them
    // to global (ctx.cc_labels) to free shared for occupancy: it freed almost nothing
    // (compiler overlays this with the spill area) and cost 25% on InteriorColor-heavy
    // puzzles. The shared cap is the smem-spill area, not this. Reverted.
    __shared__ uint8_t wall[GPU_MAX_GRID][GPU_MAX_GRID];
    __shared__ uint8_t outside[GPU_MAX_GRID][GPU_MAX_GRID];
    int total = mh * mw;
    for (int p = lane; p < total; p += 32) {
        int r = p / mw, c = p - r * mw;
        wall[r][c]    = (obj_src.mask[r][c] != GPU_TRANSPARENT) ? 1 : 0;
        outside[r][c] = 0;
    }
    __syncwarp();

    // Monotonic flood by sweeps; bounded by `total` (can't need more sweeps
    // than cells), breaks early when a sweep marks nothing new.
    for (int sweep = 0; sweep < total; ++sweep) {
        int changed = 0;
        for (int p = lane; p < total; p += 32) {
            int r = p / mw, c = p - r * mw;
            if (!wall[r][c] && !outside[r][c]) {
                bool edge = (r == 0 || r == mh - 1 || c == 0 || c == mw - 1);
                bool nb = (r > 0      && outside[r - 1][c]) ||
                          (r < mh - 1 && outside[r + 1][c]) ||
                          (c > 0      && outside[r][c - 1]) ||
                          (c < mw - 1 && outside[r][c + 1]);
                if (edge || nb) { outside[r][c] = 1; changed = 1; }
            }
        }
        __syncwarp();
        if (!__any_sync(0xffffffffu, changed)) break;
        __syncwarp();
    }

    // Paint interior: transparent cells the flood never reached. ONE PACKED BYTE
    // per lane (2 same-row cells) so the 4-bit packing has no shared-byte race —
    // a lane-striped per-cell proxy write would clobber the neighbour nibble.
    // Interior cell -> fill_value; every other cell keeps its copied source value.
    for (int b = lane; b < (int)sizeof(out.mask.bytes); b += 32) {
        int idx = arc::cells::first_cell_of_byte(b);
        int r = idx / GPU_MAX_GRID, c = idx - r * GPU_MAX_GRID;   // c even => c,c+1 same row
        int8_t v0 = (r < mh && c     < mw && !wall[r][c]     && !outside[r][c])     ? fill_value : (int8_t)obj_src.mask[r][c];
        int8_t v1 = (r < mh && c + 1 < mw && !wall[r][c + 1] && !outside[r][c + 1]) ? fill_value : (int8_t)obj_src.mask[r][c + 1];
        out.mask.bytes[b] = arc::cells::pair_byte(v0, v1);
    }
    __syncwarp();
}

// Fills the interior of each picked, non-empty object with fill(obj), asked on
// lane 0; a negative fill leaves the object as it is. Empty objects drop out.
template <class Fill>
__device__ inline void gpu_warp_fill_interiors(gpu_exec_context& ctx, int lane, Fill fill) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    const int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    int outc = 0;
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj_src = src->objects[i];
        if (obj_src.width <= 0 || obj_src.height <= 0) continue;
        int f = -1;
        if (obj_src.selected) {
            if (lane == 0) f = fill(obj_src);
            f = __shfl_sync(0xffffffffu, f, 0);
        }
        if (f < 0) {
            gpu_warp_copy_object(res->objects[outc], obj_src, lane);
            __syncwarp();
        } else {
            gpu_interior_paint_warp(obj_src, res->objects[outc], (int8_t)f, lane);
        }
        ++outc;
    }
    if (lane == 0) res->count = outc;
    __syncwarp();
}

__device__ void gpu_exec_interior_color_warp(gpu_exec_context& ctx, int8_t color, int lane) {
    gpu_warp_fill_interiors(ctx, lane, [&](const gpu_fixed_object& obj) {
        return color < 0 ? (int)gpu_obj_first_color(obj) : (int)color;
    });
}

__device__ gpu_object_list& gpu_exec_recolor(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        int8_t f = gpu_obj_resolve_color(src.objects[i], color);
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] != GPU_TRANSPARENT) obj.mask[r][c] = f;
        res.count++;
    }
    return res;
}

// ==== Shared building blocks for warp-parallel object ops ====
// Background for newcomers: a "warp" is a team of 32 GPU threads (lanes) that
// run every line together. To share work, lane L handles bytes L, L+32, L+64...
// ("striping"), so the team covers all bytes with no overlap and no races.
// Every warp op below is built from the same three steps: copy object headers,
// rebuild the mask, then lane 0 stamps the new size. Only "which source pixel
// lands in dest cell (r,c)" differs per op, so that is the one parameter.
// Adding a new op later: compute the new width/height, call gpu_warp_remap_object
// with a one-line "where does cell (r,c) come from" lambda, then stamp the new
// dims/anchor on lane 0. Ops that only touch headers or only masks can use
// gpu_warp_copy_header / gpu_warp_fill_mask alone instead of the full skeleton.

// Copy one object's header (everything before the packed mask: anchor, dims).
__device__ inline void gpu_warp_copy_header(const gpu_fixed_object& sobj, gpu_fixed_object& dobj, int lane) {
    constexpr int meta_bytes = (int)sizeof(gpu_fixed_object) - (int)sizeof(arc::cells);
    // The mask is last and the struct has no trailing padding, so "everything
    // before the mask" and "the struct minus the mask" are the same bytes. The
    // alignment arc::cells asks for makes that a layout fact worth pinning.
    static_assert(offsetof(gpu_fixed_object, mask) == meta_bytes,
                  "header copy must stop exactly where the mask starts");
    int8_t* d = reinterpret_cast<int8_t*>(&dobj);
    const int8_t* s = reinterpret_cast<const int8_t*>(&sobj);
    for (int b = lane; b < meta_bytes; b += 32) d[b] = s[b];
}

// Fill dobj's nw x nh mask cell by cell, asking `src_pixel(r, c)` for each
// cell's color; cells outside nw x nh become transparent. Each lane owns ONE
// packed dest byte = 2 dest cells in the same row (30 cells/row = 15 bytes,
// even => no row crossing), so every mask byte is written exactly once and the
// 4-bit packing is race-free with no pre-clear. No trailing sync: the caller's
// lane-0 size stamp touches different bytes, one __syncwarp after both suffices.
template <class PixelFn>
__device__ inline void gpu_warp_fill_mask(gpu_fixed_object& dobj, int nw, int nh, int lane, PixelFn src_pixel) {
    for (int bb = lane; bb < (int)sizeof(dobj.mask.bytes); bb += 32) {
        int idx = arc::cells::first_cell_of_byte(bb);
        int dr = idx / GPU_MAX_GRID;
        int dc = idx - dr * GPU_MAX_GRID;            // even
        int8_t pv[2];
        #pragma unroll
        for (int t2 = 0; t2 < 2; ++t2) {
            int ddc = dc + t2;
            pv[t2] = (dr < nh && ddc < nw) ? src_pixel(dr, ddc) : (int8_t)GPU_TRANSPARENT;
        }
        dobj.mask.bytes[bb] = arc::cells::pair_byte(pv[0], pv[1]);
    }
}

// The full per-object skeleton: header copy, sync, mask fill. After it returns,
// the caller's lane 0 stamps the new dims/anchor, then one __syncwarp.
template <class PixelFn>
__device__ inline void gpu_warp_remap_object(const gpu_fixed_object& sobj, gpu_fixed_object& dobj,
                                             int nw, int nh, int lane, PixelFn src_pixel) {
    gpu_warp_copy_header(sobj, dobj, lane);
    __syncwarp();
    gpu_warp_fill_mask(dobj, nw, nh, lane, src_pixel);
}

// Remaps each picked object of src into res; the rest ride through untouched.
// size(sobj, nw, nh) gives the new dims, pixel(sobj, r, c) each dest cell.
template <class SizeFn, class PixelFn>
__device__ inline void gpu_warp_remap_picked(const gpu_object_list* src, gpu_object_list* res, int lane,
                                             SizeFn size, PixelFn pixel) {
    const int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& sobj = src->objects[i];
        gpu_fixed_object& obj = res->objects[i];
        if (!sobj.selected) {
            gpu_warp_copy_object(obj, sobj, lane);
            if (lane == 0) res->count = i + 1;
            __syncwarp();
            continue;
        }
        int nw, nh;
        size(sobj, nw, nh);
        gpu_warp_remap_object(sobj, obj, nw, nh, lane,
                              [&](int dr, int dc) { return pixel(sobj, dr, dc); });
        if (lane == 0) {
            obj.width = (int16_t)nw; obj.height = (int16_t)nh;
            obj.x1 = obj.x0 + (int16_t)nw - 1; obj.y1 = obj.y0 + (int16_t)nh - 1;
            res->count = i + 1;
        }
        __syncwarp();
    }
}

// Warp-parallel Recolor: every non-transparent cell -> `color`. The per-cell
// recolor was a measured lane-0-serial hotspot (~70% of all kernel stall is on
// 1-lane instructions). Here the 32 lanes stripe the mask in WHOLE PACKED BYTES
// (2 cells/byte) so the 4-bit RMW has no shared-byte race (see gpu_paint_cell).
// Object metadata (x0/y0/dims) is unchanged, so we deep-copy each src object
// then overwrite only its mask bytes in parallel.
__device__ void gpu_exec_recolor_warp(gpu_exec_context& ctx, int8_t color, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    constexpr int mask_bytes = sizeof(arc::cells);
    int8_t paint = color;
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& sobj = src->objects[i];
        gpu_fixed_object& obj = res->objects[i];
        // Color-source values (10 = rarest, 11 = most common) resolve per object:
        // lane 0 scans, everyone gets the color via shuffle.
        if (color >= GPU_MAX_COLORS) {
            int f = 0;
            if (lane == 0) f = gpu_obj_resolve_color(sobj, color);
            f = __shfl_sync(0xffffffffu, f, 0);
            paint = (int8_t)f;
        }
        gpu_warp_copy_header(sobj, obj, lane);
        // Selection is per object and the flag is the only truth: a selected
        // object recolors every one of its painted cells, an unselected object
        // rides through untouched. No selection in the chain means Recolor
        // changes nothing — acts NEVER fall back to "everything".
        // Stripe mask bytes: each lane owns whole bytes (2 cells) -> no RMW race.
        bool sel = sobj.selected;
        for (int b = lane; b < mask_bytes; b += 32) {
            const uint8_t sb = sobj.mask.bytes[b];
            const int8_t first = arc::cells::dec(sb, 0);
            const int8_t second = arc::cells::dec(sb, arc::cells::CELL_BITS);
            obj.mask.bytes[b] = arc::cells::pair_byte(
                (first == GPU_TRANSPARENT || !sel) ? first : paint,
                (second == GPU_TRANSPARENT || !sel) ? second : paint);
        }
    }
    if (lane == 0) res->count = n;
    __syncwarp();
}

// THE OBJECTS A PARTNER WORD READS, as the op holds them: the box each one's
// paint fills on the canvas, whether it paints a cell there, whether it is
// picked, its color and the colors it holds -- the questions the partner
// helpers ask (each_partner and kin, 8b_op_math.hpp), asked as inference's
// partner_pieces asks them of the pieces it replays.
struct gpu_partner_view {
    const gpu_object_list& list;
    __device__ int count() const { return list.count; }
    __device__ bool is_picked(int i) const { return list.objects[i].selected; }
    __device__ int color(int i) const { return gpu_obj_first_color(list.objects[i]); }
    __device__ arc::cell_box box(int i) const {
        const gpu_fixed_object& obj = list.objects[i];
        const gpu_painted_box paint = gpu_box_of_paint(obj);
        arc::cell_box found;
        if (paint.width > 0 && paint.height > 0) {
            found.include_cell(obj.y0 + paint.y0, obj.x0 + paint.x0);
            found.include_cell(obj.y0 + paint.y0 + paint.height - 1, obj.x0 + paint.x0 + paint.width - 1);
        }
        return found;
    }
    __device__ bool painted(int i, int row, int col) const { return color_at(i, row, col) != GPU_TRANSPARENT; }
    // The blocks object i's shape is drawn in (blocks_of), its colors aside.
    __device__ arc::shape_blocks blocks(int i) const {
        const arc::cell_box own = box(i);
        return arc::blocks_of(own.width(), own.height(),
                              [&](int row, int col) { return color_at(i, own.row0 + row, own.col0 + col); },
                              arc::AS_IT_LIES);
    }
    // The color object i paints at that cell of the canvas, transparent off it.
    __device__ int color_at(int i, int row, int col) const {
        const gpu_fixed_object& obj = list.objects[i];
        const int down = row - obj.y0, across = col - obj.x0;
        return down >= 0 && across >= 0 && down < obj.height && across < obj.width ? (int)obj.mask[down][across]
                                                                                    : GPU_TRANSPARENT;
    }
    // Whether object i holds a cell of `color`.
    __device__ bool holds(int i, int color) const {
        GPU_DEV_FOR_EACH_PIXEL(list.objects[i], r, c, v) { (void)r; (void)c; if (v == color) return true; }
        return false;
    }
    // What each_partner asks (touches, shapes_alike).
    __device__ bool touching(int mine, int other, bool sides_only) const {
        return arc::touches(*this, mine, other, { 0, 0 }, sides_only);
    }
    __device__ bool alike(int mine, int other, int like) const { return arc::shapes_alike(*this, mine, other, like); }
    // The object that shows a cell of the key's drawing (key_piece_at).
    __device__ int key_piece(int row, int col) const { return arc::key_piece_at(*this, row, col); }
    // Object i's one partner by `how` among the objects not picked, or -1.
    __device__ int partner(arc::partner_by how, int i) const { return arc::partner_of(how, i, *this); }
    // Object i's one partner of `kind` (partner_of_kind) among the objects not picked, or -1.
    __device__ int partner_of_kind(int kind, int i) const { return arc::partner_of_kind(kind, i, *this); }
    // The color object i takes from its partners by `how`, or -1.
    __device__ int color_from(arc::partner_by how, int i) const { return arc::partner_color(how, i, *this); }
};

// Bridge[way]: the pieces handed back as they came, unmarked, then every bar as
// one drawing, marked, so what follows works on the bars. Which pieces pair up
// (the picked ones alone), the way they reach, and where each bar lies is
// 8b_op_math.hpp's bridge_bars, which the reading replays too; this only lays
// the bars down, each cell to the first bar that crosses it. One drawing, so a
// list that holds the pieces holds every bar too, however many gaps there are --
// one per gap ran out of room (dbc1a6ce's test, 19 dots and 17 bars). It sits
// in the whole grid's frame, as Split's pieces do.
__device__ gpu_object_list& gpu_exec_bridge(gpu_exec_context& ctx, const gpu_object_list& src, int8_t way) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        res.objects[res.count++].selected = false;
    }
    if (res.count >= GPU_KERNEL_MAX_OBJECTS) return res;
    // A pooled slot keeps whatever it last held, so every field is said.
    gpu_fixed_object& bars = res.objects[res.count];
    bars.x0 = 0; bars.y0 = 0;
    bars.x1 = (int16_t)(GPU_MAX_GRID - 1); bars.y1 = (int16_t)(GPU_MAX_GRID - 1);
    bars.width = bars.height = (int16_t)GPU_MAX_GRID;
    bars.selected = true;
    bars.mask.clear();
    bool drew = false;
    arc::bridge_bars(way, ctx.output.width, ctx.output.height, gpu_partner_view{ src },
        [&](const arc::cell_box& bar, int8_t color) {
            for (int r = max(bar.row0, 0); r <= min(bar.row1, GPU_MAX_GRID - 1); ++r)
                for (int c = max(bar.col0, 0); c <= min(bar.col1, GPU_MAX_GRID - 1); ++c)
                    if (bars.mask[r][c] == GPU_TRANSPARENT) {
                        bars.mask[r][c] = color;
                        drew = true;
                    }
            return true;
        });
    if (drew) ++res.count;
    return res;
}

// BidiColorMap[k]: the picked objects trade colors, each pixel of one color of
// a pair becoming the other. k in [0,44] is one fixed pair {a,b}, 0<=a<b<=9,
// numbered row-major (k=0 -> {0,1}, k=1 -> {0,2}, ... k=44 -> {8,9}); past
// those, the pairs a key drawn in the grid names, the objects not picked read
// along their rows [45] or their columns [46] -- 0becf7df's corner, a pair a
// row, other colors in every grid. What it trades is swaps_of's to say
// (8b_op_math.hpp), the table inference replays, so the two cannot disagree.
// Self-inverse (a swap), so it expresses involutive recolor tasks where
// Recolor/MapTo cannot: one node does a->b and b->a with identical program
// behavior, surviving examples that need the swap in either direction (e.g.
// 0d3d703e: chain Swap{1,5},{2,6},{3,4},{8,9}). Colors not in a pair are
// untouched, so disjoint swaps compose order-free.
__device__ gpu_object_list& gpu_exec_bidi_color_map(gpu_exec_context& ctx, const gpu_object_list& src, int8_t k) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    int8_t to[arc::COLOR_COUNT];
    if (!arc::swaps_of(k, gpu_partner_view{ src }, to)) return res;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        // Only what was picked swaps colors; the rest ride through as they are.
        if (!obj.selected) continue;
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c) {
                const int8_t v = obj.mask[r][c];
                if (arc::is_color(v) && to[v] >= 0) obj.mask[r][c] = to[v];
            }
    }
    return res;
}

// ColorFrom[how]: each picked object takes the color of its partner among the
// objects not picked -- the nearest, one in its rows or columns, the one around
// it or inside it, one it touches, one of its shape as it lies or turned -- and
// paints every cell it paints in it. The partner is found by partner_color
// (8b_op_math.hpp), the walk inference replays, so the two cannot disagree. An
// object with no partner, or partners that disagree, keeps its colors. Nothing
// moves and nothing is dropped.
// ColorFrom[9], past the partner ways: each picked object takes the color a key
// drawn in the grid -- the objects not picked -- shows at its place, the n-th
// picked in reading order the key's n-th entry (key_place_colors, 8b_op_math.hpp,
// the walk inference replays). Its scratch lives in a frame of its own.
__device__ __noinline__ static void gpu_paint_key_places(const gpu_object_list& src, gpu_object_list& res) {
    int8_t color[GPU_KERNEL_MAX_OBJECTS], band[GPU_KERNEL_MAX_OBJECTS];
    int16_t lows[GPU_KERNEL_MAX_OBJECTS], highs[GPU_KERNEL_MAX_OBJECTS];
    int order[GPU_KERNEL_MAX_OBJECTS];
    if (!arc::key_place_colors(gpu_partner_view{ src }, color, lows, highs, band, order)) return;
    for (int i = 0; i < res.count; ++i) {
        if (color[i] < 0) continue;
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] != GPU_TRANSPARENT) obj.mask[r][c] = color[i];
    }
}

__device__ gpu_object_list& gpu_exec_color_from(gpu_exec_context& ctx, const gpu_object_list& src, int8_t how) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    if (how == arc::KEY_BY_PLACE) {
        gpu_paint_key_places(src, res);
        return res;
    }
    const gpu_partner_view view{ src };
    // Cell by cell, from the one piece of its shape (colors_from_its_shape).
    if (how >= arc::COLORS_FROM_ITS_SHAPE) {
        for (int i = 0; i < src.count; ++i) {
            if (!src.objects[i].selected) continue;
            auto& obj = res.objects[i];
            arc::colors_from_its_shape(how - arc::COLORS_FROM_ITS_SHAPE, i, view, [&](int row, int col, int color) {
                const int r = row - obj.y0, c = col - obj.x0;
                if (r >= 0 && c >= 0 && r < obj.height && c < obj.width && arc::is_color(color))
                    obj.mask[r][c] = (int8_t)color;
            });
        }
        return res;
    }
    if (how < 0 || how >= (int8_t)arc::partner_by::count) return res;
    for (int i = 0; i < src.count; ++i) {
        if (!src.objects[i].selected) continue;
        const int color = view.color_from((arc::partner_by)how, i);
        if (color < 0) continue;
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] != GPU_TRANSPARENT) obj.mask[r][c] = (int8_t)color;
    }
    return res;
}

// WouldFit[how]: of the picked objects, only those that would fit into a gap of
// their one partner among the objects not picked stay picked -- the partner by
// `how`, as partner_of (8b_op_math.hpp) finds it, and the fit as would_fit asks
// it, turned or mirrored any way: the walk inference replays, so the two cannot
// disagree. Every partner is read off the marks as they came in. Nothing moves
// and nothing is painted; an object with no partner, or several, is not picked.
__device__ gpu_object_list& gpu_exec_would_fit(gpu_exec_context& ctx, const gpu_object_list& src, int8_t how) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    if (how < 0 || how >= (int8_t)arc::partner_by::count) return res;
    const gpu_partner_view view{ src };
    for (int i = 0; i < src.count; ++i) {
        if (!src.objects[i].selected) continue;
        const int partner = view.partner((arc::partner_by)how, i);
        res.objects[i].selected = partner >= 0 && arc::would_fit(i, partner, view);
    }
    return res;
}

// SelectLikeKey[how]: of the picked objects, only those like a piece of the key
// -- every object not picked -- stay picked: of its shape as it lies [0], or
// turned any way [1] (like_the_key, 8b_op_math.hpp), the question inference
// replays. Every likeness is read off the marks as they came in.
__device__ gpu_object_list& gpu_exec_select_like_key(gpu_exec_context& ctx, const gpu_object_list& src, int8_t how) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    const gpu_partner_view view{ src };
    for (int i = 0; i < src.count; ++i)
        if (src.objects[i].selected) res.objects[i].selected = arc::like_the_key(how, i, view);
    return res;
}

// SelectPartner[value]: the pick moves onto every partner each picked object
// has, among every object, by the way and on the side `value` names
// (each_partner_picked, 8b_op_math.hpp, which inference replays). Every
// partner is read off the marks as they came in; an object no picked one has
// for a partner comes out unpicked.
__device__ gpu_object_list& gpu_exec_select_partner(gpu_exec_context& ctx, const gpu_object_list& src, int8_t value) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = false;
    arc::each_partner_picked(value, gpu_partner_view{ src }, [&](int other) { res.objects[other].selected = true; });
    return res;
}

// SelectPartition[which]: the picked objects are walls, as Partition reads them,
// and the pick becomes every object lying in the part `which` they fence off
// (parts_the_pick_fences and lies_in_part, 8b_op_math.hpp, which inference
// replays), the parts numbered in reading order. The walls lie in no part, so
// they come out unpicked. The canvas is the input's, as Partition's is.
__device__ gpu_object_list& gpu_exec_select_partition(gpu_exec_context& ctx, const gpu_object_list& src, int which) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    const gpu_partner_view view{ src };
    int16_t part[arc::MAX_GRID_DIM][arc::MAX_GRID_DIM];
    arc::parts_the_pick_fences(view, ctx.input.width, ctx.input.height, part);
    for (int i = 0; i < src.count; ++i) res.objects[i].selected = arc::lies_in_part(view, i, which, part);
    return res;
}

// Partition: the selection marks the PARTITIONER: selected cells are walls, only
// VIEWED to fence the parts, never merged into the output. Strict, like every
// act: no selection means no walls, so the whole thing stays one part --
// Partition NEVER falls back to "all painted are walls". The parts are the ones
// the walls fence off, whatever their shape (parts_the_pick_fences, the parts
// SelectPartition picks by), each cut from the SOURCE objects: every object not
// a wall, clipped to the cells of a part, comes back in the part's box, part
// after part in reading order -- walls across the whole grid fence the tiles of
// a lattice, row by row. A source object crossing into two parts is severed
// there -- that is what partitioning means. No merge: distinct source layers stay
// distinct; want parts as single masks? place a UnionList yourself.
__device__ gpu_object_list& gpu_exec_partition(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    const int gw = ctx.input.width, gh = ctx.input.height;
    if (gw <= 0 || gh <= 0) return res;
    int16_t part[arc::MAX_GRID_DIM][arc::MAX_GRID_DIM];
    const int parts = arc::parts_the_pick_fences(gpu_partner_view{ src }, gw, gh, part);
    for (int which = 0; which < parts; ++which) {
        arc::cell_box box;
        for (int r = 0; r < gh && r < arc::MAX_GRID_DIM; ++r)
            for (int c = 0; c < gw && c < arc::MAX_GRID_DIM; ++c)
                if (part[r][c] == which) box.include_cell(r, c);
        if (box.empty()) continue;
        for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
            const auto& s = src.objects[i];
            if (s.selected) continue;  // walls are not content
            gpu_fixed_object obj;
            if (gpu_clip_to(s, box.row0, box.col0, box.row1 + 1, box.col1 + 1,
                            [&](int row, int col) { return part[row][col] == which; }, obj))
                res.objects[res.count++] = obj;
        }
    }
    return res;
}

// AlignTo[value]: each picked object slides, down or across alone, until its
// painted box starts on the row its one partner's does -- or, for values past
// the first partner_by::count, the column. The partner is found among the
// objects not picked by partner_of (8b_op_math.hpp), the walk inference
// replays: value % count says how. Partners are never picked, so none of them
// moves while the others read them. An object with no partner, or several,
// stays where it is.
__device__ gpu_object_list& gpu_exec_align_to(gpu_exec_context& ctx, const gpu_object_list& src, int8_t value) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    constexpr int ways = (int)arc::partner_by::count;
    if (value < 0 || value >= 2 * ways) return res;
    const bool columns = value >= ways;
    const auto how = (arc::partner_by)(value % ways);
    const gpu_partner_view view{ src };
    for (int i = 0; i < src.count; ++i) {
        if (!src.objects[i].selected) continue;
        const int partner = view.partner(how, i);
        if (partner < 0) continue;
        const arc::cell_box own = view.box(i), there = view.box(partner);
        auto& obj = res.objects[i];
        if (columns) { const int16_t by = (int16_t)(there.col0 - own.col0); obj.x0 += by; obj.x1 += by; }
        else         { const int16_t by = (int16_t)(there.row0 - own.row0); obj.y0 += by; obj.y1 += by; }
    }
    return res;
}

// MoveToward[value]: each picked object moves toward its one partner among the
// objects not picked -- of the kind value % PARTNER_KINDS names, each way
// partner_by finds one and then each color (partner_of_kind) -- by what value /
// PARTNER_KINDS names: one step, its own size, or until it touches the partner
// (move_toward, 8b_op_math.hpp), the move inference replays. Partners are never
// picked, so none of them moves while the others read them. An object with no
// partner, or several, stays where it is.
__device__ gpu_object_list& gpu_exec_move_toward(gpu_exec_context& ctx, const gpu_object_list& src, int8_t value) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    if (value < 0 || value >= (int)arc::toward_by::count * arc::PARTNER_KINDS) return res;
    const auto far = (arc::toward_by)(value / arc::PARTNER_KINDS);
    const int kind = value % arc::PARTNER_KINDS;
    const gpu_partner_view view{ src };
    for (int i = 0; i < src.count; ++i) {
        if (!src.objects[i].selected) continue;
        const int partner = view.partner_of_kind(kind, i);
        if (partner < 0) continue;
        const arc::direction_step by = arc::move_toward(far, i, partner, view, ctx.output.width, ctx.output.height);
        auto& obj = res.objects[i];
        obj.x0 += (int16_t)by.across; obj.x1 += (int16_t)by.across;
        obj.y0 += (int16_t)by.down;   obj.y1 += (int16_t)by.down;
    }
    return res;
}

// MirrorAround[value]: each picked object is drawn again mirrored around the
// middle of its one partner among the objects not picked -- left to right, top
// to bottom, or all four ways at once, value / MIRROR_PARTNERS -- or, for the
// last partner, around the middle of the grid itself. partner_of (8b_op_math.hpp)
// finds the partner, the walk inference replays. The picked object stays; its
// copies join the list after every object there was, picked as it is, as
// CopyTo's are -- a sentence that picked every piece paints what it holds
// without picking again. An object with no partner, or several, gets no copy.
__device__ gpu_object_list& gpu_exec_mirror_around(gpu_exec_context& ctx, const gpu_object_list& src, int8_t value) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    constexpr int partners = arc::MIRROR_PARTNERS;
    if (value < 0 || value >= (int)arc::mirror_by::count * partners) return res;
    const auto mirror = (arc::mirror_by)(value / partners);
    const int how = value % partners;
    const gpu_partner_view view{ src };
    for (int i = 0; i < src.count; ++i) {
        const gpu_fixed_object& piece = src.objects[i];
        if (!piece.selected || piece.width <= 0 || piece.height <= 0) continue;
        // The middle, the copies and where each lies are 7_infer.hpp's to say
        // (mirror_middle, each_mirror, mirrored_box), the arithmetic the reading
        // replays.
        int row2 = 0, col2 = 0;
        if (!arc::mirror_middle(how, i, view, ctx.input.width, ctx.input.height, row2, col2)) continue;
        const arc::cell_box box{ piece.y0, piece.x0, piece.y0 + piece.height - 1, piece.x0 + piece.width - 1 };
        bool full = false;
        arc::each_mirror(mirror, [&](bool down, bool across) {
            if (full || res.count >= GPU_KERNEL_MAX_OBJECTS) { full = true; return; }
            gpu_fixed_object& copy = res.objects[res.count++];
            copy = piece;
            const int pose = arc::mirror_pose(down, across);
            for (int r = 0; r < piece.height; ++r)
                for (int c = 0; c < piece.width; ++c) {
                    const auto [from_r, from_c] = arc::pose_source(pose, piece.width, piece.height, r, c);
                    copy.mask[r][c] = (int8_t)piece.mask[from_r][from_c];
                }
            const arc::cell_box to = arc::mirrored_box(box, row2, col2, down, across);
            copy.y0 = (int16_t)to.row0; copy.x0 = (int16_t)to.col0;
            copy.y1 = (int16_t)to.row1; copy.x1 = (int16_t)to.col1;
        });
        if (full) return res;
    }
    return res;
}

__device__ int gpu_object_hole_count(const gpu_fixed_object& obj) {
    return arc::object_hole_count(min((int)obj.height, GPU_MAX_GRID),
                             min((int)obj.width, GPU_MAX_GRID),
        [&](int row, int col) { return obj.mask[row][col] != GPU_TRANSPARENT; });
}

// SelectObjectsWithHoles[h]: mark the objects with exactly h enclosed holes;
// everything else rides through unmarked. Object-keyed (the `Objects` in the
// name says so). A selector answers "which" and nothing else — the acting is a
// separate op (Recolor), so the two choices cost a sum instead of the product
// the old fused HolesToColorMap[50] paid. Nothing is dropped, so several
// select/Recolor pairs chain order-free.
__device__ gpu_object_list& gpu_exec_select_objects_with_holes(gpu_exec_context& ctx, const gpu_object_list& src, int8_t h) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        if (gpu_object_hole_count(obj) == (int)h) obj.selected = true;
    }
    return res;
}

// SelectAtLeastCells[n]: mark the objects that paint at least n cells
// (paints_at_least, which the reading replays); the rest ride through
// unmarked. InvertSelection after it says "fewer than n".
__device__ gpu_object_list& gpu_exec_select_at_least_cells(gpu_exec_context& ctx, const gpu_object_list& src, int8_t n) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        if (arc::paints_at_least(min((int)obj.height, GPU_MAX_GRID), min((int)obj.width, GPU_MAX_GRID),
                                 [&](int row, int col) { return obj.mask[row][col] != GPU_TRANSPARENT; }, n))
            obj.selected = true;
    }
    return res;
}

// SelectColor[c]: mark every object that contains color c; the rest ride
// through unmarked.
__device__ gpu_object_list& gpu_exec_select_color(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height && !obj.selected; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] == color) { obj.selected = true; break; }
    }
    return res;
}

// Elect from the current layout, keeping the objects' frames (not their paint
// bounds). Like SelectColor, selection is additive and marks whole objects.
__device__ gpu_object_list& gpu_exec_select_elected_background(gpu_exec_context& ctx,
                                                              const gpu_object_list& src) {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool has_frame = false;
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        if (obj.width <= 0 || obj.height <= 0) continue;
        if (!has_frame) {
            x0 = x1 = obj.x0; y0 = y1 = obj.y0;
            has_frame = true;
        }
        x0 = min(x0, (int)obj.x0); y0 = min(y0, (int)obj.y0);
        x1 = max(x1, obj.x0 + obj.width); y1 = max(y1, obj.y0 + obj.height);
    }
    gpu_fixed_grid current;
    current.clear();
    current.width = (int16_t)max(0, min(GPU_MAX_GRID, x1 - x0));
    current.height = (int16_t)max(0, min(GPU_MAX_GRID, y1 - y0));
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            const int row = obj.y0 + r - y0, col = obj.x0 + c - x0;
            if (row >= 0 && row < current.height && col >= 0 && col < current.width)
                current.data[row][col] = v;
        }
    }
    const int8_t color = arc::confident_background(arc::elect_grid_background(current));
    // No winner adds no marks, including on wholly transparent objects.
    if (color < 0) {
        auto& res = gpu_acquire(ctx);
        gpu_copy_list(res, src);
        return res;
    }
    return gpu_exec_select_color(ctx, src, color);
}

// SelectAll: mark every object — the explicit "act on it all".
__device__ gpu_object_list& gpu_exec_select_all(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = true;
    return res;
}

// InvertSelection: flip every object's mark. With additive selectors (OR) this
// is NOT, and OR + NOT is the whole boolean algebra — "big AND red" is
// invert(invert(big) ∪ invert(red)) by De Morgan, no extra machinery.
__device__ gpu_object_list& gpu_exec_invert_selection(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    for (int i = 0; i < res.count; ++i)
        res.objects[i].selected = !res.objects[i].selected;
    return res;
}

// The color the puzzle-level table names for a piece's key of `kind` (its
// paint's key, gpu_shape_key_of_paint), as the reading looks it up
// (shape_table::color_of): -1 where it names none. The one lookup both shape
// ops make.
__device__ inline int gpu_shape_color(const gpu_exec_context& ctx, const gpu_fixed_object& obj, int kind) {
    return ctx.shape_colors.color_of(gpu_shape_key_of_paint(obj, kind));
}

// ShapeColorMap[kind]: every picked piece takes the color its own key names in
// the puzzle-level table learned host-side -- its SHAPE, the same turned either
// way or mirrored either way, or its SYMMETRY, the ways round that leave its
// drawing as it was (shape_key_of, the key the reading learned) -- or the color
// the table gives every other key. A key the table names no color for leaves
// its piece as it was; nothing else changes. A piece colored by ANOTHER's shape
// is said in words (009d5c81's key naming the figure's color): the key takes
// its shape's color, the figure the key's (ColorFrom), and the key goes
// (DeleteSelection).
__device__ gpu_object_list& gpu_exec_shape_color_map(gpu_exec_context& ctx, const gpu_object_list& src, int kind) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    if (!ctx.shape_colors.holds()) return res;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        if (!obj.selected) continue;
        const int color = gpu_shape_color(ctx, obj, kind);
        if (color < 0) continue;
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] != GPU_TRANSPARENT) obj.mask[r][c] = (int8_t)color;
    }
    return res;
}

// ShapeToInteriorColorMap: per-object interior fill whose color is picked by the
// object's OWN shape via the puzzle-level table (shape_color_map_rule::detect_interior_fill).
// Sibling of ShapeColorMap, but paints the enclosed interior (not the body). For
// each object: lane 0 looks its shape's color up (gpu_shape_color, the lookup
// ShapeColorMap[0] makes) and broadcasts it; a hit floods the interior with that
// color, a miss passes the object through unchanged. Inside-noise (a
// non-transparent pixel in the hole, e.g. a marker) is a wall to the flood, so
// it keeps its color -- the trailing Union re-stamping the separate marker
// object reproduces it either way.
__device__ void gpu_exec_shape_to_interior_color_map_warp(gpu_exec_context& ctx, int lane) {
    gpu_warp_fill_interiors(ctx, lane, [&](const gpu_fixed_object& obj) {
        return gpu_shape_color(ctx, obj, (int)arc::shape_key::shape); });
}

// OverlapCorner: lay the objects out as a diagonal corner-chain. Objects are
// ordered left-to-right (by bbox x0, ties by y0); the first is placed with its
// top-left at the canvas origin (0,0); each subsequent object's top-left cell is
// set to the previous object's bottom-right cell -- a 1-pixel corner overlap.
// Only positions change (masks/colors untouched); the trailing Union resolves
// the shared corner -- the later (further-down-chain) object wins, which falls
// out of gpu_exec_union_warp's later-overwrites-earlier paint order for free.
//
// GENERAL FORM (not built; promote when a puzzle needs it). This is the
// sequential / corner / overlap generalization of AlignToCanvas, which is the
// batch / edge-flush / gap-0 corner of the same space:
//     placement: batch (fixed canvas)   ->  sequential (running canvas)
//     anchor:    edge-flush             ->  corner-to-corner
//     offset:    gap = 0                ->  stride = 1 (overlap)
// Two future annotations, both output-determined (so search/annotation, not
// inference):
//   * direction: the 4 diagonals {DR,DL,UR,UL}. Fixed DR (down-right) here --
//     the only verified case. Add as [dir]; do NOT try to reach the other three
//     via Flip*/Turn* composition, which needs an un-flip pass to undo the mirror.
//   * stride: the per-step diagonal offset (overlap 1 today; 0 = corner-touch,
//     >0 = gap). Inseparable from this op -- it's the cascade's step, not a
//     downstream op, because it compounds with chain index (object i shifts i*).
__device__ gpu_object_list& gpu_exec_overlap_corner(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    int n = min((int)src.count, GPU_KERNEL_MAX_OBJECTS);
    if (n <= 0) { gpu_copy_list(res, src); return res; }

    // Chain order = bbox (x0, y0) ascending. Selection sort.
    int order[GPU_KERNEL_MAX_OBJECTS];
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = 0; i < n; ++i) {
        int best = i;
        for (int j = i + 1; j < n; ++j) {
            const auto& a = src.objects[order[j]];
            const auto& b = src.objects[order[best]];
            if (a.x0 < b.x0 || (a.x0 == b.x0 && a.y0 < b.y0)) best = j;
        }
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }

    // obj0 top-left at (0,0); each next top-left = previous bottom-right.
    int tx = 0, ty = 0;
    for (int k = 0; k < n; ++k) {
        const auto& o = src.objects[order[k]];
        auto& d = res.objects[res.count++];
        d = o;
        int w = o.width, h = o.height;
        d.x0 = (int16_t)tx;           d.y0 = (int16_t)ty;
        d.x1 = (int16_t)(tx + w - 1); d.y1 = (int16_t)(ty + h - 1);
        tx = d.x1; ty = d.y1;  // 1px overlap: next.TL == this.BR
    }
    return res;
}

// Snap each object flush against the bbox of currently-painted canvas content.
// Faithful port of the old binary Align(F, T) with T = "current canvas content".
// For each object: pick the axis with the larger inter-bbox gap; on that axis,
// take whichever of the two adjacent-edge snaps requires the smaller move.
// Other axis is left untouched. Gap between bboxes ends up 0 (touching).
// No direction parameter — fully auto. If canvas is empty, no-op (passes objects through).
__device__ gpu_object_list& gpu_exec_align_to_canvas(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);

    // Read the painted canvas once; moving objects do not enlarge this target.
    arc::cell_box target;
    for (int row = 0; row < ctx.output.height; ++row)
        for (int col = 0; col < ctx.output.width; ++col)
            if (ctx.output.data[row][col] != GPU_TRANSPARENT) target.include_cell(row, col);

    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }

        if (!target.empty()) {
            const auto [dy, dx] = arc::shift_to_touch({obj.y0, obj.x0, obj.y1, obj.x1}, target);

            obj.x0 = (int16_t)(obj.x0 + dx);
            obj.y0 = (int16_t)(obj.y0 + dy);
            obj.x1 = (int16_t)(obj.x1 + dx);
            obj.y1 = (int16_t)(obj.y1 + dy);
        }
        res.count++;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_origin(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }
        obj.x1 = (int16_t)(obj.width - 1);
        obj.y1 = (int16_t)(obj.height - 1);
        obj.x0 = 0; obj.y0 = 0;
        res.count++;
    }
    return res;
}

// Intersect the objects whose `selected` equals `want` into ONE object (a cell
// survives only where every such object is painted, on the shared min-size box
// at origin — the existing local-coordinate alignment). Appends it to res
// carrying the shared flag if anything survives. First-of-group provides the
// color, matching the old whole-list behavior.
__device__ void gpu_intersect_group(const gpu_object_list& src, bool want, gpu_object_list& res) {
    int n = 0, first = -1, min_w = 0, min_h = 0;
    for (int i = 0; i < src.count; ++i) {
        if (src.objects[i].selected != want) continue;
        if (first < 0) { first = i; min_w = src.objects[i].width; min_h = src.objects[i].height; }
        else {
            if (src.objects[i].width  < min_w) min_w = src.objects[i].width;
            if (src.objects[i].height < min_h) min_h = src.objects[i].height;
        }
        ++n;
    }
    if (n == 0 || min_w <= 0 || min_h <= 0 || res.count >= GPU_KERNEL_MAX_OBJECTS) return;

    gpu_fixed_object obj;
    obj.x0 = 0; obj.y0 = 0;
    obj.width = (int16_t)min_w; obj.height = (int16_t)min_h;
    obj.x1 = (int16_t)(min_w - 1); obj.y1 = (int16_t)(min_h - 1);
    obj.selected = want;
    obj.mask.clear();

    int kept = 0;
    for (int r = 0; r < min_h; ++r)
        for (int c = 0; c < min_w; ++c) {
            bool all_set = true;
            for (int i = 0; i < src.count; ++i) {
                if (src.objects[i].selected != want) continue;
                if (src.objects[i].mask[r][c] == GPU_TRANSPARENT) { all_set = false; break; }
            }
            if (all_set) { obj.mask[r][c] = src.objects[first].mask[r][c]; ++kept; }
        }
    if (kept == 0) return;
    res.objects[res.count++] = obj;
}

// IntersectionList: intersect to one object PER selection group — the selected
// objects AND into one selected object, the unselected into one unselected
// object. Same selection-respecting discipline as UnionList: the mark survives
// the fold as object identity, never dropped. No selection => one unselected
// object (plain "intersect all").
__device__ gpu_object_list& gpu_exec_intersect_list(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    if (src.count <= 0) return res;
    gpu_intersect_group(src, false, res);
    gpu_intersect_group(src, true, res);
    return res;
}

// Merge the objects whose `selected` equals `want` into ONE object using world
// coordinates. Appends it to res (carrying the shared selected flag) if any
// such object has pixels. Returns nothing painted -> res unchanged.
__device__ void gpu_union_group(const gpu_object_list& src, bool want, gpu_object_list& res) {
    int x0 = GPU_MAX_GRID, y0 = GPU_MAX_GRID, x1 = -1, y1 = -1;
    for (int i = 0; i < src.count; ++i) {
        const auto& o = src.objects[i];
        if (o.selected != want) continue;
        if (o.x0 < x0) x0 = o.x0;
        if (o.y0 < y0) y0 = o.y0;
        if (o.x1 > x1) x1 = o.x1;
        if (o.y1 > y1) y1 = o.y1;
    }
    if (x1 < 0 || y1 < 0 || res.count >= GPU_KERNEL_MAX_OBJECTS) return;
    gpu_fixed_object& obj = res.objects[res.count];
    obj.x0 = (int16_t)x0; obj.y0 = (int16_t)y0;
    obj.x1 = (int16_t)x1; obj.y1 = (int16_t)y1;
    obj.width = (int16_t)(x1 - x0 + 1); obj.height = (int16_t)(y1 - y0 + 1);
    obj.selected = want;
    obj.mask.clear();
    bool any = false;
    for (int i = 0; i < src.count; ++i) {
        const auto& s = src.objects[i];
        if (s.selected != want) continue;
        for (int r = 0; r < s.height; ++r)
            for (int c = 0; c < s.width; ++c)
                if (s.mask[r][c] != GPU_TRANSPARENT) {
                    obj.mask[s.y0 + r - y0][s.x0 + c - x0] = s.mask[r][c];
                    any = true;
                }
    }
    if (any) res.count++;
}

// UnionList: merge to one object PER selection group — all selected pixels fuse
// into one selected object, all unselected into one unselected object. The
// wall/content split (which lives in `selected`) survives the merge as object
// identity, so `UnionList Partition` cuts whole tiles with the walls still marked.
// No selection => a single unselected object (plain "merge all").
__device__ gpu_object_list& gpu_exec_union_list(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    if (src.count <= 0) return res;
    gpu_union_group(src, false, res);  // content first, then walls
    gpu_union_group(src, true, res);
    return res;
}

// Bands for Layout: objects whose spans overlap meaningfully share a band.
// rows=true reads vertical spans (row bands), false horizontal (column bands).
// Sort spans by their start, then a new band opens unless the span overlaps
// the current band by at least half its own length — a 1-2px graze between
// genuinely different rows must not weld them into one (staggered grids chain
// transitively otherwise). band_of[i] = the object's band rank.
__device__ static int gpu_layout_bands(const gpu_object_list& src, int n, bool rows, int8_t* band_of) {
    int16_t lows[GPU_KERNEL_MAX_OBJECTS], highs[GPU_KERNEL_MAX_OBJECTS];
    int order[GPU_KERNEL_MAX_OBJECTS];
    for (int i = 0; i < n; ++i) {
        lows[i] = rows ? src.objects[i].y0 : src.objects[i].x0;
        highs[i] = rows ? src.objects[i].y1 : src.objects[i].x1;
    }
    return arc::overlapping_bands(n, lows, highs, band_of, order);
}

// WHICH BAND A PIECE SITS IN, not what the piece is. Every other selector names
// something the piece carries -- its color, its holes, its size -- so none of
// them can say "the middle row of them". This one names where a piece sits among
// the others, which is a question about the arrangement and not about the piece.
// Bands come from gpu_layout_bands, the same grouping Layout reads, so a row
// means one thing wherever it is asked about. Selectors ADD, so a band past the
// last one marks nothing rather than clearing what is already picked.
__device__ gpu_object_list& gpu_exec_select_band(gpu_exec_context& ctx, const gpu_object_list& src,
                                                 int8_t which, bool rows) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    int n = min((int)res.count, GPU_KERNEL_MAX_OBJECTS);
    if (n <= 0) return res;
    int8_t band_of[GPU_KERNEL_MAX_OBJECTS];
    gpu_layout_bands(res, n, rows, band_of);
    for (int i = 0; i < n; ++i)
        if (band_of[i] == which) res.objects[i].selected = true;
    return res;
}

// SelectRowFromMiddle[from] / SelectColumnFromMiddle[from]: mark every object
// with a painted cell on the row (column) `from` lines past the middle of the
// span every object's paint covers together -- the line as many lines from
// either end, none where the span has two (line_from_middle, the reading's own
// arithmetic). Keeps every object and prior mark.
__device__ gpu_object_list& gpu_exec_select_from_middle(gpu_exec_context& ctx, const gpu_object_list& src,
                                                        int8_t from, bool rows) {
    auto& res = gpu_acquire(ctx);
    gpu_copy_list(res, src);
    const int n = min((int)res.count, GPU_KERNEL_MAX_OBJECTS);
    int lo = GPU_MAX_GRID * 2, hi = -GPU_MAX_GRID * 2;
    for (int i = 0; i < n; ++i) {
        const auto& obj = res.objects[i];
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] != GPU_TRANSPARENT) {
                    const int line = rows ? obj.y0 + r : obj.x0 + c;
                    lo = min(lo, line);
                    hi = max(hi, line);
                }
    }
    const int line = arc::line_from_middle(lo, hi, from);
    if (line < 0) return res;
    for (int i = 0; i < n; ++i) {
        auto& obj = res.objects[i];
        bool on = false;
        for (int r = 0; r < obj.height && !on; ++r)
            for (int c = 0; c < obj.width && !on; ++c)
                on = obj.mask[r][c] != GPU_TRANSPARENT && (rows ? obj.y0 + r : obj.x0 + c) == line;
        if (on) obj.selected = true;
    }
    return res;
}

// Layout[k]: arrange the objects on k-sized graph paper. Overlapping vertical
// spans share a row, overlapping horizontal spans share a column; rows sort
// top-to-bottom, columns left-to-right. Each object's drawing is copied
// UNRESIZED into its own k x k cell (cell origin = (col*k, row*k)), clipped to
// the cell. Result is one (cols*k) x (rows*k) object at the origin — grid-
// aligned by construction, so ScaleDown[k] reads exactly one object per cell.
// WHERE A PIECE SITS IS ITS FRAME; WHAT IT DRAWS IS WHAT IT HOLDS. The bands
// read the frames and the cells get the painted cells, from the top-left of
// what is painted rather than of the frame. For a trimmed piece the two are
// the same box. A pane is not trimmed -- its frame is the pane -- so after
// Panes the bands are the panes' own rows and columns while each cell gets the
// figure the pane holds, wherever it sat inside it.
// Layout[0] reads the cell off the pieces instead of taking it as a number: as
// wide as the widest drawing and as tall as the tallest.
__device__ gpu_object_list& gpu_exec_layout(gpu_exec_context& ctx, const gpu_object_list& src, int k) {
    auto& res = gpu_acquire(ctx);
    int n = min((int)src.count, GPU_KERNEL_MAX_OBJECTS);
    if (n <= 0 || k < 0) return res;

    // Where each drawing starts inside its frame, and the cell that holds them.
    int8_t top_of[GPU_KERNEL_MAX_OBJECTS], left_of[GPU_KERNEL_MAX_OBJECTS];
    int cell_w = k, cell_h = k;
    for (int i = 0; i < n; ++i) {
        arc::cell_box drawing;
        GPU_DEV_FOR_EACH_PIXEL(src.objects[i], r, c, v) { (void)v; drawing.include_cell(r, c); }
        top_of[i] = (int8_t)(drawing.empty() ? 0 : drawing.row0);
        left_of[i] = (int8_t)(drawing.empty() ? 0 : drawing.col0);
        if (k == 0) {
            cell_w = max(cell_w, drawing.width());
            cell_h = max(cell_h, drawing.height());
        }
    }
    if (cell_w <= 0 || cell_h <= 0) return res;

    int8_t row_of[GPU_KERNEL_MAX_OBJECTS], col_of[GPU_KERNEL_MAX_OBJECTS];
    int rows = gpu_layout_bands(src, n, true, row_of);
    int cols = gpu_layout_bands(src, n, false, col_of);
    int nw = min(cols * cell_w, GPU_MAX_GRID);
    int nh = min(rows * cell_h, GPU_MAX_GRID);

    gpu_fixed_object& obj = res.objects[0];
    obj.x0 = 0; obj.y0 = 0;
    obj.x1 = (int16_t)(nw - 1); obj.y1 = (int16_t)(nh - 1);
    obj.width = (int16_t)nw; obj.height = (int16_t)nh;
    obj.mask.clear();

    for (int i = 0; i < n; ++i) {
        const auto& s = src.objects[i];
        const int ox = col_of[i] * cell_w, oy = row_of[i] * cell_h;
        const int top = top_of[i], left = left_of[i];
        // Shift/Union can grow the box beyond the packed mask's actual extent.
        const int height = min((int)s.height, GPU_MAX_GRID), width = min((int)s.width, GPU_MAX_GRID);
        for (int r = 0; r < cell_h && top + r < height; ++r)
            for (int c = 0; c < cell_w && left + c < width; ++c) {
                const int gr = oy + r, gc = ox + c;
                const int8_t v = s.mask[top + r][left + c];
                if (gr < nh && gc < nw && v != GPU_TRANSPARENT) obj.mask[gr][gc] = v;
            }
    }
    res.count = 1;
    return res;
}

// DeleteSelection: erase the SELECTED cells; what survives is the unselected
// remainder, re-anchored to its tightened bbox, and objects left with nothing
// are dropped. The one delete verb — "keep the c-cells" is spelled
// `SelectColor[c] InvertSelection DeleteSelection`. Survivors carry no marks
// (they were the unselected ones).
// DeleteSelection: drop the selected OBJECTS from the list, pass the rest
// through unchanged. Selection is per object now, so this is a straight
// compaction — no per-cell erase, no bbox refit. Erasing part of an object is
// not a selection job: carve the piece out (Split/Colors) and delete that
// whole object instead. Lane 0 owns the compaction (res->count); the copy is a
// whole gpu_fixed_object, cheap enough to keep single-threaded and race-free.
__device__ void gpu_exec_delete_selection_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);  // gpu_acquire zeroed count
    __syncwarp();

    if (lane == 0) {
        int obj_count = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
        for (int i = 0; i < obj_count; ++i) {
            const auto& s = src->objects[i];
            if (s.selected) continue;                     // selected -> deleted
            if (res->count >= GPU_KERNEL_MAX_OBJECTS) break;
            res->objects[res->count++] = s;               // survivor rides through
        }
    }
    __syncwarp();
}

__device__ gpu_object_list& gpu_exec_shift(gpu_exec_context& ctx, const gpu_object_list& src, int8_t dir) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Move only selected objects; the rest ride through in place. No
        // selection means Shift moves nothing — acts never fall back to "all".
        if (!obj.selected) { res.count++; continue; }
        // Its own size away, the direction `dir` names (shift_by, 8b_op_math.hpp).
        const arc::direction_step by = arc::shift_by(dir, obj.width, obj.height);
        const int dx = by.across, dy = by.down;
        obj.x0 += (int16_t)dx; obj.y0 += (int16_t)dy;
        obj.x1 += (int16_t)dx; obj.y1 += (int16_t)dy;
        res.count++;
    }
    return res;
}

// Warp-parallel transform: all 32 lanes cooperate on pixel remapping
__device__ void gpu_exec_transform_warp(gpu_exec_context& ctx, node_type type, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    // NULL-src publishes sentinel rather than burning a pool slot.
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    // All 32 lanes call the deterministic acquire — they all compute the same
    // slot index and write the same pool_head/current_objects_idx values. No sync needed.
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);

    const bool is_turn = (type == node_type::TurnRight || type == node_type::TurnLeft);
    gpu_warp_remap_picked(src, res, lane,
        [&](const gpu_fixed_object& sobj, int& nw, int& nh) {
            nw = min((int)(is_turn ? sobj.height : sobj.width), GPU_MAX_GRID);
            nh = min((int)(is_turn ? sobj.width : sobj.height), GPU_MAX_GRID);
        },
        // Dest cell (dr, dc) pulls from its source cell via the inverse map:
        // pose_source, the one the reading's replays turn a piece through.
        [&](const gpu_fixed_object& sobj, int dr, int dc) {
            const auto [sr, sc] = arc::pose_source(pose_of(type), sobj.width, sobj.height, dr, dc);
            return (int8_t)sobj.mask[sr][sc];
        });
}

__device__ gpu_object_list& gpu_exec_move(gpu_exec_context& ctx, const gpu_object_list& src, int16_t delta, bool horizontal) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }
        if (horizontal) { obj.x0 += delta; obj.x1 += delta; }
        else { obj.y0 += delta; obj.y1 += delta; }
        res.count++;
    }
    return res;
}

// Warp-parallel Move (MoveRow/MoveColumn — the single most frequent op, ~1858
// calls). The lane-0 version's cost is the per-object 470-byte struct copy done
// serially; here the 32 lanes stripe the object bytes (the mask is the bulk), then
// lane 0 tweaks the 2 shifted metadata fields. Mirrors gpu_exec_*_warp.
__device__ void gpu_exec_move_warp(gpu_exec_context& ctx, int16_t delta, bool horizontal, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    gpu_warp_copy_objects(res, src, n, lane);
    if (lane == 0) {
        for (int i = 0; i < n; ++i) {
            gpu_fixed_object& obj = res->objects[i];
            if (!obj.selected) continue;
            if (horizontal) { obj.x0 += delta; obj.x1 += delta; }
            else { obj.y0 += delta; obj.y1 += delta; }
        }
    }
    __syncwarp();
}

__device__ gpu_object_list& gpu_exec_slam(gpu_exec_context& ctx, const gpu_object_list& src,
                                        bool rows, int16_t step) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        if (obj.selected && step != 0) {
            // Shift/Union can grow the box beyond the packed mask's actual extent.
            const int width = min((int)obj.width, GPU_MAX_GRID);
            const int height = min((int)obj.height, GPU_MAX_GRID);
            // Single-threaded, so writing cells that share a byte is safe.
            auto& drawn = *reinterpret_cast<arc::cells*>(ctx.op_scratch);
            drawn.clear();
            arc::slam_cells(width, height, rows, step,
                [&](int row, int col) { return (int)obj.mask[row][col]; },
                [&](int row, int col, int color) { drawn[row][col] = (int8_t)color; });
            obj.mask = drawn;
        }
        res.count++;
    }
    return res;
}

// Warp-parallel scale up.
__device__ void gpu_exec_scale_up_warp(gpu_exec_context& ctx, int8_t factor, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    if (factor <= 1) {
        gpu_warp_copy_objects(res, src, min((int)src->count, GPU_KERNEL_MAX_OBJECTS), lane);
        return;
    }

    gpu_warp_remap_picked(src, res, lane,
        [&](const gpu_fixed_object& sobj, int& nw, int& nh) {
            nw = min(sobj.width * factor, GPU_MAX_GRID);
            nh = min(sobj.height * factor, GPU_MAX_GRID);
        },
        // Nearest-neighbour: each dest cell pulls its source via /factor.
        [&](const gpu_fixed_object& sobj, int dr, int dc) {
            const int sr = dr / factor, sc = dc / factor;
            return (sr < sobj.height && sc < sobj.width) ? (int8_t)sobj.mask[sr][sc] : (int8_t)GPU_TRANSPARENT;
        });
}

// Warp-parallel scale down: each lane handles one output pixel's majority vote.
__device__ void gpu_exec_scale_down_warp(gpu_exec_context& ctx, int8_t factor, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    if (factor <= 1) {
        gpu_warp_copy_objects(res, src, min((int)src->count, GPU_KERNEL_MAX_OBJECTS), lane);
        return;
    }

    gpu_warp_remap_picked(src, res, lane,
        [&](const gpu_fixed_object& sobj, int& nw, int& nh) {
            nw = max((sobj.width + factor - 1) / factor, 1);
            nh = max((sobj.height + factor - 1) / factor, 1);
        },
        // Each dest cell is the majority vote of its factor x factor source block.
        [&](const gpu_fixed_object& sobj, int dr, int dc) {
            return gpu_majority_vote(sobj.mask, dr, dc, factor, sobj.height, sobj.width);
        });
}


__device__ gpu_object_list& gpu_exec_add_color(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked gains the color; the rest ride through as they are.
        if (!obj.selected) { res.count++; continue; }
        // Resolve once from the original object, before filling any holes.
        // Same literal / least-common / most-common rule as Recolor.
        const int8_t fill = gpu_obj_resolve_color(src.objects[i], color);
        if (!arc::is_color(fill)) { res.count++; continue; }
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] == GPU_TRANSPARENT) obj.mask[r][c] = fill;
        res.count++;
    }
    return res;
}

// Outline: keep each object's edge cells, drop the core to transparent. A cell
// is edge when any of its 8 neighbours is not part of the object (out-of-bbox
// counts as outside; the bbox is tight, so bbox edge = object edge). Reads the
// source mask, writes the copy, so the erosion never sees its own output.
__device__ gpu_object_list& gpu_exec_outline(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& obj_src = src.objects[i];
        res.objects[res.count] = obj_src;
        auto& obj = res.objects[res.count];
        // Only what was picked is reduced to its edge; the rest ride through
        // whole. Nothing picked means this op does nothing.
        if (!obj.selected) { res.count++; continue; }
        for (int r = 0; r < obj_src.height; ++r)
            for (int c = 0; c < obj_src.width; ++c) {
                if (obj_src.mask[r][c] == GPU_TRANSPARENT) continue;
                const bool edge = arc::on_object_edge(r, c, [&](int row, int col) {
                    return row >= 0 && col >= 0 && row < obj_src.height && col < obj_src.width &&
                           obj_src.mask[row][col] != GPU_TRANSPARENT;
                });
                if (!edge) obj.mask[r][c] = GPU_TRANSPARENT;
            }
        res.count++;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_fit(gpu_exec_context& ctx, const gpu_object_list& src, int mode) {
    const auto target = arc::fit_frame(mode, ctx.input.width, ctx.input.height, ctx.output);
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& obj_src = src.objects[i];
        if (obj_src.width <= 0 || obj_src.height <= 0) continue;
        // Only what was picked is stretched to the grid; the rest ride through.
        if (!obj_src.selected) { res.objects[res.count++] = obj_src; continue; }
        int sw = obj_src.width, sh = obj_src.height;
        if (target.empty()) continue;
        int nw = target.width(), nh = target.height();
        gpu_fixed_object obj;
        // Same object, same mark -- a fresh one would start unmarked.
        obj.selected = obj_src.selected;
        obj.x0 = (int16_t)target.col0; obj.y0 = (int16_t)target.row0;
        obj.width = (int16_t)nw; obj.height = (int16_t)nh;
        obj.x1 = (int16_t)target.col1; obj.y1 = (int16_t)target.row1;
        obj.mask.clear();
        for (int r = 0; r < nh; ++r)
            for (int c = 0; c < nw; ++c)
                obj.mask[r][c] = obj_src.mask[arc::fit_source_cell(r, sh, nh)][arc::fit_source_cell(c, sw, nw)];
        res.objects[res.count++] = obj;
    }
    return res;
}

// TODO(skew): rational-slope upgrade is UNTESTED on-device. Verify before relying
// on it: Bresenham floor shift, +/- mirror symmetry, and backward-compat that
// Skew[0]=identity / Skew[1]=45deg still match old behavior.
// Skew slope table: variant -> {num, den}; the line slope is num/den (Δcol per
// Δrow), applied as a Bresenham shift floor((y0+r)*num/den). Stern-Brocot /
// complexity order (simplest slopes first). Counts: reduced |num|,|den| <= 6
// gives 23 distinct magnitudes; with both signs + the identity (0/1) that is
// 23*2 + 1 = 47 entries. This spans essentially every slope ARC draws; the full
// 30x30 lattice has ~1080 distinct slopes (Farey order 29) -- deliberately NOT
// enumerated, to keep the search branching small. Index 0 = identity and index
// 1 = +45 deg preserve the old 2-variant Skew[0]/Skew[1] meanings.
__device__ const int8_t GPU_SKEW_SLOPES[47][2] = {
    {0,1},
    {1,1},{-1,1},   {1,2},{-1,2},   {2,1},{-2,1},   {1,3},{-1,3},   {3,1},{-3,1},
    {1,4},{-1,4},   {2,3},{-2,3},   {3,2},{-3,2},   {4,1},{-4,1},   {1,5},{-1,5},
    {5,1},{-5,1},   {1,6},{-1,6},   {2,5},{-2,5},   {3,4},{-3,4},   {4,3},{-4,3},
    {5,2},{-5,2},   {6,1},{-6,1},   {3,5},{-3,5},   {5,3},{-5,3},   {4,5},{-4,5},
    {5,4},{-5,4},   {5,6},{-5,6},   {6,5},{-6,5},
};

// Floor division toward -inf (C integer division truncates toward zero). den is
// always > 0 in the slope table, so this keeps +/- slopes mirror-symmetric.
__device__ inline int gpu_skew_shift(int row, int num, int den) {
    int a = row * num;
    int q = a / den, r = a % den;
    return (r < 0) ? q - 1 : q;
}

__device__ gpu_object_list& gpu_exec_skew(gpu_exec_context& ctx, const gpu_object_list& src, int8_t variant) {
    auto& res = gpu_acquire(ctx);
    int vi = (variant >= 0 && variant < 47) ? variant : 0;
    int num = GPU_SKEW_SLOPES[vi][0], den = GPU_SKEW_SLOPES[vi][1];
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }
        int oy0 = obj.y0;
        int min_c = 1000, max_c = -1000, min_r = 1000, max_r = -1000;
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v_) {
            (void)v_;
            int nc = c + gpu_skew_shift(oy0 + r, num, den);
            if (nc < min_c) min_c = nc; if (nc > max_c) max_c = nc;
            if (r < min_r) min_r = r; if (r > max_r) max_r = r;
        }
        if (max_c < min_c) { res.count++; continue; }
        int nw = min(max_c - min_c + 1, GPU_MAX_GRID), nh = min(max_r - min_r + 1, GPU_MAX_GRID);
        // Global scratch (coalesced), not per-thread local. Packed like the mask
        // it becomes, so the result moves across as a copy, not a conversion.
        // Single-threaded here, so sharing a byte between two cells is safe.
        arc::cells& nm = *reinterpret_cast<arc::cells*>(ctx.op_scratch);
        nm.clear();
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int nr = r - min_r, nc2 = (c + gpu_skew_shift(oy0 + r, num, den)) - min_c;
            if (nr >= 0 && nr < nh && nc2 >= 0 && nc2 < nw) nm[nr][nc2] = v;
        }
        obj.mask = nm;
        obj.width = (int16_t)nw; obj.height = (int16_t)nh;
        obj.x0 = (int16_t)(obj.x0 + min_c); obj.y0 = (int16_t)(oy0 + min_r);
        obj.x1 = obj.x0 + (int16_t)nw - 1; obj.y1 = obj.y0 + (int16_t)nh - 1;
        res.count++;
    }
    return res;
}

// Lets repeat_periods read an object: it wants width, height and data[r][c],
// and a packed mask already indexes that way. No copy, no second period rule.
struct gpu_object_view {
    const arc::cells& data;
    int width, height;
};

__device__ gpu_object_list& gpu_exec_pattern(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& obj_src = src.objects[i];
        if (obj_src.width <= 0 || obj_src.height <= 0) continue;
        // Only what was picked is reduced to its unit; the rest ride through.
        if (!obj_src.selected) { res.objects[res.count++] = obj_src; continue; }
        int best_w, best_h;
        // One finder for everyone (4_objects.hpp).
        const gpu_object_view view{ obj_src.mask, obj_src.width, obj_src.height };
        arc::repeat_periods(view, best_w, best_h);
        gpu_fixed_object obj;
        // The unit is the same object, so it keeps its mark. A fresh
        // gpu_fixed_object starts unmarked and would drop the pick silently.
        obj.selected = obj_src.selected;
        obj.x0 = obj_src.x0; obj.y0 = obj_src.y0;
        obj.width = (int16_t)best_w; obj.height = (int16_t)best_h;
        obj.x1 = obj.x0 + (int16_t)best_w - 1; obj.y1 = obj.y0 + (int16_t)best_h - 1;
        obj.mask.clear();
        // Every copy agrees with the top-left one, so the unit is that copy.
        for (int r = 0; r < best_h; ++r)
            for (int c = 0; c < best_w; ++c) obj.mask[r][c] = obj_src.mask[r][c];
        res.objects[res.count++] = obj;
    }
    return res;
}

// VoteDenoise: noise-tolerant majority vote across repeating structure. The
// arithmetic is vote_denoise_cells (8b_op_math.hpp), shared with inference.
// No period on either axis -> object passes unchanged.
__device__ gpu_object_list& gpu_exec_vote_denoise(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& s = src.objects[i];
        if (s.width <= 0 || s.height <= 0) continue;
        // Only what was picked is denoised; the rest ride through as they are.
        if (!s.selected) { res.objects[res.count++] = s; continue; }
        gpu_fixed_object obj = s;
        const int w = s.width, h = s.height;

        // The ground is read off this piece alone (denoise_ground, shared with
        // the reading): empty where a deletion upstream left it empty, else the
        // color most of the piece wears. No puzzle-level background is read.
        const int8_t bg = arc::denoise_ground(w, h, [&](int r, int c) { return (int8_t)s.mask[r][c]; });

        // A cell emptied by a background removal upstream is still the ground:
        // read it as bg, which is itself empty when the ground was deleted.
        auto ground_at = [&](int r, int c) -> int8_t {
            int8_t v = s.mask[r][c];
            return v == GPU_TRANSPARENT ? bg : v;
        };

        arc::vote_denoise_cells(w, h, bg, ground_at,
                           [&](int r, int c, int8_t v) { obj.mask[r][c] = v; });
        res.objects[res.count++] = obj;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_trim(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& obj_src = src.objects[i];
        // Fitting the box to the pixels is a deliberate act (the frame law), so
        // it happens only to what was picked; the rest keep the box they had.
        if (!obj_src.selected) { res.objects[res.count++] = obj_src; continue; }
        int16_t min_r = obj_src.height, max_r = -1, min_c = obj_src.width, max_c = -1;
        bool has_pix = false;
        GPU_DEV_FOR_EACH_PIXEL(obj_src, r, c, v_) {
            (void)v_;
            if (r < min_r) min_r = (int16_t)r; if (r > max_r) max_r = (int16_t)r;
            if (c < min_c) min_c = (int16_t)c; if (c > max_c) max_c = (int16_t)c;
            has_pix = true;
        }
        if (!has_pix) continue;
        gpu_fixed_object obj;
        // A rebuilt object is the SAME object, so it keeps its mark. A fresh
        // gpu_fixed_object starts unmarked, which would drop the pick silently.
        obj.selected = obj_src.selected;
        obj.x0 = obj_src.x0 + min_c; obj.y0 = obj_src.y0 + min_r;
        obj.width = max_c - min_c + 1; obj.height = max_r - min_r + 1;
        obj.x1 = obj.x0 + obj.width - 1; obj.y1 = obj.y0 + obj.height - 1;
        obj.mask.clear();
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                obj.mask[r][c] = obj_src.mask[min_r + r][min_c + c];
        res.objects[res.count++] = obj;
    }
    return res;
}

// Warp-parallel tile. All 32 lanes cooperate; lane 0 handles scattered merged-build
// (races unsafe to parallelize), rest is striped.
__device__ void gpu_exec_tile_warp(gpu_exec_context& ctx, int8_t period, bool horizontal, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    constexpr int obj_bytes = sizeof(gpu_fixed_object);
    int src_count = src->count;

    // Compute bbox on all lanes (tiny loop).
    int16_t min_x = GPU_MAX_GRID, min_y = GPU_MAX_GRID, max_x = -1, max_y = -1;
    for (int i = 0; i < src_count; ++i) {
        const auto& o = src->objects[i];
        if (o.x0 < min_x) min_x = o.x0;
        if (o.y0 < min_y) min_y = o.y0;
        if (o.x1 > max_x) max_x = o.x1;
        if (o.y1 > max_y) max_y = o.y1;
    }
    min_x = max(min_x, (int16_t)0); min_y = max(min_y, (int16_t)0);
    max_x = min(max_x, (int16_t)(GPU_MAX_GRID - 1));
    max_y = min(max_y, (int16_t)(GPU_MAX_GRID - 1));
    int fw = max_x - min_x + 1, fh = max_y - min_y + 1;

    // Single pass-through path for all degenerate cases: period<=0, empty src, bad bbox.
    bool passthrough = (period <= 0) || (src_count == 0) || (max_x < 0) || (max_y < 0) || (fw <= 0) || (fh <= 0);
    if (passthrough) {
        int n = min(src_count, GPU_KERNEL_MAX_OBJECTS);
        for (int i = 0; i < n; ++i) {
            int8_t* dst_obj = reinterpret_cast<int8_t*>(&res->objects[i]);
            const int8_t* src_obj = reinterpret_cast<const int8_t*>(&src->objects[i]);
            for (int b = lane; b < obj_bytes; b += 32) dst_obj[b] = src_obj[b];
        }
        if (lane == 0) res->count = n;
        __syncwarp();
        return;
    }

    // Emit one tiled output object per source object. Each output spans the
    // full canvas on the tiled axis (period = `period`, counted from where the
    // source stands, arc::tiled_source), preserving the source's own bbox on
    // the fixed axis.
    int per = (int)period;
    int n_out = min(src_count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < n_out; ++i) {
        const auto& obj = src->objects[i];
        int sw = obj.width, sh = obj.height;
        int sx = obj.x0, sy = obj.y0;

        int ow = horizontal ? ctx.output.width  : sw;
        int oh = horizontal ? sh                : ctx.output.height;
        int ox0 = horizontal ? 0   : sx;
        int oy0 = horizontal ? sy  : 0;

        auto& t = res->objects[i];
        // Only what was picked is tiled; the rest ride through as they are.
        // Nothing picked means this op does nothing.
        if (!obj.selected) {
            gpu_warp_copy_object(t, obj, lane);
            __syncwarp();
            continue;
        }
        if (lane == 0) {
            t.x0 = (int16_t)ox0; t.y0 = (int16_t)oy0;
            t.width = (int16_t)ow; t.height = (int16_t)oh;
            t.x1 = (int16_t)(ox0 + ow - 1); t.y1 = (int16_t)(oy0 + oh - 1);
            // A tiled object is the SAME object, grown -- so it keeps its mark,
            // like Trim, Fit and Pattern. A rebuilt one starts unmarked.
            t.selected = obj.selected;
        }
        // Byte-gather the mask: each lane owns ONE packed dest byte = 2 dest cells
        // in the same row. Each dest cell pulls its source via the tiling period
        // (already a dest->source map). Writes every mask byte once, race-free.
        for (int bb = lane; bb < (int)sizeof(t.mask.bytes); bb += 32) {
            int idx = arc::cells::first_cell_of_byte(bb);
            int r = idx / GPU_MAX_GRID;
            int c = idx - r * GPU_MAX_GRID;              // even
            int8_t pv[2];
            #pragma unroll
            for (int t2 = 0; t2 < 2; ++t2) {
                int cc = c + t2;
                int8_t v = GPU_TRANSPARENT;
                if (r < oh && cc < ow) {
                    const int pr = horizontal ? r : arc::tiled_source(r, sy, per, sh);
                    const int pc = horizontal ? arc::tiled_source(cc, sx, per, sw) : cc;
                    if (pr >= 0 && pr < sh && pc >= 0 && pc < sw) v = obj.mask[pr][pc];
                }
                pv[t2] = v;
            }
            t.mask.bytes[bb] = arc::cells::pair_byte(pv[0], pv[1]);
        }
        __syncwarp();
    }

    if (lane == 0) res->count = n_out;
    __syncwarp();
}

__device__ gpu_object_list& gpu_exec_diag_tile(gpu_exec_context& ctx, const gpu_object_list& src, int8_t variant) {
    bool seen[GPU_MAX_COLORS];
    memset(seen, 0, sizeof(seen));
    int color_count = 0;
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int8_t col = v;
            if (arc::is_color(col) && !seen[col]) { seen[col] = true; ++color_count; }
        }
    }
    auto& res = gpu_acquire(ctx);
    if (color_count == 0) {
        for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i)
            res.objects[res.count++] = src.objects[i];
        return res;
    }
    int8_t mapping[GPU_MAX_COLORS];
    bool mapped[GPU_MAX_COLORS];
    memset(mapping, 0, sizeof(mapping)); memset(mapped, 0, sizeof(mapped));
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int8_t col = v;
            if (!arc::is_color(col)) continue;
            int wr = obj.y0 + r, wc = obj.x0 + c;
            const int idx = arc::diagonal_seat(wr, wc, variant, color_count);
            mapping[idx] = col; mapped[idx] = true;
        }
    }
    for (int i = 0; i < color_count; ++i) {
        if (!mapped[i]) {
            for (int j = 0; j < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++j)
                res.objects[res.count++] = src.objects[j];
            return res;
        }
    }
    int w = ctx.output.width, h = ctx.output.height;
    gpu_fixed_object& out_obj = res.objects[0];
    out_obj.x0 = 0; out_obj.y0 = 0;
    out_obj.x1 = (int16_t)(w - 1); out_obj.y1 = (int16_t)(h - 1);
    out_obj.width = (int16_t)w; out_obj.height = (int16_t)h;
    out_obj.mask.clear();
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            out_obj.mask[r][c] = mapping[arc::diagonal_seat(r, c, variant, color_count)];
        }
    res.count = 1;
    return res;
}

__device__ gpu_object_list& gpu_exec_fractal(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    int w = ctx.output.width, h = ctx.output.height;
    if (w <= 0 || h <= 0) return res;
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& s = src.objects[i];
        if (s.width <= 0 || s.height <= 0) continue;
        // Only what was picked is stamped into itself; the rest ride through.
        if (!s.selected) { res.objects[res.count++] = s; continue; }
        gpu_fixed_object& out = res.objects[res.count];
        // Same object grown, so it keeps its mark (a rebuilt one starts unmarked).
        out.selected = s.selected;
        out.x0 = 0; out.y0 = 0;
        out.x1 = (int16_t)(w - 1); out.y1 = (int16_t)(h - 1);
        out.width = (int16_t)w; out.height = (int16_t)h;
        out.mask.clear();
        bool any = false;
        // gr depends only on r -> hoist out of the c-loop (was recomputed w
        // times/row). rm/cm replace r%s.height / c%s.width with wrapping
        // counters (modulo is a GPU divide).
        int rm = 0;
        for (int r = 0; r < h; ++r) {
            int gr = r * s.height / h;
            int cm = 0;
            for (int c = 0; c < w; ++c) {
                int gc = c * s.width / w;
                if (s.mask[gr][gc] != GPU_TRANSPARENT) {
                    int8_t v = s.mask[rm][cm];
                    if (v != GPU_TRANSPARENT) { out.mask[r][c] = v; any = true; }
                }
                if (++cm >= s.width) cm = 0;
            }
            if (++rm >= s.height) rm = 0;
        }
        if (any) ++res.count;
    }
    return res;
}

// Tile[0]=both axes, Tile[1]=X only, Tile[2]=Y only — period driven from object size.
// Tile[3]=seed-lattice vertical: infers the periodic lattice from the marks
// themselves (spacing = smallest row-gap, color cycle = the marks in order) and
// completes it down the canvas, each mark stamped at its own row + k*period (anchor
// preserved, rows before the first mark stay blank). Generalizes the "two pixels ->
// alternating tiled lines" pattern (puzzle 0a938d79) to N marks / repeated colors;
// use after NormalizeOrientation + TileX[1] expands each mark to a full row, then
// RestoreOrientation to restore orientation. The lattice is read from the input (a
// rule, not a constant); the line-fill/orientation stay separate ops chosen by
// search — see notes/2_dsl/design_principles.txt.
__device__ gpu_object_list& gpu_exec_tile(gpu_exec_context& ctx, const gpu_object_list& src, int8_t variant) {
    auto& res = gpu_acquire(ctx);
    int w = ctx.output.width, h = ctx.output.height;
    if (w <= 0 || h <= 0) return res;

    if (variant == 3) {
        // The lattice is read off the marks themselves (arc::mark_lattice_period):
        // with fewer than two marks, or all on one row, there is none, and the
        // marks pass through unchanged.
        const int period = arc::mark_lattice_period(src.count,
            [&](int i) { return (int)src.objects[i].y0; });
        if (period <= 0) {
            for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i)
                res.objects[res.count++] = src.objects[i];
            return res;
        }
        // One full-canvas output object per mark, stamped every `period` rows from
        // its own y0, and never above it.
        for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
            const auto& s = src.objects[i];
            if (s.width <= 0 || s.height <= 0) continue;
            gpu_fixed_object& out = res.objects[res.count];
            out.x0 = 0; out.y0 = 0;
            out.x1 = (int16_t)(w - 1); out.y1 = (int16_t)(h - 1);
            out.width = (int16_t)w; out.height = (int16_t)h;
            out.mask.clear();
            bool any = false;
            arc::each_lattice_row(s.y0, s.height, period, h, [&](int r, int sr) {
                for (int c = 0; c < w && c < s.width; ++c) {
                    int8_t v = s.mask[sr][c];
                    if (v != GPU_TRANSPARENT) { out.mask[r][c] = v; any = true; }
                }
            });
            if (any) ++res.count;
        }
        return res;
    }

    // A COPY IS A PIECE. Handing back one sheet with the pattern baked into it
    // said the tiling made one thing, and then nothing downstream could speak
    // about the copies -- no pick could reach the middle row of them, because
    // there was no middle row to reach. Want the sheet? Say UnionList after.
    // Copies past the object ceiling are dropped, the same way Split[2] drops
    // cells past it: a tiny motif over a big canvas makes more pieces than the
    // kernel can hold.
    bool tile_x = (variant == 0 || variant == 1);
    bool tile_y = (variant == 0 || variant == 2);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& s = src.objects[i];
        if (s.width <= 0 || s.height <= 0) continue;
        const int across = tile_x ? (w + s.width  - 1) / s.width  : 1;
        const int down   = tile_y ? (h + s.height - 1) / s.height : 1;
        for (int down_copy = 0; down_copy < down && res.count < GPU_KERNEL_MAX_OBJECTS; ++down_copy)
            for (int across_copy = 0; across_copy < across && res.count < GPU_KERNEL_MAX_OBJECTS; ++across_copy) {
                const int top = down_copy * s.height, left = across_copy * s.width;
                // The copy is the same object moved, so it keeps its mask and
                // its mark; the combine clips whatever hangs off the canvas.
                gpu_fixed_object& out = res.objects[res.count];
                out = s;
                out.x0 = (int16_t)left;
                out.y0 = (int16_t)top;
                out.x1 = (int16_t)(left + s.width  - 1);
                out.y1 = (int16_t)(top  + s.height - 1);
                ++res.count;
            }
    }
    return res;
}

// For each object: find the center cell, draw rays in 4 directions (the
// directions from dir_first, arc::direction) for `radius` steps using the center
// cell's color. Output is one combined object per input.
__device__ gpu_object_list& gpu_exec_extend_rays(gpu_exec_context& ctx, const gpu_object_list& src, int8_t radius, int dir_first) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const gpu_fixed_object& obj = src.objects[i];
        if (obj.width <= 0 || obj.height <= 0) continue;
        // Only what was picked grows rays; the rest ride through as they are.
        if (!obj.selected) { res.objects[res.count++] = obj; continue; }

        const auto seed = arc::nearest_ray_seed(obj.height, obj.width,
            (obj.y0 + obj.y1) / 2 - obj.y0, (obj.x0 + obj.x1) / 2 - obj.x0,
            [&](int row, int col) { return obj.mask[row][col]; });
        if (seed.row < 0) continue;
        const int8_t center_color = obj.mask[seed.row][seed.col];

        // The object's box grown by radius on every side, clipped to the canvas:
        // a ray that runs past the edge would never be painted there anyway, and
        // a frame no wider than the canvas holds every ray that shows -- one
        // anchored off the object's corner cut the long ones short on its far
        // sides.
        const int x0 = max(obj.x0 - radius, 0), y0 = max(obj.y0 - radius, 0);
        const int x1 = min(obj.x1 + radius, GPU_MAX_GRID - 1), y1 = min(obj.y1 + radius, GPU_MAX_GRID - 1);
        if (x1 < x0 || y1 < y0) continue;

        gpu_fixed_object& out = res.objects[res.count++];
        // The rays belong to the object that grew them, so they carry its mark.
        out.selected = obj.selected;
        out.x0 = (int16_t)x0; out.y0 = (int16_t)y0;
        out.x1 = (int16_t)x1; out.y1 = (int16_t)y1;
        out.width = (int16_t)(x1 - x0 + 1); out.height = (int16_t)(y1 - y0 + 1);
        out.mask.clear();

        // RAYS ONLY: do NOT copy the seed/body pixels -- output is just the rays.
        // Callers that need the originals must re-stamp them (e.g. with a
        // Union). A single-pixel seed thus contributes no cell of its own, only
        // its 4 arms. (radius 0 => empty output.)
        const int seed_row = obj.y0 + seed.row - y0, seed_col = obj.x0 + seed.col - x0;
        arc::each_extended_ray(seed, radius, dir_first == 4, [&](int down, int across) {
            const int row = seed_row + down, col = seed_col + across;
            if (row >= 0 && row < out.height && col >= 0 && col < out.width) out.mask[row][col] = center_color;
        });
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_extend_orthogonal(gpu_exec_context& ctx, const gpu_object_list& src, int8_t radius) {
    return gpu_exec_extend_rays(ctx, src, radius, 0);
}

__device__ gpu_object_list& gpu_exec_extend_diagonal(gpu_exec_context& ctx, const gpu_object_list& src, int8_t radius) {
    return gpu_exec_extend_rays(ctx, src, radius, 4);
}

__device__ inline bool gpu_obj_contains(const gpu_fixed_object& o, int x, int y) {
    int lx = x - o.x0, ly = y - o.y0;
    return lx >= 0 && lx < o.width && ly >= 0 && ly < o.height
        && o.mask[ly][lx] != GPU_TRANSPARENT;
}

// ExtendTo[dir]: every object extrudes its cells along one of 8 directions,
// painting the swept path in the object's color. A ray stops just before the
// first FOREIGN occupied cell (a wall / another object) or the grid edge, and
// passes through the object's own body (already painted). Output is one object
// per input (body + trail), composing via Union. Obstacle field = the union of
// all input objects, rasterized into op_scratch. Pass 1 measures body+trail
// extent; pass 2 paints into the packed mask with identical march logic.
// Slam to the grid edge: slide each object along dir until its painted cells
// meet the canvas edge that way, ignoring paint. Shape unchanged.
// (ExtendTo grows a trail; Slam squashes inside the box; this one TRAVELS.)
// PAST THE EDGE ALREADY, IT SLIDES BACK until its far side lies on the edge,
// rather than refusing to move. A grid larger than the canvas pushed right
// shows its right side, so a crop at a corner is the size rule's box and a
// push, with no count of cells in it. Inside the canvas nothing changes: the
// slide stops at whichever edge it meets first.
__device__ gpu_object_list& gpu_exec_move_to_edge(gpu_exec_context& ctx, const gpu_object_list& src, int8_t dir) {
    auto& res = gpu_acquire(ctx);
    const arc::direction_step step = arc::step_of(dir);
    res.count = min((int)src.count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < res.count; ++i) {
        res.objects[i] = src.objects[i];
        auto& o = res.objects[i];
        // Only what was picked travels; the rest stay where they are.
        if (!o.selected) continue;
        arc::cell_box painted;
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) { (void)v; painted.include_cell(o.y0 + r, o.x0 + c); }
        if (painted.empty()) continue;
        const int steps = arc::steps_to_edge(painted, ctx.output.width, ctx.output.height, dir);
        o.x0 = (int16_t)(o.x0 + steps * step.across);
        o.x1 = (int16_t)(o.x1 + steps * step.across);
        o.y0 = (int16_t)(o.y0 + steps * step.down);
        o.y1 = (int16_t)(o.y1 + steps * step.down);
    }
    return res;
}

// ExtendTo PAST ITS EIGHT WAYS: every picked object grows in step with the
// others along its track, lines or the open cells, until they meet --
// grow_in_step, 8b_op_math.hpp, the growth the reading replays -- and hands
// back its body and every cell it took, in its first color, as a trail hands
// back its body and its trail.
__device__ gpu_object_list& gpu_exec_grow_in_step(gpu_exec_context& ctx, const gpu_object_list& src,
                                                  arc::grown_along track) {
    auto& res = gpu_acquire(ctx);
    int gw = ctx.output.width  > 0 ? (int)ctx.output.width  : (int)ctx.input.width;
    int gh = ctx.output.height > 0 ? (int)ctx.output.height : (int)ctx.input.height;
    if (gw <= 0 || gh <= 0) return res;
    gw = min(gw, GPU_MAX_GRID); gh = min(gh, GPU_MAX_GRID);

    // Each cell's object, the first in the list that holds it.
    int16_t* owner = ctx.grow_scratch[0];
    const int16_t* turn = ctx.grow_scratch[1];
    for (int at = 0; at < gw * gh; ++at) owner[at] = -1;
    const int n = min((int)src.count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < n; ++i) {
        const auto& o = src.objects[i];
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) { (void)v;
            const int x = o.x0 + c, y = o.y0 + r;
            if (x >= 0 && x < gw && y >= 0 && y < gh && owner[y * gw + x] < 0) owner[y * gw + x] = (int16_t)i;
        }
    }
    arc::grow_in_step(gw, gh, track, [&](int piece) { return src.objects[piece].selected; },
                      owner, ctx.grow_scratch[1], ctx.grow_scratch[2], ctx.grow_ways, [](int, int, int) {});

    for (int i = 0; i < n && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& o = src.objects[i];
        if (o.width <= 0 || o.height <= 0) continue;
        // Only what was picked grows; the rest ride through as they are.
        if (!o.selected) { res.objects[res.count++] = o; continue; }
        int min_x = 1 << 30, min_y = 1 << 30, max_x = -(1 << 30), max_y = -(1 << 30);
        auto reach = [&](int y, int x) {
            if (x < min_x) min_x = x; if (x > max_x) max_x = x;
            if (y < min_y) min_y = y; if (y > max_y) max_y = y;
        };
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) { (void)v; reach(o.y0 + r, o.x0 + c); }
        for (int at = 0; at < gw * gh; ++at)
            if (turn[at] > 0 && owner[at] == i) reach(at / gw, at % gw);
        if (max_x < min_x) continue;
        const int nw = min(max_x - min_x + 1, GPU_MAX_GRID);
        const int nh = min(max_y - min_y + 1, GPU_MAX_GRID);

        gpu_fixed_object& out = res.objects[res.count];
        // The grown object is the SAME object, so it keeps the mark it was
        // picked by, as a trail's does.
        out.selected = o.selected;
        out.x0 = (int16_t)min_x; out.y0 = (int16_t)min_y;
        out.x1 = (int16_t)(min_x + nw - 1); out.y1 = (int16_t)(min_y + nh - 1);
        out.width = (int16_t)nw; out.height = (int16_t)nh;
        out.mask.clear();
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) {
            const int lx = o.x0 + c - min_x, ly = o.y0 + r - min_y;
            if (lx >= 0 && lx < nw && ly >= 0 && ly < nh) out.mask[ly][lx] = v;  // body keeps its color
        }
        const int8_t color = gpu_obj_first_color(o);
        for (int at = 0; at < gw * gh; ++at) {
            if (turn[at] <= 0 || owner[at] != i) continue;
            const int lx = at % gw - min_x, ly = at / gw - min_y;
            if (lx >= 0 && lx < nw && ly >= 0 && ly < nh) out.mask[ly][lx] = color;
        }
        ++res.count;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_extend_to(gpu_exec_context& ctx, const gpu_object_list& src, int8_t dir) {
    if (dir >= arc::direction::count) return gpu_exec_grow_in_step(ctx, src, (arc::grown_along)(dir - arc::direction::count));
    auto& res = gpu_acquire(ctx);
    const arc::direction_step step = arc::step_of(dir);
    int gw = ctx.output.width  > 0 ? (int)ctx.output.width  : (int)ctx.input.width;
    int gh = ctx.output.height > 0 ? (int)ctx.output.height : (int)ctx.input.height;
    if (gw <= 0 || gh <= 0) return res;
    gw = min(gw, GPU_MAX_GRID); gh = min(gh, GPU_MAX_GRID);

    uint8_t (*occ)[GPU_MAX_GRID] = reinterpret_cast<uint8_t(*)[GPU_MAX_GRID]>(ctx.op_scratch);
    for (int y = 0; y < GPU_MAX_GRID; ++y)
        for (int x = 0; x < GPU_MAX_GRID; ++x) occ[y][x] = 0;
    for (int i = 0; i < src.count; ++i) {
        const auto& o = src.objects[i];
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) { (void)v;
            int x = o.x0 + c, y = o.y0 + r;
            if (x >= 0 && x < GPU_MAX_GRID && y >= 0 && y < GPU_MAX_GRID) occ[y][x] = 1;
        }
    }

    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& o = src.objects[i];
        if (o.width <= 0 || o.height <= 0) continue;
        // Only what was picked is touched; the rest ride through as they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!o.selected) { res.objects[res.count++] = o; continue; }
        int8_t col = gpu_obj_first_color(o);

        // The march is 8b_op_math.hpp's (march_trail), the walk the reading replays.
        auto own = [&](int y, int x) { return gpu_obj_contains(o, x, y); };
        auto taken = [&](int y, int x) { return occ[y][x] != 0; };
        int min_x = 1 << 30, min_y = 1 << 30, max_x = -(1 << 30), max_y = -(1 << 30);
        auto reach = [&](int y, int x) {
            if (x < min_x) min_x = x; if (x > max_x) max_x = x;
            if (y < min_y) min_y = y; if (y > max_y) max_y = y;
        };
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) { (void)v;
            reach(o.y0 + r, o.x0 + c);
            arc::march_trail(o.y0 + r, o.x0 + c, step, gw, gh, own, taken, reach);
        }
        if (max_x < min_x) continue;
        int nw = min(max_x - min_x + 1, GPU_MAX_GRID);
        int nh = min(max_y - min_y + 1, GPU_MAX_GRID);

        gpu_fixed_object& out = res.objects[res.count];
        // The grown object is the SAME object, longer, so the mark it was picked
        // by rides along with it -- clears_the_marks says so, and the
        // kernel is what has to agree. Built fresh out of a slot gpu_acquire has
        // just zeroed, this had to be said out loud or the next sentence found
        // nothing picked and the chain stopped speaking to one layer.
        out.selected = o.selected;
        out.x0 = (int16_t)min_x; out.y0 = (int16_t)min_y;
        out.x1 = (int16_t)(min_x + nw - 1); out.y1 = (int16_t)(min_y + nh - 1);
        out.width = (int16_t)nw; out.height = (int16_t)nh;
        out.mask.clear();

        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) {
            int sx = o.x0 + c, sy = o.y0 + r;
            int lx = sx - min_x, ly = sy - min_y;
            if (lx >= 0 && lx < nw && ly >= 0 && ly < nh) out.mask[ly][lx] = v; // body keeps its color
            arc::march_trail(sy, sx, step, gw, gh, own, taken, [&](int y, int x) {
                int px = x - min_x, py = y - min_y;
                if (px >= 0 && px < nw && py >= 0 && py < nh) out.mask[py][px] = col;
            });
        }
        ++res.count;
    }
    return res;
}

// MaskBy[c]: keep each cell of the current objects only where the live canvas (ctx.output) is
// currently color c; drop the rest. Non-destructive — it filters the object list
// and never writes the canvas, so `…layer…, MaskBy[c], Union` paints the layer
// ONLY over cells that already hold color c (the missing per-layer mask that
// canvas-wide Intersection can't express). c == GPU_COLOR_ANY (10) means "any
// non-transparent canvas cell" (pure shape mask).
// Shared core for the canvas-relative masks (constructive AND / ANDNOT).
// Predicate on the canvas cell under each layer pixel: color 0..9 -> canvas==c;
// color == GPU_COLOR_ANY (10) -> canvas non-transparent. keep_match=true keeps
// pixels where the predicate holds (MaskBy = AND); keep_match=false keeps where
// it does NOT (MaskOut = ANDNOT). Off-grid pixels see a transparent canvas, so
// MaskOut[10] keeps exactly the empty-canvas cells (constructive "fill gaps").
// Non-destructive: filters the layer, never touches the canvas.
// ReadUnder: "what's on the canvas under this?" — one output object per source
// object, same bbox; each non-transparent source cell takes the CANVAS color at
// its world position (transparent canvas / off-grid -> transparent). Canvas
// untouched. The constructive AND that keeps canvas colors — the algebra
// table's last empty cell. Replaces the destructive clip-then-Target[10]
// staging idiom, so solutions can stay additive (Union-only deposits).
// Repaints each painted cell of the picked objects as paint(v, canvas color
// under it); an object left with no paint drops out, the unpicked ride through.
template <class Paint>
__device__ inline gpu_object_list& gpu_repaint_over_canvas(gpu_exec_context& ctx, const gpu_object_list& src,
                                                           Paint paint) {
    auto& res = gpu_acquire(ctx);
    const int ow = ctx.output.width, oh = ctx.output.height;
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& o = src.objects[i];
        if (!o.selected) { res.objects[res.count++] = o; continue; }
        gpu_fixed_object out = o;
        bool any = false;
        for (int r = 0; r < o.height; ++r)
            for (int c = 0; c < o.width; ++c) {
                const int8_t v = out.mask[r][c];
                if (v == GPU_TRANSPARENT) continue;
                const int x = o.x0 + c, y = o.y0 + r;
                const int8_t cv = (x >= 0 && x < ow && y >= 0 && y < oh) ? (int8_t)ctx.output.data[y][x]
                                                                         : (int8_t)GPU_TRANSPARENT;
                const int8_t nv = paint(v, cv);
                out.mask[r][c] = nv;
                any = any || nv != GPU_TRANSPARENT;
            }
        if (any) res.objects[res.count++] = out;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_read_under(gpu_exec_context& ctx, const gpu_object_list& src) {
    return gpu_repaint_over_canvas(ctx, src, [](int8_t, int8_t cv) { return cv; });
}

__device__ gpu_object_list& gpu_exec_mask(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color, bool keep_match) {
    const bool any_color = (color == GPU_COLOR_ANY);
    return gpu_repaint_over_canvas(ctx, src, [&](int8_t v, int8_t cv) {
        return arc::mask_keeps(cv, color, any_color, keep_match) ? v : (int8_t)GPU_TRANSPARENT;
    });
}

// Target output dims from the size rule. Single source of truth, used both at
// kernel start to size `output` and by RestoreOrientation to recover the target
// orientation after a rotate has mutated `output`. Cases named by the host rule.
__device__ void gpu_rule_output_dims(int8_t rt, const int16_t* rp,
                                     int in_w, int in_h, int& ow, int& oh) {
    arc::output_dimension_rule rule;
    rule.Chosen = (decltype(rule.Chosen))rt;
    rule.width = rp[0]; rule.height = rp[1];
    rule.width_numerator = rp[2]; rule.width_denominator = rp[3];
    rule.height_numerator = rp[4]; rule.height_denominator = rp[5];
    rule.canvas_size_for(in_w, in_h, ow, oh);
}

// Warp-parallel rotate ops. Each does two transposes — the object stream and the
// working canvas (output) — but their gating differs:
//
//   NormalizeOrientation — normalize to portrait. Objects transpose iff the (immutable)
//     INPUT is landscape; the canvas is brought to portrait ("ensure portrait":
//     swap iff currently landscape). Both read live dims safely — input is never
//     mutated and ensure-portrait is idempotent.
//   RestoreOrientation   — orient back to the target. The TARGET orientation comes from
//     the size rule (gpu_rule_output_dims), NOT from `output`, which a prior
//     NormalizeOrientation may already have transposed. Objects transpose iff the target
//     is landscape; the canvas is brought to the target orientation.
//
// This stays consistent (objects and canvas same orientation after each op) across
// all 4 input/output orientation combos, with no cached snapshot.
//
// Object transpose mirrors gpu_exec_transform_warp (all-lane deterministic acquire,
// lane-striped mask copy). Canvas transpose is an in-place square transpose of
// output.data — row r owns the upper-triangle pairs (r, c>r) so no two lanes
// touch the same cell; they can still share a byte, hence gpu_paint_cell
// — plus a width<->height swap, keeping already-painted accumulator pixels
// consistent with the rotated geometry.
// ---- Canonical pose (NormalizeOrientation[1]) ----
// What the eight poses mean is 1_readings.hpp's to say, and the card reads it
// from there, so the two sides cannot drift. The card once kept its own
// spelling with flip-rows and flip-cols on the opposite bits; nothing crossed
// the boundary so nothing was wrong, and now nothing can be.
// Every example picks the pose whose row-major cell stream is lexicographically
// smallest — the whole grid votes, so rotated/mirrored pairs land identically.

// posed(r,c) reads original(pr,pc). W,H = ORIGINAL dims.
__device__ inline void gpu_pose_src(int p, int r, int c, int W, int H, int& pr, int& pc) {
    const auto [row, col] = arc::pose_source(p, W, H, r, c);
    pr = row; pc = col;
}

// original(gr,gc) lands at posed(r,c). Exact inverse of gpu_pose_src.
__device__ inline void gpu_pose_dst(int p, int gr, int gc, int W, int H, int& r, int& c) {
    const auto [row, col] = arc::posed_place(p, W, H, gr, gc);
    r = row; c = col;
}

__device__ int gpu_best_pose(const gpu_fixed_grid& g) {
    int best = 0;
    int n = g.width * g.height;
    for (int p = 1; p < 8; ++p) {
        int pw = (p & 1) ? g.height : g.width;
        int bw = (best & 1) ? g.height : g.width;
        for (int i = 0; i < n; ++i) {
            int pr, pc, br, bc;
            gpu_pose_src(p,    i / pw, i - (i / pw) * pw, g.width, g.height, pr, pc);
            gpu_pose_src(best, i / bw, i - (i / bw) * bw, g.width, g.height, br, bc);
            int8_t vp = g.data[pr][pc], vb = g.data[br][bc];
            if (vp != vb) { if (vp < vb) best = p; break; }
        }
    }
    return best;
}

// Re-pose the canvas in place (single-threaded; op_scratch as temp).
__device__ void gpu_pose_canvas(gpu_exec_context& ctx, int p, bool undo) {
    int W = ctx.output.width, H = ctx.output.height;
    int ow = undo ? ((p & 1) ? H : W) : W;   // original-frame dims
    int oh = undo ? ((p & 1) ? W : H) : H;
    int pw = (p & 1) ? oh : ow, ph = (p & 1) ? ow : oh;
    int8_t (*tmp)[GPU_MAX_GRID] = reinterpret_cast<int8_t(*)[GPU_MAX_GRID]>(ctx.op_scratch);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) tmp[r][c] = ctx.output.data[r][c];
    for (int r = 0; r < ph; ++r)
        for (int c = 0; c < pw; ++c) {
            int pr, pc;
            gpu_pose_src(p, r, c, ow, oh, pr, pc);
            if (undo) ctx.output.data[pr][pc] = tmp[r][c];
            else      ctx.output.data[r][c]   = tmp[pr][pc];
        }
    ctx.output.width  = (int16_t)(undo ? ow : pw);
    ctx.output.height = (int16_t)(undo ? oh : ph);
}

// Normalize (pick + apply best pose) or restore (undo the remembered pose).
// Objects always re-posed; the canvas only once per program (flag-gated) and
// only while its dims live in the input's frame (Identity-rule assumption).
__device__ void gpu_exec_canonical(gpu_exec_context& ctx, bool normalize) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    // pose cached in the flag while active — recompute only on first Normalize
    int p = normalize
        ? ((ctx.orientation_flipped & 8) ? (ctx.orientation_flipped & 7) : gpu_best_pose(ctx.input))
        : (ctx.orientation_flipped & 7);
    int W = ctx.input.width, H = ctx.input.height;

    gpu_object_list& res = gpu_acquire_and_publish(ctx);
    res.count = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < res.count; ++i) {
        const gpu_fixed_object& o = src->objects[i];
        gpu_fixed_object& nb = res.objects[i];
        nb = o;
        int r0, c0, r1, c1;
        if (normalize) {
            gpu_pose_dst(p, o.y0, o.x0, W, H, r0, c0);
            gpu_pose_dst(p, o.y0 + o.height - 1, o.x0 + o.width - 1, W, H, r1, c1);
        } else {
            gpu_pose_src(p, o.y0, o.x0, W, H, r0, c0);
            gpu_pose_src(p, o.y0 + o.height - 1, o.x0 + o.width - 1, W, H, r1, c1);
        }
        nb.y0 = (int16_t)min(r0, r1);
        nb.x0 = (int16_t)min(c0, c1);
        nb.width  = (p & 1) ? o.height : o.width;
        nb.height = (p & 1) ? o.width  : o.height;
        int nh = min((int)nb.height, GPU_MAX_GRID), nw = min((int)nb.width, GPU_MAX_GRID);
        for (int r = 0; r < nh; ++r)
            for (int c = 0; c < nw; ++c) nb.mask[r][c] = GPU_TRANSPARENT;
        int oh2 = min((int)o.height, GPU_MAX_GRID), ow2 = min((int)o.width, GPU_MAX_GRID);
        for (int r = 0; r < oh2; ++r)
            for (int c = 0; c < ow2; ++c) {
                int8_t v = o.mask[r][c];
                if (v == GPU_TRANSPARENT) continue;
                int nr, nc;
                if (normalize) gpu_pose_dst(p, o.y0 + r, o.x0 + c, W, H, nr, nc);
                else           gpu_pose_src(p, o.y0 + r, o.x0 + c, W, H, nr, nc);
                nr -= nb.y0; nc -= nb.x0;
                if (nr >= 0 && nr < GPU_MAX_GRID && nc >= 0 && nc < GPU_MAX_GRID)
                    nb.mask[nr][nc] = v;
            }
    }

    if (normalize) {
        if (!(ctx.orientation_flipped & 8)) gpu_pose_canvas(ctx, p, false);
        ctx.orientation_flipped = (int8_t)(8 | p);
    } else if (ctx.orientation_flipped & 8) {
        gpu_pose_canvas(ctx, p, true);
        ctx.orientation_flipped = 0;
    }
}

__device__ void gpu_exec_rotate_warp(gpu_exec_context& ctx, bool normalize, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    bool transpose_objs, transpose_canvas;
    if (normalize) {
        bool flip = arc::lies_on_its_side(ctx.input.width, ctx.input.height);
        transpose_objs   = flip;
        transpose_canvas = arc::lies_on_its_side(ctx.output.width, ctx.output.height);  // ensure portrait
        ctx.orientation_flipped = flip ? 1 : 0;  // all lanes write same value
    } else {
        // Undo exactly what NormalizeOrientation did this program (flag-keyed;
        // target dims are blind on square grids).
        bool flip = (ctx.orientation_flipped != 0);
        transpose_objs   = flip;
        transpose_canvas = flip;
        ctx.orientation_flipped = 0;
    }

    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int obj_count = src->count;

    if (!transpose_objs) {
        // Pass src objects through verbatim.
        gpu_warp_copy_objects(res, src, min(obj_count, GPU_KERNEL_MAX_OBJECTS), lane);
    } else {
        for (int i = 0; i < obj_count && i < GPU_KERNEL_MAX_OBJECTS; ++i) {
            auto& sobj = src->objects[i];
            int w = sobj.width, h = sobj.height;
            int nw = min(h, GPU_MAX_GRID);   // transposed: new width = old height
            int nh = min(w, GPU_MAX_GRID);

            auto& obj = res->objects[i];
            // Transpose: dest cell (r,c) pulls from source cell (c,r).
            gpu_warp_remap_object(sobj, obj, nw, nh, lane, [&](int dr, int dc) {
                return (int8_t)sobj.mask[dc][dr];
            });

            if (lane == 0) {
                int16_t ox0 = sobj.x0, oy0 = sobj.y0;
                obj.x0 = oy0; obj.y0 = ox0;             // swap world anchor about origin
                obj.width = (int16_t)nw; obj.height = (int16_t)nh;
                obj.x1 = (int16_t)(oy0 + nw - 1);
                obj.y1 = (int16_t)(ox0 + nh - 1);
                res->count = i + 1;
            }
            __syncwarp();
        }
    }

    if (transpose_canvas) {
        arc::cells& od = ctx.output.data;
        for (int rr = lane; rr < GPU_MAX_GRID; rr += 32)
            for (int cc = rr + 1; cc < GPU_MAX_GRID; ++cc) {
                // Each unordered pair belongs to exactly one lane, so the two
                // reads are safe; the writes are not, because a cell shares its
                // byte with a neighbour another lane is transposing.
                int8_t a = od[rr][cc], b = od[cc][rr];
                gpu_paint_cell(od, rr, cc, b);
                gpu_paint_cell(od, cc, rr, a);
            }
        __syncwarp();
        if (lane == 0) {
            int16_t t = ctx.output.width;
            ctx.output.width = ctx.output.height;
            ctx.output.height = t;
        }
        __syncwarp();
    }
}

// Copy each selected object to each unselected destination, aligning like
// anchors of their frames. 0..8 name the 3x3 anchors in row-major order;
// 4 aligns centers. The result is copies only, all selected, in source then
// destination order. Colors, holes and sizes survive; destinations contribute
// only positions and frames. Painting/recoloring remain separate operations.
__device__ gpu_object_list& gpu_exec_copy_to(gpu_exec_context& ctx, const gpu_object_list& src, int anchor) {
    auto& res = gpu_acquire(ctx);
    int picked = 0;
    for (int i = 0; i < src.count; ++i) picked += src.objects[i].selected;
    if (!picked) { gpu_copy_list(res, src); return res; }
    // Do not silently publish only a prefix of a Cartesian product.
    if (picked * (src.count - picked) > GPU_KERNEL_MAX_OBJECTS) return res;
    for (int i = 0; i < src.count; ++i) {
        const auto& pattern = src.objects[i];
        if (!pattern.selected || pattern.width <= 0 || pattern.height <= 0) continue;
        for (int j = 0; j < src.count; ++j) {
            const auto& target = src.objects[j];
            if (target.selected || target.width <= 0 || target.height <= 0) continue;
            auto& copy = res.objects[res.count++];
            copy = pattern;
            copy.x0 = (int16_t)arc::aligned_copy_start(pattern.width, target.x0, target.width, anchor % 3);
            copy.y0 = (int16_t)arc::aligned_copy_start(pattern.height, target.y0, target.height, anchor / 3);
            copy.x1 = (int16_t)(copy.x0 + copy.width - 1);
            copy.y1 = (int16_t)(copy.y0 + copy.height - 1);
        }
    }
    return res;
}

// Project the template (src.objects[0]) onto each seed in ctx.seed_objs.
// For every seed: direction = sign(seed_center - tpl_center); first stamp aligns
// edges with the seed on the far side from the template, then copies march
// outward in (dx, dy) increments until off-grid. Each stamp recolored to seed.
// Emits the original Input grid as object 0 so painter composes stamps over it.
__device__ gpu_object_list& gpu_exec_project_on_seeds(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    if (src.count <= 0 || ctx.seed_objs.count <= 0) return res;
    const gpu_fixed_object& tpl = src.objects[0];
    if (tpl.width <= 0 || tpl.height <= 0) return res;

    // Object 0: copy of Input so painter composes stamps over it (implicit overlay).
    {
        gpu_fixed_object& base = res.objects[res.count++];
        base.x0 = 0; base.y0 = 0;
        base.x1 = (int16_t)(ctx.input.width - 1);
        base.y1 = (int16_t)(ctx.input.height - 1);
        base.width = ctx.input.width; base.height = ctx.input.height;
        base.mask = ctx.input.data;
    }

    const arc::cell_box tpl_box{tpl.y0, tpl.x0, tpl.y1, tpl.x1};
    for (int s = 0; s < ctx.seed_objs.count; ++s) {
        const auto& seed = ctx.seed_objs.objects[s];
        const int8_t scolor = gpu_obj_first_color(seed);
        const int8_t paint = (scolor >= 0) ? scolor : gpu_obj_first_color(tpl);
        arc::project_along(tpl_box, {seed.y0, seed.x0, seed.y1, seed.x1},
                           ctx.output.width, ctx.output.height, [&](int row, int col) {
            if (res.count >= GPU_KERNEL_MAX_OBJECTS) return false;
            gpu_fixed_object& out = res.objects[res.count++];
            out.x0 = (int16_t)col; out.y0 = (int16_t)row;
            out.x1 = (int16_t)(col + tpl.width - 1);
            out.y1 = (int16_t)(row + tpl.height - 1);
            out.width = tpl.width; out.height = tpl.height;
            out.mask.clear();
            for (int r = 0; r < tpl.height; ++r)
                for (int c = 0; c < tpl.width; ++c)
                    if (tpl.mask[r][c] != GPU_TRANSPARENT)
                        out.mask[r][c] = paint;
            return true;
        });
    }
    return res;
}

// SnapToAnchors: rigidly stamp each "body" onto its color-coded anchor cluster.
// Where each body goes is arc::snap_to_anchors' to say (7_infer.hpp) -- the body
// color, the runs, the keys and anchors, the eight turns and the tie-break --
// and the rule that replays this op asks it the same question, so the two
// cannot set a body down differently. What stays here is only the shape the
// answer is written into: one object per body set down, drawn turned about its
// key onto the anchor that key lands on. Output = the stamped bodies only;
// originals + loose anchors are consumed.
__device__ gpu_object_list& gpu_exec_snap_to_anchors(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);

    gpu_fixed_grid temp;
    // Full-ceiling canvas (see gpu_exec_split): don't crop a grown stream to input dims.
    gpu_paint_objects(src, GPU_MAX_GRID, GPU_MAX_GRID, temp);
    const int W = temp.width, H = temp.height;
    if (W <= 0 || H <= 0) return res;

    int8_t label[GPU_MAX_GRID][GPU_MAX_GRID];
    arc::snap_to_anchors(W, H, [&](int r, int c) { return (int)temp.data[r][c]; }, label,
        [&](int run, int way, int key_row, int key_col, int to_row, int to_col) {
            if (res.count >= GPU_KERNEL_MAX_OBJECTS) return false;
            int x0 = GPU_MAX_GRID, y0 = GPU_MAX_GRID, x1 = -1, y1 = -1;
            for (int r = 0; r < H; ++r)
                for (int c = 0; c < W; ++c) {
                    if (label[r][c] != run) continue;
                    int dr, dc; arc::snap_turn(way, r - key_row, c - key_col, dr, dc);
                    const int tr = to_row + dr, tc = to_col + dc;
                    if (tr < 0 || tr >= GPU_MAX_GRID || tc < 0 || tc >= GPU_MAX_GRID) continue;
                    if (tc < x0) x0 = tc; if (tr < y0) y0 = tr;
                    if (tc > x1) x1 = tc; if (tr > y1) y1 = tr;
                }
            if (x1 < 0) return false;
            auto& obj = res.objects[res.count];
            obj.x0 = (int16_t)x0; obj.y0 = (int16_t)y0;
            obj.x1 = (int16_t)x1; obj.y1 = (int16_t)y1;
            obj.width = (int16_t)(x1 - x0 + 1); obj.height = (int16_t)(y1 - y0 + 1);
            obj.mask.clear();
            for (int r = 0; r < H; ++r)
                for (int c = 0; c < W; ++c) {
                    if (label[r][c] != run) continue;
                    int dr, dc; arc::snap_turn(way, r - key_row, c - key_col, dr, dc);
                    const int tr = to_row + dr, tc = to_col + dc;
                    if (tr < 0 || tr >= GPU_MAX_GRID || tc < 0 || tc >= GPU_MAX_GRID) continue;
                    obj.mask[tr - y0][tc - x0] = temp.data[r][c];
                }
            res.count++;
            return true;
        });
    return res;
}

// =====================================================================
// GPU DSL Interpreter
// =====================================================================

// Warp-wide interpreter: called by ALL 32 lanes.
// Most ops run on lane 0 only; warp-parallel ops use all lanes.
// ctx.current_objects_idx is kept in sync so all lanes can resolve the current result.
__device__ gpu_object_list& gpu_eval_program_warp(gpu_exec_context& ctx, const gpu_flat_op* ops, int op_count, int lane) {
    if (lane == 0) {
        ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE;
    }
    if (op_count <= 0) {
        if (lane == 0) gpu_acquire_and_publish(ctx);
        __syncwarp();
        return ctx.pool[ctx.current_objects_idx];
    }

    for (int i = 0; i < op_count; ++i) {
        // All lanes read the op (same memory, no divergence)
        node_type type = ops[i].type;
        int8_t value = ops[i].value;
        bool has_current_objects = (ctx.current_objects_idx != gpu_exec_context::CURRENT_OBJECTS_NONE);

        // The chain starts from the input: the FIRST op sees the input rather
        // than an empty list, so no program needs a leading Input. Done here,
        // before the warp-wide op branches, because they gate on has_current_objects too.
        // Only the first op. A combine does NOT end the list -- all four pass it
        // through unchanged -- so this cannot fire later in a chain.
        if (i == 0 && !has_current_objects && !is_get_op(type)) {
            if (lane == 0) { gpu_exec_input(ctx); ctx.current_objects_idx = ctx.pool_head - 1; }
            __syncwarp();
            has_current_objects = true;
        }

        // Auto-pose: while a NormalizeOrientation is active, anything the
        // previous op fetched from the input's frame joins the canonical frame
        // by itself — one Normalize per chain instead of one per fetch.
        if ((ctx.orientation_flipped & 8) && i > 0) {
            node_type prev = ops[i - 1].type;
            if (prev == node_type::Input || prev == node_type::Seeds) {
                if (lane == 0) gpu_exec_canonical(ctx, true);
                __syncwarp();
            }
        }

        // --- Warp-parallel ops: all lanes participate ---
        bool is_transform = (type == node_type::TurnRight || type == node_type::TurnLeft ||
                             type == node_type::FlipHorizontal || type == node_type::FlipVertical);

        if (is_transform && has_current_objects) {
            gpu_exec_transform_warp(ctx, type, lane);
            continue;  // transform_warp ends with __syncwarp(), safe for all lanes
        }
        if (type == node_type::ScaleUp && has_current_objects) {
            gpu_exec_scale_up_warp(ctx, value, lane);
            continue;
        }
        if (type == node_type::ScaleDown && has_current_objects) {
            gpu_exec_scale_down_warp(ctx, value, lane);
            continue;
        }
        if (type == node_type::TileX && has_current_objects) {
            gpu_exec_tile_warp(ctx, value, true, lane);
            continue;
        }
        if (type == node_type::TileY && has_current_objects) {
            gpu_exec_tile_warp(ctx, value, false, lane);
            continue;
        }
        if (type == node_type::NormalizeOrientation && has_current_objects) {
            // variant 0: by grid shape (legacy portrait). variant 1: canonical
            // pose — whole grid votes, works on square grids.
            if (value == 1) { if (lane == 0) gpu_exec_canonical(ctx, true); __syncwarp(); }
            else gpu_exec_rotate_warp(ctx, true, lane);
            continue;
        }
        if (type == node_type::RestoreOrientation && has_current_objects) {
            if (ctx.orientation_flipped & 8) { if (lane == 0) gpu_exec_canonical(ctx, false); __syncwarp(); }
            else gpu_exec_rotate_warp(ctx, false, lane);
            continue;
        }
        if (type == node_type::DeleteSelection && has_current_objects) {
            gpu_exec_delete_selection_warp(ctx, lane);
            continue;
        }
        if (type == node_type::Union && has_current_objects) {
            gpu_exec_union_warp(ctx, lane);
            continue;
        }
        if (type == node_type::Underlay && has_current_objects) {
            gpu_exec_underlay_warp(ctx, lane);
            continue;
        }
        if (type == node_type::Withhold && has_current_objects) {
            gpu_exec_withhold_warp(ctx, lane);
            continue;
        }
        if (type == node_type::InteriorColor && has_current_objects) {
            gpu_exec_interior_color_warp(ctx, value, lane);
            continue;
        }
        if (type == node_type::ShapeToInteriorColorMap && has_current_objects) {
            gpu_exec_shape_to_interior_color_map_warp(ctx, lane);
            continue;
        }
        if (type == node_type::Recolor && has_current_objects) {
            gpu_exec_recolor_warp(ctx, value, lane);
            continue;
        }
        if (type == node_type::Intersection && has_current_objects) {
            gpu_exec_intersection_warp(ctx, lane);
            continue;
        }
        if (type == node_type::MoveRow && has_current_objects) {
            gpu_exec_move_warp(ctx, value, true, lane);
            continue;
        }
        if (type == node_type::MoveColumn && has_current_objects) {
            gpu_exec_move_warp(ctx, value, false, lane);
            continue;
        }

        // --- CurrentObjects: the objects in hand are already the ones in hand,
        // so there is nothing to fetch. It exists so a sentence can name its
        // source as "the ones I have" instead of throwing them away. ---
        if (type == node_type::CurrentObjects) continue;

        // --- Seeds: no work, just update index (all lanes agree to skip) ---
        if (type == node_type::Seeds) {
            if (lane == 0) ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_SEEDS;
            __syncwarp();
            continue;
        }

        // --- All other ops: lane 0 only ---
        if (lane == 0) {
            gpu_object_list* last = gpu_current_objects(ctx);
            gpu_object_list* result = nullptr;
            switch (type) {
            case node_type::Input:          result = &gpu_exec_input(ctx); break;
            case node_type::Output:         result = &gpu_exec_output(ctx); break;
            case node_type::Split:          result = last ? &gpu_exec_split(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Colors:         result = last ? &gpu_exec_colors(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SelectMostCommonColor: result = last ? &gpu_exec_select_most_common_color(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SelectUnique: result = last ? &gpu_exec_select_unique(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectLargestObject:  result = last ? &gpu_exec_select_object_by_size(ctx, *last, value, true) : &gpu_acquire(ctx); break;
            case node_type::SelectSmallestObject: result = last ? &gpu_exec_select_object_by_size(ctx, *last, value, false) : &gpu_acquire(ctx); break;
            case node_type::SortBySize:  result = last ? &gpu_exec_sort_by_size(ctx, *last, value == 0 ? 1 : 0) : &gpu_acquire(ctx); break;
            case node_type::SizeRankColorMap: result = last ? &gpu_exec_size_rank_color_map(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Bridge:         result = last ? &gpu_exec_bridge(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Partition:      result = last ? &gpu_exec_partition(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Panes:          result = last ? &gpu_exec_panes(ctx, *last) : &gpu_acquire(ctx); break;
            // Union handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::Union:          result = &gpu_acquire(ctx); break;
            // Withhold handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::Withhold:       result = &gpu_acquire(ctx); break;
            case node_type::Difference:     result = last ? &gpu_exec_difference(ctx, *last) : &gpu_acquire(ctx); break;
            // Intersection handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::Intersection:   result = &gpu_acquire(ctx); break;
            // InteriorColor handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::InteriorColor:  result = &gpu_acquire(ctx); break;
            // Recolor handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::Recolor:        result = &gpu_acquire(ctx); break;
            case node_type::BidiColorMap:   result = last ? &gpu_exec_bidi_color_map(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::ColorFrom:      result = last ? &gpu_exec_color_from(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::WouldFit:       result = last ? &gpu_exec_would_fit(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::AlignTo:        result = last ? &gpu_exec_align_to(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::MoveToward:     result = last ? &gpu_exec_move_toward(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::MirrorAround:   result = last ? &gpu_exec_mirror_around(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectObjectsWithHoles: result = last ? &gpu_exec_select_objects_with_holes(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectAtLeastCells: result = last ? &gpu_exec_select_at_least_cells(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectColor:    result = last ? &gpu_exec_select_color(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectElectedBackground: result = last ? &gpu_exec_select_elected_background(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SelectAll:      result = last ? &gpu_exec_select_all(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::InvertSelection: result = last ? &gpu_exec_invert_selection(ctx, *last) : &gpu_acquire(ctx); break;
            // DeleteSelection handled in the warp-parallel branch above; reaching
            // here means has_current_objects was false -> empty result.
            case node_type::DeleteSelection: result = &gpu_acquire(ctx); break;
            case node_type::Shift:          result = last ? &gpu_exec_shift(ctx, *last, value) : &gpu_acquire(ctx); break;
            // MoveRow/MoveColumn handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::MoveRow:
            case node_type::MoveColumn:     result = &gpu_acquire(ctx); break;
            case node_type::SlamRow:        result = last ? &gpu_exec_slam(ctx, *last, true, value) : &gpu_acquire(ctx); break;
            case node_type::SlamColumn:     result = last ? &gpu_exec_slam(ctx, *last, false, value) : &gpu_acquire(ctx); break;
            case node_type::AddColor:       result = last ? &gpu_exec_add_color(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Fit:            result = last ? &gpu_exec_fit(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Skew:           result = last ? &gpu_exec_skew(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Trim:           result = last ? &gpu_exec_trim(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::DiagTile:       result = last ? &gpu_exec_diag_tile(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Fractal:        result = last ? &gpu_exec_fractal(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Tile:           result = last ? &gpu_exec_tile(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Pattern:        result = last ? &gpu_exec_pattern(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::ProjectOnSeeds: result = last ? &gpu_exec_project_on_seeds(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::CopyTo:         result = last ? &gpu_exec_copy_to(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::IntersectionList:  result = last ? &gpu_exec_intersect_list(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::UnionList:         result = last ? &gpu_exec_union_list(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Layout:         result = last ? &gpu_exec_layout(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Origin:         result = last ? &gpu_exec_origin(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::AlignToCanvas:  result = last ? &gpu_exec_align_to_canvas(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::OverlapCorner:  result = last ? &gpu_exec_overlap_corner(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SnapToAnchors:  result = last ? &gpu_exec_snap_to_anchors(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::ShapeColorMap:  result = last ? &gpu_exec_shape_color_map(ctx, *last, value) : &gpu_acquire(ctx); break;
            // ShapeToInteriorColorMap is handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::ShapeToInteriorColorMap: result = &gpu_acquire(ctx); break;
            case node_type::ExtendOrthogonal:   result = last ? &gpu_exec_extend_orthogonal(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::ExtendDiagonal:     result = last ? &gpu_exec_extend_diagonal(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::ExtendTo:           result = last ? &gpu_exec_extend_to(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::MaskBy:             result = last ? &gpu_exec_mask(ctx, *last, value, true) : &gpu_acquire(ctx); break;
            case node_type::MaskOut:            result = last ? &gpu_exec_mask(ctx, *last, value, false) : &gpu_acquire(ctx); break;
            case node_type::ReadUnder:          result = last ? &gpu_exec_read_under(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::MoveToEdge:          result = last ? &gpu_exec_move_to_edge(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::VoteDenoise:         result = last ? &gpu_exec_vote_denoise(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Outline:             result = last ? &gpu_exec_outline(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SelectRow:      result = last ? &gpu_exec_select_band(ctx, *last, (int8_t)value, true) : &gpu_acquire(ctx); break;
            case node_type::SelectColumn:   result = last ? &gpu_exec_select_band(ctx, *last, (int8_t)value, false) : &gpu_acquire(ctx); break;
            case node_type::SelectRowFromMiddle:    result = last ? &gpu_exec_select_from_middle(ctx, *last, (int8_t)value, true) : &gpu_acquire(ctx); break;
            case node_type::SelectColumnFromMiddle: result = last ? &gpu_exec_select_from_middle(ctx, *last, (int8_t)value, false) : &gpu_acquire(ctx); break;
            case node_type::SelectPartition: result = last ? &gpu_exec_select_partition(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectLikeKey:  result = last ? &gpu_exec_select_like_key(ctx, *last, (int8_t)value) : &gpu_acquire(ctx); break;
            case node_type::SelectPartner:  result = last ? &gpu_exec_select_partner(ctx, *last, (int8_t)value) : &gpu_acquire(ctx); break;
            // NormalizeOrientation / RestoreOrientation are handled in the warp-parallel
            // branch above; reaching here means has_current_objects was false -> empty result.
            case node_type::NormalizeOrientation:
            case node_type::RestoreOrientation: result = &gpu_acquire(ctx); break;
            default: result = &gpu_acquire(ctx); break;
            }
            // ONE HOME FOR THE MARK. Most ops build their objects fresh, out of a
            // slot gpu_acquire has just zeroed, so a mark handed in would die
            // here unless every op remembered to carry it -- and five of them did
            // not. Now that a paint respects the mark, forgetting means printing
            // nothing, which is a silent wrong answer rather than a loud one.
            // Carried once, here, so a new op cannot forget: what came in picked
            // comes out picked. Positional when the op handed back as many
            // objects as it took, all-or-nothing otherwise, since a carve's
            // pieces have no one parent to ask.
            // Only when the op marked NOTHING itself -- an op that has an opinion
            // about which of its objects are picked keeps it. Selectors are
            // excluded by role (marking IS their work), and the two ops that
            // clear do so for reasons of their own.
            // Fit carries marks itself, even when an empty target drops every picked object.
            if (result && last && result != last && is_act_op(type) &&
                type != node_type::Fit && !clears_the_marks(type)) {
                bool came_in_picked = false, went_out_picked = false;
                for (int i = 0; i < last->count; ++i)
                    came_in_picked |= last->objects[i].selected;
                for (int i = 0; i < result->count; ++i)
                    went_out_picked |= result->objects[i].selected;
                if (came_in_picked && !went_out_picked) {
                    const bool one_for_one = (result->count == last->count);
                    for (int i = 0; i < result->count; ++i)
                        result->objects[i].selected =
                            one_for_one ? last->objects[i].selected : true;
                }
            }
            // Publish the result location for other lanes
            ctx.current_objects_idx = (int)(result - ctx.pool);
        }
        __syncwarp();
    }

    gpu_object_list* final_result = gpu_current_objects(ctx);
    if (!final_result) {
        if (lane == 0) gpu_acquire_and_publish(ctx);
        __syncwarp();
        return ctx.pool[ctx.current_objects_idx];
    }
    return *final_result;
}

// =====================================================================
// Host interface for combined exec+score
// =====================================================================

// --- Cached per-example device data (uploaded once per puzzle) ---
static constexpr int MAX_CACHED_EXAMPLES = 16;

// MSVC C4324: intentional padding; see arc::cells in 0_core.hpp.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
struct gpu_example_template {
    gpu_fixed_grid input;
    gpu_object_list seed_objs;
    int8_t pane_boxes[GPU_MAX_PANES][4];
    int8_t pane_count;
    int8_t output_colors_by_size[GPU_MAX_COLORS];
    int8_t output_colors_by_size_count;
    arc::shape_table shape_colors;
    int8_t distance_maps[GPU_MAX_COLORS][GPU_MAX_GRID][GPU_MAX_GRID];
    arc::cells target_grid;
    int8_t target_dims[2];  // w, h
    // Precomputed target CC summaries (color, w, h, count) — host-built so the
    // scoring kernel only segments the attempt grid, not the target.
    gpu_object_summary target_objs[GPU_MAX_TARGET_OBJECTS];
    int8_t target_obj_count;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// MULTISET MATCH, one home. Each attempt summary claims at most one target
// summary by (w,h,count). Position bonus is a continuous
// (1 - chebyshev/target_size) per matched pair, normalized by the target's own
// footprint rather than the grid. The canvas asks this about its components and
// the hand asks it about its objects, so the two are judged by one rule.
__device__ inline void gpu_match_summaries(const gpu_object_summary* attempt, int attempt_count,
                                           const gpu_example_template& tmpl,
                                           int& shape_matches,
                                           float& position_score) {
    bool used[GPU_MAX_TARGET_OBJECTS];
    for (int j = 0; j < GPU_MAX_TARGET_OBJECTS; ++j) used[j] = false;
    for (int i = 0; i < attempt_count; ++i) {
        int best_j = -1;
        for (int j = 0; j < tmpl.target_obj_count; ++j) {
            if (used[j]) continue;
            if (attempt[i].w != tmpl.target_objs[j].w) continue;
            if (attempt[i].h != tmpl.target_objs[j].h) continue;
            if (attempt[i].count != tmpl.target_objs[j].count) continue;
            best_j = j;
            break;
        }
        if (best_j < 0) continue;
        used[best_j] = true;
        shape_matches++;
        int acx = attempt[i].x0 * 2 + (attempt[i].w - 1);
        int acy = attempt[i].y0 * 2 + (attempt[i].h - 1);
        int tcx = tmpl.target_objs[best_j].x0 * 2 + (tmpl.target_objs[best_j].w - 1);
        int tcy = tmpl.target_objs[best_j].y0 * 2 + (tmpl.target_objs[best_j].h - 1);
        int dx = acx > tcx ? acx - tcx : tcx - acx;
        int dy = acy > tcy ? acy - tcy : tcy - acy;
        int cheb = dx > dy ? dx : dy;
        int tsize = tmpl.target_objs[best_j].w > tmpl.target_objs[best_j].h
                    ? tmpl.target_objs[best_j].w : tmpl.target_objs[best_j].h;
        if (tsize < 1) tsize = 1;
        float ps = 1.0f - (float)cheb / (float)(tsize * 2);
        if (ps < 0.0f) ps = 0.0f;
        position_score += ps;
    }
}

static gpu_example_template* d_example_templates = nullptr;
static bool d_examples_uploaded = false;

// Scratch slot: separate device storage for 1-off bridge calls (execute_flat_program).
// Kept isolated from the main cache so mid-beam calls don't evict training data.
static gpu_example_template* d_scratch_templates = nullptr;
static int8_t*  d_scratch_rule_type = nullptr;
static int16_t* d_scratch_rule_params = nullptr;
static gpu_exec_context* d_scratch_contexts = nullptr;
static gpu_flat_op*      d_scratch_programs = nullptr;
static int16_t*          d_scratch_prog_lengths = nullptr;
static int16_t*          d_scratch_node_counts = nullptr;
static float*            d_scratch_scores = nullptr;
static int8_t*           d_scratch_solved = nullptr;
static int8_t*           d_scratch_output_grids = nullptr;
static int8_t*           d_scratch_output_dims = nullptr;
static uint64_t*         d_scratch_output_hashes = nullptr;
static gpu_score_terms*  d_scratch_score_terms = nullptr;
static int               d_scratch_capacity = 0;
// Rule is per-puzzle, cache on device
static int8_t*  d_cached_rule_type = nullptr;
static int16_t* d_cached_rule_params = nullptr;

// --- Two streams for pipelining: launch batch N while waiting on batch N-1 ---
static cudaStream_t exec_stream[2] = {nullptr, nullptr};
static inline cudaStream_t get_exec_stream(int slot) {
    if (!exec_stream[slot]) cudaStreamCreate(&exec_stream[slot]);
    return exec_stream[slot];
}

// --- Per-batch device buffers (two slots for ping-pong) ---
static gpu_exec_context* d_exec_contexts[2] = {nullptr, nullptr};
static gpu_flat_op*      d_exec_programs[2] = {nullptr, nullptr};
static int16_t*          d_exec_prog_lengths[2] = {nullptr, nullptr};
static int16_t*          d_exec_node_counts[2] = {nullptr, nullptr};
static float*            d_exec_scores[2] = {nullptr, nullptr};
static int8_t*           d_exec_solved[2] = {nullptr, nullptr};
static int8_t*           d_exec_output_grids[2] = {nullptr, nullptr};
static int8_t*           d_exec_output_dims[2] = {nullptr, nullptr};
static uint64_t*         d_exec_output_hashes[2] = {nullptr, nullptr};
static gpu_score_terms*  d_exec_score_terms[2] = {nullptr, nullptr};
static int               d_exec_capacity[2] = {0, 0};

static void resize_exec_buffers(int job_count, int& capacity,
                                gpu_exec_context*& ctxs, gpu_flat_op*& progs,
                                int16_t*& prog_lens, int16_t*& node_cnts,
                                float*& scores, int8_t*& solved,
                                int8_t*& out_grids, int8_t*& out_dims,
                                uint64_t*& out_hashes, gpu_score_terms*& out_terms) {
    if (job_count <= capacity) return;

    cudaFree(ctxs);
    cudaFree(progs);
    cudaFree(prog_lens);
    cudaFree(node_cnts);
    cudaFree(scores);
    cudaFree(solved);
    cudaFree(out_grids);
    cudaFree(out_dims);
    cudaFree(out_hashes);
    cudaFree(out_terms);

    capacity = job_count;
    // Check every cudaMalloc. An unchecked device-OOM here leaves null device
    // pointers AND a sticky CUDA error on the context, which then surfaces as a
    // baffling cudaMallocHost abort on ANOTHER thread (e.g. the GUI scratch eval).
    // Fail loud at the actual source with the size that didn't fit.
    cudaError_t e = cudaSuccess;
    auto chk = [&](cudaError_t r, const char* what, size_t bytes) {
        if (r != cudaSuccess && e == cudaSuccess) {
            e = r;
            fprintf(stderr, "[FATAL] resize_exec_buffers: cudaMalloc(%s) failed for "
                    "%zu bytes at job_count=%d: %s\n", what, bytes, job_count,
                    cudaGetErrorString(r));
            fflush(stderr);
        }
    };
    chk(cudaMalloc(&ctxs,      (size_t)job_count * sizeof(gpu_exec_context)), "contexts", (size_t)job_count * sizeof(gpu_exec_context));
    chk(cudaMalloc(&progs,     (size_t)job_count * GPU_MAX_OPS * sizeof(gpu_flat_op)), "programs", (size_t)job_count * GPU_MAX_OPS * sizeof(gpu_flat_op));
    chk(cudaMalloc(&prog_lens, (size_t)job_count * sizeof(int16_t)), "prog_lens", (size_t)job_count * sizeof(int16_t));
    chk(cudaMalloc(&node_cnts, (size_t)job_count * sizeof(int16_t)), "node_cnts", (size_t)job_count * sizeof(int16_t));
    chk(cudaMalloc(&scores,    (size_t)job_count * sizeof(float)), "scores", (size_t)job_count * sizeof(float));
    chk(cudaMalloc(&solved,    (size_t)job_count * sizeof(int8_t)), "solved", (size_t)job_count);
    chk(cudaMalloc(&out_grids, (size_t)job_count * GPU_GRID_CELLS * sizeof(int8_t)), "out_grids", (size_t)job_count * GPU_GRID_CELLS);
    chk(cudaMalloc(&out_dims,  (size_t)job_count * 2 * sizeof(int8_t)), "out_dims", (size_t)job_count * 2);
    chk(cudaMalloc(&out_hashes, (size_t)job_count * sizeof(uint64_t)), "out_hashes", (size_t)job_count * sizeof(uint64_t));
    chk(cudaMalloc(&out_terms, (size_t)job_count * sizeof(gpu_score_terms)), "out_terms", (size_t)job_count * sizeof(gpu_score_terms));
    if (e != cudaSuccess) std::abort();
}

static void ensure_exec_capacity(int job_count, int slot) {
    resize_exec_buffers(job_count, d_exec_capacity[slot],
                        d_exec_contexts[slot], d_exec_programs[slot],
                        d_exec_prog_lengths[slot], d_exec_node_counts[slot],
                        d_exec_scores[slot], d_exec_solved[slot],
                        d_exec_output_grids[slot], d_exec_output_dims[slot],
                        d_exec_output_hashes[slot], d_exec_score_terms[slot]);
}

// Build host-side templates from cuda_dsl_example_data. Shared by main + scratch paths.
static void build_example_templates(const cuda_dsl_example_data& ex, int count,
                                    std::vector<gpu_example_template>& h_templates) {
    // Made where they lie: one template holds a list as long as a grid has
    // cells, too big to stand on the stack as the value assign would copy.
    h_templates.clear();
    h_templates.resize(count);
    for (int ei = 0; ei < count; ++ei) {
        auto& tmpl = h_templates[ei];

        tmpl.input.width = ex.input_dims[ei * 2 + 0];
        tmpl.input.height = ex.input_dims[ei * 2 + 1];
        tmpl.input.data = ex.input_grids[ei];

        tmpl.target_grid = ex.target_grids[ei];
        tmpl.target_dims[0] = ex.target_dims[ei * 2 + 0];
        tmpl.target_dims[1] = ex.target_dims[ei * 2 + 1];

        const int8_t* dm = ex.distance_maps + ei * GPU_MAX_COLORS * GPU_GRID_CELLS;
        memcpy(tmpl.distance_maps, dm, GPU_MAX_COLORS * GPU_GRID_CELLS);

        tmpl.seed_objs = ex.seed_lists[ei];
        memcpy(tmpl.pane_boxes, ex.pane_boxes[ei], sizeof(tmpl.pane_boxes));
        tmpl.pane_count = ex.pane_counts[ei];
        memcpy(tmpl.output_colors_by_size, ex.output_colors_by_size,
               sizeof(tmpl.output_colors_by_size));
        tmpl.output_colors_by_size_count = ex.output_colors_by_size_count;
        tmpl.shape_colors = ex.shape_colors;

        memcpy(tmpl.target_objs,
               ex.target_objs + ei * GPU_MAX_TARGET_OBJECTS,
               sizeof(gpu_object_summary) * GPU_MAX_TARGET_OBJECTS);
        tmpl.target_obj_count = ex.target_obj_counts[ei];
    }
}

// Upload per-example data to device. Call once per puzzle (or when puzzle changes).
void cuda_dsl_upload_examples(const cuda_dsl_example_data& ex) {
    if (!d_example_templates) {
        cudaMalloc(&d_example_templates, MAX_CACHED_EXAMPLES * sizeof(gpu_example_template));
        cudaMalloc(&d_cached_rule_type,   sizeof(int8_t));
        cudaMalloc(&d_cached_rule_params, 6 * sizeof(int16_t));
    }

    int count = std::min(ex.count, MAX_CACHED_EXAMPLES);
    std::vector<gpu_example_template> h_templates;
    build_example_templates(ex, count, h_templates);

    cudaMemcpy(d_example_templates, h_templates.data(), count * sizeof(gpu_example_template), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cached_rule_type,   &ex.rule_type,  sizeof(int8_t),         cudaMemcpyHostToDevice);
    cudaMemcpy(d_cached_rule_params, ex.rule_params, 6 * sizeof(int16_t),    cudaMemcpyHostToDevice);

    d_examples_uploaded = true;
}

// Upload a single example into the scratch slot. Does NOT touch the main cache.
void cuda_dsl_upload_scratch_example(const cuda_dsl_example_data& ex) {
    if (!d_scratch_templates) {
        cudaMalloc(&d_scratch_templates,   MAX_CACHED_EXAMPLES * sizeof(gpu_example_template));
        cudaMalloc(&d_scratch_rule_type,   sizeof(int8_t));
        cudaMalloc(&d_scratch_rule_params, 6 * sizeof(int16_t));
    }

    int count = std::min(ex.count, MAX_CACHED_EXAMPLES);
    std::vector<gpu_example_template> h_templates;
    build_example_templates(ex, count, h_templates);

    cudaMemcpy(d_scratch_templates,   h_templates.data(), count * sizeof(gpu_example_template), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scratch_rule_type,   &ex.rule_type,  sizeof(int8_t),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_scratch_rule_params, ex.rule_params, 6 * sizeof(int16_t), cudaMemcpyHostToDevice);
}

void cuda_dsl_invalidate_examples() {
    d_examples_uploaded = false;
}

bool cuda_dsl_examples_uploaded() {
    return d_examples_uploaded;
}

// New kernel that reads from cached example templates.
// Occupancy cap (2026-06-04): the kernel otherwise uses 255 regs (8 blocks/SM, 16.7%)
// because gpu_eval_program_warp inlines the union of all DSL ops. Once the grid was
// full (chunk accumulation), occupancy became the bottleneck (NCU: ~78% SM idle, "too
// few warps"). Measured A/B at TARGET_JOBS=2048: (32,12)=243k aps, (32,16)=343k,
// (32,21)=383k (+58%), (32,24)=374k (plateau — 80 regs + 4208B smem cap blocks at ~23).
// So (32,21) ≈ 80 regs / ~44% occupancy is the peak. Those runs sent register spills
// to shared memory with enable_smem_spilling, gone with --Ofast-compile (CMakeLists.txt).
__global__ void __launch_bounds__(32, 21) dsl_exec_and_score_kernel_v2(
    gpu_exec_context* __restrict__ contexts,
    const gpu_example_template* __restrict__ templates,
    const gpu_flat_op* __restrict__ programs,
    const int16_t* __restrict__ program_lengths,
    const int16_t* __restrict__ node_counts,
    const int8_t* __restrict__ rule_type,     // single value
    const int16_t* __restrict__ rule_params,  // 6 values
    float* __restrict__ scores,
    int8_t* __restrict__ solved_out,
    int8_t* __restrict__ output_grids,
    int8_t* __restrict__ output_dims,
    int active_examples,
    int job_count)
{
    // Warp-cooperative: one block == one warp == one job. Most DSL ops run warp-parallel
    // in gpu_eval_program_warp; paint/crop stay on lane 0; write-back + score run on
    // all 32 lanes with a warp-shuffle reduction.
    int idx = blockIdx.x;
    if (idx >= job_count) return;
    int lane = threadIdx.x;

    int ei = idx % active_examples;  // example index
    gpu_exec_context& ctx = contexts[idx];
    const gpu_example_template& tmpl = templates[ei];

    // --- Parallel context init: all 32 lanes cooperatively copy bulk data ---
    {
        // Copy only the USED objects of each read-only source list (count*obj
        // bytes) instead of the full 57 KB / 64 slots — see gpu_copy_used_objs.
        // This skips the bulk of the old init copy (the measured per-line
        // hotspot) since these lists usually hold only a few objects.
        gpu_copy_used_objs(tmpl.seed_objs,          ctx.seed_objs,          lane);

        // Copy output_colors_by_size (one byte a color, plus the count) — small,
        // lane 0 only.
        if (lane == 0) {
            for (int i = 0; i < GPU_MAX_COLORS; ++i)
                ctx.output_colors_by_size[i] = tmpl.output_colors_by_size[i];
            ctx.output_colors_by_size_count = tmpl.output_colors_by_size_count;
            // Learned shape->color lookup for ShapeColorMap (puzzle-level).
            ctx.shape_colors = tmpl.shape_colors;
            // Where this example's panes lie, for the Panes act.
            ctx.pane_count = tmpl.pane_count;
            for (int i = 0; i < GPU_MAX_PANES; ++i)
                for (int k = 0; k < 4; ++k)
                    ctx.pane_boxes[i][k] = tmpl.pane_boxes[i][k];
        }

        // Stripe-copy distance_maps (9 KB)
        const int8_t* src_dm = reinterpret_cast<const int8_t*>(tmpl.distance_maps);
        int8_t* dst_dm = reinterpret_cast<int8_t*>(ctx.distance_maps);
        constexpr int dm_bytes = sizeof(ctx.distance_maps);
        for (int i = lane; i < dm_bytes; i += 32)
            dst_dm[i] = src_dm[i];

        // Stripe-copy the input grid, whole bytes per lane
        const int8_t* src_in = reinterpret_cast<const int8_t*>(&tmpl.input);
        int8_t* dst_in = reinterpret_cast<int8_t*>(&ctx.input);
        constexpr int in_bytes = sizeof(gpu_fixed_grid);
        for (int i = lane; i < in_bytes; i += 32)
            dst_in[i] = src_in[i];

        // Stripe-clear the canvas. Whole packed bytes per lane, so no two lanes
        // share one, and the empty byte is two encoded TRANSPARENT nibbles.
        uint8_t* dst_out = ctx.output.data.bytes;
        for (int i = lane; i < (int)sizeof(ctx.output.data.bytes); i += 32)
            dst_out[i] = arc::cells::empty_byte();
    }

    // Lane 0 sets scalar fields + output dims (must complete before eval)
    if (lane == 0) {
        ctx.pool_head = 0;
        ctx.orientation_flipped = 0;

        int8_t rt = *rule_type;
        int ow, oh;
        gpu_rule_output_dims(rt, rule_params, ctx.input.width, ctx.input.height, ow, oh);
        ctx.output.width  = (int16_t)ow;
        ctx.output.height = (int16_t)oh;

        // Stash the rule so RestoreOrientation can recover the target orientation even
        // after a NormalizeOrientation has transposed ctx.output (see gpu_exec_rotate_warp).
        ctx.rule_type = rt;
        for (int k = 0; k < 6; ++k) ctx.rule_params[k] = rule_params[k];
    }
    // All lanes must finish bulk copies before lane 0 starts interpreter
    __syncwarp();

    // Execute program — warp-wide: all lanes participate, transform uses all 32
    {
        const gpu_flat_op* my_ops = programs + idx * GPU_MAX_OPS;
        int op_count = program_lengths[idx];
        gpu_eval_program_warp(ctx, my_ops, op_count, lane);
    }

    // 3D-printer model: ctx.output is the accumulator. Ops are the only writers;
    // there is no post-hoc paint of the final object list. Undeposited chains
    // get their beam gradient from the EXPLICIT Underlay twin the generator
    // emits alongside them (add_candidate_if_new) — same visibility, no
    // implicit semantics, no phantom locks.
    if (lane == 0) {
        int8_t rt = *rule_type;
        if (rt == arc::output_dimension_rule::BoundingBox) {
            // Crop to non-transparent bbox of the painted canvas.
            int16_t r0 = GPU_MAX_GRID, c0 = GPU_MAX_GRID, r1 = -1, c1 = -1;
            for (int r = 0; r < ctx.output.height; ++r) {
                for (int c = 0; c < ctx.output.width; ++c) {
                    if (ctx.output.data[r][c] != GPU_TRANSPARENT) {
                        if (r < r0) r0 = (int16_t)r;
                        if (c < c0) c0 = (int16_t)c;
                        if (r > r1) r1 = (int16_t)r;
                        if (c > c1) c1 = (int16_t)c;
                    }
                }
            }
            if (r1 >= 0) {
                int new_w = c1 - c0 + 1, new_h = r1 - r0 + 1;
                if (r0 > 0 || c0 > 0) {
                    // In-place crop — no local tmp (those are per-thread interleaved =
                    // uncoalesced L1TEX local loads, NCU's top stall). Forward copy is
                    // safe: src (r0+r, c0+c) is always >= dest (r, c) in row-major order,
                    // so it's read before any later write can overwrite it.
                    for (int r = 0; r < new_h; ++r)
                        for (int c = 0; c < new_w; ++c)
                            ctx.output.data[r][c] = ctx.output.data[r0 + r][c0 + c];
                    // Clear cells outside the cropped region (write-back + scorer read
                    // the full 30x30 grid regardless of new_w/new_h).
                    for (int r = 0; r < GPU_MAX_GRID; ++r)
                        for (int c = 0; c < GPU_MAX_GRID; ++c)
                            if (r >= new_h || c >= new_w) ctx.output.data[r][c] = GPU_TRANSPARENT;
                }
                ctx.output.width = (int16_t)new_w;
                ctx.output.height = (int16_t)new_h;
            }
        }
    }
    __syncwarp();

    // Parallel output write-back: 32 lanes stripe 900 cells.
    int8_t* my_out = output_grids + idx * GPU_GRID_CELLS;
    constexpr int kCells = GPU_MAX_GRID * GPU_MAX_GRID;
    for (int i = lane; i < kCells; i += 32) {
        int r = i / GPU_MAX_GRID;
        int c = i - r * GPU_MAX_GRID;
        my_out[i] = ctx.output.data[r][c];
    }
    // EXEC kernel ends here. ctx.output (in contexts[idx]) holds the produced
    // canvas; the separate dsl_score_kernel reads it back to score. The split
    // lets the scorer run without the op-switch's 80-reg/4.2KB-smem weight, so
    // it can reach high occupancy to hide the dependent-load (long_scoreboard)
    // latency that walls the fused kernel.
}

// SCORE kernel (split from exec): one warp = one job, reads the canvas the exec
// kernel left in contexts[idx].output and produces scores. No 47-op switch here,
// so ptxas needs far fewer registers -> higher occupancy -> latency hiding.
// 24, not 32: 24 blocks per SM is the ceiling on sm_89, so 32 was refused
// outright and the kernel ran with no occupancy request at all. It never
// needed one -- 78 registers x 32 threads leaves room for 26 blocks, past
// the ceiling already -- so this only stops asking for the impossible.
// Measured on 2026-09-05 over the first two puzzles, three runs each:
// 6.678s before, 6.687s after, identical codegen at 78 registers.
__global__ void __launch_bounds__(32, 24) dsl_score_kernel(
    gpu_exec_context* __restrict__ contexts,
    const gpu_example_template* __restrict__ templates,
    const int16_t* __restrict__ node_counts,
    float* __restrict__ scores,
    int8_t* __restrict__ solved_out,
    int8_t* __restrict__ output_dims,
    uint64_t* __restrict__ output_hashes,
    gpu_score_terms* __restrict__ score_terms,
    int active_examples,
    int job_count)
{
    int idx = blockIdx.x;
    if (idx >= job_count) return;
    int lane = threadIdx.x;
    int ei = idx % active_examples;
    gpu_exec_context& ctx = contexts[idx];
    const gpu_example_template& tmpl = templates[ei];

    // Parallel score loop: each lane accumulates private partials over a
    // stride-32 slice of the target rectangle, then we reduce across the warp.
    int tw = tmpl.target_dims[0];
    int th = tmpl.target_dims[1];
    int aw = ctx.output.width, ah = ctx.output.height;
    int nc = node_counts[idx];
    int total_target_pixels = tw * th;

    int my_mismatches = 0;
    float my_exact_err = 0.0f;
    float my_dist_err = 0.0f;
    int my_target_hist[GPU_MAX_COLORS]  = {0};
    int my_attempt_hist[GPU_MAX_COLORS] = {0};
    int my_match_by_color[GPU_MAX_COLORS] = {0};
    int my_vandal = 0;
    // vandalism = wrong paint no later layer can repair: on a KEPT cell
    // (input == target, only meaningful same-dims) or on a DONE color's cell.
    const bool same_dims_as_input = (ctx.input.width == tw && ctx.input.height == th);

    for (int i = lane; i < total_target_pixels; i += 32) {
        int r = i / tw;
        int c = i - r * tw;
        int t = tmpl.target_grid[r][c];
        int a = ctx.output.data[r][c];
        if (t != a) {
            my_mismatches++;
            if (a != GPU_TRANSPARENT) {
                my_exact_err += GPU_SCORE_W_EXACT_MATCH;
                // A kept cell is one the answer leaves alone, so painting it
                // wrong is damage nothing later repairs.
                if (same_dims_as_input && ctx.input.data[r][c] == t) my_vandal++;
            }
        } else if (arc::is_color(t)) {
            my_match_by_color[t]++;
        }
        if (arc::is_color(t)) my_target_hist[t]++;
        if (arc::is_color(a)) {
            my_attempt_hist[a]++;
            my_dist_err += (float)ctx.distance_maps[a][r][c];
        }
    }

    // Warp butterfly reduction: after this, lane 0 holds the warp-wide sums.
    constexpr unsigned kAll = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_mismatches += __shfl_xor_sync(kAll, my_mismatches, off);
        my_exact_err  += __shfl_xor_sync(kAll, my_exact_err,  off);
        my_dist_err   += __shfl_xor_sync(kAll, my_dist_err,   off);
        my_vandal     += __shfl_xor_sync(kAll, my_vandal,     off);
        #pragma unroll
        for (int k = 0; k < GPU_MAX_COLORS; ++k) {
            my_target_hist[k]     += __shfl_xor_sync(kAll, my_target_hist[k],     off);
            my_attempt_hist[k]    += __shfl_xor_sync(kAll, my_attempt_hist[k],    off);
            my_match_by_color[k]  += __shfl_xor_sync(kAll, my_match_by_color[k],  off);
        }
    }

    // ---- Warp-parallel connected-component labeling (8-connected, per color) ----
    // Replaces the old serial lane-0 BFS. Monotonic min-label propagation: every
    // non-background cell starts with its own linear index as label; each sweep,
    // a cell adopts the min label among its same-color 8-neighbours; repeat until
    // a sweep changes nothing. The component representative is therefore its min
    // linear index = its top-left (row-major-first) cell — matching the order the
    // old BFS recorded components, so scores are unchanged. All 32 lanes stripe
    // the grid, so segmentation is no longer lane-0-serial.
    if (tmpl.target_obj_count > 0) {
        int16_t* lab = ctx.cc_labels;
        int seg_cells = ah * aw;
        for (int i = lane; i < seg_cells; i += 32) {
            int r = i / aw, c = i - r * aw;
            int v = ctx.output.data[r][c];
            lab[i] = (v < 0 || v == 0) ? (int16_t)-1 : (int16_t)i;
        }
        __syncwarp();
        const int8_t lddr[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
        const int8_t lddc[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
        int iterate = 1;
        while (iterate) {
            int my_changed = 0;
            for (int i = lane; i < seg_cells; i += 32) {
                int cur = lab[i];
                if (cur < 0) continue;
                int r = i / aw, c = i - r * aw;
                int v = ctx.output.data[r][c];
                int best = cur;
                #pragma unroll
                for (int d = 0; d < 8; ++d) {
                    int nr = r + lddr[d], nc = c + lddc[d];
                    if (nr < 0 || nr >= ah || nc < 0 || nc >= aw) continue;
                    if (ctx.output.data[nr][nc] != v) continue;
                    int nl = lab[nr * aw + nc];
                    if (nl >= 0 && nl < best) best = nl;
                }
                if (best < cur) { lab[i] = (int16_t)best; my_changed = 1; }
            }
            __syncwarp();
            iterate = __any_sync(0xffffffffu, my_changed);
        }
        __syncwarp();
    }

    if (lane == 0) {
        bool dims_match = (tw == aw && th == ah);

        float rarity_score = 0.0f;
        for (int i = 0; i < GPU_MAX_COLORS; ++i)
            if (my_target_hist[i] > 0)
                rarity_score += my_match_by_color[i] * ((float)total_target_pixels / my_target_hist[i]);

        float hist_err = 0.0f;
        for (int i = 0; i < GPU_MAX_COLORS; ++i) {
            int d = my_target_hist[i] - my_attempt_hist[i];
            hist_err += (float)(d < 0 ? -d : d);
        }

        float dim_pen = (float)((tw > aw ? tw - aw : aw - tw) + (th > ah ? th - ah : ah - th));

        // CC-match: segment attempt grid into per-color CCs (8-connectivity,
        // skip background = TRANSPARENT or 0), summarize as (color,w,h,count),
        // then multiset-match against precomputed target_objs.
        // CC-match: summarize the warp-parallel labels (computed above) into
        // per-component (color,w,h,x0,y0,count), then multiset-match against
        // precomputed target_objs. The summary/match scratch lives in SHARED
        // memory (one set per block) rather than per-thread local arrays — that
        // removes the local-memory spill the old lane-0 attempt_summaries array
        // generated. Only lane 0 touches it, so no sync/race.
        __shared__ gpu_object_summary attempt_summaries[GPU_MAX_TARGET_OBJECTS];
        __shared__ int16_t sh_root[GPU_MAX_TARGET_OBJECTS];
        __shared__ int16_t sh_x1[GPU_MAX_TARGET_OBJECTS];
        __shared__ int16_t sh_y1[GPU_MAX_TARGET_OBJECTS];
        int attempt_count = 0;
        int shape_matches = 0;
        float position_score = 0.0f;
        if (tmpl.target_obj_count > 0) {
            // Single row-major pass: components appear in the same order the old
            // serial BFS recorded them (representative = min linear index = the
            // component's top-left cell, encountered first in this scan). count =
            // total cells in the component; bbox from running min/max. Identical
            // summaries to the BFS, so scores are unchanged.
            const int16_t* lab = ctx.cc_labels;
            int seg_cells = ah * aw;
            for (int i = 0; i < seg_cells; ++i) {
                int root = lab[i];
                if (root < 0) continue;
                int r = i / aw, c = i - r * aw;
                int v = ctx.output.data[r][c];
                int slot = -1;
                for (int s = 0; s < attempt_count; ++s)
                    if (sh_root[s] == root) { slot = s; break; }
                if (slot < 0) {
                    if (attempt_count >= GPU_MAX_TARGET_OBJECTS) continue;
                    slot = attempt_count++;
                    sh_root[slot] = (int16_t)root;
                    attempt_summaries[slot].x0 = (int8_t)c;
                    attempt_summaries[slot].y0 = (int8_t)r;
                    sh_x1[slot] = (int16_t)c;
                    sh_y1[slot] = (int16_t)r;
                    attempt_summaries[slot].count = 0;
                }
                if (c < attempt_summaries[slot].x0) attempt_summaries[slot].x0 = (int8_t)c;
                if (r < attempt_summaries[slot].y0) attempt_summaries[slot].y0 = (int8_t)r;
                if (c > sh_x1[slot]) sh_x1[slot] = (int16_t)c;
                if (r > sh_y1[slot]) sh_y1[slot] = (int16_t)r;
                attempt_summaries[slot].count++;
            }
            for (int s = 0; s < attempt_count; ++s) {
                attempt_summaries[s].w = (int8_t)(sh_x1[s] - attempt_summaries[s].x0 + 1);
                attempt_summaries[s].h = (int8_t)(sh_y1[s] - attempt_summaries[s].y0 + 1);
            }
            // Multiset match: each attempt CC claims at most one target CC by
            // (w,h,count). Position bonus is a
            // continuous (1 - chebyshev/target_size) per matched pair —
            // normalized by the target CC's own footprint, not the grid.
            gpu_match_summaries(attempt_summaries, attempt_count, tmpl,
                                shape_matches, position_score);
        }

        // THE HAND, JUDGED BY THE SAME RULE AS THE CANVAS. A sentence that only
        // prepares paints nothing, so the canvas terms score it as if it did
        // nothing; asking the held objects the same question is what makes the
        // preparation visible. Boxes come from the paint, not the declared bbox,
        // because a Colors layer carries the whole source frame.
        int hand_shape = 0;
        float hand_position = 0.0f;
        {
            __shared__ gpu_object_summary hand_summaries[GPU_MAX_TARGET_OBJECTS];
            int hand_count = 0;
            const gpu_object_list* held = gpu_current_objects(ctx);
            if (held) {
                int held_n = min((int)held->count, GPU_KERNEL_MAX_OBJECTS);
                for (int i = 0; i < held_n && hand_count < GPU_MAX_TARGET_OBJECTS; ++i) {
                    const gpu_fixed_object& obj = held->objects[i];
                    const gpu_painted_box box = gpu_box_of_paint(obj);
                    if (box.width <= 0 || box.height <= 0) continue;
                    int painted = 0;
                    for (int r = 0; r < box.height; ++r)
                        for (int c = 0; c < box.width; ++c) {
                            int8_t v = obj.mask[box.y0 + r][box.x0 + c];
                            if (v == GPU_TRANSPARENT) continue;
                            painted++;
                        }
                    if (painted <= 0) continue;
                    gpu_object_summary& sum = hand_summaries[hand_count++];
                    sum.w = (int8_t)box.width;
                    sum.h = (int8_t)box.height;
                    sum.x0 = (int8_t)(obj.x0 + box.x0);
                    sum.y0 = (int8_t)(obj.y0 + box.y0);
                    sum.count = (int16_t)painted;
                }
            }
            if (hand_count > 0)
                gpu_match_summaries(hand_summaries, hand_count, tmpl,
                                    hand_shape, hand_position);
        }

        // Focus term (GPU-only, like lookahead): correct cells of the focus
        // color — already painted or pending — count extra. Deliberately
        // DOMINANT, not a nudge: the machine solves one color at a time, so
        // progress on the focus color outweighs everything else it could do.
        // Unrepairable wrong paint outweighs everything a chain could gain.
        constexpr float W_VANDAL = 8.0f;
        float total = -my_vandal * W_VANDAL
                    - my_exact_err
                    - hist_err * GPU_SCORE_W_HISTOGRAM
                    - my_dist_err * GPU_SCORE_W_DISTANCE_MAP
                    + rarity_score * GPU_SCORE_W_RARITY_MATCH
                    + shape_matches * GPU_SCORE_W_SHAPE_MATCH
                    + position_score * GPU_SCORE_W_POSITION_MATCH
                    + hand_shape * GPU_SCORE_W_SHAPE_MATCH
                    + hand_position * GPU_SCORE_W_POSITION_MATCH
                    // EXPERIMENT: lookahead removed. It scores what a chain is
                    // still HOLDING as if it were about to paint it -- the answer
                    // to an unfinished chain, which the four-beat cannot produce:
                    // every program ends in a COMBINE. What is left in hand after
                    // the last paint is a leftover, not an intention, and scoring
                    // it made a solved program lose 24 to 48 points for having a
                    // non-empty hand.
                    - (nc * GPU_SCORE_W_NODE_PENALTY)
                    - dim_pen;

        output_dims[idx * 2 + 0] = (int8_t)ctx.output.width;
        output_dims[idx * 2 + 1] = (int8_t)ctx.output.height;
        // WHAT THIS JOB IS, as one number: the canvas it painted AND the objects
        // it still holds. Two chains that painted the same picture but hold
        // different objects are two different chains, because the next sentence
        // acts on the hand and not on the paint -- hashing the paint alone made
        // a sentence that only rearranges the hand look like a sentence that did
        // nothing, and the dedup deleted it.
        {
            uint64_t state = 1469598103934665603ULL;
            state = gpu_fold(state, (uint64_t)(uint8_t)ctx.output.width);
            state = gpu_fold(state, (uint64_t)(uint8_t)ctx.output.height);
            for (int row = 0; row < ctx.output.height; ++row)
                for (int col = 0; col < ctx.output.width; ++col)
                    state = gpu_fold(state, (uint64_t)(uint8_t)ctx.output.data[row][col]);
            const gpu_object_list* hand =
                (ctx.current_objects_idx >= 0) ? &ctx.pool[ctx.current_objects_idx] :
                (ctx.current_objects_idx == gpu_exec_context::CURRENT_OBJECTS_SEEDS) ? &ctx.seed_objs :
                nullptr;
            state = gpu_fold(state, (uint64_t)(hand ? hand->count : 0));
            if (hand) {
                for (int i = 0; i < hand->count; ++i) {
                    const gpu_fixed_object& obj = hand->objects[i];
                    state = gpu_fold(state, (uint64_t)(uint16_t)obj.x0);
                    state = gpu_fold(state, (uint64_t)(uint16_t)obj.y0);
                    state = gpu_fold(state, (uint64_t)(uint16_t)obj.width);
                    state = gpu_fold(state, (uint64_t)(uint16_t)obj.height);
                    state = gpu_fold(state, (uint64_t)(obj.selected ? 1 : 0));
                    // Only the object's own box, never the unused rest of the mask.
                    int w = min((int)obj.width, GPU_MAX_GRID);
                    int h = min((int)obj.height, GPU_MAX_GRID);
                    for (int row = 0; row < h; ++row)
                        for (int col = 0; col < w; ++col)
                            state = gpu_fold(state, (uint64_t)(uint8_t)obj.mask[row][col]);
                }
            }
            output_hashes[idx] = state;
        }
        scores[idx] = total;
        solved_out[idx] = (dims_match && my_mismatches == 0) ? 1 : 0;
        // Say what that was made of, so nothing has to mirror the formula.
        gpu_score_terms& terms = score_terms[idx];
        terms.vandal            = -my_vandal * W_VANDAL;
        terms.exact             = -my_exact_err;
        terms.histogram         = -hist_err * GPU_SCORE_W_HISTOGRAM;
        terms.distance          = -my_dist_err * GPU_SCORE_W_DISTANCE_MAP;
        terms.rarity            = rarity_score * GPU_SCORE_W_RARITY_MATCH;
        terms.shape             = shape_matches * GPU_SCORE_W_SHAPE_MATCH;
        terms.color             = 0.0f;
        terms.position          = position_score * GPU_SCORE_W_POSITION_MATCH;
        terms.hand              = hand_shape * GPU_SCORE_W_SHAPE_MATCH
                                + hand_position * GPU_SCORE_W_POSITION_MATCH;
        terms.node_penalty      = -(nc * GPU_SCORE_W_NODE_PENALTY);
        terms.dimension_penalty = -dim_pen;
        terms.total             = total;
    }
}

// --- DIAG: capture the first CUDA error so the worker can log it. The "short
// pause then frozen/corrupt output" symptom is consistent with a sticky CUDA
// error (kernel fault / OOM) silently no-op'ing all later launches. REMOVE later.
static char g_cuda_err_msg[512] = {0};
static bool g_cuda_err_set = false;
static void capture_cuda_err(const char* where, cudaError_t e, int slot, int n) {
    if (e != cudaSuccess && !g_cuda_err_set) {
        snprintf(g_cuda_err_msg, sizeof(g_cuda_err_msg),
                 "%s slot=%d n=%d err=%s", where, slot, n, cudaGetErrorString(e));
        g_cuda_err_set = true;
    }
}
// NOT BUILT by default since 2026-09-09, and this guard is why: building it
// costs about a third of every clean build. Measured back to back that day
// with scripts/build.bat over a deleted out/, with it and without. Trust the
// ratio and never a stopwatch reading: the same pair, run twice the same
// morning, moved by about half in absolute terms while the ratio held.
// It inlines the whole op evaluator a second time, lands at 255 registers, and
// warns about its missing __launch_bounds__ on every build, which only teaches
// us to ignore that warning. Nothing calls it yet, so today we pay all of that
// for nothing. It stays here, reasoning and all, rather than being deleted and
// guessed at again later: build it with -DARC_ENABLE_CUDA_PIECES=ON the day
// inference asks the ops for pieces instead of segmenting grids on the CPU.
#if !defined(ARC_ENABLE_CUDA_PIECES) || ARC_ENABLE_CUDA_PIECES
// --- Ask an op what the pieces of a grid are ---------------------------------
// The ops are the single home of "what is an object", so nothing on the CPU
// segments a grid itself. This runs ONE op through the same evaluator the
// solver uses and hands back what it made; no op is touched or reimplemented.
// TWO ops are sent, `Input` then the splitter, because the dispatch gives an op
// with no list before it nothing to work on and just acquires the input.
// NO __launch_bounds__ ON PURPOSE, and nvcc says so every build. Nothing calls
// this yet -- inference still segments grids itself -- so the occupancy worth
// asking for depends on grid counts nobody has passed. A number now would be a
// guess frozen into the source. See Inference Master.md, "REPLACING THESE".
__global__ void dsl_pieces_kernel(gpu_exec_context* __restrict__ contexts,
                                  const gpu_fixed_grid* __restrict__ grids,
                                  node_type op, int8_t variant, int8_t background,
                                  dsl_pieces* __restrict__ out, int grid_count) {
    int idx = blockIdx.x;
    if (idx >= grid_count) return;
    int lane = threadIdx.x;
    gpu_exec_context& ctx = contexts[idx];

    const int8_t* src = reinterpret_cast<const int8_t*>(&grids[idx]);
    int8_t* dst = reinterpret_cast<int8_t*>(&ctx.input);
    for (int i = lane; i < (int)sizeof(gpu_fixed_grid); i += 32) dst[i] = src[i];
    uint8_t* dout = ctx.output.data.bytes;
    for (int i = lane; i < (int)sizeof(ctx.output.data.bytes); i += 32)
        dout[i] = arc::cells::empty_byte();
    if (lane == 0) {
        ctx.pool_head = 0;
        ctx.orientation_flipped = 0;
        ctx.output.width = ctx.input.width;
        ctx.output.height = ctx.input.height;
        // Nothing is being painted, so the size rule is "same as the input".
        ctx.rule_type = (int8_t)arc::output_dimension_rule::Identity;
        for (int k = 0; k < 6; ++k) ctx.rule_params[k] = 0;
    }
    __syncwarp();

    gpu_flat_op prog[6];
    int count = 0;
    prog[count++] = {node_type::Input, 0};
    if (arc::is_color(background) && (op == node_type::Colors || op == node_type::Split)) {
        prog[count++] = {node_type::Colors, 0};
        prog[count++] = {node_type::SelectColor, background};
        prog[count++] = {node_type::DeleteSelection, 0};
        if (op == node_type::Split) prog[count++] = {op, variant};
    } else prog[count++] = {op, variant};
    gpu_object_list& list = gpu_eval_program_warp(ctx, prog, count, lane);

    if (lane != 0) return;
    dsl_pieces& o = out[idx];
    for (int r = 0; r < GPU_MAX_GRID; ++r)
        for (int c = 0; c < GPU_MAX_GRID; ++c) o.owner[r][c] = -1;
    int n = list.count < DSL_MAX_PIECES ? list.count : DSL_MAX_PIECES;
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& ob = list.objects[i];
        dsl_pieces::piece p;
        p.x0 = (int8_t)ob.x0; p.y0 = (int8_t)ob.y0;
        p.w = (int8_t)ob.width; p.h = (int8_t)ob.height;
        p.cells = 0;
        int one_color = -1; bool mixed = false;
        for (int r = 0; r < ob.height; ++r)
            for (int c = 0; c < ob.width; ++c) {
                int8_t v = ob.mask[r][c];
                if (v == GPU_TRANSPARENT) continue;
                p.cells++;
                int gr = ob.y0 + r, gc = ob.x0 + c;
                if (gr >= 0 && gr < GPU_MAX_GRID && gc >= 0 && gc < GPU_MAX_GRID)
                    o.owner[gr][gc] = (int8_t)i;
                if (one_color < 0) one_color = v;
                else if (one_color != v) mixed = true;
            }
        p.color = mixed ? (int8_t)-1 : (int8_t)one_color;
        o.pieces[i] = p;
    }
    o.count = n;
}

void cuda_dsl_pieces(const gpu_fixed_grid* grids, int grid_count,
                     node_type op, int8_t variant, int8_t background,
                     dsl_pieces* out) {
    if (grid_count <= 0) return;
    gpu_exec_context* d_ctx = nullptr;
    gpu_fixed_grid*   d_grids = nullptr;
    dsl_pieces*       d_out = nullptr;
    // Load-time only and a handful of grids, so allocate for the call rather
    // than keeping a pool alive for something that runs once per puzzle.
    if (cudaMalloc(&d_ctx,   (size_t)grid_count * sizeof(gpu_exec_context)) != cudaSuccess ||
        cudaMalloc(&d_grids, (size_t)grid_count * sizeof(gpu_fixed_grid))   != cudaSuccess ||
        cudaMalloc(&d_out,   (size_t)grid_count * sizeof(dsl_pieces))       != cudaSuccess) {
        cudaFree(d_ctx); cudaFree(d_grids); cudaFree(d_out);
        for (int i = 0; i < grid_count; ++i) out[i].count = 0;
        return;
    }
    cudaMemcpy(d_grids, grids, (size_t)grid_count * sizeof(gpu_fixed_grid), cudaMemcpyHostToDevice);
    // DEFAULT STREAM ON PURPOSE, and nvcc says so every build. The copies either
    // side of this are blocking, so a stream would have nothing to overlap with.
    // Whoever makes those async wants to revisit this line in the same breath.
    dsl_pieces_kernel<<<grid_count, 32>>>(d_ctx, d_grids, op, variant, background, d_out, grid_count);
    cudaMemcpy(out, d_out, (size_t)grid_count * sizeof(dsl_pieces), cudaMemcpyDeviceToHost);
    cudaFree(d_ctx); cudaFree(d_grids); cudaFree(d_out);
}

#endif // ARC_ENABLE_CUDA_PIECES

bool cuda_dsl_take_error(char* out, int outsz) {
    if (!g_cuda_err_set) return false;
    snprintf(out, outsz, "%s", g_cuda_err_msg);
    g_cuda_err_set = false;
    return true;
}

void cuda_dsl_exec_and_score(cuda_dsl_exec_batch& batch, int slot) {
    if (batch.job_count <= 0) return;

    int n = batch.job_count;
    int active = std::min((int)batch.active_examples, MAX_CACHED_EXAMPLES);
    ensure_exec_capacity(n, slot);

    cudaStream_t s = get_exec_stream(slot);

    cudaMemcpyAsync(d_exec_programs[slot],     batch.programs.data(),        n * GPU_MAX_OPS * sizeof(gpu_flat_op), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_exec_prog_lengths[slot], batch.program_lengths.data(), n * sizeof(int16_t),                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_exec_node_counts[slot],  batch.node_counts.data(),     n * sizeof(int16_t),                    cudaMemcpyHostToDevice, s);

    int threads = 32;  // one warp per job
    int blocks = n;
    dsl_exec_and_score_kernel_v2<<<blocks, threads, 0, s>>>(
        d_exec_contexts[slot], d_example_templates,
        d_exec_programs[slot], d_exec_prog_lengths[slot], d_exec_node_counts[slot],
        d_cached_rule_type, d_cached_rule_params,
        d_exec_scores[slot], d_exec_solved[slot],
        d_exec_output_grids[slot], d_exec_output_dims[slot],
        active, n);
    capture_cuda_err("launch", cudaGetLastError(), slot, n);  // launch errors (kept: cheap, catches future faults)

    // Score kernel: same stream -> runs after exec finishes writing contexts[].
    dsl_score_kernel<<<n, 32, 0, s>>>(
        d_exec_contexts[slot], d_example_templates, d_exec_node_counts[slot],
        d_exec_scores[slot], d_exec_solved[slot], d_exec_output_dims[slot],
        d_exec_output_hashes[slot], d_exec_score_terms[slot],
        active, n);
    capture_cuda_err("score-launch", cudaGetLastError(), slot, n);

    cudaMemcpyAsync(batch.scores.data(), d_exec_scores[slot], n * sizeof(float),  cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.solved.data(), d_exec_solved[slot], n * sizeof(int8_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.output_hashes.data(), d_exec_output_hashes[slot], n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.score_terms.data(), d_exec_score_terms[slot], n * sizeof(gpu_score_terms), cudaMemcpyDeviceToHost, s);
    // No sync here — caller does CPU work then calls cuda_dsl_wait_for_exec(slot).
}

void cuda_dsl_wait_for_exec(int slot) {
    if (exec_stream[slot]) capture_cuda_err("sync", cudaStreamSynchronize(exec_stream[slot]), slot, 0);  // kernel exec errors surface here
}

static long long g_fetch_copy_ns = 0;
static long long g_fetch_sync_ns = 0;

void cuda_dsl_take_fetch_ns(long long* copy_ns, long long* sync_ns) {
    if (copy_ns) *copy_ns = g_fetch_copy_ns;
    if (sync_ns) *sync_ns = g_fetch_sync_ns;
    g_fetch_copy_ns = 0;
    g_fetch_sync_ns = 0;
}

// Where a winner's picture lands. Its own pinned buffer, allocated once and
// never resized, so nothing can free it while a copy is still in flight.
static int8_t* h_winner_grids[2] = {nullptr, nullptr};
static int8_t* h_winner_dims[2]  = {nullptr, nullptr};

static inline void ensure_winner_landing(int slot) {
    if (h_winner_grids[slot]) return;
    cudaHostAlloc((void**)&h_winner_grids[slot],
                  (size_t)MAX_CACHED_EXAMPLES * GPU_GRID_CELLS, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_winner_dims[slot],
                  (size_t)MAX_CACHED_EXAMPLES * 2, cudaHostAllocDefault);
}

const int8_t* cuda_dsl_winner_grids(int slot) { return h_winner_grids[slot]; }
const int8_t* cuda_dsl_winner_dims(int slot)  { return h_winner_dims[slot]; }

void cuda_dsl_fetch_output_grids_async(int winner_idx, int active_examples, int slot) {
    int active = std::min(active_examples, MAX_CACHED_EXAMPLES);
    if (active <= 0 || winner_idx < 0) return;
    ensure_winner_landing(slot);
    if (!h_winner_grids[slot] || !h_winner_dims[slot]) return;
    int base = winner_idx * active;
    cudaStream_t s = get_exec_stream(slot);
    auto t0 = std::chrono::steady_clock::now();
    cudaMemcpyAsync(h_winner_grids[slot],
                    d_exec_output_grids[slot] + (size_t)base * GPU_GRID_CELLS,
                    (size_t)active * GPU_GRID_CELLS * sizeof(int8_t),
                    cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_winner_dims[slot],
                    d_exec_output_dims[slot] + (size_t)base * 2,
                    (size_t)active * 2 * sizeof(int8_t),
                    cudaMemcpyDeviceToHost, s);
    g_fetch_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

void cuda_dsl_fetch_output_grids(cuda_dsl_exec_batch& batch, int winner_idx, int active_examples, int slot) {
    int active = std::min(active_examples, MAX_CACHED_EXAMPLES);
    int base = winner_idx * active;
    cudaStream_t s = get_exec_stream(slot);
    auto t0 = std::chrono::steady_clock::now();
    cudaMemcpyAsync(batch.output_grids.data()       + base * GPU_GRID_CELLS,
                    d_exec_output_grids[slot]       + base * GPU_GRID_CELLS,
                    active * GPU_GRID_CELLS * sizeof(int8_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.output_dims_result.data() + base * 2,
                    d_exec_output_dims[slot]        + base * 2,
                    active * 2 * sizeof(int8_t), cudaMemcpyDeviceToHost, s);
    auto t1 = std::chrono::steady_clock::now();
    cudaStreamSynchronize(s);
    auto t2 = std::chrono::steady_clock::now();
    g_fetch_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    g_fetch_sync_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// --- Scratch exec path: reads from d_scratch_* buffers, isolated from main cache. ---

static void ensure_scratch_capacity(int job_count) {
    resize_exec_buffers(job_count, d_scratch_capacity,
                        d_scratch_contexts, d_scratch_programs,
                        d_scratch_prog_lengths, d_scratch_node_counts,
                        d_scratch_scores, d_scratch_solved,
                        d_scratch_output_grids, d_scratch_output_dims,
                        d_scratch_output_hashes, d_scratch_score_terms);
}

void cuda_dsl_exec_and_score_scratch(cuda_dsl_exec_batch& batch) {
    if (batch.job_count <= 0) return;

    int n = batch.job_count;
    int active = std::min((int)batch.active_examples, MAX_CACHED_EXAMPLES);
    ensure_scratch_capacity(n);

    cudaStream_t s = get_exec_stream(0);  // scratch uses slot 0 (independent from main pipeline)

    cudaMemcpyAsync(d_scratch_programs,     batch.programs.data(),        n * GPU_MAX_OPS * sizeof(gpu_flat_op), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_scratch_prog_lengths, batch.program_lengths.data(), n * sizeof(int16_t),                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_scratch_node_counts,  batch.node_counts.data(),     n * sizeof(int16_t),                    cudaMemcpyHostToDevice, s);

    int threads = 32;
    int blocks = n;
    dsl_exec_and_score_kernel_v2<<<blocks, threads, 0, s>>>(
        d_scratch_contexts, d_scratch_templates,
        d_scratch_programs, d_scratch_prog_lengths, d_scratch_node_counts,
        d_scratch_rule_type, d_scratch_rule_params,
        d_scratch_scores, d_scratch_solved,
        d_scratch_output_grids, d_scratch_output_dims,
        active, n);
    capture_cuda_err("scratch-launch", cudaGetLastError(), -1, n);  // GUI eval path; also surfaces sticky errors

    dsl_score_kernel<<<n, 32, 0, s>>>(
        d_scratch_contexts, d_scratch_templates, d_scratch_node_counts,
        d_scratch_scores, d_scratch_solved, d_scratch_output_dims,
        d_scratch_output_hashes, d_scratch_score_terms,
        active, n);
    capture_cuda_err("scratch-score-launch", cudaGetLastError(), -1, n);

    cudaMemcpyAsync(batch.scores.data(), d_scratch_scores, n * sizeof(float),  cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.solved.data(), d_scratch_solved, n * sizeof(int8_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.score_terms.data(), d_scratch_score_terms, n * sizeof(gpu_score_terms), cudaMemcpyDeviceToHost, s);

    cudaStreamSynchronize(s);
}

void cuda_dsl_fetch_output_grids_scratch(cuda_dsl_exec_batch& batch, int winner_idx, int active_examples) {
    int active = std::min(active_examples, MAX_CACHED_EXAMPLES);
    int base = winner_idx * active;
    cudaStream_t s = get_exec_stream(0);  // scratch uses slot 0 (independent from main pipeline)
    cudaMemcpyAsync(batch.output_grids.data()       + base * GPU_GRID_CELLS,
                    d_scratch_output_grids          + base * GPU_GRID_CELLS,
                    active * GPU_GRID_CELLS * sizeof(int8_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(batch.output_dims_result.data() + base * 2,
                    d_scratch_output_dims           + base * 2,
                    active * 2 * sizeof(int8_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
}

bool cuda_dsl_fetch_last_object_list_scratch(int job_idx, gpu_object_list& out,
                                             int* out_canvas_w, int* out_canvas_h) {
    if (!d_scratch_contexts || job_idx < 0 || job_idx >= d_scratch_capacity) return false;

    cudaStream_t s = get_exec_stream(0);  // scratch uses slot 0 (independent from main pipeline)
    gpu_exec_context* d_ctx = d_scratch_contexts + job_idx;

    // Working output-canvas dimensions after execution (including rotate ops).
    // The shared grid stores adjacent int dimensions at its start.
    if (out_canvas_w || out_canvas_h) {
        int wh[2] = {0, 0};
        cudaMemcpyAsync(wh,
                        reinterpret_cast<char*>(d_ctx) + offsetof(gpu_exec_context, output)
                            + offsetof(gpu_fixed_grid, width),
                        sizeof(wh), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        if (out_canvas_w) *out_canvas_w = wh[0];
        if (out_canvas_h) *out_canvas_h = wh[1];
    }

    int current_idx = gpu_exec_context::CURRENT_OBJECTS_NONE;
    cudaMemcpyAsync(&current_idx,
                    reinterpret_cast<char*>(d_ctx) + offsetof(gpu_exec_context, current_objects_idx),
                    sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    if (current_idx == gpu_exec_context::CURRENT_OBJECTS_NONE) return false;

    size_t src_off = 0;
    if (current_idx == gpu_exec_context::CURRENT_OBJECTS_SEEDS) {
        src_off = offsetof(gpu_exec_context, seed_objs);
        } else if (current_idx >= 0 && current_idx < GPU_KERNEL_POOL_SIZE) {
        src_off = offsetof(gpu_exec_context, pool) + current_idx * sizeof(gpu_object_list);
    } else {
        return false;
    }

    cudaMemcpyAsync(&out,
                    reinterpret_cast<char*>(d_ctx) + src_off,
                    sizeof(gpu_object_list), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return true;
}
