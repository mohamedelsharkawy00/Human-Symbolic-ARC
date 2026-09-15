#include "cuda_solver.cuh"
#include "10_dsl.cuh"
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

// Scoring weights live in 10_dsl.cuh (GPU_SCORE_W_*, single home).

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
// sits in; object_cells is aligned and padded to keep that word inside the
// store. Only for lane-striped painting -- serial code should use the proxy,
// which costs nothing.
//
// This loop is what packing the canvas costs: one puzzle went 8.6 -> 9.0,
// about 4.7% slower, measured 2026-09-09. Taken deliberately, for one cell
// type across host and device instead of two and a conversion between them.
// If it is ever wanted back, it is contained: only the union paint, the
// overlay paint, the intersection clear and the canvas transpose call this.
__device__ inline void gpu_paint_cell(gpu_packed_mask& m, int r, int c, int8_t value) {
    const int index = r * GPU_MAX_GRID + c;
    if (index < 0 || index >= gpu_packed_mask::CELLS) return;
    const int byte = index >> 1;
    const int shift = ((index & 1) * 4) + ((byte & 3) * 8);
    unsigned int* word = reinterpret_cast<unsigned int*>(m.bytes) + (byte >> 2);
    const unsigned int keep = ~(0xFu << shift);
    const unsigned int put = (unsigned int)gpu_packed_mask::enc(value) << shift;
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
    if (ctx.pool_head >= GPU_KERNEL_POOL_SIZE) ctx.pool_head = GPU_KERNEL_POOL_SIZE - 1;
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

// Init helper: stripe-copy only the USED objects of a read-only source list
// (count * sizeof(object)) into the per-job context, not all 64 slots. The
// seeds/palette/output_colors lists are read-only references that typically
// hold a handful of objects, so the old full 57 KB copy was mostly moving
// unused slots. Unused dst slots stay stale but are never read (ops are bounded
// by count). All 32 lanes cooperate; lane 0 writes the count.
__device__ inline void gpu_copy_used_objs(const gpu_object_list& src,
                                          gpu_object_list& dst, int lane) {
    int n = src.count;
    if (n < 0) n = 0;
    if (n > GPU_KERNEL_MAX_OBJECTS) n = GPU_KERNEL_MAX_OBJECTS;
    const int8_t* s = reinterpret_cast<const int8_t*>(src.objects);
    int8_t* d = reinterpret_cast<int8_t*>(dst.objects);
    int used = n * (int)sizeof(gpu_fixed_object);
    for (int i = lane; i < used; i += 32) d[i] = s[i];
    if (lane == 0) dst.count = n;
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

    __device__ void operator()(const arc::object_summary&, const int16_t* cells, int n) const {
        if (out->count >= GPU_KERNEL_MAX_OBJECTS) return;
        gpu_fixed_object& obj = out->objects[out->count++];
        // Keep the full grid frame; segmentation never tightens (Trim is the only
        // shrinker). The object's cells sit at their grid position, rest transparent
        // -- which is box-local too, the box just being the whole grid.
        obj.x0 = 0; obj.y0 = 0;
        obj.x1 = (int16_t)(g->width - 1); obj.y1 = (int16_t)(g->height - 1);
        obj.width = g->width; obj.height = g->height;
        obj.mask.clear();
        for (int i = 0; i < n; ++i) {
            const int r = cells[i] / arc::MAX_GRID_DIM, c = cells[i] % arc::MAX_GRID_DIM;
            obj.mask[r][c] = g->data[r][c];
        }
    }
};

__device__ void gpu_segment_objects(const gpu_fixed_grid& g, gpu_object_list& out, int8_t bg_color = GPU_TRANSPARENT, bool any_color = false) {
    out.count = 0;
    if (g.width <= 0 || g.height <= 0) return;
    arc::object_walk_scratch scratch;
    arc::each_object(g, (int)bg_color,
                     any_color ? arc::grouping::touching_any_color
                               : arc::grouping::touching_same_color,
                     scratch, gpu_take_object{ &out, &g });
}

__device__ int8_t gpu_majority_vote(const gpu_packed_mask& mask, int r, int c, int factor, int oh, int ow) {
    int counts[GPU_MAX_COLORS];
    memset(counts, 0, sizeof(counts));
    int non_trans = 0;
    for (int dr = 0; dr < factor; ++dr)
        for (int dc = 0; dc < factor; ++dc) {
            int sr = r * factor + dr, sc = c * factor + dc;
            if (sr >= oh || sc >= ow) continue;
            int8_t clr = mask[sr][sc];
            if (clr == GPU_TRANSPARENT) continue;
            counts[clr % GPU_MAX_COLORS]++; non_trans++;
        }
    if (non_trans == 0) return GPU_TRANSPARENT;
    int winner = -1, best = -1;
    for (int i = 0; i < GPU_MAX_COLORS; ++i)
        if (counts[i] > best) { best = counts[i]; winner = i; }
    return (int8_t)winner;
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
    if (value < GPU_MAX_COLORS) return value;
    int counts[GPU_MAX_COLORS] = {};
    for (int r = 0; r < o.height; ++r)
        for (int c = 0; c < o.width; ++c) {
            int8_t v = o.mask[r][c];
            if (v >= 0 && v < GPU_MAX_COLORS) counts[v]++;
        }
    bool rarest = (value == GPU_MAX_COLORS);
    int best = GPU_TRANSPARENT, best_n = 0;
    for (int col = 0; col < GPU_MAX_COLORS; ++col) {
        if (counts[col] == 0) continue;
        if (best < 0 || (rarest ? counts[col] < best_n : counts[col] > best_n)) {
            best = col; best_n = counts[col];
        }
    }
    return (int8_t)best;
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

// InputWithoutBackground: the input with the elected ground punched out, frame
// kept. The ground is not a figure, so the source says it once instead of every
// beat deciding for itself.
__device__ gpu_object_list& gpu_exec_input_without_background(gpu_exec_context& ctx) {
    auto& res = gpu_acquire(ctx);
    gpu_frame_to_content(res, ctx.input);
    if (ctx.background_color < 0 || res.count == 0) return res;
    gpu_fixed_object& obj = res.objects[0];
    for (int r = 0; r < obj.height; ++r)
        for (int c = 0; c < obj.width; ++c)
            if (obj.mask[r][c] == ctx.background_color) obj.mask[r][c] = GPU_TRANSPARENT;
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
    gpu_frame_to_content(res, ctx.output);
    return res;
}

__device__ gpu_object_list& gpu_exec_seeds(gpu_exec_context& ctx) { return ctx.seed_objs; }

// Split[depth]: segmentation depth, coarse to fine. [0] = connected object,
// colors mixed. [1] = connected object of one color. [2] = single pixels.
__device__ gpu_object_list& gpu_exec_split(gpu_exec_context& ctx, const gpu_object_list& src, int8_t mode) {
    auto& res = gpu_acquire(ctx);
    gpu_fixed_grid temp;
    // Size the perception canvas to the full 30x30 ceiling, not the input dims:
    // the stream may have grown past the input bbox (Tile/Shift/Extend...), and
    // input-sizing would silently crop those pixels before segmentation.
    gpu_paint_objects(src, GPU_MAX_GRID, GPU_MAX_GRID, temp);
    // The ground is not a figure: erase the elected input canvas before
    // segmenting, so no program spends nodes dropping it afterwards.
    if (ctx.background_color >= 0)
        for (int r = 0; r < GPU_MAX_GRID; ++r)
            for (int c = 0; c < GPU_MAX_GRID; ++c)
                if (temp.data[r][c] == ctx.background_color)
                    temp.data[r][c] = GPU_TRANSPARENT;
    if (mode == 2) {
        // Split[2] = pixelize: one 1x1 object per non-transparent cell (no
        // connectivity merge). Sparse scenes only -- caps at GPU_KERNEL_MAX_OBJECTS,
        // so dense grids (>64 painted cells) get truncated.
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
    gpu_segment_objects(temp, res, GPU_TRANSPARENT, mode == 0);
    return res;
}

// Colors: split each source object into its color layers, minus the elected
// input background — the ground is not a figure, so no program spends nodes
// dropping it. Every layer keeps the SOURCE object's own bbox as-is —
// segmentation never touches the frame (Trim is the only shrinker). A layer's
// mask carries just its color's cells at the object's local coordinates, the
// rest transparent.
__device__ gpu_object_list& gpu_exec_colors(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count; ++i) {
        const auto& sobj = src.objects[i];
        int w = sobj.width, h = sobj.height;
        for (int color = 0; color < GPU_MAX_COLORS; ++color) {
            // The ground is not a figure: segmentation never hands back the
            // elected input canvas, so no program spends nodes dropping it.
            if (color == ctx.background_color) continue;
            bool found = false;
            for (int r = 0; r < h && !found; ++r)
                for (int c = 0; c < w; ++c)
                    if (sobj.mask[r][c] == color) { found = true; break; }
            if (!found) continue;
            if (res.count >= GPU_KERNEL_MAX_OBJECTS) return res;
            gpu_fixed_object& obj = res.objects[res.count];
            obj.x0 = sobj.x0; obj.y0 = sobj.y0; obj.x1 = sobj.x1; obj.y1 = sobj.y1;
            obj.width = sobj.width; obj.height = sobj.height;
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
    res = src;
    int freq[GPU_MAX_COLORS];
    for (int i = 0; i < GPU_MAX_COLORS; ++i) freq[i] = 0;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c) {
                int8_t col = obj.mask[r][c];
                if (arc::is_color(col)) freq[col]++;
            }
    }
    int top = 0;
    for (int i = 1; i < GPU_MAX_COLORS; ++i) if (freq[i] > freq[top]) top = i;
    if (freq[top] == 0) return res;  // nothing painted -> nothing to mark
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
__device__ gpu_object_list& gpu_exec_select_object_by_size(gpu_exec_context& ctx, const gpu_object_list& src, int rank, bool largest) {
    auto& res = gpu_acquire(ctx);
    res = src;
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

__device__ gpu_object_list& gpu_exec_bridge(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i)
        res.objects[res.count++] = src.objects[i];

    constexpr int MAX_PAIRS = 64;
    int16_t pair_lo[MAX_PAIRS], pair_hi[MAX_PAIRS];
    int pair_count = 0;

    for (int i = 0; i < src.count; ++i) {
        int8_t ci = gpu_obj_first_color(src.objects[i]);
        if (ci < 0) continue;
        int icx = (src.objects[i].x0 + src.objects[i].x1) / 2;
        int icy = (src.objects[i].y0 + src.objects[i].y1) / 2;
        int best_j = -1, best_dist = 0x7FFF;
        for (int j = 0; j < src.count; ++j) {
            if (j == i) continue;
            int8_t cj = gpu_obj_first_color(src.objects[j]);
            if (cj != ci) continue;
            int jcx = (src.objects[j].x0 + src.objects[j].x1) / 2;
            int jcy = (src.objects[j].y0 + src.objects[j].y1) / 2;
            if (icx != jcx && icy != jcy) continue;
            int dist = abs(jcx - icx) + abs(jcy - icy);
            if (dist < best_dist) { best_dist = dist; best_j = j; }
        }
        if (best_j < 0) continue;
        int lo = min(i, best_j), hi = max(i, best_j);
        bool dup = false;
        for (int p = 0; p < pair_count; ++p)
            if (pair_lo[p] == lo && pair_hi[p] == hi) { dup = true; break; }
        if (dup || pair_count >= MAX_PAIRS) continue;
        pair_lo[pair_count] = (int16_t)lo; pair_hi[pair_count] = (int16_t)hi; pair_count++;
    }

    for (int p = 0; p < pair_count && res.count < GPU_KERNEL_MAX_OBJECTS; ++p) {
        int i = pair_lo[p], j = pair_hi[p];
        int8_t ci = gpu_obj_first_color(src.objects[i]);
        if (ci < 0) ci = 0;
        const auto& a = src.objects[i]; const auto& b = src.objects[j];
        int acx = (a.x0 + a.x1) / 2, bcx = (b.x0 + b.x1) / 2;
        int dx = abs(bcx - acx), dy = abs((a.y0 + a.y1) / 2 - (b.y0 + b.y1) / 2);
        if (dx == 0 && dy == 0) continue;
        gpu_fixed_object br;
        if (dx >= dy) {
            int x_min = min((int)a.x1, (int)b.x1) + 1, x_max = max((int)a.x0, (int)b.x0) - 1;
            if (x_min > x_max) continue;
            int y_top = min((int)a.y0, (int)b.y0), y_bot = max((int)a.y1, (int)b.y1);
            br.x0 = (int16_t)x_min; br.y0 = (int16_t)y_top;
            br.x1 = (int16_t)x_max; br.y1 = (int16_t)y_bot;
            br.width = (int16_t)min(x_max - x_min + 1, GPU_MAX_GRID);
            br.height = (int16_t)min(y_bot - y_top + 1, GPU_MAX_GRID);
        } else {
            int y_min = min((int)a.y1, (int)b.y1) + 1, y_max = max((int)a.y0, (int)b.y0) - 1;
            if (y_min > y_max) continue;
            int x_left = min((int)a.x0, (int)b.x0), x_right = max((int)a.x1, (int)b.x1);
            br.x0 = (int16_t)x_left; br.y0 = (int16_t)y_min;
            br.x1 = (int16_t)x_right; br.y1 = (int16_t)y_max;
            br.width = (int16_t)min(x_right - x_left + 1, GPU_MAX_GRID);
            br.height = (int16_t)min(y_max - y_min + 1, GPU_MAX_GRID);
        }
        br.mask.clear();
        for (int r = 0; r < br.height; ++r)
            for (int c = 0; c < br.width; ++c)
                br.mask[r][c] = ci;
        res.objects[res.count++] = br;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_partition(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    // Selection marks the PARTITIONER: selected cells are the divider walls;
    // the walls are only VIEWED to locate the cut lines, never merged into the
    // output. Strict, like every act: no selection means no walls, so the whole
    // thing stays one tile — Partition NEVER falls back to "all painted are
    // walls". The tiles are cut from the SOURCE objects (clipped to each tile
    // rectangle), so Partition views the scene as a grid but does not modify the
    // list by merging — want tiles as single masks? place a UnionList yourself.
    int gw = ctx.input.width, gh = ctx.input.height;
    if (gw <= 0 || gh <= 0) return res;

    gpu_fixed_grid walls;
    walls.clear_cells();
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
        if (!obj.selected) continue;
        GPU_DEV_FOR_EACH_PIXEL(obj, r, c, v) {
            int gr = obj.y0 + r, gc = obj.x0 + c;
            if (gr >= 0 && gr < gh && gc >= 0 && gc < gw) walls.data[gr][gc] = v;
        }
    }

    bool is_divider[GPU_MAX_GRID][GPU_MAX_GRID];
    memset(is_divider, 0, sizeof(is_divider));
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c)
            if (walls.data[r][c] != GPU_TRANSPARENT) is_divider[r][c] = true;

    bool div_row[GPU_MAX_GRID], div_col[GPU_MAX_GRID];
    memset(div_row, 0, sizeof(div_row)); memset(div_col, 0, sizeof(div_col));
    for (int r = 0; r < gh; ++r) {
        bool full = true;
        for (int c = 0; c < gw && full; ++c) if (!is_divider[r][c]) full = false;
        div_row[r] = full;
    }
    for (int c = 0; c < gw; ++c) {
        bool full = true;
        for (int r = 0; r < gh && full; ++r) if (!is_divider[r][c]) full = false;
        div_col[c] = full;
    }

    int16_t row_start[GPU_MAX_GRID], row_end[GPU_MAX_GRID]; int row_span_count = 0;
    int16_t col_start[GPU_MAX_GRID], col_end[GPU_MAX_GRID]; int col_span_count = 0;
    for (int r = 0; r < gh;) {
        if (div_row[r]) { ++r; continue; }
        int start = r; while (r < gh && !div_row[r]) ++r;
        if (row_span_count < GPU_MAX_GRID) { row_start[row_span_count] = (int16_t)start; row_end[row_span_count] = (int16_t)r; row_span_count++; }
    }
    for (int c = 0; c < gw;) {
        if (div_col[c]) { ++c; continue; }
        int start = c; while (c < gw && !div_col[c]) ++c;
        if (col_span_count < GPU_MAX_GRID) { col_start[col_span_count] = (int16_t)start; col_end[col_span_count] = (int16_t)c; col_span_count++; }
    }

    // Cut each SOURCE object (walls excluded) by the tile rectangles: clip its
    // cells to each tile, emit the clipped piece as its own object at the tile's
    // frame. A source object crossing a tile boundary is severed there — that is
    // what partitioning means. No merge: distinct source layers stay distinct.
    for (int ri = 0; ri < row_span_count; ++ri)
        for (int ci = 0; ci < col_span_count; ++ci) {
            int r0 = row_start[ri], r1 = row_end[ri], c0 = col_start[ci], c1 = col_end[ci];
            int w = c1 - c0, h = r1 - r0;
            if (w <= 0 || h <= 0) continue;
            for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
                const auto& s = src.objects[i];
                if (s.selected) continue;  // walls are not content
                gpu_fixed_object obj;
                obj.x0 = (int16_t)c0; obj.y0 = (int16_t)r0;
                obj.x1 = (int16_t)(c1 - 1); obj.y1 = (int16_t)(r1 - 1);
                obj.width = (int16_t)w; obj.height = (int16_t)h;
                obj.selected = false;
                obj.mask.clear();
                bool any = false;
                GPU_DEV_FOR_EACH_PIXEL(s, sr, sc, v) {
                    int gr = s.y0 + sr, gc = s.x0 + sc;
                    if (gr < r0 || gr >= r1 || gc < c0 || gc >= c1) continue;
                    obj.mask[gr - r0][gc - c0] = v;
                    any = true;
                }
                if (any) res.objects[res.count++] = obj;
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

    // Mark phase: every non-transparent src pixel marks its canvas cell covered.
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj = src->objects[i];
        int pw = obj.width, ph = obj.height;
        int total = pw * ph;
        int x0 = obj.x0, y0 = obj.y0;
        for (int q = lane; q < total; q += 32) {
            int r = q / pw, c = q - r * pw;
            if (obj.mask[r][c] != GPU_TRANSPARENT) {
                int gr = y0 + r, gc = x0 + c;
                if (gr >= 0 && gr < oh && gc >= 0 && gc < ow)
                    covered[gr * GPU_MAX_GRID + gc] = 1;
            }
        }
    }
    __syncwarp();

    // Clear phase: drop canvas cells not covered by any src object.
    for (int p = lane; p < oh * ow; p += 32) {
        int r = p / ow, c = p - r * ow;
        if (!covered[r * GPU_MAX_GRID + c]) gpu_paint_cell(ctx.output.data, r, c, GPU_TRANSPARENT);
    }
    __syncwarp();

    // Pass src through unchanged (fresh pool slot).
    gpu_object_list* pass = &gpu_acquire_and_publish(ctx);
    for (int i = 0; i < n; ++i) pass->objects[i] = src->objects[i];
    if (lane == 0) pass->count = n;
    __syncwarp();
}

// Warp-parallel union: composite src's pixels onto the canvas, then pass src
// through UNCHANGED (canvas := canvas ∪ src). The per-pixel paint — the measured
// ~27% hotspot, previously lane-0 serial — is striped across the 32 lanes:
// pixels of one object map to disjoint canvas cells (no write conflict), and
// objects are processed sequentially so overlapping objects still resolve
// later-wins (a warp issues its same-address stores in program order). Union
// must NOT collapse src into one object (that's UnionList); keeping src intact lets
// a downstream per-object op still see each object. Mirrors gpu_exec_*_warp.
__device__ void gpu_exec_union_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    int oh = ctx.output.height, ow = ctx.output.width;

    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj = src->objects[i];
        int pw = obj.width, ph = obj.height;
        int total = pw * ph;
        int x0 = obj.x0, y0 = obj.y0;
        for (int p = lane; p < total; p += 32) {
            int r = p / pw, c = p - r * pw;   // pw > 0 since total > 0
            int8_t v = obj.mask[r][c];
            if (v != GPU_TRANSPARENT) {
                int gr = y0 + r, gc = x0 + c;
                if (gr >= 0 && gr < oh && gc >= 0 && gc < ow)
                    gpu_paint_cell(ctx.output.data, gr, gc, v);
            }
        }
    }
    __syncwarp();

    // Pass src through unchanged as the result (fresh pool slot, src copied).
    gpu_object_list* pass = &gpu_acquire_and_publish(ctx);
    int used = n * (int)sizeof(gpu_fixed_object);
    const int8_t* s = reinterpret_cast<const int8_t*>(src->objects);
    int8_t* d = reinterpret_cast<int8_t*>(pass->objects);
    for (int b = lane; b < used; b += 32) d[b] = s[b];
    if (lane == 0) pass->count = n;
    __syncwarp();
}

// Withhold: closes the sentence and paints nothing. The hand rides through so
// the next sentence can act on it, and the canvas is left exactly as it was --
// which is the whole point: a sentence that only prepares should read as having
// done nothing, not as having damaged the picture.
__device__ void gpu_exec_withhold_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    gpu_object_list* pass = &gpu_acquire_and_publish(ctx);
    int used = n * (int)sizeof(gpu_fixed_object);
    const int8_t* source = reinterpret_cast<const int8_t*>(src->objects);
    int8_t* dest = reinterpret_cast<int8_t*>(pass->objects);
    for (int b = lane; b < used; b += 32) dest[b] = source[b];
    if (lane == 0) pass->count = n;
    __syncwarp();
}

// Underlay: like Union but paints each pixel ONLY into transparent canvas cells
// (paint-behind). One-op, non-destructive z-order -- "put this under what's
// already there" instead of reordering the chain. The only change from
// gpu_exec_union_warp is the per-cell guard: store iff the canvas cell is still
// transparent. src is passed through unchanged, same as Union.
__device__ void gpu_exec_underlay_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    int oh = ctx.output.height, ow = ctx.output.width;

    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj = src->objects[i];
        int pw = obj.width, ph = obj.height;
        int total = pw * ph;
        int x0 = obj.x0, y0 = obj.y0;
        for (int p = lane; p < total; p += 32) {
            int r = p / pw, c = p - r * pw;
            int8_t v = obj.mask[r][c];
            if (v != GPU_TRANSPARENT) {
                int gr = y0 + r, gc = x0 + c;
                if (gr >= 0 && gr < oh && gc >= 0 && gc < ow
                    && ctx.output.data[gr][gc] == GPU_TRANSPARENT)
                    gpu_paint_cell(ctx.output.data, gr, gc, v);
            }
        }
    }
    __syncwarp();

    gpu_object_list* pass = &gpu_acquire_and_publish(ctx);
    int used = n * (int)sizeof(gpu_fixed_object);
    const int8_t* s = reinterpret_cast<const int8_t*>(src->objects);
    int8_t* d = reinterpret_cast<int8_t*>(pass->objects);
    for (int b = lane; b < used; b += 32) d[b] = s[b];
    if (lane == 0) pass->count = n;
    __syncwarp();
}

__device__ gpu_object_list& gpu_exec_difference(gpu_exec_context& ctx, const gpu_object_list& src) {
    for (int i = 0; i < src.count; ++i) {
        const auto& obj = src.objects[i];
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
        int idx = b << 1;
        int r = idx / GPU_MAX_GRID, c = idx - r * GPU_MAX_GRID;   // c even => c,c+1 same row
        int8_t v0 = (r < mh && c     < mw && !wall[r][c]     && !outside[r][c])     ? fill_value : (int8_t)obj_src.mask[r][c];
        int8_t v1 = (r < mh && c + 1 < mw && !wall[r][c + 1] && !outside[r][c + 1]) ? fill_value : (int8_t)obj_src.mask[r][c + 1];
        out.mask.bytes[b] = gpu_packed_mask::enc(v0) | (gpu_packed_mask::enc(v1) << 4);
    }
    __syncwarp();
}

// Copy one object verbatim, mask and marks included. The one home for "this
// object rides through untouched", which every act needs for what wasn't picked.
__device__ inline void gpu_warp_copy_object(gpu_fixed_object& dobj, const gpu_fixed_object& sobj, int lane) {
    constexpr int obj_bytes = (int)sizeof(gpu_fixed_object);
    int8_t* d = reinterpret_cast<int8_t*>(&dobj);
    const int8_t* s = reinterpret_cast<const int8_t*>(&sobj);
    for (int b = lane; b < obj_bytes; b += 32) d[b] = s[b];
}

__device__ void gpu_exec_interior_color_warp(gpu_exec_context& ctx, int8_t color, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    int outc = 0;
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj_src = src->objects[i];
        if (obj_src.width <= 0 || obj_src.height <= 0) continue;
        // Only what was picked has its interior filled; the rest ride through.
        if (!obj_src.selected) {
            gpu_warp_copy_object(res->objects[outc], obj_src, lane);
            ++outc;
            continue;
        }
        int f = color;
        if (color < 0) {                                  // first-color fill
            if (lane == 0) f = gpu_obj_first_color(obj_src);
            f = __shfl_sync(0xffffffffu, f, 0);
        }
        gpu_interior_paint_warp(obj_src, res->objects[outc], (int8_t)f, lane);
        ++outc;
    }
    if (lane == 0) res->count = outc;
    __syncwarp();
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

// Copy the first `n` objects verbatim and set the output count.
__device__ inline void gpu_warp_copy_objects(gpu_object_list* res, const gpu_object_list* src, int n, int lane) {
    for (int i = 0; i < n; ++i) gpu_warp_copy_object(res->objects[i], src->objects[i], lane);
    if (lane == 0) res->count = n;
    __syncwarp();
}

// Copy one object's header (everything before the packed mask: anchor, dims).
__device__ inline void gpu_warp_copy_header(const gpu_fixed_object& sobj, gpu_fixed_object& dobj, int lane) {
    constexpr int meta_bytes = (int)sizeof(gpu_fixed_object) - (int)sizeof(gpu_packed_mask);
    // The mask is last and the struct has no trailing padding, so "everything
    // before the mask" and "the struct minus the mask" are the same bytes. The
    // alignment object_cells asks for makes that a layout fact worth pinning.
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
        int idx = bb << 1;
        int dr = idx / GPU_MAX_GRID;
        int dc = idx - dr * GPU_MAX_GRID;            // even
        int8_t pv[2];
        #pragma unroll
        for (int t2 = 0; t2 < 2; ++t2) {
            int ddc = dc + t2;
            pv[t2] = (dr < nh && ddc < nw) ? src_pixel(dr, ddc) : (int8_t)GPU_TRANSPARENT;
        }
        dobj.mask.bytes[bb] = gpu_packed_mask::enc(pv[0]) | (gpu_packed_mask::enc(pv[1]) << 4);
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

// Warp-parallel Recolor: every non-transparent cell -> `color`. The per-cell
// recolor was a measured lane-0-serial hotspot (~70% of all kernel stall is on
// 1-lane instructions). Here the 32 lanes stripe the mask in WHOLE PACKED BYTES
// (2 cells/byte) so the 4-bit RMW has no shared-byte race (see gpu_packed_mask
// warning). Object metadata (x0/y0/dims) is unchanged, so we deep-copy each src
// object then overwrite only its mask bytes in parallel.
__device__ void gpu_exec_recolor_warp(gpu_exec_context& ctx, int8_t color, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    constexpr int mask_bytes = sizeof(gpu_packed_mask);
    uint8_t cn = gpu_packed_mask::enc(color);  // packed nibble for the new color
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& sobj = src->objects[i];
        gpu_fixed_object& obj = res->objects[i];
        // Color-source values (10 = rarest, 11 = most common) resolve per object:
        // lane 0 scans, everyone gets the nibble via shuffle.
        if (color >= GPU_MAX_COLORS) {
            int f = 0;
            if (lane == 0) f = gpu_obj_resolve_color(sobj, color);
            f = __shfl_sync(0xffffffffu, f, 0);
            cn = gpu_packed_mask::enc((int8_t)f);
        }
        gpu_warp_copy_header(sobj, obj, lane);
        // Selection is per object and the flag is the only truth: a selected
        // object recolors every one of its painted cells, an unselected object
        // rides through untouched. No selection in the chain means Recolor
        // changes nothing — acts NEVER fall back to "everything".
        // Stripe mask bytes: each lane owns whole bytes (2 cells) -> no RMW race.
        bool sel = sobj.selected;
        for (int b = lane; b < mask_bytes; b += 32) {
            uint8_t sb = sobj.mask.bytes[b];
            uint8_t lo = sb & 0xF, hi = (sb >> 4) & 0xF;
            uint8_t nlo = (lo == 0 || !sel) ? lo : cn;   // nibble 0 == GPU_TRANSPARENT
            uint8_t nhi = (hi == 0 || !sel) ? hi : cn;
            obj.mask.bytes[b] = (uint8_t)(nlo | (nhi << 4));
        }
    }
    if (lane == 0) res->count = n;
    __syncwarp();
}

// BidiColorMap[k]: bidirectional color swap. k in [0,44] indexes one of the 45
// unordered color pairs {a,b} with 0<=a<b<=9, enumerated row-major (k=0 -> {0,1},
// k=1 -> {0,2}, ... k=44 -> {8,9}). Every pixel of color a becomes b AND every
// pixel of color b becomes a, across all objects. Self-inverse (a swap), so it
// expresses involutive recolor tasks where Recolor/MapTo cannot: one node does
// a->b and b->a with identical program behavior, surviving examples that need
// the swap in either direction (e.g. 0d3d703e: chain Swap{1,5},{2,6},{3,4},{8,9}).
// Colors not in the pair are untouched, so disjoint swaps compose order-free.
__device__ gpu_object_list& gpu_exec_bidi_color_map(gpu_exec_context& ctx, const gpu_object_list& src, int8_t k) {
    auto& res = gpu_acquire(ctx);
    if (k < 0 || k > 44) { res = src; return res; }  // out of range -> passthrough

    // Decode k -> unordered pair (a,b), a<b. Row-major over {0..9}.
    int a = 0, b = 1, idx = 0;
    for (a = 0; a < GPU_MAX_COLORS; ++a) {
        bool found = false;
        for (b = a + 1; b < GPU_MAX_COLORS; ++b) {
            if (idx == k) { found = true; break; }
            ++idx;
        }
        if (found) break;
    }

    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked swaps colors; the rest ride through as they are.
        if (!obj.selected) { res.count++; continue; }
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c) {
                int8_t v = obj.mask[r][c];
                if (v == (int8_t)a)      obj.mask[r][c] = (int8_t)b;
                else if (v == (int8_t)b) obj.mask[r][c] = (int8_t)a;
            }
        res.count++;
    }
    return res;
}

// How many enclosed holes does this object have? A hole is a 4-connected
// component of transparent cells inside the object's bbox not reachable from the
// bbox edge (the outside-flood recipe of gpu_interior_paint_warp, run serially).
// One home for the recipe: SelectObjectsWithHoles asks here, and any future hole-keyed
// detector must ask here too, or the two answers can silently disagree.
__device__ int gpu_object_hole_count(const gpu_fixed_object& obj) {
    {
        int mw = min((int)obj.width, GPU_MAX_GRID), mh = min((int)obj.height, GPU_MAX_GRID);
        if (mw <= 0 || mh <= 0) return 0;

        // mark: 1 = wall (painted cell), 2 = outside (reached from bbox edge), 3 = counted hole.
        uint8_t mark[GPU_MAX_GRID][GPU_MAX_GRID];
        for (int r = 0; r < mh; ++r)
            for (int c = 0; c < mw; ++c)
                mark[r][c] = (obj.mask[r][c] != GPU_TRANSPARENT) ? 1 : 0;

        // Outside = monotonic sweeps from the bbox edge through transparent cells.
        for (bool changed = true; changed; ) {
            changed = false;
            for (int r = 0; r < mh; ++r)
                for (int c = 0; c < mw; ++c) {
                    if (mark[r][c]) continue;
                    bool edge = (r == 0 || r == mh - 1 || c == 0 || c == mw - 1);
                    bool nb = (r > 0      && mark[r - 1][c] == 2) ||
                              (r < mh - 1 && mark[r + 1][c] == 2) ||
                              (c > 0      && mark[r][c - 1] == 2) ||
                              (c < mw - 1 && mark[r][c + 1] == 2);
                    if (edge || nb) { mark[r][c] = 2; changed = true; }
                }
        }

        // Holes = remaining unmarked components; flood each with an explicit stack.
        int holes = 0;
        int16_t stack[GPU_MAX_GRID * GPU_MAX_GRID];
        for (int r = 0; r < mh; ++r)
            for (int c = 0; c < mw; ++c) {
                if (mark[r][c]) continue;
                ++holes;
                mark[r][c] = 3;
                int top = 0;
                stack[top++] = (int16_t)(r * GPU_MAX_GRID + c);
                while (top > 0) {
                    int p = stack[--top];
                    int pr = p / GPU_MAX_GRID, pc = p % GPU_MAX_GRID;
                    const int dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
                    for (int d = 0; d < 4; ++d) {
                        int nr = pr + dr[d], nc = pc + dc[d];
                        if (nr < 0 || nr >= mh || nc < 0 || nc >= mw || mark[nr][nc]) continue;
                        mark[nr][nc] = 3;
                        stack[top++] = (int16_t)(nr * GPU_MAX_GRID + nc);
                    }
                }
            }
        return holes;
    }
}

// SelectObjectsWithHoles[h]: mark the objects with exactly h enclosed holes;
// everything else rides through unmarked. Object-keyed (the `Objects` in the
// name says so). A selector answers "which" and nothing else — the acting is a
// separate op (Recolor), so the two choices cost a sum instead of the product
// the old fused HolesToColorMap[50] paid. Nothing is dropped, so several
// select/Recolor pairs chain order-free.
__device__ gpu_object_list& gpu_exec_select_objects_with_holes(gpu_exec_context& ctx, const gpu_object_list& src, int8_t h) {
    auto& res = gpu_acquire(ctx);
    res = src;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        if (gpu_object_hole_count(obj) == (int)h) obj.selected = true;
    }
    return res;
}

// SelectColor[c]: mark every object that contains color c; all other objects
// ride through unmarked. After a carve (Colors/Split) each object is single-
// color, so this picks exactly the color-c object(s). Sub-object reach is a
// carve away, never a cell mark.
__device__ gpu_object_list& gpu_exec_select_color(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color) {
    auto& res = gpu_acquire(ctx);
    res = src;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height && !obj.selected; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] == color) { obj.selected = true; break; }
    }
    return res;
}

// SelectAll: mark every object — the explicit "act on it all".
__device__ gpu_object_list& gpu_exec_select_all(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    res = src;
    for (int i = 0; i < res.count; ++i) res.objects[i].selected = true;
    return res;
}

// InvertSelection: flip every object's mark. With additive selectors (OR) this
// is NOT, and OR + NOT is the whole boolean algebra — "big AND red" is
// invert(invert(big) ∪ invert(red)) by De Morgan, no extra machinery.
__device__ gpu_object_list& gpu_exec_invert_selection(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    res = src;
    for (int i = 0; i < res.count; ++i)
        res.objects[i].selected = !res.objects[i].selected;
    return res;
}

// ShapeColorMap: repaint the main object to a color picked by a small glyph's
// SHAPE, using the puzzle-level table learned host-side (detect_shape_color_map).
// Removes nothing: key = smallest-bbox object, target = largest-bbox object; the
// target is recolored to table[hash(key)], everything else passes through. If the
// table is empty (rejected by the consistency gate) or the glyph's shape isn't in
// it, the op is a no-op. Learn and apply share one hash (shape_occupancy_hash, 10_dsl.cuh).
__device__ gpu_object_list& gpu_exec_shape_color_map(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    res = src;  // pass everything through; default is no-op
    if (ctx.shape_color_count <= 0 || src.count <= 0) return res;

    // key = smallest painted box, target = largest. Measured on the paint so an
    // untrimmed layer ranks by its glyph, the way inference learned it.
    int key = -1, tgt = -1, ka = 1 << 30, ta = -1;
    for (int i = 0; i < src.count; ++i) {
        const gpu_painted_box b = gpu_box_of_paint(src.objects[i]);
        if (b.width <= 0 || b.height <= 0) continue;
        int a = b.width * b.height;
        if (a < ka) { ka = a; key = i; }
        if (a > ta) { ta = a; tgt = i; }
    }
    if (key < 0 || tgt < 0 || key == tgt) return res;

    // Fingerprint the key glyph's occupancy (shared recipe in 10_dsl.cuh).
    uint32_t hsh = gpu_shape_hash_of_paint(src.objects[key]);

    // Look up the color; miss -> no-op.
    int8_t color = -1;
    for (int i = 0; i < ctx.shape_color_count && i < 16; ++i)
        if (ctx.shape_color_keys[i] == hsh) { color = ctx.shape_color_vals[i]; break; }
    if (color < 0) return res;

    // Recolor the target's painted cells.
    auto& robj = res.objects[tgt];
    for (int r = 0; r < robj.height; ++r)
        for (int c = 0; c < robj.width; ++c)
            if (robj.mask[r][c] != GPU_TRANSPARENT) robj.mask[r][c] = color;
    return res;
}

// ShapeToInteriorColorMap: per-object interior fill whose color is picked by the
// object's OWN shape via the puzzle-level table (detect_shape_to_interior_color_map).
// Sibling of ShapeColorMap, but applies per object (not single key->target) and
// paints the enclosed interior (not the body). For each object: lane 0 hashes the
// occupancy (shape_occupancy_hash, the same shared recipe ShapeColorMap uses), looks
// up the color, broadcasts it; a hit floods the interior with that color, a miss
// passes the object through unchanged. Inside-noise (a non-transparent pixel in
// the hole, e.g. a marker) is a wall to the flood, so it keeps its color -- the
// trailing Union re-stamping the separate marker object reproduces it either way.
__device__ void gpu_exec_shape_to_interior_color_map_warp(gpu_exec_context& ctx, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }
    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int n = min((int)src->count, GPU_KERNEL_MAX_OBJECTS);
    int outc = 0;
    for (int i = 0; i < n; ++i) {
        const gpu_fixed_object& obj_src = src->objects[i];
        if (obj_src.width <= 0 || obj_src.height <= 0) continue;
        // Only what was picked is looked up and filled; the rest ride through.
        if (!obj_src.selected) {
            gpu_warp_copy_object(res->objects[outc], obj_src, lane);
            ++outc;
            continue;
        }

        int f = -1;
        if (lane == 0) {
            // Fingerprint the object's occupancy (shared recipe in 10_dsl.cuh).
            uint32_t hsh = gpu_shape_hash_of_paint(obj_src);
            for (int k = 0; k < ctx.shape_color_count && k < 16; ++k)
                if (ctx.shape_color_keys[k] == hsh) { f = ctx.shape_color_vals[k]; break; }
        }
        f = __shfl_sync(0xffffffffu, f, 0);

        if (f < 0) {  // no mapping for this shape -> copy object through unchanged
            const int8_t* s = reinterpret_cast<const int8_t*>(&obj_src);
            int8_t* d = reinterpret_cast<int8_t*>(&res->objects[outc]);
            for (int b = lane; b < (int)sizeof(gpu_fixed_object); b += 32) d[b] = s[b];
            __syncwarp();
        } else {
            gpu_interior_paint_warp(obj_src, res->objects[outc], (int8_t)f, lane);
        }
        ++outc;
    }
    if (lane == 0) res->count = outc;
    __syncwarp();
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
    if (n <= 0) { res = src; return res; }

    // Chain order = bbox (x0, y0) ascending. Selection sort, n <= 32.
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

    // Compute bbox of non-transparent canvas pixels.
    int gw = ctx.output.width, gh = ctx.output.height;
    int tx0 = gw, ty0 = gh, tx1 = -1, ty1 = -1;
    for (int r = 0; r < gh; ++r) {
        for (int c = 0; c < gw; ++c) {
            if (ctx.output.data[r][c] != GPU_TRANSPARENT) {
                if (c < tx0) tx0 = c;
                if (r < ty0) ty0 = r;
                if (c > tx1) tx1 = c;
                if (r > ty1) ty1 = r;
            }
        }
    }
    bool empty_canvas = (tx1 < 0);

    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }

        if (!empty_canvas) {
            // Two snap candidates per axis (touch-but-don't-overlap, gap=0):
            //   dx0: obj.x1 lands at tx0 - 1  (obj sits left of content)
            //   dx1: obj.x0 lands at tx1 + 1  (obj sits right of content)
            int dx0 = tx0 - obj.x1 - 1;
            int dx1 = tx1 - obj.x0 + 1;
            int dx  = (abs(dx0) <= abs(dx1)) ? dx0 : dx1;
            int dy0 = ty0 - obj.y1 - 1;
            int dy1 = ty1 - obj.y0 + 1;
            int dy  = (abs(dy0) <= abs(dy1)) ? dy0 : dy1;

            // Move along the axis with the larger gap between bboxes.
            // Gap is negative when bboxes overlap on that axis.
            int x_gap = max(tx0 - obj.x1, obj.x0 - tx1);
            int y_gap = max(ty0 - obj.y1, obj.y0 - ty1);
            if (x_gap >= y_gap) dy = 0; else dx = 0;

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
    int order[GPU_KERNEL_MAX_OBJECTS];
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = 1; i < n; ++i) {
        int j = i;
        while (j > 0) {
            int a = order[j], b = order[j - 1];
            int la = rows ? src.objects[a].y0 : src.objects[a].x0;
            int lb = rows ? src.objects[b].y0 : src.objects[b].x0;
            if (la >= lb) break;
            order[j] = b; order[j - 1] = a; j--;
        }
    }
    int bands = 0, band_hi = -1;
    for (int m = 0; m < n; ++m) {
        int i = order[m];
        int lo = rows ? src.objects[i].y0 : src.objects[i].x0;
        int hi = rows ? src.objects[i].y1 : src.objects[i].x1;
        // Overlap with the current band must cover half the span's own length.
        int need = (hi - lo + 2) / 2;
        if (bands == 0 || band_hi - lo + 1 < need) { bands++; band_hi = hi; }
        else if (hi > band_hi) band_hi = hi;
        band_of[i] = (int8_t)(bands - 1);
    }
    return bands;
}

// Layout[k]: arrange the objects on k-sized graph paper. Overlapping vertical
// spans share a row, overlapping horizontal spans share a column; rows sort
// top-to-bottom, columns left-to-right. Each object's pixels are copied
// UNRESIZED into its own k x k cell (cell origin = (col*k, row*k)), clipped to
// the cell. Result is one (cols*k) x (rows*k) object at the origin — grid-
// aligned by construction, so ScaleDown[k] reads exactly one object per cell.
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
    res = src;
    int n = min((int)res.count, GPU_KERNEL_MAX_OBJECTS);
    if (n <= 0) return res;
    int8_t band_of[GPU_KERNEL_MAX_OBJECTS];
    gpu_layout_bands(res, n, rows, band_of);
    for (int i = 0; i < n; ++i)
        if (band_of[i] == which) res.objects[i].selected = true;
    return res;
}

__device__ gpu_object_list& gpu_exec_layout(gpu_exec_context& ctx, const gpu_object_list& src, int k) {
    auto& res = gpu_acquire(ctx);
    int n = min((int)src.count, GPU_KERNEL_MAX_OBJECTS);
    if (n <= 0 || k <= 0) return res;

    int8_t row_of[GPU_KERNEL_MAX_OBJECTS], col_of[GPU_KERNEL_MAX_OBJECTS];
    int rows = gpu_layout_bands(src, n, true, row_of);
    int cols = gpu_layout_bands(src, n, false, col_of);
    int nw = min(cols * k, GPU_MAX_GRID);
    int nh = min(rows * k, GPU_MAX_GRID);

    gpu_fixed_object& obj = res.objects[0];
    obj.x0 = 0; obj.y0 = 0;
    obj.x1 = (int16_t)(nw - 1); obj.y1 = (int16_t)(nh - 1);
    obj.width = (int16_t)nw; obj.height = (int16_t)nh;
    obj.mask.clear();

    for (int i = 0; i < n; ++i) {
        const auto& s = src.objects[i];
        int ox = col_of[i] * k, oy = row_of[i] * k;
        for (int r = 0; r < s.height && r < k; ++r)
            for (int c = 0; c < s.width && c < k; ++c) {
                int gr = oy + r, gc = ox + c;
                if (gr < nh && gc < nw && s.mask[r][c] != GPU_TRANSPARENT)
                    obj.mask[gr][gc] = s.mask[r][c];
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
        int dx = 0, dy = 0;
        switch (dir) {
        case 0: dy = -obj.height; break; case 1: dx = obj.width; break;
        case 2: dy = obj.height; break;  case 3: dx = -obj.width; break;
        case 4: dx = obj.width; dy = obj.height; break;
        case 5: dx = -obj.width; dy = obj.height; break;
        case 6: dx = obj.width; dy = -obj.height; break;
        case 7: dx = -obj.width; dy = -obj.height; break;
        }
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

    int obj_count = src->count;
    bool is_turn = (type == node_type::TurnRight || type == node_type::TurnLeft);

    for (int i = 0; i < obj_count && i < GPU_KERNEL_MAX_OBJECTS; ++i) {
        auto& sobj = src->objects[i];
        int w = sobj.width, h = sobj.height;
        int nw = is_turn ? min(h, GPU_MAX_GRID) : min(w, GPU_MAX_GRID);
        int nh = is_turn ? min(w, GPU_MAX_GRID) : min(h, GPU_MAX_GRID);

        auto& obj = res->objects[i];
        // Only what was picked turns; the rest ride through as they are. Nothing
        // picked means this op changes nothing — no falling back to "everything".
        if (!sobj.selected) {
            gpu_warp_copy_object(obj, sobj, lane);
            if (lane == 0) res->count = i + 1;
            __syncwarp();
            continue;
        }
        // Dest cell (dr, dc) pulls from its source cell via the inverse map.
        gpu_warp_remap_object(sobj, obj, nw, nh, lane, [&](int dr, int dc) {
            int sr, sc;
            switch (type) {
            case node_type::TurnRight:      sr = h - 1 - dc; sc = dr;         break;
            case node_type::TurnLeft:       sr = dc;         sc = w - 1 - dr; break;
            case node_type::FlipHorizontal: sr = dr;         sc = w - 1 - dc; break;
            default:                        sr = h - 1 - dr; sc = dc;         break;  // FlipVertical
            }
            return (int8_t)sobj.mask[sr][sc];
        });

        // Lane 0 updates metadata; merged sync with next iteration's metadata-copy sync
        if (lane == 0) {
            obj.width = (int16_t)nw; obj.height = (int16_t)nh;
            obj.x1 = obj.x0 + (int16_t)nw - 1; obj.y1 = obj.y0 + (int16_t)nh - 1;
            res->count = i + 1;
        }
        __syncwarp();
    }
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

__device__ gpu_object_list& gpu_exec_slam_row(gpu_exec_context& ctx, const gpu_object_list& src, int16_t dx) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }
        if (dx == 0) { res.count++; continue; }
        // Clamp to the mask's actual extent: bbox dims can exceed 30 (Shift/Union can
        // push an object's bbox past the grid), but the mask is only 30x30 and `nm` is
        // a 30x30 local array — looping past 30 walks off it => illegal memory access.
        int w = min((int)obj.width, GPU_MAX_GRID), h = min((int)obj.height, GPU_MAX_GRID);
        // Global scratch (coalesced), not per-thread local. Packed like the mask
        // it becomes, so the result moves across as a copy, not a conversion.
        // Single-threaded here, so sharing a byte between two cells is safe.
        gpu_packed_mask& nm = *reinterpret_cast<gpu_packed_mask*>(ctx.op_scratch);
        nm.clear();
        if (dx > 0) {
            for (int r = 0; r < h; ++r) {
                int right_wall = w - 1;
                for (int c = w - 1; c >= 0; --c) {
                    if (obj.mask[r][c] == GPU_TRANSPARENT) continue;
                    int target = c + dx;
                    if (target > right_wall) target = right_wall;
                    if (target >= 0 && target < w) nm[r][target] = obj.mask[r][c];
                    right_wall = target - 1;
                }
            }
        } else {
            for (int r = 0; r < h; ++r) {
                int left_wall = 0;
                for (int c = 0; c < w; ++c) {
                    if (obj.mask[r][c] == GPU_TRANSPARENT) continue;
                    int target = c + dx;
                    if (target < left_wall) target = left_wall;
                    if (target >= 0 && target < w) nm[r][target] = obj.mask[r][c];
                    left_wall = target + 1;
                }
            }
        }
        obj.mask = nm;
        res.count++;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_slam_column(gpu_exec_context& ctx, const gpu_object_list& src, int16_t dy) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked moves; the rest ride through where they are.
        // Nothing picked means this op does nothing -- acts never fall back to all.
        if (!obj.selected) { res.count++; continue; }
        if (dy == 0) { res.count++; continue; }
        int w = min((int)obj.width, GPU_MAX_GRID), h = min((int)obj.height, GPU_MAX_GRID);  // clamp: bbox can exceed mask's 30x30 -> nm OOB
        // Global scratch (coalesced), not per-thread local. Packed like the mask
        // it becomes, so the result moves across as a copy, not a conversion.
        // Single-threaded here, so sharing a byte between two cells is safe.
        gpu_packed_mask& nm = *reinterpret_cast<gpu_packed_mask*>(ctx.op_scratch);
        nm.clear();
        if (dy > 0) {
            for (int c = 0; c < w; ++c) {
                int bw = h - 1;
                for (int r = h - 1; r >= 0; --r) {
                    if (obj.mask[r][c] == GPU_TRANSPARENT) continue;
                    int target = r + dy;
                    if (target > bw) target = bw;
                    if (target >= 0 && target < h) nm[target][c] = obj.mask[r][c];
                    bw = target - 1;
                }
            }
        } else {
            for (int c = 0; c < w; ++c) {
                int tw = 0;
                for (int r = 0; r < h; ++r) {
                    if (obj.mask[r][c] == GPU_TRANSPARENT) continue;
                    int target = r + dy;
                    if (target < tw) target = tw;
                    if (target >= 0 && target < h) nm[target][c] = obj.mask[r][c];
                    tw = target + 1;
                }
            }
        }
        obj.mask = nm;
        res.count++;
    }
    return res;
}


// Warp-parallel scale up: all 32 lanes cooperate on pixel expansion.
// Pattern mirrors gpu_exec_transform_warp: all-lane acquire, stripe copy with
// selective mask clear, pixel loop striped by lane, divide-free pixel advance.
__device__ void gpu_exec_scale_up_warp(gpu_exec_context& ctx, int8_t factor, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int obj_count = src->count;

    // factor <= 1: just copy objects unchanged.
    if (factor <= 1) {
        gpu_warp_copy_objects(res, src, min(obj_count, GPU_KERNEL_MAX_OBJECTS), lane);
        return;
    }

    for (int i = 0; i < obj_count && i < GPU_KERNEL_MAX_OBJECTS; ++i) {
        auto& sobj = src->objects[i];
        int ow = sobj.width, oh = sobj.height;
        int nw = min(ow * factor, GPU_MAX_GRID);
        int nh = min(oh * factor, GPU_MAX_GRID);

        auto& obj = res->objects[i];
        // Only what was picked is resized; the rest ride through untouched.
        if (!sobj.selected) {
            gpu_warp_copy_object(obj, sobj, lane);
            if (lane == 0) res->count = i + 1;
            __syncwarp();
            continue;
        }
        // Dest cell pulls its source via /factor (nearest-neighbour upscale).
        gpu_warp_remap_object(sobj, obj, nw, nh, lane, [&](int dr, int dc) {
            int sr = dr / factor, sc = dc / factor;
            return (sr < oh && sc < ow) ? (int8_t)sobj.mask[sr][sc] : (int8_t)GPU_TRANSPARENT;
        });

        if (lane == 0) {
            obj.width = (int16_t)nw; obj.height = (int16_t)nh;
            obj.x1 = obj.x0 + (int16_t)nw - 1; obj.y1 = obj.y0 + (int16_t)nh - 1;
            res->count = i + 1;
        }
        __syncwarp();
    }
}

// Warp-parallel scale down: each lane handles one output pixel's majority vote.
__device__ void gpu_exec_scale_down_warp(gpu_exec_context& ctx, int8_t factor, int lane) {
    gpu_object_list* src = gpu_current_objects(ctx);
    if (!src) { ctx.current_objects_idx = gpu_exec_context::CURRENT_OBJECTS_NONE; return; }

    gpu_object_list* res = &gpu_acquire_and_publish(ctx);
    int obj_count = src->count;

    if (factor <= 1) {
        gpu_warp_copy_objects(res, src, min(obj_count, GPU_KERNEL_MAX_OBJECTS), lane);
        return;
    }

    for (int i = 0; i < obj_count && i < GPU_KERNEL_MAX_OBJECTS; ++i) {
        auto& sobj = src->objects[i];
        int ow = sobj.width, oh = sobj.height;
        int nw = max((ow + factor - 1) / factor, 1);
        int nh = max((oh + factor - 1) / factor, 1);

        auto& obj = res->objects[i];
        // Only what was picked is resized; the rest ride through untouched.
        if (!sobj.selected) {
            gpu_warp_copy_object(obj, sobj, lane);
            if (lane == 0) res->count = i + 1;
            __syncwarp();
            continue;
        }
        // Each dest cell is the majority vote of its factor x factor source block.
        gpu_warp_remap_object(sobj, obj, nw, nh, lane, [&](int dr, int dc) {
            return gpu_majority_vote(sobj.mask, dr, dc, factor, oh, ow);
        });

        if (lane == 0) {
            obj.width = (int16_t)nw; obj.height = (int16_t)nh;
            obj.x1 = obj.x0 + (int16_t)nw - 1; obj.y1 = obj.y0 + (int16_t)nh - 1;
            res->count = i + 1;
        }
        __syncwarp();
    }
}


// SelectObjectsContainingColor[color]: mark the objects that have at least one
// pixel of `color`; the rest ride through unmarked. Object-keyed (the `Objects`
// in the name says so). Lossless — keeping only them is
// `SelectObjectsContainingColor[c] InvertSelection DeleteSelection`.
__device__ gpu_object_list& gpu_exec_select_objects_containing_color(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color) {
    auto& res = gpu_acquire(ctx);
    res = src;
    for (int i = 0; i < res.count; ++i) {
        auto& obj = res.objects[i];
        for (int r = 0; r < obj.height && !obj.selected; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] == color) { obj.selected = true; break; }
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_add_color(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        res.objects[res.count] = src.objects[i];
        auto& obj = res.objects[res.count];
        // Only what was picked gains the color; the rest ride through as they are.
        if (!obj.selected) { res.count++; continue; }
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c)
                if (obj.mask[r][c] == GPU_TRANSPARENT) obj.mask[r][c] = color;
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
                bool edge = false;
                for (int dr = -1; dr <= 1 && !edge; ++dr)
                    for (int dc = -1; dc <= 1 && !edge; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        int nr = r + dr, nc = c + dc;
                        if (nr < 0 || nc < 0 || nr >= obj_src.height || nc >= obj_src.width ||
                            obj_src.mask[nr][nc] == GPU_TRANSPARENT) edge = true;
                    }
                if (!edge) obj.mask[r][c] = GPU_TRANSPARENT;
            }
        res.count++;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_fit(gpu_exec_context& ctx, const gpu_object_list& src) {
    int gw = ctx.input.width, gh = ctx.input.height;
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& obj_src = src.objects[i];
        if (obj_src.width <= 0 || obj_src.height <= 0) continue;
        // Only what was picked is stretched to the grid; the rest ride through.
        if (!obj_src.selected) { res.objects[res.count++] = obj_src; continue; }
        int sw = obj_src.width, sh = obj_src.height;
        int nw = min((int)gw, GPU_MAX_GRID), nh = min((int)gh, GPU_MAX_GRID);
        gpu_fixed_object obj;
        // Same object, same mark -- a fresh one would start unmarked.
        obj.selected = obj_src.selected;
        obj.x0 = 0; obj.y0 = 0;
        obj.width = (int16_t)nw; obj.height = (int16_t)nh;
        obj.x1 = (int16_t)(nw - 1); obj.y1 = (int16_t)(nh - 1);
        obj.mask.clear();
        for (int r = 0; r < nh; ++r)
            for (int c = 0; c < nw; ++c)
                obj.mask[r][c] = obj_src.mask[r * sh / nh][c * sw / nw];
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
        gpu_packed_mask& nm = *reinterpret_cast<gpu_packed_mask*>(ctx.op_scratch);
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
    const gpu_packed_mask& data;
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
        // One finder for everyone (4_objects.hpp). Loose evidence on purpose:
        // naming the unit to continue with is a different question from proving
        // a grid repeats, and a motif that fits once and a bit is still the
        // motif -- 017c7c7b pair 1 is 6 tall with a 4-tall unit.
        const gpu_object_view view{ obj_src.mask, obj_src.width, obj_src.height };
        arc::repeat_periods(view, GPU_TRANSPARENT, best_w, best_h, /*must_fit_twice=*/false);
        gpu_fixed_object obj;
        // The unit is the same object, so it keeps its mark. A fresh
        // gpu_fixed_object starts unmarked and would drop the pick silently.
        obj.selected = obj_src.selected;
        obj.x0 = obj_src.x0; obj.y0 = obj_src.y0;
        obj.width = (int16_t)best_w; obj.height = (int16_t)best_h;
        obj.x1 = obj.x0 + (int16_t)best_w - 1; obj.y1 = obj.y0 + (int16_t)best_h - 1;
        obj.mask.clear();
        // Reconstruct each tile cell by majority vote over all source cells in
        // its period class, ignoring holes -- so a hole in the top-left tile is
        // filled from a repetition elsewhere. Clean patterns vote unanimously,
        // reproducing the old top-left copy.
        for (int r = 0; r < best_h; ++r)
            for (int c = 0; c < best_w; ++c) {
                int counts[GPU_MAX_COLORS]; memset(counts, 0, sizeof(counts));
                int best_v = GPU_TRANSPARENT, best_n = 0;
                for (int sr = r; sr < obj_src.height; sr += best_h)
                    for (int sc = c; sc < obj_src.width; sc += best_w) {
                        int8_t v = obj_src.mask[sr][sc];
                        if (!arc::is_color(v)) continue;       // skip holes/transparent
                        if (++counts[v] > best_n) { best_n = counts[v]; best_v = v; }
                    }
                obj.mask[r][c] = (int8_t)best_v;
            }
        res.objects[res.count++] = obj;
    }
    return res;
}

// VoteDenoise: noise-tolerant majority vote across repeating structure.
// Period per axis: a spacing qualifies when the structure it aligns agrees
// >=70% (pairs with both cells non-background), and the winner is the
// qualifier with the best agreements-minus-disagreements margin, where
// aligning structure against background counts as disagreement -- a true
// period only loses margin to noise, a false one bleeds it everywhere.
// Rows/columns whose cells match their residue-class majority <70% are
// outside the pattern and become background. Inside, each cell becomes the
// most common value among in-pattern cells whole periods away on both axes;
// ties keep the cell. No period on either axis -> object passes unchanged.
__device__ gpu_object_list& gpu_exec_vote_denoise(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& s = src.objects[i];
        if (s.width <= 0 || s.height <= 0) continue;
        // Only what was picked is denoised; the rest ride through as they are.
        if (!s.selected) { res.objects[res.count++] = s; continue; }
        gpu_fixed_object obj = s;
        const int w = s.width, h = s.height;

        // Background: puzzle-level detected color, else most frequent in the mask.
        int8_t bg = ctx.background_color;
        if (bg < 0) {
            int freq[GPU_MAX_COLORS]; memset(freq, 0, sizeof(freq));
            for (int r = 0; r < h; ++r)
                for (int c = 0; c < w; ++c) {
                    int8_t v = s.mask[r][c];
                    if (arc::is_color(v)) ++freq[v];
                }
            int bn = -1;
            for (int v = 0; v < GPU_MAX_COLORS; ++v)
                if (freq[v] > bn) { bn = freq[v]; bg = (int8_t)v; }
        }

        // A cell emptied by a background removal upstream is still the ground:
        // read it as bg, so the op behaves the same whether the background was
        // deleted or left in place.
        auto ground_at = [&](int r, int c) -> int8_t {
            int8_t v = s.mask[r][c];
            return v == GPU_TRANSPARENT ? bg : v;
        };

        // Period per axis (0 = no repeat found).
        int px = 0, py = 0;
        for (int axis = 0; axis < 2; ++axis) {
            const int dim = axis ? h : w;
            int best_margin = 0; bool found = false;
            for (int p = 2; p <= dim / 2; ++p) {
                int agree = 0, dis = 0, both = 0;
                for (int r = 0; r < (axis ? h - p : h); ++r)
                    for (int c = 0; c < (axis ? w : w - p); ++c) {
                        int8_t a = ground_at(r, c);
                        int8_t b = axis ? ground_at(r + p, c) : ground_at(r, c + p);
                        if (a == bg && b == bg) continue;
                        if (a != bg && b != bg) {
                            ++both;
                            if (a == b) ++agree; else ++dis;
                        } else ++dis;
                    }
                if (agree < 8 || agree * 10 < both * 7) continue;
                int margin = agree - dis;
                if (!found || margin > best_margin) {
                    found = true; best_margin = margin;
                    (axis ? py : px) = p;
                }
            }
        }
        if (px == 0 && py == 0) { res.objects[res.count++] = obj; continue; }

        // In-pattern lines: a line is inside iff >=70% of its cells equal the
        // per-cell majority of its residue class along that axis alone -- the
        // voted template is noise-free, so a noisy copy only pays for its own
        // noise, while an off-pattern line disagrees almost everywhere.
        bool col_in[GPU_MAX_GRID], row_in[GPU_MAX_GRID];
        for (int c = 0; c < w; ++c) col_in[c] = true;
        for (int r = 0; r < h; ++r) row_in[r] = true;
        if (px > 0)
            for (int c = 0; c < w; ++c) {
                int agree = 0;
                for (int r = 0; r < h; ++r) {
                    // One bucket past the colors, because a transparent cell can
                    // win this vote and become the template.
                    int counts[GPU_MAX_COLORS + 1]; memset(counts, 0, sizeof(counts));
                    int8_t tmpl = GPU_TRANSPARENT; int tn = 0;
                    for (int sc = c % px; sc < w; sc += px) {
                        int8_t v = ground_at(r, sc);
                        int slot = arc::is_color(v) ? v : GPU_MAX_COLORS;
                        if (++counts[slot] > tn) { tn = counts[slot]; tmpl = v; }
                    }
                    if (ground_at(r, c) == tmpl) ++agree;
                }
                col_in[c] = agree * 10 >= h * 7;
            }
        if (py > 0)
            for (int r = 0; r < h; ++r) {
                int agree = 0;
                for (int c = 0; c < w; ++c) {
                    // One bucket past the colors, same reason as above.
                    int counts[GPU_MAX_COLORS + 1]; memset(counts, 0, sizeof(counts));
                    int8_t tmpl = GPU_TRANSPARENT; int tn = 0;
                    for (int sr = r % py; sr < h; sr += py) {
                        int8_t v = ground_at(sr, c);
                        int slot = arc::is_color(v) ? v : GPU_MAX_COLORS;
                        if (++counts[slot] > tn) { tn = counts[slot]; tmpl = v; }
                    }
                    if (ground_at(r, c) == tmpl) ++agree;
                }
                row_in[r] = agree * 10 >= w * 7;
            }

        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c) {
                if (!row_in[r] || !col_in[c]) { obj.mask[r][c] = bg; continue; }
                int counts[GPU_MAX_COLORS]; memset(counts, 0, sizeof(counts));
                for (int sr = (py > 0 ? r % py : r); sr < h; sr += (py > 0 ? py : h))
                    for (int sc = (px > 0 ? c % px : c); sc < w; sc += (px > 0 ? px : w)) {
                        if (!row_in[sr] || !col_in[sc]) continue;
                        int8_t v = ground_at(sr, sc);
                        if (arc::is_color(v)) ++counts[v];
                    }
                int8_t own = ground_at(r, c);
                int8_t best_v = own;
                int best_n = arc::is_color(own) ? counts[own] : 0;
                for (int v = 0; v < GPU_MAX_COLORS; ++v)
                    if (counts[v] > best_n) { best_n = counts[v]; best_v = (int8_t)v; }
                obj.mask[r][c] = best_v;
            }
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
    // full canvas on the tiled axis (period = `period`), preserving the
    // source's own bbox on the fixed axis.
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
            int idx = bb << 1;
            int r = idx / GPU_MAX_GRID;
            int c = idx - r * GPU_MAX_GRID;              // even
            int8_t pv[2];
            #pragma unroll
            for (int t2 = 0; t2 < 2; ++t2) {
                int cc = c + t2;
                int8_t v = GPU_TRANSPARENT;
                if (r < oh && cc < ow) {
                    int pr = horizontal ? r  : (r  % per);
                    int pc = horizontal ? (cc % per) : cc;
                    if (pr >= 0 && pr < sh && pc >= 0 && pc < sw) v = obj.mask[pr][pc];
                }
                pv[t2] = v;
            }
            t.mask.bytes[bb] = gpu_packed_mask::enc(pv[0]) | (gpu_packed_mask::enc(pv[1]) << 4);
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
            int diag = (variant == 0) ? (wr + wc) : (wr - wc);
            int idx = diag % color_count;
            if (idx < 0) idx += color_count;
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
            int diag = (variant == 0) ? (r + c) : (r - c);
            int idx = diag % color_count;
            if (idx < 0) idx += color_count;
            if (idx >= 0 && idx < GPU_MAX_COLORS) out_obj.mask[r][c] = mapping[idx];
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
        // Need two marks to define a period; otherwise pass through unchanged.
        if (src.count < 2) {
            for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i)
                res.objects[res.count++] = src.objects[i];
            return res;
        }
        // Read the lattice from the marks themselves: spacing = smallest gap between
        // them, color cycle length = number of marks. A given mark recurs every
        // (count * spacing); stamping each mark at that period reconstructs the full
        // cycle, since extra / same-color marks just land on the in-between lattice
        // points. 2 marks -> 2*gap (the alternating pair); N marks -> N*gap.
        int min_gap = GPU_MAX_GRID;
        for (int i = 0; i < src.count; ++i)
            for (int j = i + 1; j < src.count; ++j) {
                int g = src.objects[i].y0 - src.objects[j].y0;
                if (g < 0) g = -g;
                if (g > 0 && g < min_gap) min_gap = g;
            }
        int period = (min_gap < GPU_MAX_GRID) ? src.count * min_gap : 0;
        if (period <= 0) {
            for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i)
                res.objects[res.count++] = src.objects[i];
            return res;
        }
        // One full-canvas output object per mark, stamped every `period` rows from
        // its own y0 (rel<0 keeps the leading rows blank).
        for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
            const auto& s = src.objects[i];
            if (s.width <= 0 || s.height <= 0) continue;
            gpu_fixed_object& out = res.objects[res.count];
            out.x0 = 0; out.y0 = 0;
            out.x1 = (int16_t)(w - 1); out.y1 = (int16_t)(h - 1);
            out.width = (int16_t)w; out.height = (int16_t)h;
            out.mask.clear();
            bool any = false;
            for (int r = 0; r < h; ++r) {
                int rel = r - s.y0;
                if (rel < 0) continue;
                int sr = rel % period;
                if (sr >= s.height) continue;
                for (int c = 0; c < w && c < s.width; ++c) {
                    int8_t v = s.mask[sr][c];
                    if (v != GPU_TRANSPARENT) { out.mask[r][c] = v; any = true; }
                }
            }
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

// 8 unit directions: up, down, left, right, then the 4 diagonals.
__device__ const int8_t GPU_DIR8[8][2] = {
    { 0,-1}, { 0, 1}, {-1, 0}, { 1, 0},
    {-1,-1}, { 1,-1}, {-1, 1}, { 1, 1}
};

// For each object: find the center cell, draw rays in 4 directions
// (GPU_DIR8[dir_first..dir_first+3]) for `radius` steps using the center
// cell's color. Output is one combined object per input.
__device__ gpu_object_list& gpu_exec_extend_rays(gpu_exec_context& ctx, const gpu_object_list& src, int8_t radius, int dir_first) {
    auto& res = gpu_acquire(ctx);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const gpu_fixed_object& obj = src.objects[i];
        if (obj.width <= 0 || obj.height <= 0) continue;
        // Only what was picked grows rays; the rest ride through as they are.
        if (!obj.selected) { res.objects[res.count++] = obj; continue; }

        // Find center cell: non-transparent pixel closest to bbox center.
        int bcx = (obj.x0 + obj.x1) / 2 - obj.x0;
        int bcy = (obj.y0 + obj.y1) / 2 - obj.y0;
        int best_r = -1, best_c = -1, best_dist = 0x7FFF;
        int8_t center_color = 0;
        for (int r = 0; r < obj.height; ++r)
            for (int c = 0; c < obj.width; ++c) {
                if (obj.mask[r][c] == GPU_TRANSPARENT) continue;
                int d = abs(r - bcy) + abs(c - bcx);
                if (d < best_dist) { best_dist = d; best_r = r; best_c = c; center_color = obj.mask[r][c]; }
            }
        if (best_r < 0) continue;

        // Expand the original object's bbox by radius on all sides.
        int x0 = obj.x0 - radius;
        int y0 = obj.y0 - radius;
        int x1 = obj.x1 + radius;
        int y1 = obj.y1 + radius;
        int nw = min(x1 - x0 + 1, GPU_MAX_GRID);
        int nh = min(y1 - y0 + 1, GPU_MAX_GRID);

        gpu_fixed_object& out = res.objects[res.count++];
        // The rays belong to the object that grew them, so they carry its mark.
        out.selected = obj.selected;
        out.x0 = (int16_t)x0; out.y0 = (int16_t)y0;
        out.x1 = (int16_t)x1; out.y1 = (int16_t)y1;
        out.width = (int16_t)nw; out.height = (int16_t)nh;
        out.mask.clear();

        // RAYS ONLY: do NOT copy the seed/body pixels -- output is just the rays.
        // Callers that need the originals must re-stamp them (e.g. with a
        // Union). A single-pixel seed thus contributes no cell of its own, only
        // its 4 arms. (radius 0 => empty output.)

        // Draw 4 rays from center cell outward.
        int cr = best_r + radius, cc = best_c + radius;
        for (int d = dir_first; d < dir_first + 4; ++d) {
            int dx = GPU_DIR8[d][0], dy = GPU_DIR8[d][1];
            for (int k = 1; k <= radius; ++k) {
                int r = cr + dy * k, c = cc + dx * k;
                if (r >= 0 && r < GPU_MAX_GRID && c >= 0 && c < GPU_MAX_GRID)
                    out.mask[r][c] = center_color;
            }
        }
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
// Slam to the grid edge: slide each object along dir as far as the grid
// allows, ignoring paint. Shape unchanged.
// (ExtendTo grows a trail; Slam squashes inside the box; this one TRAVELS.)
__device__ gpu_object_list& gpu_exec_move_to_edge(gpu_exec_context& ctx, const gpu_object_list& src, int8_t dir) {
    auto& res = gpu_acquire(ctx);
    int di = (dir >= 0 && dir < 8) ? dir : 0;
    int dx = GPU_DIR8[di][0], dy = GPU_DIR8[di][1];
    res.count = min((int)src.count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < res.count; ++i) {
        res.objects[i] = src.objects[i];
        auto& o = res.objects[i];
        // Only what was picked travels; the rest stay where they are.
        if (!o.selected) continue;
        int w = min((int)o.width, GPU_MAX_GRID), h = min((int)o.height, GPU_MAX_GRID);
        int steps = 0;
        for (; steps <= GPU_MAX_GRID; ++steps) {
            bool blocked = false;
            for (int r = 0; r < h && !blocked; ++r)
                for (int c = 0; c < w; ++c) {
                    if (o.mask[r][c] == GPU_TRANSPARENT) continue;
                    int gr = o.y0 + r + (steps + 1) * dy;
                    int gc = o.x0 + c + (steps + 1) * dx;
                    if (gr < 0 || gr >= ctx.output.height || gc < 0 || gc >= ctx.output.width) { blocked = true; break; }
                }
            if (blocked) break;
        }
        o.y0 = (int16_t)(o.y0 + steps * dy);
        o.x0 = (int16_t)(o.x0 + steps * dx);
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_extend_to(gpu_exec_context& ctx, const gpu_object_list& src, int8_t dir) {
    auto& res = gpu_acquire(ctx);
    int di = (dir >= 0 && dir < 8) ? dir : 0;
    int ddx = GPU_DIR8[di][0], ddy = GPU_DIR8[di][1];
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
        int8_t col = gpu_obj_first_color(o);

        int min_x = 1 << 30, min_y = 1 << 30, max_x = -(1 << 30), max_y = -(1 << 30);
        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) { (void)v;
            int sx = o.x0 + c, sy = o.y0 + r;
            if (sx < min_x) min_x = sx; if (sx > max_x) max_x = sx;
            if (sy < min_y) min_y = sy; if (sy > max_y) max_y = sy;
            int x = sx, y = sy;
            for (;;) {
                x += ddx; y += ddy;
                if (x < 0 || x >= gw || y < 0 || y >= gh) break;
                if (occ[y][x] && !gpu_obj_contains(o, x, y)) break;
                if (gpu_obj_contains(o, x, y)) continue;
                if (x < min_x) min_x = x; if (x > max_x) max_x = x;
                if (y < min_y) min_y = y; if (y > max_y) max_y = y;
            }
        }
        if (max_x < min_x) continue;
        int nw = min(max_x - min_x + 1, GPU_MAX_GRID);
        int nh = min(max_y - min_y + 1, GPU_MAX_GRID);

        gpu_fixed_object& out = res.objects[res.count];
        out.x0 = (int16_t)min_x; out.y0 = (int16_t)min_y;
        out.x1 = (int16_t)(min_x + nw - 1); out.y1 = (int16_t)(min_y + nh - 1);
        out.width = (int16_t)nw; out.height = (int16_t)nh;
        out.mask.clear();

        GPU_DEV_FOR_EACH_PIXEL(o, r, c, v) {
            int sx = o.x0 + c, sy = o.y0 + r;
            int lx = sx - min_x, ly = sy - min_y;
            if (lx >= 0 && lx < nw && ly >= 0 && ly < nh) out.mask[ly][lx] = v; // body keeps its color
            int x = sx, y = sy;
            for (;;) {
                x += ddx; y += ddy;
                if (x < 0 || x >= gw || y < 0 || y >= gh) break;
                if (occ[y][x] && !gpu_obj_contains(o, x, y)) break;
                if (gpu_obj_contains(o, x, y)) continue;
                int px = x - min_x, py = y - min_y;
                if (px >= 0 && px < nw && py >= 0 && py < nh) out.mask[py][px] = col;
            }
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
__device__ gpu_object_list& gpu_exec_read_under(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);
    int ow = ctx.output.width, oh = ctx.output.height;
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& o = src.objects[i];
        gpu_fixed_object out = o;
        bool any = false;
        for (int r = 0; r < o.height; ++r)
            for (int c = 0; c < o.width; ++c) {
                if (out.mask[r][c] == GPU_TRANSPARENT) continue;
                int x = o.x0 + c, y = o.y0 + r;
                bool on = (x >= 0 && x < ow && y >= 0 && y < oh);
                int8_t cv = on ? ctx.output.data[y][x] : GPU_TRANSPARENT;
                out.mask[r][c] = cv;
                if (cv != GPU_TRANSPARENT) any = true;
            }
        if (any) res.objects[res.count++] = out;
    }
    return res;
}

__device__ gpu_object_list& gpu_exec_mask(gpu_exec_context& ctx, const gpu_object_list& src, int8_t color, bool keep_match) {
    auto& res = gpu_acquire(ctx);
    int ow = ctx.output.width, oh = ctx.output.height;
    bool any_color = (color == GPU_COLOR_ANY);
    for (int i = 0; i < src.count && res.count < GPU_KERNEL_MAX_OBJECTS; ++i) {
        const auto& o = src.objects[i];
        gpu_fixed_object out = o;
        bool any = false;
        for (int r = 0; r < o.height; ++r)
            for (int c = 0; c < o.width; ++c) {
                if (out.mask[r][c] == GPU_TRANSPARENT) continue;
                int x = o.x0 + c, y = o.y0 + r;
                bool on = (x >= 0 && x < ow && y >= 0 && y < oh);
                int8_t cv = on ? ctx.output.data[y][x] : GPU_TRANSPARENT;
                bool match = any_color ? (cv != GPU_TRANSPARENT) : (cv == color);
                if (match == keep_match) any = true;
                else out.mask[r][c] = GPU_TRANSPARENT;
            }
        if (any) res.objects[res.count++] = out;
    }
    return res;
}

// Target output dims from the size rule. Single source of truth, used both at
// kernel start to size `output` and by RestoreOrientation to recover the target
// orientation after a rotate has mutated `output`. Cases named by the host rule.
__device__ void gpu_rule_output_dims(int8_t rt, const int16_t* rp,
                                     int in_w, int in_h, int& ow, int& oh) {
    if (rt == arc::output_dimension_rule::Fixed) {
        ow = min((int)rp[0], GPU_MAX_GRID);
        oh = min((int)rp[1], GPU_MAX_GRID);
    } else if (rt == arc::output_dimension_rule::Scale) {
        ow = min((int)(in_w * rp[2] / rp[3]), GPU_MAX_GRID);
        oh = min((int)(in_h * rp[4] / rp[5]), GPU_MAX_GRID);
    } else if (rt == arc::output_dimension_rule::BoundingBox) {
        // max grid now, cropped after the program runs
        ow = GPU_MAX_GRID;
        oh = GPU_MAX_GRID;
    } else if (rt == arc::output_dimension_rule::NotInferredYet) {
        // nobody filled the rule in; 0x0 makes every score wrong on purpose,
        // so a missing inference shows up instead of passing as Identity
        ow = 0;
        oh = 0;
    } else {                             // Identity: same as input
        ow = min(in_w, GPU_MAX_GRID);
        oh = min(in_h, GPU_MAX_GRID);
    }
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
        bool flip = (ctx.input.width > ctx.input.height);
        transpose_objs   = flip;
        transpose_canvas = (ctx.output.width > ctx.output.height);  // ensure portrait
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
        gpu_packed_mask& od = ctx.output.data;
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

    int tcx = (tpl.x0 + tpl.x1) / 2;
    int tcy = (tpl.y0 + tpl.y1) / 2;
    int gw = ctx.output.width, gh = ctx.output.height;

    for (int s = 0; s < ctx.seed_objs.count; ++s) {
        if (res.count >= GPU_KERNEL_MAX_OBJECTS) break;
        const auto& seed = ctx.seed_objs.objects[s];
        int scx = (seed.x0 + seed.x1) / 2;
        int scy = (seed.y0 + seed.y1) / 2;
        int dxi = scx - tcx;
        int dyi = scy - tcy;
        if (dxi == 0 && dyi == 0) continue;

        int nx0 = tpl.x0, ny0 = tpl.y0;
        if (dxi > 0)      nx0 = seed.x0;
        else if (dxi < 0) nx0 = seed.x1 - (tpl.width - 1);
        if (dyi > 0)      ny0 = seed.y0;
        else if (dyi < 0) ny0 = seed.y1 - (tpl.height - 1);

        int dx = nx0 - tpl.x0;
        int dy = ny0 - tpl.y0;
        if (dx == 0 && dy == 0) continue;

        int8_t scolor = gpu_obj_first_color(seed);

        for (int k = 1; k <= GPU_MAX_GRID; ++k) {
            if (res.count >= GPU_KERNEL_MAX_OBJECTS) break;
            int nx = tpl.x0 + k * dx;
            int ny = tpl.y0 + k * dy;
            bool on_grid = (nx + tpl.width > 0) && (nx < gw)
                        && (ny + tpl.height > 0) && (ny < gh);
            if (!on_grid) break;

            gpu_fixed_object& out = res.objects[res.count];
            out.x0 = (int16_t)nx; out.y0 = (int16_t)ny;
            out.x1 = (int16_t)(nx + tpl.width - 1);
            out.y1 = (int16_t)(ny + tpl.height - 1);
            out.width = tpl.width; out.height = tpl.height;
            out.mask.clear();
            int8_t paint = (scolor >= 0) ? scolor : gpu_obj_first_color(tpl);
            for (int r = 0; r < tpl.height; ++r)
                for (int c = 0; c < tpl.width; ++c)
                    if (tpl.mask[r][c] != GPU_TRANSPARENT)
                        out.mask[r][c] = paint;
            ++res.count;
        }
    }
    return res;
}

// --- SnapToAnchors helpers ---------------------------------------------------
// 8 dihedral transforms of a relative offset (r,c): i&3 = number of 90deg CW
// rotations ((r,c)->(c,-r)); i>=4 adds a vertical reflection (negate row).
// Mirrors the coordinate math in gpu_exec_transform_warp, without the kernel/
// pool plumbing — used here to try all rigid orientations cheaply in one op.
__device__ inline void snap_dihedral(int i, int r, int c, int& outr, int& outc) {
    int rr = r, cc = c;
    int rot = i & 3;
    for (int k = 0; k < rot; ++k) { int nr = cc, nc = -rr; rr = nr; cc = nc; }
    if (i >= 4) rr = -rr;
    outr = rr; outc = cc;
}

// SnapToAnchors: rigidly stamp each "body" onto its color-coded anchor cluster.
//   - body color  = most frequent present color (background must already be
//                   stripped upstream, e.g. via Foreground; only GPU_TRANSPARENT
//                   counts as empty here).
//   - bodies      = mixed-color 8-connected components that contain the body color.
//   - key-pixels  = a body's cells whose color != body color.
//   - loose anchors = cells in components WITHOUT the body color (the scattered
//                   single cells), bucketed by color.
// For each body, find the (anchor-key, loose-anchor, dihedral orientation) that
// maps the most key-pixels onto same-colored loose anchors (exact when all align;
// best-effort otherwise), then stamp the transformed full body there. Output =
// the stamped bodies only; originals + loose anchors are consumed.
__device__ gpu_object_list& gpu_exec_snap_to_anchors(gpu_exec_context& ctx, const gpu_object_list& src) {
    auto& res = gpu_acquire(ctx);

    gpu_fixed_grid temp;
    // Full-ceiling canvas (see gpu_exec_split): don't crop a grown stream to input dims.
    gpu_paint_objects(src, GPU_MAX_GRID, GPU_MAX_GRID, temp);
    int W = temp.width, H = temp.height;
    if (W <= 0 || H <= 0) return res;

    // 1. body color = most frequent present (non-transparent) color.
    int freq[GPU_MAX_COLORS]; for (int i = 0; i < GPU_MAX_COLORS; ++i) freq[i] = 0;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            int8_t v = temp.data[r][c];
            if (arc::is_color(v)) freq[v]++;
        }
    int bodycol = -1, bestfreq = 0;
    for (int i = 0; i < GPU_MAX_COLORS; ++i) if (freq[i] > bestfreq) { bestfreq = freq[i]; bodycol = i; }
    if (bodycol < 0) return res;

    // 2. mixed-color 8-connected component labeling over non-transparent cells.
    constexpr int MAXC = 64;
    int8_t label[GPU_MAX_GRID][GPU_MAX_GRID];
    for (int r = 0; r < GPU_MAX_GRID; ++r)
        for (int c = 0; c < GPU_MAX_GRID; ++c) label[r][c] = -1;
    bool comp_has_body[MAXC];
    int16_t qr[GPU_GRID_CELLS], qc[GPU_GRID_CELLS];
    int ncomp = 0;
    const int8_t ndr[] = {-1,-1,-1, 0, 0, 1, 1, 1};
    const int8_t ndc[] = {-1, 0, 1,-1, 1,-1, 0, 1};
    for (int r = 0; r < H && ncomp < MAXC; ++r) {
        for (int c = 0; c < W && ncomp < MAXC; ++c) {
            if (temp.data[r][c] == GPU_TRANSPARENT || label[r][c] != -1) continue;
            int id = ncomp++;
            bool hb = false;
            int head = 0, tail = 0;
            qr[tail] = (int16_t)r; qc[tail] = (int16_t)c; tail++;
            label[r][c] = (int8_t)id;
            while (head < tail) {
                int cr = qr[head], cc = qc[head]; head++;
                if (temp.data[cr][cc] == bodycol) hb = true;
                for (int d = 0; d < 8; ++d) {
                    int nr = cr + ndr[d], nc = cc + ndc[d];
                    if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                    if (temp.data[nr][nc] == GPU_TRANSPARENT || label[nr][nc] != -1) continue;
                    label[nr][nc] = (int8_t)id;
                    if (tail < GPU_GRID_CELLS) { qr[tail] = (int16_t)nr; qc[tail] = (int16_t)nc; tail++; }
                }
            }
            comp_has_body[id] = hb;
        }
    }

    // 3. loose anchors = cells in components WITHOUT the body color. Build a
    //    position lookup (loose_color) plus per-color candidate lists.
    constexpr int MAX_ANCHORS_PER_COLOR = 16;
    int8_t loose_color[GPU_MAX_GRID][GPU_MAX_GRID];
    int16_t anc_r[GPU_MAX_COLORS][MAX_ANCHORS_PER_COLOR];
    int16_t anc_c[GPU_MAX_COLORS][MAX_ANCHORS_PER_COLOR];
    int anc_n[GPU_MAX_COLORS]; for (int i = 0; i < GPU_MAX_COLORS; ++i) anc_n[i] = 0;
    for (int r = 0; r < GPU_MAX_GRID; ++r)
        for (int c = 0; c < GPU_MAX_GRID; ++c) loose_color[r][c] = GPU_TRANSPARENT;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            int8_t v = temp.data[r][c];
            if (v == GPU_TRANSPARENT) continue;
            int id = label[r][c];
            if (id < 0 || comp_has_body[id]) continue;   // body cell, not loose
            loose_color[r][c] = v;
            if (arc::is_color(v) && anc_n[v] < MAX_ANCHORS_PER_COLOR) {
                anc_r[v][anc_n[v]] = (int16_t)r; anc_c[v][anc_n[v]] = (int16_t)c; anc_n[v]++;
            }
        }

    // 4. per body: gather key-pixels, find best rigid placement, stamp.
    constexpr int MAXK = 12;
    for (int id = 0; id < ncomp; ++id) {
        if (!comp_has_body[id]) continue;
        int kcolor[MAXK], krow[MAXK], kcol_[MAXK], nk = 0;
        for (int r = 0; r < H && nk < MAXK; ++r)
            for (int c = 0; c < W && nk < MAXK; ++c)
                if (label[r][c] == id) {
                    int8_t v = temp.data[r][c];
                    if (v != bodycol && v != GPU_TRANSPARENT) {
                        kcolor[nk] = v; krow[nk] = r; kcol_[nk] = c; nk++;
                    }
                }
        if (nk == 0) continue;

        int best_score = 0, best_ak = -1, best_ar = 0, best_ac = 0, best_ori = 0;
        for (int ak = 0; ak < nk; ++ak) {
            int col = kcolor[ak], br = krow[ak], bc = kcol_[ak];
            for (int ci = 0; ci < anc_n[col]; ++ci) {
                int ar = anc_r[col][ci], ac = anc_c[col][ci];
                for (int ori = 0; ori < 8; ++ori) {
                    int score = 0;
                    for (int j = 0; j < nk; ++j) {
                        int dr, dc; snap_dihedral(ori, krow[j] - br, kcol_[j] - bc, dr, dc);
                        int tr = ar + dr, tc = ac + dc;
                        if (tr >= 0 && tr < H && tc >= 0 && tc < W &&
                            loose_color[tr][tc] == kcolor[j]) score++;
                    }
                    if (score > best_score) {
                        best_score = score; best_ak = ak;
                        best_ar = ar; best_ac = ac; best_ori = ori;
                    }
                }
            }
        }
        if (best_ak < 0 || best_score < 1) continue;

        // 5. stamp: transform every body cell about the anchor key.
        int br = krow[best_ak], bc = kcol_[best_ak];
        int x0 = GPU_MAX_GRID, y0 = GPU_MAX_GRID, x1 = -1, y1 = -1;
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                if (label[r][c] == id) {
                    int dr, dc; snap_dihedral(best_ori, r - br, c - bc, dr, dc);
                    int tr = best_ar + dr, tc = best_ac + dc;
                    if (tr < 0 || tr >= GPU_MAX_GRID || tc < 0 || tc >= GPU_MAX_GRID) continue;
                    if (tc < x0) x0 = tc; if (tr < y0) y0 = tr;
                    if (tc > x1) x1 = tc; if (tr > y1) y1 = tr;
                }
        if (x1 < 0) continue;
        if (res.count >= GPU_KERNEL_MAX_OBJECTS) break;
        auto& obj = res.objects[res.count];
        obj.x0 = (int16_t)x0; obj.y0 = (int16_t)y0;
        obj.x1 = (int16_t)x1; obj.y1 = (int16_t)y1;
        obj.width = (int16_t)(x1 - x0 + 1); obj.height = (int16_t)(y1 - y0 + 1);
        obj.mask.clear();
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                if (label[r][c] == id) {
                    int dr, dc; snap_dihedral(best_ori, r - br, c - bc, dr, dc);
                    int tr = best_ar + dr, tc = best_ac + dc;
                    if (tr < 0 || tr >= GPU_MAX_GRID || tc < 0 || tc >= GPU_MAX_GRID) continue;
                    obj.mask[tr - y0][tc - x0] = temp.data[r][c];
                }
        res.count++;

        // Consume the matched anchors so another body can't reuse this cluster.
        for (int j = 0; j < nk; ++j) {
            int dr, dc; snap_dihedral(best_ori, krow[j] - br, kcol_[j] - bc, dr, dc);
            int tr = best_ar + dr, tc = best_ac + dc;
            if (tr >= 0 && tr < H && tc >= 0 && tc < W && loose_color[tr][tc] == kcolor[j]) {
                loose_color[tr][tc] = GPU_TRANSPARENT;
                int cc = kcolor[j];
                for (int q = 0; q < anc_n[cc]; ++q)
                    if (anc_r[cc][q] == tr && anc_c[cc][q] == tc) {
                        anc_r[cc][q] = anc_r[cc][anc_n[cc]-1];
                        anc_c[cc][q] = anc_c[cc][anc_n[cc]-1];
                        anc_n[cc]--; break;
                    }
            }
        }
    }
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
            if (prev == node_type::Input || prev == node_type::Seeds ||
                prev == node_type::InputWithoutBackground) {
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
            case node_type::InputWithoutBackground:
                                result = &gpu_exec_input_without_background(ctx); break;
            case node_type::Output:         result = &gpu_exec_output(ctx); break;
            case node_type::Split:          result = last ? &gpu_exec_split(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Colors:         result = last ? &gpu_exec_colors(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SelectMostCommonColor: result = last ? &gpu_exec_select_most_common_color(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SelectLargestObject:  result = last ? &gpu_exec_select_object_by_size(ctx, *last, value, true) : &gpu_acquire(ctx); break;
            case node_type::SelectSmallestObject: result = last ? &gpu_exec_select_object_by_size(ctx, *last, value, false) : &gpu_acquire(ctx); break;
            case node_type::SortBySize:  result = last ? &gpu_exec_sort_by_size(ctx, *last, value == 0 ? 1 : 0) : &gpu_acquire(ctx); break;
            case node_type::SizeRankColorMap: result = last ? &gpu_exec_size_rank_color_map(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Bridge:         result = last ? &gpu_exec_bridge(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Partition:      result = last ? &gpu_exec_partition(ctx, *last) : &gpu_acquire(ctx); break;
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
            case node_type::SelectObjectsWithHoles: result = last ? &gpu_exec_select_objects_with_holes(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectColor:    result = last ? &gpu_exec_select_color(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectAll:      result = last ? &gpu_exec_select_all(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::InvertSelection: result = last ? &gpu_exec_invert_selection(ctx, *last) : &gpu_acquire(ctx); break;
            // DeleteSelection handled in the warp-parallel branch above; reaching
            // here means has_current_objects was false -> empty result.
            case node_type::DeleteSelection: result = &gpu_acquire(ctx); break;
            case node_type::Shift:          result = last ? &gpu_exec_shift(ctx, *last, value) : &gpu_acquire(ctx); break;
            // MoveRow/MoveColumn handled in the warp-parallel branch above; here means !has_current_objects.
            case node_type::MoveRow:
            case node_type::MoveColumn:     result = &gpu_acquire(ctx); break;
            case node_type::SlamRow:        result = last ? &gpu_exec_slam_row(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SlamColumn:     result = last ? &gpu_exec_slam_column(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::AddColor:       result = last ? &gpu_exec_add_color(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Fit:            result = last ? &gpu_exec_fit(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Skew:           result = last ? &gpu_exec_skew(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Trim:           result = last ? &gpu_exec_trim(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::DiagTile:       result = last ? &gpu_exec_diag_tile(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Fractal:        result = last ? &gpu_exec_fractal(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Tile:           result = last ? &gpu_exec_tile(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Pattern:        result = last ? &gpu_exec_pattern(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::ProjectOnSeeds: result = last ? &gpu_exec_project_on_seeds(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::IntersectionList:  result = last ? &gpu_exec_intersect_list(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::UnionList:         result = last ? &gpu_exec_union_list(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::Layout:         result = last ? &gpu_exec_layout(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::Origin:         result = last ? &gpu_exec_origin(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::AlignToCanvas:  result = last ? &gpu_exec_align_to_canvas(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::OverlapCorner:  result = last ? &gpu_exec_overlap_corner(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::SnapToAnchors:  result = last ? &gpu_exec_snap_to_anchors(ctx, *last) : &gpu_acquire(ctx); break;
            case node_type::ShapeColorMap:  result = last ? &gpu_exec_shape_color_map(ctx, *last) : &gpu_acquire(ctx); break;
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
            case node_type::SelectObjectsContainingColor: result = last ? &gpu_exec_select_objects_containing_color(ctx, *last, value) : &gpu_acquire(ctx); break;
            case node_type::SelectRow:      result = last ? &gpu_exec_select_band(ctx, *last, (int8_t)value, true) : &gpu_acquire(ctx); break;
            case node_type::SelectColumn:   result = last ? &gpu_exec_select_band(ctx, *last, (int8_t)value, false) : &gpu_acquire(ctx); break;
            // NormalizeOrientation / RestoreOrientation are handled in the warp-parallel
            // branch above; reaching here means has_current_objects was false -> empty result.
            case node_type::NormalizeOrientation:
            case node_type::RestoreOrientation: result = &gpu_acquire(ctx); break;
            default: result = &gpu_acquire(ctx); break;
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

// MSVC C4324: intentional padding; see object_cells in 1_readings.hpp.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
struct gpu_example_template {
    gpu_fixed_grid input;
    gpu_object_list seed_objs;
    int8_t output_colors_by_size[GPU_MAX_COLORS];
    int8_t output_colors_by_size_count;
    uint32_t shape_color_keys[16];
    int8_t   shape_color_vals[16];
    int8_t   shape_color_count;
    int8_t background_color;
    int8_t output_background_color;
    int8_t distance_maps[GPU_MAX_COLORS][GPU_MAX_GRID][GPU_MAX_GRID];
    gpu_packed_mask target_grid;
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
static int8_t*           d_scratch_prog_lengths = nullptr;
static int8_t*           d_scratch_node_counts = nullptr;
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
static int8_t*           d_exec_prog_lengths[2] = {nullptr, nullptr};
static int8_t*           d_exec_node_counts[2] = {nullptr, nullptr};
static float*            d_exec_scores[2] = {nullptr, nullptr};
static int8_t*           d_exec_solved[2] = {nullptr, nullptr};
static int8_t*           d_exec_output_grids[2] = {nullptr, nullptr};
static int8_t*           d_exec_output_dims[2] = {nullptr, nullptr};
static uint64_t*         d_exec_output_hashes[2] = {nullptr, nullptr};
static gpu_score_terms*  d_exec_score_terms[2] = {nullptr, nullptr};
static int               d_exec_capacity[2] = {0, 0};

static void resize_exec_buffers(int job_count, int& capacity,
                                gpu_exec_context*& ctxs, gpu_flat_op*& progs,
                                int8_t*& prog_lens, int8_t*& node_cnts,
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
    chk(cudaMalloc(&prog_lens, (size_t)job_count * sizeof(int8_t)), "prog_lens", (size_t)job_count);
    chk(cudaMalloc(&node_cnts, (size_t)job_count * sizeof(int8_t)), "node_cnts", (size_t)job_count);
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
    h_templates.assign(count, gpu_example_template{});
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
        memcpy(tmpl.output_colors_by_size, ex.output_colors_by_size,
               sizeof(tmpl.output_colors_by_size));
        tmpl.output_colors_by_size_count = ex.output_colors_by_size_count;
        memcpy(tmpl.shape_color_keys, ex.shape_color_keys, sizeof(tmpl.shape_color_keys));
        memcpy(tmpl.shape_color_vals, ex.shape_color_vals, sizeof(tmpl.shape_color_vals));
        tmpl.shape_color_count = ex.shape_color_count;
        tmpl.background_color = ex.background_color;
        tmpl.output_background_color = ex.output_background_color;

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
// So (32,21) ≈ 80 regs / ~44% occupancy is the peak. Register spills land in fast
// shared memory via enable_smem_spilling (below), which is what makes the cap a win.
__global__ void __launch_bounds__(32, 21) dsl_exec_and_score_kernel_v2(
    gpu_exec_context* __restrict__ contexts,
    const gpu_example_template* __restrict__ templates,
    const gpu_flat_op* __restrict__ programs,
    const int8_t* __restrict__ program_lengths,
    const int8_t* __restrict__ node_counts,
    const int8_t* __restrict__ rule_type,     // single value
    const int16_t* __restrict__ rule_params,  // 6 values
    float* __restrict__ scores,
    int8_t* __restrict__ solved_out,
    int8_t* __restrict__ output_grids,
    int8_t* __restrict__ output_dims,
    int active_examples,
    int job_count)
{
    // CUDA 13.0+ shared-memory register spilling: send register spills (forced by the
    // __launch_bounds__ occupancy cap above) to on-chip shared memory instead of local/
    // DRAM. We qualify: CUDA 13.2, whole-program compile, static smem only, launch bounds
    // defined. Makes the occupancy cap far cheaper (spills no longer hit DRAM latency).
    asm volatile (".pragma \"enable_smem_spilling\";");

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
            for (int i = 0; i < 16; ++i) {
                ctx.shape_color_keys[i] = tmpl.shape_color_keys[i];
                ctx.shape_color_vals[i] = tmpl.shape_color_vals[i];
            }
            ctx.shape_color_count = tmpl.shape_color_count;
            ctx.background_color = tmpl.background_color;
            ctx.output_background_color = tmpl.output_background_color;
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
            dst_out[i] = gpu_packed_mask::empty_byte();
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

    // The closing frame: whatever the program never painted is the canvas the
    // output figures sit on. Applied here, after the crop, so transparency still
    // means "nothing here" for every op that ran. No elected canvas -> untouched.
    if (ctx.output_background_color >= 0) {
        // ONE PACKED BYTE per lane (2 same-row cells, 30 per row so a byte never
        // crosses a row) rather than one cell: a lane-striped cell write would
        // read-modify-write a byte its neighbour lane also owns.
        for (int b = lane; b < (int)sizeof(ctx.output.data.bytes); b += 32) {
            const int idx = b << 1;
            const int r = idx / GPU_MAX_GRID, c = idx - r * GPU_MAX_GRID;  // c even
            if (r >= ctx.output.height) continue;
            for (int t = 0; t < 2; ++t)
                if (c + t < ctx.output.width &&
                    ctx.output.data[r][c + t] == GPU_TRANSPARENT)
                    ctx.output.data[r][c + t] = ctx.output_background_color;
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
    const int8_t* __restrict__ node_counts,
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
        dout[i] = gpu_packed_mask::empty_byte();
    if (lane == 0) {
        ctx.pool_head = 0;
        ctx.orientation_flipped = 0;
        ctx.background_color = background;
        ctx.output_background_color = background;
        ctx.output.width = ctx.input.width;
        ctx.output.height = ctx.input.height;
        // Nothing is being painted, so the size rule is "same as the input".
        ctx.rule_type = (int8_t)arc::output_dimension_rule::Identity;
        for (int k = 0; k < 6; ++k) ctx.rule_params[k] = 0;
    }
    __syncwarp();

    gpu_flat_op prog[2];
    prog[0].type = node_type::Input;  prog[0].value = 0;
    prog[1].type = op;                prog[1].value = variant;
    gpu_object_list& list = gpu_eval_program_warp(ctx, prog, 2, lane);

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
    cudaMemcpyAsync(d_exec_prog_lengths[slot], batch.program_lengths.data(), n * sizeof(int8_t),                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_exec_node_counts[slot],  batch.node_counts.data(),     n * sizeof(int8_t),                    cudaMemcpyHostToDevice, s);

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
    cudaMemcpyAsync(d_scratch_prog_lengths, batch.program_lengths.data(), n * sizeof(int8_t),                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_scratch_node_counts,  batch.node_counts.data(),     n * sizeof(int8_t),                    cudaMemcpyHostToDevice, s);

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

    // Working output-canvas dims after execution (reflect rotate ops). width and
    // height are adjacent int16_t at the end of gpu_fixed_grid, so one copy gets both.
    if (out_canvas_w || out_canvas_h) {
        int16_t wh[2] = {0, 0};
        cudaMemcpyAsync(wh,
                        reinterpret_cast<char*>(d_ctx) + offsetof(gpu_exec_context, output)
                            + offsetof(gpu_fixed_grid, width),
                        2 * sizeof(int16_t), cudaMemcpyDeviceToHost, s);
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
