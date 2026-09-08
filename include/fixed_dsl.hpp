#pragma once

// This header is included from solver_beam.hpp AFTER environment.hpp and cuda_dsl.cuh,
// so arc:: types (grid, example, output_dimension_rule, flat_program, etc.) are already available.
// Do NOT include environment.hpp here to avoid circular dependencies.
//
// Contains: fixed-size grid/object types, segmentation helpers for puzzle upload,
// and the fixed_init_context used by build_example_upload. All DSL execution is
// handled by the GPU kernel in cuda_solver.cu.

#include "cuda_dsl.cuh"
#include <cstring>
#include <algorithm>

namespace arc {

// --- Fixed-size grid (int8_t pixels, no heap) ---
struct fixed_grid {
    int8_t data[GPU_MAX_GRID][GPU_MAX_GRID];
    int16_t width = 0, height = 0;

    void clear() {
        memset(data, GPU_TRANSPARENT, sizeof(data));
        width = height = 0;
    }

    void clear_data() {
        memset(data, GPU_TRANSPARENT, sizeof(data));
    }
};

// --- Convert from environment grid (int pixels) to fixed_grid (int8_t pixels) ---
static inline void grid_to_fixed(const int src[][30], int w, int h, fixed_grid& dst) {
    dst.width = (int16_t)w;
    dst.height = (int16_t)h;
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            dst.data[r][c] = (int8_t)src[r][c];
}

// --- Flood fill for segment_objects (8-connected) ---
// Uses a fixed-size queue instead of vector. Max 900 pixels on a 30x30 grid.
// TODO: remove once fixed_init_context is replaced by GPU-side segmentation.
static inline void fixed_flood_fill(
    const fixed_grid& g, int r, int c, int8_t seed_color,
    bool visited[][GPU_MAX_GRID], fixed_object& obj)
{
    static constexpr int MAX_QUEUE = GPU_MAX_GRID * GPU_MAX_GRID;
    int16_t queue_r[MAX_QUEUE], queue_c[MAX_QUEUE];
    int head = 0, tail = 0;

    queue_r[tail] = (int16_t)r;
    queue_c[tail] = (int16_t)c;
    tail++;
    visited[r][c] = true;

    while (head < tail) {
        int cr = queue_r[head], cc = queue_c[head];
        head++;

        if (cc < obj.x0) obj.x0 = (int16_t)cc;
        if (cr < obj.y0) obj.y0 = (int16_t)cr;
        if (cc > obj.x1) obj.x1 = (int16_t)cc;
        if (cr > obj.y1) obj.y1 = (int16_t)cr;

        static constexpr int8_t dr[] = {-1,-1,-1, 0, 0, 1, 1, 1};
        static constexpr int8_t dc[] = {-1, 0, 1,-1, 1,-1, 0, 1};
        for (int d = 0; d < 8; ++d) {
            int nr = cr + dr[d], nc = cc + dc[d];
            if (nr < 0 || nr >= g.height || nc < 0 || nc >= g.width) continue;
            if (g.data[nr][nc] != seed_color || visited[nr][nc]) continue;
            if (tail >= MAX_QUEUE) continue;
            visited[nr][nc] = true;
            queue_r[tail] = (int16_t)nr;
            queue_c[tail] = (int16_t)nc;
            tail++;
        }
    }

    obj.width = (int16_t)std::min((int)(obj.x1 - obj.x0 + 1), (int)GPU_MAX_GRID);
    obj.height = (int16_t)std::min((int)(obj.y1 - obj.y0 + 1), (int)GPU_MAX_GRID);
    memset(obj.mask, GPU_TRANSPARENT, sizeof(obj.mask));

    for (int i = 0; i < tail; ++i) {
        int pr = queue_r[i] - obj.y0, pc = queue_c[i] - obj.x0;
        if (pr < 0 || pr >= obj.height || pc < 0 || pc >= obj.width) continue;
        obj.mask[pr][pc] = g.data[queue_r[i]][queue_c[i]];
    }
}

// Scan an object's mask for the first non-transparent pixel color.
// Returns GPU_TRANSPARENT if the mask is empty.
static inline int8_t fixed_obj_first_color(const fixed_object& o) {
    for (int r = 0; r < o.height; ++r)
        for (int c = 0; c < o.width; ++c)
            if (o.mask[r][c] != GPU_TRANSPARENT) return o.mask[r][c];
    return GPU_TRANSPARENT;
}

// --- segment_objects: connected-component segmentation (8-connected) ---
static inline void fixed_segment_objects(const fixed_grid& g, fixed_object_list& out, int8_t bg_color = GPU_TRANSPARENT) {
    out.clear();
    if (g.width <= 0 || g.height <= 0) return;

    bool visited[GPU_MAX_GRID][GPU_MAX_GRID]{};
    for (int r = 0; r < g.height; ++r) {
        for (int c = 0; c < g.width; ++c) {
            if (g.data[r][c] == bg_color || visited[r][c]) continue;

            fixed_object obj;
            obj.x0 = (int16_t)c; obj.y0 = (int16_t)r;
            obj.x1 = (int16_t)c; obj.y1 = (int16_t)r;
            fixed_flood_fill(g, r, c, g.data[r][c], visited, obj);

            if (!out.push(obj)) return; // full
        }
    }
}

// --- segment_by_color: one object per color ---
static inline void fixed_segment_by_color(const fixed_grid& g, fixed_object_list& out, int8_t bg_color = GPU_TRANSPARENT) {
    out.clear();
    if (g.width <= 0 || g.height <= 0) return;

    for (int color = 0; color < GPU_MAX_COLORS; ++color) {
        if (color == bg_color) continue;

        int16_t x0 = (int16_t)g.width, y0 = (int16_t)g.height;
        int16_t x1 = -1, y1 = -1;
        bool found = false;

        for (int r = 0; r < g.height; ++r) {
            for (int c = 0; c < g.width; ++c) {
                if (g.data[r][c] != color) continue;
                found = true;
                if (c < x0) x0 = (int16_t)c;
                if (r < y0) y0 = (int16_t)r;
                if (c > x1) x1 = (int16_t)c;
                if (r > y1) y1 = (int16_t)r;
            }
        }
        if (!found) continue;

        fixed_object obj;
        obj.x0 = x0; obj.y0 = y0; obj.x1 = x1; obj.y1 = y1;
        obj.width = (int16_t)std::min((int)(x1 - x0 + 1), (int)GPU_MAX_GRID);
        obj.height = (int16_t)std::min((int)(y1 - y0 + 1), (int)GPU_MAX_GRID);
        memset(obj.mask, GPU_TRANSPARENT, sizeof(obj.mask));

        for (int r = 0; r < g.height; ++r) {
            for (int c = 0; c < g.width; ++c) {
                if (g.data[r][c] != color) continue;
                int pr = r - y0, pc = c - x0;
                if (pr >= 0 && pr < obj.height && pc >= 0 && pc < obj.width)
                    obj.mask[pr][pc] = (int8_t)color;
            }
        }

        if (!out.push(obj)) return; // full
    }
}

// =====================================================================
// Execution context (fixed-size, no heap) — used by fixed_init_context
// and build_example_upload to prepare per-example data for GPU upload.
// =====================================================================

struct fixed_exec_context {
    fixed_grid input;
    fixed_grid output;
    fixed_object_list seed_objs;

    // Distance maps for scoring (from example)
    int8_t distance_maps[GPU_MAX_COLORS][GPU_MAX_GRID][GPU_MAX_GRID];
};

static inline void sort_objects_by_area_asc(fixed_object_list& list) {
    for (int i = 1; i < list.count; ++i) {
        int j = i;
        while (j > 0) {
            int a0 = list.objects[j-1].width * list.objects[j-1].height;
            int a1 = list.objects[j].width * list.objects[j].height;
            if (a1 < a0) { std::swap(list.objects[j], list.objects[j-1]); j--; }
            else break;
        }
    }
}

// Convert fixed_object_list (CPU, 900 objs) -> gpu_object_list (GPU kernel, 64 objs)
static inline gpu_object_list to_gpu_object_list(const fixed_object_list& src) {
    gpu_object_list dst;
    dst.count = std::min(src.count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < dst.count; ++i) {
        auto& d = dst.objects[i];
        const auto& s = src.objects[i];
        d.x0 = s.x0; d.y0 = s.y0; d.x1 = s.x1; d.y1 = s.y1;
        d.width = s.width; d.height = s.height;
        gpu_mask_from_grid(d.mask, s.mask);  // pack int8 grid -> 4-bit mask
    }
    return dst;
}

// =====================================================================
// Bridge functions (depend on environment.hpp types: flat_program, example, output_dimension_rule, grid)
// These are placed at the end because fixed_dsl.hpp is included after environment.hpp.
// =====================================================================

// --- Convert flat_program (node_type) to gpu_flat_op array ---
static inline int convert_program_to_gpu_ops(const flat_program& program, gpu_flat_op* out) {
    for (int i = 0; i < program.count; ++i) {
        out[i].type = program.ops[i].type;
        out[i].value = static_cast<int8_t>(program.ops[i].value);
    }
    return program.count;
}

// --- Initialize fixed_exec_context from an example and puzzle data ---
static inline void fixed_init_context(
    fixed_exec_context& ctx,
    const example& ex,
    const output_dimension_rule& rule,
    const bool* invariant_colors)
{
    // Convert input grid
    grid_to_fixed(ex.input.data, ex.input.width, ex.input.height, ctx.input);

    // Determine output dimensions
    if (rule.Chosen == output_dimension_rule::Fixed) {
        ctx.output.width = (int16_t)rule.width;
        ctx.output.height = (int16_t)rule.height;
    } else if (rule.Chosen == output_dimension_rule::Scale) {
        ctx.output.width = (int16_t)(ex.input.width * rule.width_numerator / rule.width_denominator);
        ctx.output.height = (int16_t)(ex.input.height * rule.height_numerator / rule.height_denominator);
    } else if (rule.Chosen == output_dimension_rule::BoundingBox) {
        ctx.output.width = GPU_MAX_GRID;
        ctx.output.height = GPU_MAX_GRID;
    } else {
        ctx.output.width = (int16_t)std::min(ex.input.width, (int)GPU_MAX_GRID);
        ctx.output.height = (int16_t)std::min(ex.input.height, (int)GPU_MAX_GRID);
    }
    ctx.output.width = (int16_t)std::min((int)ctx.output.width, (int)GPU_MAX_GRID);
    ctx.output.height = (int16_t)std::min((int)ctx.output.height, (int)GPU_MAX_GRID);
    ctx.output.clear_data();

    // HOW FAR EVERY CELL IS FROM THE NEAREST CELL OF EACH COLOR IN THE ANSWER,
    // which is what lets a near miss score better than a wild one. Derived
    // wholly from ex.output, so it is worked out here rather than stored on the
    // example: one recipe, and it can never be stale against the pair it
    // describes. Cells outside the answer stay 0, and the scorer never reads
    // them -- it walks the target's own pixels and no further.
    memset(ctx.distance_maps, 0, sizeof(ctx.distance_maps));
    {
        const int h = std::min(ex.output.height, (int)GPU_MAX_GRID);
        const int w = std::min(ex.output.width, (int)GPU_MAX_GRID);
        static constexpr int drs[] = {-1, 1, 0, 0};
        static constexpr int dcs[] = {0, 0, -1, 1};
        // BFS out from every cell of the color at once, so each map costs one
        // sweep of the grid rather than one per pair of cells. Cells come off in
        // the order they went on, so the first distance a cell is given is
        // already its shortest and the guard below turns away every later one --
        // which is what holds a cell to one place in the queue, and the queue to
        // one cell per grid. On the stack, keeping this file free of the heap.
        int16_t queue[GPU_GRID_CELLS];
        for (int color = 0; color < GPU_MAX_COLORS; ++color) {
            int tail = 0;
            for (int r = 0; r < h; ++r)
                for (int c = 0; c < w; ++c) {
                    if (ex.output.data[r][c] == color) {
                        ctx.distance_maps[color][r][c] = 0;
                        queue[tail++] = (int16_t)(r * GPU_MAX_GRID + c);
                    } else {
                        // Stands for "no cell of this color anywhere near", and
                        // is what every real distance is measured against below.
                        ctx.distance_maps[color][r][c] = (int8_t)DISTANCE_MAX;
                    }
                }
            for (int head = 0; head < tail; ++head) {
                const int cr = queue[head] / GPU_MAX_GRID, cc = queue[head] % GPU_MAX_GRID;
                // A cell is only ever queued once it holds a real distance, so
                // this never counts up from the stand-in and cannot run past
                // what an int8_t holds.
                const int nd = ctx.distance_maps[color][cr][cc] + 1;
                for (int i = 0; i < 4; ++i) {
                    const int nr = cr + drs[i], nc = cc + dcs[i];
                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                    if (nd >= ctx.distance_maps[color][nr][nc]) continue;
                    ctx.distance_maps[color][nr][nc] = (int8_t)nd;
                    queue[tail++] = (int16_t)(nr * GPU_MAX_GRID + nc);
                }
            }
        }
    }

    // Segment objects from input for seeds/palette
    fixed_grid input_copy;
    grid_to_fixed(ex.input.data, ex.input.width, ex.input.height, input_copy);

    static thread_local auto all_objects_ptr = std::make_unique<fixed_object_list>();
    auto& all_objects = *all_objects_ptr;
    all_objects.clear();
    fixed_segment_objects(input_copy, all_objects);

    // Remove invariant-color objects
    if (invariant_colors) {
        int dst = 0;
        for (int i = 0; i < all_objects.count; ++i) {
            int8_t c = fixed_obj_first_color(all_objects.objects[i]);
            if (is_color(c) && invariant_colors[c]) continue;
            all_objects.objects[dst++] = all_objects.objects[i];
        }
        all_objects.count = dst;
    }

    // Seeds = all objects except largest, sorted by size ascending
    ctx.seed_objs.clear();
    if (all_objects.count > 0) {
        int largest_idx = 0, largest_area = 0;
        for (int i = 0; i < all_objects.count; ++i) {
            int area = all_objects.objects[i].width * all_objects.objects[i].height;
            if (area > largest_area) { largest_area = area; largest_idx = i; }
        }
        for (int i = 0; i < all_objects.count; ++i) {
            if (i == largest_idx) continue;
            if (ctx.seed_objs.count >= GPU_MAX_OBJECTS) break;
            ctx.seed_objs.objects[ctx.seed_objs.count++] = all_objects.objects[i];
        }
        sort_objects_by_area_asc(ctx.seed_objs);
    }
}

} // namespace arc
