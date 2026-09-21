#pragma once

// This header is included from 15_solver_beam.hpp AFTER 12_environment.hpp and 11_dsl.cuh,
// so arc:: types (grid, example, output_dimension_rule, flat_program, etc.) are already available.
// Do NOT include 12_environment.hpp here to avoid circular dependencies.
//
// Contains: fixed-size grid/object types, segmentation helpers for puzzle upload,
// and the fixed_init_context used by build_example_upload. All DSL execution is
// handled by the GPU kernel in cuda_solver.cu.

#include "11_dsl.cuh"
#include <cstring>
#include <algorithm>

namespace arc {

// --- Fixed-size grid, no heap ---
// The same struct the kernel holds a grid in, under the name the host code has
// always used. One type, one layout: a grid crosses to the device as a copy.
using fixed_grid = gpu_fixed_grid;

// --- Convert from environment grid (int pixels) to the packed fixed_grid ---
static inline void grid_to_fixed(const int src[][30], int w, int h, fixed_grid& dst) {
    dst.data.clear();
    dst.width = (int16_t)w;
    dst.height = (int16_t)h;
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            dst.data[r][c] = (int8_t)src[r][c];
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
    object_walk_scratch scratch;
    each_object(g, bg_color, grouping::touching_same_color, scratch,
        [&](const object_summary& box, const int16_t* cells, int n) {
            if (out.count >= GPU_MAX_OBJECTS) return;
            fixed_object& obj = out.objects[out.count++];
            obj.clear();
            obj.x0 = box.x0; obj.y0 = box.y0;
            obj.width = box.w; obj.height = box.h;
            obj.x1 = (int16_t)(obj.x0 + obj.width - 1);
            obj.y1 = (int16_t)(obj.y0 + obj.height - 1);
            for (int i = 0; i < n; ++i) {
                const int row = cells[i] / MAX_GRID_DIM, col = cells[i] % MAX_GRID_DIM;
                obj.mask[row - obj.y0][col - obj.x0] = g.data[row][col];
            }
        });
}

// One layer per color, including disconnected cells.
static inline void fixed_segment_color_layers(const fixed_grid& g, fixed_object_list& out, int8_t bg_color = GPU_TRANSPARENT) {
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
        obj.mask.clear();

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

// Narrow fixed_object_list (CPU, 900 objs) -> gpu_object_list (GPU kernel, 64 objs).
// Only the count changes: both sides hold their cells in the same packed bytes,
// so the mask moves across as a plain copy.
static inline gpu_object_list to_gpu_object_list(const fixed_object_list& src) {
    gpu_object_list dst;
    dst.count = std::min(src.count, GPU_KERNEL_MAX_OBJECTS);
    for (int i = 0; i < dst.count; ++i) dst.objects[i] = src.objects[i];
    return dst;
}

// =====================================================================
// Bridge functions (depend on 12_environment.hpp types: flat_program, example, output_dimension_rule, grid)
// These are placed at the end because 16_fixed_dsl.hpp is included after 12_environment.hpp.
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
    ctx.output.clear_cells();

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

    // Seeds retain every color here: the rule below drops the largest box,
    // after removing invariant colors, rather than excluding elected background.
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

    // Seeds = all objects except the largest, smallest box first. The rule is
    // 4_objects.hpp's read_seeds, which inference reads too, so what an op is
    // handed here and what a reading says it will be handed cannot drift apart.
    ctx.seed_objs.clear();
    if (all_objects.count > 0) {
        vector<object_summary> boxes;
        for (int i = 0; i < all_objects.count; ++i) {
            object_summary box;
            box.w = (int8_t)all_objects.objects[i].width;
            box.h = (int8_t)all_objects.objects[i].height;
            boxes.push_back(box);
        }
        // A grid holds at most 900 cells and so at most 900 objects, which is
        // what this list holds: the cap is a guard, never a cut.
        for (int which : read_seeds(boxes).seeds) {
            if (ctx.seed_objs.count >= GPU_MAX_OBJECTS) break;
            ctx.seed_objs.objects[ctx.seed_objs.count++] = all_objects.objects[which];
        }
    }
}

} // namespace arc
