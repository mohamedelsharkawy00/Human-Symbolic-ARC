#pragma once

// GPU-based execute_flat_program bridge.
// All program execution + scoring goes through the CUDA scratch kernel.
// No CPU-side scoring — GPU score and solved flag are returned directly.

namespace arc {

	// The GPU pool must hold every object-list a max-depth program acquires
	// (up to ~2 slots/op). If MAX_SEARCH_DEPTH grows past 24, bump
	// GPU_KERNEL_POOL_SIZE in cuda_dsl.cuh in lockstep or deep candidates get
	// silently mis-scored (clamp in gpu_acquire). See that note.
	static_assert(GPU_KERNEL_POOL_SIZE >= 2 * MAX_SEARCH_DEPTH,
		"GPU pool too small for max-depth programs; raise GPU_KERNEL_POOL_SIZE");

	inline bool environment::execute_flat_program(
		const flat_program& program, const example& ex, grid& output,
		const puzzle& p, int node_count, float& out_score, bool& out_solved) {

		out_score = SCORE_MIN;
		out_solved = false;

		if (program.count <= 0)
			return true;

		// Build a single-example upload for this example
		static thread_local auto init_ctx = std::make_unique<fixed_exec_context>();
		static thread_local cuda_dsl_example_data ex_data;

		fixed_init_context(*init_ctx, ex, p.rule, p.invariant_colors);

		ex_data.count = 1;
		memcpy(ex_data.input_grids, init_ctx->input.data, GPU_GRID_CELLS);
		ex_data.input_dims[0] = (int8_t)init_ctx->input.width;
		ex_data.input_dims[1] = (int8_t)init_ctx->input.height;

		int8_t* tg = ex_data.target_grids;
		memset(tg, GPU_TRANSPARENT, GPU_GRID_CELLS);
		for (int r = 0; r < ex.output.height; ++r)
			for (int c = 0; c < ex.output.width; ++c)
				tg[r * GPU_MAX_GRID + c] = (int8_t)ex.output.data[r][c];
		ex_data.target_dims[0] = (int8_t)ex.output.width;
		ex_data.target_dims[1] = (int8_t)ex.output.height;

		memcpy(ex_data.distance_maps, init_ctx->distance_maps, GPU_MAX_COLORS * GPU_GRID_CELLS);

		ex_data.seed_lists[0] = to_gpu_object_list(init_ctx->seed_objs);

		// Target CC summaries for this single example. Recomputed rather than read
		// off p.target_objects so the scratch-path score matches the main-path
		// score — otherwise refresh_best_display_scores would set the bar
		// without the CC bonus and every CC-bearing winner would trivially beat it.
		{
			object_summary tobjs[MAX_TARGET_OBJECTS];
			int tcount = segment_objects_by_color_cpu(ex.output, tobjs, MAX_TARGET_OBJECTS);
			gpu_object_summary* dst = ex_data.target_objs; // ei=0
			for (int k = 0; k < tcount; ++k) {
				dst[k].color = tobjs[k].color;
				dst[k].w = tobjs[k].w;
				dst[k].h = tobjs[k].h;
				dst[k].x0 = tobjs[k].x0;
				dst[k].y0 = tobjs[k].y0;
				dst[k].count = tobjs[k].count;
			}
			for (int k = tcount; k < GPU_MAX_TARGET_OBJECTS; ++k) dst[k] = {};
			ex_data.target_obj_counts[0] = (int8_t)tcount;
		}

		fill_puzzle_level(ex_data, p);

		// Upload into the scratch slot — isolated from the main cache used by beam search.
		cuda_dsl_upload_scratch_example(ex_data);

		// Build a 1-job batch
		static thread_local cuda_dsl_exec_batch batch;
		batch.resize(1);
		batch.active_examples = 1;

		gpu_flat_op gpu_ops[GPU_MAX_OPS];
		int gpu_op_count = convert_program_to_gpu_ops(program, gpu_ops);
		gpu_flat_op* dst_ops = batch.programs.data();
		for (int o = 0; o < gpu_op_count; ++o) dst_ops[o] = gpu_ops[o];
		for (int o = gpu_op_count; o < GPU_MAX_OPS; ++o) dst_ops[o] = {node_type::Input, 0};
		batch.program_lengths[0] = (int8_t)gpu_op_count;
		batch.node_counts[0] = (int8_t)node_count;
		batch.program_indices[0] = 0;

		// Execute on GPU via scratch path — leaves main example cache intact.
		cuda_dsl_exec_and_score_scratch(batch);
		cuda_dsl_fetch_output_grids_scratch(batch, 0, 1);

		// Read GPU score + solved flag directly — no CPU re-scoring.
		out_score = batch.scores[0];
		out_solved = (batch.solved[0] != 0);

		// Copy output grid back
		const int8_t* og = batch.output_grids.data();
		output.width = batch.output_dims_result[0];
		output.height = batch.output_dims_result[1];
		for (int r = 0; r < output.height; ++r)
			for (int c = 0; c < output.width; ++c)
				output.data[r][c] = (int)og[r * GPU_MAX_GRID + c];

		return true;
	}
}
