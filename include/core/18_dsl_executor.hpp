#pragma once

// GPU-based execute_flat_program bridge.
// All program execution + scoring goes through the CUDA scratch kernel.
// No CPU-side scoring — GPU score and solved flag are returned directly.

namespace arc {

	// The GPU pool is a ring, so it must hold the object-lists that are live at
	// once -- the one an op reads and the one it writes -- and no longer has to
	// cover a whole program's worth of acquires. Depth may therefore grow without
	// touching it. The floor is generous on purpose, so an op that someday held a
	// third list is still safe; the ring itself is gpu_acquire in cuda_solver.cu.
	static_assert(GPU_KERNEL_POOL_SIZE >= 8,
		"GPU pool must hold more than the lists live at once; raise GPU_KERNEL_POOL_SIZE");

	// ONE EXAMPLE, READY FOR THE KERNEL: what it starts from and what it must
	// paint, into slot `ei` of an upload. The main search fills its active
	// examples with this and the scratch path its one, so a program scores the
	// same on both -- otherwise refresh_best_display_scores would set the bar
	// without the CC bonus and every CC-bearing winner would trivially beat it.
	inline void fill_example_slot(cuda_dsl_example_data& data, int ei, const example& ex,
	                              const puzzle& p) {
		static thread_local auto init_ctx = std::make_unique<fixed_exec_context>();
		fixed_init_context(*init_ctx, ex, p.rule, p.invariant_colors);

		data.input_grids[ei] = init_ctx->input.data;
		data.input_dims[ei * 2 + 0] = (int8_t)init_ctx->input.width;
		data.input_dims[ei * 2 + 1] = (int8_t)init_ctx->input.height;

		gpu_packed_mask& tg = data.target_grids[ei];
		tg.clear();
		for (int r = 0; r < ex.output.height; ++r)
			for (int c = 0; c < ex.output.width; ++c)
				tg[r][c] = (int8_t)ex.output.data[r][c];
		data.target_dims[ei * 2 + 0] = (int8_t)ex.output.width;
		data.target_dims[ei * 2 + 1] = (int8_t)ex.output.height;

		memcpy(data.distance_maps + ei * GPU_MAX_COLORS * GPU_GRID_CELLS,
		       init_ctx->distance_maps, GPU_MAX_COLORS * GPU_GRID_CELLS);

		to_gpu_object_list(init_ctx->seed_objs, data.seed_lists[ei]);

		// WHERE THIS EXAMPLE'S PANES LIE. Geometry only: Panes is an act and
		// clips whatever the chain holds, so nothing here decides what counts as
		// figure. Done in this file and not in fixed_init_context because the cut
		// is a fact the puzzle holds, and a test input has to be cut by the same
		// reading the pairs were -- the fact re-reads the grid rather than
		// remembering the pairs, so what the op cuts on and what inference said
		// cannot drift apart.
		vector<pane_split_facts::pane> boxes;
		int8_t (&out_boxes)[GPU_MAX_PANES][4] = data.pane_boxes[ei];
		memset(out_boxes, 0, sizeof(out_boxes));
		int8_t found = 0;
		if (p.inf.pane_split.holds() && p.inf.pane_split.panes_of(ex.input, boxes))
			for (const auto& box : boxes) {
				if (found >= GPU_MAX_PANES) break;
				out_boxes[found][0] = box.top;
				out_boxes[found][1] = box.left;
				out_boxes[found][2] = box.width;
				out_boxes[found][3] = box.height;
				++found;
			}
		data.pane_counts[ei] = found;

		data.target_obj_counts[ei] = environment::fill_target_objects(
			p, ex.output, data.target_objs + ei * GPU_MAX_TARGET_OBJECTS);
	}

	inline bool environment::execute_flat_program(
		const flat_program& program, const example& ex, grid& output,
		const puzzle& p, int node_count, float& out_score, bool& out_solved,
		gpu_score_terms* out_terms) {

		out_score = SCORE_MIN;
		out_solved = false;

		if (program.count <= 0)
			return true;

		// Build a single-example upload for this example. On the heap, made once a
		// thread: an upload holds a list as long as a grid has cells for every
		// example it can carry, far past what a thread's own storage should hold.
		static thread_local auto ex_data = std::make_unique<cuda_dsl_example_data>();
		ex_data->count = 1;
		fill_example_slot(*ex_data, 0, ex, p);
		fill_puzzle_level(*ex_data, p);

		// Upload into the scratch slot — isolated from the main cache used by beam search.
		cuda_dsl_upload_scratch_example(*ex_data);

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
		if (out_terms) *out_terms = batch.score_terms[0];

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
