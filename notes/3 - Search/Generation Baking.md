Bake rejection criteria into generation loop bounds
====================================================

Goal: stop generating candidates that get rejected. Move filters out of the
inner loop and into the iteration structure itself, so rejected combinations
never become iterable items in the first place.

Current shape of generate_phase_1 (solver_executor.hpp ~449):

  for each primitive:                  // ~50
    for each history entry:            // ~1000
      for each variant:                // 1-30
        build candidate                // alloc + hash
        check 4 rejection criteria:
          - check_candidate_compatibility (type mismatch)
          - depth: slot.path.len + 1 + t0->max_depth > MAX_SEARCH_DEPTH
          - tile-op cap (in add_candidate_if_new)
          - hash dedup (in add_candidate_if_new)
        if any fail -> drop the just-built candidate

Most of those iterations get rejected. The profile shows 90%+ waste during
dead zones. Each rejection still pays for create_unary_node + update_hash.


Three bake-ins, in order of payoff
----------------------------------

A. Bucket history_pool by return type
   Each primitive's expected arg type is known at compile time. Group
   history entries by their return type once, when the pool is updated.
   In the inner loop, iterate only the matching bucket -> no compat checks.

   Cost: history_pool becomes vector<vector<node*>> keyed by ast_type.
   update_history_from_beam classifies on insert.

   Effect: incompatible pairs no longer exist as iterable items. Eliminates
   the dominant rejection source (compatibility mismatch).

   Status: PICKED FIRST.

B. Pre-filter bucket by depth headroom per slot
   Each slot has fixed path.len. The max allowed history-entry max_depth
   is MAX_SEARCH_DEPTH - slot.path.len - 1, constant per slot.

   Cost: keep each bucket sorted by max_depth, or partition into depth
   sub-buckets. Per slot, take the prefix that fits the headroom.

   Effect: depth-rejected entries never enter the loop.

C. Cache tile_op_count on each node, partition bucket by tile-op count
   Add int8_t tile_op_count to node, computed once at construction
   (parent's count + 1 if this op is TileX/TileY else 0). Per generation
   pass, the max allowed history-entry tile_op_count is
   MAX_TILE_OPS - parent_tile_ops - new_op_tile_ops. Filter bucket
   prefix by that bound.

   Cost: 1 byte per node, one extra line in update_hash, partition
   logic in bucketing.

   Effect: tile-rejected entries never enter the loop.


What can't be baked in
----------------------
- visited.contains: needs the candidate hash, which needs the candidate.
  The lookup itself stays. Only the *cost of producing* the rejected
  candidate can be eliminated.
- global_cache lookup: same reason.

These two filters can be moved earlier in the pipeline (cheaper rejection)
but cannot become iteration bounds.


Expected impact
---------------
A alone: ~3-5x fewer inner-loop iterations during dead zones, because
compatibility mismatch is the most common rejection. Should compress the
12s of remaining stall time noticeably.

A + B + C together: estimated 90% reduction in inner-loop iterations
during dead zones. The dead zones in the log come from the generator
grinding ~380k rejections per level transition; with these baked in, most
of those iterations stop existing.


Implementation order
--------------------
1. A first (this is what's being implemented now).
2. Profile, measure dead-zone reduction.
3. B if dead zones still significant.
4. C if dead zones still significant.

Do not implement all three at once. Each one changes the structure of
history_pool / node; chaining unverified changes makes regressions harder
to attribute.
