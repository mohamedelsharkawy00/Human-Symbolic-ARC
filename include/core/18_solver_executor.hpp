#pragma once

namespace arc {

  inline void beam_solver::report_score(const node *n, float score, uint64_t g_hash) {
    if (!n)
      return;
    if (global_cache.size() >= MAX_CACHE_ENTRIES) {
      { unordered_map<uint64_t, cache_entry> tmp; global_cache.swap(tmp); }
      log_debug("Score cache full, cleared to maintain constant memory.");
    }
    global_cache[n->hash] = {score, g_hash};
  }

  inline beam_solver::phase_result beam_solver::handle_retry(int& batch_count) {
    if (check_yield(batch_count)) return beam_solver::phase_result::yield;
    return beam_solver::phase_result::retry;
  }

  inline bool beam_solver::check_yield(int &batch_count) {
    if (batch_count++ > 512) {
      batch_count = 0;
      return true;
    }
    return false;
  }

  inline void beam_solver::commit_and_advance_level() {
    // INJECT PREVIOUS BEAM: Monotonicity base
    for (auto &cand : beam) {
      if (!cand.n) continue;
      level_candidates.push_back({environment::copy(cand.n), cand.score});
    }

    commit_level_champion();

    float current_beam_top = beam.empty() ? SCORE_MIN : beam[0].score;
    if (current_beam_top > prev_beam_top + 0.01f) {
      prev_beam_top = current_beam_top;
      stagnant_levels = 0;
    } else {
      stagnant_levels++;
    }

    if (stagnant_levels >= MAX_STAGNANT_LEVELS) {
      log_debug(to_text("Level stagnation: %d levels at %.2f. Diversifying beam.",
                       stagnant_levels, (double)prev_beam_top));
      if (beam.size() > 4) {
        sort(beam.begin(), beam.end(), ranks_higher);
        size_t keep = beam.size() / 2;
        for (size_t i = keep; i < beam.size(); ++i)
          if (beam[i].n) environment::remove(beam[i].n);
        beam.resize(keep);
      }
      unordered_set<node_type> present_types;
      for (auto &cand : beam) if (cand.n) present_types.insert(cand.n->type);
      for (size_t i = 0; i < history_pool.size() && (int)beam.size() < BEAM_WIDTH; ++i) {
        node *h = history_pool[i];
        if (!h) continue;
        if (!present_types.insert(h->type).second) continue;
        beam.push_back({environment::copy(h), -1.0f, 0});
      }
      stagnant_levels = 0;
    }

    // Wipe the visited set between levels. Most duplicates come from the
    // within-level enumeration burst (same history pool, same primitives).
    // Cross-level repeats slip through to the GPU's global_cache, which
    // skips re-scoring. Keeps visited cache-resident.
    visited.clear();
    visited_states.clear();

    prepare_next_level_slots();
    update_history_from_beam();
    m_consec_dry_beam_entries = 0;

    log_level_timing();
  }

  // One batch of whole sentences. Level 1 has an empty beam, so a sentence IS
  // the program; later levels hang one onto each beam member. Returns true when
  // the level has nothing left to give.
  inline bool beam_solver::generate_sentence_batch() {
    if (!sweep_ready) {
      enumerate_sentences(beam.empty(), sweep);
      sweep_ready = true;
      sweep_beam_idx = 0;
      sweep_idx = 0;
      log_debug(to_text("[SENTENCES] level sweep = %zu (beam %d)",
                            sweep.size(), (int)beam.size()));
    }
    if (sweep.empty()) return true;

    int made = 0;
    // The base changes once per beam member, not once per sentence, so it is
    // flattened once and the sentence is appended in place and taken back off.
    flat_program program;
    int flattened_for = -1;
    while (made < MAX_BATCH_SIZE) {
      if (!beam.empty()) {
        if (sweep_beam_idx >= (int)beam.size()) return true;
        if (sweep_idx >= (int)sweep.size()) {
          sweep_idx = 0;
          sweep_beam_idx++;
          continue;
        }
        node *base = beam[sweep_beam_idx].n;
        if (!base) { members_skipped++; sweep_idx = 0; sweep_beam_idx++; continue; }
        if (flattened_for != sweep_beam_idx) {
          program.clear();
          if (!environment::flatten_unary_program(base, program)) {
            members_skipped++;
            sweep_idx = 0;
            sweep_beam_idx++;
            continue;
          }
          // A sentence is four ops, so the depth cap bounds how many fit.
          if (program.count + 4 > MAX_SEARCH_DEPTH) {
            members_skipped++;
            sweep_idx = 0;
            sweep_beam_idx++;
            continue;
          }
          flattened_for = sweep_beam_idx;
        }
      } else if (sweep_idx >= (int)sweep.size()) {
        return true;
      }

      const int base_count = program.count;
      pairs_walked++;
      if (!write_sentence(sweep[sweep_idx++], program)) {
        program.count = base_count;
        continue;
      }

      node *candidate = environment::build_chain_from_flat(program);
      program.count = base_count;
      if (!candidate) continue;
      if (!visited.insert_if_new(candidate->hash)) {
        if (pairs_already_seen < 5)
          log_debug(to_text("[GEN-DUP] \"%s\"",
                                environment::stringify_tree_via_flat(candidate).c_str()));
        pairs_already_seen++;
        environment::remove(candidate);
        continue;
      }
      // Same ownership contract as add_candidate_if_new: the slot list owns the
      // ref, pending_evaluation only borrows it.
      current_slot_candidates.push_back({candidate, 0.0f, 0});
      pending_evaluation.push(candidate);
      pairs_kept++;
      made++;
    }
    return false;
  }

  // Close a sentence level: keep the beam as a floor, distil, start the next
  // sweep. No slots and no history pool — a sentence needs neither.
  inline void beam_solver::commit_sentence_level() {
    merge_slot_candidates();
    for (auto &cand : beam)
      if (cand.n)
        level_candidates.push_back({environment::copy(cand.n), cand.score, cand.state_hash});
    commit_level_champion();
    // The beam's own order, so a tie decided later can be traced to where it
    // actually sat rather than guessed at.
    for (int i = 0; i < (int)beam.size(); ++i)
      if (beam[i].n)
        log_debug(to_text("[BEAM %d] %.2f  \"%s\"", i, (double)beam[i].score,
                              environment::stringify_tree_via_flat(beam[i].n).c_str()));
    log_debug(to_text(
        "[GEN] walked=%lld kept=%lld already_seen=%lld members_skipped=%lld beam=%d sweep=%d",
        pairs_walked, pairs_kept, pairs_already_seen, members_skipped,
        (int)beam.size(), (int)sweep.size()));
    pairs_walked = pairs_kept = pairs_already_seen = members_skipped = 0;
    log_level_timing();
    visited.clear();
    visited_states.clear();
    sweep_ready = false;
    sweep_beam_idx = 0;
    sweep_idx = 0;
  }

  // Where a level's time actually went, said once when it closes. Both level
  // commits report through here so the two searches read the same way.
  inline void beam_solver::log_level_timing() {
    long long fetch_copy_ns = 0, fetch_sync_ns = 0;
    cuda_dsl_take_fetch_ns(&fetch_copy_ns, &fetch_sync_ns);
    long long wait_total = m_progress.gpu_wait_ns.load();
    long long ns_wait = wait_total - ns_wait_at_last_level;
    ns_wait_at_last_level = wait_total;
    log_timing(to_text(
      "gen=%lldms launch=%lldms wait=%lldms idle=%lldms | harvest=%lldms = tally=%lldms winner=%lldms free=%lldms rest=%lldms"
      " | winner = fetch=%lldms analyze=%lldms best=%lldms rest=%lldms"
      " | fetch = copy=%lldms sync=%lldms",
      ns_generate / 1000000, ns_launch / 1000000, ns_wait / 1000000, ns_idle / 1000000,
      ns_harvest / 1000000, ns_tally / 1000000, ns_winner / 1000000, ns_free / 1000000,
      (ns_harvest - ns_tally - ns_winner - ns_free) / 1000000,
      ns_fetch / 1000000, ns_analyze / 1000000, ns_best / 1000000,
      (ns_winner - ns_fetch - ns_analyze - ns_best) / 1000000,
      fetch_copy_ns / 1000000, fetch_sync_ns / 1000000));
    ns_generate = ns_launch = ns_harvest = ns_idle = 0;
    ns_tally = ns_winner = ns_free = 0;
    ns_fetch = ns_analyze = ns_best = 0;
  }

  // Takes no puzzle: beam_solver already holds it, and this only steers its own
  // members. The parameter was a second way to say the same thing.
  inline bool beam_solver::pump_generate(atomic<long long> *global_attempts) {
    m_global_attempts = global_attempts;
    // Sentence mode never seeds starters: it has no history pool and no slots,
    // because a sentence is built whole rather than grown an op at a time.
    if (!initialized && !m_options.sentence_search) {
      seed_starter_nodes();
      initialized = true;
      log_debug("Top-Down Solver (Zero Memory) Initialized.");
    }

    if (!pending_evaluation.empty())
      return true;

    is_generating_layer = true;

    if (m_options.sentence_search) {
      // Every candidate of a level stays live until it is merged, and a level is
      // beam x sweep -- 23.8M chains at beam 128, which fills the process memory
      // limit and stops dead. Distil whenever the pile gets big, the same cap the
      // op-at-a-time path applies per slot. Safe here and only here: pending
      // BORROWS these nodes and the guard above proved it empty.
      if ((int)current_slot_candidates.size() >= MAX_CANDIDATES_PER_SLOT)
        merge_slot_candidates();
      // Commit only once the batch just made has been drained: level_candidates
      // is released on commit, and pending_evaluation only BORROWS those nodes.
      // Committing with any still queued hands step() freed memory.
      if (generate_sentence_batch() && pending_evaluation.empty())
        commit_sentence_level();
      return !pending_evaluation.empty();
    }

    for (int i = 0; i < 1024; ++i) {
      bool slot_done = generate_branch();
      if (!pending_evaluation.empty())
        return true;
      if (!slot_done)
        return false;

      log_debug(to_text("[GEN] slot=%d/%d done cand=%d lc=%d vis=%zu",
        current_slot_idx, (int)slots_to_refine.size(),
        total_candidates_this_slot, (int)level_candidates.size(),
        visited.size()));
      merge_slot_candidates();
      current_slot_idx++;
      beam_idx = 0;
      total_candidates_this_slot = 0;
      m_consec_dry_beam_entries = 0;
      reset_d2_generator();

      if (current_slot_idx >= (int)slots_to_refine.size()) {
        commit_and_advance_level();
        return !pending_evaluation.empty();
      }
    }
    return !pending_evaluation.empty();
  }

  inline node *beam_solver::get_node_at_path(node *root, int depth) {
    node *curr = root;
    for (int i = 0; i < depth && curr; ++i)
      curr = curr->arg;
    return curr;
  }

  // Predicts what the candidate's root hash *would* be if we performed
  // copy_and_replace, without allocating or copying anything. Mirrors the
  // exact fold in environment::update_hash. Used to early-reject duplicates
  // before paying for tree construction.
  inline uint64_t beam_solver::predict_replaced_hash(const node *root, int depth,
                                                    uint64_t new_subtree_hash) {
    // Walk the spine from root to the replacement point, recording each node.
    // Leaf of the walk is the swap point: its "new" hash is new_subtree_hash.
    // Then we fold back up, recomputing each ancestor's hash with the child
    // swapped to the new hash.
    const node *spine[MAX_SEARCH_DEPTH + 1];
    int spine_len = 0;
    const node *curr = root;
    for (int i = 0; i < depth; ++i) {
      if (!curr) return 0;
      spine[spine_len++] = curr;
      curr = curr->arg;
    }

    uint64_t child_hash = new_subtree_hash;
    for (int spine_at = spine_len - 1; spine_at >= 0; --spine_at) {
      const node *n = spine[spine_at];
      // The one recipe, not a copy of it; the swapped child is treated as
      // non-null.
      child_hash = environment::node_hash(to_underlying(n->type),
                                          (uint64_t)n->value, child_hash, true);
    }
    return child_hash;
  }

  inline node *beam_solver::copy_and_replace(const node *root, int depth, node *new_subtree) {
    if (depth == 0)
      return environment::copy(new_subtree);
    if (!root)
      return nullptr;

    node *res = new node();
    res->type = root->type;
    res->value = root->value;
    res->hash = root->hash;
    res->arg = copy_and_replace(root->arg, depth - 1, new_subtree);
    environment::update_hash(res);
    return res;
  }

  inline node *beam_solver::create_unary_node(node_type type, node *arg0) {
    node *n = new node();
    n->type = type;
    n->arg = environment::copy(arg0);
    environment::update_hash(n);  // finalizes depth + generation debt
    return n;
  }

  inline void beam_solver::beat_choices(op_role want, vector<flat_program_op> &out) const {
    out.clear();
    for (node_type t : primitive_types) {
      if (role_of(t) != want) continue;
      if (m_puzzle.dead_ops[(int)t]) continue;
      int variants = get_variant_count(t);
      for (int v = 0; v < variants; ++v) {
        int16_t value = get_variant_value(t, v);
        if (!color_literal_allowed(t, value)) continue;
        out.push_back({t, value});
      }
    }
  }

  inline void beam_solver::enumerate_sentences(bool sentence_opens_program,
                                               vector<sentence> &out) const {
    out.clear();
    vector<flat_program_op> gets, selects, acts, combines;
    beat_choices(op_role::get, gets);
    // A sweep must not hold a sentence the grammar forbids. Asking legal_next
    // keeps that rule in its one home, and keeps the forbidden ones from being
    // built and thrown away.
    if (sentence_opens_program)
      gets.erase(std::remove_if(gets.begin(), gets.end(),
                     [](const flat_program_op &op) { return !legal_next(nullptr, op.type); }),
                 gets.end());
    beat_choices(op_role::select, selects);
    beat_choices(op_role::combine, combines);
    // ACT is the four middle roles folded together, so it is gathered by role.
    acts.clear();
    for (op_role r : {op_role::reduce, op_role::translate, op_role::map, op_role::generate}) {
      vector<flat_program_op> part;
      beat_choices(r, part);
      acts.insert(acts.end(), part.begin(), part.end());
    }
    out.reserve(gets.size() * selects.size() * acts.size() * combines.size());
    for (const flat_program_op &g : gets)
      for (const flat_program_op &s : selects)
        for (const flat_program_op &a : acts)
          for (const flat_program_op &c : combines)
            out.push_back({g, s, a, c});
  }

  inline void beam_solver::seed_starter_nodes() {
    // The chain starts from the input, so depth 1 is every op legal with no
    // predecessor — not just the sources. Input itself is barred there
    // (legal_next): it would only respell the start every chain already has.
    vector<node *> starter_nodes;
    for (node_type t : primitive_types) {
      if (m_puzzle.dead_ops[(int)t]) continue;
      if (!legal_next(nullptr, t)) continue;
      // Annotated starters (e.g. InputFiltered[c]) seed one node per variant so the
      // search can explore each color choice independently from depth 0.
      int variants = get_variant_count(t);
      for (int v = 0; v < variants; ++v) {
        // The same pin the other two generators apply: a starter may not name a
        // color the side it speaks about never shows.
        int16_t value = get_variant_value(t, v);
        if (!color_literal_allowed(t, value)) continue;
        node *n = new node();
        n->type = t;
        n->value = value;
        environment::update_hash(n);  // finalizes depth + generation debt
        starter_nodes.push_back(n);
      }
    }

    rebuild_skeleton_beam();
    prepare_next_level_slots();

    for (node *n : starter_nodes) {
      node *h = environment::copy(n);
      history_pool.push_back(h);
      history_by_depth.push_back(h);
      pending_evaluation.push(n);
      visited.insert(n->hash);
    }
    // Starters all have max_depth=1; seed cutoffs so generate_phase_1
    // can run before update_history_from_beam fires for the first time.
    {
      int sz = (int)history_by_depth.size();
      for (int d = 0; d <= MAX_SEARCH_DEPTH; ++d)
        bucket_cutoff[d] = sz;
    }
    for (auto &cand : beam)
      if (cand.n) {
        pending_evaluation.push(cand.n);
        visited.insert(cand.n->hash);
      }

    current_slot_idx = 0;
    beam_idx = 0;
    m_gen_phase = 0;
    is_generating_layer = false;
  }

  inline void beam_solver::update_history_from_beam() {
    unordered_set<uint64_t> existing_hashes;
    for (node *n : history_pool)
      existing_hashes.insert(n->hash);

    int added = 0;
    for (auto &cand : beam)
      collect_subtrees(cand.n, existing_hashes, added);

    if (history_pool.size() > MAX_HISTORY_POOL) {
      size_t excess = history_pool.size() - MAX_HISTORY_POOL;
      for (size_t i = 0; i < excess; ++i)
        if (history_pool[i])
          environment::remove(history_pool[i]);
      history_pool.erase(history_pool.begin(), history_pool.begin() + excess);
    }

    // Rebuild the depth-sorted bucket. Compute prefix cutoffs so the generator
    // can iterate only entries shallow enough for a given slot's depth
    // headroom — depth-rejected entries never enter the inner loop.
    history_by_depth.clear();
    for (node *n : history_pool)
      if (n) history_by_depth.push_back(n);
    sort(history_by_depth.begin(), history_by_depth.end(),
         [](const node *a, const node *c) {
           if (a->max_depth != c->max_depth) return a->max_depth < c->max_depth;
           return a->hash < c->hash;
         });
    // bucket_cutoff[d] = count of entries with max_depth <= d.
    {
      int idx = 0;
      for (int d = 0; d <= MAX_SEARCH_DEPTH; ++d) {
        while (idx < (int)history_by_depth.size() && history_by_depth[idx]->max_depth <= d) ++idx;
        bucket_cutoff[d] = idx;
      }
    }

    log_debug(to_text("Updated History Pool: %zu items (+%d)",
                     history_pool.size(), added));
  }

  inline void beam_solver::collect_subtrees(node *n, unordered_set<uint64_t> &existing,
                        int &added) {
    if (!n)
      return;
    if (existing.find(n->hash) == existing.end()) {
      history_pool.push_back(environment::copy(n));
      existing.insert(n->hash);
      added++;
    }
    collect_subtrees(n->arg, existing, added);
  }

  inline void beam_solver::rebuild_skeleton_beam() {
    release_candidates(beam);

    node *inp = new node();
    inp->type = node_type::Input;
    environment::update_hash(inp);

    {
      auto add_seg = [&](node_type seg_type) {
        node* seg = new node(seg_type);
        seg->arg = environment::copy(inp);
        environment::update_hash(seg);

        node* largest_node = new node(node_type::SelectLargestObject);
        largest_node->arg = seg;
        environment::update_hash(largest_node);
        
        beam.push_back({environment::copy(largest_node), -1.0f, 0});
        environment::remove(largest_node);
      };
      
      add_seg(node_type::Split);
      add_seg(node_type::Colors);
      
      auto add_rank_seg = [&](node_type seg_type, int rank) {
        node* seg = new node(seg_type);
        seg->arg = environment::copy(inp);
        environment::update_hash(seg);

        node* ranked_node = new node(node_type::SelectLargestObject);
        ranked_node->arg = seg;
        ranked_node->value = (int16_t)rank;
        environment::update_hash(ranked_node);
        
        beam.push_back({environment::copy(ranked_node), -2.0f, 0});
        environment::remove(ranked_node);
      };
      
      add_rank_seg(node_type::Split, 1);
    }

    for (int i = 0; i < primitive_count; ++i) {
      node_type t = primitive_types[i];
      if (m_puzzle.dead_ops[(int)t]) continue;
      if (!legal_next(inp, t)) continue;  // grammar gate: t wraps chain `inp`
      if (is_parameterized(t)) {
        int variants = get_variant_count(t);
        for (int v_idx = 0; v_idx < variants; ++v_idx) {
          int16_t val = ranked_variant_value(t, v_idx);
          if (!color_literal_allowed(t, val)) continue;  // pin literal fan-out
          node *root = new node();
          root->type = t;
          root->value = val;
          root->arg = environment::copy(inp);
          environment::update_hash(root);
          beam.push_back({environment::copy(root), -18.0f, 0});
          environment::remove(root);
        }
        continue;
      }
      node *root = new node();
      root->type = t;
      root->arg = environment::copy(inp);
      environment::update_hash(root);
      beam.push_back({environment::copy(root), -18.0f, 0});
      environment::remove(root);
    }

    if (m_puzzle_loaded && m_puzzle.best_tree) {
      beam.push_back({environment::copy(m_puzzle.best_tree), 10.0f, 0});
    }

    // WHAT THE LOOKING SAID, KEPT IN THE BEAM. Trying the suggestions and then
    // searching from scratch would waste them the moment they did not finish the
    // puzzle; a sentence read off the pairs is worth extending even when it is
    // not the whole answer. Scored above the starters so they survive level 0.
    if (m_puzzle_loaded)
      for (const string &program : m_puzzle.suggested_programs)
        if (node *said = environment::parse(program)) {
          beam.push_back({environment::copy(said), 5.0f, 0});
          environment::remove(said);
        }

    environment::remove(inp);
  }

  inline void beam_solver::prepare_next_level_slots() {
    vector<slot_t> next_slots;
    set<int> visited_depths;
    for (auto &cand : beam) {
      if (!cand.n) continue;
      if (visited_depths.find(0) == visited_depths.end()) {
        next_slots.push_back({0});
        visited_depths.insert(0);
      }
      find_slots_recursive(cand.n, 0, next_slots, visited_depths);
    }
    if (next_slots.size() > 30)
      next_slots.resize(30);
    slots_to_refine = next_slots;
    current_slot_idx = 0;
    if (slots_to_refine.empty()) {
      rebuild_skeleton_beam();
      visited_depths.clear();
      for (auto &cand : beam)
        find_slots_recursive(cand.n, 0, slots_to_refine, visited_depths);
      if (slots_to_refine.empty()) {
        is_generating_layer = false;
      }
    }
    log_debug(to_text("New level: %zu slots to refine.", slots_to_refine.size()));
  }

  inline void beam_solver::find_slots_recursive(node *n, int depth, vector<beam_solver::slot_t> &out,
                            set<int> &visited_depths) {
    if (!n)
      return;
    if (depth >= MAX_SEARCH_DEPTH) return;
    node *child = n->arg;
    if (!child) return;
    int child_depth = depth + 1;
    if (visited_depths.find(child_depth) == visited_depths.end()) {
      out.push_back({child_depth});
      visited_depths.insert(child_depth);
    }
    find_slots_recursive(child, child_depth, out, visited_depths);
  }

  inline void beam_solver::reset_d2_generator() {
    m_gen_phase = 0;
    m_gen_primitive_idx = 0;
    m_gen_outer_idx = 0;
    m_gen_inner_idx = 0;

    total_candidates_this_slot = 0;
  }

  inline beam_solver::phase_result beam_solver::generate_phase_prune(node* target, int &batch_count, node *&candidate_sub) {
    if (!target) return beam_solver::phase_result::next_phase;
    if (m_gen_outer_idx < 1) {
      m_gen_outer_idx++;
      node* child = target->arg;
      if (child) {
        candidate_sub = child;
        if (check_yield(batch_count)) return beam_solver::phase_result::yield;
        return beam_solver::phase_result::candidate_ready;
      }
      return beam_solver::phase_result::retry;
    }
    return beam_solver::phase_result::next_phase;
  }

  inline beam_solver::phase_result beam_solver::generate_phase_0(const beam_solver::slot_t &slot, int &batch_count, node *&candidate_sub, [[maybe_unused]] bool &created_new) {
    if (m_gen_outer_idx < (int)history_pool.size()) {
      node *h = history_pool[m_gen_outer_idx++];
      if (slot.depth + h->max_depth <= MAX_SEARCH_DEPTH) {
        candidate_sub = h;
        if (check_yield(batch_count))
          return beam_solver::phase_result::yield;
        return beam_solver::phase_result::candidate_ready;
      }
      if (check_yield(batch_count))
        return beam_solver::phase_result::yield;
      return beam_solver::phase_result::break_loop;
    }
    return beam_solver::phase_result::next_phase;
  }

  inline beam_solver::phase_result beam_solver::generate_phase_1(const beam_solver::slot_t &slot, int &batch_count, node *&candidate_sub, bool &created_new) {
    if (m_gen_primitive_idx >= primitive_count) return beam_solver::phase_result::next_phase;

    node_type type = primitive_types[m_gen_primitive_idx];
    if (m_puzzle.dead_ops[(int)type]) {
      m_gen_primitive_idx++;
      m_gen_outer_idx = 0;
      m_gen_inner_idx = 0;
      return phase_result::break_loop;
    }

    // Bound iteration by the depth headroom for this slot; depth-overflow
    // entries never enter the loop.
    const auto &bucket = history_by_depth;
    int max_depth_allowed = MAX_SEARCH_DEPTH - slot.depth - 1;
    if (max_depth_allowed < 0) max_depth_allowed = 0;
    if (max_depth_allowed > MAX_SEARCH_DEPTH) max_depth_allowed = MAX_SEARCH_DEPTH;
    int bucket_end = bucket_cutoff[max_depth_allowed];

    if (m_gen_outer_idx >= bucket_end) {
      log_debug(to_text("[PRIM] prim=%d done: bucket_end=%d beam=%d cand=%d vis=%zu dry_streak=%d",
        m_gen_primitive_idx, bucket_end, beam_idx,
        total_candidates_this_slot, visited.size(), m_dry_yields));
      m_gen_primitive_idx++;
      m_gen_outer_idx = 0;
      m_gen_inner_idx = 0;
      return phase_result::break_loop;
    }

    node *t0 = bucket[m_gen_outer_idx];

    if (!legal_next(t0, type)) {
      m_gen_outer_idx++;
      m_gen_inner_idx = 0;
      return phase_result::retry;
    }

    candidate_sub = create_unary_node(type, t0);
    if (is_parameterized(type)) {
      int variants = get_variant_count(type);
      // Skip disallowed color-literal values (pin the fan-out). Advance the
      // inner index past any pinned-out value so we don't emit it.
      while (m_gen_inner_idx < variants &&
             !color_literal_allowed(type, ranked_variant_value(type, m_gen_inner_idx)))
        m_gen_inner_idx++;
      if (m_gen_inner_idx >= variants) {
        // All remaining variants pinned out — this outer node is done.
        environment::remove(candidate_sub);
        candidate_sub = nullptr;
        m_gen_outer_idx++;
        m_gen_inner_idx = 0;
        return phase_result::retry;
      }
      candidate_sub->value = ranked_variant_value(type, m_gen_inner_idx);
      environment::update_hash(candidate_sub);
      m_gen_inner_idx++;
      if (m_gen_inner_idx >= variants) {
        m_gen_outer_idx++;
        m_gen_inner_idx = 0;
      }
    } else {
      m_gen_outer_idx++;
      m_gen_inner_idx = 0;
    }

    created_new = true;
    if (check_yield(batch_count)) {
      environment::remove(candidate_sub);
      candidate_sub = nullptr;
      created_new = false;
      return phase_result::yield;
    }
    return phase_result::candidate_ready;
  }

  inline beam_solver::phase_result beam_solver::execute_current_phase(const beam_solver::slot_t &slot, node* target, int &batch_count, node *&candidate_sub, bool &created_new) {
    switch (m_gen_phase) {
      case 0: return generate_phase_prune(target, batch_count, candidate_sub);
      case 1: return generate_phase_0(slot, batch_count, candidate_sub, created_new);
      case 2: return generate_phase_1(slot, batch_count, candidate_sub, created_new);
    }
#ifdef _MSC_VER
    __assume(false);   // unreachable (MSVC intrinsic)
#else
    __builtin_unreachable();
#endif
  }

  inline bool beam_solver::generate_branch() {
    if (current_slot_idx >= (int)slots_to_refine.size()) return true;

    const slot_t &slot = slots_to_refine[current_slot_idx];
    int batch_count = 0;
    int items_processed = 0;

    while (beam_idx < (int)beam.size()) {
      if (++items_processed > 8) {
        return false;
      }

      int cand_before = total_candidates_this_slot;

      node *base = beam[beam_idx].n;
      node *target = get_node_at_path(base, slot.depth);
      if (!target) {
        advance_beam_idx();
        continue;
      }

      if (!process_current_phase(slot, base, target, batch_count)) {
        bool dry = (total_candidates_this_slot == m_last_cand_at_yield);
        if (dry) m_dry_yields++;
        else { m_dry_yields = 0; m_last_cand_at_yield = total_candidates_this_slot; }
        if (m_dry_yields >= 50) {
          log_debug(to_text("[GEN] dry yield exit: dry_yields=%d beam=%d cand=%d vis=%zu",
            m_dry_yields, beam_idx, total_candidates_this_slot, visited.size()));
          return true;
        }
        return false;
      }

      // Full beam entry just completed — check if it contributed anything
      if (total_candidates_this_slot == cand_before) {
        m_consec_dry_beam_entries++;
        log_debug(to_text("[GEN] dry beam entry=%d consec=%d cand=%d vis=%zu",
          beam_idx, m_consec_dry_beam_entries, total_candidates_this_slot, visited.size()));
        if (m_consec_dry_beam_entries >= 2) {
          log_debug(to_text("[GEN] early slot exit after %d dry entries at beam=%d cand=%d vis=%zu",
            m_consec_dry_beam_entries, beam_idx, total_candidates_this_slot, visited.size()));
          return true;
        }
      } else {
        m_consec_dry_beam_entries = 0;
      }

      advance_beam_idx();
    }
    log_debug(to_text("[GEN] beam exhausted: beam=%d cand=%d vis=%zu",
      beam_idx, total_candidates_this_slot, visited.size()));
    return true;
  }

  inline void beam_solver::advance_beam_idx() {
    beam_idx++;
    m_gen_phase = 0;
    reset_generator_counters();
  }

  inline bool beam_solver::process_current_phase(const beam_solver::slot_t& slot, node* base, node* target, int& batch_count) {
    m_current_base = base;  // phase_1 wraps the member itself in locked mode
    while (m_gen_phase < 3) {
      int duplicate_retries = 0;
      while (duplicate_retries < 100) {

        node *candidate_sub = nullptr;
        bool created_new = false;
        beam_solver::phase_result pres = execute_current_phase(slot, target, batch_count, candidate_sub, created_new);

        if (pres == phase_result::yield) return false;
        if (pres == phase_result::next_phase) {
          m_gen_phase++;
          reset_generator_counters();
          break;
        }
        if (pres == phase_result::retry) continue;
        if (pres == phase_result::break_loop) break;
        if (!candidate_sub) break;

        bool added = add_candidate_if_new(base, slot.depth, candidate_sub, created_new);
        if (!added) {
          duplicate_retries++;
          continue;
        }

        if (++batch_count >= MAX_BATCH_SIZE) return false;
        if (total_candidates_this_slot >= MAX_CANDIDATES_PER_SLOT) {
          beam_idx = (int)beam.size();
          return true;
        }
        break;
      }
    }
    return true;
  }

  inline bool beam_solver::add_candidate_if_new(node* base, int depth, node* candidate_sub, bool created_new) {
    // Early-reject duplicates *before* materializing the candidate tree.
    // predict_replaced_hash walks only the spine (typically 2-5 nodes) and
    // reproduces environment::update_hash exactly, so the predicted hash is
    // bit-identical to what copy_and_replace + update_hash would yield.
    // Most candidates are duplicates in this workload, so this elides a
    // copy_and_replace + remove pair on every miss.
    uint64_t predicted = predict_replaced_hash(base, depth, candidate_sub->hash);
    if (visited.contains(predicted)) {
      if (created_new) environment::remove(candidate_sub);
      return false;
    }

    node *candidate = copy_and_replace(base, depth, candidate_sub);
    if (created_new) environment::remove(candidate_sub);

    // Tile-op limit needs the materialized tree to walk; checked after build.
    if (count_tile_ops(candidate) > MAX_TILE_OPS) {
      environment::remove(candidate);
      return false;
    }

    if (visited.size() >= MAX_VISITED_NODES) {
      visited.clear();
      log_debug("Visited nodes set full, cleared to maintain constant memory.");
    }
    // insert_if_new collapses the visited.insert into a single probe walk.
    // Result is unused — predict matched the actual hash by construction, so
    // the second contains-check above already proved this is novel.
    visited.insert_if_new(candidate->hash);
    // Ownership contract:
    //   current_slot_candidates owns a ref (via copy below).
    //   pending_evaluation holds a raw borrow — no ref, no remove on dequeue.
    //   prev_original_nodes takes its own ref (copy at dequeue time).
    // ~search_state releases current_slot_candidates last, after pending is drained.
    // candidate has ref_count=1 from copy_and_replace; transfer that ref to
    // current_slot_candidates (the owner). pending_evaluation borrows.
    current_slot_candidates.push_back({candidate, 0.0f, 0});
    pending_evaluation.push(candidate);
    total_candidates_this_slot++;

    // Explicit deposit twin: a chain that doesn't end in a canvas op scores
    // as whatever it deposited so far — usually nothing — so its family dies
    // one op before the deposit that would reveal its worth (009d5c81:
    // "InputPalette ShapeColorMap" scored -148, its deposited form 922 correct
    // cells). The twin scores DIFFERENTLY from the bare chain, so it can win
    // its own beam seat — a score-alias variant (evaluate bare as deposited,
    // one program) was tried 2026-07-13 and starved the deposited trees: ties
    // never beat incumbents, so mid-chain deposits stopped earning seats.
    if (!modifies_canvas(candidate->type) &&
        candidate->max_depth + 1 <= MAX_SEARCH_DEPTH) {
      node *twin = create_unary_node(node_type::Underlay, candidate);
      if (visited.insert_if_new(twin->hash)) {
        current_slot_candidates.push_back({twin, 0.0f, 0});
        pending_evaluation.push(twin);
        total_candidates_this_slot++;
      } else {
        environment::remove(twin);
      }
    }
    return true;
  }

  inline void beam_solver::reset_generator_counters() {
    m_gen_primitive_idx = 0;
    m_gen_outer_idx = 0;
    m_gen_inner_idx = 0;
    m_dry_yields = 0;
    m_last_cand_at_yield = total_candidates_this_slot;
  }

  // Bucket by the WHOLE last sentence — its four ops and their literals. The old
  // key was the outermost two ops, which on a sentence is (COMBINE, ACT): it
  // threw away every GET and SELECT choice, so a 107,520-sentence sweep
  // collapsed to 196 survivors picked on half the slots.
  // A short chain keys on what it has; 0xff pads the beats it does not.
  static inline uint64_t bucket_key(const node* n) {
    uint64_t key = 0;
    const node* cur = n;
    for (int beat = 0; beat < 4; ++beat) {
      const uint64_t type = cur ? (uint8_t)cur->type : 0xff;
      const uint64_t value = cur ? (uint8_t)(cur->value & 0xff) : 0xff;
      key = (key << 16) | (type << 8) | value;
      if (cur) cur = cur->arg;
    }
    return key;
  }

  inline void beam_solver::merge_slot_candidates() {
    for (auto &cand : current_slot_candidates) {
      auto it = global_cache.find(cand.n->hash);
      if (it != global_cache.end()) {
        cand.score = it->second.score;
        cand.state_hash = it->second.state_hash;
      }
    }

    // STATE DEDUP: two chains in the same state are one chain, and a chain's
    // state is the canvas it painted AND the objects it still holds. The hash
    // covers both (see dsl_score_kernel) -- covering the canvas alone made a
    // sentence that only rearranges the hand look like a sentence that did
    // nothing, and this deleted it. state_hash==0 means no output came back
    // yet, so those are kept and the score-based pruning handles them.
    // Two chains in one state are one answer, so keep the one that RANKS
    // higher rather than the one that happened to arrive first. Arrival
    // carries no information -- and whether a candidate even has a hash by now
    // depends on whether its batch came back, so letting arrival choose lets
    // GPU timing pick the spelling that reaches the beam.
    static thread_local unordered_map<uint64_t, size_t> best_by_state;
    best_by_state.clear();
    int dropped = 0;
    for (auto &cand : current_slot_candidates) {
      if (!cand.n) continue;
      if (cand.state_hash == 0) {
        level_candidates.push_back(cand);
        continue;
      }
      auto seat = best_by_state.find(cand.state_hash);
      if (seat == best_by_state.end()) {
        // A canvas first seen in an EARLIER merge of this level has no
        // contender left to compare against; its winner is already downstream.
        if (!visited_states.insert_if_new(cand.state_hash)) {
          environment::remove(cand.n);
          cand.n = nullptr;
          dropped++;
          continue;
        }
        best_by_state[cand.state_hash] = level_candidates.size();
        level_candidates.push_back(cand);
        continue;
      }
      champion &held = level_candidates[seat->second];
      if (ranks_higher(cand, held)) {
        environment::remove(held.n);
        held = cand;
      } else {
        environment::remove(cand.n);
        cand.n = nullptr;
      }
      dropped++;
    }
    if (dropped > 0) {
      log_debug(to_text("[STATE-DEDUP] dropped=%d kept=%d grids_seen=%zu",
        dropped, (int)level_candidates.size(), visited_states.size()));
    }
    current_slot_candidates.clear();

    // Per-slot scratch. All three reused across slots (no per-slot realloc).
    // Bounds are structural, not heuristic, so no eviction policy is needed:
    //   best_idx_by_type: one per distinct last sentence -> bounded by the sweep
    //   kept:             one per bucket      -> bounded by best_idx_by_type
    //   keep_mask:        one per candidate   -> bounded by level cap (200k)
    // clear()/assign() at the TOP so no stale state from a prior slot leaks
    // (best_idx_by_type holds indices into the *current* level_candidates).
    static thread_local unordered_map<uint64_t, size_t> best_idx_by_type;
    best_idx_by_type.clear();
    for (size_t i = 0; i < level_candidates.size(); ++i) {
      auto &cand = level_candidates[i];
      if (!cand.n) continue;
      uint64_t k = bucket_key(cand.n);
      auto it = best_idx_by_type.find(k);
      if (it == best_idx_by_type.end() || ranks_higher(cand, level_candidates[it->second])) {
        best_idx_by_type[k] = i;
      }
    }

    static thread_local vector<champion> kept;
    kept.clear();
    static thread_local vector<char> keep_mask;
    keep_mask.assign(level_candidates.size(), 0);
    for (auto &it : best_idx_by_type) keep_mask[it.second] = 1;
    for (size_t i = 0; i < level_candidates.size(); ++i) {
      if (keep_mask[i]) {
        kept.push_back(level_candidates[i]);
      } else if (level_candidates[i].n) {
        environment::remove(level_candidates[i].n);
      }
    }
    // swap (not move-assign): move-assign would steal kept's buffer into
    // level_candidates, leaving the thread_local kept with no capacity and
    // forcing a realloc next slot. swap ping-pongs the two buffers so both
    // stay allocated; the subsequent kept.clear() (next call, at top) empties
    // the buffer it now owns (the old level_candidates storage).
    level_candidates.swap(kept);

    // Hard cap. level_candidates accumulates across every slot in a level and
    // is only cleared at the level boundary; the type-bucket prune above
    // shrinks it but does not bound it (diverse buckets => still unbounded).
    // The dump crashed at 588k. This is a funnel that distills to <=1024
    // (BEAM_WIDTH) anyway, so dropping the lowest-score entries when huge is
    // semantically almost free. Keep the top MAX_LEVEL_CANDIDATES by score;
    // release the rest so their node subtrees can be reclaimed.
    constexpr size_t MAX_LEVEL_CANDIDATES = 200000;
    if (level_candidates.size() > MAX_LEVEL_CANDIDATES) {
      std::nth_element(
          level_candidates.begin(),
          level_candidates.begin() + MAX_LEVEL_CANDIDATES,
          level_candidates.end(), ranks_higher);
      for (size_t i = MAX_LEVEL_CANDIDATES; i < level_candidates.size(); ++i)
        if (level_candidates[i].n) environment::remove(level_candidates[i].n);
      level_candidates.resize(MAX_LEVEL_CANDIDATES);
    }
  }

  inline void beam_solver::commit_level_champion() {
    if (level_candidates.empty()) {
      return;
    }

    auto top_it = max_element(
        level_candidates.begin(), level_candidates.end(),
        [](const champion &a, const champion &b) { return ranks_higher(b, a); });
    float top_score = top_it->score;

    if (top_score > all_time_best_score) {
      all_time_best_score = top_score;
    }

    // Per-level scratch, reused (no per-level realloc). Structurally bounded:
    // best_by_type holds one entry per distinct last sentence, so the sweep
    // bounds it, and it already evicts within-bucket (lower score removed below). clear() at
    // top: the owned copies it held were transferred into beam last call, so
    // clearing only drops now-stale pointers (no remove -> no double free).
    // clear() keeps the grown bucket table, and bucket count sets iteration
    // order — safe only because ranks_higher leaves no tie for order to break.
    static thread_local unordered_map<uint64_t, champion> best_by_type;
    best_by_type.clear();
    for (auto &cand : level_candidates) {
      if (!cand.n) continue;
      uint64_t k = bucket_key(cand.n);
      auto it = best_by_type.find(k);
      if (it == best_by_type.end() || ranks_higher(cand, it->second)) {
        if (it != best_by_type.end() && it->second.n) {
          environment::remove(it->second.n);
        }
        best_by_type[k] = {environment::copy(cand.n), cand.score, cand.state_hash};
      }
    }

    // Reused per level. Built from best_by_type (<=65,536), then evicted down
    // to BEAM_WIDTH (1024) by partial_sort below — that IS the eviction
    // policy (keep top-BEAM_WIDTH by score, remove the rest).
    static thread_local vector<champion> next_beam;
    next_beam.clear();
    next_beam.reserve(best_by_type.size());
    for (auto &it : best_by_type)
      next_beam.push_back(it.second);

    auto cmp = ranks_higher;
    if ((int)next_beam.size() > BEAM_WIDTH) {
      sort(next_beam.begin(), next_beam.end(), cmp);
      for (size_t i = BEAM_WIDTH; i < next_beam.size(); ++i)
        if (next_beam[i].n) environment::remove(next_beam[i].n);
      next_beam.resize(BEAM_WIDTH);
    } else {
      sort(next_beam.begin(), next_beam.end(), cmp);
    }

    release_candidates(beam);
    // swap not copy: beam takes next_beam's buffer; next_beam is left holding
    // the just-released old-beam pointers, harmlessly dropped by the clear()
    // at the top of the next call (no remove on those, no double free).
    beam.swap(next_beam);
    release_candidates(level_candidates);

    log_debug(
        to_text("Level committed. Beam top: %.2f  All-time best: %.2f",
               (double)top_score, (double)all_time_best_score));
  }

  static inline void build_example_upload(
      cuda_dsl_example_data& ex,
      const puzzle& p,
      int active_example_count) {
    int active = min(active_example_count, (int)p.train.size());
    ex.count = active;
    for (int ei = 0; ei < active; ++ei)
      fill_example_slot(ex, ei, p.train[ei], p);
    environment::fill_puzzle_level(ex, p);
  }

  static inline void pack_cuda_dsl_exec_batch(
      cuda_dsl_exec_batch& batch,
      const beam_solver::flat_candidate_chunk& chunk,
      int active) {
    int job_count = chunk.size() * active;
    batch.resize(job_count);
    batch.active_examples = active;

    for (int ci = 0; ci < chunk.size(); ++ci) {
      gpu_flat_op gpu_ops[GPU_MAX_OPS];
      int gpu_op_count = convert_program_to_gpu_ops(chunk.flat_programs[ci], gpu_ops);
      int8_t nc = (int8_t)min(chunk.node_counts[ci], 127);

      for (int ei = 0; ei < active; ++ei) {
        int j = ci * active + ei;

        gpu_flat_op* dst_ops = batch.programs.data() + j * GPU_MAX_OPS;
        for (int o = 0; o < gpu_op_count; ++o) dst_ops[o] = gpu_ops[o];
        for (int o = gpu_op_count; o < GPU_MAX_OPS; ++o) dst_ops[o] = {node_type::Input, 0};
        batch.program_lengths[j] = (int8_t)gpu_op_count;
        batch.node_counts[j] = nc;
        batch.program_indices[j] = (int16_t)ci;
      }
    }
  }

  inline void beam_solver::log_tree_record(node *tree, float score, long long attempts) {
    m_log.tree_record(tree, score, attempts);
  }

  inline void beam_solver::log_score_formula(const node *tree, const vector<gpu_score_terms> &terms) {
    if (terms.empty())
      return;
    string tree_str = environment::stringify_tree_via_flat(tree);
    for (int i = 0; i < (int)terms.size(); ++i)
      log_debug(terms[i].say(tree_str, i));
  }

  inline void beam_solver::update_best_solution(int load_id, puzzle &p, node *tree,
                                   int node_count, beam_solver::candidate_batch &res) {
    int active = min(active_example_count, (int)p.train.size());

    // WHICH DOOR, IF ANY, LETS THIS TREE IN AS THE BEST. Asked twice: once
    // without the lock, which turns most candidates away cheaply, and again
    // under it, since the best may have moved in between.
    struct gate {
      bool old_tree_solves, overrides, score_better, score_same, nodes_better, should_promote;
      float best_sum;
      bool opens() const {
        return overrides || score_better || (score_same && nodes_better) || should_promote;
      }
    };
    auto ask_gate = [&] {
      gate g;
      g.old_tree_solves = check_old_tree_solves(p, active);
      g.overrides = res.all_train_solved && !g.old_tree_solves;
      g.best_sum = 0;
      for (int i = 0; i < active; ++i)
        g.best_sum += p.best_score(false, i);
      // SOLVING OUTRANKS RESEMBLING. Nothing that fails the active examples may
      // displace something that passes them, however high it scores -- otherwise
      // the two doors trade the best tree back and forth forever, each undoing
      // the other: one asks who scores higher, the other asks who solves.
      bool would_lose_a_solve = g.old_tree_solves && !res.all_train_solved;
      g.score_better = !would_lose_a_solve &&
                       res.total_train_score > g.best_sum + 0.0001f;
      // Only treat "same score" as a valid simplification trigger if the new
      // tree is actually correct on all active examples — tying on a rarity-
      // weighted score does not imply functional equivalence.
      g.score_same = res.all_train_solved &&
                     abs(res.total_train_score - g.best_sum) < 0.0001f;
      g.nodes_better = node_count < p.best_node_count;
      g.should_promote = res.all_train_solved && (active < (int)p.train.size());
      return g;
    };

    if (!ask_gate().opens()) {
      try_update_current_display(load_id, p, tree, node_count, res);
      return;
    }
    {
      unique_lock lock(m_progress.mtx);
      if (load_id != m_load_id)
        return;
      const gate g = ask_gate();
      if (!g.opens())
        return;

      // Which door opened, so a gate that keeps reopening has to say why.
      log_debug(to_text("[GATE] all_solved=%d old_solves=%d overrides=%d better=%d same=%d fewer=%d "
                        "promote=%d score=%.4f best=%.4f nodes=%d best_nodes=%d active=%d",
                        (int)res.all_train_solved, (int)g.old_tree_solves, (int)g.overrides,
                        (int)g.score_better, (int)g.score_same, (int)g.nodes_better,
                        (int)g.should_promote, (double)res.total_train_score,
                        (double)g.best_sum, node_count, p.best_node_count, active));

      // Test eval happens in the train-finale branch below via the CPU
      // evaluate_test_after_solve path — no GPU scratch launches needed.

      int promotion_count = 0;
      bool is_train_finale = false;
      bool is_test_solve = false;

      if (res.all_train_solved && (active < (int)p.train.size())) {
        promotion_count = try_promote_examples(tree, p, active, node_count);
        // After active grows, the main GPU template cache still only holds the
        // pre-promotion prefix. The next batch eval would score against stale
        // template slots for the newly-active examples and could report
        // solved=true for a candidate that doesn't actually solve. Invalidate
        // so the next pack_cuda_dsl_exec_batch re-uploads the full active set.
        if (promotion_count > 0) cuda_dsl_invalidate_examples();
        if (active == (int)p.train.size()) {
          is_train_finale = !p.is_solved;
          is_test_solve = evaluate_test_after_solve(tree, p, node_count) && !p.is_test_solved;
        } else {
          clear_views(p, true);
        }
      } else if (res.all_train_solved && (active == (int)p.train.size()) && !p.is_solved) {
        is_train_finale = true;
        is_test_solve = evaluate_test_after_solve(tree, p, node_count) && !p.is_test_solved;
      }

      if (promotion_count == 0 && !is_train_finale && !is_test_solve) {
        if (!g.overrides && !g.score_better && !(g.score_same && g.nodes_better)) {
          update_example_results(false, res.train);
          bool full_train = res.all_train_solved && active == (int)p.train.size();
          update_test_current_views(p, tree, node_count, full_train);
          environment::remove(m_progress.current_tree);
          m_progress.current_tree = environment::copy(tree);
          changed();
          return;
        }
      }

      bool improvement_sound = (promotion_count == 0) && !is_train_finale && !is_test_solve && (g.score_better || (g.score_same && g.nodes_better));

      // res was evaluated against the pre-promotion active count; only the
      // first (active - promotion_count) entries in res.train are valid.
      // try_promote_examples already showed the promoted ones, so don't let
      // update_example_results clobber them.
      update_example_results(false, res.train, active - promotion_count);
      {
        bool full_train = res.all_train_solved && active == (int)p.train.size();
        update_test_current_views(p, tree, node_count, full_train);
      }
      changed();

      // Leave-one-out reduction: strip dead/no-op ops from this new best so the
      // stored best AND the tree the beam continues from are as short as possible.
      // Greedy, strictly-score-improving (node penalty makes a no-op a strict win),
      // solved-set preserving. Returns `tree` unchanged when nothing helps; returns
      // a freshly-owned reduced tree otherwise (which we must free at the end).
      int reduced_node_count = node_count;
      // Full-train solves get the full-depth reduction: this tree is the one
      // that gets stored, displayed, and run on test — shortest wins there.
      bool deep = res.all_train_solved && active == (int)p.train.size();
      node *reduced = reduce_best_tree(tree, p, active, reduced_node_count, deep);
      bool was_reduced = (reduced != tree);
      if (was_reduced)
        log_debug(to_text("!!!! REDUCED: %d -> %d nodes (leave-one-out) !!!!",
                              node_count, reduced_node_count));

      environment::remove(p.best_tree);
      p.best_tree = environment::copy(reduced);
      invalidate_best_tree_cache(p);

      // The tree and how many nodes it has are one fact, so they are written together.
      int previous_node_count = p.best_node_count;
      p.best_node_count = reduced_node_count;

      refresh_best_display_scores(p, active, reduced_node_count);

      environment::remove(m_progress.current_tree);
      m_progress.current_tree = environment::copy(reduced);

      // Free the reduced tree if reduce_best_tree allocated a new one (the two
      // copy() calls above took their own refs; this drops our local owner).
      if (was_reduced) environment::remove(reduced);

      bool is_simplification = (promotion_count == 0) && !is_train_finale && !is_test_solve && g.score_same && g.nodes_better && p.is_solved;
      if (is_train_finale || is_simplification) {
        if (is_train_finale) {
          log_debug("!!!! TASK SOLVED ON TRAINING SET !!!");
          p.is_test_solved = is_test_solve;
          p.is_solved = true;
        } else {
          log_debug(to_text("!!!! SIMPLIFIED SOLUTION FOUND: %d nodes (was %d) !!!!", reduced_node_count, previous_node_count));
        }
      } else if (promotion_count == 0 && !is_test_solve) {
        // Not a solve and not a simplification, just a new best, so say that.
        log_debug(to_text("!!!! NEW BEST: %d nodes (was %d) !!!!", reduced_node_count, previous_node_count));
      }

      if (is_test_solve) {
        p.is_test_solved = true;
        log_debug("!!!! TEST SET SOLVED CORRECTLY !!!!");
      }

      bool final_is_simplification = (promotion_count == 0) && !is_train_finale && !is_test_solve && g.score_same && g.nodes_better && p.is_solved;
      lock.unlock();

      if (improvement_sound || promotion_count > 0 || is_train_finale || is_test_solve || final_is_simplification) {
        search_news n;
        n.promotions = promotion_count;
        n.improved = improvement_sound;
        n.simplified = final_is_simplification;
        n.train_solved = is_train_finale;
        n.test_solved = is_test_solve;
        news(n);
      }
    }
  }

  inline bool beam_solver::load(const challenge_task &task, const vector<grid> *test_solutions,
                                const puzzle_label &label) {
    m_log.clear();
    cuda_dsl_invalidate_examples();

    {
      scoped_lock lock(m_progress.mtx);

      // Load UNDER the lock: this replaces m_puzzle, which a watcher on another
      // thread reads under this same lock. Loading outside it races that
      // reader, which could draw the new puzzle against a half-swapped one.
      m_puzzle_loaded = load_task_from_content(task.content, task.id, m_puzzle, test_solutions);
      m_label = label;
      if (m_puzzle_loaded) m_log.set_puzzle_id(m_puzzle.id);
      if (m_observer) m_observer->loaded(m_puzzle);

      m_progress.attempts = 0;
      environment::remove(m_progress.current_tree);
      m_progress.current_tree = nullptr;
      m_progress.start_time = chrono::steady_clock::now();

      active_example_count = start_example_count;
      asked_inference_yet = false;
      ++m_load_id;
      changed();

      if (m_puzzle_loaded) {
          puzzle &p = m_puzzle;
          // A puzzle just parsed has no best and no scores yet; this says so
          // to whoever is watching, and puts the search's own copies of the
          // scores at their floor.
          clear_views(p, false);
          clear_views(p, true);
          if (active_example_count > (int)p.train.size())
            active_example_count = (int)p.train.size();
          invalidate_best_tree_cache(p);
          p.best_node_count = NODE_COUNT_MAX;
          m_progress.current_attempt_depth = 0;
          m_progress.current_attempt_nodes = 0;
      }
    }
    reset_search();
    drain_pipeline();
    return m_puzzle_loaded;
  }

  inline void beam_solver::drain_pipeline() {
    for (int s = 0; s < 2; ++s) {
      if (m_slot_pending[s]) cuda_dsl_wait_for_exec(s);
      for (node *n : m_prev_nodes[s]) environment::remove(n);
      m_prev_nodes[s].clear();
      // The deferred winner holds a tree too, and node_pool.release() below
      // frees the pool out from under it.
      if (m_deferred[s].tree) environment::remove(m_deferred[s].tree);
      m_deferred[s].tree = nullptr;
      m_deferred[s].valid = false;
      m_slot_pending[s] = false;
    }
    node_pool.release();
  }

  inline void beam_solver::read_facts() {
    // THE FACTS ARE READ WHEN SOMEONE WANTS THEM. Stepping through puzzles
    // without searching costs only the parse; searching, or running a program
    // on the puzzle, asks for them, once. Under the lock, since a watcher
    // reads the same puzzle.
    if (!m_puzzle_loaded || m_puzzle.inferred) return;
    scoped_lock lock(m_progress.mtx);
    infer_into_puzzle(m_puzzle, m_label, m_options.write_logs);
    changed();
  }

  inline void beam_solver::infer() {
    if (!m_puzzle_loaded) return;
    read_facts();

    // Try what the looking spelled before spending a single beam candidate.
    if (asked_inference_yet) return;
    bool solved = false;
    {
      scoped_lock lock(m_progress.mtx);
      asked_inference_yet = true;
      if (!m_puzzle.is_solved && try_what_inference_said(m_puzzle)) {
        active_example_count = (int)m_puzzle.train.size();
        cuda_dsl_invalidate_examples();
        environment::remove(m_progress.current_tree);
        m_progress.current_tree = environment::copy(m_puzzle.best_tree);
        changed();
        solved = true;
      }
    }
    // Said with the lock let go: a driver that pauses on a solve may take it.
    if (solved) {
      search_news n;
      n.from = search_news::source::inference;
      n.train_solved = true;
      n.test_solved = m_puzzle.is_test_solved;
      news(n);
    }
  }

  inline void beam_solver::step() {
    if (!m_puzzle_loaded) return;
    // A puzzle inference just solved hands the turn back, so the driver hears
    // of it before a single candidate is spent hunting for a shorter one.
    const bool asked_before = asked_inference_yet;
    infer();
    if (!asked_before && m_puzzle.is_solved) return;

    puzzle &p = m_puzzle;
    if (p.train.empty())
      return;

    // Promote examples FIRST — BEFORE sizing the chunk. If the best tree already
    // solves the active example, promotion bumps active_example_count. Since the
    // grid is sized by job_count = chunk * active, sizing before promotion would let
    // active grow afterward and blow job_count past TARGET_JOBS -> device cudaMalloc
    // overshoot (unchecked) -> sticky CUDA error -> the GUI's cudaMallocHost aborts.
    if (p.best_tree && active_example_count < (int)p.train.size()) {
      unique_lock pl(m_progress.mtx);
      if (p.best_tree && active_example_count < (int)p.train.size()) {
        int cur = active_example_count - 1;
        if (cur >= 0 && cur < (int)p.train.size()) {
          auto [_, cslv] = eval_best_tree_train(p, cur);
          if (cslv) {
            int before = active_example_count;
            int active_ref = active_example_count;
            try_promote_examples(p.best_tree, p, active_ref, p.best_node_count);
            active_example_count = active_ref;
            if (active_example_count > before) {
              cuda_dsl_invalidate_examples();
              refresh_best_display_scores(p, active_example_count, p.best_node_count);
              changed();
            }
          }
        }
      }
    }

    // A promotion filled the train set with a verified tree (recorded in
    // try_promote_examples, from ANY promotion path — best-channel, layer
    // lock, or the block above). Commit it as the best and mark the task
    // solved.
    if (solve_found_by_promotion) {
      unique_lock fl(m_progress.mtx);
      if (!p.is_solved) {
        environment::remove(p.best_tree);
        p.best_tree = environment::copy(solve_found_by_promotion);
        p.best_node_count = solve_found_node_count;
        invalidate_best_tree_cache(p);
        cuda_dsl_invalidate_examples();
        refresh_best_display_scores(p, active_example_count, p.best_node_count);
        environment::remove(m_progress.current_tree);
        m_progress.current_tree = environment::copy(p.best_tree);
        changed();
        log_debug("!!!! SOLVE FOUND BY PROMOTION: marking task solved !!!!");
        finalize_full_train_solve(p);
      }
      environment::remove(solve_found_by_promotion);
      solve_found_by_promotion = nullptr;
    }

    // CHUNK ACCUMULATION. The GPU is the bottleneck and is starved at ~1 wave
    // (~222 blocks) per launch, while the CPU generates ~30x faster than the GPU
    // consumes (stub test: stub-off ~80k aps vs stub-on ~2.6M aps => 32x GPU-bound).
    // So accumulate a FULL multi-wave grid before launching. We size by target JOB
    // count (= chunk * active), since the grid AND VRAM scale with job_count and it
    // is independent of how many train examples are active. ~2048 jobs ≈ 8.5 waves
    // (240 blocks/wave at 12 blocks/SM), ~1.65GB VRAM across both pipeline slots.
    // This is also why throughput "felt faster after the first levels": early levels
    // have active=1 (few jobs => 1 wave); accumulation fixes the starvation directly.
    const int active_now = std::max(1, std::min(active_example_count, (int)p.train.size()));
    const int TARGET_JOBS = 2048;  // generation-bound: the chunk maxes ~1800 jobs (a beam level runs
                                   // out first), so higher buys nothing. Measured peak ~1.5GB VRAM, 0 faults.
    const int chunk_cap = std::max(1, std::min(TARGET_JOBS / active_now,
                                               flat_program_batch::MAX_PROGRAMS));

    // Prime the generator.
    { scope_ns generate_time{ns_generate};
      pump_generate(&m_progress.attempts); }

    if (pending_evaluation.empty()) {
      scope_ns idle_time{ns_idle};
      if (!is_generating_layer)
        this_thread::sleep_for(chrono::milliseconds(1));
      return;
    }

    // Reused across iterations (clear keeps capacity). trees are nulled at the
    // end of the iteration, so clear() only frees genuinely-owned leftovers.
    static thread_local flat_candidate_chunk chunk;
    chunk.clear();
    chunk.reserve(chunk_cap);
    static thread_local vector<node*> original_nodes;
    original_nodes.clear();
    original_nodes.reserve(chunk_cap);

    // Accumulate up to chunk_cap candidates, re-pumping the generator whenever
    // pending drains, so the GPU gets a full grid. STOP at a beam-level boundary:
    // pump_generate leaves pending empty after committing a level, and we must NOT
    // mix level N+1 candidates into this batch (level N+1 needs level N's scores).
    { scope_ns generate_time{ns_generate};
    auto asked_for_enough = [&] {
      return stop_after_attempts > 0 &&
             m_progress.attempts.load() >= stop_after_attempts;
    };
    while (!asked_for_enough() && (int)chunk.trees.size() < chunk_cap) {
      while (!asked_for_enough() && (int)chunk.trees.size() < chunk_cap &&
             !pending_evaluation.empty()) {
        node *orig = pending_evaluation.front();
        pending_evaluation.pop();
        if (!orig) continue;

        if (global_cache.count(orig->hash)) { continue; }

        flat_program program;
        if (!environment::flatten_unary_program(orig, program)) { continue; }

        int nc = environment::count_nodes(orig);
        if (!chunk.programs->push(program)) { break; }
        chunk.trees.push_back(orig);
        chunk.flat_programs.push_back(program);
        chunk.node_counts.push_back(nc);
        original_nodes.push_back(environment::copy(orig));
        m_progress.attempts++;
        m_progress.current_max_depth = MAX_SEARCH_DEPTH;
        m_progress.current_attempt_depth = orig->max_depth;
      }
      if ((int)chunk.trees.size() >= chunk_cap) break;
      // Chunk not full and pending drained: generate more of the CURRENT level.
      pump_generate(&m_progress.attempts);
      if (pending_evaluation.empty()) break;  // level committed / dry -> launch what we have
    }
    }

    if (chunk.size() == 0) return;

    // active is now final (promotion ran before chunk sizing above), so
    // job_count = chunk.size() * active <= chunk_cap * active <= TARGET_JOBS.
    const int active = min(active_example_count, (int)p.train.size());

    static thread_local cuda_dsl_exec_batch exec_batch[2];
    static thread_local int current_slot = 0;
    bool (&slot_pending)[2] = m_slot_pending;
    vector<node*> (&prev_original_nodes)[2] = m_prev_nodes;

    int launch_slot = current_slot;
    int wait_slot = 1 - current_slot;

    if (!cuda_dsl_examples_uploaded()) {
      // The previous batch (wait_slot) may STILL be in flight reading the old
      // example templates. Promotion bumped `active` and invalidated examples, so
      // we re-upload here — but overwriting d_example_templates while that batch
      // runs corrupts its scores/solved flags. That surfaces as runaway spurious
      // "SIMPLIFIED SOLUTION"/promotion spam and garbage predicted grids (the
      // "halt then madness" after a promotion). Sync the in-flight batch first so
      // it has finished reading before we overwrite; its results were already
      // written by the kernel, so the later wait+process of this slot stays valid.
      if (slot_pending[wait_slot]) cuda_dsl_wait_for_exec(wait_slot);
      static thread_local cuda_dsl_example_data example_data;
      build_example_upload(example_data, p, active_example_count);
      cuda_dsl_upload_examples(example_data);
    }

    // Launch the current batch FIRST so the GPU starts immediately, then do
    // the CPU-side processing of the PREVIOUS batch while this one runs on
    // the GPU. That host/GPU overlap is the whole point of the two-slot
    // pipeline. The launch_slot buffers were drained the previous iteration
    // (when this slot was the wait_slot), so overwriting them is safe.
    { scope_ns launch_time{ns_launch};
      pack_cuda_dsl_exec_batch(exec_batch[launch_slot], chunk, active);
      cuda_dsl_exec_and_score(exec_batch[launch_slot], launch_slot); }

    // Wait on and process the previous batch (the other slot) while the batch
    // just launched runs on the GPU. The wait is now per-slot, so BOTH slots
    // are synchronized before their results are read — the old single
    // slot_0_pending flag skipped this for slot 1, an unsynced read.
    if (slot_pending[wait_slot] && prev_original_nodes[wait_slot].size() > 0) {
      { gpu_wait_scope t{m_progress};
        cuda_dsl_wait_for_exec(wait_slot); }
      scope_ns harvest_time{ns_harvest};
      // GPU fault detector (permanent). CUDA errors are sticky + cudaGetLastError
      // surfaces the error from ANY preceding call, so this one check is a catch-all
      // for every kernel/memcpy/malloc. If "magic madness" returns (frozen/garbled
      // grids, score spam), grep the puzzle log for "!!! GPU ERROR" — it names the
      // CUDA error + which launch. (The real fix is the bug; this makes it loud.)
      { char _ce[512]; if (cuda_dsl_take_error(_ce, sizeof(_ce))) {
          log_debug(to_text("!!! GPU ERROR (results now garbage until restart): %s", _ce)); } }
      slot_pending[wait_slot] = false;
      // Results from previous batch are ready. Process them.
      cuda_dsl_exec_batch& prev_batch = exec_batch[wait_slot];
      int program_count = prev_batch.job_count;
      // Physically bounded: job_count = chunk.size() * active ≈ TARGET_JOBS (2048)
      // by construction (chunk_cap = TARGET_JOBS/active). Reused across iterations;
      // assign() re-inits the fill value and keeps capacity so this never
      // reallocates after the first warmup pass.
      static thread_local vector<float> program_totals;
      static thread_local vector<char> program_all_solved;
      program_totals.assign((size_t)program_count, 0.0f);
      program_all_solved.assign((size_t)program_count, 1);

      int winner_idx = -1;
      float winner_score = SCORE_MIN;
      bool winner_solved = false;
      int node_count = (int)prev_original_nodes[wait_slot].size();

      { scope_ns tally_time{ns_tally};
      for (int j = 0; j < prev_batch.job_count; ++j) {
        int pidx = prev_batch.program_indices[j];
        program_totals[pidx] += prev_batch.scores[j];
        if (!prev_batch.solved[j]) program_all_solved[pidx] = false;
      }

      for (int i = 0; i < node_count; ++i) {
        float score = program_totals[i];
        // What this candidate PAINTED, on the first active pair. Recorded for
        // every candidate, not just the winner, so the grid dedup below can
        // see two spellings of one answer and keep only one.
        {
          int first_job = i * active;
          if (first_job >= 0 && first_job < prev_batch.job_count)
            report_score(prev_original_nodes[wait_slot][i], score,
                         prev_batch.output_hashes[first_job]);
        }
        log_tree_record(prev_original_nodes[wait_slot][i], score,
                        m_global_attempts ? m_global_attempts->load() : 0);
        bool solved = program_all_solved[i];
        bool better = solved && !winner_solved;
        if (!better && solved == winner_solved && score > winner_score)
          better = true;
        if (better) {
          winner_score = score;
          winner_solved = solved;
          winner_idx = i;
        }
      }
      }

      // Promotion during the PREVIOUS batch's processing can grow active while
      // THIS batch is still in flight. That batch was packed+scored against the
      // OLD example set (prev_batch.active_examples), so indexing its results with
      // the new `active` reads the wrong candidate's scores/grids AND its
      // all_train_solved (over fewer examples) could falsely "solve"/promote. If
      // the batch's active no longer matches, it's stale — skip its winner update
      // (scores were already tallied/logged; the candidates get re-evaluated at the
      // new active). Bigger batches stay in flight longer, so this fired routinely
      // at high TARGET_JOBS and confounded scoring/promotion (the "madness").
      bool batch_stale = (prev_batch.active_examples != active);

      // The picture asked for last time round this slot has landed: the copy
      // rides that slot's stream, so the wait above ordered it. Read it BEFORE
      // asking for the next one.
      if (m_deferred[wait_slot].valid) {
        scope_ns winner_time{ns_winner};
        deferred_winner &done = m_deferred[wait_slot];
        const int8_t *grids = cuda_dsl_winner_grids(wait_slot);
        const int8_t *dims  = cuda_dsl_winner_dims(wait_slot);

        static thread_local candidate_batch winner_res;
        winner_res.train.assign(p.train.size(), candidate_result{});
        winner_res.total_train_score = 0.0f;
        winner_res.all_train_solved = true;
        static thread_local vector<gpu_score_terms> winner_breakdowns;
        winner_breakdowns.clear();
        winner_breakdowns.reserve(done.active);

        { scope_ns analyze_time{ns_analyze};
        for (int i = 0; i < done.active && grids && dims; ++i) {
          const int8_t *og = grids + (size_t)i * GPU_GRID_CELLS;
          int ow = dims[i * 2 + 0];
          int oh = dims[i * 2 + 1];
          if (ow < 0 || ow > GPU_MAX_GRID || oh < 0 || oh > GPU_MAX_GRID) { ow = 0; oh = 0; }
          winner_res.train[i].state.width = ow;
          winner_res.train[i].state.height = oh;
          for (int r = 0; r < oh; ++r)
            for (int c = 0; c < ow; ++c)
              winner_res.train[i].state.data[r][c] = (int)og[r * GPU_MAX_GRID + c];
          winner_res.train[i].valid = true;
          winner_res.train[i].score = done.score[i];
          winner_res.total_train_score += done.score[i];
          if (!done.solved[i]) winner_res.all_train_solved = false;

          winner_breakdowns.push_back(done.terms[i]);
        }
        }

        uint64_t g_hash = (!winner_res.train.empty() && winner_res.train[0].valid)
            ? grid_hash(winner_res.train[0].state) : 0;
        if (done.tree) {
          log_score_formula(done.tree, winner_breakdowns);
          report_score(done.tree, winner_res.total_train_score, g_hash);
          { scope_ns best_time{ns_best};
            update_best_solution(done.load_id, p, done.tree,
                                 done.node_count, winner_res); }
          environment::remove(done.tree);
          done.tree = nullptr;
        }
        done.valid = false;
      }

      // Ask for this batch's picture and leave it to travel.
      if (winner_idx >= 0 && !batch_stale) {
        deferred_winner &next = m_deferred[wait_slot];
        { scope_ns fetch_time{ns_fetch};
          cuda_dsl_fetch_output_grids_async(winner_idx, active, wait_slot); }
        next.tree = environment::copy(prev_original_nodes[wait_slot][winner_idx]);
        next.node_count = prev_batch.node_counts[winner_idx];
        next.active = min(active, 16);
        next.load_id = m_load_id;
        for (int i = 0; i < next.active; ++i) {
          int j = winner_idx * active + i;
          if (j < 0 || j >= prev_batch.job_count) { next.active = i; break; }
          next.score[i] = prev_batch.scores[j];
          next.solved[i] = (prev_batch.solved[j] != 0);
          next.terms[i] = prev_batch.score_terms[j];
        }
        next.valid = (next.tree != nullptr);
      }
      { scope_ns free_time{ns_free};
        for (node *n : prev_original_nodes[wait_slot]) environment::remove(n);
        prev_original_nodes[wait_slot].clear(); }
    }

    // Record this batch as the one to process next iteration (its launch
    // already happened above, before the previous batch's CPU processing).
    prev_original_nodes[launch_slot] = original_nodes;
    slot_pending[launch_slot] = true;
    current_slot = wait_slot;

    // Null out borrowed pointers so chunk destructor doesn't free beam nodes
    for (auto &t : chunk.trees) t = nullptr;

}

  // The CURRENT column: what the candidate just scored painted on each
  // example, or nothing where it painted nothing. Only a watcher wants it.
  inline void beam_solver::update_example_results(bool is_test,
                                     const vector<beam_solver::candidate_result> &results,
                                     int limit) {
    if (!m_observer)
      return;
    int n = (int)(is_test ? m_puzzle.test : m_puzzle.train).size();
    if (limit >= 0) n = min(n, limit);
    for (int i = 0; i < n; ++i) {
      if (i < (int)results.size() && results[i].valid) {
        show_current_score(is_test, i, results[i].score);
        show(is_test, i, solver_observer::column::current, &results[i].state);
      } else {
        clear_current(is_test, i);
      }
    }
  }

  // When a candidate solves all train, run it on test inputs and show its
  // predictions in the CURRENT column. Otherwise clear, to avoid noise from
  // wrong candidates churning. We never read or display the ground-truth
  // test output here — only the tree's prediction on the test input. All of
  // it is for a watcher, so none of it runs without one.
  inline void beam_solver::update_test_current_views(puzzle &p, node *tree,
                                                     int node_count,
                                                     bool all_train_solved) {
    if (!m_observer) return;
    if (!all_train_solved || !tree) {
      for (size_t i = 0; i < p.test.size(); ++i)
        clear_current(true, i);
      return;
    }
    for (size_t ei = 0; ei < p.test.size(); ++ei) {
      example &ex = p.test[ei];
      grid tg = {};
      if (ex.output.width > 0) {
        auto [ts, _slv] = environment::score_tree_output(
            tree, ex, tg, p, node_count);
        show_current_score(true, ei, ts);
      } else {
        float _s = 0; bool _slv = false;
        bool ok = environment::execute_tree_via_flat(
            tree, ex, tg, p, node_count, _s, _slv);
        if (!ok) { clear_current(true, ei); continue; }
        // No ground truth — match BEST's convention: leave score untouched
        // (will mirror best_score, set when the tree was first promoted).
      }
      show(true, ei, solver_observer::column::current, &tg);
    }
  }

  inline node *beam_solver::reduce_best_tree(node *tree, puzzle &p, int active, int &out_node_count, bool deep) {
    // Flatten to a linear op list. Non-chain trees (shouldn't occur on the
    // unary path) are left untouched.
    flat_program flat;
    if (!environment::flatten_unary_program(tree, flat) || flat.count <= 1) {
      out_node_count = environment::count_nodes(tree);
      return tree;
    }

    // Score a flat program over the active train examples. Returns the summed
    // score (node-count penalty included via node_count = #ops) and a bitmask of
    // solved examples. Builds a throwaway tree, scores on the scratch GPU path,
    // frees it.
    auto score_flat = [&](const flat_program &f, uint32_t &solved_mask) -> float {
      node *t = environment::build_chain_from_flat(f);
      if (!t) { solved_mask = 0; return SCORE_MIN; }
      float sum = 0.0f;
      solved_mask = 0;
      grid temp = {};
      for (int i = 0; i < active; ++i) {
        auto [s, solved] = environment::score_tree_output(
            t, p.train[i], temp, p, f.count);
        sum += s;
        if (solved) solved_mask |= (1u << i);
      }
      environment::remove(t);
      return sum;
    };

    uint32_t base_mask = 0;
    float base_sum = score_flat(flat, base_mask);

    // A sentence is the unit, so a sentence is what may be dropped. Taking
    // single ops out writes a program the generator could never have written,
    // and this search could never find again -- a solution in a language the
    // solver does not speak. Four tries on a four-sentence program, against
    // C(n,k) subsets, and it asks the better question: is this sentence
    // pulling its weight?
    if (m_options.sentence_search && flat.count >= 8 && flat.count % 4 == 0) {
      bool dropped_any = false;
      for (bool again = true; again;) {
        again = false;
        int sentences = flat.count / 4;
        for (int drop = 0; drop < sentences; ++drop) {
          flat_program without;
          for (int i = 0; i < flat.count; ++i)
            if (i / 4 != drop) without.push(flat.ops[i].type, flat.ops[i].value);
          uint32_t mask = 0;
          float sum = score_flat(without, mask);
          // Keep every example that was solved, and be strictly better: a
          // sentence that changes nothing sheds its node penalty and so wins.
          if ((mask & base_mask) == base_mask && sum > base_sum + 0.0001f) {
            flat = without;
            base_sum = sum;
            base_mask = mask;
            dropped_any = true;
            again = true;
            break;
          }
        }
      }
      if (!dropped_any) {
        out_node_count = flat.count;
        return tree;
      }
      node *by_sentence = environment::build_chain_from_flat(flat);
      out_node_count = flat.count;
      return by_sentence ? by_sentence : tree;
    }

    // Greedy leave-k-out: drop the k ops (never the leaf at index 0) whose
    // removal most improves the summed score, as long as it strictly improves
    // AND keeps every solved example solved. A true no-op wins automatically:
    // grids are unchanged so the per-example terms hold while node_count
    // drops -> lower penalty -> higher sum. k=1 everywhere (hot path); deep
    // mode goes to full depth — every k — to catch removals that only cancel
    // jointly (e.g. FlipHorizontal Fractal FlipHorizontal), restarting from
    // k=1 after any hit. Deep runs only at lock commits: junk frozen into a
    // prefix steals depth budget from every later layer, and the found
    // prefix is then the shortest train-equivalent version of itself.
    // Try the best k-subset drop once; apply it and report whether it improved.
    auto try_dropping = [&](int k) -> bool {
      // Dropping nothing is not a sweep, and idx[k - 1] below would step off the
      // front of the array asking about it.
      if (k < 1) return false;
      int idx[flat_program::MAX_OPS];
      int best_drop[flat_program::MAX_OPS];
      int best_found = 0;
      float best_sum = base_sum;
      uint32_t best_mask = base_mask;
      // Standard k-combination walk over droppable positions 1..count-1.
      for (int i = 0; i < k; ++i) idx[i] = i + 1;
      while (idx[k - 1] < flat.count) {
        flat_program reduced;
        for (int i = 0, j = 0; i < flat.count; ++i) {
          if (j < k && i == idx[j]) { ++j; continue; }
          reduced.push(flat.ops[i].type, flat.ops[i].value);
        }
        uint32_t m = 0;
        float s = score_flat(reduced, m);
        if ((m & base_mask) == base_mask && s > best_sum + 0.0001f) {
          best_sum = s; best_mask = m; best_found = k;
          for (int i = 0; i < k; ++i) best_drop[i] = idx[i];
        }
        // Advance: bump the rightmost index that still has room, reset the rest.
        int pos = k - 1;
        while (pos > 0 && idx[pos] == flat.count - k + pos) --pos;
        ++idx[pos];
        for (int i = pos + 1; i < k; ++i) idx[i] = idx[i - 1] + 1;
        if (pos == 0 && idx[0] > flat.count - k) break;
      }
      if (!best_found) return false;
      flat_program next;
      for (int i = 0, j = 0; i < flat.count; ++i) {
        if (j < best_found && i == best_drop[j]) { ++j; continue; }
        next.push(flat.ops[i].type, flat.ops[i].value);
      }
      flat = next;
      base_sum = best_sum;
      base_mask = best_mask;
      return true;
    };
    bool changed = false;
    for (int k = 1; k <= (deep ? flat.count - 1 : 1);) {
      // Cost guard: C(n,k) evals per sweep — full depth is fine on the short
      // spines locks have; cap the subset size once combinations pass ~20k.
      if (deep && k > 1) {
        double combos = 1;
        for (int i = 0; i < k; ++i) combos = combos * (flat.count - 1 - i) / (i + 1);
        if (combos > 20000) break;
      }
      if (try_dropping(k)) { changed = true; k = 1; }
      else ++k;
    }

    if (!changed) {
      out_node_count = flat.count;
      return tree;
    }
    node *reduced_tree = environment::build_chain_from_flat(flat);
    out_node_count = flat.count;
    return reduced_tree ? reduced_tree : tree;
  }

  inline std::pair<float, bool> beam_solver::eval_best_tree_train(puzzle &p, int idx) {
    if (p.best_tree_eval.size() != p.train.size()) p.best_tree_eval.assign(p.train.size(), {});
    auto &slot = p.best_tree_eval[idx];
    if (slot.valid) return {slot.score, slot.solved};
    grid temp = {};
    auto [s, solved] = environment::score_tree_output(
        p.best_tree, p.train[idx], temp, p, p.best_node_count);
    slot.score = s; slot.solved = solved; slot.valid = true;
    return {s, solved};
  }

  // A WHOLE PROGRAM, MADE THE BEST AND SCORED on every example. The puzzle is
  // solved if it solves the train set; its test verdict is read the same way.
  // Caller holds the lock.
  inline void beam_solver::score_program(const string &program) {
    puzzle &task = m_puzzle;
    // Re-import (or import onto a puzzle that loaded with a solved tree) would
    // otherwise leak the whole prior node subtree.
    if (task.best_tree) environment::remove(task.best_tree);
    // A failed import must not read as solved, so the old verdict goes with the old tree.
    task.is_solved = false;
    task.is_test_solved = false;
    task.best_tree = environment::parse(program);
    if (!task.best_tree) {
        log_debug("could not read this as a program: " + program);
        return;
    }
    // A program whose sentences do not each open with a source runs and draws
    // and is wrong; say so rather than let it score.
    if (const char* wrong = sentence_with_no_source(program))
        log_debug(string(wrong) + ": " + program);
    task.best_node_count = environment::count_nodes(task.best_tree);
    // Longer than one batch slot never reaches the GPU: every example would come
    // back empty, which reads as a program that draws nothing rather than one
    // that was never run. Say which it is, and keep the old verdict cleared.
    if (too_long_to_run(task.best_node_count)) {
        log_debug(to_text("%d ops is past the %d a program may be, so this one was "
                          "not run: %s", task.best_node_count,
                          flat_program::MAX_OPS, program.c_str()));
        return;
    }

    string tree_str = environment::stringify_tree_via_flat(task.best_tree);
    // Straight from the kernel that scored it, so the printed sum IS the score.
    auto log_score = [&](int ei, const gpu_score_terms &t, const char *phase) {
      log_debug(t.say(tree_str, ei, phase));
    };

    grid temp;
    bool all_solved = true;
    for (int i = 0; i < (int)task.train.size(); ++i) {
      auto &ex = task.train[i];
      gpu_score_terms terms;
      auto [s, solved] = environment::score_tree_output(
          task.best_tree, ex, temp, task, task.best_node_count, &terms);

      log_debug(to_text("Example score: %s (Solved: %s)", shortest(s).c_str(), solved ? "YES" : "NO"));
      log_score(i, terms, "import_train");

      task.best_score(false, i) = s;
      show(false, i, solver_observer::column::best, &temp);
      if (!solved)
        all_solved = false;
    }

    if (all_solved) {
      task.is_solved = true;
      bool test_all_solved = true;
      for (int i = 0; i < (int)task.test.size(); ++i) {
        auto &ex = task.test[i];
        gpu_score_terms terms;
        auto [s, solved] = environment::score_tree_output(
            task.best_tree, ex, temp, task, task.best_node_count, &terms);
        log_score(i, terms, "import_test");
        task.best_score(true, i) = s;
        show(true, i, solver_observer::column::best, &temp);
        if (!solved) test_all_solved = false;
      }
      if (!task.test.empty() && test_all_solved) {
        task.is_test_solved = true;
      }
    }
  }

  inline void beam_solver::apply_program(const string &program) {
    if (!m_puzzle_loaded || program.empty()) return;
    // Scoring wants the facts, so it asks for them. Ops that learn a table
    // (ShapeColorMap and kin) have nothing to say about a puzzle nobody has
    // read, and would paint an empty grid for every example.
    read_facts();
    scoped_lock lock(m_progress.mtx);
    score_program(program);
    if (m_puzzle.is_solved)
      active_example_count = (int)m_puzzle.train.size();
    changed();
    log_debug("Manual solution imported.");
  }

  // WHAT THE LOOKING ALREADY SPELLED, tried before the search starts. Each
  // suggestion is a whole program, so this either finishes the puzzle or costs
  // one scratch evaluation per suggestion -- which is nothing beside the beam.
  inline bool beam_solver::try_what_inference_said(puzzle &p) {
    if (p.suggested_programs.empty()) return false;
    for (const string &program : p.suggested_programs) {
      score_program(program);
      if (p.is_solved) {
        log_debug("!!!! SOLVED BY WHAT INFERENCE SAID: " + program + " !!!!");
        return true;
      }
    }
    // Nothing fitted, so the last failed tree must not stay on as the best.
    environment::remove(p.best_tree);
    p.best_tree = nullptr;
    invalidate_best_tree_cache(p);
    p.best_node_count = NODE_COUNT_MAX;
    clear_views(p, false);
    clear_views(p, true);
    return false;
  }

  // Blank every picture of the search on one side, both columns. The scores go
  // with them: an example with nothing shown has nothing scored either.
  inline void beam_solver::clear_views(puzzle &p, bool is_test) {
    for (size_t i = 0; i < (is_test ? p.test : p.train).size(); ++i) {
      clear_current(is_test, i);
      clear_best(p.best_score(is_test, i), is_test, i);
    }
  }

  inline bool beam_solver::check_old_tree_solves(puzzle &p, int active) {
    if (!p.best_tree) return false;
    for (int i = 0; i < active; ++i) {
      auto [s, solved] = eval_best_tree_train(p, i);
      if (!solved) return false;
    }
    return true;
  }

  inline int beam_solver::try_promote_examples(node *tree, puzzle &p, int &active, int node_count) {
    int promotion_count = 0;
    while (active < (int)p.train.size()) {
      grid temp = {};
      auto [s, solved] = environment::score_tree_output(
          tree, p.train[active], temp, p, node_count);

      set_example_scores(p.best_score(false, active), false, active, s);
      show_both(false, active, &temp);
      if (!solved) break;

      active++;
      promotion_count++;
      active_example_count = active;
      log_debug(to_text("==== PROMOTED: now solving %d/%zu examples ====",
                       active_example_count, p.train.size()));
    }

    // Filling the train set is the solve verdict — `tree` just verified on
    // every promoted example. Park it; step() marks the task solved.
    if (promotion_count > 0 && active == (int)p.train.size() && !p.is_solved) {
      if (solve_found_by_promotion) environment::remove(solve_found_by_promotion);
      solve_found_by_promotion = environment::copy(tree);
      solve_found_node_count = node_count;
    }
    return promotion_count;
  }

  // Shared train-finale: any path that widens the active set to the full train
  // with a solving best tree must convert that into the solved state. The
  // finale inside update_best_solution only runs for candidates that pass its
  // improvement gate — a tree that merely ties the stored best never enters,
  // so a solve discovered via promotion alone would otherwise search until
  // timeout (observed on 0692e18c: reduction found the solving tree, the step
  // promoted to 3/3, and nothing ever fired the finale).
  inline void beam_solver::finalize_full_train_solve(puzzle &p) {
    if (p.is_solved || !p.best_tree) return;
    if (active_example_count < (int)p.train.size()) return;
    bool test_ok = evaluate_test_after_solve(p.best_tree, p, p.best_node_count);
    p.is_solved = true;
    p.is_test_solved = test_ok;
    log_debug("!!!! TASK SOLVED ON TRAINING SET !!!");
    if (test_ok) log_debug("!!!! TEST SET SOLVED CORRECTLY !!!!");
    search_news n;
    n.from = search_news::source::promotion;
    n.train_solved = true;
    n.test_solved = test_ok;
    news(n);
  }

  inline bool beam_solver::evaluate_test_after_solve(node *tree, puzzle &p, int node_count) {
    bool test_all_ok = !p.test.empty();
    for (int i = 0; i < (int)p.test.size(); ++i) {
      // Always EXECUTE the tree on the test input to produce the predicted grid
      // (score_tree_output fills `tg` whether or not ground truth exists). The
      // earlier code skipped execution when there was no ground-truth output,
      // leaving the view buffer empty — that's why the GUI showed no test
      // prediction on the competition set (test outputs are blank there).
      grid tg = {};
      auto [ts, tslv] = environment::score_tree_output(
          tree, p.test[i], tg, p, node_count);
      // Set the score in both cases: the BEST/CURRENT columns are only drawn
      // when they carry a score (draw_cell suppresses scored columns with no
      // score), so without this the prediction grid would render blank. With no
      // ground truth the score is against an empty target and isn't meaningful,
      // but it still gates display — matching score_program's import path.
      set_example_scores(p.best_score(true, i), true, i, ts);
      if (p.test[i].output.width > 0) {
        // Ground truth available (eval/training sets): tslv tells us whether the
        // hidden test output was actually solved.
        if (!tslv) test_all_ok = false;
      } else {
        // No ground truth (competition test set): can't confirm a solve.
        test_all_ok = false;
      }
      show_both(true, i, &tg);
    }
    return test_all_ok;
  }

  inline void beam_solver::refresh_best_display_scores(puzzle &p, int active, int node_count) {
    bool all_train_perfect = true;
    for (int i = 0; i < (int)p.train.size(); ++i) {
      if (i >= active) {
        clear_best(p.best_score(false, i), false, i);
        continue;
      }
      grid temp = {};
      auto [s, solved] = environment::score_tree_output(
          p.best_tree, p.train[i], temp, p, node_count);
      p.best_score(false, i) = s;
      show(false, i, solver_observer::column::best, &temp);
      if (!solved) all_train_perfect = false;
    }

    if (all_train_perfect && active == (int)p.train.size()) {
      for (int i = 0; i < (int)p.test.size(); ++i) {
        // Always execute the tree to fill the predicted grid — score_tree_output
        // produces `tg` whether or not a ground-truth output exists. Gating this
        // on output.width > 0 (as before) left BEST empty on the competition
        // test set and clobbered the prediction evaluate_test_after_solve set.
        grid tg = {};
        auto [score, slv] = environment::score_tree_output(
            p.best_tree, p.test[i], tg, p, node_count);
        p.best_score(true, i) = score;
        show(true, i, solver_observer::column::best, &tg);
      }
    } else {
      for (size_t i = 0; i < p.test.size(); ++i) {
        clear_best(p.best_score(true, i), true, i);
      }
    }
  }

  inline void beam_solver::try_update_current_display(int load_id, puzzle &p, node *tree, int node_count, const candidate_batch &res) {
    unique_lock<mutex> lock(m_progress.mtx, try_to_lock);
    if (lock.owns_lock()) {
      if (load_id == m_load_id) {
        update_example_results(false, res.train);
        int active = min(active_example_count, (int)p.train.size());
        bool full_train = res.all_train_solved && active == (int)p.train.size();
        update_test_current_views(p, tree, node_count, full_train);

        environment::remove(m_progress.current_tree);
        m_progress.current_tree = environment::copy(tree);
        changed();
      }
    }
  }

}
