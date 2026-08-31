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

  inline bool beam_solver::check_candidate_compatibility(node_type type, node* arg) {
    return arg != nullptr;
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

  static inline void update_grid_if_visible(grid& target, const grid& source,
                                            vector<int>* gpu_buffer) {
    if (gpu_buffer)
      update_display_buffer(target, source, *gpu_buffer);
  }

  static inline void update_example_views(example& ex, const grid& source,
                                          vector<int>* gpu_buffer) {
    if (!gpu_buffer)
      return;
    update_display_buffer(ex.best, source, *gpu_buffer);
    update_display_buffer(ex.current, source, *gpu_buffer);
  }

  static inline void clear_current_view(example& ex) {
    ex.cur_score = SCORE_MIN - 1.0f;
    ex.current.width = 0;
  }

  static inline void clear_best_view(example& ex) {
    ex.best_score = SCORE_MIN - 1.0f;
    ex.best.width = 0;
  }

  static inline void set_example_scores(example& ex, float score) {
    ex.best_score = score;
    ex.cur_score = score;
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
      log_debug(fmt::format("Level stagnation: {} levels at {:.2f}. Diversifying beam.",
                       stagnant_levels, (double)prev_beam_top));
      // Flat search has stalled — decomposition earns its turn.
      if (!locking_armed && !locking_suspended) {
        locking_armed = true;
        log_debug("==== LOCKING ARMED: flat search stalled ====");
      }
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
        // Layer lock: lock-less injectees only feed admission rejections.
        if (locked_prefix && !contains_locked(h)) continue;
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
    visited_grids.clear();

    prepare_next_level_slots();
    update_history_from_beam();
    m_consec_dry_beam_entries = 0;

    long long fetch_copy_ns = 0, fetch_sync_ns = 0;
    cuda_dsl_take_fetch_ns(&fetch_copy_ns, &fetch_sync_ns);
    long long wait_total = m_solver_state.gpu_wait_ns.load();
    long long ns_wait = wait_total - ns_wait_at_last_level;
    ns_wait_at_last_level = wait_total;
    log_timing(fmt::format(
      "gen={}ms launch={}ms wait={}ms idle={}ms | harvest={}ms = tally={}ms winner={}ms free={}ms rest={}ms"
      " | winner = fetch={}ms analyze={}ms best={}ms rest={}ms"
      " | fetch = copy={}ms sync={}ms",
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

  // One batch of whole sentences. Level 1 has an empty beam, so a sentence IS
  // the program; later levels hang one onto each beam member. Returns true when
  // the level has nothing left to give.
  inline bool beam_solver::generate_sentence_batch() {
    if (!sweep_ready) {
      enumerate_sentences(sweep);
      sweep_ready = true;
      sweep_beam_idx = 0;
      sweep_idx = 0;
      log_debug(fmt::format("[SENTENCES] level sweep = {} (beam {})",
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
        if (!base) { sweep_idx = 0; sweep_beam_idx++; continue; }
        if (flattened_for != sweep_beam_idx) {
          program.clear();
          if (!environment::flatten_unary_program(base, program)) {
            sweep_idx = 0;
            sweep_beam_idx++;
            continue;
          }
          // A sentence is four ops, so the depth cap bounds how many fit.
          if (program.count + 4 > MAX_SEARCH_DEPTH) {
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
      if (!write_sentence(sweep[sweep_idx++], program)) {
        program.count = base_count;
        continue;
      }

      node *candidate = environment::build_chain_from_flat(program);
      program.count = base_count;
      if (!candidate) continue;
      if (!visited.insert_if_new(candidate->hash)) {
        environment::remove(candidate);
        continue;
      }
      // Same ownership contract as add_candidate_if_new: the slot list owns the
      // ref, pending_evaluation only borrows it.
      current_slot_candidates.push_back({candidate, 0.0f, 0});
      pending_evaluation.push(candidate);
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
        level_candidates.push_back({environment::copy(cand.n), cand.score, cand.grid_hash});
    commit_level_champion();
    visited.clear();
    visited_grids.clear();
    sweep_ready = false;
    sweep_beam_idx = 0;
    sweep_idx = 0;
  }

  inline bool beam_solver::pump_generate(const puzzle &p,
             atomic<long long> *global_attempts) {
    m_global_attempts = global_attempts;
    // Sentence mode never seeds starters: it has no history pool and no slots,
    // because a sentence is built whole rather than grown an op at a time.
    if (!initialized && !m_app_state.sentence_search) {
      seed_starter_nodes();
      initialized = true;
      log_debug("Top-Down Solver (Zero Memory) Initialized.");
    }

    if (!pending_evaluation.empty())
      return true;

    is_generating_layer = true;

    if (m_app_state.sentence_search) {
      // Commit only once the batch just made has been drained: level_candidates
      // is released on commit, and pending_evaluation only BORROWS those nodes.
      // Committing with any still queued hands the worker freed memory.
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

      log_debug(fmt::format("[GEN] slot={}/{} done cand={} lc={} vis={}",
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
    for (int depth = spine_len - 1; depth >= 0; --depth) {
      const node *n = spine[depth];
      // Mirror environment::update_hash exactly; the swapped child is treated
      // as non-null.
      uint64_t h = 0x9e3779b97f4a7c15ULL + to_underlying(n->type);
      h ^= (uint64_t)n->value + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= child_hash + 0x9e3779b9 + (h << 6) + (h >> 2);
      child_hash = h;
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
      int variants = is_annotated(t) ? get_variant_count(t) : 1;
      for (int v = 0; v < variants; ++v) {
        int16_t value = is_annotated(t) ? get_variant_value(t, v) : (int16_t)0;
        if (!color_literal_allowed(t, value)) continue;
        out.push_back({t, value});
      }
    }
  }

  inline void beam_solver::enumerate_sentences(vector<sentence> &out) const {
    out.clear();
    vector<flat_program_op> gets, selects, acts, combines;
    beat_choices(op_role::get, gets);
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
      int variants = is_annotated(t) ? get_variant_count(t) : 1;
      for (int v = 0; v < variants; ++v) {
        // The same pin the other two generators apply: a starter may not name a
        // color the side it speaks about never shows.
        if (!color_literal_allowed(t, (int16_t)v)) continue;
        node *n = new node();
        n->type = t;
        n->value = (int16_t)v;
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

    log_debug(fmt::format("Updated History Pool: {} items (+{})",
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

    // Layer lock: bare starters can't contain the lock — every candidate they
    // spawn would be admission-rejected (the false-Idle drain). Reseed from
    // the lock itself instead.
    if (locked_prefix) {
      beam.push_back({environment::copy(locked_prefix), 10.0f, 0});
      return;
    }

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

    environment::remove(inp);
  }

  inline void beam_solver::prepare_next_level_slots() {
    // Color-layer commit sweep: a layer that FINISHES a color deserves to lock
    // even when its beam score never beat the global best (the scorer is a
    // misleading doorman). Scan the level's survivors best-first, root-Union
    // only, capped — first commit wins and triggers the restart below.
    // Depth saturation: once the beam carries trees at the depth cap, the
    // whole-glance phase is over by construction — flat cannot go deeper,
    // only layering refocuses the budget. (05a7bcf2: flat crawled 16 min
    // with cells still improving, so no stall signal could ever fire —
    // progressing is not converging.)
    if (!locking_armed && !locking_suspended) {
      for (auto &cand : beam) {
        if (cand.n && cand.n->max_depth >= MAX_SEARCH_DEPTH) {
          locking_armed = true;
          log_debug("==== LOCKING ARMED: depth saturated ====");
          break;
        }
      }
    }
    if (m_app_state.layer_locking && locking_armed && !locking_suspended && !lock_commit_pending && m_puzzle_loaded) {
      // Solve one color at a time from the very start: before any lock exists,
      // residuals are simply the target color counts — pick a focus from them.
      if (focus_color == -1 && !locked_prefix && !m_focus_initialized) {
        m_focus_initialized = true;
        for (auto &ex : m_puzzle.train)
          for (int r = 0; r < ex.output.height; ++r)
            for (int c = 0; c < ex.output.width; ++c) {
              int v = ex.output.data[r][c];
              if (v >= 0 && v < 10) { locked_color_total[v]++; locked_color_residual[v]++; }
            }
        pick_focus_color();
      }
      const int scan_cap = 64;
      int scanned = 0;
      int active = min(active_example_count, (int)m_puzzle.train.size());
      // Coverage-ranked commit: probe every qualifying survivor, freeze only
      // the one explaining the most cells (ties: more new colors, fewer
      // nodes). The human prior: prefer the layer with the least unexplained
      // remainder, not the first that finishes something.
      node *best_cand = nullptr;
      int best_nc = 0, best_corr = -1, best_new = -1;
      for (auto &cand : beam) {
        if (scanned >= scan_cap) break;
        if (!cand.n || !modifies_canvas(cand.n->type)) continue;
        ++scanned;
        if (probe_failed.count(cand.n->hash)) continue;
        int nc = environment::count_nodes(cand.n);
        int corr = 0, nnew = 0;
        if (!try_lock_layer(cand.n, nc, m_puzzle, active, &corr, &nnew, false)) {
          probe_failed.insert(cand.n->hash);
          continue;
        }
        if (corr > best_corr ||
            (corr == best_corr && (nnew > best_new || (nnew == best_new && nc < best_nc)))) {
          best_cand = cand.n; best_nc = nc; best_corr = corr; best_new = nnew;
        }
      }
      if (best_cand)
        try_lock_layer(best_cand, best_nc, m_puzzle, active);
      // Focus rotation: a focus color that stalls for FOCUS_ROTATE_LEVELS
      // boundaries likely depends on a color not yet painted — mark it tried
      // and move on. A full circle of tried colors drops the focus entirely.
      if (focus_color >= 0 && !lock_commit_pending) {
        if (focus_progress_this_level) {
          focus_levels_no_progress = 0;
        } else if (++focus_levels_no_progress >= FOCUS_ROTATE_LEVELS) {
          focus_tried[focus_color] = true;
          focus_levels_no_progress = 0;
          // Rotation may only hand focus to another UNFINISHED color — never
          // to nothing. A stalled color keeps its focus when it's all there is.
          int prev = focus_color;
          pick_focus_color(prev);
          if (focus_color == -1) {
            // Every unfinished color has had its turn — one full circle.
            if (++focus_circles_no_lock >= 2 && locked_prefix) {
              drop_ladder();
            } else {
              for (int v = 0; v < 10; ++v) focus_tried[v] = false;
              focus_color = prev;
              focus_best_correct = locked_color_total[prev] - locked_color_residual[prev];
              cuda_dsl_set_focus_color(prev);
              log_debug(fmt::format("==== FOCUS KEPT: color {} (full circle {}) ====", prev, focus_circles_no_lock));
            }
          } else {
            log_debug(fmt::format("==== FOCUS ROTATED: color {} stalled ====", prev));
          }
        }
        focus_progress_this_level = false;
      }
    }
    if (lock_commit_pending) {
      // Restart from the committed layer: the lock is the only beam member and
      // continuations grow from it. History pool is kept — fragments stay useful.
      release_candidates(beam);
      beam.push_back({environment::copy(locked_prefix), 10.0f, 0});
      lock_commit_pending = false;
      log_debug("==== BEAM RESTARTED FROM LOCKED PREFIX ====");
    }
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
    log_debug(fmt::format("New level: {} slots to refine.", slots_to_refine.size()));
  }

  inline void beam_solver::find_slots_recursive(node *n, int depth, vector<beam_solver::slot_t> &out,
                            set<int> &visited) {
    if (!n)
      return;
    // Layer lock: the committed prefix is opaque — never recurse inside it.
    // (Slots inside it would only generate candidates that admission rejects,
    // draining the generator dry -> false Idle.)
    if (locked_prefix && n->hash == locked_hash)
      return;
    if (depth >= MAX_SEARCH_DEPTH) return;
    node *child = n->arg;
    if (!child) return;
    int child_depth = depth + 1;
    // Layer lock: the committed prefix is atomic — no slot at or inside it.
    if (locked_prefix && child->hash == locked_hash) return;
    if (visited.find(child_depth) == visited.end()) {
      out.push_back({child_depth});
      visited.insert(child_depth);
    }
    find_slots_recursive(child, child_depth, out, visited);
  }

  inline void beam_solver::reset_d2_generator() {
    m_gen_phase = 0;
    m_gen_primitive_idx = 0;
    m_gen_outer_idx = 0;
    m_gen_inner_idx = 0;

    total_candidates_this_slot = 0;
  }

  inline beam_solver::phase_result beam_solver::generate_phase_prune(const beam_solver::slot_t &slot, node* target, int &batch_count, node *&candidate_sub) {
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

  inline beam_solver::phase_result beam_solver::generate_phase_0(const beam_solver::slot_t &slot, int &batch_count, node *&candidate_sub, bool &created_new) {
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

    // Locked mode, root slot: the history bucket is almost entirely lock-less
    // fragments that admission rejects (drains the generator dry -> false
    // Idle). Grow the chain instead: wrap the beam member ITSELF with each
    // primitive — one outer item, all variants.
    if (locked_prefix && slot.depth == 0) {
      if (m_gen_outer_idx > 0 ||
          !m_current_base || m_current_base->max_depth + 1 > MAX_SEARCH_DEPTH) {
        m_gen_primitive_idx++;
        m_gen_outer_idx = 0;
        m_gen_inner_idx = 0;
        return phase_result::break_loop;
      }
      node *t0 = m_current_base;
      if (!legal_next(t0, type)) {
        m_gen_primitive_idx++;
        m_gen_outer_idx = 0;
        m_gen_inner_idx = 0;
        return phase_result::break_loop;
      }
      candidate_sub = create_unary_node(type, t0);
      if (is_parameterized(type)) {
        int variants = get_variant_count(type);
        while (m_gen_inner_idx < variants &&
               !color_literal_allowed(type, ranked_variant_value(type, m_gen_inner_idx)))
          m_gen_inner_idx++;
        if (m_gen_inner_idx >= variants) {
          environment::remove(candidate_sub);
          candidate_sub = nullptr;
          m_gen_outer_idx++;
          m_gen_inner_idx = 0;
          return phase_result::retry;
        }
        candidate_sub->value = ranked_variant_value(type, m_gen_inner_idx);
        environment::update_hash(candidate_sub);
        m_gen_inner_idx++;
        if (m_gen_inner_idx >= variants) { m_gen_outer_idx++; m_gen_inner_idx = 0; }
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

    // Bound iteration by the depth headroom for this slot; depth-overflow
    // entries never enter the loop.
    const auto &bucket = history_by_depth;
    int max_depth_allowed = MAX_SEARCH_DEPTH - slot.depth - 1;
    if (max_depth_allowed < 0) max_depth_allowed = 0;
    if (max_depth_allowed > MAX_SEARCH_DEPTH) max_depth_allowed = MAX_SEARCH_DEPTH;
    int bucket_end = bucket_cutoff[max_depth_allowed];

    if (m_gen_outer_idx >= bucket_end) {
      log_debug(fmt::format("[PRIM] prim={} done: bucket_end={} beam={} cand={} vis={} dry_streak={}",
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
      case 0: return generate_phase_prune(slot, target, batch_count, candidate_sub);
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
          log_debug(fmt::format("[GEN] dry yield exit: dry_yields={} beam={} cand={} vis={}",
            m_dry_yields, beam_idx, total_candidates_this_slot, visited.size()));
          return true;
        }
        return false;
      }

      // Full beam entry just completed — check if it contributed anything
      if (total_candidates_this_slot == cand_before) {
        m_consec_dry_beam_entries++;
        log_debug(fmt::format("[GEN] dry beam entry={} consec={} cand={} vis={}",
          beam_idx, m_consec_dry_beam_entries, total_candidates_this_slot, visited.size()));
        if (m_consec_dry_beam_entries >= 2) {
          log_debug(fmt::format("[GEN] early slot exit after {} dry entries at beam={} cand={} vis={}",
            m_consec_dry_beam_entries, beam_idx, total_candidates_this_slot, visited.size()));
          return true;
        }
      } else {
        m_consec_dry_beam_entries = 0;
      }

      advance_beam_idx();
    }
    log_debug(fmt::format("[GEN] beam exhausted: beam={} cand={} vis={}",
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

    // Layer lock: every candidate must build on the committed prefix. Checked
    // BEFORE materializing — on a unary spine the lock survives the replacement
    // unless the replaced subtree held it and the replacement doesn't.
    if (locked_prefix && !contains_locked(candidate_sub)) {
      node *replaced = get_node_at_path(base, depth);
      if (!replaced || contains_locked(replaced)) {
        if (created_new) environment::remove(candidate_sub);
        return false;
      }
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
        cand.grid_hash = it->second.grid_hash;
      }
    }

    // Grid-output dedup: two ASTs that produce the same output grid on the
    // first training pair are evidence-equivalent for the beam. Drop the
    // duplicate so its children don't get expanded next level. grid_hash==0
    // means no valid output was produced yet — keep those, the existing
    // score-based pruning handles them.
    //
    // Measured impact: marginal during real search (<1% of candidates are
    // grid-duplicates pre-first-solve — AST-hash dedup already catches almost
    // everything). Massive during post-solve simplification (~99% duplicates)
    // but that phase isn't on the critical path. Kept because it's correct
    // and cheap (one hash probe per candidate), not because it's a big win.
    // The real bottleneck is the L2 cand=1M cap, not grid redundancy.
    int dropped = 0;
    for (auto &cand : current_slot_candidates) {
      if (!cand.n) continue;
      if (cand.grid_hash == 0) {
        level_candidates.push_back(cand);
        continue;
      }
      if (visited_grids.insert_if_new(cand.grid_hash)) {
        level_candidates.push_back(cand);
      } else {
        environment::remove(cand.n);
        cand.n = nullptr;
        dropped++;
      }
    }
    if (dropped > 0) {
      log_debug(fmt::format("[GRID-DEDUP] dropped={} kept={} grids_seen={}",
        dropped, (int)level_candidates.size(), visited_grids.size()));
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
        best_by_type[k] = {environment::copy(cand.n), cand.score, cand.grid_hash};
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
      // Focus keep-alive: the focus color's staging chains score in a valley
      // (worse before the deposit), so pure rank-eviction kills them. Reserve
      // seats for the best chains that FETCH the focus color — bias, not
      // seeding: the search still writes every op itself.
      if (focus_color >= 0) {
        constexpr int FOCUS_SEATS = 64;
        auto fetches_focus = [&](node *root) {
          for (node *n = root; n; n = n->arg)
            if (n->type == node_type::SelectColor && n->value == focus_color)
              return true;
          return false;
        };
        int have = 0;
        for (int i = 0; i < (int)BEAM_WIDTH; ++i)
          if (fetches_focus(next_beam[i].n)) ++have;
        int evict_pos = (int)BEAM_WIDTH - 1;
        for (size_t i = BEAM_WIDTH; i < next_beam.size() && have < FOCUS_SEATS; ++i) {
          if (!fetches_focus(next_beam[i].n)) continue;
          while (evict_pos >= 0 && fetches_focus(next_beam[evict_pos].n)) --evict_pos;
          if (evict_pos < 0) break;
          swap(next_beam[evict_pos], next_beam[i]);
          ++have;
          --evict_pos;
        }
      }
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
        fmt::format("Level committed. Beam top: {:.2f}  All-time best: {:.2f}",
               (double)top_score, (double)all_time_best_score));
  }

  static inline void build_example_upload(
      cuda_dsl_example_data& ex,
      const puzzle& p,
      int active_example_count) {
    static thread_local auto init_ctx = std::make_unique<fixed_exec_context>();
    int active = min(active_example_count, (int)p.train.size());
    ex.count = active;

    for (int ei = 0; ei < active; ++ei) {
      fixed_init_context(*init_ctx, p.train[ei], p.rule, p.invariant_colors);

      int8_t* ig = ex.input_grids + ei * GPU_GRID_CELLS;
      memcpy(ig, init_ctx->input.data, GPU_GRID_CELLS);
      ex.input_dims[ei * 2 + 0] = (int8_t)init_ctx->input.width;
      ex.input_dims[ei * 2 + 1] = (int8_t)init_ctx->input.height;

      int8_t* tg = ex.target_grids + ei * GPU_GRID_CELLS;
      const auto& out = p.train[ei].output;
      memset(tg, GPU_TRANSPARENT, GPU_GRID_CELLS);
      for (int r = 0; r < out.height; ++r)
        for (int c = 0; c < out.width; ++c)
          tg[r * GPU_MAX_GRID + c] = (int8_t)out.data[r][c];
      ex.target_dims[ei * 2 + 0] = (int8_t)out.width;
      ex.target_dims[ei * 2 + 1] = (int8_t)out.height;

      int8_t* dm = ex.distance_maps + ei * GPU_MAX_COLORS * GPU_GRID_CELLS;
      memcpy(dm, init_ctx->distance_maps, GPU_MAX_COLORS * GPU_GRID_CELLS);

      ex.seed_lists[ei] = to_gpu_object_list(init_ctx->seed_objs);

      // Copy precomputed target CC summaries (detect_target_objects ran at load).
      gpu_object_summary* dst = ex.target_objs + ei * GPU_MAX_TARGET_OBJECTS;
      int tc = (ei < (int)p.target_objects.size()) ? p.target_objects[ei].count : 0;
      if (tc > GPU_MAX_TARGET_OBJECTS) tc = GPU_MAX_TARGET_OBJECTS;
      for (int k = 0; k < tc; ++k) {
        const auto& s = p.target_objects[ei].objs[k];
        dst[k].color = s.color;
        dst[k].w = s.w;
        dst[k].h = s.h;
        dst[k].x0 = s.x0;
        dst[k].y0 = s.y0;
        dst[k].count = s.count;
      }
      ex.target_obj_counts[ei] = (int8_t)tc;
    }

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

  inline void beam_solver::log_score_formula(const node *tree, const vector<score_breakdown> &breakdowns) {
    if (breakdowns.empty())
      return;

    string tree_str = environment::stringify_tree_via_flat(tree);
    for (int i = 0; i < (int)breakdowns.size(); ++i) {
      const auto &b = breakdowns[i];
      log_debug(fmt::format(
          "[SCORE] tree=\"{}\" ex={} mismatches={} exact_term={:.6f} histogram_error={:.6f} histogram_term={:.6f} "
          "distance_error={:.6f} distance_term={:.6f} rarity_score={:.6f} rarity_term={:.6f} "
          "shape_matches={} shape_term={:.6f} color_matches={} color_term={:.6f} "
          "position_score={:.4f} position_term={:.6f} "
          "node_penalty={:.6f} dim_penalty={:.6f} "
          "dims_match={} solved={} "
          "formula=({:.6f})+({:.6f})+({:.6f})+({:.6f})+({:.6f})+({:.6f})+({:.6f})-({:.6f})-({:.6f})={:.6f} "
          "total={:.6f}",
          tree_str,
          i,
          b.mismatches, b.exact_term,
          b.histogram_error, b.histogram_term,
          b.distance_error, b.distance_term,
          b.rarity_score, b.rarity_term,
          b.shape_matches, b.shape_term, b.color_matches, b.color_term,
          b.position_score, b.position_term,
          b.node_penalty,
          b.dimension_penalty,
          b.dims_match ? "true" : "false",
          b.solved ? "true" : "false",
          b.exact_term, b.histogram_term, b.distance_term, b.rarity_term,
          b.shape_term, b.color_term, b.position_term,
          b.node_penalty, b.dimension_penalty, b.total,
          b.total));
    }
  }

  inline void beam_solver::update_best_solution(int target_idx, puzzle &p, node *tree,
                                   int node_count, beam_solver::candidate_batch &res) {
    int active = min(active_example_count, (int)p.train.size());

    bool overrides_worse_tree = res.all_train_solved && !check_old_tree_solves(p, active);

    float current_best_sum = 0;
    for (int i = 0; i < active; ++i)
      current_best_sum += p.train[i].best_score;

    bool score_better = res.total_train_score > current_best_sum + 0.0001f;
    // Only treat "same score" as a valid simplification trigger if the new
    // tree is actually correct on all active examples — tying on a rarity-
    // weighted score does not imply functional equivalence.
    bool score_same = res.all_train_solved &&
                      abs(res.total_train_score - current_best_sum) < 0.0001f;
    bool nodes_better = node_count < p.best_node_count;
    bool should_promote = res.all_train_solved && (active < (int)p.train.size());

    if (overrides_worse_tree || score_better || (score_same && nodes_better) || should_promote) {
      unique_lock lock(m_solver_state.mtx);
      if (target_idx != m_solver_state.puzzle_idx)
        return;

      bool still_overrides = res.all_train_solved && !check_old_tree_solves(p, active);

      current_best_sum = 0;
      for (int i = 0; i < active; ++i)
        current_best_sum += p.train[i].best_score;

      bool still_score_better = res.total_train_score > current_best_sum + 0.0001f;
      bool still_score_same = res.all_train_solved &&
                              abs(res.total_train_score - current_best_sum) < 0.0001f;
      bool still_nodes_better = node_count < p.best_node_count;
      bool still_should_promote = res.all_train_solved && (active < (int)p.train.size());

      if (!still_overrides && !still_score_better && !(still_score_same && still_nodes_better) && !still_should_promote)
        return;

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
          reset_test_display(p);
        }
      } else if (res.all_train_solved && (active == (int)p.train.size()) && !p.is_solved) {
        is_train_finale = true;
        is_test_solve = evaluate_test_after_solve(tree, p, node_count) && !p.is_test_solved;
      }

      if (promotion_count == 0 && !is_train_finale && !is_test_solve) {
        if (!still_overrides && !still_score_better && !(still_score_same && still_nodes_better)) {
          update_example_results(p.train, res.train);
          bool full_train = res.all_train_solved && active == (int)p.train.size();
          update_test_current_views(p, tree, node_count, full_train);
          environment::remove(m_solver_state.current_tree);
          m_solver_state.current_tree = environment::copy(tree);
          m_app_state.needs_upload = true;
          return;
        }
      }

      bool improvement_sound = (promotion_count == 0) && !is_train_finale && !is_test_solve && (still_score_better || (still_score_same && still_nodes_better));

      // res was evaluated against the pre-promotion active count; only the
      // first (active - promotion_count) entries in res.train are valid.
      // try_promote_examples already populated views for the promoted ones,
      // so don't let update_example_results clobber them.
      update_example_results(p.train, res.train, active - promotion_count);
      {
        bool full_train = res.all_train_solved && active == (int)p.train.size();
        update_test_current_views(p, tree, node_count, full_train);
      }
      m_app_state.needs_upload = true;

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
        log_debug(fmt::format("!!!! REDUCED: {} -> {} nodes (leave-one-out) !!!!",
                              node_count, reduced_node_count));

      environment::remove(p.best_tree);
      p.best_tree = environment::copy(reduced);
      invalidate_best_tree_cache(p);

      if (promotion_count == 0 && !is_train_finale)
        p.best_node_count = reduced_node_count;


      refresh_best_display_scores(p, active, reduced_node_count);

      environment::remove(m_solver_state.current_tree);
      m_solver_state.current_tree = environment::copy(reduced);

      // Layer-lock commit: a tree that ends at a canvas op and
      // paints zero wrong cells on kept cells may freeze as the immutable
      // prefix (see try_lock_layer for the full criteria).
      try_lock_layer(reduced, reduced_node_count, p, active);

      // Free the reduced tree if reduce_best_tree allocated a new one (the two
      // copy() calls above took their own refs; this drops our local owner).
      if (was_reduced) environment::remove(reduced);

      bool is_simplification = (promotion_count == 0) && !is_train_finale && !is_test_solve && still_score_same && still_nodes_better && p.is_solved;
      if (is_train_finale || is_simplification) {
        log_debug(is_train_finale ? "!!!! TASK SOLVED ON TRAINING SET !!!" : "!!!! SIMPLIFIED SOLUTION FOUND !!!!");
        if (is_train_finale) {
          p.is_test_solved = is_test_solve;
          p.is_solved = true;
          p.best_node_count = node_count;
          for (auto &ex : p.test) ex.output.visible = true;
        } else {
          log_debug(fmt::format("!!!! SIMPLIFIED SOLUTION FOUND: {} nodes (was {}) !!!!", node_count, p.best_node_count));
        }
        if (!m_app_state.is_benchmarking) toggle_pause();
      } else if (promotion_count == 0 && !is_test_solve) {
        log_debug(fmt::format("!!!! SIMPLIFIED SOLUTION FOUND: {} nodes (was {}) !!!!", node_count, p.best_node_count));
      }

      if (is_test_solve) {
        p.is_test_solved = true;
        log_debug("!!!! TEST SET SOLVED CORRECTLY !!!!");
      }

      bool final_is_simplification = (promotion_count == 0) && !is_train_finale && !is_test_solve && still_score_same && still_nodes_better && p.is_solved;
      lock.unlock();

      if (improvement_sound || promotion_count > 0 || is_train_finale || is_test_solve || final_is_simplification) {
        vector<pair<int, int>> seq;
        seq.push_back({1047, 40});
        for (int i = 0; i < promotion_count; ++i)
          seq.push_back({1319, 80});
        if (final_is_simplification)
          seq.push_back({1319, 60});
        if (is_train_finale)
          seq.push_back({1568, 150});
        if (is_test_solve)
          seq.push_back({2093, 250});
        sound_manager::get().play_sequence(seq);
      }
    } else {
      try_update_current_display(target_idx, p, tree, node_count, res);
    }
  }

  // CPU re-eval of a root-Union tree against the given pairs; commits it as
  // the locked prefix if it qualifies. Two commit paths, both requiring zero
  // wrong paint on KEPT cells (input == output there — wrong paint on a
  // changed cell is fine, a later layer overwrites it; the underlayer case.
  // Without a kept map — non-Identity — falls back to strict zero-wrong):
  //  1. explains more total cells than the current lock, or
  //  2. FINISHES a color the lock hadn't (every target cell of that color
  //     correct, all pairs) — layered solving is one color at a time.
  // Either way it must not un-finish an already-done color.
  // Purity against the SCHEDULE: wrong paint is allowed only on cells whose
  // target color hasn't had its layer yet — that color can only lock by
  // repainting them correctly, so the repair is mandatory. (Fully strict
  // purity was tried 2026-07-03 and vetoes real solutions: the wall layer
  // paints cells the towers cover later.) When the last color locks, no
  // un-laid colors remain — the final tree is fully correct.
  inline bool beam_solver::layer_fit(node *tree, int node_count, puzzle &p, int pairs,
                                     int color_total[10], int color_correct[10], int &correct) {
    correct = 0;
    for (int v = 0; v < 10; ++v) color_total[v] = color_correct[v] = 0;
    for (int i = 0; i < pairs; ++i) {
      grid temp = {};
      environment::score_tree_output(tree, p.train[i], temp, p, node_count);
      const grid &t = p.train[i].output;
      if (temp.width != t.width || temp.height != t.height) return false;
      for (int r = 0; r < t.height; ++r)
        for (int c = 0; c < t.width; ++c) {
          int a = temp.data[r][c];
          int tv = t.data[r][c];
          if (tv >= 0 && tv < 10) {
            color_total[tv]++;
            if (a == tv) color_correct[tv]++;
          }
          if (a == t.data[r][c]) correct++;
          else if (a != -1 && (tv < 0 || tv > 9 || locked_color_done[tv]))
            return false;
        }
    }
    return true;
  }

  inline bool beam_solver::try_lock_layer(node *tree, int node_count, puzzle &p, int active,
                                          int *out_correct, int *out_new_colors, bool commit) {
    if (!m_app_state.layer_locking || !locking_armed || locking_suspended || !tree || !modifies_canvas(tree->type))
      return false;
    if (locked_prefix && tree->hash == locked_hash)
      return false;
    // A lock is permanent — it must hold on EVERY train pair, not just the
    // active set (2026-07-11, 007bbfb7: a layer perfect on the 2 actives by
    // coincidence froze, and no extension could ever solve example 3).
    int full = (int)p.train.size();
    int correct = 0;
    int color_total[10], color_correct[10];
    if (!layer_fit(tree, node_count, p, full, color_total, color_correct, correct))
      return false;
    // Focus progress: did any clean survivor push the focus color forward?
    if (focus_color >= 0 && color_correct[focus_color] > focus_best_correct) {
      focus_best_correct = color_correct[focus_color];
      focus_progress_this_level = true;
      for (int v = 0; v < 10; ++v) focus_tried[v] = false;
    }
    bool color_done[10] = {};
    bool finishes_new_color = false, breaks_done_color = false;
    for (int v = 0; v < 10; ++v) {
      color_done[v] = color_total[v] > 0 && color_correct[v] == color_total[v];
      if (color_done[v] && !locked_color_done[v]) finishes_new_color = true;
      if (!color_done[v] && locked_color_done[v]) breaks_done_color = true;
      // Hollowing note (2026-07-03): losing correct cells of an UNDONE color
      // is legal — that color is provisional until its own layer completes it,
      // same logic that allows wrong paint there. Done colors are already
      // hollow-proof: losing a cell un-finishes them -> breaks_done_color.
    }
    // A layer must FINISH its color before the next begins: completing a new
    // color is the only commit trigger. (Partial "+N cells" locks were tried
    // 2026-07-03 — greedy micro-locks burned the depth budget dry.)
    if (breaks_done_color || !finishes_new_color)
      return false;
    // Probe: qualified — report coverage, freeze nothing.
    if (!commit) {
      if (out_correct) *out_correct = correct;
      if (out_new_colors) {
        int n = 0;
        for (int v = 0; v < 10; ++v)
          if (color_done[v] && !locked_color_done[v]) ++n;
        *out_new_colors = n;
      }
      return true;
    }
    // Lean locks: this tree qualifies — strip no-ops before freezing, since
    // every junk op in a lock is stolen from all later layers' depth budget.
    // If the reduced form no longer qualifies (reduction optimizes beam score,
    // not layer criteria), fall back to committing the raw tree.
    if (!m_reducing_for_lock) {
      int rc = node_count;
      // Full depth, vetted on the full train set — same authority as the lock itself.
      node *red = reduce_best_tree(tree, p, full, rc, true);
      log_debug(fmt::format("[LOCK-REDUCE] {} -> {} nodes | in=\"{}\" out=\"{}\"",
                            node_count, rc, environment::stringify_tree_via_flat(tree),
                            environment::stringify_tree_via_flat(red)));
      if (red != tree) {
        m_reducing_for_lock = true;
        bool committed = try_lock_layer(red, rc, p, active);
        m_reducing_for_lock = false;
        environment::remove(red);
        if (committed) return true;
      }
    }
    if (locked_prefix) environment::remove(locked_prefix);
    locked_prefix = environment::copy(tree);
    locked_hash = tree->hash;
    focus_circles_no_lock = 0;
    probe_failed.clear();
    locked_correct_cells = std::max(locked_correct_cells, correct);
    string done_str;
    for (int v = 0; v < 10; ++v) {
      locked_color_done[v] = color_done[v];
      if (color_done[v]) done_str += std::to_string(v) + " ";
    }
    lock_commit_pending = true;
    log_debug(fmt::format("==== LAYER LOCKED: {} cells explained, colors done: [{}], prefix frozen ====",
                          correct, done_str));
    log_debug(fmt::format("[LOCK-PREFIX] \"{}\"", environment::stringify_tree_via_flat(locked_prefix)));
    // A lock explaining EVERY cell of the active set is a full solution for
    // it — hand it to promotion directly (the best channel may never rank it
    // first, and promotion's re-verify is the overfit gauntlet anyway).
    {
      int total_cells = 0;
      for (int v = 0; v < 10; ++v) total_cells += color_total[v];
      // If this promotion fills the train set, try_promote_examples parks the
      // prefix as the found solve; the worker loop marks the task solved.
      if (correct == total_cells && active < (int)p.train.size())
        try_promote_examples(locked_prefix, p, active_example_count, environment::count_nodes(locked_prefix));
    }
    // A lock is progress by definition: refresh residuals and re-pick focus.
    for (int v = 0; v < 10; ++v) {
      locked_color_residual[v] = color_total[v] - color_correct[v];
      locked_color_total[v] = color_total[v];
      focus_tried[v] = false;
    }
    focus_levels_no_progress = 0;
    pick_focus_color();
    return true;
  }

  // Uncommit the whole ladder and ban its opener: the rebuilt ladder must
  // start from a different decomposition. Called on the second completed
  // focus circle without a lock commit (see solver_beam.hpp).
  inline void beam_solver::drop_ladder() {
    if (!locked_prefix) return;
    environment::remove(locked_prefix);
    locked_prefix = nullptr;
    locked_hash = 0;
    locked_correct_cells = 0;
    for (int v = 0; v < 10; ++v) {
      locked_color_done[v] = false;
      locked_color_residual[v] = 0;
      locked_color_total[v] = 0;
      focus_tried[v] = false;
    }
    focus_color = -1;
    m_focus_initialized = false;
    focus_levels_no_progress = 0;
    focus_progress_this_level = false;
    focus_circles_no_lock = 0;
    lock_commit_pending = false;
    cuda_dsl_set_focus_color(-1);
    cuda_dsl_set_done_colors(0);
    // Cold restart: the whole search state is descended from the dropped
    // prefix — beam, history pool, dedup, cache — and the prefix family
    // outscores every fresh starter, so anything kept recolonizes the trap
    // (observed 2026-07-12: 25 min of "flat" search still rooted in the
    // dropped prefix). The strike means: as if the puzzle just loaded,
    // locking off.
    release_candidates(beam);
    release_nodes(history_pool);
    history_by_depth.clear();
    for (int d = 0; d <= MAX_SEARCH_DEPTH; ++d) bucket_cutoff[d] = 0;
    visited_grids.reset();
    { unordered_map<uint64_t, cache_entry> tmp; global_cache.swap(tmp); }
    seed_starter_nodes();
    probe_failed.clear();
    locking_suspended = true;
    log_debug("==== LADDER DROPPED: cold restart, locking off for this puzzle ====");
  }

  inline void beam_solver::pick_focus_color(int exclude) {
    focus_color = -1;
    for (int v = 0; v < 10; ++v) {
      if (v == exclude || focus_tried[v] || locked_color_done[v]) continue;
      if (locked_color_residual[v] <= 0) continue;
      if (focus_color == -1 || locked_color_residual[v] > locked_color_residual[focus_color])
        focus_color = v;
    }
    // The bar to beat is what the lock already explains of this color.
    focus_best_correct = focus_color >= 0
        ? locked_color_total[focus_color] - locked_color_residual[focus_color] : 0;
    cuda_dsl_set_focus_color(focus_color);
    {
      int mask = 0;
      for (int v = 0; v < 10; ++v)
        if (locked_color_done[v]) mask |= 1 << v;
      cuda_dsl_set_done_colors(mask);
    }
    log_debug(focus_color >= 0
        ? fmt::format("==== FOCUS COLOR: {} ({} residual cells) ====", focus_color, locked_color_residual[focus_color])
        : "==== FOCUS COLOR: none (free search) ====");
  }

  inline void beam_solver::reload_current_puzzle() {
    sound_manager::get().clear_queue();
    clear_logs();
    cuda_dsl_invalidate_examples();
    cuda_dsl_set_focus_color(-1);
    cuda_dsl_set_done_colors(0);

    int puzzle_idx = m_solver_state.puzzle_idx;

    {
      scoped_lock lock(m_solver_state.mtx);

      // Load UNDER the lock: load_single_puzzle mutates m_puzzle, which draw_frame
      // reads under this same lock. Loading outside it races the GUI and can render
      // the new puzzle index against the old/half-swapped puzzle. (load is file IO
      // only and ignores solver_state, so no recursive-lock/deadlock risk.)
      const challenge_task &ct = m_puzzle_tasks[puzzle_idx];
      // Infer at load only if the pane is already open, so a puzzle brought up
      // to look at is ready on the first frame. Otherwise it waits — see the
      // check by the pause gate.
      m_puzzle_loaded = load_single_puzzle(ct, m_puzzle, *m_gpu_buffer, m_base_gpu_buffer_size, solutions_for(ct.id),
                                           puzzle_idx, (int)m_puzzle_tasks.size(), m_dataset_flags.load(),
                                           m_app_state.inference_view.load() != global_app_state::INFERENCE_OFF);
      if (m_puzzle_loaded) m_log.set_puzzle_id(m_puzzle.id);

      m_solver_state.attempts = 0;
      environment::remove(m_solver_state.current_tree);
      m_solver_state.current_tree = nullptr;
      auto now = chrono::steady_clock::now();
      m_solver_state.start_time = now;
      m_app_state.pause_start_time = now;

      active_example_count = 1;
      m_app_state.needs_upload = true;

      if (m_puzzle_loaded) {
          puzzle &p = m_puzzle;
          // Blank stale predicted-output views. load_single_puzzle reuses the
          // m_puzzle object and overwrites input/target but NOT the CURRENT/BEST
          // prediction views, so without this the previous puzzle's predicted grids
          // persist on this one (and stay stuck if this puzzle never overwrites them).
          for (auto &ex : p.train) { clear_current_view(ex); clear_best_view(ex); }
          for (auto &ex : p.test)  { clear_current_view(ex); clear_best_view(ex); }
          if (!p.is_solved) {
            environment::remove(p.best_tree);
            p.best_tree = nullptr;
            invalidate_best_tree_cache(p);
            p.best_node_count = NODE_COUNT_MAX;
            p.is_solved = false;
          } else {
            active_example_count = (int)p.train.size();
            cuda_dsl_invalidate_examples();
            m_solver_state.current_tree = environment::copy(p.best_tree);
          }
          m_solver_state.current_attempt_depth = 0;
          m_solver_state.current_attempt_nodes = 0;
      }
    }
    reset_search();
  }


  inline void beam_solver::worker() {
#ifdef _MSC_VER
    // Lower priority so the solver thread doesn't starve the GUI. Irrelevant
    // headless (dedicated box, no UI), so skipped there.
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#endif

    while (!m_app_state.stop_threads) {
      static thread_local vector<node*> tl_prev_nodes[2];
      static thread_local bool tl_slot_pending[2] = {false, false};

      // Worker-side tail of the one reset: drain the GPU pipeline slots this
      // thread owns, then return the node pool to its opening state.
      auto drain_pipeline = [&] {
        for (int s = 0; s < 2; ++s) {
          if (tl_slot_pending[s]) cuda_dsl_wait_for_exec(s);
          for (node *n : tl_prev_nodes[s]) environment::remove(n);
          tl_prev_nodes[s].clear();
          tl_slot_pending[s] = false;
        }
        node_pool.release();
      };

      // A queued dataset selection (from the GUI checkmarks) rebuilds the task
      // list, then falls through to reload the current puzzle. Handled like reset.
      int new_flags = m_queued_dataset_flags.exchange(-1);
      if (new_flags > 0 && new_flags != m_dataset_flags.load()) {
        {
          scoped_lock lock(m_solver_state.mtx);
          load_dataset(new_flags);
          if (m_puzzle_tasks.empty()) {
            m_solver_state.puzzle_idx = 0;
          } else if (m_solver_state.puzzle_idx >= (int)m_puzzle_tasks.size()) {
            m_solver_state.puzzle_idx = (int)m_puzzle_tasks.size() - 1;
          }
        }
        if (!m_puzzle_tasks.empty()) reload_current_puzzle();
        drain_pipeline();
        continue;
      }

      int delta = m_queued_puzzle_delta.exchange(0);
      if (delta != 0 && !m_puzzle_tasks.empty()) {
        {
          scoped_lock lock(m_solver_state.mtx);
          m_solver_state.puzzle_idx =
              (m_solver_state.puzzle_idx + delta) % (int)m_puzzle_tasks.size();
          if (m_solver_state.puzzle_idx < 0)
              m_solver_state.puzzle_idx += static_cast<int>(m_puzzle_tasks.size());
        }
        reload_current_puzzle();
        drain_pipeline();
        continue;
      }

      if (m_reset_solver_flag.exchange(false)) {
        reload_current_puzzle();
        drain_pipeline();
        continue;
      }

      // THE FACTS ARE READ WHEN SOMEONE WANTS THEM. Stepping through puzzles
      // with the pane shut and the solver paused costs only the parse; opening
      // the pane or starting the search asks for them, once. Checked before the
      // pause gate, because while paused the loop never gets past it — and on
      // this thread under the lock, since the GUI reads the same puzzle.
      if (m_puzzle_loaded && !m_puzzle.inferred &&
          (m_app_state.inference_view.load() != global_app_state::INFERENCE_OFF ||
           !m_app_state.is_paused.load())) {
        scoped_lock lock(m_solver_state.mtx);
        infer_into_puzzle(m_puzzle, m_solver_state.puzzle_idx,
                          (int)m_puzzle_tasks.size(), m_dataset_flags.load());
        m_app_state.needs_upload = true;
      }

      int target_idx = -1;
      if (should_pause_or_wait(target_idx))
        continue;

      if (!m_puzzle_loaded) continue;
      puzzle &p = m_puzzle;
      if (p.train.empty())
        continue;

      // Promote examples FIRST — BEFORE sizing the chunk. If the best tree already
      // solves the active example, promotion bumps active_example_count. Since the
      // grid is sized by job_count = chunk * active, sizing before promotion would let
      // active grow afterward and blow job_count past TARGET_JOBS -> device cudaMalloc
      // overshoot (unchecked) -> sticky CUDA error -> the GUI's cudaMallocHost aborts.
      if (p.best_tree && active_example_count < (int)p.train.size()) {
        unique_lock pl(m_solver_state.mtx);
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
                m_app_state.needs_upload = true;
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
        unique_lock fl(m_solver_state.mtx);
        if (!p.is_solved) {
          environment::remove(p.best_tree);
          p.best_tree = environment::copy(solve_found_by_promotion);
          p.best_node_count = solve_found_node_count;
          invalidate_best_tree_cache(p);
          cuda_dsl_invalidate_examples();
          refresh_best_display_scores(p, active_example_count, p.best_node_count);
          environment::remove(m_solver_state.current_tree);
          m_solver_state.current_tree = environment::copy(p.best_tree);
          m_app_state.needs_upload = true;
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
      { scope_ns t{ns_generate};
        pump_generate(p, &m_solver_state.attempts); }

      if (pending_evaluation.empty()) {
        scope_ns t{ns_idle};
        if (!is_generating_layer)
          this_thread::sleep_for(chrono::milliseconds(1));
        continue;
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
      { scope_ns t{ns_generate};
      while ((int)chunk.trees.size() < chunk_cap) {
        while ((int)chunk.trees.size() < chunk_cap && !pending_evaluation.empty()) {
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
          m_solver_state.attempts++;
          m_solver_state.current_max_depth = MAX_SEARCH_DEPTH;
          m_solver_state.current_attempt_depth = orig->max_depth;
        }
        if ((int)chunk.trees.size() >= chunk_cap) break;
        // Chunk not full and pending drained: generate more of the CURRENT level.
        pump_generate(p, &m_solver_state.attempts);
        if (pending_evaluation.empty()) break;  // level committed / dry -> launch what we have
      }
      }

      if (chunk.size() == 0) continue;

      // active is now final (promotion ran before chunk sizing above), so
      // job_count = chunk.size() * active <= chunk_cap * active <= TARGET_JOBS.
      const int active = min(active_example_count, (int)p.train.size());

      static thread_local cuda_dsl_exec_batch exec_batch[2];
      static thread_local int current_slot = 0;
      bool (&slot_pending)[2] = tl_slot_pending;
      vector<node*> (&prev_original_nodes)[2] = tl_prev_nodes;

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
      { scope_ns t{ns_launch};
        pack_cuda_dsl_exec_batch(exec_batch[launch_slot], chunk, active);
        cuda_dsl_exec_and_score(exec_batch[launch_slot], launch_slot); }

      // Wait on and process the previous batch (the other slot) while the batch
      // just launched runs on the GPU. The wait is now per-slot, so BOTH slots
      // are synchronized before their results are read — the old single
      // slot_0_pending flag skipped this for slot 1, an unsynced read.
      if (slot_pending[wait_slot] && prev_original_nodes[wait_slot].size() > 0) {
        { gpu_wait_scope t{m_solver_state};
          cuda_dsl_wait_for_exec(wait_slot); }
        scope_ns t{ns_harvest};
        // GPU fault detector (permanent). CUDA errors are sticky + cudaGetLastError
        // surfaces the error from ANY preceding call, so this one check is a catch-all
        // for every kernel/memcpy/malloc. If "magic madness" returns (frozen/garbled
        // grids, score spam), grep the puzzle log for "!!! GPU ERROR" — it names the
        // CUDA error + which launch. (The real fix is the bug; this makes it loud.)
        { char _ce[512]; if (cuda_dsl_take_error(_ce, sizeof(_ce))) {
            log_debug(fmt::format("!!! GPU ERROR (results now garbage until restart): {}", _ce)); } }
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

        { scope_ns t{ns_tally};
        for (int j = 0; j < prev_batch.job_count; ++j) {
          int pidx = prev_batch.program_indices[j];
          if (pidx < 0 || pidx >= program_count) {
            diag_log(fmt::format("[ASSERT pidx-OOB] puz={} j={} pidx={} program_count={} active={} batch_active={} job_count={}",
                                 p.id, j, pidx, program_count, active, prev_batch.active_examples, prev_batch.job_count));
            continue;
          }
          program_totals[pidx] += prev_batch.scores[j];
          if (!prev_batch.solved[j]) program_all_solved[pidx] = false;
        }

        for (int i = 0; i < node_count; ++i) {
          float score = program_totals[i];
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
        if (winner_idx >= 0 && !batch_stale) {
          scope_ns t{ns_winner};
          const int winner_node_count = prev_batch.node_counts[winner_idx];
          // Also a GPU block: its stream sync waits on the running kernel, so
          // it belongs in gpu_wait_ns or the busy readout under-reports.
          { scope_ns tf{ns_fetch};
            gpu_wait_scope tg{m_solver_state};
            cuda_dsl_fetch_output_grids(prev_batch, winner_idx, active, wait_slot); }

          // Reused across iterations. Bounded by p.train.size() / active,
          // both <= MAX_TRAIN_EXAMPLES (~10). assign() (not resize) so the
          // trailing [active, train.size()) slots are freshly defaulted
          // rather than carrying stale data from a prior pass; clear()+
          // reserve keeps capacity so neither reallocates after warmup.
          static thread_local candidate_batch winner_res;
          winner_res.train.assign(p.train.size(), candidate_result{});
          winner_res.total_train_score = 0.0f;
          winner_res.all_train_solved = true;
          static thread_local vector<score_breakdown> winner_breakdowns;
          winner_breakdowns.clear();
          winner_breakdowns.reserve(active);

          { scope_ns ta{ns_analyze};
          for (int i = 0; i < active; ++i) {
            int j = winner_idx * active + i;
            if (j < 0 || j >= prev_batch.job_count)
              diag_log(fmt::format("[ASSERT j-OOB] puz={} winner={} active={} batch_active={} i={} j={} job_count={}",
                                   p.id, winner_idx, active, prev_batch.active_examples, i, j, prev_batch.job_count));
            float gpu_score = prev_batch.scores[j];
            bool gpu_solved = (prev_batch.solved[j] != 0);

            const int8_t* og = prev_batch.output_grids.data() + j * GPU_GRID_CELLS;
            int ow = prev_batch.output_dims_result[j * 2 + 0];
            int oh = prev_batch.output_dims_result[j * 2 + 1];
            if (ow < 0 || ow > GPU_MAX_GRID || oh < 0 || oh > GPU_MAX_GRID)
              diag_log(fmt::format("[ASSERT dims-OOB] puz={} ex={} j={} ow={} oh={} winner={} active={} batch_active={}",
                                   p.id, i, j, ow, oh, winner_idx, active, prev_batch.active_examples));
            winner_res.train[i].state.width = ow;
            winner_res.train[i].state.height = oh;
            for (int r = 0; r < oh; ++r)
              for (int c = 0; c < ow; ++c)
                winner_res.train[i].state.data[r][c] = (int)og[r * GPU_MAX_GRID + c];
            winner_res.train[i].valid = true;
            winner_res.train[i].score = gpu_score;
            winner_res.total_train_score += gpu_score;
            if (!gpu_solved) winner_res.all_train_solved = false;

            const object_summary* tobjs = nullptr;
            int tobj_count = 0;
            if (i < (int)p.target_objects.size()) {
              tobjs = p.target_objects[i].objs;
              tobj_count = p.target_objects[i].count;
            }
            score_breakdown breakdown = environment::analyze_output_score(
                winner_res.train[i].state, p.train[i], winner_node_count,
                tobjs, tobj_count);
            winner_breakdowns.push_back(breakdown);
          }
          }

          uint64_t g_hash = (!winner_res.train.empty() && winner_res.train[0].valid)
              ? winner_res.train[0].state.hash() : 0;

          node *winner_tree = environment::copy(prev_original_nodes[wait_slot][winner_idx]);
          if (winner_tree) {
            log_score_formula(winner_tree, winner_breakdowns);
            report_score(winner_tree, winner_res.total_train_score, g_hash);
            { scope_ns tb{ns_best};
              update_best_solution(target_idx, p, winner_tree,
                                   winner_node_count, winner_res); }
            environment::remove(winner_tree);
          }
        }
        { scope_ns t{ns_free};
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
  }

  inline void beam_solver::update_example_results(vector<example> &examples,
                                     const vector<beam_solver::candidate_result> &results,
                                     int limit) {
    if (!m_gpu_buffer)
      return;
    int n = (int)examples.size();
    if (limit >= 0) n = min(n, limit);
    for (int i = 0; i < n; ++i) {
      example &ex = examples[i];
      if (i < results.size() && results[i].valid) {
        const grid& g = results[i].state;
        ex.cur_score = results[i].score;
        update_display_buffer(ex.current, g, *m_gpu_buffer);
      } else {
        clear_current_view(ex);
      }
    }
  }

  inline void beam_solver::clear_test_current_views(puzzle &p) {
    if (!m_gpu_buffer) return;
    for (auto &ex : p.test) clear_current_view(ex);
  }

  // When a candidate solves all train, run it on test inputs and show its
  // predictions in the CURRENT column. Otherwise clear, to avoid noise from
  // wrong candidates churning. We never read or display the ground-truth
  // test output here — only the tree's prediction on the test input.
  inline void beam_solver::update_test_current_views(puzzle &p, node *tree,
                                                     int node_count,
                                                     bool all_train_solved) {
    if (!m_gpu_buffer) return;
    if (!all_train_solved || !tree) {
      for (auto &ex : p.test) clear_current_view(ex);
      return;
    }
    for (auto &ex : p.test) {
      grid tg = {};
      if (ex.output.width > 0) {
        auto [ts, _slv] = environment::score_tree_output(
            tree, ex, tg, p, node_count);
        ex.cur_score = ts;
      } else {
        float _s = 0; bool _slv = false;
        bool ok = environment::execute_tree_via_flat(
            tree, ex, tg, p, node_count, _s, _slv);
        if (!ok) { clear_current_view(ex); continue; }
        // No ground truth — match BEST's convention: leave score untouched
        // (will mirror best_score, set when the tree was first promoted).
      }
      update_display_buffer(ex.current, tg, *m_gpu_buffer);
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

    // Lock-bound (deep) acceptance judges deletions by what the lock will
    // certify — cells explained and colors finished — never by beam score.
    // Unpainted cells score leniently (provisional paint), so score-judged
    // cleaning happily deletes the deposit op itself ("blank canvas scores
    // fine"), the gutted tree then fails lock qualification, and the
    // all-or-nothing fallback freezes the RAW tree, junk included (the
    // cliff — observed on 007bbfb7: "Output AddColor[0] FlipVertical Union"
    // reduced to "Output AddColor[0]", rejected, raw committed with the no-op
    // flip frozen).
    auto cert_flat = [&](const flat_program &f, int &correct, int &done_mask) -> bool {
      node *t = environment::build_chain_from_flat(f);
      if (!t) return false;
      int ct[10], cc[10];
      bool ok = layer_fit(t, f.count, p, active, ct, cc, correct);
      environment::remove(t);
      if (!ok) return false;
      done_mask = 0;
      for (int v = 0; v < 10; ++v)
        if (ct[v] > 0 && cc[v] == ct[v]) done_mask |= 1 << v;
      return true;
    };
    int base_correct = 0, base_done = 0;
    bool cert_mode = deep && cert_flat(flat, base_correct, base_done);

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
    auto sweep = [&](int k) -> bool {
      int idx[flat_program::MAX_OPS];
      int best_drop[flat_program::MAX_OPS];
      int best_found = 0;
      float best_sum = base_sum;
      uint32_t best_mask = base_mask;
      int best_correct = base_correct;
      // Standard k-combination walk over droppable positions 1..count-1.
      for (int i = 0; i < k; ++i) idx[i] = i + 1;
      while (idx[k - 1] < flat.count) {
        flat_program reduced;
        for (int i = 0, j = 0; i < flat.count; ++i) {
          if (j < k && i == idx[j]) { ++j; continue; }
          reduced.push(flat.ops[i].type, flat.ops[i].value);
        }
        if (cert_mode) {
          // Accept when certification is preserved: no explained cell lost, no
          // finished color un-finished — fewer nodes is the win, ties included.
          int c = 0, dm = 0;
          if (cert_flat(reduced, c, dm) && c >= base_correct &&
              (dm & base_done) == base_done && (best_found == 0 || c > best_correct)) {
            best_correct = c; best_found = k;
            for (int i = 0; i < k; ++i) best_drop[i] = idx[i];
          }
        } else {
          uint32_t m = 0;
          float s = score_flat(reduced, m);
          if ((m & base_mask) == base_mask && s > best_sum + 0.0001f) {
            best_sum = s; best_mask = m; best_found = k;
            for (int i = 0; i < k; ++i) best_drop[i] = idx[i];
          }
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
      base_correct = best_correct;
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
      if (sweep(k)) { changed = true; k = 1; }
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

      set_example_scores(p.train[active], s);
      update_example_views(p.train[active], temp, m_gpu_buffer);
      if (!solved) break;

      active++;
      promotion_count++;
      active_example_count = active;
      log_debug(fmt::format("==== PROMOTED: now solving {}/{} examples ====",
                       active_example_count, p.train.size()));
    }

    // No overfit re-verify needed here: locks are vetted against the FULL
    // train set at commit time (try_lock_layer), so widening the active set
    // cannot invalidate one.

    // Filling the train set is the solve verdict — `tree` just verified on
    // every promoted example. Park it; the worker loop marks the task solved.
    if (promotion_count > 0 && active == (int)p.train.size() && !p.is_solved) {
      if (solve_found_by_promotion) environment::remove(solve_found_by_promotion);
      solve_found_by_promotion = environment::copy(tree);
      solve_found_node_count = node_count;
    }
    return promotion_count;
  }

  // Shared train-finale: any path that widens the active set to the full train
  // with a solving best tree must convert that into the solved state + pause.
  // The finale inside update_best_solution only runs for candidates that pass
  // its improvement gate — a tree that merely ties the stored best never enters,
  // so a solve discovered via promotion alone would otherwise search until
  // timeout (observed on 0692e18c: reduction found the solving tree, the worker
  // loop promoted to 3/3, and nothing ever fired the finale).
  inline void beam_solver::finalize_full_train_solve(puzzle &p) {
    if (p.is_solved || !p.best_tree) return;
    if (active_example_count < (int)p.train.size()) return;
    bool test_ok = evaluate_test_after_solve(p.best_tree, p, p.best_node_count);
    p.is_solved = true;
    p.is_test_solved = test_ok;
    for (auto &ex : p.test) ex.output.visible = true;
    log_debug("!!!! TASK SOLVED ON TRAINING SET !!!");
    if (test_ok) log_debug("!!!! TEST SET SOLVED CORRECTLY !!!!");
    vector<pair<int, int>> seq;
    seq.push_back({1568, 150});
    if (test_ok) seq.push_back({2093, 250});
    sound_manager::get().play_sequence(seq);
    if (!m_app_state.is_benchmarking) toggle_pause();
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
      // but it still gates display — matching apply_solution's reload path.
      set_example_scores(p.test[i], ts);
      if (p.test[i].output.width > 0) {
        // Ground truth available (eval/training sets): tslv tells us whether the
        // hidden test output was actually solved.
        if (!tslv) test_all_ok = false;
      } else {
        // No ground truth (competition test set): can't confirm a solve.
        test_all_ok = false;
      }
      p.test[i].output.visible = true;
      update_example_views(p.test[i], tg, m_gpu_buffer);
    }
    return test_all_ok;
  }

  inline void beam_solver::reset_test_display(puzzle &p) {
    p.best_node_count = NODE_COUNT_MAX;
    for (auto &ex : p.test) {
      if (m_gpu_buffer) {
        clear_current_view(ex);
        clear_best_view(ex);
      }
    }
  }

  inline void beam_solver::refresh_best_display_scores(puzzle &p, int active, int node_count) {
    bool all_train_perfect = true;
    for (int i = 0; i < (int)p.train.size(); ++i) {
      if (i >= active) {
        if (m_gpu_buffer) clear_best_view(p.train[i]);
        continue;
      }
      grid temp = {};
      auto [s, solved] = environment::score_tree_output(
          p.best_tree, p.train[i], temp, p, node_count);
      p.train[i].best_score = s;
      update_grid_if_visible(p.train[i].best, temp, m_gpu_buffer);
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
        p.test[i].best_score = score;
        update_grid_if_visible(p.test[i].best, tg, m_gpu_buffer);
      }
    } else {
      for (int i = 0; i < (int)p.test.size(); ++i) {
        clear_best_view(p.test[i]);
      }
    }
  }

  inline void beam_solver::try_update_current_display(int target_idx, puzzle &p, node *tree, int node_count, const candidate_batch &res) {
    unique_lock<mutex> lock(m_solver_state.mtx, try_to_lock);
    if (lock.owns_lock()) {
      if (target_idx == m_solver_state.puzzle_idx) {
        update_example_results(p.train, res.train);
        int active = min(active_example_count, (int)p.train.size());
        bool full_train = res.all_train_solved && active == (int)p.train.size();
        update_test_current_views(p, tree, node_count, full_train);

        environment::remove(m_solver_state.current_tree);
        m_solver_state.current_tree = environment::copy(tree);
        m_app_state.needs_upload = true;
      }
    }
  }

}
