ARC Project Status & Developer Guide (CUDA Transition)

Project Overview
The project is currently transitioning from a tree-based recursive executor to a batched CUDA-native solver. We have completed the migration to a flat executor (active) and are currently in the "Shadow CUDA Authority" phase.

Current Milestone: Guarded CUDA Authority
- Status: Active
- Core Goal: Move from proving that CUDA agrees with CPU to actually consuming CUDA-selected backend payloads in the worker flow.
- Agreement Level: 100% (1.0000) consistently observed in current benchmarks (accept_rate, use_rate, select_rate).

Quick Reality Check
- Unary execution path is now Flat.
- Legacy recursive executor is Retired (preserved in /notes/legacy_recursive_executor.txt).
- CUDA validates and mirrors chunk-level scores in a Sidecar path.
- APS is healthy (~30k-40k) with sampled logging (Period = 100).

Architecture & Key Files

File: include/solver_executor.hpp
Purpose: Primary Workflow
Critical Notes: Contains the worker() loop, CUDA score service, and shadow/authority logic.

File: include/solver_beam.hpp
Purpose: Beam & Chunking
Critical Notes: Owns flat_candidate_chunk and handles candidate lifecycle.

File: include/environment.hpp
Purpose: Flat Engine
Critical Notes: Defines flat_program, batch packing, and DSL flattening logic.

File: src/cuda_solver.cu
Purpose: CUDA Implementation
Critical Notes: Host/Device interface for batch scoring.

File: scripts/build.bat
Purpose: Preferred Build
Critical Notes: Modern script that kills running ARC.exe, builds via presets, and auto-launches.

Historical Context & Troubleshooting
Important lessons from previous sessions to avoid regression:

1. Stack Size Overflows: flat_program_batch is ~99KB. DO NOT allocate it on the stack. It has been moved to the heap inside flat_candidate_chunk via unique_ptr.
2. Move/Copy Constructor Bug: flat_candidate_chunk needs custom move logic to prevent crashes during worker cleanup. Ensure clear() and move assignments are consistent.
3. Pointer Lifetime: Mismatches previously occurred because the chunk validator held pointers to temporary grids. Fixed: cuda_chunk_score_validation_state now owns copied grids in attempt_storage.
4. Logging Performance: Always-on CUDA cross-checks tanked APS. Current validation is sampled (CUDA_VALIDATION_PERIOD = 100). Keep logging to sampled mode by default.

Next Phase: "Full CUDA" Execution

Step 1: Guarded Shadow Winner Path
- Use cuda_backend_selected_program as a shadow winner.
- Track how often the worker could have consumed the CUDA payload without deviation.

Step 2: Unified Winner Payload
- Create a selected_backend_winner_payload object containing the flat_program, node count, and train scores.
- This object should be the single "source of truth" for what the solver chooses next.

Step 3: Guarded Worker Consumption
- Sparse sampling: let the worker() choose the CUDA-selected program as the winner.
- Compare the downstream solver state-if it stays identical, we are ready for step 4.

Step 4: Shadow Consume
- Derive candidate choice and payload handling from CUDA for sampled chunks.
- Keep CPU as an auditor/authority for final safety checks.

Critical Reminder - Build System
We have consolidated to scripts/build.bat.
- Use scripts/build.bat to ensure a clean build-run loop that handles process termination automatically.

Last Updated: 2026-04-04
