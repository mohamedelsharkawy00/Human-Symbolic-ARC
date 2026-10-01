Plan For:
Auto-Pilot Benchmark Mode

Goal:
Implement a visual benchmarking mode directly inside the existing GUI, no command line,
allowing the solver to automatically rip through puzzles and log its results
without requiring the user to manually click "Next".

Remaining Work:

1. State: Add 'is_benchmarking', 'benchmark_timeout_seconds', and
   'benchmark_puzzle_limit' to global_app_state.
2. Trigger: Hook up a '[B]' key command and a "BENCHMARK" button in the
   GUI header to turn the mode on/off.
3. Execution Loop (main.cpp): Each frame, check if puzzle is solved or
   timed out. Log result via format_solution_result(), then auto-advance
   with change_puzzle(1).
4. Summary Block: After all puzzles complete, append totals —
   Puzzles Run, Train Solved (%), Test Solved (%), Bench Time, Avg APS.
5. Incremental Flushing: Flush benchmark_results.txt after every puzzle
   so a crash doesn't lose prior results.
6. Version & Date Headers: Print Date/Time at the top of
   benchmark_results.txt for traceability.

The 2-Minute "Smoke Test" Profile
To quickly verify if a codebase change broke anything without waiting 8+ hours:
- Target Limit: First 100 puzzles (--limit=100)
- Timeout Limit: 1.2 seconds per puzzle (--benchmark=1.2)
- Total Duration: ~2 minutes max.
- Reasoning: 1.2 seconds provides the C++ solver plenty of time to explore
  millions of nodes. A 2-minute loop allows for extremely rapid iteration.
