# Which Design Pattern Fits (2026-10-23)

Asked: the code names almost no design patterns; which one standard pattern
fits it best? **Blackboard.** Below it, every other pattern that fits a part
of the code or nearly fits, for learning.

## Blackboard: the one that fits

A shared board, specialists who each write the part they know on it, and a
controller who picks who writes next. The standard pattern for problems with
no fixed recipe, such as speech and image recognition, since the 1970s.

The reading already is one, unnamed:

- **Board**: each pair's left-over cells. Saying a word crosses off the cells
  it explains (`say_fired`, `6_correspondence.hpp`). The facts the detectors
  write first (`13_infer.hpp`) are on it too.
- **Specialists**: the 19 kinds in `finish_the_line` (`12_line.hpp`), 8 words
  and 11 paints, each a lambda that offers the cells it would claim and how to
  say it.
- **Controller**: the loop at the end of `finish_the_line`. Every kind offers,
  the one claiming the most cells nothing claims yet is said, and all are
  asked again until no offer claims a cell.
- **What stays on the board**: the missing word (`list_what_is_missing`).

Naming it would mean:

- Each kind becomes its own unit with one shape: an offer (the cells it would
  claim) and how to say it.
- The locals the lambdas share today (the hand, the canvas, the cells laid
  wrong, the caches) become the board's own fields. A guess: most of the work
  is there.
- `finish_the_line` (1,716 lines) shrinks to the controller loop: the shape
  for the open item "finish_the_line as a struct".
- A kind's name lives with it. Today `kinds[]` and `inference_times::names`
  are two lists kept in step by a `static_assert`.
- Not changed: the ops (`src/ops.cpp`), and the search inside each specialist
  (every pick and value tried, the best kept). Naming the pattern gives that
  search one home per kind; it does not remove it.

How to know it held: a pure move, so a full training run against the last
baseline shows said.txt, tally.txt and every readout identical, as the
7_infer split did (af30bca).

Status: suggested, not agreed as work, not started.

## The others that fit a part

| Pattern | In plain words | Where it is here |
|---|---|---|
| Interpreter | a small language, run by walking its program | the DSL: `parse_pipeline` reads a program, `one_op`'s switch runs it (`src/ops.cpp`), `program_run` holds what one run reads and writes |
| Pipes and filters | each step takes one stream and hands one on | every op takes one object list and hands one on (`run_ops`) |
| Layers | a lower layer never knows the one above | `include/core` never includes `include/gui`; the build enforces it |
| Facade | one simple door to a big system | `infer()`, `environment::draw`, and `solver`: "give it a puzzle, ask it to read" |
| Table-driven | one table, everything else derived from it | `NODE_PRIMITIVES`: every op's name, range and role |
| Observer | the doer announces, watchers listen | `solver_observer`: the solver says what it did, whoever drives it listens |
| Ports and adapters | the core reports through one fixed shape, adapters plug in | `search_stream_writer` turns `solver_observer` calls into text lines for the GUI |
| Command | an action as a value, run by one executor | `ui_action`: keys, clicks and the command port all go through `execute_action`; in the reading, an offer's `say` is an action kept for later |
| Strategy | the varying part handed in | `command_server` is handed what commands mean by `main()` |
| Template method | a fixed skeleton, one step varying | `slides_hold` is the skeleton; `align_holds`, `toward_holds`, `move_holds`, `to_edge_holds` and `shift_holds` hand in only where each piece goes. Done with a function and a lambda, not a base class |
| Adapter | make one type answer another's questions | `partner_view` lets the op's objects answer the questions `15_op_math.hpp` asks; `partner_pieces` does it for the reading |
| Iterator (internal) | hand the loop body in, not the loop out | `each_painted`, `for_each_colored_cell`, `each_object`, `each_tile_offset`, `each_program_word` |
| Specification | rules for picking, combined with and, or, not | the DSL's picks: selectors add (or), `InvertSelection` flips (not), `WouldFit` and `SelectLikeKey` keep only some (and) |
| Chain of responsibility | ask in order; the first that can, does | the whole-program rules at the top of `finish_the_line`: the first that holds is the whole line |
| Decorator | wrap to add a check, same shape out | `weigh` wraps a claim with "not in a finished pair", `weigh_paint` with "not already drawn by the line" |
| Double buffer | read one, write the other, swap | `program_run::fresh()`: two lists, an op reads one and writes the other |
| Null object | a harmless empty stand-in instead of nothing | `no_table` in `environment::draw`; the empty puzzle `solver::get_puzzle` hands back |
| Singleton | one instance, reached from anywhere | `sound_manager::get()` |
| Memoization | keep an answer, never work it out twice | `figure_readings`, `replayed_once`, `laid_once`, the GUI's `preview_cache` |
| Generate and test | try every candidate, keep the best | `says_the_most`: each cut, each opening, each tried on its own copy of the reading. The search the searchless work aims to remove |
| Process per task (bulkhead) | each job walled off, so a failure stays put | one `arc_headless` process per puzzle (README section 5) |
| Immediate-mode GUI | redraw everything from the state every frame | `draw_frame` draws each frame from `global_app_state`; the click zones are rebuilt every frame |

## Near misses

| Pattern | Why not here |
|---|---|
| Production system (rule engine) | The runner-up: its match, pick, act cycle is the controller's loop. Blackboard fits better because the specialists write partial answers on a shared board, and what stays unclaimed means something: the missing word. |
| Strategy, as the main pattern | Strategy picks one way up front. Here every kind is asked each round and the best is said, which is the blackboard's controller. |
| Composite | Needs trees. Programs are chains: every op takes one input (no binary ops, AGENTS.md). |
| Visitor | Pays off with many operations over a fixed set of node types. Programs have one operation, run, and one switch does it. |
| Factory (method or abstract) | Nothing chooses between families of objects; the ops are plain functions. |
| Memento, Prototype | A reading is copied before each try (`inference read = inf;`), as a plain value copy; neither pattern's machinery is needed. |
| Table-driven, in the GUI | `ui_action` and `ui_action_names` are two lists kept in step by hand, where the ops use one table. |

Not checked: Bridge, Builder, Flyweight, Mediator, Proxy, State.
