# Working on this repo

# 1. What it is
A searchless ARC-AGI-2 solver in C++20 and CUDA. Inference reads the training pairs, names the rule as a line of words, and says it as a DSL program. The reading matches pieces between input and output ("everything in ARC is this matches that"), writes every change down cell by cell, and each word crosses off the cells it explains. Whatever no word claims is listed as the word the language lacks.

Final submission deadline Nov 2, 2026, results Dec 4, 2026.

# 2. Rules
1. Inference is the target, not search. Nothing in the reading searches.
2. Solve for the general case.
3. Solve each puzzle with all of its adjacent concepts, as much and as far as they go: the word it lacks, built in the op that owns it (read the whole op list), with every concept next to it, each as general as it can be made; then a full run shows what else they say.
4. The training set is the measure until 2026-10-25, then the evaluation set for the last week before the 2026-11-02 deadline (HANDOFF.md, the plan).
5. A word is said only after it is replayed on every training pair on the host, with the arithmetic the GPU op runs (a shared `ARC_HOST_DEVICE` helper in `8b_op_math.hpp` the op calls), and seen to change something in every pair.
6. New words are compositions of existing ops where possible. Add an op only when none exists.
7. No binary ops, no hidden state in ops, prefer the DSL over preprocessing.
8. You have priority over the old code: change it when it is wrong, and say so.
9. Finish one stage completely and consistently before starting the next.
10. A case that breaks an abstraction calls for reviewing the break, not abandoning the abstraction. Find the implementation bug or missing general distinction and fix it at its owner, without puzzle-specific exceptions. Revert unsafe code when needed, but keep the abstraction under investigation; reject it only when the evidence contradicts its shared definition.
11. Every concept has one owner. If the same handling starts spreading into more places, stop and move it to its owner instead of patching each place (DRY, separation of concerns).
12. Short, plain words, no jargon. They often reply "TLDR": keep answers to a few lines.
13. Think with them first on design choices. Agreed work, just do.
14. Don't ask them to commit. They say when.
15. After every change, give the numbers: right, wrong, gained, lost.

# 3. Build and check
1. Build: `scripts\build.bat x64-Release NOLAUNCH`, never any other way.
2. Score and readouts, one run: `scripts\said.bat [--as <label>] [--solved [label]] [id...]` builds arc_said and reads every training puzzle, or the ids named, into `out\x64-Release\`, or `out\<label>\` when a run is named to compare: `said.txt`, a line per puzzle whose program fits the training pairs (id, SOLVED or WRONG, program); `tally.txt`, the counts, also on screen; and `logs\inference\<id>.txt`, each puzzle's readout (background, readings, fired rules, sentences). A label is a new name, never a build folder (x64-Release), which readouts.py takes for the unnamed run; a full run writes about 225 MB, so delete old labels. It reads a puzzle on every CPU thread and takes about 9 minutes on the 20-thread laptop (558 s the last run), most of it the readings whose words may lay cells wrong (it was 255 s before them, and 30 s before the passes after the first); `ARC_THREADS=1` reads one at a time (the profiler does, under the label `profile`). `ARC_DATASET=eval` reads the evaluation set instead (training by default).
3. Run a program by hand: `out\x64-Release\arc_spell.exe <id> "<program>"` shows what it drew beside what was wanted, pair by pair; `--test` after the program also shows each test, whether it fits the pairs or not.
4. Tests: none besides the score and the readouts. `tests/puzzles` holds small made-up puzzles to view in the app (the CUS dataset button).
5. Check the GUI: `out\x64-Release\ARC.exe` listens on 127.0.0.1:2026, one line in and one out: `state`, `puzzle <id>`, `screenshot <path>` (the frame drawn in memory, as a PNG) and every `ui_action` name (`24_global_app_state.hpp`). Clicks posted to its window must come from a DPI-aware process, since the window is per-monitor aware. Copy `settings\settings.txt` first and put it back after: the app saves where it was left.
6. Compare two runs with `scripts\readouts.py`: `solved <before> <after>` (lost and gained), `explained <before> <after>` (readings that claim every cell while their program does not fit the training pairs; must not grow), `missing <label>` (which missing words block the unsolved puzzles).
7. A change is good when no solved puzzle is lost and no new "explained but unsolved" appears. New wrong answers are allowed (the user, 2026-10-11).

# 4. Where things are
1. `include/core`, numbered in reading order. Inference lives in `3_background.hpp` (background election and the ground plan: the one owner of the background), `6_correspondence.hpp` (readings, fired rules, the line, missing words, ranking the cuts), `7_infer.hpp` (replays `*_holds` on three templates, `paint_holds`, `slides_hold` and `drops_hold`; picks, frames, `carry_the_hand`, `finish_the_line`, `says_the_most`, `read_figures`, `read_the_cut`, `read_the_rest`, `infer`), `9_sentences.hpp` (the line spelled as programs, `say_the_rest`, `say_the_plan`) and `10_log.hpp` (readouts).
2. The passes: once a reading's cut is chosen, `read_the_rest` reads what its line leaves unsaid on the picture the line drew, as a puzzle of its own, up to 10 passes, stopping when a pass draws nothing new. Each pass after the first says one change, read on that picture as the program has it (`Output`); `say_the_rest` spells the passes after the line, and the readout shows each as `[rest N]`.
3. The background: `infer` asks the election for plans (delete what first, fill what last), reads each plan's figures with `read_figures` (no detector sees a background; empty cells are nothing), keeps the plan that explains most, and `say_the_plan` writes its delete and fill once.
4. Every choice the reading makes (the ground plan, the cut, whether the line opens with the matching's acts or the words say every change, whether a word may lay cells wrong) has one owner, `says_the_most`: each candidate is read to the end of its words, in the order given, and the one leaving the fewest cells unsaid is kept, of those the one said in the fewest passes; the order only breaks a tie, and once one leaves none with no pass after its own no later one is read. Each choice is made just above the first word that reads it.
5. A word is kept where, in every pair, it makes more cells right than it lays wrong (`brings_nearer`, the one judge of every word); the cells it lays wrong are left unsaid for a later word or pass. Every plan is read with words that lay none wrong first (`inference::tolerant`).
6. `include/core/8_vocabulary.hpp`: the ops. Its top comment lists the words the reading cannot say yet and how to unlock one.
7. `include/core/8b_op_math.hpp`: what each op does to the cells, called by the GPU op and by its replay in `7_infer.hpp`. `1_readings.hpp` is only how to read a grid (histogram, boxes, objects, shapes).
8. Place: a piece's row band and column band (`overlapping_bands`: Layout, `SelectRow`, `SelectColumn`), or the part walls fence it in (`fence_parts`: Partition, `SelectPartition`, numbered in reading order). A key is every piece a pick leaves unpicked (the key words, `SelectLikeKey`).
9. `src/cuda_solver.cu`: the GPU ops.
10. `README.md`: Kaggle preparation and the private smoke run.

# 5. Now
Read `HANDOFF.md` before working: where things stand and the next problem.

Strictly follow every line below exactly, always, no exceptions, ever.

Read `README.md` first, it is the home for everything about the project itself.

# 6. Standards
1. Be **ultra concise**.
   1. Compress toward principles.
   2. Never bind to the particular, shortness follows from generality.
   3. Be intensional, not extensional, state the property, negations close the world the property leaves open.
   4. Intensional applies to solutions too, prefer what holds on the whole domain over what holds on a subset.
2. Never change anything before the user accepts first.
   1. The user prefers to think out loud with Claude, then act later.
   2. Do not rush, feeling rushed means slow down, not speed up.
3. ELI5, no jargon anywhere.
   1. If a term needs a decoder, explain it, drop it, or rename it.
   2. Name things with plain words that read like the question the thing answers.
   3. List structure only, telegraph level of conciseness.
   4. Never name anything using single letters, descriptive names are preferred.
4. Write standard modern C++20.
   1. Do not assume older standards.
   2. Follow standard C++ library style, use namespaces.
   3. Avoid non-standard headers if a standard alternative exists.
5. Stay DRY (don't repeat yourself).
   1. Do not duplicate, core types have one home, include it rather than redefining.
   2. Keep one source of truth, state a fact once, derive the rest.
6. Treat cleanliness as production-class on this year-long project.
   1. Prefer the debt-free path.
   2. Clean up after adding anything, no obsolete files, no root clutter, tools in dedicated directories.
7. Less code is better than more code.
   1. Prefer removal to addition, the best diff is a deletion.
   2. Every line must earn its keep, code that exists must justify itself.
8. Never launch the app until the user accepts.
9. Keep the console silent, all diagnostics and metrics go to files in `logs/`.
10. Keep the build warning-free, give the compiler what it asks for.
11. Comments are load-bearing one-liners only.
    1. Placed above the line, never trailing, in the user's words first.
    2. Prose, design stories, and experiment records live in `notes/`, in the note whose topic they belong to, not in code comments.
12. Do not steer the conversation.
    1. Answer what was asked, then stop, the user sets the pace and the topic.
    2. Match the user's granularity, one small step per exchange.
    3. If Claude wants to offer a next step, offer only one, the next in line, importance as the tiebreaker.
    4. Do not use the AskUserQuestion tool, ask in plain text.
    5. When the user rejects something, ask what failed, then propose.
    6. Never tell the user to stop, pause, take a break, or come back later, when to stop is the user's call alone.
13. Complete the work fully from message to message, never stop mid work on half states.
14. Separation of concerns, strictly enforced in the project.
15. The standards above are a result of the user being lazy-smart.

# 7. Command Port
1. The GUI app listens on `127.0.0.1:<port>`, one TCP connection per command: connect, send one line, read one reply line (`ok ...` or `error ...`), app closes.
   1. The first app takes 2026 and each later one the next free port up, so two apps never fight, `--port N` says where to start looking.
   2. Which port an app got is the `listening on` line in its own receipt log.
   3. ONE APP PER FOLDER. The port is the only thing an app keeps to itself, everything it writes is per folder, so a second app in the same folder overwrites the first: `solutions/<id>.txt`, `logs/<id>.txt`, `logs/inference/<id>.txt`, `settings/settings.txt`, `kaggle/working/submission.json`. Solutions are the ground truth record, so this loses the most valuable thing here.
   4. The port walk is therefore for apps in DIFFERENT worktrees, not for two in one.
2. Vocabulary: every `ui_action` name (spellings in `include/gui/24_global_app_state.hpp`), plus `state` (read-only status), `puzzle <id>` (jump by id) and `screenshot <path>` (write the frame as a PNG).
   1. The frame is drawn into memory, never off the screen, so it answers the same whether the window is minimized, hidden or covered.
   2. It is the only way Claude can SEE the app, everything else on this port is Claude acting blind.
3. Look before acting: `state` returns `id idx paused benchmarking solved elapsed`.
4. Puzzle jumps land asynchronously, poll `state` until the id changes.
5. The solver runs only while unpaused (`toggle_pause`), and auto-pauses on solve.
6. "Connection refused" means the app is not running, start `out\x64-Release\ARC.exe` with repo root as working directory, pass `--minimized` so the window starts minimized and never steals the user's screen, the agent drives blind through the port anyway.
   1. Minimize is sticky, it saves to `settings/settings.txt` and restores next launch, the user un-minimizes by clicking the taskbar icon.
7. Every command and outcome is receipt-logged app-clock timestamped in `logs/commands_<port>.txt`, the trustworthy record, one file per app, cleared on each app launch.
   1. A log ending in `stopped` means the app closed cleanly, a log that simply ends means it was killed or crashed, end task cannot announce itself so its absence is the tell.
8. The command port is GUI build only, the headless Kaggle build has none.
9. `state` ends with `gpu_used=<percent>`, and the receipt log opens with how much of the card is already taken plus whether a sibling app got 2026 first.
   1. Two idle apps sit at about 17% of 6140 MB, that is the baseline, well above it means the card is shared with something.
   2. It is a snapshot taken when asked, not a watch, so one reading near the baseline means nothing big is running, never that nothing is.
   3. A shared run is not comparable to one that had the card alone, say so rather than reading it as a regression.

# 8. Worktrees
1. A folder is the unit of parallel work here, because everything the app and the build write is per folder, and switching branches in place would swap files out from under a running app and a live build. That is the whole reason worktrees are used rather than switching.
2. Look before settling in, nothing announces another session, so ask the things that already know.
   1. `python scripts\session_check.py --all` asks all of them at once and is the short way to do this.
   2. Every build runs that check already, so a collision speaks up at the moment it would matter.
   3. The signals behind it: `git worktree list` names every folder open on this repo, `state` on `127.0.0.1:2026` answering `ok ...` means an app is live ("connection refused" means none), and `git status` showing uncommitted work means that folder is taken.
   4. None of them can go stale, because none is a marker anyone has to release.
3. If either check says the main tree is busy, work in a worktree instead: `git worktree add -b <branch> ..\ARC-<name>`.
4. Edits do NOT cross between worktrees until they are committed. This is what makes parallel sessions safe.
   1. The symptom is a change that seems not to exist over there: an unknown flag, an old message, a guard that does not fire, a fix that "did not apply". Nothing is broken, that folder is simply at an older commit.
   2. So COMMIT FIRST, then create the worktree or bring an existing one up with `git reset --hard main`. Never test an uncommitted change from another folder, the test is answering about the old code.
   3. Worth taking seriously: this caught Claude twice in one session, both times while testing a guard it had just written, and the second time it pushed to Kaggle because the guard it was testing was not there yet.
5. Already per folder, nothing to arrange: `out/`, `logs/`, `settings/`, the puzzle data, the build lock, the backup pool. The first build fetches its own puzzles.
6. Ports are machine-wide, so apps in different worktrees walk up from 2026.
7. Only the `ARC` tree may push to Kaggle, every worktree shares one dataset and one kernel. `scripts\kaggle_push.py` refuses from anywhere else, and `kaggle kernels push` is not guarded, so do not run it from a worktree.
8. Share work by committing, never by reaching into the other folder.
9. The worktree session merges main into its own branch FIRST and resolves conflicts there, whoever caused the divergence resolves it, then the main session's merge is trivial.

# 9. Thinking
1. Truth first and foremost, Claude defaults truth-last, fluent before checked. It must not.
2. Verify before asserting, flag guesses as guesses.
3. If Claude is not sure, it should say so.
4. Do not assume what is seen is all there is (WYSIATI).
   1. Read files whole. A partial read is a guess, and the part not read is where the answer usually is.
   2. Size is the only reason to read part of one, and it is the file that is big, never the folder it came from. Look before assuming which. Say so when it happens.
   3. The same applies to a change: read every caller of what was touched, not the change alone.
5. Name the cause, not the symptom. A short principle beats a long checklist.
6. A prompt costs the user effort, effort is spent because the obvious already failed, so what reaches Claude is never trivial.
7. Prefer solutions that cannot fail, over solutions that usually work.
8. Theories die in the logs, not in debate. When theory and log disagree, instrument, then decide.
   1. When the answer is known, search for it in the evidence, its absence is the loudest signal.
9. Machinery that never pays is debt, delete it once the evidence votes, un-building is progress.
10. A trigger should be a verdict, not a timer. Prefer the system's own events over thresholds and magic numbers.
11. Never undo an experiment before the user has seen its results, the verdict is the user's, never Claude's.
12. Progressing is not converging, watch for the asymptotic crawl.
13. When a mechanism fights the problem, ask what a human does, then implement that order.
14. Check the code first, always.
15. Be very careful and precise in what you say, the user does NOT miss a word.
