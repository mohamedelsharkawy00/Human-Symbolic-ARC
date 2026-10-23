# 1. What is this project?
1. A searchless ARC-AGI-2 solver in C++20.
2. Inference reads the training pairs, names the rule as a line of words, and says it as a DSL program.
3. The reading matches pieces between input and output ("everything in ARC is this matches that").
4. It writes every change down cell by cell.
5. Each word crosses off the cells it explains.
6. Whatever no word claims is listed as the word the language lacks.
7. Final submission deadline Nov 2, 2026, results Dec 4, 2026.
8. Read `README.md` first, it is the home for everything about the project itself.
9. Read `HANDOFF.md` before working: where things stand and the next problem.
# 2. Rules
1. Inference is the target, not search. Nothing in the reading should be doing any search.
2. Prefer solving for the general case rather than the specific puzzle given.
3. Solve each puzzle with all of its adjacent concepts, as much and as far as they go: the word it lacks, built in the op that owns it (read the whole op list), with every concept next to it, each as general as it can be made; then a full run shows what else they say.
4. The training set is the measure until 2026-10-25, then the evaluation set for the last week before the 2026-11-02 deadline (HANDOFF.md, the plan).
5. A word is said only after it is replayed on every training pair on the host, and seen to change something in every pair.
   1. The replay runs the arithmetic the op runs: a shared helper in `9_op_math.hpp` the op (`src/ops.cpp`) calls.
   2. Anything less would let the language say a word the pairs don't back.
6. New words are compositions of existing ops where possible. Add an op only when none exists.
7. No binary ops, no hidden state in ops, prefer the DSL over preprocessing.
8. The agent has priority over the old code: change it when it is wrong, and say so.
9. Finish one stage completely and consistently before starting the next.
   1. Complete the work fully from message to message, never stop mid work on half states.
10. A case that breaks an abstraction calls for reviewing the break, not abandoning the abstraction. Find the implementation bug or missing general distinction and fix it at its owner, without puzzle-specific exceptions. Revert unsafe code when needed, but keep the abstraction under investigation; reject it only when the evidence contradicts its shared definition.
11. Every concept has one owner. If the same handling starts spreading into more places, stop and move it to its owner instead of patching each place (DRY, separation of concerns).
   1. Stay DRY (don't repeat yourself).
      1. Do not duplicate, core types have one home, include it rather than redefining.
      2. Keep one source of truth, state a fact once, derive the rest.
   2. Separation of concerns, strictly enforced in the project.
12. Short, plain words, no jargon. The user often replies "TLDR": keep answers to a few lines.
   1. ELI5, no jargon anywhere.
      1. If a term needs a decoder, explain it, drop it, or rename it.
      2. Name things with plain words that read like the question the thing answers.
      3. List structure only, telegraph level of conciseness.
      4. Never name anything using single letters, descriptive names are preferred.
13. Think with the user first on design choices. Agreed work, just do.
   1. Never change anything before the user accepts first.
      1. The user prefers to think out loud with the agent, then act later.
      2. Do not rush, feeling rushed means slow down, not speed up.
14. Don't ask the user to commit. The user says when.
15. After every change, give the numbers: right, wrong, gained, lost.
16. Be **ultra concise**.
   1. Compress toward principles.
   2. Never bind to the particular, shortness follows from generality.
   3. Be intensional, not extensional, state the property, negations close the world the property leaves open.
   4. Intensional applies to solutions too, prefer what holds on the whole domain over what holds on a subset.
17. Write standard modern C++20.
   1. Do not assume older standards.
   2. Follow standard C++ library style, use namespaces.
   3. Avoid non-standard headers if a standard alternative exists.
   4. Write text files with LF: binary output mode in C++, newline="\n" in Python.
18. Treat cleanliness as production-class on this year-long project.
   1. Prefer the debt-free path.
   2. Clean up after adding anything, no obsolete files, no root clutter, tools in dedicated directories.
19. Less code is better than more code.
   1. Prefer removal to addition, the best diff is a deletion.
   2. Every line must earn its keep, code that exists must justify itself.
20. Never launch the app until the user accepts.
21. Keep the console silent, all diagnostics and metrics go to files in `logs/`.
22. Keep the build warning-free, give the compiler what it asks for.
23. Comments are for the person reading the code, never for the agent. Keep only the comments that person needs to understand the code.
   1. Placed above the line, never trailing.
   2. Prose, design stories, and experiment records live in `notes/`, in the note whose topic they belong to, not in code comments.
   3. A comment the reader does not need is bloat, read by every person and every agent who opens the file, every time.
24. Do not steer the conversation.
   1. Answer what was asked, then stop, the user sets the pace and the topic.
   2. Match the user's granularity, one small step per exchange.
   3. If the agent wants to offer a next step, offer only one, the next in line, importance as the tiebreaker.
   4. Do not use the AskUserQuestion tool, ask in plain text.
   5. When the user rejects something, ask what failed, then propose.
   6. Never tell the user to stop, pause, take a break, or come back later, when to stop is the user's call alone.
25. Truth first and foremost, the agent defaults truth-last, fluent before checked. It must not.
26. Verify before asserting, flag guesses as guesses.
27. If the agent is not sure, it should say so.
28. Do not assume what is seen is all there is (WYSIATI).
   1. Read files whole. A partial read is a guess, and the part not read is where the answer usually is.
   2. Size is the only reason to read part of one, and it is the file that is big, never the folder it came from. Look before assuming which. Say so when it happens.
   3. The same applies to a change: read every caller of what was touched, not the change alone.
29. Name the cause, not the symptom. A short principle beats a long checklist.
30. A prompt costs the user effort, effort is spent because the obvious already failed, so what reaches the agent is never trivial.
31. Prefer solutions that cannot fail, over solutions that usually work.
32. Theories die in the logs, not in debate. When theory and log disagree, instrument, then decide.
   1. When the answer is known, search for it in the evidence, its absence is the loudest signal.
33. Machinery that never pays is debt, delete it once the evidence votes, un-building is progress.
34. A trigger should be a verdict, not a timer. Prefer the system's own events over thresholds and magic numbers.
35. Never undo an experiment before the user has seen its results, the verdict is the user's, never the agent's.
36. Progressing is not converging, watch for the asymptotic crawl.
37. When a mechanism fights the problem, ask what a human does, then implement that order.
38. Check the code first, always.
39. Be very careful and precise in what the agent says, the user does NOT miss a word.
40. Compare two runs with `scripts\readouts.py`: `solved <before> <after>` (lost and gained), `explained <before> <after>` (readings that claim every cell while their program does not fit the training pairs; must not grow), `missing <label>` (which missing words block the unsolved puzzles).
41. A change is good when no solved puzzle is lost and no new "explained but unsolved" appears. New wrong answers are allowed (the user, 2026-10-11).
42. The rules above are a result of the user being lazy-smart.
# 3. Command Port
1. The GUI app listens on `127.0.0.1:<port>`, one TCP connection per command: connect, send one line, read one reply line (`ok ...` or `error ...`), app closes.
   1. The first app takes 2026 and each later one the next free port up, so two apps never fight, `--port N` says where to start looking.
   2. Which port an app got is the `listening on` line in its own receipt log.
   3. ONE APP PER FOLDER. The port is the only thing an app keeps to itself, everything it writes is per folder, so a second app in the same folder overwrites the first: `solutions/<id>.txt`, `logs/<id>.txt`, `logs/inference/<id>.txt`, `settings/settings.txt`, `kaggle/working/submission.json`. Solutions are the ground truth record, so this loses the most valuable thing here.
   4. The port walk is therefore for apps in DIFFERENT worktrees, not for two in one.
2. Vocabulary: every `ui_action` name (spellings in `include/gui/22_global_app_state.hpp`), plus `state` (read-only status), `puzzle <id>` (jump by id) and `screenshot <path>` (write the frame as a PNG).
   1. The frame is drawn into memory, never off the screen, so it answers the same whether the window is minimized, hidden or covered.
   2. It is the only way the agent can SEE the app, everything else on this port is the agent acting blind.
3. Look before acting: `state` returns `id idx paused benchmarking solved elapsed`.
4. Puzzle jumps land asynchronously, poll `state` until the id changes.
5. Unpausing (`toggle_pause`, SPACE) reads the puzzle and runs what the reading spells; it pauses itself when the reading is done.
6. "Connection refused" means the app is not running, start `out\x64-Release\ARC.exe` with repo root as working directory, pass `--minimized` so the window starts minimized and never steals the user's screen, the agent drives blind through the port anyway.
   1. Minimize is sticky, it saves to `settings/settings.txt` and restores next launch, the user un-minimizes by clicking the taskbar icon.
   2. Copy `settings\settings.txt` first and put it back after: the app saves where it was left.
   3. Clicks posted to its window must come from a DPI-aware process, since the window is per-monitor aware.
7. Every command and outcome is receipt-logged app-clock timestamped in `logs/commands_<port>.txt`, the trustworthy record, one file per app, cleared on each app launch.
   1. A log ending in `stopped` means the app closed cleanly, a log that simply ends means it was killed or crashed, end task cannot announce itself so its absence is the tell.
8. The command port is GUI build only, the headless Kaggle build has none.
9. The receipt log opens with which binary is answering and whether a sibling app got 2026 first.
# 4. Worktrees
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
   3. Worth taking seriously: this caught the agent twice in one session, both times while testing a guard it had just written, and the second time it pushed to Kaggle because the guard it was testing was not there yet.
5. Already per folder, nothing to arrange: `out/`, `logs/`, `settings/`, the puzzle data, the build lock, the backup pool.
6. Ports are machine-wide, so apps in different worktrees walk up from 2026.
7. Every folder pushes to one Kaggle dataset and one notebook (`solver-source-code`, `arc-build`), the submission and the scored runs alike, through `scripts\kaggle_push.py`, never `kaggle kernels push` by hand. A run builds whatever was pushed last, each version's message the commit it came from.
8. Share work by committing, never by reaching into the other folder.
9. The worktree session merges main into its own branch FIRST and resolves conflicts there, whoever caused the divergence resolves it, then the main session's merge is trivial.
