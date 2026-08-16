Strictly follow every line below exactly, always, no exceptions, ever.

Read `README.md` first, it is the home for everything about the project itself.

# 1. Context
1. Final submission deadline Nov 2, 2026, results Dec 4, 2026.

# 2. Standards
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
8. Build only via `scripts\build.bat x64-Release NOLAUNCH`, never any other way, and never launch the app until the user accepts.
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

# 3. Command Port
1. The GUI app listens on `127.0.0.1:<port>`, one TCP connection per command: connect, send one line, read one reply line (`ok ...` or `error ...`), app closes.
   1. The first app takes 2026 and each later one the next free port up, so two apps never fight, `--port N` says where to start looking.
   2. Which port an app got is the `listening on` line in its own receipt log.
2. Vocabulary: every `ui_action` name (spellings in `include/global_app_state.hpp`), plus `state` (read-only status) and `puzzle <id>` (jump by id).
3. Look before acting: `state` returns `id idx paused benchmarking solved elapsed`.
4. Puzzle jumps land asynchronously, poll `state` until the id changes.
5. The solver runs only while unpaused (`toggle_pause`), and auto-pauses on solve.
6. "Connection refused" means the app is not running, start `out\x64-Release\ARC.exe` with repo root as working directory, pass `--minimized` so the window starts minimized and never steals the user's screen, the agent drives blind through the port anyway.
   1. Minimize is sticky, it saves to `settings/settings.txt` and restores next launch, the user un-minimizes by clicking the taskbar icon.
7. Every command and outcome is receipt-logged app-clock timestamped in `logs/commands_<port>.txt`, the trustworthy record, one file per app, cleared on each app launch.
8. The command port is GUI build only, the headless Kaggle build has none.
9. `state` ends with `gpu_used=<percent>`, and the receipt log opens with how much of the card is already taken plus whether a sibling app got 2026 first.
   1. Two idle apps sit at about 17% of 6140 MB, that is the baseline, well above it means the card is shared with something.
   2. A shared run is not comparable to one that had the card alone, say so rather than reading it as a regression.

# 4. Worktrees
1. Look before settling in, nothing announces another session, so ask the two things that already know.
   1. `git worktree list` names every folder open on this repo, more than one means someone else is set up.
   2. Send `state` to `127.0.0.1:2026`, an `ok ...` line back means an app is already running, so a session is live even if only one folder exists, and "connection refused" means none is.
   3. `git status` in a folder someone else is using shows their uncommitted work, that folder is taken.
2. If either check says the main tree is busy, work in a worktree instead, `git worktree add -b <branch> ..\ARC-<name>`, sharing one history, and `git worktree remove` throws it away.
3. Edits do NOT cross between worktrees until they are committed, a worktree only ever sees committed code.
   1. This is what makes parallel sessions safe, and it is the thing that looks like a bug when a fix seems not to apply.
4. Everything mutable is already per folder: `out/`, `logs/`, `settings/`, the puzzle data, the build lock, and the backup pool.
5. The first build in a new worktree fetches its own puzzles, so nothing needs copying or linking by hand.
6. Ports are machine-wide, not per folder, so apps from different worktrees still walk up from 2026.
7. Only the `ARC` tree may push to Kaggle, every worktree shares one dataset slug and `scripts\kaggle_push.py` refuses from anywhere else.
8. Share work between worktrees by committing, never by reaching into the other folder.

# 5. Thinking
1. Truth first and foremost, Claude defaults truth-last, fluent before checked. It must not.
2. Verify before asserting, flag guesses as guesses.
3. If Claude is not sure, it should say so.
4. Do not assume what is seen is all there is (WYSIATI).
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
