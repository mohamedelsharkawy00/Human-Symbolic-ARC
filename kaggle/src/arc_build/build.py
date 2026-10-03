# ARC headless solver: build + run on Kaggle, produce submission.json.
# (Authored as a .py for clarity; converted to notebook cells for the kernel.)
import os, json, glob, subprocess, sys, time

# A separate private notebook sets this for a short end-to-end check.
# The submission notebook always leaves it off.
QUICK_CHECK = os.environ.get("ARC_SMOKE_CHECK") == "1"

# --- 0. FILESYSTEM PROBE (diagnostic; off by default) ---------------------
# Make the runtime layout visible. VERIFIED by this probe: datasets mount FLAT at
# /kaggle/input/<slug>/ (NO datasets/<owner> nesting), competitions at
# /kaggle/input/competitions/<slug>/. The web UI hides these paths.
# IMPORTANT: Kaggle keeps only a ROLLING WINDOW of stdout -- a full '/' walk
# (tens of thousands of system files) floods it and EVICTS the part you care
# about. So: stdout gets ONLY /kaggle (small, always survives); the full
# everything-from-/ tree is written to a FILE in /kaggle/working/ (lands in the
# Output tab, downloadable, NOT subject to log eviction). /proc,/sys,/dev are
# pruned (virtual kernel fs, not real files, would explode/hang).
PROBE = False          # set True to dump the filesystem layout at startup
PROBE_ONLY = False     # set True to probe then EXIT (fast run -> saves fs_tree.txt
                       # to the Output tab without waiting on the full solve)
def _tree(root, out, max_lines=2000000, max_depth=99, prune=()):
    root = root.rstrip("/") or "/"
    n = 0
    for dp, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if os.path.join(dp, d) not in prune)
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        ind = "  " * depth
        print(f"{ind}{os.path.basename(dp) or dp}/", file=out)
        n += 1
        for f in sorted(files):
            try: sz = os.path.getsize(os.path.join(dp, f))
            except OSError: sz = -1
            print(f"{ind}  {f}  ({sz} B)", file=out)
            n += 1
            if n >= max_lines:
                print(f"  ... [capped at {max_lines} lines]", file=out); return
def _probe():
    print("="*60); print("FILESYSTEM PROBE")
    print("cwd:", os.getcwd())
    print("whoami:", subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip())
    print("KAGGLE_KERNEL_RUN_TYPE:", os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    print("\n--- NOTEBOOK ROOT (cwd, before any chdir) ---")
    _tree(os.getcwd(), sys.stdout)          # where the kernel actually runs from
    print("\n--- /kaggle (FULL: input mounts + working dir) ---")
    _tree("/kaggle", sys.stdout)            # small -> safe to stdout
    # Everything from / -> durable file (survives log eviction; grab from Output tab).
    full = "/kaggle/working/fs_tree.txt"
    os.makedirs("/kaggle/working", exist_ok=True)
    with open(full, "w") as fh:
        _tree("/", fh, prune={"/proc", "/sys", "/dev"})
    print(f"\n--- full '/' tree written to {full} "
          f"({os.path.getsize(full)} B) -- download from the Output tab ---")
    print("="*60, "\n", sep="")
if PROBE or PROBE_ONLY:
    _probe()
    if PROBE_ONLY:
        print("PROBE_ONLY: exiting before build/solve."); sys.exit(0)

# --- locate inputs --------------------------------------------------------
# Source dataset (auto-discover via glob -- mount path varies: seen flat at
# /kaggle/input/<slug>/ and nested at /kaggle/input/datasets/<owner>/<slug>/)
src_cands = glob.glob("/kaggle/input/**/src/kaggle_main.cpp", recursive=True)
assert src_cands, "could not find src/kaggle_main.cpp under /kaggle/input"
SRC_ROOT = src_cands[0].rsplit("/src/", 1)[0]
INC = SRC_ROOT + "/include"
print("source root:", SRC_ROOT)

# This notebook is the SUBMISSION path: always solve the competition test set and
# produce submission.json. (Scored eval runs are done locally in the Windows GUI,
# which loads arc-agi_evaluation_*.json directly -- no eval branch needed here.)
comp_cands = glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)
assert comp_cands, "could not find arc-agi_test_challenges.json"
TEST_JSON = comp_cands[0]
print("TEST MODE -- challenges:", TEST_JSON)

# --- 1. read challenges for output validation -----------------------------
# The C++ loader reads the same competition file from its fixed Kaggle path.
challenges = json.load(open(TEST_JSON))
print(f"{len(challenges)} tasks -> {TEST_JSON} (sliced in-memory by the solver)")

# --- 2. compile -----------------------------------------------------------
# Compile for the GPUs actually assigned, including T4 fallback machines.
gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
gpu_arches = [line.strip().replace(".", "") for line in gpu_info.splitlines() if line.strip()]
assert gpu_arches and all(a.isdigit() for a in gpu_arches), gpu_info
print(subprocess.check_output(["nvcc", "--version"], text=True))
BIN = "/kaggle/working/arc_solver"
cmd = [
    "nvcc", "-std=c++20", "--expt-relaxed-constexpr",
    "-O2", "-DARC_ENABLE_CUDA_PIECES=0",
    "-I", INC,
    SRC_ROOT + "/src/kaggle_main.cpp", SRC_ROOT + "/src/cuda_solver.cu",
    "-o", BIN,
]
for arch in sorted(set(gpu_arches)):
    cmd.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
print("compiling:\n  " + " ".join(cmd))
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"compile took {time.time()-t0:.1f}s, exit={r.returncode}")
if r.returncode != 0:
    open("/kaggle/working/compile.log", "w").write(r.stdout + r.stderr)
    print("=== STDERR ===")
    print(r.stderr[:8000])
    sys.exit(1)
print("compiled OK ->", BIN)

# --- 3. run: one shard PER VISIBLE GPU -------------------------------------
# Launch one process per physical GPU, pinned via CUDA_VISIBLE_DEVICES, each
# solving a contiguous shard and writing submission_<shard>.json. Each of those
# solves every puzzle of its shard in a fresh child process of its own (the
# child inherits the GPU pin), so no puzzle runs on memory another left behind.
#
# SHARD COUNT = number of GPUs ACTUALLY PRESENT (nvidia-smi -L), not a hardcoded
# 4. This is load-bearing: hardcoding 4 on a 2-GPU box (the default T4x2, which
# the accelerator dropdown silently falls back to) launches 2 extra shards with
# no GPU -- they crash on cudaMallocHost (pinned-memory exhaustion) and SILENTLY
# drop 120 of 240 tasks. Sizing to the real GPU count means the run always covers
# all 240 tasks on whatever hardware we land on (L4x4 -> 4 fast shards; T4x2 -> 2
# slower shards, still complete).
os.chdir("/kaggle/working")
# The private smoke notebook runs inference on three tasks. It checks
# build, execution and output format; it does not measure solve rate.
NGPU = len(gpu_arches)
SHARDS = 1 if QUICK_CHECK else NGPU
PUZZLE_LIMIT = "3" if QUICK_CHECK else "0"   # 0 = full run
# There is no search budget: each task stops after its emitted programs.
print(f"detected {NGPU} GPU(s) -> launching {SHARDS} shards, "
      f"PUZZLE_LIMIT={PUZZLE_LIMIT} MODE=inference-only (QUICK_CHECK={QUICK_CHECK})")

procs = []
# A rerun must never merge output left by an earlier run.
for stale in glob.glob("/kaggle/working/submission*.json"):
    os.remove(stale)
t0 = time.time()
for s in range(SHARDS):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(s)   # pin this process to physical GPU s
    env["ARC_SHARD_INDEX"] = str(s)
    env["ARC_SHARD_TOTAL"] = str(SHARDS)
    env["ARC_PUZZLE_LIMIT"] = PUZZLE_LIMIT  # overall task cap (shard takes its slice)
    env["ARC_INFERENCE_ONLY"] = "1"  # Emit, replay on all training pairs, predict. No search.
    env["ARC_DATASET"] = "test"   # which set to solve; the solver finds the file itself,
                                  # walking up from /kaggle/working to the /kaggle/input mount
    lf = open(f"/kaggle/working/shard_{s}.log", "w")
    procs.append((s, subprocess.Popen([BIN], stdout=lf, stderr=subprocess.STDOUT, env=env), lf))

for s, p, lf in procs:
    p.wait(); lf.close()
    print(f"shard {s} exit={p.returncode}")
assert all(p.returncode == 0 for _, p, _ in procs), "a solver shard failed; inspect shard logs"
print(f"all shards took {time.time()-t0:.1f}s")
for s, _, _ in procs:
    print(f"--- shard_{s}.log tail ---")
    shard_log = open(f"/kaggle/working/shard_{s}.log").read()
    print(shard_log[-500:])
    assert "its run exited with" not in shard_log and "handed back nothing" not in shard_log, \
        "a puzzle process failed; refusing to submit fallback inputs from a failed run"

# --- 4. merge shard outputs -> submission.json ----------------------------
# CROSS-FILE CONTRACT: the "submission_<n>.json" name below is produced by the C++
# solver (src/kaggle_main.cpp, the shard_index write path). If you rename it in one place
# you MUST rename it in the other, or this merge silently finds nothing.
# (The Windows build never shards -- it writes a single submission.json directly.)
# NAMING CONTRACT (src/kaggle_main.cpp): with >1 shard each writes submission_<n>.json;
# with a SINGLE shard it writes submission.json directly (no suffix). The merge
# must read whatever kaggle_main.cpp wrote, or it silently produces an empty file.
if SHARDS > 1:
    merged = {}
    for s in range(SHARDS):
        f = f"/kaggle/working/submission_{s}.json"
        if os.path.exists(f):
            part = json.load(open(f))
            dupes = set(merged) & set(part)
            assert not dupes, f"duplicate tasks in shard {s}: {sorted(dupes)[:5]}"
            merged.update(part)
        else:
            raise RuntimeError(f"missing shard output: {f}")
    json.dump(merged, open("/kaggle/working/submission.json", "w"))
    print(f"merged -> submission.json: {len(merged)} tasks from {SHARDS} shards")
else:
    # Single shard wrote submission.json directly -- nothing to merge.
    f = "/kaggle/working/submission.json"
    merged = json.load(open(f)) if os.path.exists(f) else {}
    print(f"single shard -> submission.json: {len(merged)} tasks")

# --- 5. verify ------------------------------------------------------------
sub = merged
expected = set(challenges.keys())
if QUICK_CHECK:
    expected = set(sorted(expected)[:int(PUZZLE_LIMIT)])
got = set(sub.keys())
missing = expected - got
print("got tasks:", len(got), "| expected:", len(expected))
print("MISSING task_ids:", len(missing), list(missing)[:10])
print("extra (not in challenges):", len(got - expected), list(got - expected)[:5])
assert got == expected, "submission task IDs do not match the requested tasks"

# Validate every grid: rectangular + all cells int 0-9 (else "Scoring Error").
bad_cell = bad_shape = 0
for tid, preds in sub.items():
    assert isinstance(preds, list) and len(preds) == len(challenges[tid]["test"]), tid
    for entry in preds:
        assert isinstance(entry, dict) and set(entry) == {"attempt_1", "attempt_2"}, tid
        for key in ("attempt_1", "attempt_2"):
            g = entry.get(key)
            if not g or not isinstance(g, list) or not all(isinstance(r, list) for r in g):
                bad_shape += 1; continue
            wid = len(g[0]) if g else 0
            if not (1 <= len(g) <= 30 and 1 <= wid <= 30):
                bad_shape += 1
            for row in g:
                if len(row) != wid: bad_shape += 1; break
            for row in g:
                for v in row:
                    if type(v) is not int or v < 0 or v > 9:
                        bad_cell += 1; break
print(f"VALIDATION: bad cells (not 0-9): {bad_cell} | bad shapes (ragged/empty): {bad_shape}")
assert bad_cell == 0 and bad_shape == 0, "invalid submission grids"
if QUICK_CHECK:
    # Deliberately not a submission artifact: only three tasks were run.
    os.replace("/kaggle/working/submission.json", "/kaggle/working/smoke_predictions.json")
    print("SMOKE CHECK PASSED (3 tasks; not a competition submission)")

if got:
    k = next(iter(sub)); print("sample entry:", k, "->", json.dumps(sub[k])[:120])
