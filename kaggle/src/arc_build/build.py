# ARC headless solver: build + run on Kaggle, produce submission.json.
# (Authored as a .py for clarity; converted to notebook cells for the kernel.)
import os, json, glob, subprocess, sys, time

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
src_cands = glob.glob("/kaggle/input/**/src/main.cpp", recursive=True)
assert src_cands, "could not find src/main.cpp under /kaggle/input"
SRC_ROOT = src_cands[0].rsplit("/src/", 1)[0]
INC = SRC_ROOT + "/include"
TP  = INC + "/third_party"
print("source root:", SRC_ROOT)

# This notebook is the SUBMISSION path: always solve the competition test set and
# produce submission.json. (Scored eval runs are done locally in the Windows GUI,
# which loads arc-agi_evaluation_*.json directly -- no eval branch needed here.)
comp_cands = glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)
assert comp_cands, "could not find arc-agi_test_challenges.json"
TEST_JSON = comp_cands[0]
print("TEST MODE -- challenges:", TEST_JSON)

# --- 1. point the solver at the big challenges JSON ------------------------
# No on-disk split anymore: the C++ solver reads TEST_JSON as-is and slices it
# in-memory (load_challenges_json). We just pass the path via ARC_CHALLENGES_JSON
# below. (challenges is still loaded here for the verify step at the end.)
challenges = json.load(open(TEST_JSON))
print(f"{len(challenges)} tasks -> {TEST_JSON} (sliced in-memory by the solver)")

# --- 2. compile -----------------------------------------------------------
# Target sm_89 (NVIDIA L4, Ada) -- the submission accelerator (GPU L4 x4). Embed
# PTX too (compute_89) so the binary still JIT-loads if a run lands on a NEWER
# arch when L4 is supply-constrained. (If it ever lands on an OLDER card, add a
# lower -gencode, e.g. arch=compute_75,code=sm_75.)
BIN = "/kaggle/working/arc_solver"
cmd = [
    "nvcc", "-std=c++20",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_89,code=compute_89",
    "-O2",
    "-DFMT_HEADER_ONLY",   # fmtlib header-only: pulls in vformat definition (no separate .cc to link)
    "-I", INC, "-I", TP,
    SRC_ROOT + "/src/main.cpp", SRC_ROOT + "/src/cuda_solver.cu",
    "-o", BIN,
]
print("compiling:\n  " + " ".join(cmd))
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"compile took {time.time()-t0:.1f}s, exit={r.returncode}")
if r.returncode != 0:
    print("=== STDERR ===")
    print(r.stderr[:8000])
    sys.exit(1)
print("compiled OK ->", BIN)

# --- 3. run: one shard PER VISIBLE GPU -------------------------------------
# Launch one process per physical GPU, pinned via CUDA_VISIBLE_DEVICES, each
# solving a contiguous shard and writing submission_<shard>.json.
#
# SHARD COUNT = number of GPUs ACTUALLY PRESENT (nvidia-smi -L), not a hardcoded
# 4. This is load-bearing: hardcoding 4 on a 2-GPU box (the default T4x2, which
# the accelerator dropdown silently falls back to) launches 2 extra shards with
# no GPU -- they crash on cudaMallocHost (pinned-memory exhaustion) and SILENTLY
# drop 120 of 240 tasks. Sizing to the real GPU count means the run always covers
# all 240 tasks on whatever hardware we land on (L4x4 -> 4 fast shards; T4x2 -> 2
# slower shards, still complete).
def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        n = sum(1 for line in out.stdout.splitlines() if line.strip().startswith("GPU "))
        return max(1, n)
    except Exception:
        return 1

os.chdir("/kaggle/working")
# QUICK_CHECK: solve only the first 10 tasks on a single shard to eyeball the
# solve rate fast (still writes a real submission.json, just 10 entries). The
# solver prints a "SOLVE SUMMARY" to its shard log at the end. Set False for the
# full submission run. TIMEOUT stays at the submission value so the sampled rate
# reflects real submission-time behaviour.
QUICK_CHECK = False
NGPU = gpu_count()
SHARDS = 1 if QUICK_CHECK else NGPU
PUZZLE_LIMIT = "10" if QUICK_CHECK else "0"   # 0 = FULL RUN: all 240 tasks
TIMEOUT = "350"       # 350s/puzzle. Budget target ~6h on L4x4: 240/4=60 tasks/shard
                      # x 350s = ~5.8h worst case (+compile/IO). ASSUMES L4x4 (4
                      # shards) -- on a T4x2 fallback it's 120x350=~11.7h (under the
                      # 12h kill but blows the 6h target), so confirm L4x4 in the GUI.
print(f"detected {NGPU} GPU(s) -> launching {SHARDS} shards, "
      f"PUZZLE_LIMIT={PUZZLE_LIMIT} TIMEOUT={TIMEOUT} (QUICK_CHECK={QUICK_CHECK})")

procs = []
t0 = time.time()
for s in range(SHARDS):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(s)   # pin this process to physical GPU s
    env["ARC_SHARD_INDEX"] = str(s)
    env["ARC_SHARD_TOTAL"] = str(SHARDS)
    env["ARC_PUZZLE_LIMIT"] = PUZZLE_LIMIT  # overall task cap (shard takes its slice)
    env["ARC_TIMEOUT"] = TIMEOUT
    env["ARC_CHALLENGES_JSON"] = TEST_JSON   # solver slices this big JSON in-memory
    lf = open(f"/kaggle/working/shard_{s}.log", "w")
    procs.append((s, subprocess.Popen([BIN], stdout=lf, stderr=subprocess.STDOUT, env=env), lf))

for s, p, lf in procs:
    p.wait(); lf.close()
    print(f"shard {s} exit={p.returncode}")
print(f"all shards took {time.time()-t0:.1f}s")
for s, _, _ in procs:
    print(f"--- shard_{s}.log tail ---")
    print(open(f"/kaggle/working/shard_{s}.log").read()[-500:])

# --- 4. merge shard outputs -> submission.json ----------------------------
# CROSS-FILE CONTRACT: the "submission_<n>.json" name below is produced by the C++
# solver (src/main.cpp, the shard_index write path). If you rename it in one place
# you MUST rename it in the other, or this merge silently finds nothing.
# (The Windows build never shards -- it writes a single submission.json directly.)
# NAMING CONTRACT (src/main.cpp): with >1 shard each writes submission_<n>.json;
# with a SINGLE shard it writes submission.json directly (no suffix). The merge
# must read whatever main.cpp wrote, or it silently produces an empty file.
if SHARDS > 1:
    merged = {}
    for s in range(SHARDS):
        f = f"/kaggle/working/submission_{s}.json"
        if os.path.exists(f):
            part = json.load(open(f))
            dupes = set(merged) & set(part)
            if dupes: print(f"WARNING shard {s} dupes:", list(dupes)[:5])
            merged.update(part)
        else:
            print(f"WARNING: {f} missing (shard {s} produced no output)")
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
got = set(sub.keys())
missing = expected - got
print("got tasks:", len(got), "| expected:", len(expected))
print("MISSING task_ids:", len(missing), list(missing)[:10])
print("extra (not in challenges):", len(got - expected), list(got - expected)[:5])
print("extra (not in challenges):", len(got - expected), list(got - expected)[:5])

# Validate every grid: rectangular + all cells int 0-9 (else "Scoring Error").
bad_cell = bad_shape = 0
for tid, preds in sub.items():
    for entry in preds:
        for key in ("attempt_1", "attempt_2"):
            g = entry.get(key)
            if not g or not isinstance(g, list) or not all(isinstance(r, list) for r in g):
                bad_shape += 1; continue
            wid = len(g[0]) if g else 0
            for row in g:
                if len(row) != wid: bad_shape += 1; break
            for row in g:
                for v in row:
                    if not isinstance(v, int) or v < 0 or v > 9:
                        bad_cell += 1; break
print(f"VALIDATION: bad cells (not 0-9): {bad_cell} | bad shapes (ragged/empty): {bad_shape}")

if got:
    k = next(iter(sub)); print("sample entry:", k, "->", json.dumps(sub[k])[:120])
