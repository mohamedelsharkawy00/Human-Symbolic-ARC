"""Sync the headless source into a staging dir (kaggle/input/<slug>/, mirroring
the Kaggle mount path) and push it to Kaggle as a new dataset version, then wait
until it's processed (ready).

Why this exists: a source dataset (the submission's is
mohamedelsharkawy0/solver-source-code) is a staging COPY of the headless-relevant
files. Copying by hand is error-prone -- a forgotten header means Kaggle silently
compiles stale code. This script copies a FIXED, complete file set every time so
that can't happen.

The staging dir is 100% generated (incl. dataset-metadata.json) -- the entire
folder can be deleted safely; sync() rebuilds it from scratch.

Pushing the source does NOT run the build kernel -- you control that (validation
vs full run).

Usage:
    python scripts/kaggle_push.py -m "commit message"   the submission's source
    python scripts/kaggle_push.py --kernel              re-pin the submission notebook
    python scripts/kaggle_push.py --smoke               its private 3-task check
    python scripts/kaggle_push.py --run train [id...]   a scored run on Kaggle's GPU
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "mohamedelsharkawy0/solver-source-code"
# A scored run's own source and notebook. The submission never reads them, so
# any folder may push them.
RUN_DATASET = "mohamedelsharkawy0/arc-run-source"
# Every worktree pushes to this one dataset, so only one of them is allowed to.
# Pick the tree that owns submissions; the others stop here instead of silently
# overwriting an upload. One submission a day makes racing pointless anyway.
OWNER_TREE = "ARC"
# Remembers the version we last pushed FROM HERE, so a push that would land on
# top of someone else's is refused rather than discovered later.
PUSH_STAMP = os.path.join(ROOT, "kaggle", "input", ".last_push.json")

# Staging dir mirrors the REAL Kaggle mount path (verified by the build kernel's
# filesystem probe): datasets mount FLAT at /kaggle/input/<slug>/ -- no datasets/
# subdir, no <owner> in the path. (The handoff's "datasets/<owner>/<slug>" was wrong.)
def staging(dataset):
    return os.path.join(ROOT, "kaggle", "input", dataset.split("/")[1])

# The complete set of files the Kaggle build needs: its two mains (the
# submission's and the scored run's), the CUDA kernel, the one header arc_said
# keeps beside itself, and every header in include/core. The folder IS the list
# -- the solver is built from include/core alone, locally too (the arc_headless
# target), so a header it needs cannot be missing from it, and none of
# include/gui is sent.
CORE = os.path.join(ROOT, "include", "core")
FOLDERS = {
    "src": (os.path.join(ROOT, "src"), ["kaggle_main.cpp", "cuda_solver.cu"]),
    "scripts": (os.path.join(ROOT, "scripts"), ["said.cpp", "in_parallel.hpp"]),
    "include": (CORE, sorted(f for f in os.listdir(CORE) if os.path.isfile(os.path.join(CORE, f)))),
}

def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout.strip()

# Which commit the upload is made from; the notebook prints it and keeps it.
def commit():
    head = git("rev-parse", "--short", "HEAD")
    return head + " with uncommitted changes" if git("status", "--porcelain") else head

def sync(dataset):
    pkg = staging(dataset)
    for name, (origin, files) in FOLDERS.items():
        folder = os.path.join(pkg, name)
        os.makedirs(folder, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(origin, f), os.path.join(folder, f))
        # Anything else in there is left from an older push (a header since
        # moved to include/gui or renamed, an old folder) and would be uploaded
        # with the rest. Named by what it is rather than one by one, so the
        # next file to go away needs no edit here.
        for entry in os.listdir(folder):
            stale = os.path.join(folder, entry)
            if os.path.isdir(stale):
                shutil.rmtree(stale)
            elif entry not in files:
                os.remove(stale)
    # Kaggle requires dataset-metadata.json at the root of the uploaded folder
    # (the CLI reads it to know WHICH dataset to push to). We own it here so the
    # staging dir is 100% generated.
    with open(os.path.join(pkg, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": dataset.split("/")[1].replace("-", " ").title(), "id": dataset,
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    with open(os.path.join(pkg, "commit.txt"), "w") as fh:
        fh.write(commit() + "\n")
    print(f"synced {sum(len(files) for _, files in FOLDERS.values())} files -> {os.path.relpath(pkg, ROOT)}")

def exists(dataset):
    return subprocess.run(["kaggle", "datasets", "status", dataset], capture_output=True).returncode == 0

def remote_version():
    """Newest version number Kaggle reports for the dataset, or None."""
    r = subprocess.run(["kaggle", "datasets", "status", DATASET],
                       capture_output=True, text=True)
    return (r.stdout or "").strip() or None

def check_owner():
    """Only the owning tree may push, and only if nobody pushed since we did."""
    here = os.path.basename(ROOT)
    if here != OWNER_TREE:
        sys.exit(f"refusing: pushes come from '{OWNER_TREE}', this is '{here}'. "
                 f"Every worktree shares one dataset, so pushing here would "
                 f"overwrite whatever '{OWNER_TREE}' last sent.")
    try:
        with open(PUSH_STAMP) as fh:
            seen = json.load(fh).get("status")
    except (OSError, ValueError):
        return
    now = remote_version()
    if now and seen and now != seen:
        sys.exit("refusing: the dataset moved since this tree last pushed "
                 "(someone or something else uploaded). Re-check before pushing, "
                 f"then delete {os.path.relpath(PUSH_STAMP, ROOT)} to proceed.")

def record_push():
    os.makedirs(os.path.dirname(PUSH_STAMP), exist_ok=True)
    with open(PUSH_STAMP, "w") as fh:
        json.dump({"status": remote_version(), "at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

# A variant is the notebook under a name of its own, told what to do by the
# environment it starts with: (name, title, environment, dataset, timeout).
SMOKE = ("arc-smoke-check", "ARC Smoke Check", {"ARC_SMOKE_CHECK": "1"}, DATASET, 1200)

def run_variant(which, ids):
    return ("arc-run", "ARC Run", {"ARC_DATASET": which, "ARC_RUN_IDS": " ".join(ids)}, RUN_DATASET, None)

def push_kernel(variant=None):
    """Re-pin the notebook to the newest dataset version, or start a variant.

    The raw `kaggle kernels push` has no guard of its own, and every worktree
    shares one kernel, so going through here is what makes the rule real.
    """
    kernel_dir = os.path.join(ROOT, "kaggle", "src", "arc_build")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_to_ipynb.py")], check=True)
    timeout = None
    if variant:
        name, title, environment, dataset, timeout = variant
        with open(os.path.join(kernel_dir, "kernel-metadata.json")) as fh:
            meta = json.load(fh)
        with open(os.path.join(kernel_dir, meta["code_file"]), encoding="utf-8") as fh:
            notebook = json.load(fh)
        notebook["cells"][0]["source"].insert(0, f"import os; os.environ.update({environment!r})\n")
        meta["id"] = meta["id"].rsplit("/", 1)[0] + "/" + name
        meta["title"] = title
        meta["is_private"] = True
        meta["dataset_sources"] = [dataset]
        kernel_dir = os.path.join(ROOT, "out", name)
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        with open(os.path.join(kernel_dir, meta["code_file"]), "w", encoding="utf-8") as fh:
            json.dump(notebook, fh, indent=1)
    command = ["kaggle", "kernels", "push", "-p", kernel_dir]
    if timeout:
        command += ["--timeout", str(timeout)]
    r = subprocess.run(command,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()); sys.exit(1)
    print(f"{variant[1]} started." if variant else "kernel re-pinned to the latest dataset version.")

def push(dataset, msg):
    # A dataset not there yet, or deleted, is made anew, private.
    command = (["kaggle", "datasets", "version", "-m", msg] if exists(dataset)
               else ["kaggle", "datasets", "create"])
    r = subprocess.run(command + ["-p", staging(dataset), "--dir-mode", "zip"],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()); sys.exit(1)

def wait_ready(dataset, timeout_s=180):
    print("waiting for dataset to be ready...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run(["kaggle", "datasets", "status", dataset],
                           capture_output=True, text=True)
        status = (r.stdout or "").strip().lower()
        if "ready" in status:
            print(" ready.")
            return True
        print(".", end="", flush=True)
        time.sleep(5)
    print(" TIMEOUT (still processing -- check before running the kernel).")
    return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="update source", help="dataset version message")
    ap.add_argument("--kernel", action="store_true",
                    help="re-pin the notebook instead of pushing the source")
    ap.add_argument("--smoke", action="store_true",
                    help="run a separate private 3-task notebook on the uploaded source")
    ap.add_argument("--run", nargs="+", metavar=("SET", "ID"),
                    help="a scored run on Kaggle's GPU of train or eval, every puzzle or the ids named")
    args = ap.parse_args()
    if args.run:
        which, ids = args.run[0], args.run[1:]
        if which not in ("train", "eval"):
            sys.exit("--run reads train or eval, the sets whose answers are known")
        sync(RUN_DATASET)
        push(RUN_DATASET, commit())
        if not wait_ready(RUN_DATASET):
            sys.exit(1)
        push_kernel(run_variant(which, ids))
        print("When it is done, into out/<label> for scripts/readouts.py:")
        print("    kaggle kernels output mohamedelsharkawy0/arc-run -p out/<label>")
        print("    tar -xzf out/<label>/said.tar.gz -C out/<label>")
        sys.exit(0)
    check_owner()
    if args.kernel or args.smoke:
        push_kernel(SMOKE if args.smoke else None)
        sys.exit(0)
    sync(DATASET)
    push(DATASET, args.message)
    wait_ready(DATASET)
    record_push()
    print("done.")
    print()
    print("NEXT: re-pin the notebook to THIS dataset version, or it builds stale code:")
    print("    python scripts/kaggle_push.py --kernel")
    print("  (go through this script, not `kaggle kernels push` directly. Every")
    print("   worktree shares one kernel and the raw command has no guard, so a")
    print("   push from elsewhere silently overwrites whatever was pinned last.)")
    print("  (the push auto-pins latest -> the browser won't prompt 'update dataset")
    print("   version'. Then open arc-build in the browser, set accelerator = L4x4,")
    print("   Save & Run All. Skipping the push means you must click 'Update' in the GUI.)")
