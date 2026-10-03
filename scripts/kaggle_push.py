"""Sync the headless source into the staging dir (kaggle/input/datasets/<owner>/
<slug>/, mirroring the Kaggle mount path) and push it to Kaggle as a new dataset
version, then wait until it's processed (ready).

Why this exists: the source dataset (mohamedelsharkawy0/solver-source-code) is a
staging COPY of the headless-relevant files. Copying by hand is error-prone -- a
forgotten header means Kaggle silently compiles stale code. This script copies a
FIXED, complete file set every time so that can't happen.

The staging dir is 100% generated (incl. dataset-metadata.json) -- the entire
folder can be deleted safely; sync() rebuilds it from scratch.

Does NOT run the build kernel -- you control that (validation vs full run).

Usage:
    python scripts/kaggle_push.py -m "commit message"
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "mohamedelsharkawy0/solver-source-code"
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
PKG  = os.path.join(ROOT, "kaggle", "input", DATASET.split("/")[1])

# Kaggle requires dataset-metadata.json at the root of the uploaded folder (the
# CLI reads it to know WHICH dataset to push to). We own it here so the staging
# dir is 100% generated -- the whole folder can be deleted and this script rebuilds it.
DATASET_METADATA = {
    "title": "Solver Source Code",
    "id": DATASET,
    "licenses": [{"name": "CC0-1.0"}],
}

# The complete set of files the Kaggle build needs: its main, the CUDA kernel,
# and every header in include/core. The folder IS the list -- the solver is
# built from include/core alone, locally too (the arc_headless target), so a
# header it needs cannot be missing from it, and none of include/gui is sent.
SRC_FILES = ["kaggle_main.cpp", "cuda_solver.cu"]
CORE = os.path.join(ROOT, "include", "core")
INC_FILES = sorted(f for f in os.listdir(CORE) if os.path.isfile(os.path.join(CORE, f)))

def sync():
    os.makedirs(PKG, exist_ok=True)
    os.makedirs(os.path.join(PKG, "src"), exist_ok=True)
    os.makedirs(os.path.join(PKG, "include"), exist_ok=True)
    # dataset identity manifest (Kaggle CLI looks for it at the upload-folder root)
    with open(os.path.join(PKG, "dataset-metadata.json"), "w") as fh:
        json.dump(DATASET_METADATA, fh, indent=2)
    for f in SRC_FILES:
        shutil.copy2(os.path.join(ROOT, "src", f), os.path.join(PKG, "src", f))
    for f in INC_FILES:
        dst = os.path.join(PKG, "include", f)
        shutil.copy2(os.path.join(CORE, f), dst)
    # Every header is copied flat, so any folder still sitting in there is left
    # over from an older push (third_party, inference) and would be uploaded
    # with the rest. Named by what it is rather than one by one, so the next
    # folder to go away needs no edit here.
    inc = os.path.join(PKG, "include")
    for entry in os.listdir(inc):
        stale = os.path.join(inc, entry)
        if os.path.isdir(stale):
            shutil.rmtree(stale)
        elif entry not in INC_FILES:
            # Left from an older push: a header since moved to include/gui or
            # renamed. Uploaded, it would be dead weight at best.
            os.remove(stale)
    src = os.path.join(PKG, "src")
    for entry in os.listdir(src):
        if entry not in SRC_FILES:
            os.remove(os.path.join(src, entry))
    print(f"synced {len(SRC_FILES)} src + {len(INC_FILES)} headers -> {os.path.relpath(PKG, ROOT)}")

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

def push_kernel(smoke=False):
    """Re-pin the notebook to the newest dataset version, guarded the same way.

    The raw `kaggle kernels push` has no guard of its own, and every worktree
    shares one kernel, so going through here is what makes the rule real.
    """
    kernel_dir = os.path.join(ROOT, "kaggle", "src", "arc_build")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_to_ipynb.py")], check=True)
    if smoke:
        with open(os.path.join(kernel_dir, "kernel-metadata.json")) as fh:
            meta = json.load(fh)
        with open(os.path.join(kernel_dir, meta["code_file"]), encoding="utf-8") as fh:
            notebook = json.load(fh)
        notebook["cells"][0]["source"].insert(0, 'import os; os.environ["ARC_SMOKE_CHECK"] = "1"\n')
        meta["id"] = meta["id"].rsplit("/", 1)[0] + "/arc-smoke-check"
        meta["title"] = "ARC Smoke Check"
        meta["is_private"] = True
        kernel_dir = os.path.join(ROOT, "out", "kaggle-smoke")
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        with open(os.path.join(kernel_dir, meta["code_file"]), "w", encoding="utf-8") as fh:
            json.dump(notebook, fh, indent=1)
    command = ["kaggle", "kernels", "push", "-p", kernel_dir]
    if smoke:
        command += ["--timeout", "1200"]
    r = subprocess.run(command,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()); sys.exit(1)
    print("private smoke check started." if smoke else "kernel re-pinned to the latest dataset version.")

def push(msg):
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", PKG, "-m", msg, "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()); sys.exit(1)

def wait_ready(timeout_s=180):
    print("waiting for dataset to be ready...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run(["kaggle", "datasets", "status", DATASET],
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
    args = ap.parse_args()
    check_owner()
    if args.kernel or args.smoke:
        push_kernel(smoke=args.smoke)
        sys.exit(0)
    sync()
    push(args.message)
    wait_ready()
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
