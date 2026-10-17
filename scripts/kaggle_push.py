"""Sync the headless source into the staging dir (kaggle/input/<slug>/, mirroring
the Kaggle mount path) and push it to Kaggle as a new dataset version, then wait
until Kaggle serves it.

Why this exists: the source dataset (mohamedelsharkawy0/solver-source-code) is a
staging COPY of the headless-relevant files. Copying by hand is error-prone -- a
forgotten header means Kaggle silently compiles stale code. This script copies a
FIXED, complete file set every time so that can't happen.

The staging dir is 100% generated (incl. dataset-metadata.json) -- the entire
folder can be deleted safely; sync() rebuilds it from scratch.

ONE DATASET AND ONE NOTEBOOK (arc-build) for everything -- the submission, its
smoke check and the scored runs -- pushed from any folder. A run builds whatever
was pushed last; each version's message is the commit it came from.

Usage:
    python scripts/kaggle_push.py [-m "message"]        push the source
    python scripts/kaggle_push.py --kernel              run the notebook: the submission
    python scripts/kaggle_push.py --smoke               run it on three test tasks
    python scripts/kaggle_push.py --run train [id...]   push the source and run it scored
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "mohamedelsharkawy0/solver-source-code"
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

# Which commit the upload is made from, the version's message by default.
def commit():
    head = git("rev-parse", "--short", "HEAD")
    return head + " with uncommitted changes" if git("status", "--porcelain") else head

def sync():
    for name, (origin, files) in FOLDERS.items():
        folder = os.path.join(PKG, name)
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
    with open(os.path.join(PKG, "dataset-metadata.json"), "w") as fh:
        json.dump(DATASET_METADATA, fh, indent=2)
    print(f"synced {sum(len(files) for _, files in FOLDERS.values())} files -> {os.path.relpath(PKG, ROOT)}")

def push(msg):
    # A dataset not there yet, or deleted, is made anew, private. Asked of the
    # list of our own: a refused question (too many requests) is not "not there".
    mine = subprocess.run(["kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
    if mine.returncode != 0:
        print((mine.stderr or mine.stdout).strip()); sys.exit(1)
    command = (["kaggle", "datasets", "version", "-m", msg] if DATASET in mine.stdout
               else ["kaggle", "datasets", "create"])
    r = subprocess.run(command + ["-p", PKG, "--dir-mode", "zip"], capture_output=True, text=True)
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

def push_kernel(environment=None, timeout=None):
    """Run the notebook on the source pushed last, told what to do by the
    environment it starts with: none is the submission."""
    kernel_dir = os.path.join(ROOT, "kaggle", "src", "arc_build")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_to_ipynb.py")], check=True)
    if environment:
        with open(os.path.join(kernel_dir, "kernel-metadata.json")) as fh:
            meta = json.load(fh)
        with open(os.path.join(kernel_dir, meta["code_file"]), encoding="utf-8") as fh:
            notebook = json.load(fh)
        notebook["cells"][0]["source"].insert(0, f"import os; os.environ.update({environment!r})\n")
        kernel_dir = os.path.join(ROOT, "out", "arc-build")
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
    print(f"arc-build started on the source pushed last{', told ' + repr(environment) if environment else ''}.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", help="dataset version message (the commit by default)")
    ap.add_argument("--kernel", action="store_true",
                    help="run the notebook as the submission, on the source pushed last")
    ap.add_argument("--smoke", action="store_true",
                    help="run the notebook on three test tasks, on the source pushed last")
    ap.add_argument("--run", nargs="+", metavar=("SET", "ID"),
                    help="push the source and run it scored on train or eval, every puzzle or the ids named")
    args = ap.parse_args()
    if args.kernel or args.smoke:
        push_kernel({"ARC_SMOKE_CHECK": "1"}, 1200) if args.smoke else push_kernel()
        sys.exit(0)
    which, ids = (args.run[0], args.run[1:]) if args.run else (None, [])
    if which not in (None, "train", "eval"):
        sys.exit("--run reads train or eval, the sets whose answers are known")
    sync()
    push(args.message or commit())
    if not wait_ready():
        sys.exit(1)
    if which:
        push_kernel({"ARC_DATASET": which, "ARC_RUN_IDS": " ".join(ids)})
        print("When it is done, into out/<label> for scripts/readouts.py:")
        print("    kaggle kernels output mohamedelsharkawy0/arc-build -p out/<label>")
        print("    tar -xzf out/<label>/said.tar.gz -C out/<label>")
        sys.exit(0)
    print("done.")
    print()
    print("NEXT: run the notebook on this source, through this script, which writes")
    print("the notebook from build.py first:")
    print("    python scripts/kaggle_push.py --smoke    three test tasks")
    print("    python scripts/kaggle_push.py --kernel   the submission")
    print("  (--kernel last before submitting: the notebook's newest version is what")
    print("   the browser shows. Then open arc-build in the browser, set accelerator =")
    print("   L4x4, Save & Run All.)")
