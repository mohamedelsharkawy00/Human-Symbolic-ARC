"""Build the CPU-only inference readout tool for Linux cloud development.

Run from any directory: python3 scripts/cloud_setup.py
Use --build-only when puzzle data is supplied separately or networking is off.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true", help="do not fetch puzzle data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    if not compiler or not shutil.which(compiler[0]):
        parser.error("a C++20 compiler is needed: install g++ or set CXX=clang++")
    out = root / "out" / "cloud"
    out.mkdir(parents=True, exist_ok=True)
    command = compiler + [
        "-std=c++20", "-O2", "-pthread", "-DARC_ENABLE_CUDA_PIECES=0",
        "-I", str(root / "include" / "core"),
        str(root / "scripts" / "probe.cpp"), "-o", str(out / "arc_probe"),
    ]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    if not args.build_only:
        data = root / "kaggle/input/competitions/arc-prize-2026-arc-agi-2"
        needed = ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json")
        if not all((data / name).is_file() for name in needed):
            subprocess.run([sys.executable, str(root / "scripts/fetch_data.py")], cwd=root, check=True)
    print("CPU inference build ready. From the repo root:")
    print("  out/cloud/arc_probe out/probe/cloud 00576224")
    print("Readouts only; arc_said and CopyTo still require the CUDA build.")


if __name__ == "__main__":
    main()
