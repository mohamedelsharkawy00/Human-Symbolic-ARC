"""Time a clean build: python scripts/profile_build.py [x64-Release].

Each run gets a new out/build-profile-* directory. Release flags are unchanged.
Reports include wall-clock phases, individual compiler/linker/backup timings,
and Ninja edges (which overlap; do not sum them to get elapsed build time).
This clears no OS caches and reuses installed tools and existing puzzle data.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from build import build_lock, toolchain_environment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def time_command():
    directory, label = sys.argv[2:4]
    command = sys.argv[5:]
    start = time.perf_counter()
    code = subprocess.call(command)
    record = {"name": label, "seconds": time.perf_counter() - start,
              "exit_code": code, "command": command}
    (Path(directory) / f"{label}-{uuid.uuid4().hex}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return code


def ninja_edges(path):
    edges = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                continue
            start, end, _, output, command_hash = line.split("\t")
            key = (start, end, command_hash)
            edge = edges.setdefault(key, {"start_seconds": int(start) / 1000,
                "seconds": (int(end) - int(start)) / 1000, "outputs": []})
            edge["outputs"].append(output)
    return sorted(edges.values(), key=lambda row: row["start_seconds"])


def profile():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default="x64-Release",
                        choices=["x64-Release", "x64-RelWithDebInfo", "x64-Debug"])
    args = parser.parse_args()
    run = ROOT / "out" / ("build-profile-" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    timing = run / "timings"
    timing.mkdir(parents=True)
    phases = []
    env = os.environ.copy()
    total_start = time.perf_counter()
    print(f"Report directory: {run}", flush=True)

    def phase(name, command, required=True, environment=None):
        active_environment = env if environment is None else environment
        print(f"\n>>> {name}", flush=True)
        start = time.perf_counter()
        search_path = next((v for k, v in active_environment.items() if k.upper() == "PATH"), "")
        command = [shutil.which(command[0], path=search_path) or command[0], *command[1:]]
        with (run / f"{name}.log").open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=active_environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        elapsed = time.perf_counter() - start
        phases.append({"name": name, "seconds": elapsed, "exit_code": code})
        print(f"<<< {name}: {elapsed:.3f}s (exit {code})", flush=True)
        if code and required:
            raise subprocess.CalledProcessError(code, command)

    code = 0
    try:
        data = ROOT / "kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
        phase("puzzle-data", [sys.executable, "scripts/fetch_data.py"] +
              (["--check"] if data.exists() else []), required=False)
        phase("session-check", [sys.executable, "scripts/session_check.py"], required=False)
        env = toolchain_environment(lambda command, environment:
            phase("toolchain-setup", command, environment=environment))
        env["NINJA_STATUS"] = "[%f/%t, %es] "
        command = ["cmake", "--preset", args.preset, "-B", str(run / "build"),
                   f"-DARC_BUILD_TIMING_DIR={timing.as_posix()}"]
        for kind, label in (("COMPILER", "compile"), ("LINKER", "link")):
            launcher = ";".join([Path(sys.executable).as_posix(), SCRIPT.as_posix(),
                "--time-command", timing.as_posix(), f"CXX-{label}", "--"])
            command.append(f"-DCMAKE_CXX_{kind}_LAUNCHER={launcher}")
        phase("configure", command)
        phase("build", ["cmake", "--build", str(run / "build"), "--", "-v", "-d", "stats"])
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Profile stopped: {error}", flush=True)
        code = 1
    finally:
        total = time.perf_counter() - total_start
        commands = [json.loads(p.read_text(encoding="utf-8")) for p in timing.glob("*.json")]
        edges = ninja_edges(run / "build/.ninja_log")
        report = {"preset": args.preset, "total_seconds": total, "exit_code": code,
                  "phases": phases, "commands": commands, "ninja_edges": edges}
        (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
        lines = [f"Fresh {args.preset} build: {total:.3f}s; exit {code}", "", "Wall-clock phases:"]
        lines += [f"  {p['seconds']:9.3f}s  {p['name']} (exit {p['exit_code']})" for p in phases]
        lines += ["", "Individual commands (overlap; included in build phase):"]
        lines += [f"  {p['seconds']:9.3f}s  {p['name']} (exit {p['exit_code']})"
                  for p in sorted(commands, key=lambda p: -p["seconds"])]
        lines += ["", "Ninja edges (link edge includes backup and linker wrapper):"]
        lines += [f"  {p['seconds']:9.3f}s  {', '.join(p['outputs'])}" for p in edges]
        lines += ["", "Fresh outputs, including shaders; installed tools, puzzle data and OS caches reused.",
                  "Compiler and linker timings include subprocess overhead; analysis/LTCG remain enabled in Release."]
        summary = "\n".join(lines) + "\n"
        (run / "report.txt").write_text(summary, encoding="utf-8", newline="\n")
        print("\n" + summary + f"Reports: {run}", flush=True)
    return code


def main():
    try:
        with build_lock():
            return profile()
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(time_command() if len(sys.argv) > 1 and sys.argv[1] == "--time-command" else main())
