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

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def time_command():
    directory, label = sys.argv[2:4]
    command = sys.argv[5:]
    if label == "CUDA-compile" and os.environ.get("ARC_PROFILE_CUDA") == "1":
        prefix = Path(directory) / ("cuda-" + uuid.uuid4().hex)
        command += ["--time", str(prefix.with_suffix(".csv")),
                    "--fdevice-time-trace", str(prefix.with_suffix(".trace"))]
    start = time.perf_counter()
    code = subprocess.call(command)
    record = {"name": label, "seconds": time.perf_counter() - start,
              "exit_code": code, "command": command}
    (Path(directory) / f"{label}-{uuid.uuid4().hex}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default="x64-Release",
                        choices=["x64-Release", "x64-RelWithDebInfo", "x64-Debug"])
    parser.add_argument("--cuda", action="store_true",
                        help="record NVCC phase CSV and device compiler JSON timeline (CUDA 12.8+)")
    args = parser.parse_args()
    run = ROOT / "out" / ("build-profile-" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    timing = run / "timings"
    timing.mkdir(parents=True)
    phases = []
    env = os.environ.copy()
    total_start = time.perf_counter()
    print(f"Report directory: {run}", flush=True)

    def phase(name, command, required=True):
        print(f"\n>>> {name}", flush=True)
        start = time.perf_counter()
        search_path = next((v for k, v in env.items() if k.upper() == "PATH"), "")
        command = [shutil.which(command[0], path=search_path) or command[0], *command[1:]]
        with (run / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
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
        # Capture VsDevCmd's environment through a generated batch file. Paths
        # are quoted as batch paths, never interpolated as PowerShell code.
        setup = run / "environment.cmd"
        environment_file = run / "environment.json"
        exporter = run / "export_environment.py"
        exporter.write_text("import json, os, pathlib, sys\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n",
            encoding="utf-8")
        setup.write_text('@echo off\n'
            'if not defined DevEnvDir call "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\Tools\\VsDevCmd.bat" -arch=x64\n'
            'if errorlevel 1 exit /b 1\n'
            f'"{sys.executable}" "{exporter}" "{environment_file}"\n', encoding="utf-8")
        installer = str(Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer")
        env["PATH"] = installer + os.pathsep + env.get("PATH", "")
        phase("toolchain-setup", ["cmd.exe", "/d", "/c", str(setup)])
        env = json.loads(environment_file.read_text(encoding="utf-8"))
        environment_file.unlink()  # The environment is not a report artifact.
        env["NINJA_STATUS"] = "[%f/%t, %es] "
        env["ARC_PROFILE_CUDA"] = "1" if args.cuda else "0"
        command = ["cmake", "--preset", args.preset, "-B", str(run / "build"),
                   f"-DARC_BUILD_TIMING_DIR={timing.as_posix()}"]
        for language in ("CXX", "CUDA"):
            for kind, label in (("COMPILER", "compile"), ("LINKER", "link")):
                launcher = ";".join([Path(sys.executable).as_posix(), SCRIPT.as_posix(),
                    "--time-command", timing.as_posix(), f"{language}-{label}", "--"])
                command.append(f"-DCMAKE_{language}_{kind}_LAUNCHER={launcher}")
        phase("configure", command)
        phase("build", ["cmake", "--build", str(run / "build"), "--", "-v", "-d", "stats"])
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Profile stopped: {error}", flush=True)
        code = 1
    finally:
        total = time.perf_counter() - total_start
        commands = [json.loads(p.read_text(encoding="utf-8")) for p in timing.glob("*.json")
                    if not p.name.startswith("cuda-")]
        edges = ninja_edges(run / "build/.ninja_log")
        report = {"preset": args.preset, "total_seconds": total, "exit_code": code,
                  "cuda_trace": args.cuda,
                  "phases": phases, "commands": commands, "ninja_edges": edges}
        (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
        (run / "report.txt").write_text(summary, encoding="utf-8")
        print("\n" + summary + f"Reports: {run}", flush=True)
        if args.cuda:
            from summarize_cuda_build import summarize
            print(summarize(run) or "No CUDA phase timings were produced.", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(time_command() if len(sys.argv) > 1 and sys.argv[1] == "--time-command" else main())
