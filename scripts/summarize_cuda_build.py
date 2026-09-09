"""Summarize an existing build profile's NVCC CSV and compiler timeline."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys


def summarize(run):
    phases = []
    for path in sorted((run / "timings").glob("cuda-*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle, skipinitialspace=True):
                row = {k.strip(): v.strip() for k, v in raw.items()}
                phases.append({"name": row["phase name"],
                    "seconds": float(row["metric"]) / (1000 if row["unit"] == "ms" else 1),
                    "source": row["source file name"], "tool": row["tool"]})
    if not phases:
        return None
    lines = ["CUDA compilation phases (NVCC --time):", ""]
    lines += [f"  {p['seconds']:9.3f}s  {p['name']}" for p in phases]
    traces = sorted((run / "timings").glob("cuda-*.trace*.json"))
    events = []
    for path in traces:
        data = json.loads(path.read_text(encoding="utf-8"))
        for event in data.get("traceEvents", []):
            if event.get("ph") != "X":
                continue
            args = event.get("args", {})
            symbol = args.get("name", args.get("detail", ""))
            match = re.match(r"_Z(\d+)", symbol)
            if match:
                symbol = symbol[match.end():match.end() + int(match[1])]
            events.append({"name": event["name"], "detail": symbol,
                           "seconds": event["dur"] / 1_000_000})
    interesting = [e for e in events if e["name"] in
                   ("Optimizing compilation unit", "NVVM Optimizer", "NVVM CodeGen")]
    lines += ["", "Selected device compiler events (inclusive, nested within phases):"]
    lines += [f"  {e['seconds']:9.3f}s  {e['name']}  {e['detail']}"
              for e in sorted(interesting, key=lambda e: -e["seconds"])]
    lines += ["", "Device compiler timelines:"] + [f"  {p}" for p in traces]
    lines += ["", "Open a timeline in edge://tracing or chrome://tracing.",
              "Trace events can be nested; inclusive event times must not be summed.",
              "Instrumentation adds overhead; compare optimization experiments using the same profiling mode."]
    # Use the advisor shipped beside the exact NVCC used by this build.
    commands = list((run / "timings").glob("CUDA-compile-*.json"))
    if commands and traces:
        command = json.loads(commands[0].read_text(encoding="utf-8"))["command"]
        advisor = Path(command[0]).parent / "ctadvisor.exe"
        if advisor.exists():
            outputs = []
            for trace in traces:
                result = subprocess.run([str(advisor), "--trace-file-path", str(trace),
                    "--max-hash-table-size", "1000000"], cwd=run,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
                outputs.append(f"Trace: {trace}\nExit: {result.returncode}\n{result.stdout}")
            (run / "cuda-advisor.txt").write_text("\n".join(outputs), encoding="utf-8")
            lines += [f"Advisor analysis: {run / 'cuda-advisor.txt'}"]
    report = {"phases": phases, "traces": [str(p) for p in traces], "device_events": interesting}
    (run / "cuda-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run / "cuda-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(Path(sys.argv[1]).resolve()) or "No CUDA timing CSV found.")
