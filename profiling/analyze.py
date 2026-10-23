"""Summarize VTune CPU profiles.

Usage: python profiling/analyze.py [run-directory]
"""

import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "hotspots.txt")

# VTune CSV paths
VTUNE_FUNC_CSV = os.path.join(OUTPUT_DIR, "cpu_stacks_vtune.csv")
VTUNE_LINE_CSV = os.path.join(OUTPUT_DIR, "source_lines.csv")
VTUNE_THREAD_CSV = os.path.join(OUTPUT_DIR, "threads.csv")

TOP_N = 80

# --- Source files of interest for line-level breakdown ---
HOT_SOURCE_FILES = {
    "1_readings.hpp", "3_background.hpp", "4_objects.hpp",
    "5_matching.hpp", "6_correspondence.hpp", "7_infer.hpp", "8b_op_math.hpp",
    "9_sentences.hpp", "said.cpp",
    "ops.cpp",
    "13_environment.hpp",
}

# --- Category definitions ---
INDIRECT_MEMORY = {"vcruntime140"}
INDIRECT_HEAP = {"ntdll", "ucrtbase"}
GPU_DRIVER = {"nvwgf2umx", "nvlddmkm.sys", "nvoglv64", "nvapi64",
              "nvldumdx", "nvgpucomp64", "nvrtum64", "nvspcap64",
              "nvmemmapstoragex", "nvppex",
              "vulkan-1", "d3d12core", "d3d12", "dxgi", "dxcore",
              "dxgkrnl.sys", "dxgmms2.sys", "dcomp", "opengl32"}
OS_KERNEL = {"ntoskrnl", "win32kfull.sys", "win32kbase.sys", "win32k.sys",
             "win32u", "kernelbase", "kernel32", "user32", "gdi32",
             "gdi32full", "fltmgr.sys", "ntfs.sys", "classpnp.sys",
             "storport.sys", "stornvme.sys", "msvcp140"}
NOISE = {"imagehlp", "dbghelp", "graphics-hook64", "apphelp",
         "wpprecorder.sys", "directxdatabasehelper", "shcore"}

ARC_MODULES = {"arc", "arc_said", "arc_headless"}
CAT_LABELS = {
    "ARC":    "ARC",
    "MEM":    "ARC:mem",
    "HEAP":   "ARC:heap",
    "GPU":    "GPU/Vulkan",
    "OS":     "OS",
    "NOISE":  "noise",
    "OTHER":  "other",
}


def get_module(name: str) -> str:
    if "!" in name:
        return name.split("!")[0].strip()
    return ""


def get_function(name: str) -> str:
    if "!" in name:
        return name.split("!", 1)[1].strip()
    return name.strip()


def simplify_name(name: str) -> str:
    """Simplify a full symbol for readability."""
    mod = get_module(name)
    func = get_function(name)

    if categorize(mod) != "ARC":
        return name

    # ILT thunks
    if func.startswith("ILT+"):
        paren = func.find("(")
        if paren != -1:
            inner = func[paren + 1:].rstrip(")")
            if inner.startswith("?"):
                inner = inner[1:]
            return f"arc::[thunk] {inner}"
        return f"arc::{func}"

    # std::_Hash containers
    if func.startswith("std::_Hash<"):
        ops = ["::_Try_emplace", "::clear", "::emplace", "::_Forced_rehash"]
        for op in ops:
            if op in func:
                if "_Umap_traits" in func:
                    return f"arc::unordered_map{op}"
                if "_Uset_traits" in func:
                    return f"arc::unordered_set{op}"
                return f"arc::hash_container{op}"
        return "arc::hash_container::?"

    # std::vector
    if func.startswith("std::vector<"):
        for op in ["::_Resize_reallocate", "::resize", "::_Change_array",
                    "::insert", "::~vector"]:
            if op in func:
                return f"arc::vector{op}"
        return "arc::vector::?"

    # std:: other
    if func.startswith("std::"):
        return f"arc::{func}"

    # arc:: namespace
    if func.startswith("arc::"):
        return func

    # Lambdas
    if func.startswith("`arc::"):
        return func

    return f"arc::{func}"


def categorize(module: str) -> str:
    mod = module.lower().strip()
    if mod.endswith(".exe") or mod.endswith(".dll") or mod.endswith(".sys"):
        mod = mod.rsplit(".", 1)[0]

    if mod in ARC_MODULES:
        return "ARC"
    if mod in INDIRECT_MEMORY or mod in {m.lower() for m in INDIRECT_MEMORY}:
        return "MEM"
    if mod in INDIRECT_HEAP or mod in {m.lower() for m in INDIRECT_HEAP}:
        return "HEAP"
    if mod in GPU_DRIVER or mod in {m.lower() for m in GPU_DRIVER}:
        return "GPU"
    if mod in OS_KERNEL or mod in {m.lower() for m in OS_KERNEL}:
        return "OS"
    if mod in NOISE or mod in {m.lower() for m in NOISE}:
        return "NOISE"
    if mod.startswith("thread ") or mod.startswith("process"):
        return "OTHER"
    if mod in ("root", "broken", "?", ""):
        return "OTHER"
    return "OTHER"


def parse_csv(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# --- VTune extraction ---

def extract_vtune(rows: list[dict]) -> list[tuple[str, float, float, int]]:
    """Extract hotspots from VTune function-level CSV."""
    out: list[tuple[str, float, float, int]] = []
    time_key = None
    for row in rows[:1]:
        for k in row.keys():
            if "cpu time" in k.lower():
                time_key = k
                break
    if not time_key and rows:
        return out

    total_time = sum(float(r.get(time_key, "0") or "0") for r in rows)
    if total_time <= 0:
        return out

    for row in rows:
        func = row.get("Function", row.get("function", "")).strip()
        module = row.get("Module", row.get("module", "")).strip()
        if not func:
            continue
        try:
            cpu_time = float(row.get(time_key, "0") or "0")
        except ValueError:
            continue
        exc_pct = (cpu_time / total_time) * 100.0 if total_time > 0 else 0
        name = f"{module}!{func}" if module else func
        out.append((name, exc_pct, exc_pct, int(cpu_time * 1000)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# --- Source-line breakdown for hot files ---

def build_source_breakdown(source_line_csv: str) -> dict[str, list[tuple[int, str, float]]]:
    """Build per-file source line breakdown from VTune source-line CSV."""
    if not os.path.isfile(source_line_csv):
        return {}

    rows = parse_csv(source_line_csv)
    if not rows:
        return {}

    time_key = None
    for k in rows[0].keys():
        if "cpu time" in k.lower():
            time_key = k
            break
    if not time_key:
        return {}

    total_time = sum(float(r.get(time_key, "0") or "0") for r in rows)
    if total_time <= 0:
        return {}

    file_lines: dict[str, list[tuple[int, str, float]]] = defaultdict(list)

    for row in rows:
        src = row.get("Source File", row.get("source file", "")).strip()
        line_str = row.get("Source Line", row.get("source line", "")).strip()
        func = row.get("Function", row.get("function", "")).strip()
        if not src or not line_str:
            continue

        basename = os.path.basename(src).lower()
        matched = None
        for hot in HOT_SOURCE_FILES:
            if basename == hot.lower():
                matched = hot
                break
        if not matched:
            continue

        try:
            lineno = int(line_str)
            cpu_time = float(row.get(time_key, "0") or "0")
        except ValueError:
            continue

        exc_pct = (cpu_time / total_time) * 100.0
        if exc_pct >= 0.1:
            file_lines[matched].append((lineno, func, exc_pct))

    for fname in file_lines:
        file_lines[fname].sort(key=lambda x: x[2], reverse=True)
        file_lines[fname] = file_lines[fname][:30]

    return file_lines


# --- Functional area breakdown ---

def build_area_breakdown(hotspots: list[tuple[str, float, float, int]]) -> list[tuple[str, float]]:
    """Group ARC functions into logical areas."""
    areas: dict[str, float] = defaultdict(float)

    area_patterns = [
        ("word_replay", ["_holds", "replay", "partner_pieces"]),
        ("matching", ["matching", "correspondence", "segmentation"]),
        ("object_cells", ["arc::object::", "arc::cells::", "arc::pose_source"]),
        ("dsl_emission", ["say_the_", "sentence", "spell"]),
        ("inference", ["infer", "read_figures", "read_the_cut", "finish_the_line", "carry_the_hand", "says_the_most"]),
        ("tree_ops",        ["copy", "remove", "flatten", "update_hash", "stringify"]),
        ("program_runs",    ["::op_", "environment::draw", "run_ops", "one_op"]),
        ("hash_cache",      ["unordered_map", "hash_container"]),
        ("display_ui",      ["update_grid", "update_example", "update_display",
                             "render", "present", "swapchain"]),
    ]

    for name, exc, _, _ in hotspots:
        func = get_function(name).lower()
        mod = get_module(name).lower()
        if categorize(mod) != "ARC":
            continue

        matched = False
        for area_name, patterns in area_patterns:
            for pat in patterns:
                if pat.lower() in func:
                    areas[area_name] += exc
                    matched = True
                    break
            if matched:
                break
        if not matched:
            areas["other_arc"] += exc

    results = [(name, pct) for name, pct in areas.items() if pct >= 0.1]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def build_thread_breakdown(thread_csv: str) -> list[tuple[str, float, int]]:
    """Build per-thread CPU time breakdown from VTune thread-grouped CSV."""
    if not os.path.isfile(thread_csv):
        return []
    rows = parse_csv(thread_csv)
    if not rows:
        return []

    time_key = None
    for k in rows[0].keys():
        if "cpu time" in k.lower():
            time_key = k
            break
    if not time_key:
        return []

    total_time = sum(float(r.get(time_key, "0") or "0") for r in rows)
    if total_time <= 0:
        return []

    results = []
    for row in rows:
        thread = row.get("Thread Name", row.get("thread name", "")).strip()
        if not thread:
            thread = row.get("Thread", row.get("thread", "")).strip()
        if not thread:
            continue
        try:
            cpu_time = float(row.get(time_key, "0") or "0")
        except ValueError:
            continue
        pct = (cpu_time / total_time) * 100.0
        results.append((thread, pct, int(cpu_time * 1000)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# --- Report output ---

def write_report(hotspots: list[tuple[str, float, float, int]], out_path: str,
                 area_breakdown: list[tuple[str, float]] | None = None,
                 source_breakdown: dict[str, list[tuple[int, str, float]]] | None = None,
                 thread_breakdown: list[tuple[str, float, int]] | None = None) -> None:
    total = sum(h[3] for h in hotspots)
    cats: dict[str, float] = defaultdict(float)
    for name, exc, _, _ in hotspots:
        cats[categorize(get_module(name))] += exc

    arc_all = cats["ARC"] + cats["MEM"] + cats["HEAP"]

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        def w(s: str = "") -> None:
            f.write(s + "\n")

        w(f"ARC Profiling Report  |  {datetime.now():%Y-%m-%d %H:%M:%S}  |  {total} sampled CPU ms  |  VTune")
        w("=" * 120)

        # =========================================================
        # CPU Section (VTune)
        # =========================================================
        w()
        w("  CPU Profile (VTune)")
        w("-" * 80)
        w()
        w("  CPU Time Breakdown by Category")
        w(f"  ARC total (direct + indirect)  {arc_all:7.2f}%")
        w(f"    |- Direct code               {cats['ARC']:7.2f}%")
        w(f"    |- C runtime (incl. memory)    {cats['MEM']:7.2f}%")
        w(f"    |- Heap (new/delete/malloc)   {cats['HEAP']:7.2f}%")
        w(f"  GPU / Vulkan                  {cats['GPU']:7.2f}%")
        w(f"  OS / Kernel                   {cats['OS']:7.2f}%")
        w()

        # Functional area breakdown
        if area_breakdown:
            w("=" * 120)
            w("  ARC Functional Area Breakdown (function-name estimates)")
            w(f"  {'Area':<25}  {'CPU%':>7}  {'Bar':<30}")
            w(f"  {'---':<25}  {'---':>7}  {'---':<30}")
            for area, pct in area_breakdown:
                bar = "#" * max(1, int(pct * 0.5))
                w(f"  {area:<25}  {pct:6.2f}%  {bar}")
            w()

        # Top functions
        w("=" * 120)
        w(f"  {'#':>4}  {'Exc%':>7}  {'CPU ms':>6}  {'Cat':>9}  {'Bar':<20}  Function")
        w(f"  {'---':>4}  {'---':>7}  {'---':>6}  {'---':>9}  {'---':<20}  {'---':<55}")

        for i, (name, exc, inc, cnt) in enumerate(hotspots[:TOP_N], 1):
            mod = get_module(name)
            cat = CAT_LABELS.get(categorize(mod), "other")
            display = simplify_name(name)
            bar = "#" * max(1, int(exc * 0.5))
            w(f"  {i:4d}  {exc:6.2f}%  {cnt:6d}  {cat:>9}  {bar:<20}  {display}")

        w()
        w(f"  {min(TOP_N, len(hotspots))} of {len(hotspots)} functions shown.")

        # Source-line breakdown per hot file
        if source_breakdown:
            w()
            w("=" * 120)
            w("  Hot Source Lines (by file)")
            for fname in sorted(source_breakdown.keys()):
                lines = source_breakdown[fname]
                file_total = sum(p for _, _, p in lines)
                w()
                w(f"  --- {fname} ({file_total:.2f}% in shown lines) ---")
                w(f"  {'Line':>6}  {'CPU%':>7}  Function")
                for lineno, func, pct in lines:
                    w(f"  {lineno:6d}  {pct:6.2f}%  {func}")

        # Thread breakdown
        if thread_breakdown:
            w()
            w("=" * 120)
            w("  Thread Utilization")
            w(f"  {'#':>4}  {'CPU%':>7}  {'Bar':<20}  Thread")
            w(f"  {'---':>4}  {'---':>7}  {'---':<20}  {'---':<40}")
            for i, (thread, pct, _) in enumerate(thread_breakdown, 1):
                bar = "#" * max(1, int(pct * 0.2))
                w(f"  {i:4d}  {pct:6.2f}%  {bar:<20}  {thread}")
            w()


def main() -> None:
    if not os.path.isfile(VTUNE_FUNC_CSV):
        print(f"ERROR: No profiling data found. Run profile.bat first.")
        sys.exit(1)

    hotspots = extract_vtune(parse_csv(VTUNE_FUNC_CSV))
    area_breakdown = build_area_breakdown(hotspots)
    source_breakdown = build_source_breakdown(VTUNE_LINE_CSV)
    thread_breakdown = build_thread_breakdown(VTUNE_THREAD_CSV)

    write_report(hotspots, REPORT_PATH, area_breakdown, source_breakdown, thread_breakdown)
    print(f"Report written to {REPORT_PATH}")

    if hotspots:
        print(f"Sampled CPU time (ms): {sum(h[3] for h in hotspots)}")

    # Console summary
    print()
    if hotspots:
        print("CPU Category breakdown:")
        cats: dict[str, float] = defaultdict(float)
        for name, exc, _, _ in hotspots:
            cats[categorize(get_module(name))] += exc
        for cat in ["ARC", "MEM", "HEAP", "GPU", "OS"]:
            if cats[cat] > 0:
                label = CAT_LABELS.get(cat, cat)
                print(f"  {label:<12}  {cats[cat]:6.2f}%")

        if area_breakdown:
            print(f"\nFunctional areas:")
            for area, pct in area_breakdown[:10]:
                print(f"  {area:<25}  {pct:6.2f}%")

        print(f"\nTop 15 CPU functions:")
        for i, (name, exc, inc, cnt) in enumerate(hotspots[:15], 1):
            cat = CAT_LABELS.get(categorize(get_module(name)), "other")
            display = simplify_name(name)
            print(f"  {i:2d}. {exc:6.2f}%  [{cat:>9}]  {display}")


if __name__ == "__main__":
    main()
