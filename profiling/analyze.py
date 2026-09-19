"""Analyze VTune CPU stacks CSV + Nsight Systems GPU CSVs and produce a unified report.

Updated April 2026 for full-CUDA architecture:
- Hot path: GPU kernel (dsl_exec_and_score_kernel_v2)
- CPU hot path: beam search generation + program packing
- Source-line breakdown targets: 26_solver_executor.hpp, 23_solver_beam.hpp,
  cuda_solver.cu, 24_fixed_dsl.hpp
- Nsight Systems: CUDA kernel times, memory transfers, API overhead
"""

import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "hotspots.txt")

# VTune CSV paths
VTUNE_FUNC_CSV = os.path.join(OUTPUT_DIR, "cpu_stacks_vtune.csv")
VTUNE_LINE_CSV = os.path.join(OUTPUT_DIR, "source_lines.csv")
VTUNE_THREAD_CSV = os.path.join(OUTPUT_DIR, "threads.csv")

# Nsight Systems CSV paths
NSYS_KERN_CSV = os.path.join(OUTPUT_DIR, "nsys_stats_cuda_gpu_kern_sum.csv")
NSYS_MEM_CSV = os.path.join(OUTPUT_DIR, "nsys_stats_cuda_gpu_mem_time_sum.csv")
NSYS_API_CSV = os.path.join(OUTPUT_DIR, "nsys_stats_cuda_api_sum.csv")
NSYS_EXEC_CSV = os.path.join(OUTPUT_DIR, "nsys_stats_cuda_kern_exec_sum.csv")

# Nsight Compute CSV (per-kernel metrics deep-dive)
NCU_CSV = os.path.join(OUTPUT_DIR, "ncu_kernel.csv")

TOP_N = 80

# --- Source files of interest for line-level breakdown ---
HOT_SOURCE_FILES = {
    "26_solver_executor.hpp",
    "23_solver_beam.hpp",
    "cuda_solver.cu",
    "24_fixed_dsl.hpp",
    "11_dsl.cuh",
    "12_environment.hpp",
}

# --- Category definitions ---
INDIRECT_MEMORY = {"vcruntime140"}
INDIRECT_HEAP = {"ntdll", "ucrtbase"}
GPU_DRIVER = {"nvwgf2umx", "nvlddmkm.sys", "nvoglv64", "nvapi64",
              "nvldumdx", "nvgpucomp64", "nvrtum64", "nvspcap64",
              "nvmemmapstoragex", "nvppex",
              "vulkan-1", "d3d12core", "d3d12", "dxgi", "dxcore",
              "dxgkrnl.sys", "dxgmms2.sys", "dcomp", "opengl32"}
CUDA_RUNTIME = {"cudart64_12", "cudart64", "nvcuda", "cublas64_12",
                "cublaslt64_12", "nvcuda64", "nvcudart_hybrid64"}
OS_KERNEL = {"ntoskrnl", "win32kfull.sys", "win32kbase.sys", "win32k.sys",
             "win32u", "kernelbase", "kernel32", "user32", "gdi32",
             "gdi32full", "fltmgr.sys", "ntfs.sys", "classpnp.sys",
             "storport.sys", "stornvme.sys", "msvcp140"}
NOISE = {"imagehlp", "dbghelp", "graphics-hook64", "apphelp",
         "wpprecorder.sys", "directxdatabasehelper", "shcore"}

ARC_MODULE = "arc"
CAT_LABELS = {
    "ARC":    "ARC",
    "MEM":    "ARC:mem",
    "HEAP":   "ARC:heap",
    "GPU":    "GPU/Vulkan",
    "CUDA":   "CUDA",
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

    if mod.lower() != ARC_MODULE:
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

    # CUDA-related
    if "cuda" in func.lower() or "memcpy" in func.lower():
        return f"arc::{func}"

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

    if mod == ARC_MODULE:
        return "ARC"
    if mod in INDIRECT_MEMORY or mod in {m.lower() for m in INDIRECT_MEMORY}:
        return "MEM"
    if mod in INDIRECT_HEAP or mod in {m.lower() for m in INDIRECT_HEAP}:
        return "HEAP"
    if mod in CUDA_RUNTIME or mod in {m.lower() for m in CUDA_RUNTIME}:
        return "CUDA"
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
        ("beam_search",     ["step", "generate_phase", "generate_binary",
                             "find_slots", "rebuild_skeleton", "advance_level",
                             "execute_current_phase", "check_yield"]),
        ("tree_ops",        ["copy", "remove", "flatten", "create_unary",
                             "create_binary", "update_hash", "stringify"]),
        ("candidate_pack",  ["collect_flat_candidate", "pack_cuda_dsl",
                             "convert_program_to_gpu", "append_flat_candidate",
                             "build_single_flat"]),
        ("gpu_interface",   ["cuda_dsl_exec", "cuda_dsl_upload", "cuda_dsl_invalidate",
                             "cuda_init", "ensure_exec_capacity"]),
        ("winner_select",   ["update_best_solution", "report_score",
                             "log_tree_record", "select_cuda"]),
        ("hash_cache",      ["global_cache", "unordered_map", "hash_container"]),
        ("fixed_dsl",       ["fixed_init_context", "fixed_execute", "fixed_exec_",
                             "to_gpu_object_list"]),
        ("display_ui",      ["update_grid", "update_example", "update_display",
                             "render", "draw", "present", "swap"]),
    ]

    for name, exc, _, _ in hotspots:
        func = get_function(name).lower()
        mod = get_module(name).lower()
        if mod != ARC_MODULE:
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


# =====================================================================
# Nsight Systems ingestion
# =====================================================================

def fmt_ns(ns: float) -> str:
    """Format nanoseconds into human-readable time."""
    if ns >= 1e9:
        return f"{ns / 1e9:.3f}s"
    if ns >= 1e6:
        return f"{ns / 1e6:.3f}ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.1f}us"
    return f"{ns:.0f}ns"


def fmt_pct(pct: float) -> str:
    return f"{pct:.1f}%"


def parse_nsys_kernels(path: str) -> list[dict]:
    """Parse cuda_gpu_kern_sum CSV."""
    if not os.path.isfile(path):
        return []
    rows = parse_csv(path)
    out = []
    for row in rows:
        name = row.get("Name", "").strip()
        if not name:
            continue
        # Simplify kernel name: extract function name before first (
        short = name.split("(")[0].strip() if "(" in name else name
        # Remove return type prefix if present
        if " " in short:
            short = short.rsplit(" ", 1)[-1]
        out.append({
            "name": short,
            "full_name": name,
            "time_pct": float(row.get("Time (%)", "0") or "0"),
            "total_ns": float(row.get("Total Time (ns)", "0") or "0"),
            "instances": int(row.get("Instances", "0") or "0"),
            "avg_ns": float(row.get("Avg (ns)", "0") or "0"),
            "min_ns": float(row.get("Min (ns)", "0") or "0"),
            "max_ns": float(row.get("Max (ns)", "0") or "0"),
        })
    out.sort(key=lambda x: x["total_ns"], reverse=True)
    return out


def parse_nsys_memops(path: str) -> list[dict]:
    """Parse cuda_gpu_mem_time_sum CSV."""
    if not os.path.isfile(path):
        return []
    rows = parse_csv(path)
    out = []
    for row in rows:
        op = row.get("Operation", "").strip()
        if not op:
            continue
        out.append({
            "operation": op,
            "time_pct": float(row.get("Time (%)", "0") or "0"),
            "total_ns": float(row.get("Total Time (ns)", "0") or "0"),
            "count": int(row.get("Count", "0") or "0"),
            "avg_ns": float(row.get("Avg (ns)", "0") or "0"),
            "min_ns": float(row.get("Min (ns)", "0") or "0"),
            "max_ns": float(row.get("Max (ns)", "0") or "0"),
        })
    out.sort(key=lambda x: x["total_ns"], reverse=True)
    return out


def parse_nsys_api(path: str) -> list[dict]:
    """Parse cuda_api_sum CSV."""
    if not os.path.isfile(path):
        return []
    rows = parse_csv(path)
    out = []
    for row in rows:
        name = row.get("Name", "").strip()
        if not name:
            continue
        out.append({
            "name": name,
            "time_pct": float(row.get("Time (%)", "0") or "0"),
            "total_ns": float(row.get("Total Time (ns)", "0") or "0"),
            "calls": int(row.get("Num Calls", row.get("Instances", "0")) or "0"),
            "avg_ns": float(row.get("Avg (ns)", "0") or "0"),
            "min_ns": float(row.get("Min (ns)", "0") or "0"),
            "max_ns": float(row.get("Max (ns)", "0") or "0"),
        })
    out.sort(key=lambda x: x["total_ns"], reverse=True)
    return out


def parse_nsys_kern_exec(path: str) -> list[dict]:
    """Parse cuda_kern_exec_sum CSV (launch-to-exec latency)."""
    if not os.path.isfile(path):
        return []
    rows = parse_csv(path)
    out = []
    for row in rows:
        name = row.get("Name", "").strip()
        if not name:
            continue
        short = name.split("(")[0].strip() if "(" in name else name
        if " " in short:
            short = short.rsplit(" ", 1)[-1]
        out.append({
            "name": short,
            "instances": int(row.get("Instances", "0") or "0"),
            "exec_ns": float(row.get("Avg Exec Time (ns)", row.get("Avg (ns)", "0")) or "0"),
            "launch_ns": float(row.get("Avg Launch Time (ns)", "0") or "0"),
        })
    return out


# =====================================================================
# Nsight Compute ingestion
# =====================================================================

def parse_ncu(path: str) -> list[dict]:
    """Parse Nsight Compute --csv output.

    NCU CSV is metric-per-row: each kernel launch produces many rows.
    Returns a list of {kernel_name, launch_id, metrics: {name: value}}.
    """
    if not os.path.isfile(path):
        return []
    try:
        rows = parse_csv(path)
    except Exception:
        return []
    if not rows:
        return []

    launches: dict[tuple, dict] = {}
    for row in rows:
        kname = (row.get("Kernel Name") or row.get("kernel name") or "").strip()
        launch_id = (row.get("ID") or row.get("id") or "").strip()
        metric = (row.get("Metric Name") or row.get("metric name") or "").strip()
        unit = (row.get("Metric Unit") or row.get("metric unit") or "").strip()
        value = (row.get("Metric Value") or row.get("metric value") or "").strip()
        if not kname or not metric:
            continue
        short = kname.split("(")[0].strip() if "(" in kname else kname
        if " " in short:
            short = short.rsplit(" ", 1)[-1]
        key = (short, launch_id)
        if key not in launches:
            launches[key] = {"name": short, "launch_id": launch_id, "metrics": {}}
        launches[key]["metrics"][metric] = (value, unit)
    return list(launches.values())


def ncu_get(metrics: dict, key: str) -> str:
    v = metrics.get(key)
    if not v:
        return "-"
    val, unit = v
    return f"{val} {unit}".strip()


# --- Report output ---

def write_report(hotspots: list[tuple[str, float, float, int]], out_path: str,
                 area_breakdown: list[tuple[str, float]] | None = None,
                 source_breakdown: dict[str, list[tuple[int, str, float]]] | None = None,
                 thread_breakdown: list[tuple[str, float, int]] | None = None,
                 gpu_kernels: list[dict] | None = None,
                 gpu_memops: list[dict] | None = None,
                 gpu_api: list[dict] | None = None,
                 gpu_kern_exec: list[dict] | None = None,
                 ncu_launches: list[dict] | None = None) -> None:
    total = sum(h[3] for h in hotspots)
    cats: dict[str, float] = defaultdict(float)
    for name, exc, _, _ in hotspots:
        cats[categorize(get_module(name))] += exc

    arc_all = cats["ARC"] + cats["MEM"] + cats["HEAP"]

    has_gpu = bool(gpu_kernels or gpu_memops or gpu_api)
    has_ncu = bool(ncu_launches)

    with open(out_path, "w", encoding="utf-8") as f:
        def w(s: str = "") -> None:
            f.write(s + "\n")

        sources = []
        if hotspots:
            sources.append("VTune")
        if has_gpu:
            sources.append("Nsight Systems")
        if has_ncu:
            sources.append("Nsight Compute")

        w(f"ARC Profiling Report  |  {datetime.now():%Y-%m-%d %H:%M:%S}  |  {total} samples  |  {' + '.join(sources)}")
        w("=" * 120)

        # =========================================================
        # GPU Section (Nsight Systems)
        # =========================================================
        if has_gpu:
            w()
            w("  GPU Profile (Nsight Systems)")
            w("-" * 80)

            if gpu_kernels:
                total_kern_ns = sum(k["total_ns"] for k in gpu_kernels)
                w()
                w(f"  CUDA Kernels  (total GPU time: {fmt_ns(total_kern_ns)})")
                w(f"  {'Kernel':<45}  {'Time':>10}  {'%':>6}  {'Calls':>7}  {'Avg':>10}  {'Min':>10}  {'Max':>10}")
                w(f"  {'-'*45}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
                for k in gpu_kernels:
                    w(f"  {k['name']:<45}  {fmt_ns(k['total_ns']):>10}  {fmt_pct(k['time_pct']):>6}  {k['instances']:>7}  {fmt_ns(k['avg_ns']):>10}  {fmt_ns(k['min_ns']):>10}  {fmt_ns(k['max_ns']):>10}")

            if gpu_memops:
                total_mem_ns = sum(m["total_ns"] for m in gpu_memops)
                w()
                w(f"  CUDA Memory Transfers  (total: {fmt_ns(total_mem_ns)})")
                w(f"  {'Operation':<40}  {'Time':>10}  {'%':>6}  {'Count':>7}  {'Avg':>10}  {'Min':>10}  {'Max':>10}")
                w(f"  {'-'*40}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
                for m in gpu_memops:
                    w(f"  {m['operation']:<40}  {fmt_ns(m['total_ns']):>10}  {fmt_pct(m['time_pct']):>6}  {m['count']:>7}  {fmt_ns(m['avg_ns']):>10}  {fmt_ns(m['min_ns']):>10}  {fmt_ns(m['max_ns']):>10}")

            if gpu_api:
                w()
                w(f"  CUDA API Calls (CPU-side overhead)")
                w(f"  {'API':<45}  {'Time':>10}  {'%':>6}  {'Calls':>7}  {'Avg':>10}")
                w(f"  {'-'*45}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*10}")
                for a in gpu_api[:15]:
                    w(f"  {a['name']:<45}  {fmt_ns(a['total_ns']):>10}  {fmt_pct(a['time_pct']):>6}  {a['calls']:>7}  {fmt_ns(a['avg_ns']):>10}")

            if gpu_kern_exec:
                w()
                w(f"  Kernel Launch-to-Execution Latency")
                w(f"  {'Kernel':<45}  {'Instances':>10}  {'Avg Exec':>12}  {'Avg Launch':>12}")
                w(f"  {'-'*45}  {'-'*10}  {'-'*12}  {'-'*12}")
                for k in gpu_kern_exec:
                    if k["instances"] > 0:
                        w(f"  {k['name']:<45}  {k['instances']:>10}  {fmt_ns(k['exec_ns']):>12}  {fmt_ns(k['launch_ns']):>12}")

            # GPU utilization summary
            if gpu_kernels and gpu_memops:
                total_kern_ns = sum(k["total_ns"] for k in gpu_kernels)
                total_mem_ns = sum(m["total_ns"] for m in gpu_memops)
                total_gpu_ns = total_kern_ns + total_mem_ns
                w()
                w(f"  GPU Time Split:")
                w(f"    Compute: {fmt_ns(total_kern_ns):>10}  ({total_kern_ns / total_gpu_ns * 100:.1f}%)")
                w(f"    MemOps:  {fmt_ns(total_mem_ns):>10}  ({total_mem_ns / total_gpu_ns * 100:.1f}%)")
                w(f"    Total:   {fmt_ns(total_gpu_ns):>10}")

            w()
            w("=" * 120)

        # =========================================================
        # GPU Kernel Deep-Dive (Nsight Compute)
        # =========================================================
        if has_ncu:
            w()
            w("  GPU Kernel Deep-Dive (Nsight Compute)")
            w("-" * 80)
            w(f"  {len(ncu_launches)} kernel launch(es) profiled")
            w()

            for launch in ncu_launches:
                w(f"  Kernel: {launch['name']}  (launch #{launch['launch_id']})")
                for metric in sorted(launch["metrics"].keys()):
                    val, unit = launch["metrics"][metric]
                    w(f"    {metric:<60}  {val} {unit}".rstrip())
                w()

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
        w(f"    |- Memory ops (memcpy/set)    {cats['MEM']:7.2f}%")
        w(f"    |- Heap (new/delete/malloc)   {cats['HEAP']:7.2f}%")
        w(f"  CUDA runtime                  {cats['CUDA']:7.2f}%  <- CPU waiting on GPU / kernel launch")
        w(f"  GPU / Vulkan                  {cats['GPU']:7.2f}%  <- display rendering")
        w(f"  OS / Kernel                   {cats['OS']:7.2f}%")
        w()

        # Functional area breakdown
        if area_breakdown:
            w("=" * 120)
            w("  ARC Functional Area Breakdown")
            w(f"  {'Area':<25}  {'CPU%':>7}  {'Bar':<30}")
            w(f"  {'---':<25}  {'---':>7}  {'---':<30}")
            for area, pct in area_breakdown:
                bar = "#" * max(1, int(pct * 0.5))
                w(f"  {area:<25}  {pct:6.2f}%  {bar}")
            w()

        # Top functions
        w("=" * 120)
        w(f"  {'#':>4}  {'Exc%':>7}  {'Smpl':>6}  {'Cat':>9}  {'Bar':<20}  Function")
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
                w(f"  --- {fname} ({file_total:.2f}% total) ---")
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
    has_vtune = os.path.isfile(VTUNE_FUNC_CSV)
    has_nsys = any(os.path.isfile(p) for p in [NSYS_KERN_CSV, NSYS_MEM_CSV, NSYS_API_CSV])
    has_ncu = os.path.isfile(NCU_CSV)

    if not has_vtune and not has_nsys and not has_ncu:
        print(f"ERROR: No profiling data found. Run profile.bat first.")
        sys.exit(1)

    # VTune data
    hotspots = []
    area_breakdown = []
    source_breakdown = {}
    thread_breakdown = []
    if has_vtune:
        rows = parse_csv(VTUNE_FUNC_CSV)
        hotspots = extract_vtune(rows)
        area_breakdown = build_area_breakdown(hotspots)
        source_breakdown = build_source_breakdown(VTUNE_LINE_CSV)
        thread_breakdown = build_thread_breakdown(VTUNE_THREAD_CSV)

    # Nsight data
    gpu_kernels = parse_nsys_kernels(NSYS_KERN_CSV)
    gpu_memops = parse_nsys_memops(NSYS_MEM_CSV)
    gpu_api = parse_nsys_api(NSYS_API_CSV)
    gpu_kern_exec = parse_nsys_kern_exec(NSYS_EXEC_CSV)

    # Nsight Compute data
    ncu_launches = parse_ncu(NCU_CSV)

    write_report(hotspots, REPORT_PATH, area_breakdown, source_breakdown,
                 thread_breakdown, gpu_kernels, gpu_memops, gpu_api, gpu_kern_exec,
                 ncu_launches)
    print(f"Report written to {REPORT_PATH}")

    if hotspots:
        print(f"Total CPU samples: {sum(h[3] for h in hotspots)}")

    # Console summary
    print()
    if hotspots:
        print("CPU Category breakdown:")
        cats: dict[str, float] = defaultdict(float)
        for name, exc, _, _ in hotspots:
            cats[categorize(get_module(name))] += exc
        for cat in ["ARC", "MEM", "HEAP", "CUDA", "GPU", "OS"]:
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

    if gpu_kernels:
        print(f"\nGPU Kernels:")
        for k in gpu_kernels:
            print(f"  {k['name']:<45}  {fmt_ns(k['total_ns']):>10}  ({k['instances']} calls, avg {fmt_ns(k['avg_ns'])})")

    if ncu_launches:
        print(f"\nGPU Kernel Deep-Dive ({len(ncu_launches)} launches):")
        for launch in ncu_launches:
            print(f"  Kernel: {launch['name']}  (launch #{launch['launch_id']})")
            for metric in sorted(launch["metrics"].keys()):
                val, unit = launch["metrics"][metric]
                print(f"    {metric:<60}  {val} {unit}".rstrip())
            print()

    if gpu_memops:
        total_mem = sum(m["total_ns"] for m in gpu_memops)
        print(f"\nGPU Memory Transfers (total: {fmt_ns(total_mem)}):")
        for m in gpu_memops:
            print(f"  {m['operation']:<40}  {fmt_ns(m['total_ns']):>10}  ({m['count']} ops)")


if __name__ == "__main__":
    main()
