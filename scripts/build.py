"""Build ARC: build.py [preset] [NOLAUNCH] [target]."""
from contextlib import contextmanager
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def build_lock():
    """One build per folder; the OS releases the lock even if the build dies."""
    (ROOT / "out").mkdir(exist_ok=True)
    with (ROOT / "out/build.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another build is already running in this folder.") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def toolchain_environment(run):
    """Capture MSVC's environment; run owns logging and checks the exit code."""
    with tempfile.TemporaryDirectory(dir=ROOT / "out", prefix="build-env-") as directory:
        folder = Path(directory)
        exporter = folder / "export.py"
        result = folder / "environment.json"
        setup = folder / "environment.cmd"
        exporter.write_text("import json, os, pathlib, sys\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)), "
            "encoding='utf-8', newline='\\n')\n", encoding="utf-8", newline="\n")
        setup.write_text('@echo off\n'
            'if not defined DevEnvDir call "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\Tools\\VsDevCmd.bat" -arch=x64 -no_logo\n'
            'if errorlevel 1 exit /b 1\n'
            f'"{sys.executable}" "{exporter}" "{result}"\n', encoding="utf-8", newline="\n")
        environment = os.environ.copy()
        installer = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer"
        environment["PATH"] = str(installer) + os.pathsep + environment.get("PATH", "")
        run(["cmd.exe", "/d", "/c", str(setup)], environment)
        return json.loads(result.read_text(encoding="utf-8"))


def build(preset, target):
    data = ROOT / "kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
    fetched = subprocess.call([sys.executable, "scripts/fetch_data.py"] +
                              (["--check"] if data.exists() else []), cwd=ROOT)
    if fetched:
        print("Fetch failed. Run python scripts/fetch_data.py once online.")
    subprocess.call([sys.executable, "scripts/session_check.py"], cwd=ROOT)
    environment = toolchain_environment(lambda command, env:
        subprocess.run(command, cwd=ROOT, env=env, check=True))
    output = ROOT / "out" / preset
    for old in output.glob("*.old"):
        try:
            old.unlink()
        except PermissionError:
            pass
    # Windows allows renaming a running executable, but not overwriting it.
    for name in ("ARC", "arc_headless"):
        executable = output / f"{name}.exe"
        if executable.exists():
            try:
                with executable.open("ab"):
                    pass
            except PermissionError:
                executable.rename(executable.with_suffix(f".exe.{uuid.uuid4().hex}.old"))
    cmake = shutil.which("cmake", path=environment.get("PATH")) or "cmake"
    if not output.exists():
        subprocess.run([cmake, "--preset", preset], cwd=ROOT, env=environment, check=True)
    subprocess.run([cmake, "--build", "--preset", preset] +
                   (["--target", target] if target else []) + ["--", "-v"],
                   cwd=ROOT, env=environment, check=True)


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "x64-Release"
    no_launch = len(sys.argv) > 2 and sys.argv[2].upper() == "NOLAUNCH"
    target = sys.argv[3] if len(sys.argv) > 3 else None
    try:
        with build_lock():
            build(preset, target)
        if not no_launch:
            subprocess.Popen([str(ROOT / "out" / preset / "ARC.exe")], cwd=ROOT,
                             close_fds=True)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
