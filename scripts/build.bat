@echo off
setlocal
cd /d "%~dp0.."

:: Single source of truth for building ARC. Callable by the profiling scripts.
::   %1  preset    - CMake preset name (default: x64-Release)
::   %2  NOLAUNCH  - if set to NOLAUNCH, build only; do not start ARC.exe
::                   (profilers launch the binary under their own collector)
set "PRESET=%~1"
if not defined PRESET set "PRESET=x64-Release"
set "NOLAUNCH=%~2"

:: Clear the screen only for standalone dev builds; when a profiling script
:: calls us with NOLAUNCH we must preserve its progress output.
if not defined NOLAUNCH cls

:: Ensure directories exist
if not exist "out" mkdir "out"

:: One build at a time. The lock is a file handle held open for the whole run,
:: so a second build stops here instead of corrupting the shared object files,
:: and the handle is released by the operating system even if this build dies.
set "BUILD_RAN="
9>"out\build.lock" call :build
if not defined BUILD_RAN (
    echo.
    echo Another build is already running in this folder. Wait for it to finish.
    exit /b 1
)
exit /b %BUILD_STATUS%

:build
set "BUILD_RAN=1"
set "BUILD_STATUS=0"

:: The repo ships no puzzles (the rules forbid redistributing them), so fetch
:: them the first time and never again. One file answers "is the data here"; a
:: fetch that fails is reported and the build carries on, since a machine with
:: no internet can still want a binary.
if not exist "kaggle\input\competitions\arc-prize-2026-arc-agi-2\arc-agi_training_challenges.json" (
    echo Puzzle data missing. Fetching it once...
    python "scripts\fetch_data.py"
    if errorlevel 1 echo [!] Fetch failed. Run "python scripts\fetch_data.py" once you are online.
) else (
    REM Only reports; updating stays a deliberate act because changing the data
    REM makes benchmark numbers before and after incomparable. Silent offline.
    python "scripts\fetch_data.py" --check
)

:: Nothing announces a second session, so ask here, where a collision would
:: matter, instead of hoping someone remembers to look. Silent when alone.
python "scripts\session_check.py"

:: Ensure MSVC environment if not already present
:: VsDevCmd invokes vswhere by name; make it discoverable without current-directory lookup.
:: SETLOCAL above keeps this PATH change confined to the build.
if not defined DevEnvDir set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
if not defined DevEnvDir (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
)

:: Clear leftovers from earlier builds; a still-running old binary stays locked
:: and is simply skipped, then swept on a later run.
del /q "out\%PRESET%\*.old" 2>nul

:: Rename instead of kill. Windows forbids overwriting a running exe but allows
:: renaming it, so the path frees up for the linker while another session keeps
:: driving the old binary. The random suffix keeps a second build from colliding
:: with an .old that is still held open.
if exist "out\%PRESET%\ARC.exe" ren "out\%PRESET%\ARC.exe" "ARC.exe.%RANDOM%.old" 2>nul

:: Run configuration if build directory is missing
if not exist "out\%PRESET%" (
    cmake --preset %PRESET%
)

cmake --build --preset %PRESET% -- -v
if errorlevel 1 (
    set "BUILD_STATUS=1"
    exit /b 1
)

:: Launch the freshly built binary unless the caller asked us not to.
if /i "%NOLAUNCH%"=="NOLAUNCH" exit /b 0
start "" .\out\%PRESET%\ARC.exe
exit /b 0
