@echo off
setlocal
cd /d "%~dp0.."

:: Single source of truth for building ARC. Callable by the profiling scripts
:: and scripts\said.bat.
::   %1  preset    - CMake preset name (default: x64-Release)
::   %2  NOLAUNCH  - if set to NOLAUNCH, build only; do not start ARC.exe
::                   (profilers launch the binary under their own collector)
::   %3  target    - build only this target (said.bat builds arc_said alone)
set "PRESET=%~1"
if not defined PRESET set "PRESET=x64-Release"
set "NOLAUNCH=%~2"
set "ONLY_TARGET="
if not "%~3"=="" set "ONLY_TARGET=--target %~3"

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
:: Launch the freshly built binary unless the caller asked us not to. Only
:: now, with the lock let go: started inside it, the app inherited the lock's
:: handle and held it for as long as it ran, so no build could run beside it.
if "%BUILD_STATUS%"=="0" if /i not "%NOLAUNCH%"=="NOLAUNCH" start "" .\out\%PRESET%\ARC.exe
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
:: ...but only when it is actually locked, i.e. a copy is running. Renaming
:: unconditionally left ninja with no ARC.exe every single time, so it redid the
:: LTCG link on builds where nothing had changed -- most of a no-op build, spent
:: producing the same bytes again (2026-09-09). The test is an append of nothing,
:: which opens the file for writing without touching its bytes and fails exactly
:: when the linker would.
:: arc_headless.exe too: a GUI benchmark runs it, one puzzle at a time.
for %%E in (ARC arc_headless) do (
    if exist "out\%PRESET%\%%E.exe" (
        2>nul ( >>"out\%PRESET%\%%E.exe" call ) || (
            ren "out\%PRESET%\%%E.exe" "%%E.exe.%RANDOM%.old" 2>nul
        )
    )
)

:: Run configuration if build directory is missing
if not exist "out\%PRESET%" (
    cmake --preset %PRESET%
)

cmake --build --preset %PRESET% %ONLY_TARGET% -- -v
if errorlevel 1 (
    set "BUILD_STATUS=1"
    exit /b 1
)

exit /b 0
