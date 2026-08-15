@echo off
setlocal
cls
cd /d "%~dp0.."

:: Request Admin required for ETW / HW sampling
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [*] Requesting Administrative Privileges for ETW collection...
    powershell -Command "Start-Process cmd -ArgumentList '/k \"%~f0\"' -Verb RunAs -WindowStyle Maximized"
    exit /b
)

:: Profiling-specific output dir (build/out dirs are handled by build.bat)
if not exist "profiling\output" mkdir "profiling\output"

:: Terminate any profiler processes that would lock files / the result dir.
:: (build.bat terminates ARC.exe itself before the build.)
taskkill /F /IM vtune.exe /T 2>nul
taskkill /F /IM vtune-gui.exe /T 2>nul
taskkill /F /IM amplxe-runss.exe /T 2>nul

:: Crash recovery: restore VTune's vulkan-1.dll if a prior run died mid-collect
set "VTUNE_BIN=C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64"
set "VTUNE_VK=%VTUNE_BIN%\vulkan-1.dll"
set "VTUNE_VK_BAK=%VTUNE_VK%.profile-bak"
call :recover_vtune_vulkan

:: Build via build.bat (single source of truth). NOLAUNCH: don't start ARC.exe;
:: VTune launches it under the collector below.
echo [1/4] Building ARC for Profiling - RelWithDebInfo
call "%~dp0build.bat" x64-RelWithDebInfo NOLAUNCH
if errorlevel 1 exit /b 1

set _NT_SYMBOL_PATH=srv*c:\symbols*https://msdl.microsoft.com/download/symbols;%CD%\out\x64-RelWithDebInfo

:: ===================================================================
:: Part 1: VTune CPU profiling
:: ===================================================================

set "VTUNE_EXE=%VTUNE_BIN%\vtune.exe"
if not exist "%VTUNE_EXE%" set "VTUNE_EXE="

if not defined VTUNE_EXE goto skip_vtune

:: VTune bundles its own vulkan-1.dll in bin64. Renaming it (to force ARC
:: onto the System32 loader) was found to break VTune's own collector init
:: with "Unauthorized control server connection" + corrupt trace. The
:: Windows DLL search order resolves System32 before VTune's bin64 anyway
:: since ARC.exe lives in out\x64-RelWithDebInfo, not in VTune's directory.
:: if exist "%VTUNE_VK%" ren "%VTUNE_VK%" "vulkan-1.dll.profile-bak"

echo [2/4] Collecting CPU profile - VTune hotspots
if exist "profiling\output\vtune_results" rmdir /s /q "profiling\output\vtune_results"
"%VTUNE_EXE%" -q -collect hotspots -knob sampling-mode=hw -knob enable-stack-collection=true -finalization-mode=full -search-dir "%VTUNE_BIN%" -duration=70 -data-limit=0 -result-dir profiling\output\vtune_results -- .\out\x64-RelWithDebInfo\ARC.exe 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -File profiling\filter_vtune.ps1
taskkill /F /IM ARC.exe /T 2>nul

echo [3/4] Exporting VTune data...
call :vtune_report function    cpu_stacks_vtune.csv
call :vtune_report source-line source_lines.csv
call :vtune_report thread      threads.csv

:: Restore VTune's bundled vulkan-1.dll (rename trick disabled above; this
:: is now only a safety net if a stale .profile-bak somehow exists from an
:: older run, which the recover_vtune_vulkan subroutine also handles).
if exist "%VTUNE_VK_BAK%" ren "%VTUNE_VK_BAK%" "vulkan-1.dll"
goto done_vtune

:skip_vtune
echo [2/4] SKIPPED - Intel VTune not found
echo [3/4] SKIPPED

:done_vtune

:: Nsight Compute (GPU kernel deep-dive) and its required vtss.sys
:: driver-unload step are removed. To re-enable, restore from git history.

:: ===================================================================
:: Part 2: Python analysis
:: ===================================================================

echo [4/4] Running Python analysis...
if exist "profiling\output\cpu_stacks_vtune.csv" (
    python profiling\analyze.py
) else (
    echo SKIPPED - No VTune data to analyze
)

echo.
echo Done! Results:
if exist "profiling\output\hotspots.txt" echo   CPU report:     profiling\output\hotspots.txt

exit /b 0

:: ===================================================================
:: Subroutines
:: ===================================================================

:: Export a VTune hotspots report as CSV. %1 = group-by, %2 = output filename.
:vtune_report
"%VTUNE_EXE%" -q -report hotspots -result-dir profiling\output\vtune_results -group-by %~1 -format csv -csv-delimiter comma -report-output profiling\output\%~2
exit /b 0

:: Restore VTune's vulkan-1.dll if a prior run left a .profile-bak behind.
:: Uses %VTUNE_VK% and %VTUNE_VK_BAK% set by the caller.
:recover_vtune_vulkan
if not exist "%VTUNE_VK_BAK%" exit /b 0
if exist "%VTUNE_VK%" goto :recover_vtune_vulkan_orphan
echo [*] Found stale vulkan-1.dll.profile-bak from a previous run - restoring...
ren "%VTUNE_VK_BAK%" "vulkan-1.dll"
exit /b 0
:recover_vtune_vulkan_orphan
echo [*] Removing orphan vulkan-1.dll.profile-bak (main DLL already present)...
del "%VTUNE_VK_BAK%"
exit /b 0
