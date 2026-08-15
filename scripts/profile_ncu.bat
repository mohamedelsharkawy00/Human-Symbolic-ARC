@echo off
setlocal
cls
cd /d "%~dp0.."

:: Request Admin - NCU needs it to access GPU hardware performance counters
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [*] Requesting Administrative Privileges for Nsight Compute HW counters...
    powershell -Command "Start-Process cmd -ArgumentList '/k \"%~f0\" %*' -Verb RunAs -WindowStyle Maximized"
    exit /b
)

:: Profiling-specific output dir (build/out dirs are handled by build.bat)
if not exist "profiling\output" mkdir "profiling\output"

:: Terminate any profiler processes that would lock files / the GPU.
:: (build.bat terminates ARC.exe itself before the build.)
taskkill /F /IM ncu.exe /T 2>nul
taskkill /F /IM nv-nsight-cu.exe /T 2>nul
taskkill /F /IM nv-nsight-cu-cli.exe /T 2>nul

:: ===================================================================
:: This script intentionally does NOT run VTune. Nsight Compute and VTune's
:: vtss.sys driver cannot coexist: with vtss.sys loaded, NCU instrumenting
:: nvlddmkm.sys BSODs the machine (LIST_ENTRY corruption). NCU was removed
:: from profile.bat for exactly that reason. Keeping NCU standalone - no
:: VTune, no vtss.sys - is what makes it safe. Do NOT merge this back into
:: profile.bat.
:: ===================================================================

:: SAFETY: a recent profile.bat (VTune) run can leave vtss.sys loaded in the
:: kernel. If it is loaded when NCU instruments nvlddmkm.sys, the machine
:: BSODs. Detect and ABORT with guidance rather than auto-unloading - silently
:: unloading vtss breaks VTune's HW sampling until it is reinstalled. A reboot
:: clears the driver.
sc query vtss   2>nul | find "RUNNING" >nul
if not errorlevel 1 goto vtune_driver_loaded
sc query vtsspp 2>nul | find "RUNNING" >nul
if not errorlevel 1 goto vtune_driver_loaded
goto vtune_driver_clear
:vtune_driver_loaded
echo [!] FATAL: a VTune driver ^(vtss.sys / vtsspp.sys^) is currently loaded.
echo [!] Running Nsight Compute now risks a kernel BSOD ^(nvlddmkm LIST_ENTRY corruption^).
echo [!] Reboot to clear the driver, then run this script again.
exit /b 1
:vtune_driver_clear

:: Build via build.bat (single source of truth). NOLAUNCH: NCU launches ARC
:: itself under the collector below.
echo [1/2] Building ARC for Profiling - RelWithDebInfo
call "%~dp0build.bat" x64-RelWithDebInfo NOLAUNCH
if errorlevel 1 exit /b 1

set _NT_SYMBOL_PATH=srv*c:\symbols*https://msdl.microsoft.com/download/symbols;%CD%\out\x64-RelWithDebInfo

:: ===================================================================
:: Sampling mode. Default is STRIDED; pass "consecutive" to switch back.
::   strided     (default): every 1000th launch via NCU's --kernel-id
::               invocation-number regex (.*000$ => 1000,2000,...,10000),
::               spread across the run. Captures the DYNAMIC variance
::               (duration, memory) behind the 0.24-25.9 ms spread.
::   consecutive: 10 back-to-back launches after a 10000-launch warmup. Fine
::               for STATIC metrics (registers/occupancy are identical every
::               launch) but all 10 may come from one program phase.
:: Strided is default because the UAC relaunch above can drop a forwarded arg,
:: and strided is the more representative measurement anyway. The two modes
:: write SEPARATE report files, so results never overwrite each other.
:: ===================================================================
set "NCU_OUT=kernel_profile_strided"
set "NCU_CSV=ncu_kernel_strided.csv"
set "NCU_FILTER=--kernel-id ::dsl_exec_and_score_kernel_v2:.*000$ --launch-count 10"
set "NCU_MODE=strided (every 1000th launch, 10 samples)"
if /i "%~1"=="consecutive" (
    set "NCU_OUT=kernel_profile"
    set "NCU_CSV=ncu_kernel.csv"
    set "NCU_FILTER=--kernel-name dsl_exec_and_score_kernel_v2 --launch-skip 10000 --launch-count 10"
    set "NCU_MODE=consecutive (10 launches after 10000 warmup)"
)

:: ===================================================================
:: Locate ncu - prefer newest installed Nsight Compute
:: ===================================================================
set "NCU_EXE="
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\NVIDIA Corporation\Nsight Compute *" 2^>nul') do (
    if not defined NCU_EXE set "NCU_EXE=C:\Program Files\NVIDIA Corporation\%%D\ncu.bat"
)
if defined NCU_EXE if not exist "%NCU_EXE%" set "NCU_EXE="
if not defined NCU_EXE goto skip_ncu
echo [*] Using %NCU_EXE%
:: The GUI launcher sits next to ncu.bat as ncu-ui.bat (used to auto-open below).
set "NCU_UI=%NCU_EXE:ncu.bat=ncu-ui.bat%"

:: ===================================================================
:: Nsight Compute - single-kernel deep dive on the bottleneck kernel.
:: --set full adds Scheduler Statistics ("No Eligible %") + Warp State Statistics
:: (per-reason stall cycles: Long Scoreboard=memory, Barrier=__syncwarp, Wait=dep)
:: on top of detailed's occupancy/registers/memory. These explain SM-idle: whether
:: it's "too few warps" (occupancy) vs a specific stall reason. Slower replay than
:: detailed, fine for strided sampling. --import-source yes embeds source (-lineinfo).
:: --kill yes ends ARC.exe once profiling completes. NCU REPLAYS each kernel
:: ~17x to gather HW counters, so we sample a handful, not all ~5000. The
:: sampling filter and output names come from NCU_FILTER / NCU_OUT above.
:: ===================================================================
echo [2/2] Profiling CUDA kernel with Nsight Compute - %NCU_MODE%...
echo       ARC will appear FROZEN during NCU replay passes - this is normal.
echo       --kill yes ends ARC.exe automatically once profiling completes.
call "%NCU_EXE%" --set full --import-source yes %NCU_FILTER% --kill yes -f -o profiling\output\%NCU_OUT% -- .\out\x64-RelWithDebInfo\ARC.exe

:: Export a flat CSV summary alongside the .ncu-rep (open the latter in the GUI).
call "%NCU_EXE%" -i profiling\output\%NCU_OUT%.ncu-rep --csv > profiling\output\%NCU_CSV%
:: Also export the RAW page: every collected metric incl. the per-reason warp-stall
:: breakdown (warps_issue_stalled_*) and scheduler stats (No Eligible %, Eligible
:: Warps/Scheduler). The default --csv page omits these, so without this you'd have
:: to dig through the GUI's link-maze to find why the SM idles. Read ncu_raw.csv.
call "%NCU_EXE%" -i profiling\output\%NCU_OUT%.ncu-rep --csv --page raw > profiling\output\ncu_raw.csv
goto done_ncu

:skip_ncu
echo [2/2] SKIPPED - NVIDIA Nsight Compute not found under "C:\Program Files\NVIDIA Corporation\"

:done_ncu

echo.
echo Done! Results:
if exist "profiling\output\%NCU_OUT%.ncu-rep" echo   GPU kernel:     profiling\output\%NCU_OUT%.ncu-rep  (open in Nsight Compute UI)
if exist "profiling\output\%NCU_CSV%" echo   GPU kernel CSV: profiling\output\%NCU_CSV%

:: Auto-open the freshly captured report in the Nsight Compute GUI. Launch via
:: PowerShell Start-Process: plain `start` mis-parses the space in "Program
:: Files" ('C:\Program' is not recognized).
if defined NCU_UI if exist "profiling\output\%NCU_OUT%.ncu-rep" (
    echo [*] Opening report in Nsight Compute UI...
    powershell -NoProfile -Command "Start-Process -FilePath '%NCU_UI%' -ArgumentList '%CD%\profiling\output\%NCU_OUT%.ncu-rep'"
)

exit /b 0
