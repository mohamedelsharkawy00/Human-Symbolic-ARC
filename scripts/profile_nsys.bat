@echo off
setlocal
cls
cd /d "%~dp0.."

:: Request Admin required for nsys CPU sampling / context switches
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [*] Requesting Administrative Privileges for nsys CPU sampling...
    powershell -Command "Start-Process cmd -ArgumentList '/k \"%~f0\"' -Verb RunAs -WindowStyle Maximized"
    exit /b
)

:: Profiling-specific output dir (build/out dirs are handled by build.bat)
if not exist "profiling\output" mkdir "profiling\output"

:: Terminate any profiler processes that would lock files / the GPU.
:: (build.bat terminates ARC.exe itself before the build.)
taskkill /F /IM nsys.exe /T 2>nul
taskkill /F /IM nsight-sys.exe /T 2>nul
taskkill /F /IM QuadDaemon.exe /T 2>nul

:: Build via build.bat (single source of truth). NOLAUNCH: don't start ARC.exe;
:: nsys launches it under the collector below.
echo [1/3] Building ARC for Profiling - RelWithDebInfo
call "%~dp0build.bat" x64-RelWithDebInfo NOLAUNCH
if errorlevel 1 exit /b 1

set _NT_SYMBOL_PATH=srv*c:\symbols*https://msdl.microsoft.com/download/symbols;%CD%\out\x64-RelWithDebInfo

:: ===================================================================
:: Locate nsys - prefer newest installed version
:: ===================================================================
set "NSYS_EXE="
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\NVIDIA Corporation\Nsight Systems *" 2^>nul') do (
    if not defined NSYS_EXE set "NSYS_EXE=C:\Program Files\NVIDIA Corporation\%%D\target-windows-x64\nsys.exe"
)
if not defined NSYS_EXE goto skip_nsys
echo [*] Using %NSYS_EXE%

:: ===================================================================
:: Collect nsys timeline (70s — matches profile.bat VTune window so the
:: two reports describe the same workload).
:: --duration + --kill is unreliable on Windows with CUDA; we run for the
:: window then taskkill below as a safety net.
:: --cpuctxsw=true reveals when CPU threads are *blocked* (the question we
:: actually need to answer: is the CPU idle waiting on GPU sync?).
:: --trace adds Windows GPU scheduler (wddm), Vulkan, and NVTX on top of CUDA.
:: (osrt is Linux-only; on Windows, wddm + cpuctxsw cover the equivalent
:: visibility into thread-blocked vs GPU-queue states.)
:: ===================================================================
echo [2/3] Collecting nsys timeline (70s)...
"%NSYS_EXE%" profile ^
    --duration=70 ^
    --trace=cuda,nvtx,vulkan,wddm ^
    --sample=cpu ^
    --cpuctxsw=process-tree ^
    --output=profiling/output/arc_timeline ^
    --force-overwrite=true ^
    .\out\x64-RelWithDebInfo\ARC.exe
taskkill /F /IM ARC.exe /T 2>nul

echo [3/3] Exporting nsys stats...
:: cuda_api_sum/gpu_kern_sum/mem_*_sum are all AGGREGATE summaries. The
:: per-call cuda_api_trace report (thousands of rows) was removed — it buried
:: the summaries and is what to open the .nsys-rep timeline in the GUI for.
"%NSYS_EXE%" stats --force-export=true --report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum profiling/output/arc_timeline.nsys-rep
goto done_nsys

:skip_nsys
echo [2/3] SKIPPED - Nsight Systems not found under "C:\Program Files\NVIDIA Corporation\"
echo [3/3] SKIPPED

:done_nsys

echo.
echo Done! Results:
if exist "profiling\output\arc_timeline.nsys-rep" echo   Timeline: profiling\output\arc_timeline.nsys-rep  (open in Nsight Systems UI)

exit /b 0
