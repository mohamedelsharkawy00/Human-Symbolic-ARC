@echo off
setlocal
cd /d "%~dp0.."

:: CPU profile of dataset inference and DSL replay, through normal completion.
:: VTune attaches once the puzzles are loaded, so the profile leaves startup out.
::   scripts\profile.bat [id...]   the puzzles named, else every training puzzle.
set "VTUNE_EXE=C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
if not exist "%VTUNE_EXE%" (
    echo Intel VTune was not found.
    exit /b 1
)
call "%~dp0build.bat" x64-RelWithDebInfo NOLAUNCH
if errorlevel 1 exit /b 1

:: One puzzle at a time, so the profile reads like every one before it
:: (arc_said otherwise reads a puzzle on every CPU thread).
set "ARC_THREADS=1"

:: Keep each run separate, including score output.
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss-fff"') do set "PROFILE_OUT=profiling\output\cpu-%%T"
mkdir "%PROFILE_OUT%"
if errorlevel 1 exit /b 1
echo Collecting full dataset CPU profile in %PROFILE_OUT%...
powershell -NoProfile -ExecutionPolicy Bypass -File profiling\collect_cpu.ps1 -Vtune "%VTUNE_EXE%" -OutputDirectory "%PROFILE_OUT%" %*
if errorlevel 1 exit /b 1
call :report function cpu_stacks_vtune.csv
if errorlevel 1 exit /b 1
call :report source-line source_lines.csv
if errorlevel 1 exit /b 1
call :report thread threads.csv
if errorlevel 1 exit /b 1
"%VTUNE_EXE%" -q -report callstacks -result-dir "%PROFILE_OUT%\vtune_results" -format csv -csv-delimiter comma -report-output "%PROFILE_OUT%\callstacks.csv"
if errorlevel 1 exit /b 1
python profiling\analyze.py "%PROFILE_OUT%"
exit /b %errorlevel%

:report
"%VTUNE_EXE%" -q -report hotspots -result-dir "%PROFILE_OUT%\vtune_results" -group-by %~1 -format csv -csv-delimiter comma -report-output "%PROFILE_OUT%\%~2"
exit /b %errorlevel%
