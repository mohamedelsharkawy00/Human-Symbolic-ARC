@echo off
setlocal
cd /d "%~dp0.."

:: EVERY READOUT, WRITTEN DOWN: builds arc_probe (left out of the default build)
:: through scripts\build.bat and writes out\probe\<label>\logs\inference\<id>.txt
:: and summary.txt.
::   scripts\probe.bat <label> [id...]
:: Then compare two labels with scripts\readouts.py.
if "%~1"=="" (
    echo usage: scripts\probe.bat ^<label^> [id...]
    exit /b 2
)
call "%~dp0build.bat" x64-Release NOLAUNCH arc_probe
if errorlevel 1 exit /b 1
"out\x64-Release\arc_probe.exe" "out\probe\%~1" %2 %3 %4 %5 %6 %7 %8 %9
