@echo off
setlocal
cd /d "%~dp0.."

:: EVERY READOUT, WRITTEN DOWN: builds arc_probe (left out of the default build)
:: and writes out\probe\<label>\logs\inference\<id>.txt and summary.txt.
::   scripts\probe.bat <label> [id...]
:: Then compare two labels with scripts\readouts.py.
if "%~1"=="" (
    echo usage: scripts\probe.bat ^<label^> [id...]
    exit /b 2
)
if not defined DevEnvDir set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
if not defined DevEnvDir (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
)
if not exist "out\x64-Release" cmake --preset x64-Release
cmake --build --preset x64-Release --target arc_probe
if errorlevel 1 exit /b 1
"out\x64-Release\arc_probe.exe" "out\probe\%~1" %2 %3 %4 %5 %6 %7 %8 %9
