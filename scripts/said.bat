@echo off
setlocal
cd /d "%~dp0.."

:: EVERY PUZZLE READ, SCORED AND WRITTEN DOWN, in one run: builds arc_said
:: through scripts\build.bat and runs it, writing out\<label>\said.txt,
:: tally.txt and logs\inference\<id>.txt.
::   scripts\said.bat <label> [id...]
:: Then compare two labels with scripts\readouts.py.
if "%~1"=="" (
    echo usage: scripts\said.bat ^<label^> [id...]
    exit /b 2
)
call "%~dp0build.bat" x64-Release NOLAUNCH arc_said
if errorlevel 1 exit /b 1
"out\x64-Release\arc_said.exe" %*
