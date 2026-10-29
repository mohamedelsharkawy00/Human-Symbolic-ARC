@echo off
setlocal
cd /d "%~dp0.."

:: EVERY PUZZLE READ, SCORED AND WRITTEN DOWN, in one run: builds arc_headless
:: through scripts\build.bat and runs it --said, writing said.txt, tally.txt and
:: logs\inference\<id>.txt into out\x64-Release\, or out\<label>\ when named.
::   scripts\said.bat [--as <label>] [--solved [label]] [id...]
:: Then compare two runs with scripts\readouts.py (the unnamed one is x64-Release).
call "%~dp0build.bat" x64-Release NOLAUNCH arc_headless
if errorlevel 1 exit /b 1
"out\x64-Release\arc_headless.exe" --said %*
