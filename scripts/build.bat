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

:: Ensure MSVC environment if not already present
if not defined DevEnvDir (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
)

:: Auto-terminate any running instances to prevent link errors
taskkill /F /IM ARC.exe /T 2>nul

:: Run configuration if build directory is missing
if not exist "out\%PRESET%" (
    cmake --preset %PRESET%
)

cmake --build --preset %PRESET% -- -v
if errorlevel 1 exit /b 1

:: Launch the freshly built binary unless the caller asked us not to.
if /i "%NOLAUNCH%"=="NOLAUNCH" exit /b 0
start "" .\out\%PRESET%\ARC.exe
exit /b 0
