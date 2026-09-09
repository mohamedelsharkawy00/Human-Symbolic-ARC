@echo off
setlocal
cd /d "%~dp0.."
if not exist "out" mkdir "out"
set "PROFILE_RAN="
9>"out\build.lock" call :profile %*
if not defined PROFILE_RAN (
    echo Another build is already running in this folder. Wait for it to finish.
    exit /b 1
)
exit /b %PROFILE_STATUS%

:profile
set "PROFILE_RAN=1"
python "scripts\profile_build.py" %*
set "PROFILE_STATUS=%ERRORLEVEL%"
exit /b %PROFILE_STATUS%
