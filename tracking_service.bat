@echo off
:: Check if we're starting or stopping the service
if "%1"=="stop" (
    echo Stopping tracking service...
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq FileTracking"
    exit /b
)

:: Start the tracking service in a minimized window
start "FileTracking" /min python track_updates.py --last-run "2025-08-07 06:00:00"
echo Tracking service started in background

:: Wait a moment and show the tracking file path
timeout /t 2 /nobreak >nul
for /f "delims=" %%i in ('dir /b /a-d updated_files_*.txt') do echo Current tracking file: %%i
