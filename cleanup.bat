@echo off
echo Cleaning up temporary files...

:: Remove Python cache files
del /s /q *.pyc
del /s /q *.pyo
del /s /q *.pyd
rd /s /q __pycache__ 2>nul

:: Remove build and distribution directories
rd /s /q build 2>nul
rd /s /q dist 2>nul
rd /s /q *.egg-info 2>nul

:: Remove IDE specific files
del /s /q .coverage 2>nul
rd /s /q .pytest_cache 2>nul
rd /s /q htmlcov 2>nul

:: Remove temporary files
del /s /q *.log 2>nul
del /s /q *.bak 2>nul
del /s /q *~ 2>nul

echo Cleanup complete!
