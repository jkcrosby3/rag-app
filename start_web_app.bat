@echo off
REM Batch file to start the RAG web application

echo Starting RAG Web Application...
echo ===================================

echo Available UIs:
echo 1. Unified Interface (default) - Combined query and document management
echo 2. Query Interface - Just the query interface
echo 3. Document Management - Just the document management UI
echo 4. Flask Web App - Traditional web interface

echo.
set /p choice="Choose UI [1-4] (default: 1): "

if "%choice%"=="" set choice=1
if "%choice%"=="1" set UI=unified
if "%choice%"=="2" set UI=streamlit
if "%choice%"=="3" set UI=gradio
if "%choice%"=="4" set UI=flask

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating Python virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated.
)

REM Install the package in development mode
echo Installing package in development mode...
pip install -e .

REM Run the startup script with the selected UI
echo Starting the %UI% web application...
python start_web_app.py --ui %UI%

REM Pause to see any error messages
pause
