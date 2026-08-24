@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo   Local AI Document Translator & Reflow Publisher (Version 1.0)
echo   Local-first, Offline-capable Document Publishing System
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python installation
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=py
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set PYTHON_CMD=python
    ) else (
        echo [ERROR] Python was not found. Please install Python 3.11+ and add it to PATH.
        pause
        exit /b 1
    )
)

:: 2. Setup Virtual Environment
if not exist "venv\Scripts\python.exe" (
    echo [SETUP] Creating Python virtual environment in .\venv...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Build Frontend if dist not found
if not exist "frontend\dist\index.html" (
    echo [SETUP] Building frontend distribution...
    where npm >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        cd frontend
        call npm install --silent
        call npm run build
        cd ..
    ) else (
        echo [WARNING] npm not found. Running in API-only backend mode.
    )
)

:: 4. Launch Browser
echo [LAUNCH] Opening browser at http://127.0.0.1:8765...
start "" "http://127.0.0.1:8765"

:: 5. Start FastAPI Server
echo [RUNNING] Starting Local AI Document Publisher on http://127.0.0.1:8765
echo Press Ctrl+C in this terminal to stop the application.
echo.
set PYTHONPATH=%CD%\backend
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
pause
