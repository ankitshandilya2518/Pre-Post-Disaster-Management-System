@echo off
setlocal enabledelayedexpansion
title DisasterVision - Building Damage Classifier Dashboard
color 0B
cls

echo ==============================================================================
echo       DISASTERVISION: PRE/POST-DISASTER BUILDING-DAMAGE CLASSIFIER            
echo             CSE471 Final Year Project - Computer Vision Dashboard             
echo ==============================================================================
echo.
echo [*] Initializing application environment...
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\streamlit.exe!
    echo Please make sure you are in the project folder.
    pause
    exit /b 1
)

echo [*] Starting Streamlit Server...
echo [*] Your default web browser will open automatically at http://localhost:8501
echo.
echo ------------------------------------------------------------------------------
echo Keep this window open while using the dashboard. Press Ctrl+C to stop.
echo ------------------------------------------------------------------------------
echo.

rem Wait 2 seconds and launch browser just in case
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8501"

rem Run Streamlit
".\.venv\Scripts\streamlit.exe" run app.py --server.port 8501

echo.
echo Dashboard server stopped.
pause
