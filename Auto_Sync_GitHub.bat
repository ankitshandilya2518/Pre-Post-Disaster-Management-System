@echo off
setlocal enabledelayedexpansion
title DisasterVision - GitHub Real-Time Auto-Sync
color 0B
cls

cd /d "%~dp0"

echo ==============================================================================
echo       DISASTERVISION: PRE/POST-DISASTER BUILDING-DAMAGE CLASSIFIER            
echo             Automatic GitHub Real-Time Synchronizer Launcher                 
echo ==============================================================================
echo.
echo [*] Launching background filesystem monitor...
echo [*] Any file edits will be automatically committed and pushed to GitHub.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\auto_sync.ps1"

echo.
pause
