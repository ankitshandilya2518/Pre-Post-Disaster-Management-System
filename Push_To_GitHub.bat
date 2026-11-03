@echo off
setlocal enabledelayedexpansion
title DisasterVision - Quick Push To GitHub
color 0A
cls

cd /d "%~dp0"

echo ==============================================================================
echo       DISASTERVISION: PRE/POST-DISASTER BUILDING-DAMAGE CLASSIFIER            
echo                     One-Click Push to GitHub Remote                           
echo ==============================================================================
echo.
echo Remote Repository:
echo   https://github.com/ankitshandilya2518/Pre-Post-Disaster-Management-System
echo.

rem 1. Verify remote is set
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    git remote add origin https://github.com/ankitshandilya2518/Pre-Post-Disaster-Management-System.git
)

rem 2. Stage all modifications
echo [*] Staging files (git add -A)...
git add -A

rem 3. Check for staged changes
git diff --cached --quiet
if %errorlevel% neq 0 (
    echo [*] Changes detected.
    echo.
    set /p USER_MSG="Enter commit description (Press Enter for automatic timestamp): "
    if "!USER_MSG!"=="" (
        set USER_MSG=Update: %date% %time%
    )
    echo.
    echo [*] Committing: "!USER_MSG!"...
    git commit -m "!USER_MSG!"
) else (
    echo [*] No new uncommitted file changes detected.
)

rem 4. Push to GitHub
echo.
echo [*] Pushing to GitHub (origin main)...
git push origin main

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================================
    echo [SUCCESS] GitHub is now completely up to date!
    echo Check your repository:
    echo https://github.com/ankitshandilya2518/Pre-Post-Disaster-Management-System
    echo ==============================================================================
) else (
    echo.
    echo ==============================================================================
    echo [NOTICE] Push encountered an issue. Please check your internet connection 
    echo          or GitHub sign-in permissions.
    echo ==============================================================================
)

echo.
pause
