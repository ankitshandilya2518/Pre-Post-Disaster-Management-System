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

rem 1. Check if git remote exists
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo [ACTION REQUIRED] No GitHub remote repository is currently linked.
    echo.
    echo Please enter your GitHub repository URL:
    echo (e.g. https://github.com/your-username/CSE471-DisasterVision.git)
    echo.
    set /p REPO_URL="Repository URL: "
    if "!REPO_URL!"=="" (
        echo [ERROR] No URL entered. Aborting.
        pause
        exit /b 1
    )
    git remote add origin !REPO_URL!
    echo [OK] Linked remote origin to: !REPO_URL!
    echo.
)

rem 2. Staging all files
echo [*] Staging project changes...
git add -A

rem 3. Ask for commit message or default to timestamp
set /p USER_MSG="Enter commit message (Press Enter for automatic timestamp): "
if "!USER_MSG!"=="" (
    set USER_MSG=Update: %date% %time%
)

echo [*] Committing: "!USER_MSG!"...
git commit -m "!USER_MSG!"

rem 4. Push to main
echo [*] Pushing to GitHub (origin main)...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================================
    echo [SUCCESS] Your project has been successfully pushed and updated on GitHub!
    echo ==============================================================================
) else (
    echo.
    echo ==============================================================================
    echo [NOTICE] If this is your first push, please complete GitHub sign-in in the 
    echo          browser popup, or verify your repository permissions.
    echo ==============================================================================
)

echo.
pause
