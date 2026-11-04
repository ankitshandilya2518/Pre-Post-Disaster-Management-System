@echo off
setlocal
title DisasterVision - Export All Project Reports
cd /d "%~dp0"

echo ======================================================================
echo    DisasterVision: Generating All Official Reports into CSE471
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" export_all_reports.py
) else (
    python export_all_reports.py
)

echo.
echo ======================================================================
echo  [DONE] All reports have been exported directly into this folder!
echo ======================================================================
echo.
pause
