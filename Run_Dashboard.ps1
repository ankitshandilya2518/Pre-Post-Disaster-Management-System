Set-Location -Path $PSScriptRoot
Write-Host "Launching DisasterVision Dashboard..." -ForegroundColor Cyan
Start-Process "http://localhost:8501"
& ".\.venv\Scripts\streamlit.exe" run app.py --server.port 8501
