# ==============================================================================
# DISASTERVISION: AUTOMATIC GITHUB REAL-TIME SYNCHRONIZATION ENGINE
# ==============================================================================
# Monitors workspace files for modifications and automatically commits & pushes
# changes to your remote GitHub repository in real time.

param (
    [int]$DebounceSeconds = 3
)

$Host.UI.RawUI.WindowTitle = "DisasterVision - GitHub Auto-Sync Engine (Active)"
Clear-Host

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "           DISASTERVISION: AUTOMATIC GITHUB REPOSITORY SYNC                   " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

$repoRoot = (Get-Item -Path $PSScriptRoot\..).FullName
Set-Location $repoRoot

# 1. Verify Git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "[!] Git repository not found. Initializing..." -ForegroundColor Yellow
    git init -b main
}

# 2. Check Remote URL
$remoteUrl = git remote get-url origin 2>$null
if (-not $remoteUrl) {
    Write-Host "==============================================================================" -ForegroundColor Yellow
    Write-Host " [ACTION REQUIRED] GitHub Remote Repository Not Yet Configured                 " -ForegroundColor Yellow
    Write-Host "==============================================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please enter your GitHub repository URL (HTTPS or SSH):" -ForegroundColor White
    Write-Host "Example: https://github.com/your-username/CSE471-DisasterVision.git" -ForegroundColor DarkGray
    Write-Host ""
    $inputUrl = Read-Host "GitHub Repository URL"
    if ([string]::IsNullOrWhiteSpace($inputUrl)) {
        Write-Host "[ERROR] No URL provided. Cannot synchronize with GitHub." -ForegroundColor Red
        Write-Host "Press any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
    git remote add origin $inputUrl.Trim()
    $remoteUrl = $inputUrl.Trim()
    Write-Host "[OK] Added remote 'origin': $remoteUrl" -ForegroundColor Green
    Write-Host ""
}

Write-Host "[*] Active Remote: $remoteUrl" -ForegroundColor Green
Write-Host "[*] Branch: main" -ForegroundColor Green
Write-Host "[*] Watching directory: $repoRoot" -ForegroundColor Green
Write-Host "[*] Debounce interval: $DebounceSeconds seconds" -ForegroundColor Gray
Write-Host ""
Write-Host "==============================================================================" -ForegroundColor DarkGray
Write-Host "Auto-Sync is now RUNNING. Whenever you edit or save any file," -ForegroundColor White
Write-Host "it will be automatically staged, committed, and pushed to GitHub." -ForegroundColor White
Write-Host "Keep this window open in the background. Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host "==============================================================================" -ForegroundColor DarkGray
Write-Host ""

# Function to perform Git Push
function Sync-ToGitHub {
    param([string]$TriggerReason)
    
    try {
        $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "[$timestamp] Detected change: $TriggerReason" -ForegroundColor Yellow
        Write-Host "[$timestamp] Staging files..." -ForegroundColor Gray
        
        git add -A
        
        # Check if there are staged changes
        git diff --cached --quiet
        if ($LASTEXITCODE -ne 0) {
            $commitMsg = "Auto-update: $timestamp ($TriggerReason)"
            Write-Host "[$timestamp] Committing: '$commitMsg'..." -ForegroundColor Cyan
            git commit -m "$commitMsg" | Out-Null
            
            Write-Host "[$timestamp] Pushing to GitHub (origin main)..." -ForegroundColor Cyan
            $pushOut = git push -u origin main 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[$timestamp] [SUCCESS] Automatically synced to GitHub!" -ForegroundColor Green
            } else {
                Write-Host "[$timestamp] [PUSH NOTICE] $pushOut" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[$timestamp] Workspace already in sync (no new changes)." -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "[!] Error during sync: $_" -ForegroundColor Red
    }
    Write-Host ""
}

# Initial synchronization on startup
Sync-ToGitHub -TriggerReason "Auto-Sync Engine Startup"

# Configure FileSystemWatcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $repoRoot
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, LastWrite, DirectoryName'

# Ignored path patterns (regex)
$ignoredPatterns = @(
    "\\\.git\\",
    "\\\.venv\\",
    "\\__pycache__\\",
    "\\\.matplotlib_cache\\",
    "\\scratch\\",
    "\\\.tmp$"
)

$lastChangeTime = [DateTime]::MinValue
$pendingReason = ""

$action = {
    param($source, $eventArgs)
    $path = $eventArgs.FullPath
    $changeType = $eventArgs.ChangeType
    $name = $eventArgs.Name

    foreach ($pat in $ignoredPatterns) {
        if ($path -match $pat) { return }
    }

    # Record event
    $script:lastChangeTime = [DateTime]::Now
    $script:pendingReason = "$changeType $name"
}

Register-ObjectEvent $watcher "Changed" -Action $action | Out-Null
Register-ObjectEvent $watcher "Created" -Action $action | Out-Null
Register-ObjectEvent $watcher "Deleted" -Action $action | Out-Null
Register-ObjectEvent $watcher "Renamed" -Action $action | Out-Null

# Main Loop with Debouncing
try {
    while ($true) {
        Start-Sleep -Milliseconds 500
        if ($lastChangeTime -ne [DateTime]::MinValue) {
            $elapsed = ([DateTime]::Now - $lastChangeTime).TotalSeconds
            if ($elapsed -ge $DebounceSeconds) {
                $reasonToSync = $pendingReason
                $lastChangeTime = [DateTime]::MinValue
                $pendingReason = ""
                Sync-ToGitHub -TriggerReason $reasonToSync
            }
        }
    }
} finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Get-EventSubscriber | Unregister-Event
    Write-Host "[*] Auto-Sync Engine stopped." -ForegroundColor Yellow
}
