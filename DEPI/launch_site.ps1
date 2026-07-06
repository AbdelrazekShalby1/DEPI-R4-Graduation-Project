param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSCommandPath
if (-not $ProjectDir) {
    $ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$SiteUrl = "http://127.0.0.1:8000/"
$HealthUrl = "http://127.0.0.1:8000/health"
$Python = Join-Path $ProjectDir ".venv\Scripts\pythonw.exe"
$FallbackPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$ServerScript = Join-Path $ProjectDir "run_server.py"
$ErrorLog = Join-Path $ProjectDir "server.err"

function Show-StartupError {
    param([string]$Message)

    try {
        $shell = New-Object -ComObject WScript.Shell
        $shell.Popup($Message, 12, "AI Medical Center", 48) | Out-Null
    } catch {
        Write-Host $Message
    }

    if (Test-Path $ErrorLog) {
        Start-Process notepad.exe $ErrorLog
    }
}

function Test-SiteReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Path $Python)) {
    $Python = $FallbackPython
}

if (-not (Test-Path $Python)) {
    Show-StartupError "Python environment was not found in $ProjectDir\.venv."
    exit 1
}

if (-not (Test-Path $ServerScript)) {
    Show-StartupError "run_server.py was not found in $ProjectDir."
    exit 1
}

if (-not (Test-SiteReady)) {
    Start-Process `
        -FilePath $Python `
        -ArgumentList "run_server.py" `
        -WorkingDirectory $ProjectDir `
        -WindowStyle Hidden | Out-Null

    $isReady = $false
    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        Start-Sleep -Milliseconds 500
        if (Test-SiteReady) {
            $isReady = $true
            break
        }
    }

    if (-not $isReady) {
        Show-StartupError "AI Medical Center server did not start. Check server.err for details."
        exit 1
    }
}

if (-not $NoBrowser) {
    Start-Process $SiteUrl
}

exit 0
