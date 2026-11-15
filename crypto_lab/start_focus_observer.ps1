$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$focusFolder = Join-Path $PSScriptRoot 'runtime\focus'
if (Test-Path -LiteralPath (Join-Path $focusFolder 'STOP')) { throw 'Focus STOP marker exists. Remove it deliberately before resuming.' }
$statusFile = Join-Path $focusFolder 'status.json'
if (Test-Path -LiteralPath $statusFile) {
    $focusStatus = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
    $focusExistingProcess = if ($focusStatus.pid) { Get-CimInstance Win32_Process -Filter "ProcessId = $($focusStatus.pid)" -ErrorAction SilentlyContinue } else { $null }
    if ($focusExistingProcess -and $focusExistingProcess.CommandLine -like '*sentinel_crypto.cli*focus-watch*') {
        Write-Output "Focus observer already running: $($focusStatus.pid)"
        exit 0
    }
}
New-Item -ItemType Directory -Force -Path $focusFolder | Out-Null
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
$focusProcess = Start-Process -FilePath $pythonCommand -ArgumentList @('-u','-m','sentinel_crypto.cli','focus-watch') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $focusFolder 'observer.stdout.log') -RedirectStandardError (Join-Path $focusFolder 'observer.stderr.log')
$focusProcess.Id | Set-Content -LiteralPath (Join-Path $focusFolder 'observer.pid')
Write-Output "Started prospective focus observer PID $($focusProcess.Id)"
