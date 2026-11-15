param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$challengerFolder = Join-Path $PSScriptRoot 'runtime\challengers'
if (Test-Path -LiteralPath (Join-Path $challengerFolder 'STOP')) { throw 'Challenger STOP marker exists. Remove it deliberately before resuming.' }
if (-not (Test-Path -LiteralPath (Join-Path $challengerFolder 'manifest.json'))) { throw 'Prepare frozen forward models in the neural environment first.' }
$challengerStatusPath = Join-Path $challengerFolder 'status.json'
if (Test-Path -LiteralPath $challengerStatusPath) {
    $challengerStatus = Get-Content -LiteralPath $challengerStatusPath -Raw | ConvertFrom-Json
    $challengerExisting = if ($challengerStatus.pid) { Get-CimInstance Win32_Process -Filter "ProcessId = $($challengerStatus.pid)" -ErrorAction SilentlyContinue } else { $null }
    if ($challengerExisting -and $challengerExisting.CommandLine -like '*sentinel_crypto.cli*challenger-watch*') {
        Write-Output "Challenger worker already running: $($challengerStatus.pid)"
        exit 0
    }
}
$challengerPython = (Get-Command $Python -ErrorAction Stop).Source
$challengerProcess = Start-Process -FilePath $challengerPython -ArgumentList @('-u','-m','sentinel_crypto.cli','challenger-watch') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $challengerFolder 'worker.stdout.log') -RedirectStandardError (Join-Path $challengerFolder 'worker.stderr.log')
$challengerProcess.Id | Set-Content -LiteralPath (Join-Path $challengerFolder 'worker.pid')
Write-Output "Started real-data paper challenger worker PID $($challengerProcess.Id)"
