$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'runtime\STOP')) {
    throw 'A stop marker exists. Remove crypto_lab\runtime\STOP deliberately before resuming.'
}
$runtimeFolder = Join-Path $PSScriptRoot 'runtime'
New-Item -ItemType Directory -Force -Path $runtimeFolder | Out-Null
$statusPath = Join-Path $runtimeFolder 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $workerStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    $workerExistingProcess = if ($workerStatus.pid) { Get-CimInstance Win32_Process -Filter "ProcessId = $($workerStatus.pid)" -ErrorAction SilentlyContinue } else { $null }
    if ($workerExistingProcess -and $workerExistingProcess.CommandLine -like '*sentinel_crypto.cli*paper*') {
        Write-Output "Worker already has a process: $($workerStatus.pid)"
        exit 0
    }
}
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
$workerProcess = Start-Process -FilePath $pythonCommand -ArgumentList @('-u','-m','sentinel_crypto.cli','paper') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeFolder 'worker.stdout.log') -RedirectStandardError (Join-Path $runtimeFolder 'worker.stderr.log')
$workerProcess.Id | Set-Content -LiteralPath (Join-Path $runtimeFolder 'worker.pid')
Write-Output "Started forward paper worker PID $($workerProcess.Id)"
