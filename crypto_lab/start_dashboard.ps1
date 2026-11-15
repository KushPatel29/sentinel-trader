$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeFolder = Join-Path $PSScriptRoot 'runtime'
New-Item -ItemType Directory -Force -Path $runtimeFolder | Out-Null
try {
    $healthResponse = Invoke-WebRequest -Uri 'http://127.0.0.1:8502/_stcore/health' -TimeoutSec 2
    if ($healthResponse.StatusCode -eq 200) {
        Write-Output 'Dashboard already available at http://localhost:8502'
        exit 0
    }
} catch {}
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
$dashboardPath = Join-Path $PSScriptRoot 'dashboard.py'
$dashboardProcess = Start-Process -FilePath $pythonCommand -ArgumentList @('-m','streamlit','run',('"' + $dashboardPath + '"'),'--server.address','127.0.0.1','--server.port','8502','--server.headless','true','--browser.gatherUsageStats','false') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeFolder 'dashboard.stdout.log') -RedirectStandardError (Join-Path $runtimeFolder 'dashboard.stderr.log')
$dashboardProcess.Id | Set-Content -LiteralPath (Join-Path $runtimeFolder 'dashboard.pid')
Write-Output "Started dashboard PID $($dashboardProcess.Id): http://localhost:8502"
