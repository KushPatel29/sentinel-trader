@echo off
cd /d "%~dp0crypto_lab"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0crypto_lab\start_paper.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0crypto_lab\start_dashboard.ps1"
start "" http://localhost:8502
