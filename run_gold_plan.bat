@echo off
REM Runs the daily gold plan once on your Alpaca PAPER account (GLD): places, trails or closes orders.
REM Add --dry-run to see what it would do without sending orders.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
sentinel gold-plan %*
echo.
if "%SENTINEL_NO_PAUSE%"=="" pause
