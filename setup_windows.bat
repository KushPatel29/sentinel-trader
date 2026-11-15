@echo off
REM One-time setup on Windows: creates a virtual environment and installs Sentinel.
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
if not exist .venv (%PY% -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1))
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dashboard]" || (echo Install failed & pause & exit /b 1)
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Notepad will open .env: paste your Alpaca PAPER keys, save, close,
echo then double-click start_bot.bat
notepad .env
pause
