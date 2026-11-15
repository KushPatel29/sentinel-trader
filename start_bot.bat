@echo off
REM Starts the paper-trading bot. First run downloads ~2 years of data and trains the model.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not exist models\model_bundle.pkl (
  echo First run: downloading history from Alpaca and training the model, this takes a few minutes...
  sentinel fetch || (pause & exit /b 1)
  sentinel train || (pause & exit /b 1)
)
echo Starting Sentinel in PAPER mode. Close this window or press Ctrl+C to stop.
echo To flatten everything and stop: create a file named KILL in this folder.
sentinel live
pause
