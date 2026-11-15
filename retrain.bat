@echo off
REM Weekly refit (schedule with Windows Task Scheduler, e.g. Saturdays). Restart the bot afterwards.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
sentinel fetch && sentinel train
