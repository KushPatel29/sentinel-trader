@echo off
REM Schedules run_gold_plan.bat every weekday at 08:00 (your local time), after the US close and before the next open.
cd /d "%~dp0"
schtasks /Create /F /SC WEEKLY /D MON,TUE,WED,THU,FRI,SAT /ST 08:00 /TN "Sentinel Gold Plan" /TR "cmd /c set SENTINEL_NO_PAUSE=1 && \"%~dp0run_gold_plan.bat\" >> \"%~dp0gold_plan_log.txt\" 2>&1"
echo Scheduled. Remove with: schtasks /Delete /TN "Sentinel Gold Plan" /F
pause
