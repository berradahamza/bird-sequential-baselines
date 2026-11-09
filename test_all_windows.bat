@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u run_campaign.py test %*
if errorlevel 1 goto fail
echo Scores: runs\summary\results_final.csv
pause
exit /b 0
:fail
echo Test cannot start until every requested run is trained.
pause
exit /b 1
