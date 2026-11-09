@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u run_campaign.py train %*
if errorlevel 1 goto fail
echo Training complete. Now run test_all_windows.bat to evaluate frozen test checkpoints.
pause
exit /b 0
:fail
echo Training stopped. Run the same command again to resume after a completed epoch.
pause
exit /b 1
