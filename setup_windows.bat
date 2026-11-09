@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Use Python 3.10 to 3.13 (3.12 recommended)'"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" run.py doctor --device cpu
if errorlevel 1 goto fail
echo Installation complete. Run train_all_windows.bat next.
pause
exit /b 0
:fail
echo Installation failed. Read the error above. Python 3.12 is recommended.
pause
exit /b 1
