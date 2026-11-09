@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
echo Installing PyTorch 2.8 CUDA 12.6. Requires a compatible NVIDIA GPU and driver.
".venv\Scripts\python.exe" -m pip install --force-reinstall "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto fail
".venv\Scripts\python.exe" run.py doctor --device cuda
if errorlevel 1 goto fail
echo CUDA ready. Training scripts will select it automatically.
pause
exit /b 0
:fail
echo CUDA setup failed. You can still train with --device cpu. See README.md.
pause
exit /b 1
