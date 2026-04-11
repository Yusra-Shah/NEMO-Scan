@echo off
echo ============================================
echo  NEMO Scan - Environment Setup (Windows)
echo ============================================

:: Step 1: Create virtual environment
echo.
echo [1/4] Creating virtual environment...
python -m venv nemo_env
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.12 is installed.
    pause
    exit /b 1
)
echo Done.

:: Step 2: Activate it
echo.
echo [2/4] Activating environment...
call nemo_env\Scripts\activate.bat

:: Step 3: Upgrade pip
echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

:: Step 4: Install all dependencies
echo.
echo [4/4] Installing all libraries (this will take 5-10 minutes)...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup Complete. Verifying imports...
echo ============================================
python verify_env.py

pause