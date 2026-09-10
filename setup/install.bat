@echo off
REM setup\install.bat for snomed_methods
REM Installs dependencies in a virtual environment and registers it as a Jupyter kernel

setlocal enabledelayedexpansion

REM Default values
set "VENV_NAME=%~1"
if "%VENV_NAME%"=="" set VENV_NAME=snomed_methods_env

set "INSTALL_MODE=%~2"
if "%INSTALL_MODE%"=="" set INSTALL_MODE=

echo === SNOMED Methods Installation ===
echo Virtual environment: %VENV_NAME%
echo Install mode: %INSTALL_MODE%

REM Find Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found in PATH
    exit /b 1
)

for /f "tokens=*" %%i in ('where python') do set PYTHON_CMD=%%i
echo Using Python: %PYTHON_CMD%

REM Check if virtual environment already exists
if exist "%VENV_NAME%" (
    echo Virtual environment already exists: %VENV_NAME%
) else (
    echo Creating virtual environment...
    call %PYTHON_CMD% -m venv "%VENV_NAME%"
)

REM Activate virtual environment
call "%VENV_NAME%\Scripts\activate.bat"

REM Upgrade base packages
echo Installing dependencies...
pip install --upgrade pip setuptools wheel

REM Install the package in editable mode
if "%INSTALL_MODE%"=="dev" (
    REM Install with development dependencies
    pip install -e ".[dev,medcat]"
) else (
    REM Install with production dependencies only
    pip install -e "."
)

REM Register as Jupyter kernel
echo Registering as Jupyter kernel...
python -m ipykernel install --user --name "%VENV_NAME%" --display-name "Python (%VENV_NAME%)"

REM Verify installation
echo.
echo === Installation Complete ===
echo Virtual environment: %VENV_NAME%
echo.
echo To use the virtual environment:
echo   call "%VENV_NAME%\Scripts\activate.bat"
echo.
echo Installed kernel specs:
python -m jupyter kernelspec list
