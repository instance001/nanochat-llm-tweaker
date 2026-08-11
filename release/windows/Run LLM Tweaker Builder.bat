@echo off
setlocal

set "RELEASE_DIR=%~dp0"
set "APP_ROOT=%RELEASE_DIR%..\.."
set "LAUNCHER=%APP_ROOT%\nanochat-master\launch-local-builder.ps1"

if not exist "%LAUNCHER%" (
    echo Could not find the LLM Tweaker launcher:
    echo "%LAUNCHER%"
    echo.
    echo Make sure this file is still inside release\windows in the llm-tweaker repo.
    pause
    exit /b 1
)

cd /d "%APP_ROOT%\nanochat-master"

echo Starting LLM Tweaker Builder...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Port 8000 -RuntimePort 8091 %*

if errorlevel 1 (
    echo.
    echo LLM Tweaker Builder stopped with an error.
    pause
)
