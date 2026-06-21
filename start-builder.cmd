@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0nanochat-master\launch-local-builder.ps1" -Port 8765 -RuntimePort 8766 -NoBrowser %*
