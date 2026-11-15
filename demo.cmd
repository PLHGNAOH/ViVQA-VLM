@echo off
REM ViVQA-VLM demo.  .\demo = offline, tu mo trinh duyet.  .\demo share = tao link gradio.live
if /I "%~1"=="share" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\prototype\run_local_windows.ps1" -Share
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\prototype\run_local_windows.ps1" -Offline
)