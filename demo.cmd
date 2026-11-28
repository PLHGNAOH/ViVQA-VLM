@echo off
REM ViVQA-VLM demo.
REM   .\demo       = offline, parallel OCR+model startup (nhanh)
REM   .\demo safe  = offline, sequential startup — chậm hơn ~20 s, ít tranh RAM
REM   .\demo share = tao link gradio.live
REM   .\demo nhom  = chi Final Model, khong co bo chon
REM Bang Vi du tat mac dinh; bat lai: chay run_local_windows.ps1 -Offline -Examples
if /I "%~1"=="share" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\prototype\run_local_windows.ps1" -Share
) else if /I "%~1"=="safe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\prototype\run_local_windows.ps1" -Offline -SequentialStart
) else if /I "%~1"=="nhom" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\prototype\run_local_windows.ps1" -Offline -NoCompare
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\prototype\run_local_windows.ps1" -Offline
)
