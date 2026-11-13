# =============================================================================
# run_local_windows.ps1 — chạy prototype Gradio trên laptop Windows (RTX 4 GB)
#
# Mục đích: demo bảo vệ — Qwen2.5-VL-3B 4-bit + adapter SH trên GPU, OCR trên CPU
#           (giữ nguyên prompt / tham số sinh / min-max pixels / engine OCR như eval).
#
# Lệnh chạy (từ bất kỳ thư mục nào):
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Share
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Mock
#
# Khởi động đo ở B5 (2026-11-13, RTX 3050 Ti 4 GB): ~78 s (OCR CPU ~31 s + Qwen/adapter).
# Máy phải bật suốt lúc demo. Hôm bảo vệ dùng -Offline, mở http://127.0.0.1:7860
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Share,
    [switch]$Offline,
    [switch]$Mock,
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

$Repo = "D:\FPT_university\termno9\ViVQA-VLM"
$EnvDir = "D:\envs\vivqa-demo"
$HfHome = "D:\hf_cache"
$Adapter = "D:\vivqa_models\SH\epoch_1"
$ExpectSha = "c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354"
$ExamplesJson = "D:\vivqa_models\measure_2026-11-13\examples_local.json"

$Activate = Join-Path $EnvDir "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    throw "Không thấy venv: $EnvDir — tạo lại theo README mục Windows (RTX 4 GB)."
}
. $Activate

$env:HF_HOME = $HfHome
$env:HF_HUB_CACHE = Join-Path $HfHome "hub"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

if ($Offline) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    Write-Host "OFFLINE=1 (HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE)"
} else {
    Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
    Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
}

Set-Location $Repo

$pyArgs = @("-m", "src.prototype.app", "--port", "$Port")
if ($Mock) {
    $pyArgs += "--mock"
} else {
    $pyArgs += @("--adapter", $Adapter, "--expect_adapter_sha", $ExpectSha, "--ocr_device", "cpu")
}
if ($Share) {
    $pyArgs += "--share"
}
if ((-not $Mock) -and (Test-Path $ExamplesJson)) {
    $pyArgs += @("--examples_json", $ExamplesJson)
}

Write-Host "python $($pyArgs -join ' ')"
Write-Host "Mở http://127.0.0.1:$Port  (máy phải bật suốt lúc demo)"
& python @pyArgs
exit $LASTEXITCODE
