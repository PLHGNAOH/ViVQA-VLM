# =============================================================================
# run_local_windows.ps1 — chạy prototype Gradio trên laptop Windows (RTX 4 GB)
#
# Mục đích: demo bảo vệ — Qwen2.5-VL-3B 4-bit + adapter SH trên GPU, OCR trên CPU
#           (giữ nguyên prompt / tham số sinh / min-max pixels / engine OCR như eval).
#
# Cách dùng nhanh (từ gốc repo):
#   .\demo              → offline, song song OCR+model, tự mở trình duyệt
#   .\demo safe         → offline + -SequentialStart (chậm ~20 s, ít tranh RAM)
#   .\demo share        → tạo link *.gradio.live
#   .\demo nhom         → offline, một thẻ (không so sánh) = -NoCompare
#
# Lệnh chạy (từ bất kỳ thư mục nào):
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Share
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Mock
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -OcrThreads 12
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -NoBrowser
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Dark
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -SequentialStart
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline -Examples
#   powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline -NoCompare
#
# Khởi động: OCR worker song song với nạp Qwen (mặc định). -SequentialStart = .\demo safe.
# Bảng Ví dụ tắt mặc định; bật lại: -Offline -Examples.
# Hôm bảo vệ dùng -Offline (hoặc .\demo), mở http://127.0.0.1:7860/?__theme=light
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Share,
    [switch]$Offline,
    [switch]$Mock,
    [switch]$NoBrowser,
    [switch]$Dark,
    [switch]$SequentialStart,
    [switch]$Examples,
    [switch]$NoCompare,
    [int]$Port = 7860,
    [int]$OcrThreads = 12
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

# Paddle mặc định gán OMP_NUM_THREADS=1 khi import nếu biến chưa có → OCR CPU chậm.
# Đặt trước khi spawn app/ocr_worker (tiến trình con thừa kế env). Không bật mkldnn.
if ($OcrThreads -lt 1) { throw "-OcrThreads phải >= 1" }
$env:OMP_NUM_THREADS = "$OcrThreads"
$env:MKL_NUM_THREADS = "$OcrThreads"
Write-Host "OCR CPU threads: OMP_NUM_THREADS=$OcrThreads MKL_NUM_THREADS=$OcrThreads"

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
if ($SequentialStart) {
    $pyArgs += "--sequential-start"
    Write-Host "STARTUP=sequential (OCR ready rồi mới nạp Qwen)"
} else {
    Write-Host "STARTUP=parallel (OCR worker song song nạp Qwen)"
}
if ($Share) {
    $pyArgs += "--share"
}
# Bảng Ví dụ tắt mặc định; chỉ truyền --examples_json khi -Examples
if ($Examples -and (-not $Mock) -and (Test-Path $ExamplesJson)) {
    $pyArgs += @("--examples_json", $ExamplesJson)
    Write-Host "EXAMPLES=on ($ExamplesJson)"
} else {
    Write-Host "EXAMPLES=off (bật lại: -Examples)"
}
if ($NoCompare) {
    $pyArgs += @("--compare_default", "nhom")
}

Write-Host "python $($pyArgs -join ' ')"
Write-Host "Đang nạp model. Trình duyệt sẽ tự mở khi sẵn sàng; hoặc chờ dòng 'Running on local URL' rồi mở http://127.0.0.1:$Port (nền sáng; muốn nền tối: thêm -Dark). Tắt: Ctrl+C."
$browserJob = $null
$exitCode = 1
if (-not $NoBrowser -and -not $Mock) {
    $pollUrl = "http://127.0.0.1:$Port"
    if ($Dark) {
        $openUrl = $pollUrl
    } else {
        $openUrl = "$pollUrl/?__theme=light"
    }
    $browserJob = Start-Job -ScriptBlock {
        param($poll, $open)
        $deadline = (Get-Date).AddSeconds(300)
        while ((Get-Date) -lt $deadline) {
            try {
                $r = Invoke-WebRequest -Uri $poll -UseBasicParsing -TimeoutSec 2
                if ($r.StatusCode -eq 200) {
                    Start-Process $open
                    Write-Output "Đã mở trình duyệt: $open"
                    return
                }
            } catch {}
            Start-Sleep -Seconds 3
        }
        Write-Output "Hết 300 s chưa mở được trình duyệt ($open)"
    } -ArgumentList $pollUrl, $openUrl
}

try {
    & python @pyArgs
    $exitCode = $LASTEXITCODE
} finally {
    if ($null -ne $browserJob) {
        try {
            $out = Receive-Job -Job $browserJob -ErrorAction SilentlyContinue
            if ($out) { $out | ForEach-Object { Write-Host $_ } }
        } catch {}
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
}
exit $exitCode
