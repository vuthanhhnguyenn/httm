# Chạy lại 2 clip phòng thi bằng code hiện tại, ra video đánh dấu đỏ/vàng + báo cáo.
# Thiếu video thì tự tải từ YouTube bằng yt-dlp (cần uv). Chạy trong thư mục repo:
#   powershell -ExecutionPolicy Bypass -File scripts\chay_lai_2_clip.ps1
# Có GPU: mặc định dùng .venv-gpu (card 0). Không có GPU thì thêm: -Device cpu
param(
    [string]$Repo = "E:\httm",
    [string]$Clip1 = "storage\videos\EPf9O7UXRoM.mp4",
    [string]$Clip2 = "storage\videos\LUy81HI7wS0.mp4",
    [string]$Device = "0",
    [string]$Python = ".\.venv-gpu\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location $Repo
$ids = @{ $Clip1 = "EPf9O7UXRoM"; $Clip2 = "LUy81HI7wS0" }
foreach ($video in @($Clip1, $Clip2)) {
    if (-not (Test-Path $video)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $video) | Out-Null
        Write-Host "Tải $video từ YouTube..."
        uvx yt-dlp -f "bv*[height<=720][ext=mp4]" -o $video "https://www.youtube.com/watch?v=$($ids[$video])"
    }
}
if (-not (Test-Path $Python)) { $Python = ".\.venv\Scripts\python.exe" }
foreach ($pair in @(@($Clip1, "clip1"), @($Clip2, "clip2"))) {
    $video, $name = $pair
    if (-not (Test-Path $video)) { Write-Host "Không thấy $video, bỏ qua."; continue }
    $out = "storage\ket-qua\$name"
    Write-Host "Chạy $video -> $out"
    & $Python scripts\process_video.py $video --all-seats --device $Device --out $out
}
Write-Host "Xong. Mở storage\ket-qua\clip1\annotated.mp4 và report.html (clip2 tương tự)."
