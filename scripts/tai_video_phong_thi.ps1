# Tải vài clip phòng thi (CCTV / camera lớp khi thi) từ YouTube để thử analyze_video_multi.py.
# Chỉ dùng để thử nội bộ, không phát lại hay đưa vào dataset công khai.
# Chạy trong thư mục repo:
#   powershell -ExecutionPolicy Bypass -File scripts\tai_video_phong_thi.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\tai_video_phong_thi.ps1 -Url "https://www.youtube.com/watch?v=..."
# Có -Url thì chỉ tải link đó; thêm -AlsoSearch nếu muốn tìm thêm theo từ khóa.
param(
    [string[]]$Query = @(
        "exam hall cctv footage students",
        "camera giám sát phòng thi thí sinh",
        "examination room cctv cheating caught"
    ),
    [string[]]$Url = @(),     # link cụ thể nếu bạn đã chọn được clip
    [switch]$AlsoSearch,      # vẫn tìm theo từ khóa khi đã có -Url
    [int]$PerQuery = 3,       # số kết quả lấy cho mỗi từ khóa
    [int]$MaxMinutes = 15,    # khi tìm theo từ khóa: bỏ qua video dài hơn (thường là phóng sự)
    [int]$Seconds = 120,      # cắt giữ N giây đầu
    [string]$OutDir = "storage\videos\phong-thi",
    [string]$Python = ".\.venv-gpu\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$trim = @'
import sys, cv2
src, dst, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
n = 0
while n < fps * secs:
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame); n += 1
out.release(); cap.release()
print(f"  -> {dst}: {n} frame, {w}x{h}, {fps:.1f} fps")
'@
$trimPy = Join-Path $OutDir "_trim.py"
Set-Content -Path $trimPy -Value $trim -Encoding UTF8

$rawTemplate = Join-Path $OutDir "raw\%(id)s.%(ext)s"
$sources = Join-Path $OutDir "nguon.tsv"
function Fetch([string]$target, [string[]]$extra) {
    Write-Host "Tải: $target"
    uvx yt-dlp --no-playlist --ignore-errors -f "b[height<=720]/b" @extra `
        --print-to-file "%(id)s`t%(title)s`t%(webpage_url)s" $sources `
        -o $rawTemplate $target
}
foreach ($u in $Url) { Fetch $u @() }
if ($Url.Count -eq 0 -or $AlsoSearch) {
    foreach ($q in $Query) { Fetch "ytsearch$($PerQuery):$q" @("--match-filter", "duration < $($MaxMinutes * 60)") }
}

Get-ChildItem (Join-Path $OutDir "raw") -File | ForEach-Object {
    $clip = Join-Path $OutDir ($_.BaseName + ".mp4")
    if (-not (Test-Path $clip)) { & $Python $trimPy $_.FullName $clip $Seconds }
}
Write-Host "Xong. Clip đã cắt nằm trong $OutDir, nguồn gốc ghi ở nguon.tsv."
Write-Host "Xem qua rồi chọn clip có 2-6 thí sinh rõ mặt để chạy analyze_video_multi.py."
