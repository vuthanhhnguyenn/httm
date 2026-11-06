# Phase 2g: Video vào, video đánh dấu ra (offline, Kaggle T4) — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một lệnh `scripts/process_video.py` biến một video phòng thi thành `annotated.mp4` (H.264, cảnh báo đỏ/vàng), `report.html`, `events.csv/jsonl`, trong ≤ 2 × thời lượng video trên Kaggle T4.

**Architecture:** Ba lượt offline. Lượt nhanh (tầng nhanh có sẵn, thêm `tracks.jsonl`) → lượt kỹ trọn đoạn ở 5 khung/giây nhưng giải mã một lần, crop dùng chung, batch lớn, FP16, 1–2 card → gộp sự kiện, gán mức tin cậy theo `configs/confidence.yaml`, dựng video qua ffmpeg và báo cáo HTML.

**Tech Stack:** Python 3.12, Ultralytics 8.4.160 (yolo11n-pose, yolo11x-pose, yolo11x), OpenCV, Pillow (chữ tiếng Việt), ffmpeg (libx264), pytest, Kaggle Notebook.

**Spec:** `docs/superpowers/specs/2026-11-06-phase2g-offline-video-design.md`

## Global Constraints

- Mốc tốc độ: `total_s ≤ 2 × video_s` trên Kaggle T4, đo trong `timing.json` (tính cả nạp model, không tính cài thư viện/tải model).
- Lượt kỹ: trọn đoạn, `fps = 5`, `warmup_ms = 6000`, model `yolo11x-pose` + `yolo11x`, `imgsz 960` (`--quality max`) hoặc `640` (`--quality balanced`).
- Kết quả lượt kỹ mới phải giống cách cũ (`analyse_window` trên khung đọc từng đoạn) với sai số ≤ 1e-4 khi dùng cùng detector.
- Clip 1 (fixture `packages/proctoring_ai/tests/fixtures/cctv_clip1_review`): `true_covered == 9`, `false_red == 0`, `false_opened ≤ 7`.
- Mức tin cậy theo đúng bảng mục 4 của spec; chữ trên video và báo cáo bằng tiếng Việt.
- Chạy được không GPU (CPU, chậm) để kiểm thử trên cloud. Không có ffmpeg thì rơi về `mp4v` và ghi lý do.
- Thư viện mới: không thêm phụ thuộc Python mới (Pillow, OpenCV đã có); ffmpeg là chương trình ngoài, tuỳ chọn.
- Commit kết thúc bằng hai dòng `Co-Authored-By` / `Claude-Session` như các commit trước. Làm trên nhánh `claude/project-thread-g4frkm` (PR #42), không merge.

## Review Focus

1. Video có tốc độ khung lẻ (29,97), hoặc `CAP_PROP_FRAME_COUNT` sai: lượt dựng phải ra đúng số khung gốc đọc được, cảnh báo đúng giây.
2. Đoạn nghi vấn chạm cuối video hoặc bắt đầu trước 6 s: giải mã một lần không được thiếu hay lặp khung so với cách cũ.
3. Hai đoạn khác ghế có vùng crop gần trùng: gộp crop không được gán pose của ghế này sang ghế kia.
4. Video không có đoạn nghi vấn nào: vẫn ra `annotated.mp4` (chỉ khung ghế), báo cáo trống có dòng "Không có sự kiện".
5. ffmpeg chết giữa chừng (ống bị đóng): không treo, giữ bản `mp4v`, `timing.json` ghi lỗi.

---

### Task 1: Thiết bị và FP16 cho `ReviewDetectors`

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_detectors.py`
- Modify: `scripts/review_suspects.py` (`--device X` thành `device_override` cho mọi model, thêm `--half`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_detectors.py`

**Interfaces:**
- Produces: `ReviewDetectors(specs, device=None, loader=None, image_size=960, batch_size=8, *, half: bool = False, device_override: Mapping[str, str] | None = None)`. `device_override[name]` thắng `spec.device` và được đưa thẳng vào `resolve_device`. `half=True` thêm `half=True` vào `predict` chỉ khi thiết bị của model là CUDA (`"cuda"`, `"0"`, `"cuda:1"`…).
  Lý do: `configs/review_models.yaml` đặt `yolo11x` ở `intel:gpu`; trên Kaggle không có card Intel nên sẽ rơi về CPU nếu không ghi đè.

- [ ] **Step 1:** Test `test_device_override_beats_config`: spec `device: intel:gpu`, `device_override={"m": "cuda:1"}`, loader giả, `_cuda_available` bị monkeypatch thành True → `detectors.devices["m"] == "cuda:1"`, không có reason "intel". Test `test_half_only_on_cuda`: model giả ghi lại kwargs của `predict`; trên `"cpu"` không có `half`, trên `"0"` có `half=True`. Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt. Chạy `uv run --no-sync pytest -q packages/proctoring_ai/tests/multi/test_cctv_review_detectors.py`, phải PASS.
- [ ] **Step 3:** Commit `feat(cctv): per-model device override and FP16 for the deep tier`.

### Task 2: Đo tốc độ trên Kaggle (làm sớm để Nguyên chạy song song)

**Files:**
- Create: `scripts/bench_review.py`
- Create: `notebooks/kaggle_benchmark.ipynb`
- Test: `tests/test_bench_review.py`

**Interfaces:**
- Consumes: `ReviewDetectors(..., half, device_override)` (Task 1).
- Produces: `bench(video: Path, *, batches: Sequence[int], halves: Sequence[bool], devices: Sequence[str], image_sizes: Sequence[int], crops: int = 64, loader=None) -> list[dict]`. Mỗi dòng `{model, batch, half, device, imgsz, ms_per_crop}`. CLI in bảng và ghi `bench.json`.

- [ ] **Step 1:** Test `test_bench_rows`: video tổng hợp 2 s, loader giả (predict ngủ 1 ms mỗi ảnh) → số dòng = 2 model × |batches| × |halves| × |devices| × |imgsz|, `ms_per_crop > 0`. Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt: lấy `crops` crop 960×960 ngẫu nhiên từ khung video, khởi động mỗi cấu hình 1 lần rồi đo 3 lần, lấy trung vị.
- [ ] **Step 3:** Notebook: ô 1 hướng dẫn (bật GPU T4 ×2 và Internet, Dataset Private), ô 2 clone nhánh `claude/project-thread-g4frkm` + `pip install -e` các gói, ô 3 `download_model.py`, ô 4 `bench_review.py` (batch 8/16/32, FP16 bật/tắt, `0` và `1`, imgsz 960/640), ô 5 chạy `analyze_video_multi.py --scene cctv --mode full --fps 5 --suspects --no-video` + `review_suspects.py --fps 5 --max-frames 0 --device 0` trên clip 1 và in thời gian. Kiểm tra JSON hợp lệ bằng `python -c "import json; json.load(open(...))"`.
- [ ] **Step 4:** Commit; chép notebook vào `/mnt/project-files/phase2g/`.

### Task 3: Luật ghi thêm bằng chứng (`evidence_kind`, `raised`)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_rules.py`

**Interfaces:**
- Produces: bản ghi `opened` của PASSING_OBJECT có `evidence_kind ∈ {"object_between", "transfer", "hands_only"}`; của PHONE_DETECTED có `raised: bool` (điện thoại cao hơn mép bàn ở ít nhất một khung trong lần chạy) và `confidence: float` (cao nhất trong lần chạy). Kết quả luật (loại, mốc) không đổi.

- [ ] **Step 1:** Test: tay chạm tay không vật → `hands_only`; có vật giữa → `object_between`; `test_passing_by_object_transfer` → `transfer`; phone giơ cao → `raised True`, trên bàn → `False`. Thêm assert vào test hồi quy clip 1: danh sách (type, mốc) giống trước khi đổi. Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt (truyền thêm chi tiết qua `emit(..., **details)` sẵn có). PASS, cả `pytest -q`.
- [ ] **Step 3:** Commit.

### Task 4: Mức tin cậy

**Files:**
- Create: `configs/confidence.yaml`
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/confidence.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_score.py`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_confidence.py`, `test_cctv_review_cache.py`

**Interfaces:**
- Produces: `load_confidence(path: Path) -> ConfidenceRules`; `assign_levels(review: list[dict], fast: list[dict], rules: ConfidenceRules) -> list[dict]` trả bản sao các bản ghi `opened`/`merged` có thêm `level` (`"red"`/`"yellow"`) và `level_reason` (tiếng Việt). `fast` là các sự kiện luật `opened` của tầng nhanh (`seat_id`, `video_offset_ms`, `type`).
- Produces: `score_review` thêm `true_red`, `false_red` (verdict khớp sự kiện `level == "red"`).
- YAML: mỗi loại một khoá với `always: red|yellow` hoặc điều kiện `red_if: {evidence_kind: [...]}`, `{raised: true, min_confidence: 0.6}`, `{fast_agrees_s: 3}`; `UNCLASSIFIED_SUSPECT`, `REVIEW_SKIPPED`, `LOOK_AT_LAP`: `always: yellow`.

- [ ] **Step 1:** Một test cho mỗi dòng bảng mục 4 của spec (8 test) + `test_unknown_type_is_yellow`. Test `score_review` có `false_red`. Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, PASS.
- [ ] **Step 3:** Thêm vào `tests/test_cctv_review_regression.py` (offline): sau `assign_levels` (fast = rỗng vì fixture không có sự kiện tầng nhanh), `false_red == 0` và `true_covered == 9`. Nếu `false_red > 0` thì báo số thật, không nới test.
- [ ] **Step 4:** Commit.

### Task 5: Lượt kỹ giải mã một lần, crop dùng chung, batch lớn

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_offline.py`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_offline.py`

**Interfaces:**
- Consumes: `review_seats`, `crop_region`, `crop_image`, `assign_poses`, `review_window`, `ReviewDeduper`, `save_window` (có sẵn); `ReviewDetectors.pose_many/objects_many`.
- Produces: `plan_frames(windows, *, fps=5.0, warmup_ms=6000, duration_ms=None) -> dict[int, list[int]]` (mốc ms đích → chỉ số đoạn cần nó; mốc đích = lưới `start − warmup` bước `1000/fps` của từng đoạn, như `_frames`).
- Produces: `review_offline(video: Path, windows: list[SuspectWindow], detectors, *, fps=5.0, warmup_ms=6000, share_iou=0.7, cache_path: Path | None = None, parallel: bool = False, on_progress=None) -> OfflineResult` với `OfflineResult(records: list[dict], frames: dict[window_key, list[ReviewFrame]], stats: dict)`. `stats` gồm `crops`, `crops_shared`, `decode_s`, `pose_s`, `objects_s`, `rules_s`.
- Thuật toán: đọc video tuần tự một lần, với mỗi khung đọc được thì khớp mốc đích gần nhất phía sau (đúng quy tắc của `_frames`); gom crop vào hàng chờ; khi đủ `batch_size` hoặc hết video thì chạy `pose_many` và `objects_many` (song song trên 2 luồng nếu `parallel` và hai model ở hai thiết bị khác nhau). Ở cùng một khung, đoạn có vùng crop IoU ≥ `share_iou` với một crop đã có thì dùng lại crop đó (vùng = hợp hai vùng). Không giữ ảnh gốc sau khi đã crop (bộ nhớ).

- [ ] **Step 1:** Test:
  - `test_plan_frames_matches_frames_grid`: 2 đoạn chồng nhau trên video tổng hợp 10 fps → mốc giống `_frames` của `review_suspects.py`.
  - `test_same_records_as_per_window` (Review Focus 2): video tổng hợp 6 s, 3 đoạn (một chạm cuối video, một bắt đầu ở 1 s), detector giả trả keypoint theo toạ độ crop → `records` (bỏ `id`) và `ReviewFrame` giống `analyse_window` trên khung đọc từng đoạn, sai số ≤ 1e-4.
  - `test_shared_crop_calls_model_once` và `test_shared_crop_keeps_seats_apart` (Review Focus 3).
  - `test_parallel_two_devices` (detector giả ghi luồng gọi).
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, PASS. `pytest -q`.
- [ ] **Step 3:** Commit.

### Task 6: `tracks.jsonl` từ lượt nhanh

**Files:**
- Modify: `packages/proctoring_runtime/src/proctoring_runtime/multi_replay.py` (tham số `tracks_path: Path | None = None`)
- Modify: `scripts/analyze_video_multi.py` (`--tracks`)
- Test: `packages/proctoring_runtime/tests/test_multi_replay*.py` (file đang có)

**Interfaces:**
- Produces: mỗi khung đã phân tích một dòng `{"t": offset_ms, "seats": {seat_id: [x, y, w, h, role]}}` (role: `candidate`/`walker`/`pending`, số làm tròn 4 chữ số). `load_tracks(path) -> list[tuple[int, dict]]` trong `proctoring_runtime/tracks.py`, `tracks_at(tracks, t_ms, max_gap_ms=200) -> dict | None`.

- [ ] **Step 1:** Test ghi/đọc tracks qua `run_multi_replay` với pipeline giả có sẵn trong test; `tracks_at` trả `None` khi lệch > 200 ms. FAIL.
- [ ] **Step 2:** Cài đặt, PASS. Commit.

### Task 7: Dựng video và chữ tiếng Việt

**Files:**
- Create: `packages/proctoring_runtime/src/proctoring_runtime/text.py` (chuyển `put_text`, `_FONTS` từ `scripts/review_suspects.py`, script import lại)
- Create: `packages/proctoring_runtime/src/proctoring_runtime/render.py`
- Test: `packages/proctoring_runtime/tests/test_render.py`

**Interfaces:**
- Produces: `render_video(video: Path, out: Path, tracks, events: list[dict], *, min_show_ms=2000, ffmpeg: str | None = "ffmpeg") -> dict` trả `{"frames": n, "encoder": "libx264"|"mp4v", "error": str|None, "seconds": s}`. Sự kiện hiện từ `video_offset_ms` tới `max(closed, opened + min_show_ms)`; ghế có sự kiện vẽ dày màu đỏ `(0,0,255)` hoặc vàng `(0,200,255)` với nhãn tiếng Việt `event_name(type)`; ghế khác xám mảnh; góc trên "Đỏ: n · Vàng: m" (đếm sự kiện đã mở tới thời điểm đó). Ghi qua `subprocess.Popen([ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps, "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", out])`; `BrokenPipeError` hoặc mã thoát ≠ 0 → ghi lại bằng `cv2.VideoWriter(mp4v)` và đặt `error`.

- [ ] **Step 1:** Test: video tổng hợp 3 s 10 fps → đọc lại đủ 30 khung (Review Focus 1, thêm một video 29,97 fps); sự kiện đỏ 1–2 s → khung 1,5 s có điểm ảnh đỏ trong vùng ghế, khung 0,5 s không có; `ffmpeg="khong-co"` → `encoder == "mp4v"`, file tồn tại; ffmpeg giả thoát sớm (script `sh -c 'exit 1'`) → không treo, có `error` (Review Focus 5); không có sự kiện → vẫn ra file (Review Focus 4). FAIL.
- [ ] **Step 2:** Cài đặt, PASS (cloud có ffmpeg tại `/usr/bin/ffmpeg`). Commit.

### Task 8: Báo cáo HTML và CSV

**Files:**
- Create: `packages/proctoring_runtime/src/proctoring_runtime/report.py`
- Test: `packages/proctoring_runtime/tests/test_report.py`

**Interfaces:**
- Produces: `write_report(out_dir: Path, events: list[dict], *, video_name: str, timing: Mapping) -> None` ghi `report.html` (một file, không tải gì từ mạng; video `annotated.mp4` nhúng bằng `<video>`; mỗi dòng: màu, loại tiếng Việt, mốc `mm:ss.s` bấm được đặt `currentTime`, ghế, lý do mức tin cậy, ảnh `evidence/<id>.jpg`; sắp đỏ trước rồi theo thời gian; bộ lọc đỏ/vàng/loại bằng JS nội tuyến) và `events.csv` (UTF-8 BOM để Excel đọc tiếng Việt: `level,type,label_vi,time_s,seat_id,reason,evidence`).

- [ ] **Step 1:** Test: 3 sự kiện (2 vàng, 1 đỏ) → HTML có 3 dòng, dòng đầu là đỏ; không có `http://`/`https://` trong HTML; danh sách rỗng → có "Không có sự kiện"; CSV đọc lại bằng `csv` đúng 3 dòng. FAIL.
- [ ] **Step 2:** Cài đặt, PASS. Commit.

### Task 9: `process_video.py`

**Files:**
- Create: `scripts/process_video.py`
- Test: `tests/test_process_video.py`, `tests/test_cctv_review_regression.py`

**Interfaces:**
- Consumes: Task 1, 4–8; `analyze_video_multi.run` (gọi trực tiếp với `--no-video --suspects --tracks`).
- Produces: `process(video: Path, out: Path, *, quality="max", device=None, device2=None, detectors=None, fast_runner=None) -> dict` (timing). CLI: `process_video.py VIDEO --out DIR [--quality max|balanced] [--device 0] [--device2 1] [--batch 16]`. `timing.json`: `video_s`, `fast_s`, `review_s`, `render_s`, `report_s`, `total_s`, `ratio`, `over_budget` (`ratio > 2`), `devices`, `device_reasons`, thống kê lượt kỹ, `encoder`. Bằng chứng: ảnh lấy từ khung đã giải mã ở lượt kỹ cho sự kiện tầng kỹ (vẽ như `_draw`), tầng nhanh có sẵn `evidence/`.

- [ ] **Step 1:** Test đầu-cuối trên CPU: video tổng hợp 4 s, `fast_runner` giả trả 1 đoạn nghi vấn + tracks, detector giả → đủ file đầu ra, `timing["ratio"]` có mặt, sự kiện trong `events.jsonl` có `level`. Video không mở được → mã 1. Không có đoạn nghi vấn → vẫn ra video và báo cáo. FAIL.
- [ ] **Step 2:** Cài đặt, PASS. `pytest -q`, ruff, mypy (cả `scripts`).
- [ ] **Step 3:** Chạy thật trên cloud CPU với 15 s đầu clip 1 (`ffmpeg -t 15 -c copy`) để thấy đủ đầu ra; xem vài khung của `annotated.mp4`. Commit.

### Task 10: Notebook Kaggle chạy trọn quy trình

**Files:**
- Create: `notebooks/kaggle_process_video.ipynb`
- Modify: `docs/evaluation-results.md`, `README.md` (mục "Chạy trên Kaggle")

- [ ] **Step 1:** Notebook theo mục 6 của spec: biến `BRANCH`, `DATASET_DIR`; clone (Kaggle Secret `GITHUB_TOKEN` nếu repo private); cài; tải model; lặp mọi `*.mp4` → `/kaggle/working/ket-qua/<tên>/`; bảng tóm tắt (thời lượng, thời gian, tỉ lệ, đỏ, vàng); nén `ket-qua.zip`. Kiểm tra JSON hợp lệ.
- [ ] **Step 2:** README: 6–10 dòng cách dùng. Commit; chép notebook và hướng dẫn vào `/mnt/project-files/phase2g/`.

### Task 11: Chốt trên clip 1, clip 2 và đo trên Kaggle

**Files:**
- Modify: `docs/evaluation-results.md`

- [ ] **Step 1:** Cloud CPU: chạy `process_video.py` trọn clip 1 và clip 2 (chậm, chạy nền). Chấm clip 1 bằng `score_review` (sau `assign_levels`): `true_covered == 9`, `false_red == 0`. Ghép ảnh sự kiện clip 2 kèm màu để Nguyên chấm.
- [ ] **Step 2:** Đẩy nhánh, cập nhật mô tả PR #42, nhờ Nguyên chạy `kaggle_benchmark.ipynb` và `kaggle_process_video.ipynb` rồi gửi `timing.json`/`bench.json`. Đạt khi `ratio ≤ 2` cho clip 1 và clip 2. Không đạt: chọn từ `bench.json` cấu hình nhanh nhất còn giữ chất lượng (batch, FP16, 2 card), thử `balanced` và chấm lại clip 1 bằng cache ở imgsz 640 (cần chạy lại lượt kỹ trên Kaggle); nếu vẫn không đạt thì báo số và đề xuất `yolo11l`/TensorRT để Nguyên quyết.
- [ ] **Step 3:** Ghi số đo vào `docs/evaluation-results.md`. Commit.
