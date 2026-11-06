# Phase 3: Lõi nhận diện học từ nhãn — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mọi ghế được xét theo từng cửa sổ 3 s và một model LightGBM học từ nhãn quyết định sự kiện thay cho ngưỡng viết tay, chạy bằng `process_video.py --core learned`.

**Architecture:** Tầng 1 dùng lại `review_offline`, nhưng mỗi ghế là một `SuspectWindow` phủ cả video. Cache của nó (định dạng `review_cache`) chính là dòng thời gian. `features.py` cắt dòng thời gian thành cửa sổ 3 s và tính đặc trưng; luật cũ chạy một lần mỗi ghế và chỉ thành cờ. `train_core.py` train LightGBM rồi chấm theo từng video để riêng. `learned.py` đổi xác suất thành bản ghi `opened`/`closed` cùng dạng hiện có, nên render và báo cáo giữ nguyên.

**Tech Stack:** Python 3.12 (cú pháp 3.11 hợp lệ), numpy, OpenCV, ultralytics (đã có), lightgbm 4.x (mới), pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-phase3-learned-core-design.md`

## Global Constraints

- Lõi cũ là mặc định: `--core rules`. `--core learned` thiếu `--model` hoặc file model thì `ProcessError` có hướng dẫn, không quay về lõi cũ.
- Cửa sổ 3 000 ms, bước 1 000 ms, 5 khung/giây. Trễ hai ngưỡng: mở khi `p ≥ 0.5` ở 2 cửa sổ liền nhau, đóng khi `p < 0.35`.
- Đỏ: độ đúng ≥ 90% trên tập kiểm tra. Vàng: ≥ 50%.
- 8 lớp: `NORMAL, LOOK_AT_NEIGHBOR, HEADS_TOGETHER, PASSING_OBJECT, PHONE_DETECTED, DOCUMENT_DETECTED, LOOK_AT_LAP, TURN_AROUND`, cộng `UNSURE` (chỉ có trong nhãn, không bao giờ train). `TURN_AROUND` có dưới 10 mẫu thì gộp vào `LOOK_AT_NEIGHBOR`.
- Không đưa ảnh thí sinh vào repo. Ảnh ghép nằm trong project files hoặc output Kaggle.
- Sự kiện cặp (`PASSING_OBJECT`, `HEADS_TOGETHER`) gán cho người vươn sang (quy tắc phase 2d).
- Cú pháp phải parse được với `feature_version=(3, 11)` (`tests/test_python311_compat.py`).
- Không gọi `asyncio.run` theo từng khung (lỗi treo trên Windows). Luật cũ được gọi qua `review_window`, vốn dùng sẵn một `asyncio.Runner` cho mỗi cửa sổ.
- Chuỗi hiển thị cho người dùng viết bằng tiếng Việt.
- Commit kết thúc bằng hai dòng trailer của phiên: `Co-Authored-By` và `Claude-Session`.

## Review Focus

1. Ghế có người chỉ một phần video (đi vắng, vào muộn): cửa sổ không có khung nào của ghế bị bỏ qua, không ra hàng toàn số 0 (test ở Task 3).
2. Ghế xuất hiện dưới 10 s: mốc nền yaw/lift lấy trung vị các ghế cùng hàng, không chia cho 0 hay NaN (test ở Task 3).
3. Model train với danh sách đặc trưng khác danh sách lúc chạy: báo lỗi nêu tên cột thiếu (test ở Task 6).
4. Video ngắn hơn 3 s: không có cửa sổ, `events.jsonl` rỗng, render vẫn chạy (test ở Task 7).
5. Nhãn của một lần chạy được dùng với timeline của lần chạy khác (id ghế khác): khớp theo IoU hộp ghế ≥ 0.5, không khớp thì bỏ và đếm (test ở Task 4).

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `packages/proctoring_ai/src/proctoring_ai/multi/cctv/timeline.py` | `tracks.jsonl` → một `SuspectWindow` cho mỗi ghế, phủ cả video |
| `scripts/build_timeline.py` | Tầng nhanh, rồi `review_offline` trên các cửa sổ đó, ghi `timeline.json.gz` và `timeline_timing.json` |
| `packages/proctoring_ai/src/proctoring_ai/multi/cctv/features.py` | Cửa sổ 3 s, `FEATURE_NAMES`, `FeatureRow`, ghi/đọc `features.csv` |
| `packages/proctoring_ai/src/proctoring_ai/multi/cctv/labels.py` | Định dạng nhãn, chọn đoạn cần gán, khớp nhãn với hàng đặc trưng |
| `scripts/label_sheets.py` | Ảnh ghép 6 khung mỗi đoạn và `index.jsonl` |
| `scripts/train_core.py` | Train, chấm từng video để riêng, chọn ngưỡng, ghi model và `meta.json` |
| `packages/proctoring_ai/src/proctoring_ai/multi/cctv/learned.py` | Nạp model, dự đoán, trễ hai ngưỡng, sinh bản ghi sự kiện |
| `scripts/process_video.py` | `--core learned\|rules`, `--model` |
| `notebooks/kaggle_timeline.ipynb` | Chạy `build_timeline` và `label_sheets` cho mọi video, zip kết quả |
| `notebooks/kaggle_xem_dataset_anh.ipynb` | Xem cấu trúc, các lớp và giấy phép của dataset ảnh |
| `data/labels/learned/<video>.jsonl` | Nhãn (chỉ toạ độ, không có ảnh) |

---

### Task 1: Cửa sổ toàn video cho mọi ghế

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/timeline.py`
- Modify: `pyproject.toml` (thêm `"lightgbm>=4.3,<5.0"` vào nhóm `ai`; `uv lock`)
- Test: `packages/proctoring_ai/tests/test_cctv_timeline.py`

**Interfaces:**
- Produces: `all_seat_windows(tracks: Iterable[Mapping[str, Any]], duration_ms: int, *, min_candidate_share: float = 0.5, neighbor_widths: float = 2.5, neighbor_rows_heights: float = 1.2) -> list[SuspectWindow]`
  - Mỗi dòng của `tracks` có dạng `{"t": ms, "seats": {uuid: [x, y, w, h, role]}}`.
  - Một ghế được giữ khi role là `CANDIDATE` ở ít nhất `min_candidate_share` số dòng có ghế đó.
  - Hộp ghế là trung vị từng toạ độ.
  - Cửa sổ có dạng `SuspectWindow(seat_id, None, neighbor_ids, 0, duration_ms, ("ALL_SEATS",), box, seats={seat và các ghế kề: hộp})`.
  - Ghế kề: tâm lệch ngang < `neighbor_widths` × bề rộng ghế, và lệch dọc < `neighbor_rows_heights` × chiều cao ghế.
  - Kết quả sắp theo `str(seat_id)`.
- Produces: `load_tracks(path: Path) -> list[dict[str, Any]]`

- [ ] **Step 1: Viết test.**
  - `test_candidate_seats_get_one_window_each`: 3 ghế, ghế thứ ba là `walker` trong 80% số dòng, kết quả có 2 cửa sổ, `start_ms == 0`, `end_ms == duration_ms`, `reasons == ("ALL_SEATS",)`.
  - `test_box_is_median`: hộp ghế là trung vị của các hộp `[0.1,…]`, `[0.2,…]`, `[0.3,…]`, tức `x == 0.2`.
  - `test_neighbours_same_row_only`: ghế A và B cùng hàng, cách 1 bề rộng, nên kề nhau. Ghế C ở dưới 2 chiều cao nên không kề A. `seats` của cửa sổ A chứa A và B.
- [ ] **Step 2:** `uv run pytest packages/proctoring_ai/tests/test_cctv_timeline.py -q`. Expected: FAIL, vì chưa có module `timeline`.
- [ ] **Step 3:** Viết `timeline.py` và thêm dependency.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: 3 passed.
- [ ] **Step 5:** Commit `feat(cctv): one whole-video window per seat for the learned core`.

### Task 2: `build_timeline.py` cùng notebook Kaggle

**Files:**
- Create: `scripts/build_timeline.py`
- Create: `notebooks/kaggle_timeline.ipynb`
- Test: `tests/test_build_timeline.py`

**Interfaces:**
- Consumes: `all_seat_windows`, `load_tracks` (Task 1); từ `process_video.py`: `run_fast`, `build_detectors`, `_probe`, `ProcessError`; `review_offline(video, windows, detectors, fps=5.0, warmup_ms=0, share_iou=..., cache_path=..., parallel=..., batch_size=...)`.
- Produces: `build(video: Path, out: Path, *, quality: str = "max", device: str | None = None, device2: str | None = None, detectors: Any = None, fast_runner: FastRunner | None = None, reuse_fast: bool = False, half: bool = True) -> dict[str, Any]`.
  - Ghi `out/tracks.jsonl`, `out/fast/`, `out/timeline.json.gz` (định dạng `save_window`, mỗi ghế một dòng) và `out/timeline_timing.json`.
  - `timeline_timing.json` có các khoá `video_s, fast_s, timeline_s, total_s, ratio, seats, crops, crops_shared`.
  - CLI: `build_timeline.py VIDEO --out DIR [--quality] [--device] [--device2] [--reuse-fast] [--fp32]`.

- [ ] **Step 1: Viết test.**
  - `test_build_writes_one_timeline_line_per_seat`: dùng `fast_runner` giả ghi `tracks.jsonl` có 2 ghế candidate, detectors giả lấy từ `tests/test_process_video.py` (một người mỗi crop), video tổng hợp 4 s bằng `cv2.VideoWriter`. `load_cache(out/"timeline.json.gz")` có 2 khoá, và mỗi khoá có khoảng 20 khung (5 khung/giây × 4 s, sai số ±2).
  - `test_timing_has_ratio`: `timeline_timing.json["seats"] == 2`, có `ratio > 0`.
- [ ] **Step 2:** `uv run pytest tests/test_build_timeline.py -q`. Expected: FAIL, vì chưa có `build_timeline`.
- [ ] **Step 3:** Viết `build_timeline.py`. Lượt nhanh gọi `run_fast`, tôn trọng `reuse_fast` giống `process.py`. Lượt kỹ gọi `review_offline` với `warmup_ms=0` và in tiến độ bằng tiếng Việt.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: 2 passed.
- [ ] **Step 5:** Tạo notebook `kaggle_timeline.ipynb`, cùng khung với `kaggle_chan_doan.ipynb`:
  - `BRANCH = "claude/project-thread-g4frkm"`; clone hoặc fetch rồi reset; cài `uv sync`.
  - Tìm mọi `.mp4` dưới `/kaggle/input`.
  - Với mỗi video: chạy `build_timeline.py` vào `/kaggle/working/timeline/<stem>/`, sau đó chạy `label_sheets.py` (Task 4).
  - In bảng ratio và zip `timeline.zip`.
  - Kiểm tra JSON hợp lệ: `python -c "import json;json.load(open('notebooks/kaggle_timeline.ipynb'))"`.
- [ ] **Step 6:** Commit `feat(cctv): build_timeline script and Kaggle notebook`.

### Task 3: Đặc trưng cửa sổ 3 s

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/features.py`
- Test: `packages/proctoring_ai/tests/test_cctv_features.py`

**Interfaces:**
- Consumes: `ReviewFrame`, `review_window` (`review_rules.py`); `head_yaw_index`, `head_lift`, `head_point`, `point` (`head.py`); `DeskLine` (`suspects.py`); `LEFT_WRIST`, `RIGHT_WRIST` (`pose_detector.py`); `window_key`, `load_cache` (`review_cache.py`).
- Produces:
  - `WIN_MS = 3000`, `STEP_MS = 1000`, `RULE_TYPES` (tuple 8 lớp trừ `NORMAL`), `FEATURE_NAMES: tuple[str, ...]`.
  - `@dataclass(frozen=True, slots=True) class FeatureRow: video: str; seat_id: UUID; start_ms: int; end_ms: int; seat_box: Box; values: Mapping[str, float]; reacher_for: Mapping[str, str]`. `reacher_for` cho biết với mỗi lớp cặp thì người kề nào là bên kia.
  - `video_features(video: str, windows: Sequence[SuspectWindow], frames: Mapping[CacheKey, list[ReviewFrame]]) -> list[FeatureRow]`
  - `write_csv(path: Path, rows: Sequence[FeatureRow]) -> None` và `read_csv(path: Path) -> list[FeatureRow]` (đọc lại ra đúng giá trị).
- Nội dung `FEATURE_NAMES`, mỗi nhóm đủ các tên sau, thứ tự cố định:
  - **Đầu**, so với trung vị yaw và lift của ghế trên cả video: `yaw_dev_mean, yaw_dev_max, yaw_dev_std, yaw_turn_share` (tỷ lệ khung |lệch| > 0.35), `lift_dev_mean, lift_min, head_down_share`.
  - **Tay**:
    - `wrist_below_desk_share`: tỷ lệ khung mọi cổ tay thấy được đều dưới `DeskLine`; khung không thấy cổ tay tính là dưới.
    - `wrist_speed_max`: tính theo bề rộng ghế mỗi giây.
    - `reach_out_max`: phần cổ tay vươn ra ngoài hộp ghế, tính theo bề rộng ghế.
  - **Cặp**, với người kề có `wrist_gap_min` nhỏ nhất: `wrist_gap_min` (theo chiều cao ghế), `head_gap_ratio_min` (khoảng cách đầu–đầu chia cho trung vị của cặp), `lean_toward`.
  - **Vật**: `phone_conf_max, phone_near_wrist_share, phone_raised, paper_conf_max, paper_near_wrist_share, object_moved`.
  - **Luật**: `rule_<TYPE>` có giá trị 1.0 nếu `review_window` trên cửa sổ toàn video của ghế có sự kiện loại đó, của ghế chính, chồng lên cửa sổ 3 s; ngược lại 0.0.
  - **Độ tin cậy**: `kp_conf_mean, seat_height, frames_seen_share`.
- Quy tắc biên:
  - Cửa sổ có `frames_seen_share == 0` thì không ra hàng.
  - Ghế có tổng thời gian thấy được dưới 10 000 ms lấy mốc nền là trung vị các ghế khác cùng hàng (tâm dọc lệch < 0.5 chiều cao). Không có ghế nào cùng hàng thì dùng 0.0.
  - Không giá trị nào là NaN.

- [ ] **Step 1: Viết test.** Khung giả dựng bằng helper `make_frame(ms, seat_kps, objects)`.
  - `test_head_turn_shows_in_yaw_dev`: ghế quay đầu (yaw lệch 0.6) trong 2 s giữa video 20 s. Cửa sổ chứa đoạn đó có `yaw_turn_share ≥ 0.5`; cửa sổ ở giây 0–3 có `yaw_turn_share == 0`.
  - `test_reach_and_wrist_gap`: cổ tay ghế A nằm trong hộp ghế B ở giây 5–6. Cửa sổ 4–7 có `reach_out_max > 0.5`, `wrist_gap_min < 0.2`, và `reacher_for["PASSING_OBJECT"] == str(B)`.
  - `test_phone_near_wrist`: điện thoại conf 0.7 cạnh cổ tay. Kết quả `phone_conf_max == 0.7` và `phone_near_wrist_share > 0`.
  - `test_absent_seat_has_no_rows`: ghế không có pose ở giây 10–20 thì không có hàng nào có `start_ms ≥ 10000` và `end_ms ≤ 20000`.
  - `test_short_seat_uses_row_baseline`: ghế chỉ thấy 5 s. Mọi giá trị hữu hạn, và mốc nền bằng trung vị của ghế cùng hàng.
  - `test_no_nan_and_fixed_columns`: mọi hàng có đúng `set(FEATURE_NAMES)` và không có NaN.
  - `test_csv_roundtrip`: `read_csv(write_csv(rows)) == rows`, giá trị làm tròn 6 chữ số.
  - `test_clip1_fixture_runs`: chạy trên `tests/fixtures/cctv_clip1_review`, coi mỗi cửa sổ của fixture là một ghế. Ra hơn 0 hàng, không lỗi.
- [ ] **Step 2:** `uv run pytest packages/proctoring_ai/tests/test_cctv_features.py -q`. Expected: FAIL, vì chưa có module.
- [ ] **Step 3:** Viết `features.py`. `review_window` gọi đúng một lần cho mỗi cửa sổ toàn video. Hai khung liền nhau cách quá 1 000 ms thì không tính tốc độ tay giữa hai khung đó.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: 8 passed.
- [ ] **Step 5:** Commit `feat(cctv): 3 s window features for the learned core`.

### Task 4: Nhãn, chọn đoạn và ảnh ghép

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/labels.py`
- Create: `scripts/label_sheets.py`
- Test: `packages/proctoring_ai/tests/test_cctv_labels.py`

**Interfaces:**
- Consumes: `FeatureRow`, `read_csv`, `RULE_TYPES` (Task 3).
- Produces:
  - `LABELS = ("NORMAL", *RULE_TYPES, "UNSURE")`.
  - `@dataclass(frozen=True, slots=True) class Label: video: str; seat_box: Box; start_ms: int; end_ms: int; label: str; labeler: str; note: str = ""`.
  - `save_labels(path, labels)` và `load_labels(path) -> list[Label]`: jsonl; nhãn ngoài `LABELS` thì `ValueError`.
  - `@dataclass(frozen=True, slots=True) class Segment: video: str; seat_id: UUID; seat_box: Box; start_ms: int; end_ms: int; why: str`. Trong đó `why` là `"rule"`, `"outlier"` hoặc `"quiet"`.
  - `select_segments(rows: Sequence[FeatureRow], *, quiet_fraction: float = 0.2, seed: int = 0, outlier_q: float = 0.95) -> list[Segment]`.
    - Hàng "nổi bật": có ít nhất một `rule_*` bằng 1, hoặc một trong các đặc trưng `yaw_dev_max, reach_out_max, phone_conf_max, paper_conf_max, wrist_below_desk_share` vượt phân vị `outlier_q` của chính ghế đó.
    - Hàng nổi bật liền nhau của cùng ghế gộp thành một đoạn.
    - Thêm `quiet_fraction` × số hàng yên lặng, chọn ngẫu nhiên với `random.Random(seed)`.
  - `match(rows: Sequence[FeatureRow], labels: Sequence[Label], *, min_iou: float = 0.5) -> tuple[list[tuple[FeatureRow, str]], int]`.
    - Một hàng nhận nhãn khi cùng video, IoU hộp ≥ `min_iou`, và trung điểm cửa sổ 3 s nằm trong `[start_ms, end_ms]` của nhãn.
    - Bỏ `UNSURE`.
    - Trả về các cặp đã khớp và số nhãn không khớp được hàng nào.
  - `label_sheets.py VIDEO FEATURES_CSV --out DIR`:
    - Với mỗi `Segment`, ghi `DIR/<n:04d>.jpg`: lưới 2×3 gồm 6 khung đều nhau trong đoạn, cắt quanh `crop_region([seat_box] + ghế kề)`, viền ghế đích màu đỏ, góc trên ghi `#n t=…s`.
    - Ghi `DIR/index.jsonl`, mỗi dòng một `Segment` kèm `n`.

- [ ] **Step 1: Viết test.**
  - `test_select_merges_adjacent_rule_rows`: 3 hàng liền nhau của một ghế đều có `rule_LOOK_AT_LAP = 1`, nên ra đúng 1 đoạn `why == "rule"`, kéo dài từ start của hàng đầu tới end của hàng cuối.
  - `test_quiet_sample_is_seeded`: 100 hàng yên lặng, `quiet_fraction = 0.2`, ra 20 đoạn `"quiet"`; cùng seed thì cùng kết quả.
  - `test_match_by_box_iou`: nhãn có hộp lệch 0.01 vẫn khớp. Nhãn có hộp ở ghế khác (IoU 0) thì không khớp và được đếm vào số thứ hai.
  - `test_unsure_is_dropped` và `test_unknown_label_raises`.
  - `test_sheet_written`: video tổng hợp 4 s cùng 1 đoạn sinh ra `0000.jpg` đọc được bằng `cv2.imread` và `index.jsonl` có 1 dòng.
- [ ] **Step 2:** `uv run pytest packages/proctoring_ai/tests/test_cctv_labels.py -q`. Expected: FAIL.
- [ ] **Step 3:** Viết `labels.py` và `label_sheets.py`.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: 6 passed.
- [ ] **Step 5:** Commit `feat(cctv): label format, segment selection and contact sheets`.

### Task 5: Train và chấm theo từng video để riêng

**Files:**
- Create: `scripts/train_core.py`
- Test: `tests/test_train_core.py`

**Interfaces:**
- Consumes: `FEATURE_NAMES`, `read_csv` (Task 3); `load_labels`, `match` (Task 4).
- Produces:
  - `CLASSES: tuple[str, ...]`: 8 lớp, sau khi đã gộp `TURN_AROUND` nếu cần.
  - `train(pairs: Sequence[tuple[FeatureRow, str]], *, seed: int = 0) -> tuple[Any, tuple[str, ...]]` trả về booster LightGBM và các lớp thật sự dùng.
    - Tham số: `objective="multiclass"`, `num_leaves=15`, `min_data_in_leaf=5`, `learning_rate=0.05`, `num_boost_round=300`, trọng số mẫu tỷ lệ nghịch với tần suất lớp.
    - Lớp có 0 mẫu thì bỏ khỏi danh sách. `TURN_AROUND` có dưới 10 mẫu thì đổi nhãn thành `LOOK_AT_NEIGHBOR` và in ra dòng ghi lại quyết định đó.
  - `leave_one_out(pairs_by_video: Mapping[str, Sequence[tuple[FeatureRow, str]]]) -> dict[str, dict[str, Any]]`: với mỗi video, train trên các video còn lại, dự đoán trên video đó, và báo `precision`, `recall` cho từng lớp (khác `NORMAL`) ở mức cửa sổ.
  - `pick_thresholds(probs: Sequence[float], truth: Sequence[bool], *, red_precision: float = 0.9, yellow_precision: float = 0.5) -> tuple[float, float]`: chọn ngưỡng nhỏ nhất mà độ đúng của các cửa sổ có p ≥ ngưỡng đạt yêu cầu. Không đạt thì trả về 1.01, nghĩa là không bao giờ báo mức đó.
  - `save_model(dir: Path, booster, classes, thresholds: Mapping[str, tuple[float, float]], meta_extra: Mapping[str, Any]) -> None` ghi `dir/model.txt` và `dir/meta.json`. `meta.json` gồm `feature_names, classes, thresholds, videos, frame_size, seat_height_median, trained_at, licenses`.
  - CLI: `train_core.py --features DIR_OR_CSV... --labels DIR --out models/learned_core/<tên> [--leave-one-out]`. Ngưỡng cuối cùng chọn trên xác suất leave-one-out gộp của mọi video. Model cuối train trên toàn bộ dữ liệu.

- [ ] **Step 1: Viết test.** Dữ liệu tổng hợp dựng sẵn:
  - `test_train_separates_synthetic_classes`: 200 hàng `NORMAL` với đặc trưng ngẫu nhiên nhỏ, và 40 hàng `LOOK_AT_NEIGHBOR` có `yaw_turn_share = 1`. Dự đoán trên chính tập này đúng ≥ 95%.
  - `test_turn_around_merged_when_rare`: 5 nhãn `TURN_AROUND` nên `"TURN_AROUND" not in classes`.
  - `test_pick_thresholds`: với p `[0.9, 0.8, 0.6, 0.4]` và truth `[T, T, F, T]`, kết quả `(0.8, 0.4)`.
  - `test_pick_thresholds_never_reached` cho ra `1.01`.
  - `test_leave_one_out_reports_each_video`: 2 video tổng hợp cho ra 2 khoá, mỗi khoá có `recall` của `LOOK_AT_NEIGHBOR`.
  - `test_meta_lists_features`: `meta.json["feature_names"] == list(FEATURE_NAMES)`.
- [ ] **Step 2:** `uv run pytest tests/test_train_core.py -q`. Expected: FAIL.
- [ ] **Step 3:** Viết `train_core.py`.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: 6 passed.
- [ ] **Step 5:** Commit `feat(cctv): train the learned core with leave-one-video-out scoring`.

### Task 6: Lõi học khi chạy

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/learned.py`
- Test: `packages/proctoring_ai/tests/test_cctv_learned.py`

**Interfaces:**
- Consumes: `FeatureRow`, `FEATURE_NAMES` (Task 3); `model.txt` và `meta.json` (Task 5).
- Produces:
  - `class LearnedCore` với các thành phần:
    - `@classmethod load(cls, model_dir: Path) -> LearnedCore`. Thiếu file thì `FileNotFoundError` có thông điệp tiếng Việt. `meta.feature_names` khác `FEATURE_NAMES` thì `ValueError` nêu các cột thiếu và thừa.
    - `predict(rows: Sequence[FeatureRow]) -> list[dict[str, float]]`: xác suất từng lớp cho mỗi hàng.
    - `events(rows, probs, *, p_open: float = 0.5, p_close: float = 0.35) -> list[dict[str, Any]]`.
  - Thuật toán của `events`:
    - Với mỗi ghế và mỗi lớp khác `NORMAL`, duyệt các hàng theo `start_ms`.
    - Mở khi hai hàng liền nhau đều có `p ≥ p_open`. `opened` lấy `start_ms` của hàng đầu tiên.
    - Đóng ở hàng đầu tiên có `p < p_close`. `closed` lấy `end_ms` của hàng cuối còn ≥ `p_close`.
    - Hết video mà sự kiện chưa đóng thì đóng ở `end_ms` của hàng cuối.
    - Các bản ghi có cùng dạng `review_window`: `id, type, seat_id, scene: "cctv", stage: "learned", action, video_offset_ms, duration_ms` (bản ghi closed), `probability` (max p), `head` (tâm trên của hộp ghế), `level` (`"red"` nếu `probability ≥ thresholds[type][0]`, `"yellow"` nếu ≥ `[1]`, ngược lại không ghi sự kiện), `level_reason: "learned"`.
    - Lớp cặp ghi `seat_id` là người vươn sang, `neighbor_seat_id` lấy từ `reacher_for`.
  - Hàm `domain_warning(meta, frame_size, seat_height_median) -> str | None`: trả về câu cảnh báo tiếng Việt khi cỡ khung hoặc chiều cao ghế trung vị lệch quá 2 lần so với lúc train.

- [ ] **Step 1: Viết test.**
  - `test_hysteresis_needs_two_windows`: p theo thời gian `[0.6, 0.3, 0.6, 0.6, 0.4, 0.2]` chỉ ra 1 sự kiện, mở ở hàng thứ 3 và đóng ở `end_ms` của hàng thứ 5.
  - `test_open_until_end`: p `[0.7, 0.7, 0.7]` cho bản ghi closed tại `end_ms` cuối.
  - `test_levels_from_meta`: ngưỡng `(0.8, 0.5)`, max p bằng 0.85 nên `"red"`, bằng 0.6 nên `"yellow"`. Với max p 0.55 nhưng ngưỡng vàng 0.6 thì không có sự kiện.
  - `test_pair_event_goes_to_reacher`: lớp `PASSING_OBJECT` có `reacher_for = B` khi hàng thuộc ghế A, nên `seat_id == A` và `neighbor_seat_id == B`. Nếu `reacher_for` chỉ ra rằng chính B vươn sang A thì sự kiện thuộc B.
  - `test_feature_mismatch_raises`: `meta.feature_names` thiếu `phone_conf_max` thì `ValueError` có chuỗi `phone_conf_max`.
  - `test_load_roundtrip`: train nhanh bằng `train_core.train` trên dữ liệu tổng hợp, `save_model`, `LearnedCore.load`, rồi `predict` cho xác suất có tổng ≈ 1.
  - `test_domain_warning`: khung 640×360 so với 1280×720 lúc train thì không cảnh báo; 320×180 thì có.
- [ ] **Step 2:** `uv run pytest packages/proctoring_ai/tests/test_cctv_learned.py -q`. Expected: FAIL.
- [ ] **Step 3:** Viết `learned.py`.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: 7 passed.
- [ ] **Step 5:** Commit `feat(cctv): learned core inference with hysteresis and red/yellow levels`.

### Task 7: `process_video.py --core learned`

**Files:**
- Modify: `scripts/process_video.py`, gồm `process()`, `main()` và phần lượt kỹ.
- Test: `tests/test_process_video.py`

**Interfaces:**
- Consumes: `all_seat_windows`, `load_tracks` (Task 1); `video_features` (Task 3); `LearnedCore`, `domain_warning` (Task 6); `review_offline`.
- Produces:
  - Chữ ký mới: `process(..., core: str = "rules", model: Path | None = None)`.
  - CLI: `--core {rules,learned}` (mặc định `rules`), `--model DIR`.
  - Với `learned`:
    - Lượt 2 gọi `review_offline` trên `all_seat_windows` với `warmup_ms=0`, cache ghi vào `out/timeline.json.gz`.
    - Sau đó chạy `video_features`, ghi `out/features.csv`, gọi `LearnedCore.events`, rồi đưa qua `ReviewDeduper` và render cùng báo cáo như cũ.
    - Không gọi `assign_levels`, vì level đã có từ model.
    - `timing.json["config"]["core"] = "learned"` và `["model"] = tên thư mục.
    - `domain_warning` khác `None` thì in ra và đưa vào `report.html`.
  - Lỗi:
    - `--core learned` mà không có `--model`: `ProcessError("--core learned cần --model <thư mục model>")`.
    - Thư mục model thiếu file: `ProcessError` kèm thông điệp từ `LearnedCore.load`.

- [ ] **Step 1: Viết test.**
  - `test_learned_core_requires_model`: chạy `main([... , "--core", "learned"])` trả về mã lỗi khác 0, stderr có `--model`.
  - `test_learned_core_end_to_end`:
    - Đầu vào: fast runner giả với 2 ghế, detectors giả, video tổng hợp 6 s, và model tổng hợp train trong test (`train_core.train` + `save_model`) học được lớp `LOOK_AT_NEIGHBOR` khi `yaw_turn_share` cao.
    - Kỳ vọng: `events.jsonl` có ít nhất 1 bản ghi `stage == "learned"` và có `level`; tồn tại `annotated.mp4` khi có ffmpeg (nếu không có thì truyền `ffmpeg=None` như các test hiện có); `timing.json["config"]["core"] == "learned"`.
  - `test_learned_core_short_video`: video 2 s nên `events.jsonl` rỗng, process vẫn trả về timing.
  - `test_rules_core_unchanged`: các test hiện có của `process_video` vẫn pass, không phải sửa.
- [ ] **Step 2:** `uv run pytest tests/test_process_video.py -q`. Expected: 3 test mới FAIL.
- [ ] **Step 3:** Sửa `process_video.py`.
- [ ] **Step 4:** Chạy `uv run pytest -q`. Expected: toàn bộ suite pass, cùng `uv run ruff check .`.
- [ ] **Step 5:** Commit `feat(process_video): --core learned runs the learned detection core`.

### Task 8: Notebook xem dataset ảnh

**Files:**
- Create: `notebooks/kaggle_xem_dataset_anh.ipynb`

**Interfaces:**
- Produces: một notebook in ra:
  - Cây thư mục 3 cấp dưới `/kaggle/input` (bỏ `.mp4`), cùng số file theo đuôi.
  - Nếu có `data.yaml` hoặc `*.yaml`: nội dung đầy đủ, tức tên lớp.
  - Nếu có thư mục `labels/*.txt` dạng YOLO: số hộp theo từng lớp.
  - Nếu là thư mục phân lớp (`train/<lớp>/*.jpg`): số ảnh mỗi lớp.
  - Nội dung `dataset-metadata.json` hoặc README nếu có, để xem giấy phép.
  - Một ảnh ghép 4×4 mẫu ngẫu nhiên, lưu thành `/kaggle/working/mau.jpg`.

- [ ] **Step 1:** Viết notebook. Kiểm tra JSON hợp lệ bằng `python -c "import json;json.load(open('notebooks/kaggle_xem_dataset_anh.ipynb'))"`, rồi chạy riêng code các ô trên thư mục giả dựng trong scratchpad (một `data.yaml` và 2 nhãn YOLO). Expected: in đúng 2 lớp và số hộp.
- [ ] **Step 2:** Commit `feat(kaggle): notebook to inspect the exam image dataset`.

### Task 9: Gán nhãn, train và nghiệm thu trên 3 video (cần Nguyên chạy Kaggle)

Phần dữ liệu thật. Đây không phải TDD. Bằng chứng là bảng số liệu ghi vào `docs/evaluation-results.md`.

- [ ] **Step 1:** Nguyên chạy `kaggle_timeline.ipynb` trên dataset video-test-rule và video-test-deep, rồi gửi `timeline.zip`. Ghi `ratio` của tầng 1 cho từng video. Nếu ratio lớn hơn 3 thì báo Nguyên và thử lại với `--quality balanced`.
- [ ] **Step 2:** Claude xem từng ảnh ghép trong `sheets/<video>/` và ghi `data/labels/learned/<video>.jsonl`, với `labeler: "claude"`.
- [ ] **Step 3:** Đo độ khớp với 23 phán quyết ở `data/labels/EPf9O7UXRoM-review.yaml`. Một phán quyết khớp nhãn khi nhãn cùng ghế (đầu cách ≤ 0.05) và cùng mốc ±1.5 s. Phán quyết `true` phải ra đúng lớp, phán quyết `false` phải ra `NORMAL` hoặc lớp khác. Ghi tỷ lệ khớp. Nếu dưới 85% thì gán lại các lớp lệch.
- [ ] **Step 4:** Chọn ngẫu nhiên 50 đoạn (seed 0) thành một trang `phase3/duyet-nhan.jpg` trong project files để Nguyên duyệt. Sửa nhãn theo phản hồi.
- [ ] **Step 5:** Chạy `train_core.py --leave-one-out`. Clip 11 s gần như không có sự kiện, nên chỉ dùng làm dữ liệu train, không chấm riêng. Ghi bảng precision và recall từng lớp cho clip 1 và clip 2.
- [ ] **Step 6:** Train model cuối vào `models/learned_core/v1/`. Chạy `process_video.py --core learned` trên clip 1 bằng notebook Kaggle hoặc bằng timeline sẵn có. Chấm bằng `score_review` với các mục tiêu: bắt ≥ 8/9 ca đúng, số ảnh nhầm ≤ một nửa của lần chạy 2026-11-06, `false_red == 0`, ratio ≤ 3.
- [ ] **Step 7:** Commit nhãn, model và kết quả: `feat(cctv): learned core v1 trained on 3 videos`. Lưu ý kết quả trên clip 1 là kết quả leave-one-out; model cuối đã thấy clip 1 nên không dùng để chấm.
