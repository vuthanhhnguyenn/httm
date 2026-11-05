# Phase 2f: Ít báo nhầm hơn và tầng kỹ gần thời gian thực Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trên clip 1, tầng kỹ giữ 9/9 sự kiện đúng và giảm sự kiện nhầm từ 14 xuống ≤ 7. Tầng kỹ chạy nền cùng tầng nhanh, cho kết quả trong ≤ 60 s (p95) trên M1200, và tầng nhanh vẫn giữ ≥ 10 FPS.

**Architecture:** Phần A làm trước. Tầng kỹ lưu đầu ra model theo từng đoạn (`review_cache`), để `scripts/replay_review.py` chấm lại luật trong vài giây theo nhãn đúng/nhầm của Nguyên. Sau đó sửa luật trong `review_rules.py` và thêm `ReviewDeduper` để gộp sự kiện trùng giữa các đoạn. Phần B: tầng nhanh giữ khung gần nhất trong `FrameRing`. `SuspectTracker.poll` đẩy đoạn đã đóng vào `ReviewQueue`. `ReviewWorker` (một luồng nền) lấy tối đa 12 khung mỗi đoạn, chạy model theo batch (pose trên CUDA, vật thể trên OpenVINO card Intel) rồi trả sự kiện về `run_multi_replay`.

**Tech Stack:** Python 3.12, Ultralytics 8.4.160 (yolo11x-pose, yolo11x), OpenVINO (mới), OpenCV, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-11-05-phase2f-cctv-precision-speed-design.md`

## Global Constraints

- Model tầng kỹ vẫn là `yolo11x-pose` và `yolo11x` ở 960 px. Không đổi sang model m/l khi chưa có Nguyên đồng ý.
- Không train model và không thêm lớp vật thể.
- Không tắt luật `LOOK_AT_LAP`. Các ca đúng 43.2, 51.2, 59.0, 63.0 phải còn.
- Mỗi ngưỡng mới nằm trong dataclass cấu hình (`ReviewRuleConfig`, `CctvSuspectConfig`, `ReviewWorkerConfig`) và có một dòng chú thích lý do bằng lời. Không được dò số cho khớp clip 1.
- Sau mỗi task phần A, chạy replay và ghi số đúng/nhầm vào `docs/evaluation-results.md` (mục "Phase 2f").
- Toạ độ trong JSONL chuẩn hoá theo khung hình gốc (0..1). Chữ hiển thị cho người dùng bằng tiếng Việt.
- Không dùng FP16. Tầng nhanh không bao giờ phải dừng hay chờ vì tầng kỹ.
- Các giá trị chép từ spec: `dedupe_ms = 10000`, tối đa 12 khung mỗi đoạn (3 khung/giây trong đoạn, 2 khung khởi động), đoạn đang mở được gửi đi sau 10 s, `review_deadline_ms = 60000`, bộ đệm 20 s ở 5 khung/giây, batch tối đa 8, hàng chờ đọc video tối đa 8 khung.

## Review Focus

- **Đoạn nghi vấn bắt đầu trước khung cũ nhất còn trong `FrameRing`** (vì chờ lâu hoặc nằm ở đầu clip): nếu nguồn là file thì đọc lại từ file, nếu không thì `REVIEW_SKIPPED`. Không được trả về đoạn rỗng một cách im lặng. Test ở Task 10.
- **Khung thưa làm hỏng các luật "giữ đủ lâu"**: `gap_ms = 300` của luật cũ nhỏ hơn khoảng cách 333 ms giữa hai khung, nên mọi đoạn liên tục bị cắt vụn. Khoảng gap phải tính theo khoảng cách khung thật. Test ở Task 7.
- **Hết VRAM khi tầng kỹ chạy cùng tầng nhanh**: chia đôi batch, xuống tới batch 1 vẫn lỗi thì chuyển sang CPU. Tầng nhanh không bị ảnh hưởng. Test ở Task 8.
- **Máy không có card Intel hoặc không cài OpenVINO** (cả cloud): tự chuyển sang CPU, ghi lý do vào `timing.json`, không dừng. Test ở Task 9.
- **Video kết thúc khi hàng đợi còn đoạn**: `close(flush=True)` phải xử lý hết hoặc ghi `REVIEW_SKIPPED`. `events.jsonl` không được thiếu đoạn nào. Test ở Task 11.

---

### Task 1: Bộ kiểm tra hồi quy cho tầng kỹ (A1)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_cache.py`
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_score.py`
- Create: `scripts/replay_review.py`
- Create: `data/labels/EPf9O7UXRoM-review.yaml`
- Modify: `scripts/review_suspects.py` (thêm `--cache-dir`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_cache.py`, `packages/proctoring_ai/tests/multi/test_cctv_review_regression.py`
- Fixture: `packages/proctoring_ai/tests/fixtures/cctv_clip1_review/` (`suspects.jsonl`, `cache.json.gz`)

**Interfaces:**
- Produces: `save_window(path: Path, window: SuspectWindow, frames: Sequence[ReviewFrame]) -> None`, `load_cache(path: Path) -> dict[tuple[str, int, int], list[ReviewFrame]]` (khoá `(seat_id, start_ms, end_ms)`). File `cache.json.gz` ghép mọi đoạn, số làm tròn 4 chữ số.
- Produces: `Verdict(type: str, t_s: float, head: tuple[float, float], verdict: bool)`, `load_verdicts(path) -> list[Verdict]`, `score_review(records: Iterable[dict], verdicts: Sequence[Verdict]) -> dict` trả `{"true_found": int, "true_total": int, "false_opened": int, "unjudged": int, "missed": [str]}`. Một sự kiện `opened` khớp mục khi cùng `type`, `|t - t_s| ≤ 1.5` và khoảng cách đầu ≤ 0.05. `UNCLASSIFIED_SUSPECT` và sự kiện `merged` không được tính.
- Produces: `replay(suspects: Path, cache: Path, *, config: ReviewRuleConfig | None = None, max_frames: int | None = None) -> list[dict]`. Task 2 bổ sung khử trùng, Task 7 bổ sung chọn khung.

- [ ] **Step 1:** Viết `data/labels/EPf9O7UXRoM-review.yaml` từ bảng 2.1 của spec. Toạ độ `head` của mỗi mục chép từ `/mnt/project-files/phase2e/clip1/events.jsonl` (bản ghi `opened` cùng loại và cùng mốc). Mỗi ô PASSING_OBJECT nhầm phải có mục cho **cả hai** ghế.
- [ ] **Step 2:** Test round-trip `save_window` → `load_cache` (2 đoạn, pose và vật thể giữ nguyên tới 4 chữ số). Test `score_review` trên dữ liệu tự dựng: 1 đúng khớp, 1 nhầm khớp, 1 lệch 2 s là `unjudged`, 1 `merged` bị bỏ qua. Chạy, phải FAIL.
- [ ] **Step 3:** Cài đặt `review_cache.py` và `review_score.py`. Thêm `--cache-dir` vào `review_suspects.py`: khi có cờ này thì ghi `ReviewFrame` của từng đoạn qua `save_window`. Chạy test, phải PASS.
- [ ] **Step 4:** Tạo cache clip 1 trên cloud (chạy nền, khoảng 1 giờ trên CPU). Lệnh: `analyze_video_multi.py /mnt/project-files/EPf9O7UXRoM.mp4 --scene cctv --mode full --fps 5 --suspects --out storage/multi/p2f-clip1`, rồi `review_suspects.py ... --cache-dir`. Chép `suspects.jsonl` và `cache.json.gz` vào fixture. Nếu fixture lớn hơn 3 MB thì chỉ giữ pose của các ghế trong `window.seats` (vốn đã như vậy) và bỏ keypoint có độ tin cậy < 0.2.
- [ ] **Step 5:** Viết `scripts/replay_review.py <suspects.jsonl> <cache.json.gz> <labels.yaml> [--max-frames N]`, in ra `đúng 9/9, nhầm 14, chưa chấm N`. Viết test hồi quy `test_clip1_baseline`: `true_found == 9` và `false_opened == 14`. Test này xác nhận cache tái tạo đúng kết quả phase 2e. Nếu không ra 14 thì dừng lại tìm nguyên nhân (thứ tự khung, làm tròn số) trước khi đi tiếp.
- [ ] **Step 6:** Ghi mục "Phase 2f" vào `docs/evaluation-results.md` với số gốc. Commit.

### Task 2: Gộp trùng và chỉ báo người với tay (A2)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_dedupe.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (khối (b) passing)
- Modify: `scripts/review_suspects.py`, `scripts/replay_review.py` (cho mọi bản ghi đi qua bộ khử trùng)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_dedupe.py`, `test_cctv_review_rules.py`

**Interfaces:**
- Produces: `class ReviewDeduper(dedupe_ms: int = 10000)` có `admit(records: list[dict]) -> list[dict]`. Bản ghi `opened` của cùng `(seat_id, type)` trong vòng `dedupe_ms` kể từ lần mở trước được đổi thành `{"action": "merged", "merged_into": <id cũ>, ...}`, không có `evidence`. Bản ghi `closed` tương ứng bị bỏ. Hàm giữ trạng thái giữa các lần gọi theo thứ tự `video_offset_ms`.

- [ ] **Step 1:** Viết các test:
  - `test_same_seat_type_within_10s_merged`: mở ở 17200 và 17500 cùng ghế cùng loại → bản thứ hai là `merged`.
  - `test_after_10s_opens_again`: mở ở 17200 và 28000 → hai lần `opened`.
  - `test_passing_only_reacher`: ghế A có cổ tay vượt qua đường giữa hai tâm ghế, ghế B thì không → chỉ có PASSING_OBJECT cho A.
  - `test_passing_both_reach_farther_wins`: cả hai cùng với tay → chỉ báo người có cổ tay vượt xa đường giữa hơn.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt `ReviewDeduper`. Trong `review_window`, thay hai lời gọi `emit` đối xứng bằng một lời gọi cho người với tay. "Vượt" là khoảng cách theo trục x từ cổ tay tới đường giữa, tính về phía ghế kia, lấy giá trị lớn nhất trong đoạn. Không ai vượt thì báo ghế `main`. Chạy test, phải PASS. Chạy `pytest packages/proctoring_ai/tests/multi -q`, phải PASS hết.
- [ ] **Step 3:** Chạy replay, ghi số vào `docs/evaluation-results.md`. Nếu `true_found < 9` thì dừng lại sửa, không đi tiếp. Commit.

### Task 3: Vật thể phải được cầm, không nằm im (A3)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (khối (a) phone và (d) document, `ReviewRuleConfig`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_rules.py`

**Interfaces:**
- Produces: các trường `ReviewRuleConfig` `object_min_frames: int = 2`, `object_track_iou: float = 0.3`, `object_wrist_widths: float = 0.6`, `object_move_widths: float = 0.5`, `calculator_on_paper: bool = False`, cùng hàm `_alive(frames, label_set, box, cfg) -> set[int]`, trả chỉ số các khung có vật "sống".

- [ ] **Step 1:** Viết các test:
  - `test_still_phone_far_from_wrist_ignored`: điện thoại đứng yên ở cùng vị trí trong 12 khung, cách cổ tay 1 chiều rộng khung người → không có PHONE_DETECTED.
  - `test_held_phone_reported`: điện thoại cách cổ tay 0.3 chiều rộng, thấy ở 3 khung → có PHONE_DETECTED.
  - `test_single_frame_object_ignored`: tài liệu chỉ có ở 1 khung → không có DOCUMENT_DETECTED.
  - `test_moving_book_reported`: `book` dưới mép bàn, dịch chuyển 0.6 chiều rộng của nó → có DOCUMENT_DETECTED.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt. Các khung có vật "sống" là đầu vào bắt buộc của điều kiện phone và document đang có (giữ nguyên `phone_min_conf` và luật "cao hơn đầu thì là bàn trước"). Hai lần thấy là cùng một vật nếu IoU > `object_track_iou` giữa hai khung liền nhau có vật. Chạy test, phải PASS.
- [ ] **Step 3:** Chạy replay. Nếu 31.6 (máy tính bỏ túi) vẫn bị báo thì bật `calculator_on_paper`: `phone` có tâm nằm trong khung `book`/`paper` của cùng ghế, mở rộng 20%, thì không tính. Thêm test `test_phone_on_paper_ignored_when_enabled`. Ghi số. Commit.

### Task 4: "Tay chạm tay" tính theo cỡ người (A4)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (khối (b), `ReviewRuleConfig`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_rules.py`

**Interfaces:**
- Produces: các trường `passing_wrist_heights: float = 0.15` (khoảng cách cổ tay chia cho chiều cao khung người nhỏ hơn; thay cho `passing_wrist_widths` ở khối (b), còn `_transfers` vẫn giữ `passing_wrist_widths`), `passing_wrist_conf: float = 0.5`, `far_row_height: float = 0.12`, `far_row_inside_frames: int = 2`.

- [ ] **Step 1:** Viết các test:
  - `test_far_row_hands_meet_without_object_ignored`: hai ghế có khung cao 0.08, cổ tay chạm nhau trong 1 s, không có vật thể, cổ tay không vào khung ghế bên → không có PASSING_OBJECT.
  - `test_far_row_wrist_inside_other_seat_reported`: như trên, nhưng cổ tay A nằm trong khung ghế B ở 2 khung → có.
  - `test_near_row_hands_meet_reported`: khung cao 0.25, cổ tay chạm nhau → có.
  - `test_low_conf_wrist_ignored`: cổ tay có độ tin cậy 0.4 → không có.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, chạy test, phải PASS. Chạy replay: các ca 17.2, 20.1, 22.1 phải hết, còn 25.5 phải còn. Nếu mất 25.5 thì xem lại `passing_wrist_heights` bằng dữ liệu debug của ca đó, ghi lý do vào chú thích, rồi chạy lại. Ghi số. Commit.

### Task 5: `LOOK_AT_LAP` cần tay ở dưới bàn (A5)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (khối lap)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_rules.py`

**Interfaces:**
- Produces: trường `ReviewRuleConfig.lap_needs_both_wrists: bool = True`.

- [ ] **Step 1:** Viết các test:
  - `test_writer_one_hand_on_desk_not_lap`: đầu cúi 4 s, một cổ tay trên mép bàn và một cổ tay dưới → không có LOOK_AT_LAP.
  - `test_both_hands_below_desk_is_lap`: cả hai cổ tay dưới mép bàn trong 4 s → có.
  - `test_hands_hidden_is_lap`: đầu cúi 4 s, không thấy cổ tay nào → có.
  Chạy, phải FAIL (với test thứ 3, luật hiện tại cần thấy ít nhất một cổ tay).
- [ ] **Step 2:** Cài đặt: khi `lap_needs_both_wrists` bật, một khung tính là lap nếu đầu cúi và (cả hai cổ tay thấy được và đều dưới mép bàn, hoặc không thấy cổ tay nào). Chạy test, phải PASS.
- [ ] **Step 3:** Chạy replay. Nếu mất một trong các ca 43.2, 51.2, 59.0, 63.0 thì đặt mặc định `lap_needs_both_wrists = False`, ghi lý do vào chú thích và vào `docs/evaluation-results.md`. Ghi số. Commit.

### Task 6: Chốt phần A — ngưỡng hồi quy và clip 2

**Files:**
- Modify: `packages/proctoring_ai/tests/multi/test_cctv_review_regression.py`
- Modify: `docs/evaluation-results.md`

- [ ] **Step 1:** Thêm test `test_clip1_phase2f`: `true_found == 9` và `false_opened <= 7`. Chạy, phải PASS. Nếu không đạt thì báo số thật cho Nguyên, đừng nới test.
- [ ] **Step 2:** Chạy cả hai tầng trên clip 2 (`LUy81HI7wS0`). Nếu clip này không có trong project files thì hỏi Nguyên tải lên. Chạy trước bằng code `main` (worktree tạm) và sau bằng nhánh này, so số sự kiện tầng kỹ: số sau không được nhiều hơn số trước. Ghép ảnh các sự kiện thành `/mnt/project-files/phase2f/clip2/su-kien.jpg` để Nguyên chấm.
- [ ] **Step 3:** Ghi kết quả clip 1 và clip 2 vào `docs/evaluation-results.md`. Commit.

### Task 7: Ít khung hơn cho mỗi đoạn (B2)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_frames.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/suspects.py` (`SuspectWindow.peak_ms`, `_Span.peak`)
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (gap theo khoảng cách khung)
- Modify: `scripts/replay_review.py`, `scripts/review_suspects.py` (`--max-frames`, mặc định 12)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_frames.py`, `test_cctv_review_rules.py`, `test_cctv_review_regression.py`

**Interfaces:**
- Produces: `SuspectWindow.peak_ms: int | None = None`, có trong `to_json`/`from_json`. Mốc này là lúc có nhiều lý do nghi vấn cùng lúc nhất; nếu bằng nhau thì lấy lúc sớm nhất.
- Produces: `select_offsets(window: SuspectWindow, available: Sequence[int], *, per_s: float = 3.0, warmup_frames: int = 2, warmup_ms: int = 6000, max_frames: int = 12) -> list[int]`. Kết quả là các mốc lấy từ `available`, tăng dần, có `peak_ms` nếu có, và các khung trong đoạn được rải đều.
- Produces: trong `review_window`, khoảng gap thật là `max(cfg.gap_ms, int(1.5 * median_spacing))`, với `median_spacing` là trung vị khoảng cách giữa các khung. `phone_frames` được hiểu là `min(cfg.phone_frames, số khung trong phone_span_ms)`.

- [ ] **Step 1:** Viết các test:
  - `test_select_caps_and_keeps_peak`: đoạn 30 s, `available` 10 khung/giây → 12 mốc, trong đó có `peak_ms`, có đúng 2 mốc trước `start_ms`.
  - `test_select_short_window`: đoạn 1 s → 2 + 3 mốc.
  - `test_sparse_frames_keep_hold`: LOOK_AT_LAP với khung cách nhau 333 ms trong 4 s vẫn mở.
  - `test_peak_ms_roundtrip`.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, chạy test, phải PASS. Chạy `pytest packages/proctoring_ai/tests/multi -q`.
- [ ] **Step 3:** Chạy replay với `--max-frames 12` (chọn khung trong cache dày 10 khung/giây). Thêm test `test_clip1_sparse12`: `true_found == 9` và `false_opened <= 7`. Nếu không đạt thì thử 16 khung và ghi lại số. Mặc định chỉ đổi thành 16 khi 12 không đạt mà 16 đạt. Ghi số. Commit.

### Task 8: Gom batch ở tầng kỹ, chịu được hết VRAM (B3)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_detectors.py`
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_core.py`
- Modify: `scripts/review_suspects.py` (dùng `review_core`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_detectors.py`, `test_cctv_review_core.py`

**Interfaces:**
- Produces: `ReviewDetectors.pose_many(crops: Sequence[Any], transforms: Sequence[CropTransform]) -> list[list[dict]]` và `objects_many(...) -> list[list[ReviewObject]]`. Mỗi model chỉ một lời gọi `predict(source=list)` cho mỗi batch, `batch_size=8`. Khi gặp lỗi CUDA hết bộ nhớ (`torch.cuda.OutOfMemoryError` hoặc `RuntimeError` có chữ "out of memory"), chia đôi batch và thử lại. Xuống tới 1 mà vẫn lỗi thì chuyển model đó sang `cpu` và ghi `status[name] = "cpu: hết VRAM"`. `pose`/`objects` cũ trở thành lớp bọc một phần tử.
- Produces: `review_core.analyse_window(window, images: Sequence[tuple[int, Any]], detectors, *, config=None) -> tuple[list[dict], list[ReviewFrame]]`. Hàm cắt ảnh (`crop_region`, `crop_image`), chạy `*_many`, `assign_poses`, rồi `review_window`. `review_suspects.py` và `ReviewWorker` cùng gọi hàm này.

- [ ] **Step 1:** Viết các test với model giả (`loader=`):
  - `test_one_predict_per_model_per_batch`: 10 ảnh → mỗi model 2 lần gọi (8 + 2).
  - `test_oom_halves_then_cpu`: model giả ném OOM khi batch > 2 → chạy xong với batch 2. Model giả ném OOM mọi lúc trên cuda → chuyển sang `cpu`, có `status`.
  - `test_analyse_window_matches_single`: kết quả của `analyse_window` giống cách chạy từng khung cũ trên cùng dữ liệu giả.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, chạy test, phải PASS. Chạy `review_suspects.py` trên 2 đoạn của clip 1 và so `events.jsonl` với replay: phải giống nhau. Commit.

### Task 9: Chọn card cho từng model, OpenVINO trên card Intel (B4)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/devices.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_detectors.py` (`ModelSpec.device`)
- Modify: `configs/review_models.yaml` (`device: cuda` cho pose, `device: intel:gpu` cho `yolo11x`)
- Modify: `scripts/download_model.py` (`--export-openvino`)
- Modify: `scripts/analyze_video_multi.py`, `packages/proctoring_ai/src/proctoring_ai/analyzer_factory.py` (`--phone-device`)
- Modify: `pyproject.toml`, `uv.lock` (thêm `openvino`, Apache-2.0)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_devices.py`

**Interfaces:**
- Produces: `resolve_device(wanted: str, model_path: Path) -> tuple[str, Path, str]` trả về `(device, path model sẽ nạp, lý do)`.
  - `auto` → `cuda:0` nếu có CUDA, không thì `cpu`.
  - `intel:gpu` → `(intel:gpu, <stem>_openvino_model/)` nếu đã import được `openvino`, thư mục xuất đã có và `openvino.Core().available_devices` có `GPU`. Thiếu một trong ba thì trả `("cpu", model_path, "<lý do bằng tiếng Việt>")`.
  - `cuda` mà không có CUDA → `cpu`.
- Produces: `ModelSpec.device: str = "auto"`. `ReviewDetectors.status[name]` ghi thiết bị thật, ví dụ `"ok (intel:gpu)"` hoặc `"ok (cpu: không có card Intel)"`. `timing.json` → `models` lấy giá trị này.
- Produces: `--phone-device {auto,cuda,intel:gpu,cpu}` của `analyze_video_multi.py`, truyền tới máy dò điện thoại của tầng nhanh qua `resolve_device`.

- [ ] **Step 1:** Viết test cho `resolve_device` bằng cách monkeypatch import `openvino`, `torch.cuda.is_available` và sự tồn tại của thư mục xuất: đủ 4 nhánh trên, và lý do phải là chuỗi tiếng Việt không rỗng khi chuyển sang CPU. Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt. Kiểm tra trong mã nguồn Ultralytics 8.4.160 đã khoá rằng `device="intel:gpu"` được hỗ trợ cho model OpenVINO. Nếu không được thì truyền `device` qua `YOLO(path, task=...)` theo đúng API của bản đó, rồi ghi cách làm vào chú thích. Chạy test, phải PASS.
- [ ] **Step 3:** Thêm `openvino` bằng `uv add openvino`. Ghi license vào `docs/licensing.md`. `download_model.py --export-openvino` xuất `yolo11x` (và model phone của tầng nhanh) bằng `YOLO(pt).export(format="openvino", imgsz=960)`, bỏ qua nếu đã có. Chạy trên cloud: thấy lý do CPU trong `timing.json`, lượt chạy không lỗi. Commit.

### Task 10: `FrameRing` và `ReviewQueue` (B1, B5 — phần không có luồng)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/frame_ring.py`
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_queue.py`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_frame_ring.py`, `test_cctv_review_queue.py`

**Interfaces:**
- Produces: `FrameRing(span_ms: int = 20000, fps: float = 5.0, jpeg_quality: int | None = None)` có:
  - `add(offset_ms: int, image: ndarray) -> None`: bỏ khung cách khung giữ trước dưới `1000/fps` ms, và bỏ khung cũ hơn `span_ms`.
  - `oldest_ms() -> int | None`.
  - `get(offset_ms: int) -> tuple[int, ndarray] | None`: trả khung gần nhất trong ±`1000/fps`.
  - `offsets() -> list[int]`.
- Produces: `ReviewQueue(deadline_ms: int = 60000, max_frames: int = 12)` có `push(window: SuspectWindow, now_wall_ms: int) -> None` (gộp với đoạn đang chờ của cùng ghế nếu chồng nhau), `pop() -> SuspectWindow | None` và `expired(now_wall_ms: int) -> list[SuspectWindow]`. Ưu tiên: có `"RULE_EVENT"` trong `reasons`, rồi nhiều lý do hơn, rồi `push` sớm hơn. Thời hạn tính theo giờ thật (wall clock), không theo giờ video.
- Produces: `EventType.REVIEW_SKIPPED` trong `proctoring_core/types.py`, nhãn `"Chưa kiểm tra kỹ"` trong `proctoring_runtime/messages.py`, trọng số rủi ro `0.0` trong `risk/engine.py`. Thêm hàm `skipped_record(window: SuspectWindow, reason: str) -> dict` trong `review_queue.py`, với `reason` thuộc `"deadline" | "frames_gone" | "shutdown"`.

- [ ] **Step 1:** Viết các test:
  - Ring: giữ đúng 5 khung/giây; khung cũ hơn 20 s bị bỏ; `get` lệch quá 200 ms trả `None`; chế độ JPEG trả ảnh cùng kích thước.
  - Queue: đoạn có RULE_EVENT ra trước; hai đoạn chồng nhau của cùng ghế thành một; đoạn chờ 60001 ms nằm trong `expired` và không còn trong `pop`.
  - `skipped_record` có `type == "REVIEW_SKIPPED"`, `stage == "review"`, `video_offset_ms == window.peak_ms or window.start_ms`.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, chạy test, phải PASS. Chạy cả `pytest -q` (vì có EventType mới). Commit.

### Task 11: `ReviewWorker` chạy nền và nối vào tầng nhanh (B1, B5)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_worker.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/suspects.py` (`SuspectTracker.poll`)
- Modify: `packages/proctoring_runtime/src/proctoring_runtime/multi_replay.py` (tham số `side_tasks`)
- Modify: `scripts/analyze_video_multi.py` (`--review-live`, `--ring-jpeg`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_worker.py`, `test_cctv_suspects.py`, `packages/proctoring_runtime/tests/test_multi_replay*.py` (file đang có)

**Interfaces:**
- Consumes: `FrameRing`, `ReviewQueue`, `skipped_record` (Task 10); `analyse_window` (Task 8); `select_offsets` (Task 7); `ReviewDeduper` (Task 2).
- Produces: `SuspectTracker.poll(offset_ms: int, *, max_open_ms: int = 10000) -> list[SuspectWindow]` trả các đoạn đã đóng (kết thúc + `pad_ms` < `offset_ms`), cùng phần 10 s đầu của đoạn còn mở lâu. Đoạn đã trả rồi thì không trả lại; phần sau của đoạn mở lâu bắt đầu từ chỗ đã cắt. `finish()` chỉ trả phần chưa được `poll` trả.
- Produces: `ReviewWorker(detectors, ring: FrameRing, *, video_path: Path | None, config: ReviewWorkerConfig, clock=time.monotonic)` có:
  - `submit(windows: list[SuspectWindow]) -> None`.
  - `drain() -> list[dict]`: các bản ghi đã xong, đã qua `ReviewDeduper`.
  - `close(flush: bool = True, timeout_s: float = 600) -> list[dict]`.
  - `stats() -> dict`, gồm `review_latency_ms` {p50, p95, max}, `review_queue_max`, `review_skipped`.
  
  Worker chạy trên một `threading.Thread`. Khung lấy từ `ring`; nếu `window.start_ms - warmup_ms < ring.oldest_ms()` thì đọc lại từ `video_path`, còn nếu không có `video_path` thì tạo `skipped_record(..., "frames_gone")`. `ReviewWorkerConfig` gồm `max_frames=12, per_s=3.0, warmup_frames=2, warmup_ms=6000, deadline_ms=60000, dedupe_ms=10000`.
- Produces: `run_multi_replay(..., side_tasks: Sequence[SideTask] = ())`, với `SideTask` là Protocol có `on_frame(frame: dict, offset_ms: int) -> list[dict]` và `finish(offset_ms: int) -> list[dict]`. Bản ghi trả về được ghi vào `events.jsonl` như sự kiện luật (ảnh bằng chứng đã có sẵn, `evidence` đã điền). `summary()` của side task (nếu có) được gộp vào `timing.json`. Lớp `LiveReview(SideTask)` trong `review_worker.py` nối ring, tracker.poll và worker lại với nhau.

- [ ] **Step 1:** Viết các test (detector giả, đồng hồ giả):
  - `test_poll_returns_closed_once`: một đoạn đóng chỉ được `poll` trả một lần, và `finish` không trả lại.
  - `test_poll_splits_long_open_span_at_10s`.
  - `test_worker_reviews_from_ring`: 3 đoạn → `drain` có bản ghi, detector không đọc file.
  - `test_worker_reads_file_when_ring_too_new` và `test_worker_skips_when_no_file`.
  - `test_deadline_skips_with_record`: đồng hồ nhảy 61 s → có `REVIEW_SKIPPED` với reason `deadline`.
  - `test_close_flush_leaves_no_window_untracked`: số đoạn đã gửi bằng số đoạn có bản ghi review cộng số bản ghi skipped.
  - `test_side_task_records_written`: `run_multi_replay` với một side task giả ghi bản ghi vào `events.jsonl` và gộp `summary` vào `timing.json`.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, chạy test, phải PASS. Chạy cả `pytest -q`.
- [ ] **Step 3:** Chạy trên cloud: `analyze_video_multi.py /mnt/project-files/EPf9O7UXRoM.mp4 --scene cctv --mode full --fps 5 --suspects --review-live`. Chấm `events.jsonl` với `review_score` (lọc `stage == "review"`): `true_found == 9` và `false_opened <= 7`. Ghi `review_latency_ms` (trên CPU, chỉ để tham khảo). Commit.

### Task 12: Tầng nhanh gom batch và đọc video ở luồng riêng (B6)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/pose_detector.py` (`detect_many`)
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/tiling.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/pipeline.py` (`gather` khi khác card)
- Create: `packages/proctoring_ai/src/proctoring_ai/capture/prefetch_source.py`
- Modify: `scripts/analyze_video_multi.py` (bọc nguồn bằng `PrefetchFrameSource`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_tiling.py` (mới hoặc đang có), `test_pipeline.py`, `packages/proctoring_ai/tests/test_prefetch_source.py`

**Interfaces:**
- Produces: `YoloPoseDetector.detect_many(frames: Sequence[dict]) -> list[list[dict]]`, một lời gọi `predict(source=[...])`, mỗi kết quả được `parse_pose_result` theo `frame_size` của chính ảnh đó. `TiledPoseDetector.detect` dùng `detect_many` nếu `base` có hàm này, không thì quay về cách cũ.
- Produces: `CctvPipeline(..., parallel_objects: bool = False)`. Khi bật, pose và phone chạy bằng `asyncio.gather`. `build_cctv_pipeline` bật cờ này khi thiết bị thật của phone khác thiết bị thật của pose.
- Produces: `PrefetchFrameSource(source: FrameSource, max_queue: int = 8)` có cùng giao diện `open/read/close`. Một task nền đọc trước vào `asyncio.Queue(maxsize=8)`. `read()` trả `None` khi hết nguồn, và ném lại lỗi của nguồn ở lần `read()` tiếp theo.

- [ ] **Step 1:** Viết các test:
  - `test_tiles_one_predict_call`: base giả đếm số lần gọi → 1 lần `detect_many` với 3 ảnh ở khung có ô cắt, và 1 ảnh ở khung không có. Kết quả giống cách cũ.
  - `test_parallel_objects_overlap`: hai detector giả, mỗi cái chờ 50 ms → `detect` mất dưới 90 ms khi bật cờ, và ≥ 100 ms khi tắt.
  - `test_prefetch_order_end_and_error`.
  Chạy, phải FAIL.
- [ ] **Step 2:** Cài đặt, chạy test, phải PASS. Chạy clip 1 trên cloud ở `--fps 5`: số sự kiện tầng nhanh và độ phủ `eval_cctv.py --suspects` (6/6) giống trước khi sửa. Ghi FPS trên CPU trước và sau. Commit.

### Task 13: Đo trên M1200 và báo kết quả

**Files:**
- Create: `/mnt/project-files/phase2f/do-m1200.ps1`
- Modify: `docs/evaluation-results.md`

- [ ] **Step 1:** Viết `do-m1200.ps1` theo mẫu `phase2e/chay-2-video-gpu.ps1`. Script cài `openvino` vào `.venv-gpu` bằng `uv pip install --python .venv-gpu\Scripts\python.exe openvino` (vì `.venv-gpu` không có pip), chạy `download_model.py --export-openvino`, rồi chạy `analyze_video_multi.py EPf9O7UXRoM.mp4 --scene cctv --mode realtime --suspects --review-live --phone-device intel:gpu` và in `timing.json`.
- [ ] **Step 2:** Đẩy nhánh, mở PR vào `main` (không merge), nhờ Nguyên chạy script và gửi `timing.json`. Đạt khi tầng nhanh ≥ 10 FPS và `review_latency_ms.p95` ≤ 60000. Nếu không đạt thì báo số đo và đề xuất dùng model l hoặc m (spec 6.4), chờ Nguyên quyết định.
- [ ] **Step 3:** Ghi số đo M1200 vào `docs/evaluation-results.md`. Commit.
