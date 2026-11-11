# Lõi luật v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Gán người vào ghế theo dõi qua các khung, đo góc đầu bằng HHP-Net, chọn loại sự kiện theo bảng ưu tiên, để cảnh báo đúng loại trên clip 1 (dev) và clip 2 (giữ lại).

**Architecture:** Cache lưu người thô mỗi khung (P0) để đổi cách gán chỉ cần phát lại, không chạy lại model. Theo dõi người (P1) cho chủ ghế ổn định. Yaw HHP-Net (P2) và bảng chọn loại (P3) chạy trên khung xương đã gán. Mỗi thay đổi hành vi là một cờ trong `ReviewRuleConfig`, mặc định tắt cho tới khi qua cổng đo.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), pytest, ruff, mypy, OpenVINO (CPU), tf2onnx + onnxruntime (chỉ để chuyển HHP-Net, một lần).

**Spec:** `docs/superpowers/specs/2026-11-11-rules-core-v2-design.md`

## Global Constraints

- Không dùng model ngôn ngữ-hình ảnh; chạy CPU; thời gian chạy luật không quá 1,5 lần hiện tại.
- Ảnh, video, crop học sinh không vào repo; nhãn chỉ có toạ độ và kết luận. File lớn ghi vào `/mnt/project-files/rules-core/`.
- Commit kết thúc bằng đúng hai dòng: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` và `Claude-Session: https://claude.ai/code/session_01L3BFxLgvrqyRet2ft36w8e`. GITHUB_TOKEN không xuất hiện trong URL, argv, `.git/config`, output.
- Làm trên nhánh `claude/project-thread-g4frkm`, PR #43; chỉ gộp khi Nguyên yêu cầu.
- Một tiến trình model một lúc (giới hạn bộ nhớ chung; chạy hai cái là bị kill). Dựng timeline dùng `--chunk-seats 3 --resume` và giữ một tác vụ nền được theo dõi để container không dừng.
- ĐO: `.venv/bin/python scripts/process_video.py <video> --all-seats --reuse-fast --device intel:cpu --out /home/claude/rc/<clip>` rồi `scripts/score_alerts.py EVENTS data/labels/<stem>-*.yaml`. Video: `/home/claude/tl/EPf9O7UXRoM.mp4` (clip 1), `/home/claude/tl/LUy81HI7wS0.mp4` (clip 2).
- Luật dừng: một bước làm "bắt được" clip 1 tụt quá 1 sự kiện so với 10/14 thì không bật mặc định; ghi vào ledger và notes.

## Review Focus

- Người đổi chỗ, đứng dậy hoặc mất khung xương vài giây: ghế không được "nhảy" sang người khác, và không được có đầu giả.
- Hai người ngồi sát cùng cúi (trao bài) không được báo thành nhìn đùi.
- Cache cũ (không có `people`) vẫn đọc được.
- ONNX HHP-Net phải khớp SavedModel (< 0,5°) và xử lý thiếu điểm mặt (độ tin 0).
- Cờ mới tắt thì kết quả y hệt hiện tại (replay hồi quy `test_clip1_phase2f` vẫn 9/9).

### Task 1: Cache lưu người thô

**Files:** Modify `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (`ReviewFrame`), `review_cache.py`, `review_offline.py` (`flush`). Test: `packages/proctoring_ai/tests/multi/test_cctv_review_cache.py`.

**Interfaces:**
- Produces: `ReviewFrame.people: Sequence[Mapping[str, Any]] = ()` (mỗi mục có `bbox` và `keypoints`, toạ độ chuẩn hoá toàn khung như `found` từ `pose_many`); JSON key `"people"`.

- [ ] Viết test: `_frame_json` rồi `_frame_from` giữ nguyên `people`; một dòng cache cũ không có khoá `people` vẫn đọc ra `people == ()`. Chạy, thấy FAIL.
- [ ] Thêm trường và (de)serialize; làm tròn 4 chữ số như các trường khác.
- [ ] `flush()` trong `review_offline.py` truyền `people=found` (chỉ những người có tâm box trong `regions[index]`) vào `ReviewFrame`.
- [ ] Chạy `pytest packages/proctoring_ai/tests/multi/test_cctv_review_cache.py test_cctv_review_offline.py`, rồi cả bộ, ruff, mypy. Commit `feat(cctv): cache keeps every detected person`.

### Task 2: Phát lại từ người thô

**Files:** Create `scripts/reassign_timeline.py`. Modify `review_offline.py` hoặc nơi `process_video` đọc cache để cho phép gán lại. Test: `test_cctv_review_cache.py`.

**Interfaces:**
- Consumes: `ReviewFrame.people`, `assign_poses(seats, people, by_body=False)`.
- Produces: `reassign_frames(frames, assigner) -> list[ReviewFrame]` (cùng `seats/objects/aspect`, `poses` tính lại từ `people`).

- [ ] Test: với khung có `people` giả, `reassign_frames(..., assign_poses)` ra đúng `poses` như `assign_poses` trực tiếp; khung không có `people` giữ nguyên `poses`.
- [ ] Cài `reassign_frames`; thêm cờ `process_video --reassign {old,tracked}` (mặc định không đụng cache).
- [ ] Commit `feat(cctv): replay seat assignment from cached people`.

### Task 3: Dựng lại timeline clip 1 và clip 2 (chạy, không code)

- [ ] Dựng `--chunk-seats 3 --resume` cho clip 1 vào `/home/claude/rc/clip1-v2` (~100 phút) rồi clip 2 `/home/claude/rc/clip2-v2` (~67 phút), từng cái một, có tác vụ nền được theo dõi.
- [ ] Cổng: phát lại bằng gán cũ (`--reassign old`) cho clip 1 ra bắt được 10/14 và 14 cảnh báo (sai số dựng < 0,02: chấp nhận ±1 cảnh báo). Ghi dòng "P0" vào `docs/superpowers/notes/2026-11-11-rules-core-accuracy.md`. Nếu không đạt, dừng và báo.

### Task 4: Theo dõi người

**Files:** Create `packages/proctoring_ai/src/proctoring_ai/multi/cctv/person_tracks.py`; test `packages/proctoring_ai/tests/multi/test_cctv_person_tracks.py`.

**Interfaces:**
- Produces: `link_people(frames_people: Sequence[Sequence[Mapping[str, Any]]], offsets_ms: Sequence[int]) -> list[list[int]]` (ID track cho từng người từng khung; nối bằng IoU box + khoảng cách giữa hai vai, tham lam, hai chiều thời gian, ngắt khi mất > 1 s).

- [ ] Test: hai người đứng yên được ID ổn định 50 khung; một người mất 5 khung rồi hiện lại vẫn cùng ID; hai người đổi chỗ không đổi ID chéo (khoảng cách vai); người mới xuất hiện cho ID mới.
- [ ] Cài; commit `feat(cctv): link detected people across frames`.

### Task 5: Chủ ghế theo track

**Files:** Modify `review_rules.py` (`ReviewRuleConfig.assign_tracked: bool = False`, `owner_window_ms: int = 3000`, `owner_switch_ms: int = 1500`), `person_tracks.py` (`assign_tracked`), `review_offline.py`/`reassign_timeline.py`. Test: `test_cctv_person_tracks.py`.

**Interfaces:**
- Produces: `assign_tracked(frames: Sequence[ReviewFrame], track_ids, cfg) -> list[ReviewFrame]`: chủ ghế = track có tổng phần chung với ô ghế (chia diện tích box người) lớn nhất trong cửa sổ `owner_window_ms`; đổi chủ chỉ khi track khác hơn hẳn liên tục ≥ `owner_switch_ms`. Khung mà chủ ghế không được phát hiện: ghế **không có** khung xương (không điền box ghế làm đầu).

- [ ] Test: đầu hàng sau lọt vào ô ghế trong 1 s không cướp ghế; chủ ghế mất xương 2 s thì ghế không có `poses`, không có đầu giả; chủ ghế thật sự đổi người (đứng dậy, người khác ngồi) thì đổi sau ≥ `owner_switch_ms`.
- [ ] Cài, nối vào `--reassign tracked`.
- [ ] ĐO clip 1 với `assign_tracked`: cổng bắt ≥ 10/14 và (tuỳ chọn) bật cùng `heads_same_row_only` (patch `/home/claude/rc/task5b-heads-same-row.patch`, `git apply`) cho bắt ≥ 10/14 và giảm nhầm chụm đầu giây 42, 44, 68. Ghi notes. Commit `feat(cctv): seat owner from person tracks (off by default)`; đặt mặc định bật chỉ nếu qua cổng.

### Task 6: Góc đầu HHP-Net

**Files:** Create `scripts/export_hhp_onnx.py` (một lần, trong venv riêng `/home/claude/ext/tfvenv`), `packages/proctoring_ai/src/proctoring_ai/multi/cctv/head_pose.py`, model `models/pretrained/hhp_net.onnx` (nhỏ, MIT, ghi nguồn trong README); sửa `rules.py`/`review_rules.py` (`ReviewRuleConfig.yaw_source: str = "index"`, giá trị `"hhp"` hoặc `"both"`; `yaw_deg_threshold=25`, `yaw_hold_ms=800`). Test: `packages/proctoring_ai/tests/multi/test_cctv_head_pose.py`.

**Interfaces:**
- Produces: `hhp_yaw_pitch(kps: Keypoints, aspect: float) -> tuple[float, float, float] | None` (yaw, pitch, độ bất định yaw; None nếu < 3 điểm mặt đủ tin cậy), chuẩn hoá đầu vào đúng như `normalize_wrt_maximum_distance_point` của HHP-Net (nhân x với `aspect` trước, điểm thiếu = (0, 0, 0)).

- [ ] Xuất ONNX bằng tf2onnx từ SavedModel `/home/claude/ext/HHP-Net/models/model_w300_biwi`; test: sai lệch yaw/pitch giữa ONNX và SavedModel < 0,5° trên 50 đầu vào lấy từ cache clip 1 (kiểm trong script xuất, ghi số vào notes).
- [ ] Test `hhp_yaw_pitch`: ảnh thiếu mũi ra None; mặt chính diện giả ra |yaw| < 10°.
- [ ] Luật quay: |yaw − mốc ghế| > `yaw_deg_threshold` liên tục ≥ `yaw_hold_ms` **và** chỉ số mũi–vai cùng chiều ở chế độ `both`; pitch chỉ loại cúi sâu. Độ bất định chỉ làm trọng số.
- [ ] Quét θ ∈ {20, 25, 30}, hold ∈ {0,8; 1,2} trên clip 1; cổng: bắt được các lần quay giây 24, 27, 44 và số cảnh báo nhầm tăng ≤ 20%. Ghi notes. Commit.

### Task 7: Bảng chọn loại

**Files:** Modify `review_rules.py` (hàm `choose_type(evidence) -> str` và dùng ở nơi hiện chọn loại cho cảnh báo), `ReviewRuleConfig.type_table: bool = False`. Test: `test_cctv_review_rules.py`.

**Interfaces:**
- Produces: `choose_type(evidence: Mapping[str, Any]) -> EventType`, thứ tự ưu tiên PHONE > PASSING > HEADS_TOGETHER (cùng hàng) > LOOK_AT_NEIGHBOR > LOOK_AT_LAP > TURN_AROUND.

- [ ] Test từng hàng ưu tiên bằng khung giả: hai người cúi sát + cổ tay vào ô người kia → PASSING chứ không LAP; hai đầu khác hàng quay vào nhau → LOOK_AT_NEIGHBOR; cùng hàng và xích lại → HEADS_TOGETHER.
- [ ] Cài; ĐO "đúng loại" clip 1 (cổng: tăng so với 4, không tụt bắt được). Ghi notes. Commit.

### Task 8: Clip 2 giữ lại, tổng kết

- [ ] Chạy ĐO clip 2 với các cờ đã qua cổng, chấm bằng `data/labels/LUy81HI7wS0-judging.yaml` (cần Nguyên chấm 6 ô trước). Chỉ ghi số, không chỉnh luật theo clip 2.
- [ ] Bảng "Bước v2" trong notes: bắt được / đúng / đúng loại / đỏ / thời gian cho từng cờ trên cả hai clip, kèm sai số ±8–10%.
- [ ] Đánh giá toàn nhánh bằng một người xem độc lập (model mạnh nhất), sửa một lượt, rồi báo Nguyên. Sau đó mới tới thử mấp máy miệng.
