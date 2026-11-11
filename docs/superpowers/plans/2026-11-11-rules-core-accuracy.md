# Tối ưu lõi luật CCTV lên 90% đúng — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đưa chế độ `--all-seats` lên bắt được ≥ 90% và đúng ≥ 90% bằng cách sửa từng nhóm lỗi của luật tầng kỹ, đo trước/sau mỗi bước.

**Architecture:** Mỗi cải tiến là một cờ mới trong `ReviewRuleConfig` (`review_rules.py`), bật mặc định khi số đo tốt. Đo bằng cách chạy lại luật trên dòng thời gian đã dựng (`--all-seats --reuse-fast`), chấm bằng `scripts/score_alerts.py` mới, ghi từng dòng vào sổ ghi chép. Chỉ bước lật ảnh (Task 7) cần dựng lại dòng thời gian.

**Tech Stack:** Python 3.12, numpy, OpenCV, Ultralytics YOLO (OpenVINO trên CPU), pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-11-11-rules-core-accuracy-design.md`

## Global Constraints

- Không dùng model ngôn ngữ-hình ảnh (API ngoài hay model mở).
- Video học sinh là dữ liệu cá nhân: tờ chấm và ảnh chỉ ghi vào `/mnt/project-files/rules-core/`, không vào repo; nhãn trong repo chỉ có tọa độ và kết luận.
- Ghép cảnh báo với nhãn: khoảng cách đầu ≤ 0,06, cửa sổ [đầu − 2 s, cuối + 2 s].
- Đích: bắt được ≥ 90%, đúng ≥ 90%, thời gian ≤ 1,5 lần hiện tại.
- Luật dừng: bước nào làm tụt "bắt được" trên tập dev quá 1 sự kiện thì sửa ngưỡng hoặc bỏ phần đó trước khi sang bước sau.
- Tập giữ lại (clip 2) chỉ chạy để ghi số, không nhìn để chỉnh luật.
- Nhánh `claude/project-thread-g4frkm`; trailer commit `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` và `Claude-Session: https://claude.ai/code/session_01L3BFxLgvrqyRet2ft36w8e`. Không gộp PR khi Nguyên chưa bảo.
- Trước mỗi lần đẩy: `pytest -q` và `mypy` xanh.
- Container khởi động lại khi không có tác vụ nền: lệnh dài chạy bằng `nohup` cùng Monitor.

## Review Focus

- Ghế không có người (không có khung xương) trong cả cửa sổ: mọi luật mới phải bỏ qua, không lỗi. Test trong Task 3 và Task 6.
- Khung xương thiếu khuỷu tay (độ tin thấp): LOOK_AT_LAP không được mở. Test trong Task 3.
- Điện thoại chỉ xuất hiện 1 khung trên giấy: phân số trên giấy tính theo số lần thấy, không chia cho 0. Test trong Task 4.
- Hai người chồng nhau trong một ô ghế: `assign_poses` không gán một người cho hai ghế. Test trong Task 6.
- Ảnh cắt lật mà model không thấy ai: ghép lật trả về đúng kết quả ảnh gốc. Test trong Task 7.

## Thư mục đo (ngoài repo)

- `/home/claude/rc/clip1`: chép từ `/home/claude/demo1` (có `fast/`, `suspects.jsonl`, `tracks.jsonl`, `timeline.json.gz`).
- `/home/claude/rc/clip2`: chép `/home/claude/tl/clip2` vào; nếu thiếu `fast/` thì lần chạy đầu sẽ chạy lại tầng nhanh.
- Video: `/home/claude/tl/EPf9O7UXRoM.mp4`, `/home/claude/tl/LUy81HI7wS0.mp4`.
- Lệnh đo một clip (dùng ở mọi task, gọi là **ĐO**):

```
python scripts/process_video.py <video> --all-seats --reuse-fast --device intel:cpu --out /home/claude/rc/<clip> --overwrite
python scripts/score_alerts.py /home/claude/rc/<clip>/events.jsonl data/labels/<video-stem>-*.yaml
```

---

### Task 1: Công cụ chấm chính thức và số đo gốc

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/alert_score.py`
- Create: `scripts/score_alerts.py`
- Create: `docs/superpowers/notes/2026-11-11-rules-core-accuracy.md`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_alert_score.py`

**Interfaces:**
- Consumes: `review_score.load_verdicts(path) -> list[Verdict]` (Verdict có `type, t_s, heads, verdict`).
- Produces: `score_alerts(records: Iterable[dict], verdicts: Sequence[Verdict], *, max_head: float = 0.06, pad_s: float = 2.0) -> dict` với khóa `caught`, `true_total`, `alerts`, `alerts_correct`, `alerts_judged`, `false_flagged`, `false_total`, `red`, `by_type` (dict type → `{alerts, correct}`). `scripts/score_alerts.py EVENTS LABELS... [--json]`.

Định nghĩa (khớp `anyalert.py` hiện tại, mở rộng thêm độ đúng):
- Cảnh báo = bản ghi `action == "opened"` có `head`, loại khác `UNCLASSIFIED_SUSPECT`, `REVIEW_SKIPPED`.
- Cảnh báo phủ một nhãn khi đầu cách một trong `heads` ≤ `max_head` và `start − pad_s ≤ t_s ≤ end + pad_s` (end lấy từ bản ghi `closed` cùng `id`, thiếu thì start + 3 s). Không xét loại.
- `caught` = số nhãn đúng được phủ; `false_flagged` = số nhãn nhầm được phủ.
- Một cảnh báo là **đúng** nếu phủ ít nhất một nhãn đúng, **nhầm** nếu chỉ phủ nhãn nhầm, **chưa chấm** nếu không phủ nhãn nào. `alerts_correct` / `alerts_judged` là độ đúng.

- [ ] **Step 1: Viết test hỏng**

```python
def test_alert_covering_a_true_label_is_correct():
    v = [Verdict("LOOK_AT_LAP", 10.0, ((0.5, 0.5),), True), Verdict("PHONE_DETECTED", 30.0, ((0.2, 0.2),), False)]
    recs = [opened("a", "LOOK_AT_LAP", 9000, (0.52, 0.5)), closed("a", 12000),
            opened("b", "PHONE_DETECTED", 29000, (0.2, 0.21)), closed("b", 29500),
            opened("c", "TURN_AROUND", 50000, (0.9, 0.9))]
    s = score_alerts(recs, v)
    assert (s["caught"], s["true_total"], s["false_flagged"], s["false_total"]) == (1, 1, 1, 1)
    assert (s["alerts"], s["alerts_correct"], s["alerts_judged"]) == (3, 1, 2)

def test_head_farther_than_006_does_not_cover():
    v = [Verdict("LOOK_AT_LAP", 10.0, ((0.5, 0.5),), True)]
    assert score_alerts([opened("a", "LOOK_AT_LAP", 10000, (0.57, 0.5))], v)["caught"] == 0

def test_window_padding_two_seconds_after_close():
    v = [Verdict("LOOK_AT_LAP", 14.0, ((0.5, 0.5),), True)]
    recs = [opened("a", "LOOK_AT_LAP", 9000, (0.5, 0.5)), closed("a", 12000)]
    assert score_alerts(recs, v)["caught"] == 1
```

(`opened(id, type, ms, head)` và `closed(id, ms)` là hàm dựng bản ghi trong file test.)

- [ ] **Step 2: Chạy, thấy hỏng** — `pytest packages/proctoring_ai/tests/multi/test_cctv_alert_score.py -q` → ImportError.
- [ ] **Step 3: Viết `alert_score.py` và `scripts/score_alerts.py`** (in một dòng tóm tắt và bảng theo loại; `--json` in dict).
- [ ] **Step 4: Chạy test** → PASS.
- [ ] **Step 5: Số đo gốc.** Dựng `/home/claude/rc/{clip1,clip2}` như trên, chạy **ĐO** cho clip 1. Expected: khớp số cũ (bắt được 8/9, nhầm bị phủ 3/14, đỏ 2). Ghi bảng đầu tiên vào sổ ghi chép với các cột: Bước, Bắt được, Đúng/đã chấm, Nhầm bị phủ, Đỏ, Tổng cảnh báo, Thời gian luật (s), Ghi chú. Clip 2 chạy một lần để có `events.jsonl` cho Task 2 (chưa có nhãn nên chưa có số).
- [ ] **Step 6: Commit** `feat(cctv): official alert scorer and baseline numbers`.

### Task 2: Tờ chấm rộng (bước 6 của spec)

**Files:**
- Modify: `scripts/process_video.py` (thêm `--keep-blips`)
- Create: `scripts/make_judging_sheet.py`
- Test: `tests/test_process_video.py`, `tests/test_make_judging_sheet.py`

**Interfaces:**
- Consumes: `score_alerts` (Task 1) để biết cảnh báo nào chưa được chấm.
- Produces: `data/labels/<video-stem>-judging.yaml` cùng định dạng `verdicts:` với `EPf9O7UXRoM-review.yaml` (`type, t_s, heads, verdict`), đọc được bằng `load_verdicts`. Mục chưa chấm có `verdict: null` và bị `load_verdicts` bỏ qua (sửa `load_verdicts` để bỏ mục `verdict` là `None`).

- [ ] **Step 1: Test hỏng cho `--keep-blips`:** `test_keep_blips_skips_require_sustained` — chạy `process(..., all_seats=True, keep_blips=True)` với detector giả cho ra một cảnh báo 500 ms; assert cảnh báo có trong `events.jsonl` (mặc định thì bị bỏ). Cấu hình trong `timing.json` có `"keep_blips": True`.
- [ ] **Step 2: Test hỏng cho tờ chấm:** `test_sheet_lists_only_unjudged_alerts` — events có 3 cảnh báo, nhãn đã phủ 1; `select_candidates(records, verdicts, limit=30)` trả về 2 mục, sắp theo thời gian, mỗi mục có `id, type, t_s, heads`. `test_sheet_merges_alerts_on_same_head_within_2s` — hai cảnh báo cùng đầu cách nhau 1 s thành một ô.
- [ ] **Step 3: Chạy, thấy hỏng.**
- [ ] **Step 4: Cài đặt.** `process(..., keep_blips: bool = False)`; khi `True` thì không gọi `require_sustained`. `make_judging_sheet.py VIDEO EVENTS --labels ... --out DIR --limit 30`: chọn ứng viên bằng `select_candidates`, vẽ mỗi ô (ảnh tại `t_s`, cắt quanh đầu, khung đỏ, số thứ tự) bằng cách dùng lại `_tile` của `scripts/label_sheets.py`, ghép tối đa 12 ô một ảnh, ghi `DIR/sheet-*.jpg` và `DIR/judging.yaml` (`verdict: null`).
- [ ] **Step 5: Test xanh**, rồi chạy thật: **ĐO** với `--keep-blips --out /home/claude/rc/<clip>-loose` cho cả hai clip, tạo tờ chấm vào `/mnt/project-files/rules-core/clip1/` và `/mnt/project-files/rules-core/clip2/` (clip 1 khoảng 10 ô mới, clip 2 khoảng 20 ô). Chép `judging.yaml` (chỉ tọa độ) vào `data/labels/<stem>-judging.yaml`.
- [ ] **Step 6: Commit** `feat(cctv): judging sheet for a wider labelled set`, gửi tờ chấm cho Nguyên (Nguyên trả lời danh sách số ô đúng). Không chờ: làm tiếp Task 3 trên tập dev. Khi Nguyên trả lời, điền `verdict` vào hai file yaml, commit `data(labels): Nguyên's judging verdicts`, rồi chạy lại **ĐO** cho dòng gốc.

### Task 3: LOOK_AT_LAP chặt hơn (bước 1)

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/pose_detector.py` (thêm `LEFT_ELBOW, RIGHT_ELBOW = 7, 8`)
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py` (khối LAP, khoảng dòng 283–289; `ReviewRuleConfig`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_rules.py`, `tests/test_cctv_review_regression.py`

**Interfaces:**
- Produces: `ReviewRuleConfig.lap_hidden_needs_low_elbow: bool = True`, `lap_still_widths: float = 0.15`, `lap_paper_blocks: bool = True`. Bỏ `lap_needs_both_wrists`.

Luật mới cho mỗi khung (ghế `main`, đường mép bàn `line`):
1. Có cổ tay thấy được ở trên `line`, hoặc (khi `lap_paper_blocks`) nằm trong tờ giấy/sách của ghế (dùng hình hộp nới 10% như `_on_paper`) → khung không tính.
2. Mỗi cổ tay không thấy được chỉ tính là dưới bàn nếu khuỷu cùng bên thấy được (độ tin ≥ 0,4) và nằm dưới `line`. Nếu một cổ tay vắng mà khuỷu cũng không đạt → khung không tính.
3. Cuối cùng, mỗi đoạn LAP mở ra (theo `_runs` với `lap_hold_ms`) chỉ giữ nếu mọi cổ tay thấy được trong đoạn dịch không quá `lap_still_widths × box.width` (khoảng cách lớn nhất giữa các vị trí của cùng một cổ tay).

- [ ] **Step 1: Test hỏng.** Sửa `test_hands_hidden_is_lap` thành `test_hands_hidden_with_low_elbows_is_lap` (thêm khuỷu ở y 0,55 với độ tin 0,9). Thêm:
  - `test_hands_hidden_without_elbows_is_not_lap`
  - `test_wrist_on_exam_paper_below_line_is_not_lap` (giấy `ReviewObject("paper", ...)` phủ cổ tay ở y 0,5)
  - `test_wrist_moving_along_the_line_is_not_lap` (cổ tay chạy x 0,26 → 0,34 trong 4 s, y 0,55)
  - `test_empty_seat_gives_no_lap` (pose luôn `None`)
- [ ] **Step 2: Chạy, thấy hỏng** (3 test mới hỏng, `test_both_hands_below_desk_is_lap` vẫn xanh).
- [ ] **Step 3: Cài đặt** theo luật trên.
- [ ] **Step 4: Test đơn vị xanh.** Chạy `pytest tests/test_cctv_review_regression.py -q`. `test_clip1_phase2f` phải giữ `true_covered == 9`; nếu tụt thì đây là luật dừng. `test_clip1_opened_events_unchanged` sẽ đỏ vì đổi luật có chủ đích: tạo lại `opened_golden.json` bằng `replay_review.py` và ghi ruling vào commit.
- [ ] **Step 5: ĐO clip 1 và clip 2**, thêm dòng "Bước 1: LAP" vào sổ ghi chép.
- [ ] **Step 6: Commit** `feat(cctv): look-at-lap needs hands really under the desk`.

### Task 4: Điện thoại phải được cầm (bước 3)

**Files:**
- Modify: `review_rules.py` (khối (a) phone, khoảng dòng 205–246; `ReviewRuleConfig`)
- Test: `test_cctv_review_rules.py`, regression

**Interfaces:**
- Produces: `ReviewRuleConfig.phone_paper_fraction: float = 0.6` (thay cho `calculator_on_paper`, bỏ cờ đó). Bản ghi `PHONE_DETECTED` có thêm `aspect: float` (cao/rộng lớn nhất của các khung đạt), chỉ là bằng chứng.

Luật: với mỗi vật "phone" sống (`_alive`) trong ghế, đếm số khung nó nằm trên giấy (theo `_on_paper`) trên số khung nó được thấy, nối theo IoU như `_alive`. Nếu tỷ lệ ≥ `phone_paper_fraction` thì vật đó là máy tính và không được tính ở khung nào.

- [ ] **Step 1: Test hỏng:** `test_phone_on_paper_most_of_the_time_is_calculator` (5 khung, 4 trên giấy, được cầm và nhấc như điện thoại → không mở), `test_phone_lifted_off_paper_is_phone` (1/5 khung trên giấy → mở, có `aspect`), `test_single_sighting_on_paper_does_not_crash` (1 khung).
- [ ] **Step 2: Chạy, thấy hỏng.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Test xanh**, regression như Task 3 Step 4.
- [ ] **Step 5: ĐO hai clip**, thêm dòng "Bước 3: điện thoại".
- [ ] **Step 6: Commit** `feat(cctv): a phone lying on the exam paper is a calculator`.

### Task 5: Sự kiện cặp cần hai tín hiệu (bước 2)

**Files:**
- Modify: `review_rules.py` (`_hands_meet`, `_head_rules`; `ReviewRuleConfig`)
- Test: `test_cctv_review_rules.py`, regression

**Interfaces:**
- Produces: `ReviewRuleConfig.passing_inside_frames: int = 2` (áp cho mọi hàng, thay `far_row_inside_frames` chỉ cho hàng xa), `heads_lifted_fraction: float = 0.5`.

Ruling về spec 3.4: luật HEADS_TOGETHER ở `rules.py` **đã** đòi khoảng cách hai đầu < 0,7 lần mốc thường (`together_ratio`), chặt hơn mức 0,8 spec đề xuất, nên không thêm điều kiện khoảng cách. Các ca nhầm là hai người cùng cúi viết, nên tín hiệu thứ hai là: người bị gắn HEADS_TOGETHER phải **ngẩng đầu** (`head_lift ≥ min_head_lift`) trong ít nhất `heads_lifted_fraction` số khung của sự kiện; nếu không thì bỏ cả bản ghi opened lẫn closed của nó. Lọc này đặt sau `_head_rules` trong `review_window`, không sửa `rules.py` (tầng nhanh giữ nguyên).

PASSING loại `hands_only`: ngoài điều kiện hiện có, cần cổ tay của một bên nằm trong ô ghế bên kia ở ít nhất `passing_inside_frames` khung trong đoạn đó. `object_between` và `transfer` giữ như cũ.

- [ ] **Step 1: Test hỏng:** `test_heads_together_of_two_writers_bent_over_is_dropped`, `test_heads_together_with_head_up_is_kept`, `test_near_row_hands_meet_without_crossing_is_not_passing`, `test_near_row_wrist_inside_neighbour_seat_is_passing`. Sửa `test_near_row_hands_meet_reported` và `test_near_row_hands_a_fifth_of_a_seat_apart_reported` để cổ tay vào hẳn ô bên kia (giữ ý nghĩa: hai tay chạm nhau qua ranh giới thì báo).
- [ ] **Step 2: Chạy, thấy hỏng.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Test xanh**, regression như Task 3 Step 4 (sự kiện 25,5 s PASSING và 47,4 s HEADS phải vẫn được phủ).
- [ ] **Step 5: ĐO hai clip**, thêm dòng "Bước 2: sự kiện cặp".
- [ ] **Step 6: Commit** `feat(cctv): pair events need a second signal`.

### Task 6: Gán người theo vai (bước 4)

**Files:**
- Modify: `review_rules.py` (`assign_poses`, dòng 101–121)
- Test: `test_cctv_review_rules.py`, regression

**Interfaces:**
- Consumes: `head.shoulders(kps)` (trung điểm vai), `_person_head(item)`.
- Produces: chữ ký giữ nguyên `assign_poses(seats, people) -> dict[UUID, Keypoints]`. Hằng số mô-đun `ASSIGN_MIN_OVERLAP = 0.5`, `BACK_ROW_HEIGHTS = 0.1`.

Điểm của người `p` cho ghế `s`:
- Loại nếu đầu cao hơn đỉnh ô quá `BACK_ROW_HEIGHTS × box.height`, hoặc phần chung khung người với ô / diện tích khung người < `ASSIGN_MIN_OVERLAP` (không có `bbox` thì dùng luật cũ: đầu trong ô).
- Điểm = phần chung + 1 nếu trung điểm vai nằm trong nửa trên ô, trừ khoảng cách đầu tới chỗ đầu thường (`y + 0.25 h`) chia cho chiều cao ô.
- Ghép tham lam theo điểm từ cao xuống trên mọi cặp (ghế, người); mỗi người một ghế, mỗi ghế một người.

- [ ] **Step 1: Test hỏng:** `test_back_row_head_inside_seat_does_not_take_it`, `test_two_overlapping_people_get_two_seats`, `test_person_without_bbox_falls_back_to_head_inside`. Hai test cũ `test_assign_poses_picks_head_inside_seat`, `test_faceless_neighbour_does_not_take_the_seat` phải vẫn xanh.
- [ ] **Step 2: Chạy, thấy hỏng.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Test xanh**, regression như Task 3 Step 4.
- [ ] **Step 5: ĐO hai clip**, thêm dòng "Bước 4: gán người".
- [ ] **Step 6: Commit** `feat(cctv): assign people to seats by body overlap and shoulders`.

### Task 7: Khung xương ổn định (bước 5)

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/pose_smooth.py`
- Modify: `review_rules.py` (đầu `review_window`), `review_offline.py` (`_pose`, khoảng dòng 96–101), `scripts/build_timeline.py`, `scripts/process_video.py` (cờ `--pose-flip`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_pose_smooth.py`, `test_cctv_review_offline.py`

**Interfaces:**
- Produces:
  - `smooth_frames(frames: Sequence[ReviewFrame], *, min_cutoff: float = 1.0, beta: float = 0.05) -> list[ReviewFrame]`: bộ lọc One Euro cho từng điểm của từng ghế theo `offset_ms`; điểm độ tin < 0,4 đi qua nguyên và không cập nhật bộ lọc; ghế mất khung xương thì bộ lọc của ghế đó bắt đầu lại.
  - `ReviewRuleConfig.smooth_poses: bool = True` (gọi `smooth_frames` ở đầu `review_window`).
  - `merge_flipped(people: list[dict], flipped: list[dict], mirror_x: float) -> list[dict]`: lật x của `flipped` quanh `mirror_x` (`x = 2·mirror_x − x'`, cả `bbox` lẫn điểm), đổi cặp trái-phải COCO `(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)`, ghép với `people` theo IoU ≥ 0,5, lấy trung bình điểm có trọng số độ tin; người chỉ có một bên thì giữ nguyên.
  - `review_offline(..., pose_flip: bool = False)`; khi bật, `_pose` gọi thêm `pose_many` trên ảnh cắt lật ngang với cùng `transforms`, rồi `merge_flipped` với `mirror_x = transform.to_frame(crop_width / 2, 0)[0]`.
  - CLI `--pose-flip` ở `build_timeline.py` và `process_video.py`; `timing.json` có `"pose_flip"`.

- [ ] **Step 1: Test hỏng:** `test_smoothing_removes_one_frame_jitter` (cổ tay nhảy 0,05 ở một khung rồi về: sau lọc lệch < 0,02), `test_smoothing_follows_a_real_move` (dịch 0,1 và giữ: sau 1 s lệch < 0,01), `test_smoothing_restarts_after_missing_seat`; `test_flip_merge_swaps_left_and_right`, `test_flip_merge_of_symmetric_person_is_identity`, `test_flip_merge_with_no_flipped_people_returns_originals`; `test_review_offline_pose_flip_calls_pose_twice` (detector giả đếm số lần gọi).
- [ ] **Step 2: Chạy, thấy hỏng.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Test xanh**, regression như Task 3 Step 4.
- [ ] **Step 5: ĐO làm mượt** (không cần dựng lại) trên hai clip, thêm dòng "Bước 5a: làm mượt". Sau đó dựng lại dòng thời gian clip 1 với `--pose-flip` vào `/home/claude/rc/clip1-flip` (nohup + Monitor, khoảng 2 giờ CPU), **ĐO**, thêm dòng "Bước 5b: lật ảnh" có cột thời gian dựng. Nếu chậm quá 1,5 lần và lợi < 1 cảnh báo đúng, để `--pose-flip` mặc định tắt và ghi lý do.
- [ ] **Step 6: Commit** `feat(cctv): steadier poses (smoothing, optional flip)`.

### Task 8: Tổng kết

**Files:**
- Modify: `docs/superpowers/notes/2026-11-11-rules-core-accuracy.md`, `notebooks/kaggle_process_video.ipynb` (nếu có cờ mới cần bật), `README.md` (mục `--all-seats`)

- [ ] **Step 1:** Chạy **ĐO** cuối trên hai clip với mọi cờ ở mặc định. Viết phần kết luận: số trên tập dev, số trên tập giữ lại (nếu Nguyên đã chấm), sai số do tập nhỏ, các nhóm lỗi còn lại.
- [ ] **Step 2:** Chép demo cuối clip 1 (`annotated.mp4`, `report.html`, `events.csv`) vào `/mnt/project-files/rules-core/demo-clip1/`.
- [ ] **Step 3:** `pytest -q` và `mypy` xanh, commit `docs: rules core accuracy results`, đẩy, cập nhật mô tả PR #43.
