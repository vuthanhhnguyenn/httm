# Phase 2b CCTV mode — Implementation Plan

**Spec:** docs/superpowers/specs/2026-10-29-phase2b-cctv-pose-design.md

**Goal:** `analyze_video_multi.py --scene cctv` tracks a whole exam room with one YOLO-pose pass and
emits head-turn / turn-around / heads-together / hand-to-neighbour / phone / leaving-seat events.

## Global Constraints
- Webcam mode output unchanged; existing tests stay green.
- New code in `packages/proctoring_ai/src/proctoring_ai/multi/cctv/`; no MediaPipe in CCTV mode.
- Model `yolo11n-pose.pt` declared in `models/manifest.yaml` (task `pose_detect`, SHA-256 pinned).
- Pose 960 px, conf 0.25; tracker `max_subjects=60`; all timings in ms of video time.

## Tasks
1. **Event types** — `TURN_AROUND`, `HEADS_TOGETHER` in `EventType`, risk weights, Vietnamese names.
2. **Pose detector** — `YoloPoseDetector.detect(frame) -> [{"class":"person","confidence","bbox","keypoints"}]`
   (17×[x,y,conf], normalised). Manifest entry + factory `_initialize_pose_detector`. Test: parse fake result.
3. **Roles** — `SeatRoleClassifier.update(seat_id, box, ms) -> "pending"|"candidate"|"walker"`,
   `.left_seat(seat_id, ms) -> bool`. Tests: still→candidate after 3 s; walker; candidate moving away 5 s.
4. **Head** — `head_yaw_index(kps)`, `facing(kps)`, `head_point(kps, box)`, `RollingBaseline`.
   Tests on synthetic keypoints.
5. **Rules** — `CctvSeatRules.step(result, offset) -> [(seat_id, record)]`, `.finish`, `.labels`,
   `.roles`. Tests per event incl. walker never emits.
6. **Pipeline + replay + CLI** — `CctvPipeline` producing `MultiFrameResult` with `poses`;
   `run_multi_replay(..., rules=...)`; writer colours walkers grey; `--scene/--pose-size/--pose-conf/--phone-every`.
   Test: fake pose detector end to end, no face events.
7. **Smoke** — real model on the 3 user screenshots: ≥15 people; push; user measures on M1200.
