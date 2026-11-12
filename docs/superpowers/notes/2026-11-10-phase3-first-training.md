# Phase 3: first training on the cloud (2026-11-10)

## What ran
- Timelines built on the cloud CPU with OpenVINO (`--device intel:cpu`, `--chunk-seats 3 --resume`, max quality, FP32):
  clip 1 (EPf9O7UXRoM, 71.7 s, 22 seats) in 100 min, clip 2 (LUy81HI7wS0, 103 s, 10 seats) in 67 min.
  Outputs (features.csv, timeline.json.gz, tracks) are in project files `phase3/timeline/`, not in the repo.
- Labels in `data/labels/learned/` (coordinates only):
  - clip 1: Nguyên's 23 verdicts of 2026-11-05 (9 true events, 14 false) mapped to seats by head position,
    plus NORMAL for the random quiet windows. 503 labelled windows: LAP 22, LOOK 9, PASSING 8, DOCUMENT 4.
    HEADS_TOGETHER (47.4 s) is lost because the LOOK label at 47.0 s on the same seat claims those windows.
  - clip 2: labelled by Claude from zoomed crops; only clear cases (writing = NORMAL, one LOOK_AT_NEIGHBOR at
    8-13 s), everything unclear UNSURE. Clip 2 cuts between several cameras, so its seat map mixes scenes.

## Result
- Leave-one-video-out: nothing learned (clip 2 has one positive event), every threshold stays NEVER (1.01),
  so the trained model would raise no alert. Not shipped.
- Leave-seats-out (5 folds over both clips): LOOK_AT_LAP precision 0.75, recall 0.68 (22 windows); every other
  class 0. Only LAP has a signal, and it rests on two students.

## Conclusion
The learned core cannot beat the rules until there are more labelled exam videos (target from the spec:
5+ videos with several examples of each behaviour). Keep `--core rules` as default.

## Rules core over every seat (`process_video.py --all-seats`), clip 1
The same timeline run through the hand-written rules instead of only the fast tier's suspect windows, with
`require_sustained` (red must last >= 3.2 s, alerts under 1 s dropped except phones). Scored against
Nguyên's 23 verdicts; "flagged" = any alert on that student within 2 s of the judged moment. The 17 alerts
nobody judged were checked by Claude on zoomed crops.

| Run | True events flagged | False pictures flagged | Red alerts |
|---|---|---|---|
| Kaggle 2x T4, suspect windows (2026-11-06) | 5/9 | - | - |
| Saved suspect-window cache, rules replay | 8/9 | 7/14 | 0 (before levels) |
| All seats, no red rule | 8/9 | 3/14 | 9 (at least 4 false: students writing) |
| **All seats + `require_sustained`** | **8/9** | **3/14** | **2, both true** (26.4 s, 47.0 s) |

The 3.2 s threshold was picked on clip 1 itself: optimistic until checked on another video. Cost: every seat
is cropped (6821 crops for clip 1 against 2721 for suspect windows), about twice the deep-tier time; on a
T4 roughly 5x the video length at 960 px, about 2.5x at `--quality balanced`. Outputs in project files
`phase3/all-seats-clip1/`.
