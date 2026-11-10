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
