# Feature Specification: Smart Exam Proctoring System

**Feature Branch**: `N/A (project is not a Git repository)`

**Created**: 2026-09-19

**Status**: Draft

**Input**: User description: "Use README.md as the baseline product specification."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Monitor One Candidate Safely (Priority: P1)

As a proctor, I start a webcam-based exam session for one candidate and see the live camera,
candidate location, camera and tracking status, current risk level, and suspicious events. The
system helps me find moments that deserve review without declaring that the candidate cheated.

**Why this priority**: This is the Phase 1 product value and establishes the observation, temporal
analysis, event, evidence, risk, and human-review foundations used by every later capability.

**Independent Test**: Run the required recorded single-candidate scenarios and verify that a proctor
can start and stop a session, observe its status, receive explainable events, and review evidence.

**Acceptance Scenarios**:

1. **Given** a connected camera and an idle session, **When** the proctor starts monitoring, **Then**
   the live view shows one anonymous candidate track, processing status, update rate, and current risk.
2. **Given** a candidate briefly glances aside below the configured duration, **When** frames are
   analyzed, **Then** no head-turn event is created from that isolated movement.
3. **Given** a candidate turns their head beyond the configured angle for the configured duration,
   **When** the condition persists, **Then** one explainable left- or right-head-turn event is created.
4. **Given** a phone remains visible with sufficient confidence and duration, **When** the session is
   active, **Then** a high-severity phone event with timing, confidence, reason, and evidence is shown.
5. **Given** the candidate leaves the view or another person remains present long enough, **When** the
   configured temporal rule is met, **Then** the corresponding missing-person or multiple-person
   event is created with the correct severity.
6. **Given** suspicious activity stops, **When** no new qualifying event occurs, **Then** the risk
   score decays over time and remains within 0 to 100.

---

### User Story 2 - Review Explainable Evidence and Apply Exam Policy (Priority: P2)

As an authorized reviewer, I inspect a session timeline, filter or open its suspicious events, view
the evidence and explanation for each event, and make the final judgment. As an administrator, I can
configure thresholds and allowed materials so alerts reflect the rules of a particular exam.

**Why this priority**: Detection without reviewable evidence and exam-specific policy would create
unmanageable false alerts and unsafe automated conclusions.

**Independent Test**: Configure one exam to allow scratch paper but disallow books and phones, run a
recorded mixed-behavior session, and verify event suppression, event creation, evidence access, risk
changes, and auditability.

**Acceptance Scenarios**:

1. **Given** scratch paper is allowed, **When** allowed paper is visible, **Then** it does not create
   an unauthorized-document event or increase risk.
2. **Given** phones are disallowed, **When** a phone satisfies its confidence and duration rules,
   **Then** the event appears on the timeline with evidence and contributes its configured risk.
3. **Given** an authorized reviewer opens an event, **When** evidence is available, **Then** the
   reviewer sees what happened, when, to which anonymous track, for how long, with what confidence,
   why it was flagged, and what evidence was retained.
4. **Given** an unauthorized user requests evidence, **When** access is evaluated, **Then** access is
   denied and the attempt is recorded according to the audit policy.
5. **Given** a retention period expires, **When** retention processing runs, **Then** the associated
   sensitive evidence is deleted and is no longer retrievable.

---

### User Story 3 - Monitor a Room With Multiple Candidates (Priority: P3)

As a room proctor, I see multiple anonymous candidate tracks, each with independent behavior state,
risk, and event history. I can prioritize tracks with more suspicious activity, inspect a candidate,
and review the room event stream without treating ranking as a cheating verdict.

**Why this priority**: This is the Phase 2 expansion and depends on the event, policy, evidence, and
human-review behavior proven in Phase 1.

**Independent Test**: Replay room recordings containing 1, 5, 10, and, where target hardware permits,
20 people; verify track continuity, per-track risk isolation, object association, room display, event
history, and reported tracking and performance metrics.

**Acceptance Scenarios**:

1. **Given** several candidates are visible, **When** monitoring begins, **Then** each has a distinct
   anonymous track, independent state, risk score, and history.
2. **Given** a phone appears near one tracked candidate, **When** spatial and temporal association is
   sufficient, **Then** the phone event is assigned only to that candidate.
3. **Given** candidates overlap or a track is temporarily lost, **When** visibility returns, **Then**
   the system applies its recovery policy and records identity switches, losses, and fragmentation.
4. **Given** a fixed room layout with configured seats, **When** tracks move within the monitored
   area, **Then** tracks can be mapped to seats to reduce ambiguous identity and object association.
5. **Given** the risk overview is displayed, **When** candidates are ordered by risk, **Then** the UI
   clearly states that the ordering is a review priority and not a cheating determination.

### Edge Cases

- A person, face, object, or pose is detected intermittently near a confidence threshold.
- A second person crosses the background for less than the configured minimum duration.
- The candidate is present but their face is hidden, they turn fully away, or they bend below view.
- The camera becomes dark, blurred, covered, disconnected, or redirected while a session is active.
- A permitted document and a prohibited document look visually similar.
- A phone is visible on a desk but cannot be reliably associated with a person in a multi-person view.
- A candidate is occluded, leaves and re-enters, or receives a new track identifier.
- An event starts near session start or ends near session stop, leaving less pre/post evidence than
  the configured evidence window.
- An optional analyzer is unavailable or too slow; the operator must see degraded capability rather
  than receiving silently incomplete monitoring.
- Multiple related events overlap and would otherwise cause risk to be counted repeatedly without a cap.
- System time changes or frames arrive late, duplicated, or out of order.
- The evidence store is unavailable or full while an event is created.

## Requirements *(mandatory)*

### Scope Boundaries

**Phase 1 — committed MVP scope**:

- One webcam and one expected candidate per exam session.
- Live monitoring, temporal behavior analysis, event aggregation, risk, evidence, configuration,
  session history, and reviewer timeline.
- Head turn, looking down, abnormal head movement, phone, unauthorized document, multiple person,
  person missing, leaving seat, camera blocked, face not visible, and supporting suspicious-hand signals.
- Possible-talking remains experimental and cannot independently cause a large risk increase.

**Phase 2 — planned expansion**:

- Multiple simultaneous anonymous candidate tracks, per-track state and risk, object-to-person
  association, optional seat mapping, room overview, risk prioritization, and per-candidate history.

**Out of scope**:

- Automatic punishment or a final cheating/not-cheating verdict.
- Face recognition, biometric identity, emotion or personality inference, lie detection, or remote
  biometric identification.
- Audio analysis in the Phase 1 MVP.
- Storing full-session video by default.

### Functional Requirements

- **FR-001**: The system MUST let an operator create, start, inspect, stop, and revisit an exam session.
- **FR-002**: The system MUST show live camera output, camera status, processing status, tracking
  status, current risk, and the effective frame update rate during an active session.
- **FR-003**: The system MUST use anonymous track identifiers and MUST NOT require biometric identity.
- **FR-004**: The system MUST distinguish observations, suspicious events, risk, and the reviewer's
  final judgment; none of the first three may be presented as proof of cheating.
- **FR-005**: Significant events MUST require configurable duration, frequency, or corroborating
  signals and MUST NOT be created from a single frame alone.
- **FR-006**: The system MUST support configurable thresholds for detection confidence, head angles,
  event durations, event frequency, camera parameters, risk weights, decay, and maximum risk.
- **FR-007**: The system MUST support an exam policy that independently allows or disallows books,
  scratch paper, phones, and other configured materials.
- **FR-008**: The system MUST estimate left/right head direction and create `HEAD_TURN_LEFT` or
  `HEAD_TURN_RIGHT` only after the configured angle and duration conditions are met.
- **FR-009**: The system MUST create `LOOK_DOWN` according to configured pitch, duration, and
  frequency, and MUST NOT treat this event alone as evidence of cheating.
- **FR-010**: The system MUST aggregate repeated head-direction changes over a configurable sliding
  window and create `ABNORMAL_HEAD_MOVEMENT` when the configured pattern is met.
- **FR-011**: The system MUST create `PHONE_DETECTED` only when a phone meets configured confidence
  and minimum-duration requirements and the exam policy prohibits it.
- **FR-012**: The system MUST create `DOCUMENT_DETECTED` only when a material meets configured
  persistence requirements and the exam policy does not allow that material.
- **FR-013**: In a single-candidate session, the system MUST create `MULTIPLE_PERSON_DETECTED` only
  when more than one person persists beyond the configured duration.
- **FR-014**: The system MUST create `PERSON_MISSING` when no candidate remains visible beyond the
  configured duration.
- **FR-015**: The system MUST create `LEAVING_SEAT` from sustained body position, torso, pose, or
  allowed-zone evidence rather than an isolated keypoint change.
- **FR-016**: The system MUST create `CAMERA_BLOCKED` from sustained changes in visibility or image
  quality and MUST distinguish blocking from ordinary short exposure changes where possible.
- **FR-017**: The system MUST create `FACE_NOT_VISIBLE` when a candidate remains present but their
  face is unavailable beyond the configured duration.
- **FR-018**: The system MAY create `SUSPICIOUS_HAND_ACTIVITY` from repeated or sustained hand
  behavior, but the signal MUST have low default severity and MUST NOT independently drive high risk.
- **FR-019**: The system MAY create `POSSIBLE_TALKING` from sustained mouth movement as an
  experimental supporting signal, but it MUST NOT independently drive a large risk increase.
- **FR-020**: Event severity MUST use `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`; default severity MUST be
  configurable while respecting safety constraints on supporting signals.
- **FR-021**: Every event MUST include a unique event identifier, session identifier, anonymous track
  identifier when known, event type, severity, confidence, risk contribution or resulting risk,
  start and end time, duration, explanation, and evidence reference when evidence exists.
- **FR-022**: The system MUST merge frame-level signals into event lifecycles so that one sustained
  condition produces one event with a start and end rather than duplicate alerts per frame.
- **FR-023**: The current risk MUST be bounded to 0–100, increase with configured event contribution,
  persistence, repetition, and correlated signals, decay during normal behavior, and never accumulate
  without limit.
- **FR-024**: Correlated phone, looking-down, and hand-activity events MAY increase risk more than the
  same signals considered independently, but the calculation MUST remain explainable.
- **FR-025**: The operator MUST receive new events and risk/status changes during an active session
  without manually refreshing the session view.
- **FR-026**: Authorized users MUST be able to retrieve session history, event history, configuration,
  exam policy, and individual event details.
- **FR-027**: The system MUST retain a representative event frame and MAY retain a short policy-enabled
  clip around the event; it MUST NOT retain full-session video by default.
- **FR-028**: Evidence access MUST be restricted to authorized roles and viewing or downloading
  evidence MUST be auditable in production use.
- **FR-029**: Evidence and related sensitive session data MUST be deleted after the configured
  retention period.
- **FR-030**: The system MUST record session lifecycle, camera connection, analyzer readiness,
  effective update rate, processing latency, track lifecycle, event lifecycle, risk changes, and errors.
- **FR-031**: The system MUST avoid logging images or identifying data unless the data is required by
  an explicit evidence or diagnostic policy.
- **FR-032**: Failures in camera input, analysis, evidence retention, persistence, or realtime delivery
  MUST be visible to the operator and MUST NOT silently imply complete monitoring.
- **FR-033**: Phase 2 MUST maintain independent temporal buffers, behavior state, risk, and event
  history for each active track.
- **FR-034**: Phase 2 MUST associate detected objects to a track using spatial proximity, body/hand
  relationship, and temporal consistency; ambiguous objects MUST remain unassigned rather than guessed.
- **FR-035**: Phase 2 MUST record track identity switches, track loss, fragmentation, and recovery from
  occlusion as evaluation outcomes.
- **FR-036**: Phase 2 SHOULD support configured seat regions and track-to-seat mapping for fixed rooms.
- **FR-037**: Phase 2 MUST provide a room overview, candidate grid, risk-prioritized review list, room
  event stream, candidate detail, event history, and evidence replay.
- **FR-038**: Every behavior feature MUST provide configuration, error handling, logging,
  documentation, representative test recordings, and measured evaluation results before completion.
- **FR-039**: The fixed regression set MUST include normal, head-turn, phone-use, look-down,
  second-person, leave-seat, camera-blocked, and mixed-behavior recordings.
- **FR-040**: Evaluation MUST report detector precision/recall, tracking metrics where applicable,
  event precision/recall/F1, false alerts per minute, missed events per session, detection delay,
  effective update rate, and end-to-end latency.

### Key Entities *(include if feature involves data)*

- **Exam Session**: A bounded monitoring period with lifecycle state, camera status, exam policy,
  active tracks, risk history, timestamps, and retention metadata.
- **Candidate Track**: An anonymous subject within a session, including track identifier, location,
  visibility, current feature state, current risk, and event history.
- **Observation**: A timestamped, frame-level measurement such as person count, object confidence,
  head orientation, pose, hand position, brightness, blur, or visibility; it is not itself an event.
- **Suspicious Event**: A temporally aggregated, explainable occurrence with type, severity,
  confidence, timing, duration, reason, risk effect, track association, and evidence reference.
- **Risk State**: The current bounded review-priority score and level for a session or track,
  including contributing events and decay history.
- **Evidence Artifact**: A protected event frame or short clip with creation, retention, access, and
  deletion metadata.
- **Exam Policy**: The allowed/prohibited materials and activities, evidence rules, and other
  exam-specific decisions applied to observations.
- **System Configuration**: Versioned thresholds, durations, windows, weights, decay, camera targets,
  and feature flags that govern repeatable system behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A proctor can start or stop a single-candidate monitoring session and see the resulting
  state within 2 seconds for at least 95% of attempts on the declared reference environment.
- **SC-002**: At 720p on the declared GPU reference environment, the monitoring view updates at least
  15 times per second and 95% of event/status updates are visible within 300 ms of qualification.
- **SC-003**: 100% of created suspicious events contain event type, severity, confidence, start time,
  duration or open state, explanation, risk effect, and anonymous subject association when known.
- **SC-004**: In the required normal-behavior regression recordings, there are zero false
  `HIGH` or `CRITICAL` events and all false alerts per minute are reported.
- **SC-005**: Each required suspicious-behavior regression recording produces the expected event type
  within its configured temporal tolerance, and missed events and detection delay are reported.
- **SC-006**: Isolated one-frame detections and sub-threshold-duration movements create zero
  significant events across the temporal-rule unit and recorded-video tests.
- **SC-007**: Risk remains within 0–100 in 100% of tests, decreases during a sustained normal period,
  and produces a traceable explanation for every change.
- **SC-008**: In policy tests, 100% of allowed-material scenarios are suppressed and 100% of
  prohibited-material scenarios that meet configured thresholds create the corresponding event.
- **SC-009**: 100% of evidence access attempts are authorized and audited, and all artifacts selected
  for expiry are unavailable after a successful retention run.
- **SC-010**: Phase 2 evaluation reports results separately for 1, 5, 10, and, when supported by the
  target profile, 20 visible people, including identity switches, losses, fragmentation, recovery,
  false alerts, event delay, update rate, and latency.
- **SC-011**: In labeled multi-person object-association tests, every scored phone/document event is
  either assigned to the labeled track or explicitly marked ambiguous; none is silently assigned to
  a known wrong track.
- **SC-012**: A reviewer can open any timeline event, understand why it was flagged, inspect available
  evidence, and record a human disposition in under 60 seconds in at least 90% of usability trials.

## Assumptions

- The README is the authoritative baseline for this initial product specification; later discoveries
  update this living specification before downstream plans and tasks.
- Phase 1 is the first delivery target. Phase 2 requirements are retained here to protect forward
  compatibility but are planned and implemented only after Phase 1 acceptance criteria are met.
- A single-person session expects exactly one candidate; other visible people are suspicious only
  after the configured persistence rule is satisfied.
- Default angles, durations, confidence thresholds, event weights, and risk decay values in the
  README are starting baselines, not validated production guarantees.
- The reference performance environment includes a supported GPU; every benchmark declares its
  hardware, input resolution, subject count, and processing profile.
- The organization operating the system defines authorized reviewer roles and a lawful retention
  period before production use.
- Short evidence clips, when enabled, target the event plus up to five seconds before and after,
  constrained by session boundaries and policy.
- Recorded evaluation data includes diverse lighting, camera angles, skin tones, clothing, glasses,
  backgrounds, desk layouts, phone types, distances, occlusions, and substantial normal behavior.
