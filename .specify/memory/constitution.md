<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Modified principles: placeholder set -> six project-specific principles
- Added sections: Product and Safety Constraints; Development Workflow and Quality Gates
- Removed sections: none
- Follow-up TODOs: none
-->

# Smart Exam Proctoring System Constitution

## Core Principles

### I. Human Review Is the Final Authority (NON-NEGOTIABLE)
The system MUST only observe, measure, aggregate, and flag suspicious events. It MUST NOT label a
person as cheating, apply punishment, or present a model output as a final verdict. Every alert MUST
remain reviewable by an authorized human and MUST expose its confidence, risk contribution, timing,
duration, reason, subject or anonymous track, and available evidence. This protects candidates from
automated high-impact decisions and keeps the product aligned with its stated purpose.

### II. Temporal, Multi-Signal Evidence Over Single Frames
No important behavior or risk decision MAY be triggered by one frame alone. Rules MUST use a
minimum duration, a minimum frequency, temporal aggregation, or corroborating signals. The system
MUST preserve the distinction between observations, suspicious events, accumulated risk, and human
judgment. This reduces false alerts and makes behavior decisions reproducible.

### III. Policy and Thresholds Are Configuration
Detection thresholds, minimum durations, event weights, risk decay, camera parameters, retention,
and exam permissions MUST be configurable outside behavior implementation. Exam policy MUST decide
whether otherwise detected objects or activities are allowed. Defaults MUST be documented and
treated as baselines that require validation against representative data.

### IV. AI Inference Is Separate From Business Decisions
Detection, tracking, landmark estimation, and feature extraction MUST produce observations only.
Behavior evaluation, event lifecycle, severity, policy enforcement, and risk scoring MUST live in a
separate business layer with testable contracts. A model component MUST NOT embed an exam-specific
cheating rule. This separation enables model replacement, policy variation, and deterministic rule
testing.

### V. Privacy and Data Minimization by Design
The system MUST collect and retain only the camera and evidence data necessary for review. Full-session
video MUST NOT be stored by default. Phase 1 MUST use anonymous track identifiers and MUST NOT add face
recognition, biometric identity, emotion, personality, or lie inference. Production evidence access
MUST be restricted and auditable, sensitive stored data MUST be protected, and retention expiry MUST
result in deletion. Users MUST be clearly informed that camera monitoring is active.

### VI. Measured Quality Before Completion
An AI feature is complete only when it includes implementation, configuration, error handling,
logging, documentation, repeatable tests, representative recorded-video evaluation, and reported
metrics. Evaluation MUST cover detector, tracking where applicable, and event-level behavior. False
alerts per minute, missed events, detection delay, throughput, and latency MUST be measured against a
documented hardware and input profile. A demonstration alone is not evidence of completion.

## Product and Safety Constraints

- Phase 1 delivers a single-person webcam experience; Phase 2 extends the same event and review model
  to multiple anonymous tracks without weakening any Phase 1 safeguards.
- Risk scores MUST remain bounded from 0 to 100, decay when suspicious behavior stops, and avoid
  unbounded accumulation.
- Evidence MUST be event-focused: a representative frame and, when enabled by policy, a short clip
  around the event rather than unconditional full-session recording.
- False-positive-prone signals such as mouth or hand movement MUST remain supporting or experimental
  signals and MUST NOT independently cause a large risk increase.
- The MVP excludes automatic punishment, automatic cheating verdicts, face recognition, emotion
  recognition, personality inference, lie detection, and remote biometric identification.
- The system SHOULD continue operating in a degraded, visible state when an optional analyzer is
  unavailable; camera, model, tracking, and processing failures MUST be surfaced to the operator.

## Development Workflow and Quality Gates

1. Define testable behavior, policy assumptions, measurable success criteria, and privacy impact
   before implementation.
2. Add configuration and deterministic unit tests for temporal rules, event aggregation, and risk
   scoring before treating a behavior as complete.
3. Validate the end-to-end path from recorded input through observation, event, persistence, and
   realtime delivery using representative normal and suspicious scenarios.
4. Run the fixed recorded-video regression suite after every material AI change and compare event
   metrics, especially false alerts per minute, against the previous baseline.
5. Benchmark performance on a declared hardware, resolution, model, precision, and subject-count
   profile before making realtime claims.
6. Review every change for explainability, data minimization, access control, retention, and
   adherence to the non-goals.

## Governance

This constitution takes precedence over informal practices and generated implementation plans.
Amendments MUST be documented with rationale, migration impact, and an updated semantic version.
Removing or redefining a principle requires a MAJOR version bump; adding or materially expanding a
principle requires MINOR; wording clarifications require PATCH. Every feature specification, plan,
task list, and review MUST include an explicit constitution compliance check. Exceptions require a
written, time-bounded justification and MUST NOT override the human-review, privacy, or no-single-frame
principles.

**Version**: 1.0.0 | **Ratified**: 2026-09-19 | **Last Amended**: 2026-09-19
