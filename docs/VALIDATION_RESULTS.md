# Validation Results

Latest packaged validation:

```text
=== MonitorMe Step 17B validation ===
........... [100%]
=== MonitorMe Step 17B validation PASSED ===
```

The Step 17B validation suite does not seed demo CCTV data. It validates the real capture runner using a test-only frame source so CI/development machines without a physical C922 can still prove persistence, artifacts, assistant grounding, FactGuard, API routes, and reports.

Node1 physical validation should be run on the machine with the Logitech C922:

```bash
./scripts/validate_node1_c922_live.sh
```

Expected physical output:

- a completed `capture_sessions` row;
- real keyframe artifacts under `data/captures/<session_id>/keyframes/` if motion is detected;
- real `motion_detected` event rows;
- assistant answer containing `event_id`, `session_id`, and `frame_id`.
