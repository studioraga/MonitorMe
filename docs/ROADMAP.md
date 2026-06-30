# Roadmap

## Step 17A: MonitorMe Node1 Local Evidence Assistant v0.1

Status: implemented in this repository.

- normalized event DB
- evidence packs
- incident reports
- false-positive tracker
- grounded assistant answer flow
- FactGuard tests

## Step 17B: Node1 real camera producer

- C922 `/dev/video0` local capture
- motion gate
- YOLO ONNX detector
- keyframe and clip artifacts
- non-blocking assistant queue

## Step 18: Qwen VLM keyframe analysis

- only after event trigger
- analyze selected keyframes/evidence crops
- store VLM facts as normalized rows
- no external upload of private frames

## Step 19: SmolVLM2 clip experiments

- local clip summaries
- latency and quality experiments

## Step 20: SAM 2 evidence crops and privacy masking

- object masks
- crop relevant objects
- blur background/non-relevant regions

## Step 21: Grounding DINO after trigger

- open-vocabulary detection only after trigger
- never run broad surveillance prompts by default

## Step 22: CLIP/SigLIP visual search

- local embeddings
- similar event search
- Qdrant integration

## Step 23: multi-camera correlation

- Node2/Node3/Node4 event producers
- same normalized event schema
- cross-camera timeline queries


## Step 17D detector health

MonitorMe v0.1.8 adds `python -m monitor_me.cli detector-health` and `GET /models/detector/health` so Node1 can validate the YOLO ONNX model path, checksum, ONNX Runtime providers, and model input/output metadata before live camera capture. See `docs/STEP17D_DETECTOR_HEALTH.md`.

- Step 17E: evidence visualization overlays with raw-keyframe preservation.
