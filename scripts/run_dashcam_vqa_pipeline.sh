#!/usr/bin/env bash
# Dashcam VQA pipeline: run VQA on a video (JSONL) then burn captions on the same video.
#
# Edit the variables below, or override when invoking:
#   VIDEO=/path/in.mp4 JSONL=/path/answers.jsonl OUTPUT_MP4=/path/out.mp4 \
#     QUESTION="Your question" ./scripts/run_dashcam_vqa_pipeline.sh
#
# Run only inference (step 1):
#   STEP=inference ./scripts/run_dashcam_vqa_pipeline.sh
# Run only overlay (step 2; needs existing JSONL):
#   STEP=overlay ./scripts/run_dashcam_vqa_pipeline.sh
#
# Raw commands (from repo root, package import path uses underscore + dot):
#
#   python -m alpamayo1_5.dashcam_vqa_inference \
#     --video INPUT.mp4 \
#     --output-jsonl answers.jsonl \
#     --question "Your question"
#
#   python -m alpamayo1_5.dashcam_vqa_overlay \
#     --video INPUT.mp4 \
#     --jsonl answers.jsonl \
#     --output OUTPUT.mp4
#
# Show built-in help:
#   ./scripts/run_dashcam_vqa_pipeline.sh help
#
# Extra Python CLI flags (optional, space-separated words):
#   EXTRA_INFERENCE_ARGS="--max-anchors 3 --no-progress" STEP=inference ./scripts/run_dashcam_vqa_pipeline.sh
#   EXTRA_OVERLAY_ARGS="--position top" STEP=overlay ./scripts/run_dashcam_vqa_pipeline.sh

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<'EOF'
Dashcam VQA pipeline (inference JSONL, then burn captions on the video).

Environment variables (defaults shown):
  VIDEO       Input MP4  (/root/data/input.mp4)
  JSONL       VQA output (/root/data/out.jsonl)
  OUTPUT_MP4  Final MP4  (/root/data/output_with_captions.mp4)
  QUESTION    VQA prompt ("Describe the scene.")
  STEP        all | inference | overlay (default: all)
  EXTRA_INFERENCE_ARGS  extra flags for dashcam_vqa_inference
  EXTRA_OVERLAY_ARGS    extra flags for dashcam_vqa_overlay

Examples:
  ./scripts/run_dashcam_vqa_pipeline.sh
  VIDEO=/path/in.mp4 JSONL=/path/a.jsonl OUTPUT_MP4=/path/out.mp4 QUESTION="..." ./scripts/run_dashcam_vqa_pipeline.sh
  STEP=overlay ./scripts/run_dashcam_vqa_pipeline.sh

Equivalent python -m commands (use underscore in package name: alpamayo1_5):
  python -m alpamayo1_5.dashcam_vqa_inference --video ... --output-jsonl ... --question "..."
  python -m alpamayo1_5.dashcam_vqa_overlay --video ... --jsonl ... --output ...
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Default PYTHONPATH so `python -m alpamayo1_5...` works without editable install.
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}/src}"

# --- paths (override via environment) ---
VIDEO="${VIDEO:-/root/data/input.mp4}"
JSONL="${JSONL:-/root/data/out.jsonl}"
OUTPUT_MP4="${OUTPUT_MP4:-/root/data/output_with_captions.mp4}"
QUESTION="${QUESTION:-Describe the scene.}"

# all | inference | overlay
STEP="${STEP:-all}"

run_inference() {
  echo "[dashcam] inference -> $JSONL"
  python -m alpamayo1_5.dashcam_vqa_inference \
    --video "$VIDEO" \
    --output-jsonl "$JSONL" \
    --question "$QUESTION" \
    ${EXTRA_INFERENCE_ARGS:-}
}

run_overlay() {
  echo "[dashcam] overlay -> $OUTPUT_MP4"
  python -m alpamayo1_5.dashcam_vqa_overlay \
    --video "$VIDEO" \
    --jsonl "$JSONL" \
    --output "$OUTPUT_MP4" \
    ${EXTRA_OVERLAY_ARGS:-}
}

case "$STEP" in
  all)
    run_inference
    run_overlay
    ;;
  inference)
    run_inference
    ;;
  overlay)
    run_overlay
    ;;
  *)
    echo "error: STEP must be all, inference, or overlay (got: $STEP)" >&2
    exit 1
    ;;
esac
