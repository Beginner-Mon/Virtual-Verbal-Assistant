#!/bin/bash
set -e

SKELETON_DIR="/workspace/kimodo/assets/skeletons/smplx22"
S3_BUCKET="${S3_ASSETS_BUCKET:-s3://eca-kimodo-assets}"

mkdir -p "$SKELETON_DIR"

# Download SMPLX_NEUTRAL from S3 if not already present
if [ ! -f "$SKELETON_DIR/SMPLX_NEUTRAL.npz" ]; then
    echo "[entrypoint] Downloading SMPLX_NEUTRAL.npz from $S3_BUCKET ..."
    aws s3 cp "$S3_BUCKET/smplx/SMPLX_NEUTRAL.npz" "$SKELETON_DIR/SMPLX_NEUTRAL.npz"
fi

# Setup HuggingFace token from env var (injected by Secrets Manager)
if [ -n "$HF_TOKEN" ]; then
    mkdir -p /workspace/.cache/huggingface
    echo -n "$HF_TOKEN" > /workspace/.cache/huggingface/token
    echo "[entrypoint] HF token configured"
fi

echo "[entrypoint] Starting: $@"
exec "$@"
