#!/bin/bash
set -e

SKELETON_DIR="/workspace/kimodo/assets/skeletons/smplx22"
S3_BUCKET="${S3_ASSETS_BUCKET:-s3://eca-kimodo-assets}"
CHECKPOINT_DIR="/workspace/checkpoints"
TEXT_ENCODERS_DIR="/workspace/text-encoders"

mkdir -p "$SKELETON_DIR" "$CHECKPOINT_DIR" "$TEXT_ENCODERS_DIR"

# Download SMPLX_NEUTRAL from S3
echo "[entrypoint] Downloading SMPLX_NEUTRAL.npz..."
aws s3 cp "$S3_BUCKET/smplx/SMPLX_NEUTRAL.npz" "$SKELETON_DIR/SMPLX_NEUTRAL.npz"

# Sync Kimodo diffusion checkpoint from S3
echo "[entrypoint] Syncing checkpoints..."
aws s3 sync "$S3_BUCKET/models/checkpoints/" "$CHECKPOINT_DIR/"

# Sync LLM2Vec text encoder models from S3
echo "[entrypoint] Syncing text encoders..."
aws s3 sync "$S3_BUCKET/models/text-encoders/" "$TEXT_ENCODERS_DIR/"

# Patch adapter configs to point to local base model path
for cfg in \
    "$TEXT_ENCODERS_DIR/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/adapter_config.json" \
    "$TEXT_ENCODERS_DIR/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_config.json"
do
    if [ -f "$cfg" ]; then
        sed -i 's|"meta-llama/Meta-Llama-3-8B-Instruct"|"/workspace/text-encoders/meta-llama/Meta-Llama-3-8B-Instruct"|' "$cfg"
        echo "[entrypoint] Patched adapter path in $cfg"
    fi
done

export CHECKPOINT_DIR
export TEXT_ENCODERS_DIR

# # Setup HuggingFace token from env var (injected by Secrets Manager)
# if [ -n "$HF_TOKEN" ]; then
#     mkdir -p /workspace/.cache/huggingface
#     echo -n "$HF_TOKEN" > /workspace/.cache/huggingface/token
#     echo "[entrypoint] HF token configured"
# fi

echo "[entrypoint] Starting: $@"
exec "$@"
