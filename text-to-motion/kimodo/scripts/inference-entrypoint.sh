#!/bin/bash
set -e

# kimodo is installed as a wheel, so its assets live in site-packages rather than at
# /workspace/kimodo. Ask the package where they are instead of hardcoding a path that
# changes with the install layout.
SKELETON_DIR="$(python -c 'from kimodo.assets import SKELETONS_ROOT; print(SKELETONS_ROOT / "smplx22")')"
S3_BUCKET="${S3_ASSETS_BUCKET:-s3://eca-kimodo-assets}"
HF_STORE="/workspace/.cache/huggingface"
CHECKPOINT_DIR="$HF_STORE/checkpoints"
TEXT_ENCODERS_DIR="$HF_STORE/text-encoders"

mkdir -p "$SKELETON_DIR" "$HF_STORE" "$CHECKPOINT_DIR" "$TEXT_ENCODERS_DIR"

# S3 downloads go through boto3 (already in the venv) rather than the AWS CLI, which
# cost ~220 MB of image for these three calls. See scripts/s3_sync.py.
S3_SYNC="python /usr/local/bin/s3_sync.py"

# Download skeleton files from S3
echo "[entrypoint] Syncing skeleton files..."
$S3_SYNC "$S3_BUCKET/skeletons/smplx22/" "$SKELETON_DIR/"

# Sync Kimodo diffusion checkpoint from S3
echo "[entrypoint] Syncing checkpoints..."
$S3_SYNC "$S3_BUCKET/models/checkpoints/" "$CHECKPOINT_DIR/"

# Sync LLM2Vec text encoder models from S3
echo "[entrypoint] Syncing text encoders..."
$S3_SYNC "$S3_BUCKET/models/text-encoders/" "$TEXT_ENCODERS_DIR/"

# Patch adapter configs to point to local base model path
for cfg in \
    "$TEXT_ENCODERS_DIR/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/adapter_config.json" \
    "$TEXT_ENCODERS_DIR/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_config.json"
do
    if [ -f "$cfg" ]; then
        sed -i 's|"meta-llama/Meta-Llama-3-8B-Instruct"|"/workspace/.cache/huggingface/text-encoders/meta-llama/Meta-Llama-3-8B-Instruct"|' "$cfg"
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
