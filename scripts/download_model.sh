#!/bin/bash
# Downloads the quantized CLIP ONNX model into models/ with resume support,
# so flaky connections that cut long transfers still complete eventually.
set -u
BASE="https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
DEST="$(dirname "$0")/../models/Xenova/clip-vit-base-patch32"
mkdir -p "$DEST/onnx"

FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "preprocessor_config.json"
  "onnx/text_model_quantized.onnx"
  "onnx/vision_model_quantized.onnx"
)

for f in "${FILES[@]}"; do
  out="$DEST/$f"
  echo "== $f"
  for attempt in $(seq 1 60); do
    curl -L -C - --fail --silent --show-error \
      --connect-timeout 20 --speed-time 30 --speed-limit 1024 \
      -o "$out" "$BASE/$f" && break
    echo "   attempt $attempt failed (have $(du -h "$out" 2>/dev/null | cut -f1 || echo 0)), retrying in 5s..."
    sleep 5
  done
  if [ ! -s "$out" ]; then
    echo "FAILED: $f" >&2
    exit 1
  fi
done

echo "All files downloaded:"
find "$DEST" -type f -exec du -h {} +
