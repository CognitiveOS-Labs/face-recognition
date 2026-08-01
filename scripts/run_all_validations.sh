#!/bin/sh
# run_all_validations.sh — end-to-end reproducible validation of the
# face-recognition model conversion pipeline.
#
# Steps:
#   1. Extract TF.js weights (tfjs weightsLoaderFactory) -> npy + mapping.json
#   2. Bitwise-verify extraction against an independent .bin parser
#   3. Convert to safetensors (roundtrip bitwise check)
#   4. Convert to gguf
#   5. Read back every gguf and bitwise-check vs safetensors (reference loader)
#   (optional) 6. Cross-check gguf files with the external `gguf` PyPI reader
#
# Environment:
#   PYTHON  python interpreter with numpy+safetensors (+ optional gguf)
#           (default: python3)
#   NODE    node with @tensorflow/tfjs installed in ./scripts
#           (default: node)
#
# Exit code: 0 on success, nonzero on any failure.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCRIPTS="$ROOT/scripts"
ARTIFACTS="$ROOT/artifacts"
MODEL_DIR="$ROOT/upstream/face-api/model"
PYTHON="${PYTHON:-python3}"
NODE="${NODE:-node}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: $MODEL_DIR not found."
  echo "Clone the upstream repo at the pinned commit first, e.g.:"
  echo "  git clone --filter=blob:none https://github.com/vladmandic/face-api.git $MODEL_DIR"
  echo "  (cd $MODEL_DIR && git checkout 189226d63aabb48cb40776fd1c453ebc0fa722f1)"
  exit 2
fi

echo "==> [1/5] Extract TF.js weights (tfjs weightsLoaderFactory)"
( cd "$SCRIPTS" && "$NODE" extract-tfjs-weights.js )

echo "==> [2/5] Bitwise-verify extraction vs independent .bin parser"
"$PYTHON" "$SCRIPTS/verify_tfjs_bin.py"

echo "==> [3/5] Convert to safetensors (roundtrip bitwise check)"
"$PYTHON" "$SCRIPTS/convert-to-safetensors.py"

echo "==> [4/5] Convert to gguf"
"$PYTHON" "$SCRIPTS/convert-to-gguf.py"

echo "==> [5/5] Read back gguf + bitwise check vs safetensors (reference loader)"
"$PYTHON" "$SCRIPTS/read-gguf.py" --all

echo "==> [6] Optional external gguf cross-check (spec compliance)"
if "$PYTHON" -c "import gguf" 2>/dev/null; then
  "$PYTHON" "$SCRIPTS/check_gguf_external.py"
else
  echo "    'gguf' package not installed; skipping (pip install -r $SCRIPTS/requirements.txt)"
fi

echo
echo "ALL VALIDATIONS PASSED"
