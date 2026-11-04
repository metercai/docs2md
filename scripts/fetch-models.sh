#!/usr/bin/env bash
# fetch-models.sh — download the EXTERNAL models docs2md needs.
#
# The L1 PP-OCRv6-small det/rec/dict are EMBEDDED in the binary (not fetched
# here). This script fetches only the pieces that are deliberately external:
#
#   1. the PP-DocLayout model (G1 routing; ~130MB, over GitHub's 100MB
#      single-file limit, so it is not in the repo)
#   2. optional larger PP-OCRv6 tiers (medium/tiny) for [l1] model_dir
#   3. optional L2 VLM GGUF (PaddleOCR-VL) for the llama-server wrapper
#
# Sources are the same as rusto-rs (ModelScope RapidOCR) and the PaddleOCR-VL
# GGUF build. Everything is written under ./models by default.

set -euo pipefail

DEST="${DOCS2MD_MODELS_DIR:-models}"
mkdir -p "$DEST/DocOCR" "$DEST"

echo "== [1/3] PP-DocLayout (G1 routing) =="
LAYOUT="$DEST/DocOCR/pp_doclayout_plus_l.onnx"
if [ -f "$LAYOUT" ]; then
  echo "  already present: $LAYOUT"
else
  echo "  downloading (this is a ~130MB file)..."
  # rusto-rs ships this under models/DocOCR; the canonical copy is in the
  # local rusto-rs working tree if present, otherwise ModelScope.
  if [ -f /home/ubuntu/rusto-rs/models/DocOCR/pp_doclayout_plus_l.onnx ]; then
    cp /home/ubuntu/rusto-rs/models/DocOCR/pp_doclayout_plus_l.onnx "$LAYOUT"
  else
    curl -fL --progress-bar \
      -o "$LAYOUT" \
      "https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath=onnx%2FPP-DocLayout%2Fpp_doclayout_plus_l.onnx"
  fi
fi
echo "  => $LAYOUT"

echo "== [2/3] optional larger PP-OCRv6 tiers =="
TIER="${DOCS2MD_TIER:-medium}"
TDIR="$DEST/PPOCR_v6_${TIER}_rten"
mkdir -p "$TDIR"
for f in det rec; do
  OUT="$TDIR/$f.onnx"
  if [ -f "/home/ubuntu/rusto-rs/models/PPOCR_v6_${TIER}_rten/$f.onnx" ]; then
    echo "  copying $f.onnx from local rusto-rs tree"
    cp "/home/ubuntu/rusto-rs/models/PPOCR_v6_${TIER}_rten/$f.onnx" "$OUT"
  elif [ ! -f "$OUT" ]; then
    echo "  (optional) $TIER tier not fetched — set [l1] tier=$TIER only if you fetch it"
  fi
done
[ -f "/home/ubuntu/rusto-rs/models/PPOCR_v6_${TIER}_rten/dict.txt" ] && \
  cp "/home/ubuntu/rusto-rs/models/PPOCR_v6_${TIER}_rten/dict.txt" "$TDIR/dict.txt" || true

echo "== [3/3] optional L2 VLM (PaddleOCR-VL GGUF) =="
if [ -n "${L2_GGUF_DIR:-}" ]; then
  echo "  L2_GGUF_DIR=$L2_GGUF_DIR (point [l2] at a llama-server serving it)"
fi
# The PaddleOCR-VL GGUF + mmproj are large (~1GB) and are NOT auto-downloaded.
# Build/obtain them separately and run:
#   docs2md-l2d --model /path/PaddleOCR-VL-1.6.gguf --mmproj /path/PaddleOCR-VL-1.6-mmproj.gguf --port 8080
# then set [l2] base_url = "http://127.0.0.1:8080" in docs2md.toml.

echo ""
echo "Done. Example [l1] layout config:"
echo "  [l1]"
echo "  layout_model = \"$LAYOUT\""
