#!/usr/bin/env python3
"""Generate the P5 reference: the product's reading-order stage per page.

For each sample page the full indra-ocr product path (orientation -> detect
-> recognize (LM on) -> assemble) runs, with ``OrtOCR.reading_order``
spied: the (boxes, texts, wh) the model saw (post geo mirror, post
tidy_lines/split_line) and its (order, cls, same, tab, a) are captured.
``line_features`` and ``_cluster`` outputs are stored too, so the gate can
walk the diagnostic chain feats -> logits -> decode -> order/blocks.

Per page:
  <page>.ord.json  {
    "boxes":  [[x0, y0, x1, y1] * N]  (f32 values as seen by the model),
    "wh":     [w, h],
    "texts":  [str * N],
    "k":      orientation class (0 for the 32-page set; vertical pages are
              skipped by the gate, they bypass the model order),
    "feats":  [[16] * N] f32  (line_features),
    "cls":    [[10] * N] f32,
    "same":   [[N] * N] f32,
    "tab":    [[N][N] * N] f32 or null,
    "a":      [[N] * N] f32,
    "order":  [int * N],
    "blocks": [[int] * B]  (_cluster(same)),
  }

    <indra-ocr-venv>/python gen-p5-reference.py <samples-dir> <out-dir>
"""
import json
import os
import sys

import cv2
import numpy as np

from indra_ocr.order_post import _cluster, line_features
from indra_ocr.pipeline import DESKEW_MIN, OrtOCR, estimate_skew, preview, rotate_page


def main():
    samples, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    p = OrtOCR("/tmp/indra-models", threads=1)
    assert p.order is not None, "the order model must be loaded"
    assert not p.table_heads, "table heads must be off (one order call per page)"

    captured = []
    orig = p.reading_order

    def spy(boxes, texts, wh):
        order, layout, a = orig(boxes, texts, wh)
        captured.append(
            {
                "boxes": np.array(boxes, np.float32)[:, :4],
                "wh": (int(wh[0]), int(wh[1])),
                "texts": [str(t) for t in texts],
                "order": [int(v) for v in order],
                "cls": None if layout is None else layout[0].astype(np.float32),
                "same": None if layout is None else layout[1].astype(np.float32),
                "tab": None if layout is None else (layout[2] if layout[2] is None
                                                   else layout[2].astype(np.float32)),
                "a": None if a is None else a.astype(np.float32),
            }
        )
        return order, layout, a

    for name in sorted(os.listdir(samples)):
        if not name.endswith(".png"):
            continue
        tag = name[:-4]
        gray = cv2.imread(os.path.join(samples, name), cv2.IMREAD_GRAYSCALE)
        small = preview(gray)
        a = estimate_skew(small, pool=None)
        k = p.orientation(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)

        captured.clear()
        p.reading_order = spy
        try:
            p(gray)
        finally:
            p.reading_order = orig

        assert len(captured) == 1, f"{tag}: {len(captured)} reading_order calls"
        c = captured[0]
        n = len(c["boxes"])
        feats = line_features(c["boxes"], c["wh"], c["texts"]).astype(np.float32)
        blocks = (
            _cluster(c["same"]) if c["same"] is not None else None
        )
        rec = {
            "boxes": [[float(v) for v in row] for row in c["boxes"]],
            "wh": list(c["wh"]),
            "texts": c["texts"],
            "k": int(k),
            "feats": [[float(v) for v in row] for row in feats],
            "cls": (
                None
                if c["cls"] is None
                else [[float(v) for v in row] for row in c["cls"]]
            ),
            "same": (
                None
                if c["same"] is None
                else [[float(v) for v in row] for row in c["same"]]
            ),
            "tab": (
                None
                if c["tab"] is None
                else [[float(v) for v in row] for row in c["tab"]]
            ),
            "a": (
                None
                if c["a"] is None
                else [[float(v) for v in row] for row in c["a"]]
            ),
            "order": c["order"],
            "blocks": (
                None if blocks is None else [[int(v) for v in b] for b in blocks]
            ),
        }
        json.dump(rec, open(os.path.join(out, tag + ".ord.json"), "w"))
        print(f"{tag}: N={n} k={k} blocks={0 if blocks is None else len(blocks)}", flush=True)


if __name__ == "__main__":
    main()
