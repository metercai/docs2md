#!/usr/bin/env python3
"""Generate the numpy reference for the P1 preprocessing gate.

Runs the actual indra-ocr product code (indra_ocr.pipeline) over the 32-page
sample and records, per page, every preprocessing decision and every model
input, losslessly:

  <page>.meta.json   dimensions, scale, k/a/skew, adaptive re-detect,
                     det input size, boxes, pictures, dark flags, crop widths
  <page>.upright.png the page after _turn (quarter turn + deskew)
  <page>.det.png     page_to_input ink map x255 (uint8; 0 = white) at the
                     first detect long side
  <page>.det2.png    same, at the adaptive re-detect long side (when it fired)
  <page>.line{i}.png window-1 crop ink map x255 (recognizer input)
  <page>.line{i}t.png window-3 tall crop ink map x255

The Rust side recomputes the chain and the gate (tests/p1-gate.rs) compares:
same sizes, max|delta| <= 1/255, is_dark_line 100% match.

    <indra-ocr-venv>/python gen-p1-reference.py <samples-dir> <out-dir>
"""
import json
import os
import sys

import cv2
import numpy as np

from indra_ocr.pipeline import (
    DESKEW_MIN,
    ADAPTIVE_MAX_SIDE,
    ADAPTIVE_MIN_H,
    ADAPTIVE_TARGET_H,
    OrtOCR,
    estimate_skew,
    preview,
    rotate_page,
)
from indra_ocr.det_post import page_to_input
from indra_ocr.line_mask import crop_lines, line_darkness


def main():
    samples, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    p = OrtOCR("/tmp/indra-models", threads=1)
    print(f"long_side={p.long_side} height={p.height} window={p.window} "
          f"mask={bool(p.mask)} adaptive={p.adaptive}", flush=True)

    for name in sorted(os.listdir(samples)):
        if not name.endswith(".png"):
            continue
        tag = name[:-4]
        gray = cv2.imread(os.path.join(samples, name), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape

        small = preview(gray)
        a = estimate_skew(small, pool=None)
        k = p.orientation(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)
        gray2, k2, skew = p._turn(gray, k, a, small)
        first, _s1 = p._detect_at(gray2, p.long_side)  # first detect level
        boxes, pics = p.detect_all(gray2)
        dark = line_darkness(gray2, boxes[:, :4]) if len(boxes) else []
        b4 = boxes[:, :4] if len(boxes) else np.zeros((0, 4), np.float32)
        crops1 = crop_lines(gray2, b4, p.height, None, pool=None, dark=dark) if len(boxes) else []
        crops3 = crop_lines(gray2, b4, p.height, window=p.window, pool=None, dark=dark) if len(boxes) else []

        meta = {
            "src": name, "w": w, "h": h,
            "small": [small.shape[0], small.shape[1]],
            "a": a, "k": k, "skew": skew,
            "detect": {"boxes": boxes.tolist(), "pictures": pics.tolist()},
            "dark": list(dark),
            "crops1": [list(c.shape) for c in crops1],
            "crops3": [list(c.shape) for c in crops3],
        }
        cv2.imwrite(os.path.join(out, tag + ".upright.png"), gray2)

        # det input(s)
        x1, s1 = page_to_input(gray2, p.long_side)
        meta["det1"] = {"scale": s1, "out": [x1.shape[0], x1.shape[1]]}
        cv2.imwrite(os.path.join(out, tag + ".det.png"), np.rint(x1 * 255).astype(np.uint8))
        # Adaptive re-detect (detect_all already ran detect(); replay the same
        # rule on the FIRST-level boxes — the boxes detect() sees — to find the
        # second long side. Replaying on the post-adaptive boxes under-records:
        # the rule can fire inside detect() yet no longer triggers on the
        # taller re-detected lines.
        if p.adaptive and len(first) >= 8:
            med_h = float(np.median(first[:, 3] - first[:, 1]))
            if med_h < ADAPTIVE_MIN_H:
                ls = int(min(ADAPTIVE_MAX_SIDE,
                             p.long_side * ADAPTIVE_TARGET_H / max(med_h, 1.0))) // 32 * 32
                if ls > p.long_side:
                    x2, s2 = page_to_input(gray2, ls)
                    meta["detect"]["adaptive"] = {"ls": ls, "med_h": med_h}
                    meta["det2"] = {"scale": s2, "out": [x2.shape[0], x2.shape[1]]}
                    cv2.imwrite(os.path.join(out, tag + ".det2.png"),
                                np.rint(x2 * 255).astype(np.uint8))

        for i, c in enumerate(crops1):
            cv2.imwrite(os.path.join(out, f"{tag}.line{i}.png"),
                        np.rint(c * 255).astype(np.uint8))
        for i, c in enumerate(crops3):
            cv2.imwrite(os.path.join(out, f"{tag}.line{i}t.png"),
                        np.rint(c * 255).astype(np.uint8))

        with open(os.path.join(out, tag + ".meta.json"), "w") as f:
            json.dump(meta, f)
        nlines = len(boxes)
        print(f"{tag} {w}x{h} a={a:.2f} k={k} boxes={nlines} "
              f"det={meta['det1']['out']}"
              + (f" det2={meta['det2']['out']}" if "det2" in meta else ""), flush=True)


if __name__ == "__main__":
    main()
