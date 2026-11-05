#!/usr/bin/env python3
"""Generate the P6 reference: the product's markdown-assembly stage per page.

Extends the P5 reference (the product's reading-order stage) with the inputs
``page_markdown`` actually saw and its output, plus the raw calls of the three
pure line-splitting functions so the Rust port can be replayed against them:

  cells    — per model line, the product's split_line(box, toks, CELL_GAP)
             cells, each (geo'd box [4], text).  Table assembly consumes these.
  pictures — geo'd picture regions [P][4] (None when the page has none).
  conf     — per model line, the mean token confidence (from Page.lines,
             which are in reading order, mapped back to model index via order).
  md       — the product's page_markdown output (str, or None for pages whose
             layout head is absent).
  split_calls   — every OrtOCR.split_line call: (box [4], gap, at or null,
                  toks, [(box [4], text), ...]).
  gutter_calls  — every OrtOCR.gutter_cuts call: (boxes, toks per line,
                  cuts per line).
  tidy_calls    — every pipeline.tidy_lines call: (boxes, [text, conf, toks],
                  rtl, space, (out_boxes, [text, conf, toks])).

Tokens serialize as [kind, value, x]: kind 0 = recognizer vocab id (value int),
kind 1 = a single character (value str).  Math tokens (one token holds a whole
line) never occur here: the product venv has the math module disabled.

Per page:
  <page>.md.json  {
    "boxes", "wh", "texts", "k", "feats", "cls", "same", "tab", "a",
    "order", "blocks",            # as P5
    "cells": {line_idx: [[box4, text], ...]},
    "pictures": [[x0,y0,x1,y1] * P] or null,
    "conf": [float * N],
    "md": str or null,
    "split_calls":  [ [box4, gap, at_or_null, toks, [[box4, text], ...]], ... ],
    "gutter_calls": [ [boxes, [toks per line], [cuts per line]], ... ],
    "tidy_calls":   [ [boxes, [ [text, conf, toks], ...], rtl, space,
                       [out_boxes, [ [text, conf, toks], ...]]], ... ],
  }

    <indra-ocr-venv>/python gen-p6-reference.py <samples-dir> <out-dir>
"""
import json
import os
import sys

import cv2
import numpy as np

import indra_ocr.pipeline as pl
from indra_ocr.order_post import _cluster, line_features
from indra_ocr.pipeline import DESKEW_MIN, OrtOCR, estimate_skew, preview, rotate_page


def ser_toks(toks):
    """[(tok, x)] -> [[kind, value, x], ...]; id token -> [0, int, x],
    char token -> [1, str, x]."""
    out = []
    for tk in toks:
        tok, x = tk[0], float(tk[1])
        if isinstance(tok, str):
            out.append([1, tok, x])
        else:
            out.append([0, int(tok), x])
    return out


def ser_rec(rec):
    return [[r[0], float(r[1]), ser_toks(r[2])] for r in rec]


def main():
    samples, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    p = OrtOCR("/tmp/indra-models", threads=1)
    assert p.order is not None, "the order model must be loaded"
    assert not p.table_heads, "table heads must be off (one order call per page)"

    orig_ro = p.reading_order
    orig_pm = pl.page_markdown
    orig_split = p.split_line
    orig_gutter = p.gutter_cuts
    orig_tidy = pl.tidy_lines

    for name in sorted(os.listdir(samples)):
        if not name.endswith(".png"):
            continue
        tag = name[:-4]
        gray = cv2.imread(os.path.join(samples, name), cv2.IMREAD_GRAYSCALE)
        small = preview(gray)
        a = estimate_skew(small, pool=None)
        k = p.orientation(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)

        cap_ro, cap_pm, cap_split, cap_gutter, cap_tidy = [], [], [], [], []

        def spy_ro(boxes, texts, wh, _o=orig_ro, _c=cap_ro):
            order, layout, a_ = _o(boxes, texts, wh)
            _c.append(
                {
                    "boxes": np.array(boxes, np.float32)[:, :4],
                    "wh": (int(wh[0]), int(wh[1])),
                    "texts": [str(t) for t in texts],
                    "order": [int(v) for v in order],
                    "cls": None if layout is None else layout[0].astype(np.float32),
                    "same": None if layout is None else layout[1].astype(np.float32),
                    "tab": None
                    if layout is None
                    else (
                        layout[2] if layout[2] is None else layout[2].astype(np.float32)
                    ),
                    "a": None if a_ is None else a_.astype(np.float32),
                }
            )
            return order, layout, a_

        def spy_pm(boxes, texts, order, cls, same, _o=orig_pm, _c=cap_pm, **kw):
            out = _o(boxes, texts, order, cls, same, **kw)
            cells = kw.get("cells")
            pics = kw.get("pictures")
            _c.append(
                {
                    "cells": {
                        str(i): [[[float(v) for v in bx], str(tx)] for bx, tx in cs]
                        for i, cs in cells.items()
                    }
                    if cells
                    else {},
                    "pictures": (
                        [[float(v) for v in row[:4]] for row in np.asarray(pics)]
                        if pics is not None and len(pics)
                        else None
                    ),
                    "md": out,
                }
            )
            return out

        def spy_split(box, toks, gap, at=None, _o=orig_split, _c=cap_split):
            out = _o(box, toks, gap, at)
            _c.append(
                [
                    [float(v) for v in np.asarray(box, np.float32)[:4]],
                    float(gap),
                    None if at is None else [float(v) for v in at],
                    ser_toks(toks),
                    [[[float(v) for v in bx[:4]], str(tx)] for bx, tx, _ in out],
                ]
            )
            return out

        def spy_gutter(boxes, rec, _o=orig_gutter, _c=cap_gutter):
            out = _o(boxes, rec)
            _c.append(
                [
                    [[float(v) for v in row[:4]] for row in np.asarray(boxes, np.float32)],
                    [ser_toks(r[2]) for r in rec],
                    [[float(v) for v in c] for c in out],
                ]
            )
            return out

        def spy_tidy(boxes, rec, rtl, space, _o=orig_tidy, _c=cap_tidy):
            # Snapshot the input BEFORE the call: tidy_lines rebinds rec via
            # list(rec) (caller untouched) but mutates the boxes array IN
            # PLACE (boxes[best] = union), so recording after the call would
            # capture the post-mutation union boxes.
            in_boxes = [[float(v) for v in row[:4]] for row in np.asarray(boxes, np.float32)]
            in_rec = ser_rec(rec)
            ob, or_ = _o(boxes, rec, rtl, space)
            _c.append(
                [
                    in_boxes,
                    in_rec,
                    bool(rtl),
                    None if space is None else int(space),
                    [
                        [[float(v) for v in row[:4]]
                         for row in np.asarray(ob, np.float32)],
                        ser_rec(or_),
                    ],
                ]
            )
            return ob, or_

        p.reading_order = spy_ro
        pl.page_markdown = spy_pm
        p.split_line = spy_split
        p.gutter_cuts = spy_gutter
        pl.tidy_lines = spy_tidy
        try:
            page = p(gray)
        finally:
            p.reading_order = orig_ro
            pl.page_markdown = orig_pm
            p.split_line = orig_split
            p.gutter_cuts = orig_gutter
            pl.tidy_lines = orig_tidy

        assert len(cap_ro) == 1, f"{tag}: {len(cap_ro)} reading_order calls"
        assert len(cap_pm) == 1, f"{tag}: {len(cap_pm)} page_markdown calls"
        c, m = cap_ro[0], cap_pm[0]
        n = len(c["boxes"])

        conf = [0.0] * n
        for pos, idx in enumerate(c["order"]):
            conf[idx] = float(page.lines[pos].conf)

        blocks = _cluster(c["same"]) if c["same"] is not None else None
        rec = {
            "boxes": [[float(v) for v in row] for row in c["boxes"]],
            "wh": list(c["wh"]),
            "texts": c["texts"],
            "k": int(k),
            "feats": [[float(v) for v in row] for row in
                      line_features(c["boxes"], c["wh"], c["texts"]).astype(np.float32)],
            "cls": None if c["cls"] is None
            else [[float(v) for v in row] for row in c["cls"]],
            "same": None if c["same"] is None
            else [[float(v) for v in row] for row in c["same"]],
            "tab": None if c["tab"] is None
            else [[float(v) for v in row] for row in c["tab"]],
            "a": None if c["a"] is None
            else [[float(v) for v in row] for row in c["a"]],
            "order": c["order"],
            "blocks": None if blocks is None else [[int(v) for v in b] for b in blocks],
            "cells": m["cells"],
            "pictures": m["pictures"],
            "conf": conf,
            "md": m["md"],
            "split_calls": cap_split,
            "gutter_calls": cap_gutter,
            "tidy_calls": cap_tidy,
        }
        json.dump(rec, open(os.path.join(out, tag + ".md.json"), "w"))
        mdlen = 0 if rec["md"] is None else len(rec["md"])
        print(
            f"{tag}: N={n} k={k} cells_lines={len(m['cells'])} "
            f"split={len(cap_split)} gutter={len(cap_gutter)} tidy={len(cap_tidy)} md={mdlen}ch",
            flush=True,
        )


if __name__ == "__main__":
    main()
