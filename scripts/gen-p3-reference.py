#!/usr/bin/env python3
"""Generate the P3 reference: greedy (text, conf, tokens) per line for the 32
sample pages, running the actual indra-ocr product code.

LM is DISABLED (p.lms = {}): the P3 gate compares the pure greedy reading;
P4 re-runs the same gate WITH the language model (plan .hermes/plans §P4.3).
Crop inputs are the P1 reference PNGs (measured lossless: max|delta| = 0.0 —
crop values are exact multiples of 1/255), so the Rust gate compares the same
recognizer inputs the Python reference consumed.

Per page:
  <page>.rec.json   {"lines": [{"text", "conf", "group", "xs": [..]}]}
                    text = the greedy reading ("" under MIN_CONF=0.5),
                    conf = mean CTC peak probability (0.0 when no kept step),
                    group = the script group that produced the winning read
                            ("base" when the classifier is unsure),
                    xs = page x of each kept CTC step.

    <indra-ocr-venv>/python gen-p3-reference.py <samples-dir> <ref-dir> <out-dir>
"""
import json
import os
import sys

import cv2
import numpy as np

from indra_ocr.pipeline import (
    MIN_CONF,
    DESKEW_MIN,
    ROUTE_SURE,
    OrtOCR,
    estimate_skew,
    preview,
    rotate_page,
    width_batches,
)
from indra_ocr.line_mask import crop_lines, line_darkness
from indra_ocr.rec_preproc import pad_batch
from indra_ocr.vocab import Vocab


def main():
    samples, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    p = OrtOCR("/tmp/indra-models", threads=1)
    # P3 gate: greedy only. The reference must not read via the LM.
    p.lms = {}
    p.lm = None
    print(
        f"long_side={p.long_side} height={p.height} window={p.window} "
        f"groups={list(p.groups)} script={p.script is not None} "
        f"lm={'ON' if p.lm is not None else 'off (P3 greedy)'}",
        flush=True,
    )
    vocabs = {"base": p.vocab}
    for g, (_s, v) in p.groups.items():
        vocabs[g] = v

    for name in sorted(os.listdir(samples)):
        if not name.endswith(".png"):
            continue
        tag = name[:-4]
        gray = cv2.imread(os.path.join(samples, name), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        small = preview(gray)
        a = estimate_skew(small, pool=None)
        k = p.orientation(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)
        gray2, _k2, _skew = p._turn(gray, k, a, small)

        meta = json.load(open(os.path.join(ref, tag + ".meta.json")))
        boxes = np.array(meta["detect"]["boxes"])
        if not len(boxes):
            json.dump({"lines": []}, open(os.path.join(out, tag + ".rec.json"), "w"))
            print(f"{tag}: 0 lines", flush=True)
            continue
        b4 = boxes[:, :4]

        # P1 references: window-1 crops (group recognizers, script classifier)
        # and window-3 tall crops (base recognizer, rec_window=3).
        crops = [
            cv2.imread(os.path.join(ref, f"{tag}.line{i}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            for i in range(len(b4))
        ]
        tall = [
            cv2.imread(os.path.join(ref, f"{tag}.line{i}t.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            for i in range(len(b4))
        ]

        # Script routing (the product's line_groups, batched exactly as
        # pipeline.py does it).
        names = ["base"] * len(crops)
        sure = [True] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops]):
            arr, widths = pad_batch([crops[j] for j in idx])
            q = p.script.run(None, {"x": arr, "widths": widths})[0]
            for kk, j in enumerate(idx):
                g = p.script_groups[int(q[kk].argmax())]
                names[j] = g if g in p.groups else "base"
                sure[j] = float(q[kk].max()) > ROUTE_SURE

        # Grouping exactly as recognize(): base reads its own group's lines
        # plus every line the classifier is UNSURE about; each group recognizer
        # reads only the lines routed to it.
        base_sel = [j for j in range(len(crops)) if names[j] == "base" or not sure[j]]
        groups = [
            ("base", base_sel, False)
        ] + [
            (g, [j for j, n in enumerate(names) if n == g], True)
            for g, (_s, _v) in p.groups.items()
        ]

        best_conf = [-1.0] * len(crops)
        res = [("", 0.0, "base", []) for _ in range(len(crops))]
        for gname, sel, as_chars in groups:
            if not sel:
                continue
            sess = p.rec if gname == "base" else p.groups[gname][0]
            vocab = vocabs[gname]
            src = tall if gname == "base" else crops
            for part in width_batches([src[j].shape[1] for j in sel]):
                idx = [sel[i] for i in part]
                arr, widths = pad_batch([src[j] for j in idx])
                out_t = sess.run(["ids", "conf", "lengths"], {"x": arr, "widths": widths})
                best, pmax, lengths = out_t
                stride = arr.shape[-1] / max(best.shape[1], 1)
                for kk, j in enumerate(idx):
                    n = int(lengths[kk])
                    seq, pm = best[kk, :n], pmax[kk, :n]
                    keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                    x0, y0, x1, y1 = (float(v) for v in b4[j])
                    hh = max(1.0, y1 - y0)
                    ix0 = np.floor(x0 - 0.3 * hh)
                    cw = np.ceil(x1 + 0.3 * hh) - ix0
                    # x scale uses the window-1 crop width (crops[j]), NOT the
                    # source (tall) — matches pipeline.recognize exactly.
                    f = cw / crops[j].shape[1]
                    t = np.nonzero(keep)[0]
                    ids = seq[t].tolist()
                    conf = float(pm[keep].mean()) if keep.any() else 0.0
                    xs = (ix0 + (t + 0.5) * stride * f).tolist()
                    if conf <= best_conf[j]:
                        continue
                    best_conf[j] = conf
                    res[j] = (
                        (vocab.decode(ids), conf, gname, xs)
                        if conf >= MIN_CONF
                        else ("", conf, gname, [])
                    )

        lines = [
            {"text": r[0], "conf": r[1], "group": r[2], "xs": r[3]} for r in res
        ]
        json.dump({"lines": lines}, open(os.path.join(out, tag + ".rec.json"), "w"))
        ng = {}
        for r in res:
            ng[r[2]] = ng.get(r[2], 0) + 1
        print(f"{tag}: {len(res)} lines {ng}", flush=True)


if __name__ == "__main__":
    main()
