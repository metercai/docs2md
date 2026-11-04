#!/usr/bin/env python3
"""Generate the P4 reference: (text, conf) per line WITH the language-model
beam search enabled, running the actual indra-ocr product code.

This re-runs the P3 gate input (the same P1 reference crop PNGs, measured
lossless) through recognize() with p.lms intact: a line whose greedy mean
CTC peak conf falls in [MIN_CONF, group gate) is re-read by the CTC prefix
beam (beam=8, cand_floor=3e-3); its ids/steps replace the greedy ones, the
conf does not.

Per page:
  <page>.rec4.json  {"lines": [{text, conf, group, xs}]} — same shape as the
                    P3 reference, now LM-inclusive.
  <page>.beam.json  [{"line": j, "group", "conf",
                      "n": lengths,
                      "blank": [f32]*n, "topv": [f32]*(n*8), "topi": [i64]*(n*8),
                      "greedy": {"ids": [...], "steps": [...]},
                      "beam":   {"ids": [...], "steps": [...]}}]
                    one entry per line the beam re-read; blank/topv/topi are
                    the Python engine's raw step slices (gate 2 feeds the same
                    tensors into the Rust beam_search and requires ids/steps
                    to come back byte-identical).

    <indra-ocr-venv>/python gen-p4-reference.py <samples-dir> <ref-dir> <out-dir>
"""
import json
import os
import sys

import cv2
import numpy as np

from indra_ocr.lm import beam_search
from indra_ocr.pipeline import (
    MIN_CONF,
    DESKEW_MIN,
    LM_BEAM,
    LM_CAND_FLOOR,
    OUT_GREEDY,
    OUT_LM,
    ROUTE_SURE,
    OrtOCR,
    estimate_skew,
    preview,
    rotate_page,
    width_batches,
)
from indra_ocr.rec_preproc import pad_batch


def main():
    samples, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    p = OrtOCR("/tmp/indra-models", threads=1)
    # P4: the LM stays on — p.lms maps id(session) -> (NgramLM, alpha, beta,
    # gate), built from meta.json "lm_params" in __init__.
    assert p.lm is not None, "the base recognizer must have its lm.npz"
    gates = {}
    for name, sess in [("base", p.rec)] + [(g, s) for g, (s, _) in p.groups.items()]:
        if id(sess) in p.lms:
            gates[name] = (p.lms[id(sess)][1], p.lms[id(sess)][2], p.lms[id(sess)][3])
    print(f"lm gates: {gates}", flush=True)
    vocabs = {"base": p.vocab}
    for g, (_s, v) in p.groups.items():
        vocabs[g] = v

    for name in sorted(os.listdir(samples)):
        if not name.endswith(".png"):
            continue
        tag = name[:-4]
        gray = cv2.imread(os.path.join(samples, name), cv2.IMREAD_GRAYSCALE)
        small = preview(gray)
        a = estimate_skew(small, pool=None)
        k = p.orientation(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)
        gray2, _k2, _skew = p._turn(gray, k, a, small)

        meta = json.load(open(os.path.join(ref, tag + ".meta.json")))
        boxes = np.array(meta["detect"]["boxes"])
        if not len(boxes):
            json.dump({"lines": []}, open(os.path.join(out, tag + ".rec4.json"), "w"))
            json.dump([], open(os.path.join(out, tag + ".beam.json"), "w"))
            print(f"{tag}: 0 lines", flush=True)
            continue
        b4 = boxes[:, :4]
        crops = [
            cv2.imread(os.path.join(ref, f"{tag}.line{i}.png"), cv2.IMREAD_GRAYSCALE)
            .astype(np.float32)
            / 255.0
            for i in range(len(b4))
        ]
        tall = [
            cv2.imread(os.path.join(ref, f"{tag}.line{i}t.png"), cv2.IMREAD_GRAYSCALE)
            .astype(np.float32)
            / 255.0
            for i in range(len(b4))
        ]

        names = ["base"] * len(crops)
        sure = [True] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops]):
            arr, widths = pad_batch([crops[j] for j in idx])
            q = p.script.run(None, {"x": arr, "widths": widths})[0]
            for kk, j in enumerate(idx):
                g = p.script_groups[int(q[kk].argmax())]
                names[j] = g if g in p.groups else "base"
                sure[j] = float(q[kk].max()) > ROUTE_SURE

        base_sel = [j for j in range(len(crops)) if names[j] == "base" or not sure[j]]
        groups = [
            ("base", base_sel, False)
        ] + [
            (g, [j for j, n in enumerate(names) if n == g], True)
            for g, (_s, _v) in p.groups.items()
        ]

        best_conf = [-1.0] * len(crops)
        res = [("", 0.0, "base", []) for _ in range(len(crops))]
        beams = []
        for gname, sel, as_chars in groups:
            if not sel:
                continue
            sess = p.rec if gname == "base" else p.groups[gname][0]
            vocab = vocabs[gname]
            lm = p.lms.get(id(sess))
            src = tall if gname == "base" else crops
            # Group the batch by the window-1 crop width (recognize:
            # width_batches over crops[j].shape[1]), then pad the source crop.
            for part in width_batches([crops[j].shape[1] for j in sel]):
                idx = [sel[i] for i in part]
                arr, widths = pad_batch([src[j] for j in idx])
                outs = sess.run(
                    OUT_GREEDY + (OUT_LM if lm is not None else []),
                    {"x": arr, "widths": widths},
                )
                best, pmax, lengths = outs[:3]
                blank, topv, topi = outs[3:] if lm is not None else (None, None, None)
                stride = arr.shape[-1] / max(best.shape[1], 1)
                for kk, j in enumerate(idx):
                    n = int(lengths[kk])
                    seq, pm = best[kk, :n], pmax[kk, :n]
                    keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                    x0, y0, x1, y1 = (float(v) for v in b4[j])
                    hh = max(1.0, y1 - y0)
                    ix0 = np.floor(x0 - 0.3 * hh)
                    cw = np.ceil(x1 + 0.3 * hh) - ix0
                    f = cw / crops[j].shape[1]
                    t = np.nonzero(keep)[0]
                    ids = seq[t].tolist()
                    conf = float(pm[keep].mean()) if keep.any() else 0.0
                    steps = t.tolist()
                    if lm is not None and MIN_CONF <= conf < lm[3]:
                        b_ids, b_steps = beam_search(
                            blank[kk, :n],
                            topv[kk, :n],
                            topi[kk, :n],
                            lm[0],
                            lm[1],
                            lm[2],
                            LM_BEAM,
                            cand_floor=LM_CAND_FLOOR,
                        )
                        if conf <= best_conf[j]:
                            continue
                        beams.append(
                            {
                                "line": j,
                                "group": gname,
                                "conf": conf,
                                "n": n,
                                "blank": [float(x) for x in blank[kk, :n]],
                                "topv": [float(x) for x in topv[kk, :n].ravel()],
                                "topi": [int(x) for x in topi[kk, :n].ravel()],
                                "greedy": {"ids": ids, "steps": steps},
                                "beam": {"ids": b_ids, "steps": b_steps},
                            }
                        )
                        ids, steps = b_ids, b_steps
                    if conf <= best_conf[j]:
                        continue
                    best_conf[j] = conf
                    xs = (ix0 + (np.array(steps) + 0.5) * stride * f).tolist()
                    res[j] = (
                        (vocab.decode(ids), conf, gname, xs)
                        if conf >= MIN_CONF
                        else ("", conf, gname, [])
                    )

        lines = [
            {"text": r[0], "conf": r[1], "group": r[2], "xs": r[3]} for r in res
        ]
        json.dump({"lines": lines}, open(os.path.join(out, tag + ".rec4.json"), "w"))
        json.dump(beams, open(os.path.join(out, tag + ".beam.json"), "w"))
        ng = {}
        for r in res:
            ng[r[2]] = ng.get(r[2], 0) + 1
        print(f"{tag}: {len(res)} lines {ng} beams={len(beams)}", flush=True)


if __name__ == "__main__":
    main()
