#!/usr/bin/env python3
"""Generate cross-language RTL test vectors from the product indra_ocr.rtl.

The P7 gate (rtl.rs tests) replays these vectors and asserts the Rust port
produces byte-identical results: flip (rtl base / ltr base / base_rtl forced),
visual_label, label_from_visual, normalize, clusters, and brackets_as_shapes.

Samples are synthetic Arabic/Hebrew lines (the fixture corpus has no RTL)
covering: mixed Latin runs, digits, brackets (mirror), tatweel, presentation
forms, combining marks, Hebrew niqqud, empty/neutral-only lines.
"""

import json
import sys

from indra_ocr import rtl


def main(out: str) -> None:
    # (line, base_rtl or None) — base_rtl forced to True/False exercises the
    # explicit-paragraph-direction branch; None exercises rtl_base().
    lines = [
        "مرحبا بالعالم",
        "مرحبا (test 123) العالم",
        "hello مرحبا world",
        "בוקר טוב everybody",
        "היי (בוקר) שלום",
        "1234567890123مرحبا",
        "٣٤٥ ٦٧٨ arabic indic digits",
        "أبجد هوز حطي كلمن",
        "ب\u064bح\u0650ر\u064e",  # Arabic harakat on each base
        "סֵפֶר עִבְרִית",  # Hebrew niqqud
        "text with tatweel ـ in the middle",
        "\u0640leading tatweel dropped",
        "\u064bleading combining mark dropped",
        "\u0640\u064b",  # only tatweel + a lone combining mark
        "a (b) [c] {d} <e> «f» ‹g› brackets",
        "مرحبا (test) ب",
        "abc 123 def",  # no RTL at all: passthrough
        "   ",  # neutral only
        "cafe\u00e9 no rtl but combining accent",
        "العربية and English mix (2024)",
        "שָׁלוֹם peace",  # niqqud + dagesh
        "(\u0645\u0631\u062d\u0628\u0627)",  # whole line is a bracketed word
        ")\u0627\u0645 \u0628\u064e\u064a",  # leading bracket then RTL
        "mixed \u05e9\u05dc\u05d5\u05dd (hebrew) and \u0645\u0631\u062d\u0628\u0627 (arabic)",
    ]
    cases = []
    for ln in lines:
        for base in (None, True, False):
            cases.append(
                {
                    "line": ln,
                    "base_rtl": base,
                    "flip": rtl.flip(ln, mirror=True, base_rtl=base),
                    "flip_nomirror": rtl.flip(ln, mirror=False, base_rtl=base),
                    "visual_label": rtl.visual_label(ln, base_rtl=base),
                    "label_from_visual": rtl.label_from_visual(ln, shapes=False),
                    "label_from_visual_shapes": rtl.label_from_visual(ln, shapes=True),
                    "normalize": rtl.normalize(ln),
                    "clusters": rtl.clusters(ln),
                    "has_rtl": rtl.has_rtl(ln),
                    "rtl_base": rtl.rtl_base(ln),
                }
            )
    vec = {
        "cases": cases,
        "brackets": [
            {"lines": l, "shapes": rtl.brackets_as_shapes(l)}
            for l in [
                ["\u0645\u0631\u062d\u0628\u0627 )", "\u0645\u0631\u062d\u0628\u0627 )", "\u0645\u0631\u062d\u0628\u062a ("],
                ["\u0645\u0631\u062d\u0628\u0627 (", "\u0645\u0631\u062d\u0628\u0627 (", "\u0645\u0631\u062d\u0628\u0629 )"],
                ["hello world", "no rtl here"],
                ["\u0645\u0631\u062d\u0628\u0627 )", "hello \u0645\u0631\u062d\u0628\u0629 ("],  # latin lines don't vote
            ]
        ],
    }
    with open(out, "w") as f:
        json.dump(vec, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(cases)} cases, {len(vec['brackets'])} brackets")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/p1/ref/rtl-vectors.json")
