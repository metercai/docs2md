#!/usr/bin/env python3
"""Render the 32-page stratified sample (8 fixture files) to grayscale PNGs.

One-time input generation for the P1 preprocessing gate (and later P2-P5
gates). Needs `fitz` (PyMuPDF); the product itself never uses it — the PNGs
are the gate's fixed input.

    python3 render-p1-samples.py <fixtures-dir> <out-dir>

Sampling: 8 files, stratified page indices, 32 pages total:
  RFdiffusion3      42p  -> 4
  WO2024133739A1    105p -> 4
  WO2025114501A1    415p -> 10
  云手机_CPH         46p  -> 4
  小升初简历          9p  -> 4
  北京小车指标        1p  -> 1
  移民局预约表        2p  -> 2
  扫描文件            3p  -> 3
"""
import sys
import os

import fitz  # PyMuPDF

SAMPLES = {
    "RFdiffusion3-2025.09.18.676967v1.full.pdf": [3, 15, 27, 39],
    "WO2024133739A1_amylin_NN_patent_oral.pdf": [6, 10, 60, 90],
    "WO2025114501A1.pdf": [5, 10, 100, 150, 200, 250, 296, 350, 400, 410],
    "云手机_CPH_最佳实践.pdf": [2, 14, 28, 44],
    "小升初-五一小学-蔡现霖-简历V3.pdf": [0, 2, 5, 8],
    "北京市个人小客车更新指标通知书.pdf": [0],
    "国家移民局政务服务平台出入境预约申请表.pdf": [0, 1],
    "扫描文件_20250502_231411.pdf": [0, 1, 2],
}
DPI = 150


def main():
    fixtures, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    n = 0
    for doc, idxs in SAMPLES.items():
        d = fitz.open(os.path.join(fixtures, doc))
        for i, idx in enumerate(idxs):
            pix = d[idx].get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
            tag = f"{n:02d}_{os.path.basename(doc)[:-4][:12]}_p{idx}"
            pix.save(os.path.join(out, tag + ".png"))
            print(f"{tag}  {pix.width}x{pix.height}")
            n += 1
    print(f"rendered {n} pages -> {out}")


if __name__ == "__main__":
    main()
