//! P2: LineNet detection decode + detect orchestration (faithful port of
//! indra-ocr det_post.py `decode` + pipeline.py `detect` / `detect_all`).
//!
//! The det.onnx model outputs raw logits [1, C, h/2, w/2] at half the
//! detector input resolution: channel 0 is the core-band logit (sigmoided
//! here, P2), channels 1/2 are the top/bottom distances to the line edges.
//! When C > 3, channel 3 is a picture map decoded by
//! preproc::picture_boxes; the embedded model has C == 3, so the picture
//! path stays dormant.

use super::preproc::{self, Gray, InkMap};
use super::Session;

/// Core-band threshold on the sigmoid probability.
pub const DET_THRESH: f32 = 0.5;
/// Minimum component area (1/2-resolution px) to keep a line.
pub const DET_MIN_AREA: i64 = 2;
/// The core is shorter than the line by this fraction of its height, at
/// each end (det_post.py CORE_X_SHRINK).
const CORE_X_SHRINK: f32 = 0.15;
/// Decoder output is at half the detector input resolution.
const OUT_STRIDE: f32 = 2.0;

/// One decoded line in INPUT pixels (x0, y0, x1, y1, score).
pub type LineBox = [f32; 5];

/// decode (det_post.py): sigmoid the core logit, take 4-connected
/// components, and per component average the top/bottom distance maps and
/// the probability (numpy `bincount` = scan order, reproduced exactly by a
/// second scan-order pass). X is widened by CORE_X_SHRINK of the line
/// height on each side. `ch` holds the three channels, each [h, w]
/// row-major, half resolution. Returns the lines in scan order of the
/// components, in INPUT pixels.
pub fn decode(ch: [&[f32]; 3], h: usize, w: usize) -> Vec<LineBox> {
    decode_with(ch, h, w, DET_THRESH, DET_MIN_AREA, 0.0, 0.0)
}

/// decode with explicit threshold / min area / grow factors (the
/// `grow_y` / `grow_x` of det_post.py widen every box by that fraction of
/// its height; 0/0 in the pipeline).
pub fn decode_with(
    ch: [&[f32]; 3],
    h: usize,
    w: usize,
    thresh: f32,
    min_area: i64,
    grow_y: f32,
    grow_x: f32,
) -> Vec<LineBox> {
    let n = h * w;
    let mut prob = vec![0.0f32; n];
    let mut mask = vec![0u8; n];
    for i in 0..n {
        let p = 1.0f32 / (1.0f32 + (-ch[0][i]).exp());
        prob[i] = p;
        if p > thresh {
            mask[i] = 1;
        }
    }
    // Pass 1: 4-connected labelling in scan order (cv2
    // connectedComponentsWithStats, connectivity=4).
    let mut labels = vec![0i32; n];
    let mut stats: Vec<[i64; 5]> = Vec::new(); // left, top, width, height, area
    for start in 0..n {
        if mask[start] == 0 || labels[start] != 0 {
            continue;
        }
        let label = (stats.len() + 1) as i32;
        // left, top, max_x, max_y, area — bbox finalized when the component
        // is complete (an incremental width/height update would depend on
        // DFS visit order; cv2's stats are bounding boxes, order-free).
        let mut s: [i64; 5] = [
            start as i64,
            (start / w) as i64,
            (start % w) as i64,
            (start / w) as i64,
            0,
        ];
        let mut stack = vec![start];
        labels[start] = label;
        while let Some(p) = stack.pop() {
            s[4] += 1;
            let (px, py) = (p % w, p / w);
            s[0] = s[0].min(px as i64);
            s[1] = s[1].min(py as i64);
            s[2] = s[2].max(px as i64);
            s[3] = s[3].max(py as i64);
            // four neighbours
            if px > 0 {
                let q = p - 1;
                if mask[q] != 0 && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
            if px + 1 < w {
                let q = p + 1;
                if mask[q] != 0 && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
            if py > 0 {
                let q = p - w;
                if mask[q] != 0 && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
            if py + 1 < h {
                let q = p + w;
                if mask[q] != 0 && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
        }
        stats.push(s);
    }
    let ncomp = stats.len();
    if ncomp == 0 {
        return Vec::new();
    }
    // Pass 2: scan-order accumulation (numpy bincount semantics).
    let mut top_sum = vec![0.0f32; ncomp + 1];
    let mut bot_sum = vec![0.0f32; ncomp + 1];
    let mut prob_sum = vec![0.0f32; ncomp + 1];
    let mut cnt = vec![0.0f32; ncomp + 1];
    for y in 0..h {
        let ys = ((y as f32) + 0.5) * OUT_STRIDE;
        for x in 0..w {
            let i = y * w + x;
            let l = labels[i] as usize;
            if l == 0 {
                continue;
            }
            top_sum[l] += ys - ch[1][i];
            bot_sum[l] += ys + ch[2][i];
            prob_sum[l] += prob[i];
            cnt[l] += 1.0;
        }
    }
    let mut boxes = Vec::with_capacity(ncomp);
    for i in 1..=ncomp {
        if stats[i - 1][4] < min_area {
            continue;
        }
        let (left, max_x) = (stats[i - 1][0], stats[i - 1][2]);
        let width = max_x - left + 1;
        let y0 = top_sum[i] / cnt[i].max(1.0);
        let y1 = bot_sum[i] / cnt[i].max(1.0);
        let score = prob_sum[i] / cnt[i].max(1.0);
        let lh = (y1 - y0).max(1.0);
        let ex = (CORE_X_SHRINK * lh).min(0.5 * lh);
        boxes.push([
            left as f32 * OUT_STRIDE - ex - grow_x * lh,
            y0 - grow_y * lh,
            (left + width) as f32 * OUT_STRIDE + ex + grow_x * lh,
            y1 + grow_y * lh,
            score,
        ]);
    }
    boxes
}

/// One detect pass: page -> det input -> model -> decode. Returns the lines
/// in INPUT pixels, the input scale, and the picture channel (half
/// resolution, when the model has more than 3 output channels).
fn detect_at(det: &Session, gray: &Gray, long_side: usize) -> (Vec<LineBox>, f64, Option<InkMap>) {
    let (ink, s) = preproc::page_to_input(gray, long_side);
    let outs = det
        .run_f("x", &[1, 1, ink.h, ink.w], &ink.data)
        .expect("det run");
    let t = &outs[0];
    let c = t.shape()[1].min(4);
    let (h, w) = (t.shape()[2], t.shape()[3]);
    let d = t.as_float().expect("det float output");
    let boxes = decode(
        [&d[..h * w], &d[h * w..2 * h * w], &d[2 * h * w..3 * h * w]],
        h,
        w,
    );
    let pics = (c > 3).then(|| InkMap {
        w,
        h,
        data: d[3 * h * w..4 * h * w].to_vec(),
    });
    (boxes, s, pics)
}

/// detect (pipeline.py): lines in PAGE pixels (clipped), plus picture boxes
/// [P, 4] in page pixels (empty without the picture channel). The adaptive
/// re-detect rule fires on the FIRST pass's input-px boxes (median line
/// height under preproc::ADAPTIVE_MIN_H).
pub fn detect(
    det: &Session,
    gray: &Gray,
    long_side: usize,
    adaptive: bool,
) -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
    let (mut boxes, mut s, mut pics) = detect_at(det, gray, long_side);
    if let Some(ls) = preproc::adaptive_long_side(
        long_side,
        &boxes.iter().map(|b| [b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64]).collect::<Vec<_>>(),
        adaptive,
    ) {
        (boxes, s, pics) = detect_at(det, gray, ls);
    }
    let (pw, ph) = (gray.w as f64, gray.h as f64);
    let lines = boxes
        .iter()
        .map(|b| {
            [
                (b[0] as f64 / s).clamp(0.0, pw),
                (b[1] as f64 / s).clamp(0.0, ph),
                (b[2] as f64 / s).clamp(0.0, pw),
                (b[3] as f64 / s).clamp(0.0, ph),
            ]
        })
        .collect();
    let pictures = pics.map(|p| preproc::picture_boxes(&p, s, (gray.h, gray.w)));
    (lines, pictures.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic half-resolution map: one horizontal line band at rows
    /// 2..4 (input rows 4..8), full width, prob 1, distances to the band.
    fn one_line(h: usize, w: usize) -> [Vec<f32>; 3] {
        let mut core = vec![10.0f32; h * w]; // sigmoid(10) ~ 1
        for i in 0..h * w {
            let y = i / w;
            if !(2..5).contains(&y) {
                core[i] = -10.0; // sigmoid ~ 0
            }
        }
        let zero = vec![0.0f32; h * w];
        let mut top = zero.clone();
        let mut bot = zero.clone();
        // distances: row y (1/2 res) centre = (y+0.5)*2 input px; the line
        // spans input rows 4..8, so top dist = centre-4, bottom = 8-centre.
        for i in 0..h * w {
            let y = i / w;
            if (2..5).contains(&y) {
                let cy = (y as f32 + 0.5) * 2.0;
                top[i] = cy - 4.0;
                bot[i] = 8.0 - cy;
            }
        }
        [core, top, bot]
    }

    #[test]
    fn decode_single_line_band() {
        let ch = one_line(8, 16);
        let boxes = decode(
            [&ch[0], &ch[1], &ch[2]],
            8,
            16,
        );
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        // rows 2..4 (1/2 res) -> input y0=4, y1=8; full width 0..32 input
        // px (16 at half res), widened by 0.15*4 = 0.6 on each side.
        assert!((b[1] - 4.0).abs() < 1e-3, "y0={}", b[1]);
        assert!((b[3] - 8.0).abs() < 1e-3, "y1={}", b[3]);
        assert!((b[0] + 0.6).abs() < 1e-3, "x0={}", b[0]);
        assert!((b[2] - 32.6).abs() < 1e-3, "x1={}", b[2]);
        assert!(b[4] > 0.999);
    }

    #[test]
    fn decode_empty_and_min_area() {
        let zero = vec![0.0f32; 64];
        let boxes = decode([&zero, &zero, &zero], 8, 8);
        assert!(boxes.is_empty());
        // two-pixel-tall component fails min_area=2 (area 2 passes: >=).
        // Build a 1-px band: area = w px; a 1x1 blob has area 1 < 2.
        let mut core = vec![-10.0f32; 64];
        core[20] = 10.0;
        let boxes = decode([&core, &zero, &zero], 8, 8);
        assert!(boxes.is_empty(), "1 px blob must fail min_area=2");
    }

    /// Bounding box must be order-free (cv2 connectedComponentsWithStats
    /// semantics): the top row starts at x=10 and the bottom row extends
    /// left to x=2, so the component's leftmost pixel is visited late in
    /// DFS. An incremental `width = max(px - left + 1)` update computed
    /// with the pre-shrink left underestimates the width by the left
    /// shift (this was the P2 gate's x1 bug).
    #[test]
    fn decode_bbox_is_independent_of_dfs_order() {
        let (h, w) = (2, 20);
        let mut core = vec![-10.0f32; h * w];
        let mut top = vec![0.0f32; h * w];
        let mut bot = vec![0.0f32; h * w];
        for x in 10..20 {
            core[x] = 10.0; // top row, x 10..19
        }
        for x in 2..20 {
            core[w + x] = 10.0; // bottom row, x 2..19
        }
        for (y, row) in (0..h).enumerate() {
            let cy = (y as f32 + 0.5) * 2.0; // line spans input rows 0..4
            for x in 0..w {
                if core[row * w + x] > 0.0 {
                    top[row * w + x] = cy;
                    bot[row * w + x] = 4.0 - cy;
                }
            }
        }
        let boxes = decode([&core, &top, &bot], h, w);
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        // left=2, right+1=20 (1/2 res) -> input px, lh=4, ex=0.15*4=0.6.
        assert!((b[0] - (4.0 - 0.6)).abs() < 1e-3, "x0={}", b[0]);
        assert!((b[2] - (40.0 + 0.6)).abs() < 1e-3, "x1={}", b[2]);
        assert!((b[1] - 0.0).abs() < 1e-3);
        assert!((b[3] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn grow_widens_by_fraction_of_height() {
        let ch = one_line(8, 16);
        let b0 = decode([&ch[0], &ch[1], &ch[2]], 8, 16)[0];
        let b1 = decode_with(
            [&ch[0], &ch[1], &ch[2]],
            8,
            16,
            DET_THRESH,
            DET_MIN_AREA,
            0.5,
            0.5,
        )[0];
        // lh = 4 -> y0/y1 out by 2, x by 2 each side.
        assert!((b1[1] - (b0[1] - 2.0)).abs() < 1e-3);
        assert!((b1[3] - (b0[3] + 2.0)).abs() < 1e-3);
        assert!((b1[0] - (b0[0] - 2.0)).abs() < 1e-3);
        assert!((b1[2] - (b0[2] + 2.0)).abs() < 1e-3);
    }
}
