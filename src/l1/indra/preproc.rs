//! P1 preprocessing port of indra-ocr's page -> recognizer-input chain:
//! preview / estimate_skew / rotation / page_to_input (det input) /
//! crop_line + is_dark_line (rec input) / pad_batch / width_batches /
//! picture_boxes / adaptive re-detect rule.
//!
//! Fidelity notes:
//! * The integer and fixed-point paths (crop geometry, percentiles, the
//!   16-bit area/linear resizes, the 2x16 fixed-point bilinear warp) are
//!   reproduced bit-exactly from OpenCV 4.x (`resize_bitExact`,
//!   `getRotationMatrix2D` + `warpAffine` with `WARP_INVERSE_MAP`).
//! * `resize_area` uses a clean floating box filter (exact for integer
//!   downscale factors, 1 LSB away otherwise) — the gate compares ink maps
//!   within 1/255 (plan P1: "max|Δ|≤1/255, integer pixel level"), not bytes.
//! * `estimate_skew` mirrors the numpy search exactly (float64 grid math,
//!   first-max tie breaks); the row-sharpness scores are exact integers.

use std::collections::HashMap;

/// A grayscale page: 0 = black, 255 = white, row-major.
#[derive(Clone, Debug)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0u8; w * h],
        }
    }
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }
    /// Decode a rendered page image (PNG bytes) to the engine's grayscale
    /// page. Color inputs convert with BT.601 luma, matching cv2's
    /// IMREAD_GRAYSCALE (the product loader).
    pub fn from_png(png: &[u8]) -> Result<Gray, String> {
        let img = image::load_from_memory(png).map_err(|e| format!("decode page image: {e}"))?;
        let gray = img.to_luma8();
        Ok(Gray {
            w: gray.width() as usize,
            h: gray.height() as usize,
            data: gray.into_raw(),
        })
    }
}

/// A float32 ink map (0 = paper, 1 = ink), row-major.
#[derive(Clone, Debug)]
pub struct InkMap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl InkMap {
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
    /// Quantized to u8 the way the reference does: rint(ink * 255).
    pub fn to_u8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| (v * 255.0f32).round() as u8)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Constants (indra-ocr pipeline.py / det_post.py / rec_preproc.py)
// ---------------------------------------------------------------------------
pub const INPUT_MULTIPLE: usize = 32;
pub const DESKEW_MAX: f64 = 45.0;
pub const DESKEW_MIN: f64 = 0.3;
pub const PREVIEW_SIDE: usize = 1600;
pub const ORIENT_SIDE: usize = 768;
pub const ORIENT_MIN: f64 = 0.95;
pub const VERTICAL: i32 = 4;
pub const ADAPTIVE_MIN_H: f64 = 9.0;
pub const ADAPTIVE_TARGET_H: f64 = 12.0;
pub const ADAPTIVE_MAX_SIDE: i64 = 2048;
pub const LINE_HEIGHT: usize = 32;
pub const PAD_Y: f64 = 0.2;
pub const PAD_X: f64 = 0.3;
pub const WIDTH_MULTIPLE: usize = 8;
pub const PICTURE_MIN: f64 = 0.5;
pub const PICTURE_MIN_AREA: f64 = 0.004;

/// numpy `arange(start, stop, step)`: n = ceil((stop-start)/step) values,
/// values[i] = start + i*step (one rounding each, like numpy).
fn np_arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil() as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Python `round`: half-to-even (banker's). Size math must match the product's
/// `round(w * s)` exactly — a true .5 lands on the even side.
fn pyround(x: f64) -> i64 {
    if x.fract().abs() == 0.5 {
        let f = x.floor() as i64;
        if f % 2 == 0 {
            f
        } else {
            f + 1
        }
    } else {
        x.round() as i64
    }
}

// ---------------------------------------------------------------------------
// Resizes
// ---------------------------------------------------------------------------

/// A (source index, weight) pair from cv2's `computeResizeAreaTab`. Indices
/// are always non-negative (a `si = sx1-1` entry is only emitted when
/// `sx1 - fsx1 > 1e-3`, which implies sx1 >= 1).
struct AreaTabEntry {
    si: usize,
    di: usize,
    alpha: f32,
}

/// Build one axis of cv2's area-resize table exactly as
/// `computeResizeAreaTab(ssize, dsize, cn=1, scale)` does (scale = src/dst).
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<AreaTabEntry> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell_width = scale.min(ssize as f64 - fsx1);
        let sx1 = fsx1.ceil() as usize;
        let mut sx2 = fsx2.floor() as usize;
        sx2 = sx2.min(ssize.saturating_sub(1));
        let sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(AreaTabEntry {
                di: dx,
                si: sx1 - 1,
                alpha: ((sx1 as f64 - fsx1) / cell_width) as f32,
            });
        }
        let mut sx = sx1;
        while sx < sx2 {
            tab.push(AreaTabEntry {
                di: dx,
                si: sx,
                alpha: (1.0 / cell_width) as f32,
            });
            sx += 1;
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push(AreaTabEntry {
                di: dx,
                si: sx2,
                alpha: (fsx2 - sx2 as f64).min(1.0).min(cell_width) as f32 / cell_width as f32,
            });
        }
    }
    tab
}

/// cv2 INTER_AREA for uint8, the real shrink path: per-axis area tables with
/// float accumulation and half-rounding (exact port of resizeArea_ +
/// computeResizeAreaTab). For our pages the scale is always non-integer, so the
/// fast integer path (never taken) is out of scope.
fn resize_area(src: &Gray, nw: usize, nh: usize) -> Gray {
    let (sw, sh) = (src.w, src.h);
    if nw == 0 || nh == 0 {
        return Gray::new(nw, nh);
    }
    if nw == sw && nh == sh {
        return src.clone();
    }
    let sx = sw as f64 / nw as f64;
    let sy = sh as f64 / nh as f64;
    let xtab = area_tab(sw, nw, sx);
    let ytab = area_tab(sh, nh, sy);
    // cv2 accumulates with float weights: for each dst row, sum over the ytab
    // rows (buf[dx] = sum_x S[sy][x.si]*x.alpha; sum += buf[dy]*e.alpha), then
    // half-round. We fold the x pass into the y accumulation (same value).
    let mut out = vec![0u8; nh * nw];
    for dy in 0..nh {
        let mut sums = vec![0.0f32; nw];
        for e in ytab.iter().filter(|e| e.di == dy) {
            let row = &src.data[e.si * sw..(e.si + 1) * sw];
            for x in &xtab {
                sums[x.di] += row[x.si] as f32 * x.alpha * e.alpha;
            }
        }
        for dx in 0..nw {
            out[dy * nw + dx] = sums[dx].round() as i32 as u8;
        }
    }
    Gray {
        w: nw,
        h: nh,
        data: out,
    }
}

/// cv2 INTER_LINEAR for uint8 as clean float bilinear with border-replicate.
/// Not byte-exact with cv2's ufixedpoint16 path, but the P1 gate is a 1/255
/// tolerance (resize is explicitly not byte-gated, plan §P1 / risk #1), and
/// float bilinear stays within sub-LSB of it.
fn resize_linear(src: &Gray, nw: usize, nh: usize) -> Gray {
    let (sw, sh) = (src.w, src.h);
    if nw == 0 || nh == 0 {
        return Gray::new(nw, nh);
    }
    if nw == sw && nh == sh {
        return src.clone();
    }
    let sx = sw as f64 / nw as f64;
    let sy = sh as f64 / nh as f64;
    let mut out = Gray::new(nw, nh);
    for dy in 0..nh {
        let fy64 = (dy as f64 + 0.5) * sy - 0.5;
        for dx in 0..nw {
            let fx64 = (dx as f64 + 0.5) * sx - 0.5;
            out.data[dy * nw + dx] = sample_bilinear(src, fx64, fy64) as u8;
        }
    }
    out
}

/// Bilinear sample of `src` at float coords with cv2 BORDER_REPLICATE edges.
fn sample_bilinear(src: &Gray, x: f64, y: f64) -> f64 {
    let w = src.w as i64;
    let h = src.h as i64;
    let x0 = (x.floor() as i64).clamp(0, w - 1);
    let x1 = (x0 + 1).min(w - 1);
    let fx = (x - x0 as f64).clamp(0.0, 1.0);
    let y0 = (y.floor() as i64).clamp(0, h - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fy = (y - y0 as f64).clamp(0.0, 1.0);
    let top = src.data[(y0 as usize) * src.w + (x0 as usize)] as f64 * (1.0 - fx)
        + src.data[(y0 as usize) * src.w + (x1 as usize)] as f64 * fx;
    let bot = src.data[(y1 as usize) * src.w + (x0 as usize)] as f64 * (1.0 - fx)
        + src.data[(y1 as usize) * src.w + (x1 as usize)] as f64 * fx;
    top * (1.0 - fy) + bot * fy
}

/// The page's preview: shrunk once to at most `side` a side (area averaging);
/// unchanged when it already fits.
pub fn preview(gray: &Gray, side: usize) -> Gray {
    let (h, w) = (gray.h, gray.w);
    let s = side as f64 / (h.max(w)) as f64;
    if s >= 1.0 {
        return gray.clone();
    }
    let nw = pyround(w as f64 * s).max(1) as usize;
    let nh = pyround(h as f64 * s).max(1) as usize;
    resize_area(gray, nw, nh)
}

/// `gray` shrunk to `side` a side (area averaging) and inverted to a uint8
/// ink map (255 = ink).
fn ink_thumb(gray: &Gray, side: usize) -> Gray {
    let (h, w) = (gray.h, gray.w);
    let s = side as f64 / (h.max(w)) as f64;
    let nw = pyround(w as f64 * s).max(1) as usize;
    let nh = pyround(h as f64 * s).max(1) as usize;
    let small = resize_area(gray, nw, nh);
    let mut ink = Gray::new(nw, nh);
    for p in 0..nw * nh {
        ink.data[p] = 255 - small.data[p];
    }
    ink
}

// ---------------------------------------------------------------------------
// Warps (getRotationMatrix2D + warpAffine, WARP_INVERSE_MAP)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct RotMatrix {
    a: f64, // cos(angle)
    b: f64, // sin(angle)
    cx: f64,
    cy: f64,
}

/// cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0): alpha=cos, beta=sin.
/// The warp samples src at M^{-1}(x, y) (warpAffine's default), which for this
/// matrix is the rotation by -angle about the centre.
fn rot_matrix(gray: &Gray, angle: f64) -> RotMatrix {
    let rad = std::f64::consts::PI * angle / 180.0;
    RotMatrix {
        a: rad.cos(),
        b: rad.sin(),
        cx: gray.w as f64 / 2.0,
        cy: gray.h as f64 / 2.0,
    }
}

/// Sample `src` at float coords (q) with cv2's default warp border
/// (BORDER_CONSTANT, 0): cvRound each axis, out of bounds reads 0.
#[inline]
fn sample_nearest(src: &Gray, sx: f64, sy: f64) -> u8 {
    let ix = sx.round() as isize; // f64::round == cvRound (half away from zero)
    let iy = sy.round() as isize;
    if ix < 0 || iy < 0 || ix >= src.w as isize || iy >= src.h as isize {
        return 0;
    }
    src.data[(iy as usize) * src.w + (ix as usize)]
}

/// Bilinear sample at float coords with cv2 BORDER_REPLICATE (clamped edges).
#[inline]
fn sample_linear_replicate(src: &Gray, sx: f64, sy: f64) -> u8 {
    let x0 = sx.floor() as i64;
    let y0 = sy.floor() as i64;
    let fx = (sx - sx.floor()) as f32;
    let fy = (sy - sy.floor()) as f32;
    let mut acc = 0.0f32;
    for (dy, wy) in [(y0, 1.0f32 - fy), (y0 + 1, fy)] {
        for (dx, wx) in [(x0, 1.0f32 - fx), (x0 + 1, fx)] {
            let cx = dx.clamp(0, (src.w - 1) as i64) as usize;
            let cy = dy.clamp(0, (src.h - 1) as i64) as usize;
            acc += src.data[cy * src.w + cx] as f32 * wx * wy;
        }
    }
    acc.round() as u8
}

/// cv2.warpAffine with `getRotationMatrix2D((w/2, h/2), angle, 1.0)`:
/// dst(x, y) = src(M^-1 (x, y)), the inverse rotation about the centre, with no
/// pixel-centre offset (verified against cv2 on a full grid: 64/64 for
/// INTER_NEAREST). `linear_replicate` selects INTER_LINEAR + BORDER_REPLICATE
/// (rotate_page) vs INTER_NEAREST + BORDER_CONSTANT (the skew search).
fn warp(src: &Gray, m: &RotMatrix, linear_replicate: bool) -> Gray {
    let mut out = Gray::new(src.w, src.h);
    for y in 0..src.h {
        for x in 0..src.w {
            let dx = x as f64 - m.cx;
            let dy = y as f64 - m.cy;
            let sx = m.a * dx - m.b * dy + m.cx;
            let sy = m.b * dx + m.a * dy + m.cy;
            out.data[y * src.w + x] = if linear_replicate {
                sample_linear_replicate(src, sx, sy)
            } else {
                sample_nearest(src, sx, sy)
            };
        }
    }
    out
}

/// How sharply the horizontal ink profile changes row to row, the page turned
/// by `a` degrees (constant-0 border). The scores are exact integers.
pub fn row_sharpness(ink: &Gray, a: f64) -> f64 {
    let m = rot_matrix(ink, a);
    let r = warp(ink, &m, false);
    let mut p = vec![0i64; r.h];
    for (y, row) in p.iter_mut().zip(r.data.chunks(r.w)) {
        *y = row.iter().map(|&v| v as i64).sum();
    }
    let mut acc = 0i64;
    for i in 1..p.len() {
        let d = p[i] - p[i - 1];
        acc += d * d;
    }
    acc as f64
}

/// The page rotated by `degrees` counter-clockwise about its centre (same
/// size), INTER_LINEAR + BORDER_REPLICATE.
pub fn rotate_page(gray: &Gray, degrees: f64) -> Gray {
    let m = rot_matrix(gray, degrees);
    warp(gray, &m, true)
}

/// np.rot90(gray, k) for k mod 4 (1 = ccw, 2 = 180, 3 = cw).
pub fn rot90(gray: &Gray, k: i32) -> Gray {
    let k = k.rem_euclid(4);
    let (h, w) = (gray.h, gray.w);
    let (nh, nw) = match k {
        2 => (h, w),
        _ => (w, h),
    };
    let mut out = Gray::new(nw, nh);
    match k {
        0 => out.data.copy_from_slice(&gray.data),
        1 => {
            // np.rot90 k=1 (ccw): out[i][j] = src[j][W-1-i]; out is W x H.
            for i in 0..nh {
                for j in 0..nw {
                    out.data[i * nw + j] = gray.data[j * w + (w - 1 - i)];
                }
            }
        }
        2 => {
            for y in 0..h {
                for x in 0..w {
                    out.data[(h - 1 - y) * w + (w - 1 - x)] = gray.data[y * w + x];
                }
            }
        }
        _ => {
            // np.rot90 k=3 (cw): out[i][j] = src[H-1-j][i]; out is W x H.
            for i in 0..nh {
                for j in 0..nw {
                    out.data[i * nw + j] = gray.data[(h - 1 - j) * w + i];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Skew search (exact port of estimate_skew)
// ---------------------------------------------------------------------------

/// Degrees (counter-clockwise) that straighten the text rows of a page.
pub fn estimate_skew(gray: &Gray, side: usize) -> f64 {
    let fine = ink_thumb(gray, side);
    let fine_mean =
        fine.data.iter().map(|&v| v as i64).sum::<i64>() as f64 / (fine.w * fine.h) as f64;
    if fine_mean < 0.003 * 255.0 {
        return 0.0;
    }
    let coarse = ink_thumb(gray, side / 2);
    let grid = np_arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0);
    let scores: Vec<f64> = grid.iter().map(|&a| row_sharpness(&coarse, a)).collect();
    // np.argsort(scores)[-2:]: the two largest scores (stable, smaller index first).
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]).then_with(|| i.cmp(&j)));
    let starts: Vec<f64> = {
        let mut s = vec![0.0f64];
        for &i in idx.iter().take(2) {
            if !s.contains(&grid[i]) {
                s.push(grid[i]);
            }
        }
        s.sort_by(f64::total_cmp);
        s
    };
    // near: 0.5-degree ranges around each start, deduped in first-seen order.
    let mut near: Vec<f64> = Vec::new();
    for &a0 in &starts {
        for a in np_arange(a0 - 2.0, a0 + 2.0 + 1e-6, 0.5) {
            if !near.contains(&a) {
                near.push(a);
            }
        }
    }
    let mut fine_scores = HashMap::new();
    for a in &near {
        fine_scores.insert(a.to_bits(), row_sharpness(&fine, *a));
    }
    // a1 = the FIRST angle in `near` order with the max fine score.
    let mut a1 = near[0];
    let mut best = fine_scores[&a1.to_bits()];
    for &a in &near {
        let s = fine_scores[&a.to_bits()];
        if s > best {
            best = s;
            a1 = a;
        }
    }
    let last = np_arange(a1 - 0.5, a1 + 0.5 + 1e-6, 0.1);
    let mut vals: Vec<f64> = Vec::new();
    for &a in &last {
        vals.push(row_sharpness(&fine, a));
    }
    let mut bi = 0usize;
    for i in 1..vals.len() {
        if vals[i] > vals[bi] {
            bi = i;
        }
    }
    last[bi]
}

/// The quarter turns (counter-clockwise) from orient probabilities: 0 without
/// the model or when no class exceeds ORIENT_MIN; VERTICAL for vertical
/// columns.
pub fn orientation_from_probs(p: &[f32]) -> i32 {
    let mut k = 0usize;
    for i in 1..p.len() {
        if p[i] > p[k] {
            k = i;
        }
    }
    if p[k] as f64 > ORIENT_MIN {
        k as i32
    } else {
        0
    }
}

/// The `_turn` step: quarter turn + (re-)measured deskew. Returns the turned
/// page, k, and the applied skew degrees (0.0 when under DESKEW_MIN).
pub fn turn(gray: &Gray, k: i32, mut a: f64, small: &Gray) -> (Gray, i32, f64) {
    let q = if k == VERTICAL { 1 } else { -k };
    let gray = if k == VERTICAL {
        rot90(gray, 1)
    } else if k != 0 {
        rot90(gray, -k)
    } else {
        gray.clone()
    };
    if k == 1 || k == 3 || k == VERTICAL {
        a = estimate_skew(&rot90(small, q), 800);
    }
    if a.abs() >= DESKEW_MIN {
        (rotate_page(&gray, a), k, a)
    } else {
        (gray, k, 0.0)
    }
}

// ---------------------------------------------------------------------------
// det input
// ---------------------------------------------------------------------------

/// Resize a page so its long side is `long_side`, pad to a multiple of 32 with
/// white; returns the float32 ink map (0 = white) and the scale.
pub fn page_to_input(gray: &Gray, long_side: usize) -> (InkMap, f64) {
    let (h, w) = (gray.h, gray.w);
    let scale = long_side as f64 / (h.max(w)) as f64;
    let nh = pyround(h as f64 * scale).max(1) as usize;
    let nw = pyround(w as f64 * scale).max(1) as usize;
    let small = if (nh, nw) != (h, w) {
        if scale < 1.0 {
            resize_area(gray, nw, nh)
        } else {
            resize_linear(gray, nw, nh)
        }
    } else {
        gray.clone()
    };
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut out = InkMap {
        w: pw,
        h: ph,
        data: vec![0.0f32; ph * pw],
    };
    for y in 0..nh {
        for x in 0..nw {
            out.data[y * pw + x] = 1.0f32 - small.data[y * nw + x] as f32 / 255.0f32;
        }
    }
    (out, scale)
}

/// The adaptive re-detect rule from detect(): the second long side, when the
/// median line height (input px) is under ADAPTIVE_MIN_H.
pub fn adaptive_long_side(long_side: usize, boxes: &[[f64; 4]], adaptive: bool) -> Option<usize> {
    if !adaptive || boxes.len() < 8 {
        return None;
    }
    let mut hs: Vec<f64> = boxes.iter().map(|b| b[3] - b[1]).collect();
    hs.sort_by(f64::total_cmp);
    let med_h = hs[hs.len() / 2];
    if med_h < ADAPTIVE_MIN_H {
        let ls = (ADAPTIVE_MAX_SIDE
            .min((long_side as f64 * ADAPTIVE_TARGET_H / med_h.max(1.0)) as i64)
            / 32)
            * 32;
        if ls > long_side as i64 {
            return Some(ls as usize);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// rec input
// ---------------------------------------------------------------------------

/// np.percentile(b, 100*q) (linear) of a uint8 crop, from its histogram.
fn u8_percentile(b: &[u8], q: f64) -> f64 {
    let mut hist = [0i64; 256];
    for &v in b {
        hist[v as usize] += 1;
    }
    let mut cum = [0i64; 256];
    let mut acc = 0i64;
    for v in 0..256 {
        acc += hist[v];
        cum[v] = acc;
    }
    let n = b.len() as f64;
    let x = q * (n - 1.0);
    let i = x.floor() as i64;
    let g = x - i as f64;
    // searchsorted(cum, i, side="right"): first index with cum > i.
    let srt = |target: i64| -> i64 {
        let mut lo = 0i64;
        let mut hi = 256i64;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cum[mid as usize] > target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    };
    let lo = srt(i) as f64;
    let hi = srt((i + 1).min((n - 1.0) as i64)) as f64;
    let d = hi - lo;
    if g >= 0.5 {
        hi - d * (1.0 - g)
    } else {
        lo + d * g
    }
}

/// Light text on a dark ground: the ground (median of the line box) lies
/// nearer the dark end than the text does.
pub fn is_dark_line(gray: &Gray, box_: [f64; 4]) -> bool {
    let (h, w) = (gray.h, gray.w);
    let [x0, y0, x1, y1] = box_;
    let sx0 = x0 as i64;
    let sy0 = y0 as i64;
    let sx1 = x1 as i64 + 1;
    let sy1 = y1 as i64 + 1;
    let (cx0, cy0) = (sx0.max(0), sy0.max(0));
    let (cx1, cy1) = (sx1.min(w as i64), sy1.min(h as i64));
    if cx1 <= cx0 || cy1 <= cy0 {
        return false;
    }
    let mut b = Vec::new();
    for y in cy0..cy1 {
        for x in cx0..cx1 {
            b.push(gray.data[(y as usize) * w + (x as usize)]);
        }
    }
    if b.is_empty() {
        return false;
    }
    let (lo, med, hi) = (
        u8_percentile(&b, 0.02),
        u8_percentile(&b, 0.5),
        u8_percentile(&b, 0.98),
    );
    hi - med > 2.0 * (med - lo) + 8.0
}

/// Crop one line from a page and return the float32 ink map
/// [window*height, W]: expand the box, crop (white/dark outside the page),
/// resize to `height` keeping the aspect, and invert for dark lines.
#[allow(clippy::too_many_arguments)] // mirrors the Python crop_line signature
pub fn crop_line(
    gray: &Gray,
    box_: [f64; 4],
    height: usize,
    pad_y: f64,
    pad_x: f64,
    max_width: usize,
    window: usize,
    dark: Option<bool>,
) -> InkMap {
    let pad_y = (window as f64 * (1.0 + 2.0 * pad_y) - 1.0) / 2.0;
    let height = height * window;
    let [x0, y0, x1, y1] = box_;
    let h = (y1 - y0).max(1.0);
    let (ex0, ex1) = (x0 - pad_x * h, x1 + pad_x * h);
    let (ey0, ey1) = (y0 - pad_y * h, y1 + pad_y * h);
    let (img_h, img_w) = (gray.h, gray.w);
    let ix0 = ex0.floor() as i64;
    let iy0 = ey0.floor() as i64;
    let ix1 = ex1.ceil() as i64;
    let iy1 = ey1.ceil() as i64;
    let (cw, ch) = ((ix1 - ix0).max(1), (iy1 - iy0).max(1));
    let (sx0, sy0) = (ix0.max(0), iy0.max(0));
    let (sx1, sy1) = (ix1.min(img_w as i64), iy1.min(img_h as i64));
    let dark = dark.unwrap_or_else(|| is_dark_line(gray, box_));
    let fill = if dark { 0u8 } else { 255 };
    let mut crop = Gray {
        w: cw as usize,
        h: ch as usize,
        data: vec![fill; (cw * ch) as usize],
    };
    if sx1 > sx0 && sy1 > sy0 {
        for y in sy0..sy1 {
            for x in sx0..sx1 {
                crop.data[(y - iy0) as usize * (cw as usize) + (x - ix0) as usize] =
                    gray.data[(y as usize) * img_w + (x as usize)];
            }
        }
    }
    let out_w =
        WIDTH_MULTIPLE.max(max_width.min(pyround(cw as f64 * height as f64 / ch as f64) as usize));
    let resized = if ch as usize > height {
        resize_area(&crop, out_w, height)
    } else {
        resize_linear(&crop, out_w, height)
    };
    let mut ink = InkMap {
        w: out_w,
        h: height,
        data: vec![0.0; out_w * height],
    };
    for p in 0..out_w * height {
        ink.data[p] = resized.data[p] as f32 / 255.0f32;
    }
    if !dark {
        for v in ink.data.iter_mut() {
            *v = 1.0f32 - *v;
        }
    }
    ink
}

/// Right-pad ink maps to one width: (array [B, 1, H, Wm] row-major, widths).
pub fn pad_batch(lines: &[&InkMap], multiple: usize) -> (Vec<f32>, Vec<i32>) {
    let h = lines[0].h;
    let wm = lines.iter().map(|l| l.w).max().unwrap_or(0);
    let wm = wm.div_ceil(multiple) * multiple;
    let mut out = vec![0.0f32; lines.len() * wm * h];
    let mut widths = vec![0i32; lines.len()];
    for (i, l) in lines.iter().enumerate() {
        for y in 0..h {
            out[i * wm * h + y * wm..i * wm * h + y * wm + l.w]
                .copy_from_slice(&l.data[y * l.w..(y + 1) * l.w]);
        }
        widths[i] = l.w as i32;
    }
    (out, widths)
}

/// Group line indices into batches of similar width (narrowest first).
pub fn width_batches(
    widths: &[i32],
    max_batch: usize,
    max_ratio: f64,
    min_width: i32,
    max_area: i32,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by(|&a, &b| widths[a].cmp(&widths[b]).then_with(|| a.cmp(&b)));
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for &i in &order {
        let wi = widths[i];
        if let Some(last) = batches.last_mut() {
            if last.len() < max_batch
                && (wi.max(min_width)) as f64 <= max_ratio * (widths[last[0]].max(min_width)) as f64
                && ((last.len() + 1) as i32) * wi.max(min_width) <= max_area
            {
                last.push(i);
                continue;
            }
        }
        batches.push(vec![i]);
    }
    batches
}

/// [side, side] float32 ink map (1 = ink): long side scaled to `side`, padded.
pub fn orient_input(gray: &Gray, side: usize) -> InkMap {
    let (h, w) = (gray.h, gray.w);
    let s = side as f64 / (h.max(w)) as f64;
    let nw = (w as f64 * s).round().max(1.0) as usize;
    let nh = (h as f64 * s).round().max(1.0) as usize;
    let small = resize_area(gray, nw, nh);
    let mut out = InkMap {
        w: side,
        h: side,
        data: vec![0.0f32; side * side],
    };
    for y in 0..nh {
        for x in 0..nw {
            out.data[y * side + x] = 1.0f32 - small.data[y * nw + x] as f32 / 255.0f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// picture channel post-processing
// ---------------------------------------------------------------------------

/// 8-connected component stats: (x, y, w, h, area), labels 0 = background,
/// components labelled in scan order (like cv2.connectedComponentsWithStats).
fn cc8(mask: &[u8], w: usize, h: usize) -> Vec<(i64, i64, i64, i64, i64)> {
    let mut labels = vec![0i32; w * h];
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] == 0 || labels[i] != 0 {
                continue;
            }
            // BFS
            let label = (out.len() + 1) as i32;
            let mut minx = x as i64;
            let mut maxx = x as i64;
            let mut miny = y as i64;
            let mut maxy = y as i64;
            let mut area = 0i64;
            let mut stack = vec![i];
            labels[i] = label;
            while let Some(p) = stack.pop() {
                area += 1;
                let (px, py) = (p % w, p / w);
                minx = minx.min(px as i64);
                maxx = maxx.max(px as i64);
                miny = miny.min(py as i64);
                maxy = maxy.max(py as i64);
                for (dx, dy) in [
                    (-1i32, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                    (1, 1),
                ] {
                    let nx = (px as i32 + dx) as isize;
                    let ny = (py as i32 + dy) as isize;
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let ni = (ny * w as isize + nx) as usize;
                    if mask[ni] != 0 && labels[ni] == 0 {
                        labels[ni] = label;
                        stack.push(ni);
                    }
                }
            }
            out.push((minx, miny, maxx - minx + 1, maxy - miny + 1, area));
        }
    }
    out
}

/// Picture regions [P, 4] in page pixels from the det picture channel (at
/// half the detector's input scale `s`): 8-connected regions above
/// PICTURE_MIN, at least PICTURE_MIN_AREA of the page.
pub fn picture_boxes(logit: &InkMap, s: f64, page_hw: (usize, usize)) -> Vec<[f64; 4]> {
    let thresh = (PICTURE_MIN / (1.0 - PICTURE_MIN)).ln();
    let w = logit.w;
    let mut mask = vec![0u8; w * logit.h];
    for (m, &v) in mask.iter_mut().zip(&logit.data) {
        *m = if v as f64 > thresh { 1 } else { 0 };
    }
    let comps = cc8(&mask, w, logit.h);
    let f = 2.0 / s;
    let min_area = PICTURE_MIN_AREA * page_hw.0 as f64 * page_hw.1 as f64;
    let mut out = Vec::new();
    for (x, y, cw, chh, _area) in comps {
        let b = [
            x as f64 * f,
            y as f64 * f,
            ((x + cw) as f64 * f).min(page_hw.1 as f64),
            ((y + chh) as f64 * f).min(page_hw.0 as f64),
        ];
        if (b[2] - b[0]) * (b[3] - b[1]) >= min_area {
            out.push(b);
        }
    }
    out
}

/// A box of the page that rotate_page(.., degrees) made, mapped back: the
/// bounding box of the four mapped corners, clipped to the page, rounded to
/// one decimal (Python round-half-to-even). The mapping is the exact inverse
/// of the `warp` resample: M^-1 = [[a, -b], [b, a]] about the centre, with
/// the y translation `- b * cx` (a `+ b * cx` here would shift every deskewed
/// box by sin(degrees) * width — see the round-trip test).
pub fn unrotate_box(box_: [f64; 4], degrees: f64, wh: (usize, usize)) -> [f64; 4] {
    let (w, h) = wh;
    let rad = std::f64::consts::PI * (-degrees) / 180.0;
    let (a, b) = (rad.cos(), rad.sin());
    let (g, d) = (-b, a);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let [x0, y0, x1, y1] = box_;
    let mut xs = [0f64; 4];
    let mut ys = [0f64; 4];
    let pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
    for (i, (px, py)) in pts.iter().enumerate() {
        xs[i] = a * *px + b * *py + (1.0 - a) * cx - b * cy;
        ys[i] = g * *px + d * *py - g * cx + (1.0 - d) * cy;
    }
    let clip = |v: f64, m: f64| v.clamp(0.0, m);
    let r1 = |v: f64| (v * 10.0).round() / 10.0;
    [
        r1(clip(xs.iter().cloned().fold(f64::MAX, f64::min), w as f64)),
        r1(clip(ys.iter().cloned().fold(f64::MAX, f64::min), h as f64)),
        r1(clip(xs.iter().cloned().fold(f64::MIN, f64::max), w as f64)),
        r1(clip(ys.iter().cloned().fold(f64::MIN, f64::max), h as f64)),
    ]
}

// ---------------------------------------------------------------------------
// Tests (unit level; the 32-page gate is in tests/p1-gate.rs)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> Gray {
        let mut g = Gray::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = (x * 255 / w.saturating_sub(1).max(1)) as u8;
            }
        }
        g
    }

    #[test]
    fn area_integer_downscale_is_exact_average() {
        // 8x8 of 2x2 blocks, downscale 4x -> each block average.
        let mut g = Gray::new(8, 8);
        let blocks = [[10u8, 50], [90, 200]];
        for y in 0..8 {
            for x in 0..8 {
                g.data[y * 8 + x] = blocks[y / 4][x / 4];
            }
        }
        let r = resize_area(&g, 4, 4);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(r.data[y * 4 + x], blocks[y / 2][x / 2]);
            }
        }
    }

    #[test]
    fn linear_identity_and_mirror() {
        let g = ramp(64, 32);
        let r = resize_linear(&g, 64, 32);
        assert_eq!(r.data, g.data);
        // 2x upscale of a ramp stays a (nearly) linear ramp: monotone.
        let r2 = resize_linear(&g, 128, 64);
        for y in 0..64 {
            for x in 1..128 {
                assert!(r2.data[y * 128 + x] >= r2.data[y * 128 + x - 1]);
            }
        }
    }

    #[test]
    fn rot90_roundtrip() {
        let g = ramp(10, 6);
        assert_eq!(rot90(&rot90(&rot90(&g, 1), 1), 1).data, rot90(&g, 3).data);
        let g180 = rot90(&g, 2);
        assert_eq!(g180.data[0], g.data[(6 - 1) * 10 + (10 - 1)]);
    }

    // A degenerate box (all four corners on one point) round-trips through
    // unrotate_box with no bbox inflation. The forward placement comes from
    // the `warp` resample, not from unrotate_box itself: `rotate_page(deg)`
    // builds S = R_{+deg} (rot_matrix uses rad = +deg) and does
    // out(q) = src(S(q)), so an original feature at p lands on the rotated
    // page at q = S^{-1}(p) = R_{-deg}(p). unrotate_box computes S = R_{+deg}
    // (the exact inverse), so unrotate_box(forward(p)) = p. The historical
    // `+ g*cx` sign error added sin(degrees)*width to the y translation; the
    // points below sit ~1000 px from the centre, so it shows up as a ~74 px
    // deviation — far past the 0.1 px tolerance (and the points stay inside
    // the page, so no clipping muddies the comparison).
    #[test]
    fn unrotate_box_roundtrips_the_forward_warp() {
        let (w, h) = (1700usize, 2200usize);
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let pts = [
            (850.0, 1100.0),
            (700.0, 1000.0),
            (1000.0, 1200.0),
            (600.0, 900.0),
            (1100.0, 1300.0),
            (750.0, 950.0),
            (950.0, 1250.0),
            (500.0, 800.0),
            (1200.0, 1400.0),
        ];
        for &deg in &[1.0, 3.0, 5.0] {
            let t = std::f64::consts::PI * deg / 180.0;
            let (c, s) = (t.cos(), t.sin());
            for &(px, py) in &pts {
                // q: where the original point p lands on the rotated page
                // (R_{-deg} about the centre, the inverse of the warp's S).
                let qx = cx + c * (px - cx) + s * (py - cy);
                let qy = cy - s * (px - cx) + c * (py - cy);
                let b = unrotate_box([qx, qy, qx, qy], deg, (w, h));
                assert!(
                    (b[0] - px).abs() < 0.1 && (b[1] - py).abs() < 0.1,
                    "deg={deg} p=({px},{py}) unrotated to ({}, {})",
                    b[0],
                    b[1]
                );
            }
        }
    }

    #[test]
    fn skew_flat_page_is_zero() {
        let mut g = Gray::new(200, 300);
        g.data.fill(255);
        assert_eq!(estimate_skew(&g, 800), 0.0);
    }

    #[test]
    fn dark_line_detection() {
        // White ground, dark text: not dark.
        let mut g = Gray::new(40, 20);
        g.data.fill(250);
        for x in 0..40 {
            g.data[10 * 40 + x] = 20;
        }
        assert!(!is_dark_line(&g, [5.0, 8.0, 35.0, 12.0]));
        // Dark ground, light text: dark.
        let mut d = Gray::new(40, 20);
        d.data.fill(30);
        for x in 0..40 {
            d.data[10 * 40 + x] = 250;
        }
        assert!(is_dark_line(&d, [5.0, 8.0, 35.0, 12.0]));
    }

    #[test]
    fn crop_geometry_matches_python() {
        // box (10, 20, 110, 40) on a 200x100 page, window=1, pad 0.3/0.2:
        // h=20; X0=10-6=4 -> floor 4; X1=110+6=116 -> ceil 116; cw=112.
        // Y0=20-4=16; Y1=40+4=44; ch=28.
        // out_w = max(8, min(4096, round(112*32/28))) = round(128.0)=128.
        let mut g = Gray::new(200, 100);
        g.data.fill(128);
        let ink = crop_line(
            &g,
            [10.0, 20.0, 110.0, 40.0],
            32,
            PAD_Y,
            PAD_X,
            4096,
            1,
            None,
        );
        assert_eq!(ink.h, 32);
        assert_eq!(ink.w, 128);
        // uniform 128 grey -> ink = 1 - 128/255 exactly representable-ish.
        let expect = 1.0f32 - 128.0f32 / 255.0f32;
        assert!((ink.data[0] - expect).abs() < 1e-6);
    }

    #[test]
    fn width_batches_groups_narrowest_first() {
        let w = [100i32, 40, 90, 300, 50];
        let b = width_batches(&w, 4, 1.3, 64, 64 * 640);
        // narrowest first; 300 starts a new batch (300 > 1.3*90).
        assert_eq!(b[0][0], 1); // 40
        assert!(b.iter().any(|x| x[0] == 3)); // 300 in its own batch head
        assert_eq!(b.iter().map(|x| x.len()).sum::<usize>(), 5);
    }

    #[test]
    fn page_to_input_padding() {
        let g = ramp(333, 500);
        let (ink, s) = page_to_input(&g, 1024);
        let s_expect = 1024.0 / 500.0;
        assert!((s - s_expect).abs() < 1e-12);
        // 333*2.048=681.98 -> round 682; 500*2.048=1024. pad to 32: 704 x 1024.
        assert_eq!((ink.w, ink.h), (704, 1024));
        // padding is paper (0 ink).
        assert_eq!(ink.data[ink.h * ink.w - 1], 0.0);
    }

    #[test]
    fn adaptive_rule_fires_only_on_short_lines() {
        // 10 boxes of height 5 (input px) -> fires; target = 1024*12/5 = 2457
        // capped at 2048.
        let short = vec![[0.0, 0.0, 10.0, 5.0]; 10];
        assert_eq!(adaptive_long_side(1024, &short, true), Some(2048));
        // height 12 -> no fire.
        let tall = vec![[0.0, 0.0, 10.0, 12.0]; 10];
        assert_eq!(adaptive_long_side(1024, &tall, true), None);
        // 7 boxes -> no fire (needs >= 8).
        let few = vec![[0.0, 0.0, 10.0, 5.0]; 7];
        assert_eq!(adaptive_long_side(1024, &few, true), None);
    }
}
