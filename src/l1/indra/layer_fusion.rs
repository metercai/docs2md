//! C2 (T4): text-layer fusion. Assigns each text-layer character to the
//! detected line whose box contains its centre (a run spanning two table
//! cells splits between them), so a covered line is read from the exact PDF
//! characters instead of the OCR guess. Uncovered lines keep the OCR result.
//!
//! Coordinate chain (all y-down pixels unless noted):
//!   PDF user space (y-UP, points)
//!     --[display map, rot=0]-->  display frame (render bitmap, top-left)
//!     --[rot90(k)]-->            post-turn frame
//!     --[deskew skew]-->         working frame  (== the det box frame)
//! The det boxes live in the working frame, so every layer character is mapped
//! there before assignment. The deskew step is the exact inverse of the
//! `preproc::rotate_page` warp (verified against the real warp in a unit
//! test); it is deliberately NOT derived from `preproc::unrotate_box`, whose
//! y-translation carries a pre-existing sign error.

use std::collections::BTreeMap;

use crate::l1::indra::preproc::{self, VERTICAL};
use crate::l1::indra::rtl;
use crate::l1::{LayerItem, LayerPage};

/// One text-layer character with its centre in the working (det) frame.
#[derive(Debug, Clone)]
struct CharPt {
    c: char,
    x: f64,
    y: f64,
    /// The run (content) index this char came from.
    run: usize,
}

/// Map a PDF-user-space point (y-up, points) to the display frame (y-down,
/// pixels) for an upright page (`rot = 0`, the only case `extract_layer`
/// emits). `box_*` is the render content box in user space, `sx`/`sy` the
/// pixels-per-point scale in each axis. Matches `blocks::pdf_box_to_norm`
/// (the 0° branch) scaled to pixels.
fn user_to_display(
    x: f64,
    y: f64,
    box_x0: f64,
    box_y0: f64,
    box_h: f64,
    sx: f64,
    sy: f64,
) -> (f64, f64) {
    let px = (x - box_x0) * sx;
    let py = (box_y0 + box_h - y) * sy;
    (px, py)
}

/// Map a display-frame point through the quarter turn `k` (the `np.rot90` the
/// pipeline applies) into the post-turn frame. `(W, H)` is the display size.
fn display_to_turned(p: (f64, f64), k: i32, w: usize, h: usize) -> (f64, f64) {
    let (x, y) = p;
    let (ww, hh) = (w as f64, h as f64);
    match k {
        0 => (x, y),
        // turn applies rot90(-k): k=1 -> rot90(3) (CW), k=2 -> rot90(2),
        // k=3 -> rot90(1) (CCW). VERTICAL uses rot90(1) (CCW).
        1 => (hh - 1.0 - y, x),                  // rot90(3) CW
        2 => (ww - 1.0 - x, hh - 1.0 - y),       // rot90(2)
        3 => (y, ww - 1.0 - x),                  // rot90(1) CCW
        _ if k == VERTICAL => (y, ww - 1.0 - x), // rot90(1) CCW
        _ => (x, y),
    }
}

/// The forward display→deskew (post-turn→working) map. The det boxes live in
/// the `turn`ed frame, which `preproc::turn` builds with
/// `rotate_page(gray, a)`; the `warp` there computes
/// `dst(x,y) = src(c + R(a)·(dst−c))`, i.e. a source point `q` lands at
/// `dst = c + R(−a)·(q−c)`. This is the exact inverse of that warp, so a
/// post-turn point maps into the working frame where the det boxes are.
/// (Deliberately NOT derived from `preproc::unrotate_box`, whose y-translation
/// carries a pre-existing `+2·sa·cx` sign error.)
fn turn_frame_to_working(p: (f64, f64), a: f64, ww: usize, wh: usize) -> (f64, f64) {
    if a == 0.0 {
        return p;
    }
    let (px, py) = p;
    let rad = std::f64::consts::PI * a / 180.0;
    let ca = rad.cos();
    let sa = rad.sin();
    let (cx, cy) = (ww as f64 / 2.0, wh as f64 / 2.0);
    (
        cx + ca * (px - cx) + sa * (py - cy),
        cy - sa * (px - cx) + ca * (py - cy),
    )
}

/// Map a PDF-user-space point (y-up) to the working (det) frame.
fn user_to_working(
    x: f64,
    y: f64,
    layer: &LayerPage,
    k: i32,
    a: f64,
    gray: &preproc::Gray,
    render_wh: (usize, usize),
) -> (f64, f64) {
    let (bh,) = (layer.box_rect.3,);
    let (rw, rh) = render_wh;
    let sx = if layer.box_rect.2 > 0.0 {
        rw as f64 / layer.box_rect.2
    } else {
        1.0
    };
    let sy = if bh > 0.0 { rh as f64 / bh } else { 1.0 };
    let disp = user_to_display(x, y, layer.box_rect.0, layer.box_rect.1, bh, sx, sy);
    let turned = display_to_turned(disp, k, rw, rh);
    turn_frame_to_working(turned, a, gray.w, gray.h)
}

/// Expand one run-level text item into per-character centres in the working
/// frame, distributing the run box evenly across its (non-empty) characters.
/// pdf-inspector is run-level (no per-char boxes), so the run's width is
/// split evenly — good enough for line-centre placement. Spaces are kept
/// (assign_runs re-anchors them to the previous character).
fn run_char_pts(
    item: &LayerItem,
    layer: &LayerPage,
    k: i32,
    a: f64,
    gray: &preproc::Gray,
    render_wh: (usize, usize),
    run_idx: usize,
) -> Vec<CharPt> {
    // Only upright horizontal runs are fused for C2; a rotated or
    // baseline-shifted run is skipped (its line falls back to OCR).
    if item.rotation.abs() > 1.0 || item.baseline_shift.abs() > 0.5 {
        return Vec::new();
    }
    let chars: Vec<char> = item.text.chars().collect();
    if chars.is_empty() || item.x1 <= item.x0 || item.y1 <= item.y0 {
        return Vec::new();
    }
    let n = chars.len() as f64;
    let cw = (item.x1 - item.x0) / n;
    // An RTL run stores its characters right-to-left (the first PDF char is
    // the rightmost), so its x is mirrored: char i lands at x1 - (i+0.5)*cw.
    // Otherwise char i lands at x0 + (i+0.5)*cw. Either way this is the
    // physical (visual) x, which is what assign_runs needs for box containment
    // and for the left-to-right token order.
    let rtl_run = rtl::rtl_base(&item.text);
    let mut out = Vec::with_capacity(chars.len());
    for (i, c) in chars.iter().enumerate() {
        let t = (i as f64 + 0.5) * cw;
        let cxu = if rtl_run { item.x1 - t } else { item.x0 + t };
        let cyu = (item.y0 + item.y1) / 2.0;
        let (px, py) = user_to_working(cxu, cyu, layer, k, a, gray, render_wh);
        out.push(CharPt {
            c: *c,
            x: px,
            y: py,
            run: run_idx,
        });
    }
    out
}

/// Port of the reference `assign_runs`: each non-space character goes to the
/// smallest det box containing its centre (a run spanning two table cells
/// splits between them); a space follows the character before it (same box,
/// previous character's x). Then each line's runs are ordered by x (by -x
/// when the line is RTL) and joined with a single space. Returns, per det
/// line, the ordered `(char, working-x)` token list (empty for an uncovered
/// line) plus the number of characters that fell outside every box.
fn assign_runs(char_pts: &[CharPt], boxes: &[[f64; 4]]) -> (Vec<Vec<(char, f64)>>, usize) {
    let n = boxes.len();
    // line -> run idx -> ordered (char, x)
    let mut parts: Vec<BTreeMap<usize, Vec<(char, f64)>>> =
        (0..n).map(|_| BTreeMap::new()).collect();
    let mut lost = 0usize;
    if n == 0 {
        let lost = char_pts.iter().filter(|c| !c.c.is_whitespace()).count();
        return (Vec::new(), lost);
    }
    let area: Vec<f64> = boxes
        .iter()
        .map(|b| (b[2] - b[0]).max(1e-6) * (b[3] - b[1]).max(1e-6))
        .collect();
    // (line, x) of the last placed non-space char, for space follow-up.
    let mut last: Option<(usize, f64)> = None;
    for cp in char_pts {
        if cp.c.is_whitespace() {
            if let Some((j, lx)) = last {
                parts[j].entry(cp.run).or_default().push((cp.c, lx));
            }
            continue;
        }
        let inside: Vec<usize> = (0..n)
            .filter(|&i| {
                boxes[i][0] <= cp.x
                    && cp.x <= boxes[i][2]
                    && boxes[i][1] <= cp.y
                    && cp.y <= boxes[i][3]
            })
            .collect();
        if inside.is_empty() {
            lost += 1;
            last = None;
            continue;
        }
        let j = *inside
            .iter()
            .min_by(|a, b| area[**a].total_cmp(&area[**b]))
            .unwrap();
        parts[j].entry(cp.run).or_default().push((cp.c, cp.x));
        last = Some((j, cp.x));
    }
    // Visual (left-to-right) token order per line. Within a run the tokens
    // keep their content order, except that an RTL run — stored right-to-left
    // — is reversed, because on the page its characters read right-to-left.
    // The runs themselves are ordered left to right (ascending min-x). That
    // visual order is what markdown::decode_tokens expects for Ch tokens: it
    // joins them and applies the RTL flip to reach the logical reading order —
    // byte-equivalent to the reference's 3-tuple path (logical order, no
    // flip). Never x-sort inside a run: a space carries the previous char's
    // x and would be reordered off its position.
    let mut lines: Vec<Vec<(char, f64)>> = Vec::with_capacity(n);
    for line in parts.iter() {
        if line.is_empty() {
            lines.push(Vec::new());
            continue;
        }
        let mut runs: Vec<Vec<(char, f64)>> = line
            .values()
            .map(|v| {
                let s: String = v.iter().map(|(c, _)| *c).collect();
                let mut t = v.clone();
                if rtl::rtl_base(&s) {
                    t.reverse();
                }
                t
            })
            .collect();
        runs.sort_by(|a, b| {
            let mina = a.iter().map(|(_, x)| *x).fold(f64::INFINITY, f64::min);
            let minb = b.iter().map(|(_, x)| *x).fold(f64::INFINITY, f64::min);
            mina.total_cmp(&minb)
        });
        let mut tk: Vec<(char, f64)> = Vec::new();
        for part in runs {
            if !tk.is_empty() {
                let last_x = tk.last().map(|(_, x)| *x).unwrap_or(0.0);
                tk.push((' ', last_x));
            }
            tk.extend(part);
        }
        lines.push(tk);
    }
    (lines, lost)
}

/// Decode a line's ordered characters into its text: join, collapse
/// whitespace runs to one space. The NFC and RTL flip are applied downstream
/// by `markdown::decode_tokens`, so they are not repeated here.
pub(crate) fn decode_layer_line(line: &[(char, f64)]) -> String {
    let joined: String = line.iter().map(|(c, _)| *c).collect();
    collapse_ws(&joined)
}

/// ' '.join(s.split()): collapse every whitespace run to one space, trim.
fn collapse_ws(s: &str) -> String {
    let mut out = String::new();
    let mut pending = false;
    for c in s.chars() {
        if c.is_whitespace() {
            pending = true;
        } else {
            if pending && !out.is_empty() {
                out.push(' ');
            }
            pending = false;
            out.push(c);
        }
    }
    out
}

/// The fusion result for one det line: the ordered `(char, working-frame x)`
/// token list. `None` when the line has no layer characters (keep the OCR
/// result).
#[derive(Debug, Clone)]
pub(crate) struct FusedLine {
    pub tokens: Vec<(char, f64)>,
}

/// Fuse the page's text layer onto the detected lines. Returns one entry per
/// det line: `Some` when the line is covered by layer characters (use them),
/// `None` when it is not (keep the OCR reading). Also returns the number of
/// layer characters that fell outside every box (for the diagnostic note).
pub(crate) fn fuse_lines(
    layer: &LayerPage,
    boxes: &[[f64; 4]],
    k: i32,
    a: f64,
    gray: &preproc::Gray,
    render_wh: (usize, usize),
) -> (Vec<Option<FusedLine>>, usize) {
    let mut char_pts: Vec<CharPt> = Vec::new();
    for (i, item) in layer.items.iter().enumerate() {
        char_pts.extend(run_char_pts(item, layer, k, a, gray, render_wh, i));
    }
    if char_pts.is_empty() {
        return (vec![None; boxes.len()], 0);
    }
    let (lines, lost) = assign_runs(&char_pts, boxes);
    let out = lines
        .into_iter()
        .map(|l| {
            let text = decode_layer_line(&l);
            if text.trim().is_empty() {
                None
            } else {
                Some(FusedLine { tokens: l })
            }
        })
        .collect();
    (out, lost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l1::indra::preproc::Gray;

    fn gray(w: usize, h: usize) -> Gray {
        Gray {
            w,
            h,
            data: vec![255u8; w * h],
        }
    }

    // The display map (rot=0) must agree with the in-repo, tested
    // blocks::pdf_box_to_norm 0° branch, scaled to pixels.
    #[test]
    fn display_map_matches_blocks() {
        // box at origin 0,0 size 100x200 pts, scale 2 px/pt -> 200x400 px.
        let (px, py) = user_to_display(10.0, 50.0, 0.0, 0.0, 200.0, 2.0, 2.0);
        // pdf_box_to_norm(10,50,10,50,...) 0°: x 0.1, y_top 0.75 -> px 20, py 300.
        assert!((px - 20.0).abs() < 1e-9);
        assert!((py - 300.0).abs() < 1e-9);
    }

    // The deskew forward map must agree with the REAL warp: paint one ink
    // pixel at a post-turn point, run preproc::rotate_page (the exact warp the
    // pipeline applies), and the darkest dst pixel must land within 1 px of
    // the predicted working coordinate.
    #[test]
    fn deskew_forward_matches_real_warp() {
        let (ww, wh) = (200usize, 300usize);
        for &a in &[0.7f64, -1.2, 2.5] {
            let q = (57.0, 243.0); // post-turn point
            let mut src = gray(ww, wh);
            src.data[(q.1 as usize) * ww + (q.0 as usize)] = 0; // ink
            let dst = preproc::rotate_page(&src, a);
            let (bx, by) = turn_frame_to_working(q, a, ww, wh);
            let mut best: (usize, usize) = (0, 0);
            let mut best_v = u8::MAX;
            for y in 0..wh {
                for x in 0..ww {
                    if dst.data[y * ww + x] < best_v {
                        best_v = dst.data[y * ww + x];
                        best = (x, y);
                    }
                }
            }
            assert!(best_v < 250, "ink lost a={a}");
            assert!(
                (best.0 as f64 - bx.round()).abs() <= 1.0
                    && (best.1 as f64 - by.round()).abs() <= 1.0,
                "a={a} predicted=({bx},{by}) actual={best:?}"
            );
        }
    }

    // The rot90 maps must agree with the REAL preproc::rot90: paint one ink
    // pixel, rotate, and the darkest dst pixel must land within 1 px of the
    // predicted coordinate for each quarter turn.
    #[test]
    fn rot90_maps_match_real() {
        for (k, (w, h)) in [
            (1i32, (100usize, 200usize)),
            (2, (100, 200)),
            (3, (100, 200)),
            (VERTICAL, (100, 200)),
        ] {
            let (w, h) = (w, h);
            let q = (10.0, 30.0); // display point
            let mut src = gray(w, h);
            src.data[(q.1 as usize) * w + (q.0 as usize)] = 0;
            // turn applies rot90(-k) for k in {1,2,3}; VERTICAL -> rot90(1).
            let r90k = if k == VERTICAL { 1 } else { (-k).rem_euclid(4) };
            let dst = preproc::rot90(&src, r90k);
            let (bx, by) = display_to_turned(q, k, w, h);
            let mut best: (usize, usize) = (0, 0);
            let mut best_v = u8::MAX;
            for y in 0..dst.h {
                for x in 0..dst.w {
                    if dst.data[y * dst.w + x] < best_v {
                        best_v = dst.data[y * dst.w + x];
                        best = (x, y);
                    }
                }
            }
            assert!(best_v < 250, "ink lost k={k}");
            assert!(
                (best.0 as f64 - bx.round()).abs() <= 1.0
                    && (best.1 as f64 - by.round()).abs() <= 1.0,
                "k={k} predicted=({bx},{by}) actual={best:?}"
            );
        }
    }

    // assign_runs: a run spanning two side-by-side boxes splits between them;
    // an in-run space is kept (follows the previous char); a char outside
    // every box is lost.
    #[test]
    fn assign_runs_splits_cells_keeps_spaces_counts_lost() {
        // Two boxes side by side.
        let boxes = [[0.0, 0.0, 50.0, 20.0], [60.0, 0.0, 120.0, 20.0]];
        // One run: "ab" in the left box, "cd" in the right (spans both).
        let cps = vec![
            CharPt {
                c: 'a',
                x: 5.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'b',
                x: 20.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'c',
                x: 70.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'd',
                x: 90.0,
                y: 10.0,
                run: 0,
            },
        ];
        let (lines, lost) = assign_runs(&cps, &boxes);
        assert_eq!(lost, 0);
        assert_eq!(decode_layer_line(&lines[0]), "ab");
        assert_eq!(decode_layer_line(&lines[1]), "cd");

        // In-run space is kept: "he llo" across the left box (run 0) plus a
        // second run "world" -> "he llo world" with the space preserved.
        let cps2 = vec![
            CharPt {
                c: 'h',
                x: 5.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'e',
                x: 10.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: ' ',
                x: 15.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'l',
                x: 20.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'l',
                x: 25.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'o',
                x: 30.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'w',
                x: 40.0,
                y: 10.0,
                run: 1,
            },
            CharPt {
                c: 'o',
                x: 45.0,
                y: 10.0,
                run: 1,
            },
            CharPt {
                c: 'r',
                x: 48.0,
                y: 10.0,
                run: 1,
            },
            CharPt {
                c: 'l',
                x: 51.0,
                y: 10.0,
                run: 1,
            },
            CharPt {
                c: 'd',
                x: 54.0,
                y: 10.0,
                run: 1,
            },
        ];
        // All in one box (0..130) so they land on a single line.
        let boxes1 = [[0.0, 0.0, 130.0, 20.0]];
        let (lines2, lost2) = assign_runs(&cps2, &boxes1);
        assert_eq!(lost2, 0);
        assert_eq!(decode_layer_line(&lines2[0]), "he llo world");

        // A char outside every box is lost.
        let cps3 = vec![
            CharPt {
                c: 'a',
                x: 5.0,
                y: 10.0,
                run: 0,
            },
            CharPt {
                c: 'z',
                x: 300.0,
                y: 10.0,
                run: 0,
            },
        ];
        let (_, lost3) = assign_runs(&cps3, &boxes);
        assert_eq!(lost3, 1);
    }

    // C2c byte-parity gate (R2): run the SAME per-character centres the
    // reference indra_ocr.document.assign_runs sees (even advance + RTL
    // mirror, working-frame units) through MY assign_runs, then through the
    // REAL markdown::decode_tokens, and assert the final text is byte-identical
    // to the reference oracle (/tmp/c2c/gen.py = assign_runs + _decode, which
    // emits logical order WITHOUT a flip). This isolates the algorithm (box
    // assignment, cell split, space follow, run ordering) + the decode path;
    // the coordinate transform is covered by the geometry tests above.
    fn centers(
        text: &str,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        rtl: bool,
        run: usize,
    ) -> Vec<CharPt> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let n = chars.len() as f64;
        let cw = (x1 - x0) / n;
        chars
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let t = (i as f64 + 0.5) * cw;
                let cx = if rtl { x1 - t } else { x0 + t };
                CharPt {
                    c,
                    x: cx,
                    y: (y0 + y1) / 2.0,
                    run,
                }
            })
            .collect()
    }

    // Decode a fused line exactly as the pipeline does: Tok::Ch sequence ->
    // the real markdown::decode_tokens with the page-level rtl flag.
    fn decode_page_line(line: &[(char, f64)], page_rtl: bool) -> String {
        let vocab = crate::l1::indra::vocab::Vocab {
            chars: vec!["a".into(), "b".into(), " ".into()],
            has_commands: false,
            jamo: false,
        };
        let toks: Vec<crate::l1::indra::markdown::Tok> = line
            .iter()
            .map(|(c, x)| crate::l1::indra::markdown::Tok::Ch(*c, *x))
            .collect();
        crate::l1::indra::markdown::decode_tokens(&toks, &vocab, page_rtl)
    }

    #[test]
    fn assign_runs_matches_reference_oracle() {
        // One oracle case: char points, det boxes, page rtl flag, the
        // expected decoded text per box (reference indra_ocr oracle values).
        struct Case {
            name: &'static str,
            cps: Vec<CharPt>,
            boxes: Vec<[f64; 4]>,
            page_rtl: bool,
            expected: Vec<String>,
        }
        let cases: Vec<Case> = vec![
            Case {
                name: "ltr_cell_split",
                cps: centers("abcd", 0.0, 0.0, 120.0, 20.0, false, 0),
                boxes: vec![[0.0, 0.0, 50.0, 20.0], [60.0, 0.0, 120.0, 20.0]],
                page_rtl: false,
                expected: vec!["ab".into(), "cd".into()],
            },
            Case {
                name: "ltr_inrun_space_two_runs",
                cps: centers("he llo", 0.0, 0.0, 40.0, 20.0, false, 0)
                    .into_iter()
                    .chain(centers("world", 40.0, 0.0, 80.0, 20.0, false, 1))
                    .collect(),
                boxes: vec![[0.0, 0.0, 80.0, 20.0]],
                page_rtl: false,
                expected: vec!["he llo world".into()],
            },
            Case {
                name: "ltr_run_order_by_x",
                cps: centers("gamma", 60.0, 0.0, 90.0, 20.0, false, 0)
                    .into_iter()
                    .chain(centers("alpha", 0.0, 0.0, 30.0, 20.0, false, 1))
                    .chain(centers("beta", 30.0, 0.0, 60.0, 20.0, false, 2))
                    .collect(),
                boxes: vec![[0.0, 0.0, 90.0, 20.0]],
                page_rtl: false,
                expected: vec!["alpha beta gamma".into()],
            },
            Case {
                name: "ltr_two_stacked_lines",
                cps: centers("top line", 0.0, 0.0, 80.0, 10.0, false, 0)
                    .into_iter()
                    .chain(centers("bottom line", 0.0, 30.0, 80.0, 40.0, false, 1))
                    .collect(),
                boxes: vec![[0.0, 0.0, 80.0, 10.0], [0.0, 30.0, 80.0, 40.0]],
                page_rtl: false,
                expected: vec!["top line".into(), "bottom line".into()],
            },
            // RTL: the reference emits logical order (right run first) WITHOUT
            // a flip; my visual-order tokens + decode_tokens flip must land on
            // the same bytes.
            Case {
                name: "rtl_single_run",
                cps: centers("שלום עולם", 0.0, 0.0, 160.0, 20.0, true, 0),
                boxes: vec![[0.0, 0.0, 160.0, 20.0]],
                page_rtl: true,
                expected: vec!["שלום עולם".into()],
            },
            Case {
                name: "rtl_two_runs_right_first",
                cps: centers("בבב", 60.0, 0.0, 90.0, 20.0, true, 0)
                    .into_iter()
                    .chain(centers("אאא", 0.0, 0.0, 30.0, 20.0, true, 1))
                    .collect(),
                boxes: vec![[0.0, 0.0, 90.0, 20.0]],
                page_rtl: true,
                expected: vec!["בבב אאא".into()],
            },
            Case {
                name: "rtl_cell_split",
                cps: centers("abcdef", 0.0, 0.0, 120.0, 20.0, false, 0),
                boxes: vec![[0.0, 0.0, 50.0, 20.0], [60.0, 0.0, 120.0, 20.0]],
                page_rtl: false,
                expected: vec!["abc".into(), "def".into()],
            },
        ];

        for case in &cases {
            let (lines, _lost) = assign_runs(&case.cps, &case.boxes);
            assert_eq!(
                lines.len(),
                case.expected.len(),
                "{}: line count",
                case.name
            );
            for (i, line) in lines.iter().enumerate() {
                let got = decode_page_line(line, case.page_rtl);
                assert_eq!(
                    got, case.expected[i],
                    "{}: line {i} (got {got:?})",
                    case.name
                );
            }
        }
    }
}
