//! Routing: G1 (layout-based L1/L2 split) + G3 (output-level fallback gate).
//!
//! G1 runs the external PP-DocLayout model on a rendered page and applies
//! the layout rules (validated on the 623-page fixture corpus):
//!   * any table-caption region            => L2
//!   * a COMPACT complex region (table /
//!     figure / equation whose bbox covers
//!     < compact_frac of the page and
//!     conf > compact_conf)                => L2
//!   * a DENSE-TEXT blob (a TEXT region
//!     covering >= dense_frac of the page
//!     with conf >= dense_conf)            => L2
//!   * otherwise                           => L1
//!
//! G3 is the output-level gate applied to an L1 engine's result (per the
//! fit plan). It is the primary check on the Indra output (chars = markdown
//! length) and the fallback check on the PP-OCRv6 output (chars = line-text
//! total); both share the `[route.g3]` thresholds:
//!
//! * chars < min_chars AND mean_conf < hard_conf  => HARD fail
//! * chars < min_chars AND mean_conf < mid_conf   => MEDIUM doubt
//! * otherwise                                    => OK
//!
//! A HARD/MEDIUM page falls back to the other engine (Indra -> PP-OCRv6); if
//! the fallback is unavailable or also fails, the page is escalated to L2 (if
//! available); otherwise the best L1 output is kept and the page is marked
//! "needs a higher-tier OCR model".

use rusto::{LayoutDetector, LayoutOutput, LayoutRegion, LayoutType};

use crate::config::RouteConfig;
#[cfg(test)]
use crate::l1::OcrLine;
use crate::l1::OcrPage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    L1,
    L2,
}

/// G3 verdict for one L1 engine's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G3Verdict {
    /// Keep the engine output.
    Ok,
    /// Fall back / escalate (medium doubt — conservative).
    Medium,
    /// Fall back / escalate (hard fail — low char count and low conf).
    Hard,
}

/// Apply G1 layout routing. `page_area` is the rendered page pixel area.
pub fn route_from_layout(regions: &[LayoutRegion], page_area: f64, rc: &RouteConfig) -> Tier {
    // (a) any table-caption region => L2 (the strongest complex signal).
    for r in regions {
        if r.layout_type == LayoutType::TableCaption {
            return Tier::L2;
        }
    }
    // (b) compact complex region => L2.
    for r in regions {
        if matches!(
            r.layout_type,
            LayoutType::Table | LayoutType::Figure | LayoutType::Equation
        ) {
            let rw = (r.bbox.x_max - r.bbox.x_min) as f64;
            let rh = (r.bbox.y_max - r.bbox.y_min) as f64;
            let frac = if page_area > 0.0 {
                (rw * rh) / page_area
            } else {
                1.0
            };
            if frac < rc.compact_frac as f64 && r.confidence > rc.compact_conf {
                return Tier::L2;
            }
        }
    }
    // (c) dense-text blob => L2. TEXT type only: the layout model emits a
    // full-page high-conf table for ordinary text pages, so a full-page
    // TABLE is a known false positive and must not count as "dense".
    for r in regions {
        if r.layout_type == LayoutType::Text {
            let rw = (r.bbox.x_max - r.bbox.x_min) as f64;
            let rh = (r.bbox.y_max - r.bbox.y_min) as f64;
            let frac = if page_area > 0.0 {
                (rw * rh) / page_area
            } else {
                1.0
            };
            if frac >= rc.dense_frac as f64 && r.confidence >= rc.dense_conf {
                return Tier::L2;
            }
        }
    }
    Tier::L1
}

/// Structural signal for the S0 reroute check (plan C1): true when the page
/// carries a table-caption region (rule a) or a COMPACT complex region
/// (rule b). Deliberately excludes rule c (dense text): an S0 text page
/// legitimately contains large high-conf TEXT regions, dense text is not a
/// structural block, and OCR would be slower and less accurate than the
/// native layer for it.
pub fn structural_from_layout(
    regions: &[LayoutRegion],
    page_area: f64,
    rc: &RouteConfig,
) -> bool {
    // (a) any table-caption region.
    for r in regions {
        if r.layout_type == LayoutType::TableCaption {
            return true;
        }
    }
    // (b) compact complex region.
    for r in regions {
        if matches!(
            r.layout_type,
            LayoutType::Table | LayoutType::Figure | LayoutType::Equation
        ) {
            let rw = (r.bbox.x_max - r.bbox.x_min) as f64;
            let rh = (r.bbox.y_max - r.bbox.y_min) as f64;
            let frac = if page_area > 0.0 {
                (rw * rh) / page_area
            } else {
                1.0
            };
            if frac < rc.compact_frac as f64 && r.confidence > rc.compact_conf {
                return true;
            }
        }
    }
    false
}

/// Run the structural check (see [`structural_from_layout`]) on a rendered
/// PNG. `None` when no layout detector is available or detection fails.
pub fn g1_structural(
    layout: Option<&mut LayoutDetector>,
    png: &[u8],
    page_area: f64,
    rc: &RouteConfig,
) -> Option<bool> {
    let ld = layout?;
    let dyn_img = image::load_from_memory(png).ok()?;
    let mat = rusto::image_impl::Mat::new(dyn_img);
    let out: LayoutOutput = ld.detect(&mat).ok()?;
    Some(structural_from_layout(&out.regions, page_area, rc))
}

/// Run G1 on a rendered PNG. `layout` is the (lazily built) detector; when it
/// is `None` (no external layout model configured / failed to load) every
/// OCR page defaults to L1 and G3 may still escalate to L2.
pub fn g1(
    layout: Option<&mut LayoutDetector>,
    png: &[u8],
    page_area: f64,
    rc: &RouteConfig,
) -> Option<Tier> {
    let ld = layout?;
    let dyn_img = image::load_from_memory(png).ok()?;
    let mat = rusto::image_impl::Mat::new(dyn_img);
    let out: LayoutOutput = ld.detect(&mat).ok()?;
    Some(route_from_layout(&out.regions, page_area, rc))
}

/// Apply G3 (the output-level gate) to an L1 engine's result. `chars` is the
/// engine's output-size signal — the Indra markdown length for the primary
/// (G3) check, or the PP-OCRv6 line-text total for the fallback (G2) check;
/// `mean_conf` is the mean per-line confidence. Both applications share the
/// `[route.g3]` thresholds (only the `chars` signal differs).
pub fn g3(chars: usize, mean_conf: f32, rc: &RouteConfig) -> G3Verdict {
    if chars >= rc.g3.min_chars {
        return G3Verdict::Ok;
    }
    if mean_conf < rc.g3.hard_conf {
        G3Verdict::Hard
    } else if mean_conf < rc.g3.mid_conf {
        G3Verdict::Medium
    } else {
        G3Verdict::Ok
    }
}

/// Terminal decision for one G1=L1 page in the dual-engine G3 flow (plan
/// §集成语义, fit §7.2). Pure over engine *results* so the branches are
/// unit-testable without models: the primary (Indra) is gated by G3 on its
/// markdown length; a failure runs the PP-OCRv6 fallback at most once, gated
/// by the same thresholds on its line-text total (the "G2" check); a second
/// failure escalates to L2.
///
/// `fallback`: `None` = no PP-OCRv6 engine (not configured, or init failed,
/// ¬A_P — escalate with the primary output kept), `Some(Err)` = the fallback
/// ran but the page failed (same: keep the primary), `Some(Ok)` = its page
/// result. `Escalate.keep` is the output retained when L2 is unavailable:
/// the PP output when it produced one, otherwise the primary's.
#[derive(Debug)]
pub enum L1Outcome {
    /// S1a: the primary's output is kept.
    Primary(OcrPage),
    /// S1b: the fallback's output is kept (the primary failed G3; PP plain
    /// text, no Indra tables).
    Fallback(OcrPage),
    /// Escalate to L2. `tag` is the fallback gate's `hard`/`medium` verdict;
    /// `keep` is the output to retain when L2 is unavailable (with a
    /// "needs a higher-tier OCR model" note, plan line 117).
    Escalate {
        tag: &'static str,
        keep: OcrPage,
    },
}

/// Output-size signal for the output-level gate: the engine markdown's
/// length when the engine produces one (Indra), otherwise the line-text
/// total (PP-OCRv6).
pub fn output_chars(page: &OcrPage) -> usize {
    page.md.as_ref().map_or_else(|| page.char_count(), |m| m.chars().count())
}

/// Decide the dual-engine G3 flow for one rendered page (see
/// [`L1Outcome`]). `primary` is the (already successful) Indra result;
/// `fallback` is the PP-OCRv6 result when it was run (at most once per
/// page). Logs the fallback decision (page number, trigger criterion, both
/// engines' char/conf summary) so the route is observable (fit plan §5.2).
pub fn decide_l1_flow(
    page1: u32,
    primary: OcrPage,
    fallback: Option<Result<OcrPage, String>>,
    rc: &RouteConfig,
) -> L1Outcome {
    let chars = output_chars(&primary);
    let conf = primary.mean_conf();
    if g3(chars, conf, rc) == G3Verdict::Ok {
        return L1Outcome::Primary(primary);
    }
    // G3 failed: fall back to PP-OCRv6 (at most once per page).
    let vtag = if g3(chars, conf, rc) == G3Verdict::Hard { "hard" } else { "medium" };
    let fb = match fallback {
        None => {
            // No PP-OCRv6 engine (¬A_P): escalate, keep the primary output.
            eprintln!(
                "[p{page1}] G3 {vtag} (md_chars={chars}, conf={conf:.3}) — no PP-OCRv6 \
                 fallback — escalating to L2"
            );
            return L1Outcome::Escalate {
                tag: vtag,
                keep: primary,
            };
        }
        Some(Err(e)) => {
            // The fallback ran but failed: escalate, keep the primary output.
            eprintln!(
                "[p{page1}] G3 {vtag} (md_chars={chars}, conf={conf:.3}); PP-OCRv6 \
                 fallback failed: {e} — escalating to L2"
            );
            return L1Outcome::Escalate {
                tag: vtag,
                keep: primary,
            };
        }
        Some(Ok(pp)) => pp,
    };
    // G2 (same thresholds, line-text total) on the fallback output.
    let pc = fb.char_count();
    let pconf = fb.mean_conf();
    match g3(pc, pconf, rc) {
        G3Verdict::Ok => {
            eprintln!(
                "[p{page1}] G3 {vtag} (md_chars={chars}, conf={conf:.3}); PP-OCRv6 ok \
                 (chars={pc}, conf={pconf:.3}) — using PP-OCRv6"
            );
            L1Outcome::Fallback(fb)
        }
        v @ (G3Verdict::Hard | G3Verdict::Medium) => {
            let t = if v == G3Verdict::Hard { "hard" } else { "medium" };
            eprintln!(
                "[p{page1}] G3 {vtag} (md_chars={chars}, conf={conf:.3}); G2 {t} \
                 (chars={pc}, conf={pconf:.3}) — escalating to L2"
            );
            // Plan line 117: when L2 is unavailable, the retained output is
            // the PP result (not the primary's).
            L1Outcome::Escalate { tag: t, keep: fb }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusto::layout::BBox;

    fn region(t: LayoutType, x0: i32, y0: i32, x1: i32, y1: i32, conf: f32) -> LayoutRegion {
        LayoutRegion {
            bbox: BBox {
                x_min: x0,
                y_min: y0,
                x_max: x1,
                y_max: y1,
            },
            layout_type: t,
            confidence: conf,
        }
    }

    fn rc() -> RouteConfig {
        RouteConfig::default()
    }

    #[test]
    fn table_caption_always_l2() {
        let r = region(LayoutType::TableCaption, 10, 10, 50, 60, 0.1);
        assert_eq!(route_from_layout(&[r], 1000.0, &rc()), Tier::L2);
    }

    #[test]
    fn compact_table_l2() {
        // bbox covers 24% of a 1000x1000 page, conf 0.79
        let r = region(LayoutType::Table, 100, 100, 500, 700, 0.79);
        let area = 1000.0 * 1000.0;
        assert_eq!(route_from_layout(&[r], area, &rc()), Tier::L2);
    }

    #[test]
    fn full_page_table_is_not_l2_by_area_rule() {
        // full-page high-conf table (the known layout false positive on text
        // pages): frac ~= 1.0 (not < 0.5), so the compact rule does not fire.
        let r = region(LayoutType::Table, 0, 0, 999, 999, 0.96);
        let area = 1000.0 * 1000.0;
        assert_eq!(route_from_layout(&[r], area, &rc()), Tier::L1);
    }

    #[test]
    fn dense_text_blob_l2() {
        // a TEXT region covering >= 85% of the page with conf >= 0.3
        let r = region(LayoutType::Text, 0, 0, 960, 900, 0.45);
        let area = 1000.0 * 1000.0;
        assert_eq!(route_from_layout(&[r], area, &rc()), Tier::L2);
    }

    #[test]
    fn plain_text_page_l1() {
        // many compact low-conf text bars (normal reading page)
        let rs: Vec<LayoutRegion> = (0..5)
            .map(|i| region(LayoutType::Text, 0, i * 200, 900, i * 200 + 150, 0.05))
            .collect();
        let area = 1000.0 * 1000.0;
        assert_eq!(route_from_layout(&rs, area, &rc()), Tier::L1);
    }

    #[test]
    fn g3_hard_fail() {
        // 20 chars (< 60) and mean conf 0.3 (< 0.5) => hard
        assert_eq!(g3(20, 0.3, &rc()), G3Verdict::Hard);
    }

    #[test]
    fn g3_medium_doubt() {
        // 20 chars (< 60) and mean conf 0.6 (>= 0.5, < 0.75) => medium
        assert_eq!(g3(20, 0.6, &rc()), G3Verdict::Medium);
    }

    #[test]
    fn g3_ok_low_conf_but_enough_chars() {
        // 500 chars (>= 60) with low conf => ok (high-output low-conf page)
        assert_eq!(g3(500, 0.2, &rc()), G3Verdict::Ok);
    }

    #[test]
    fn g3_ok_enough_chars() {
        assert_eq!(g3(300, 0.9, &rc()), G3Verdict::Ok);
    }

    // --- dual-engine G3 flow (plan §集成语义) -----------------------------
    /// An Indra-style page: markdown of `mdlen` chars, one line at `conf`.
    fn indra(mdlen: usize, conf: f32) -> OcrPage {
        OcrPage {
            lines: vec![OcrLine {
                text: "x".to_string(),
                conf,
                top: 0.0,
                left: 0.0,
                width: 0.0,
                height: 0.0,
            }],
            md: Some("x".repeat(mdlen)),
            blocks: Vec::new(),
        }
    }
    /// A PP-OCRv6-style page: `nchars` of line text, one line at `conf`.
    fn pp(nchars: usize, conf: f32) -> OcrPage {
        OcrPage {
            lines: vec![OcrLine {
                text: "y".repeat(nchars),
                conf,
                top: 0.0,
                left: 0.0,
                width: 0.0,
                height: 0.0,
            }],
            md: None,
            blocks: Vec::new(),
        }
    }

    // S1a: G3 passes on the primary markdown → keep the Indra output.
    #[test]
    fn flow_s1a_primary_passes() {
        let out = decide_l1_flow(1, indra(500, 0.9), None, &rc());
        assert!(matches!(out, L1Outcome::Primary(_)));
    }

    // S1b: G3 fails on a short/low-conf primary, PP passes → PP output kept.
    #[test]
    fn flow_s1b_fallback() {
        let out =
            decide_l1_flow(1, indra(20, 0.3), Some(Ok(pp(200, 0.9))), &rc());
        match out {
            L1Outcome::Fallback(p) => assert_eq!(p.char_count(), 200),
            other => panic!("expected Fallback, got {other:?}"),
        }
    }

    // ¬A_P: G3 fails and there is no PP engine → escalate, keep the primary.
    #[test]
    fn flow_no_fallback_escalates_keep_primary() {
        let out = decide_l1_flow(1, indra(20, 0.3), None, &rc());
        match out {
            L1Outcome::Escalate { tag, keep } => {
                assert_eq!(tag, "hard");
                assert_eq!(keep.md.as_deref(), Some("x".repeat(20).as_str()));
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // G2 re-fail: PP ran but its output also fails → escalate, keep the PP.
    #[test]
    fn flow_g2_refail_escalates_keep_pp() {
        let out =
            decide_l1_flow(1, indra(20, 0.3), Some(Ok(pp(10, 0.2))), &rc());
        match out {
            L1Outcome::Escalate { tag, keep } => {
                assert_eq!(tag, "hard");
                // keep is the PP output (plan line 117), not the primary.
                assert_eq!(keep.char_count(), 10);
                assert!(keep.md.is_none());
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // PP ran but errored → escalate, keep the primary.
    #[test]
    fn flow_fallback_error_escalates_keep_primary() {
        let out = decide_l1_flow(
            1,
            indra(20, 0.6),
            Some(Err("boom".to_string())),
            &rc(),
        );
        match out {
            L1Outcome::Escalate { tag, keep } => {
                assert_eq!(tag, "medium");
                assert_eq!(keep.md.as_deref(), Some("x".repeat(20).as_str()));
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // output_chars: the engine markdown length wins over the line-text total.
    #[test]
    fn output_chars_prefers_md() {
        assert_eq!(output_chars(&indra(123, 0.9)), 123);
        assert_eq!(output_chars(&pp(77, 0.9)), 77);
    }
}
