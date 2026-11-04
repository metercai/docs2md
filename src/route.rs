//! Routing: G1 (layout-based L1/L2 split) + G2 (output-level fallback).
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
//! G2 is the output-level gate applied to an L1 result (per the fit plan):
//!   * md_chars < min_chars AND mean_conf < hard_conf  => HARD fail
//!   * md_chars < min_chars AND mean_conf < mid_conf   => MEDIUM doubt
//!   * otherwise                                       => OK
//! A HARD/MEDIUM page is escalated to L2 (if available); if L2 is not
//! available the L1 output is kept and the page is marked "needs a
//! higher-tier OCR model".

use rusto::{LayoutDetector, LayoutOutput, LayoutRegion, LayoutType};

use crate::config::RouteConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    L1,
    L2,
}

/// G2 verdict for one L1 result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G2Verdict {
    /// Keep the L1 output.
    Ok,
    /// Escalate to L2 (medium doubt — conservative).
    Medium,
    /// Escalate to L2 (hard fail — low char count and low conf).
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

/// Run G1 on a rendered PNG. `layout` is the (lazily built) detector; when it
/// is `None` (no external layout model configured / failed to load) every
/// OCR page defaults to L1 and G2 may still escalate to L2.
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

/// Apply G2 to an L1 result.
pub fn g2(chars: usize, mean_conf: f32, rc: &RouteConfig) -> G2Verdict {
    if chars >= rc.g2.min_chars {
        return G2Verdict::Ok;
    }
    if mean_conf < rc.g2.hard_conf {
        G2Verdict::Hard
    } else if mean_conf < rc.g2.mid_conf {
        G2Verdict::Medium
    } else {
        G2Verdict::Ok
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
    fn g2_hard_fail() {
        // 20 chars (< 60) and mean conf 0.3 (< 0.5) => hard
        assert_eq!(g2(20, 0.3, &rc()), G2Verdict::Hard);
    }

    #[test]
    fn g2_medium_doubt() {
        // 20 chars (< 60) and mean conf 0.6 (>= 0.5, < 0.75) => medium
        assert_eq!(g2(20, 0.6, &rc()), G2Verdict::Medium);
    }

    #[test]
    fn g2_ok_low_conf_but_enough_chars() {
        // 500 chars (>= 60) with low conf => ok (high-output low-conf page)
        assert_eq!(g2(500, 0.2, &rc()), G2Verdict::Ok);
    }

    #[test]
    fn g2_ok_enough_chars() {
        assert_eq!(g2(300, 0.9, &rc()), G2Verdict::Ok);
    }
}
