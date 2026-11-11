//! Routing: the L1→L2 serial metric chain (plan §2.1).
//!
//! The chain is evaluated as the indra pipeline advances — after det, after
//! the script pre-pass, after rec, after assemble — and the first hit stops
//! the pipeline and escalates the page to L2 (PP-OCRv6). Every stage reads
//! an intermediate quantity the pipeline already computes (no extra
//! inference); the thresholds live in `[route.chain]` (config.rs, the
//! single source of truth).
//!
//! The chain replaces the former G1 layout routing (render + PP-DocLayout)
//! and the former G3 output-level dual-engine flow: routing no longer loads
//! or runs any external layout model (decision B), and PP-OCRv6 is reached
//! only through the chain (decision E) — or through [`output_gate`] on the
//! `use_indra = false` rollback path, where the pipeline intermediates do
//! not exist.

use crate::config::{L3Config, RouteConfig};
use crate::l1::OcrPage;

/// The script pre-pass metrics for one page (plan §2.1 [script] stage),
/// computed by `rec::route_script` from the per-line script routing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScriptMetrics {
    /// Total routed lines.
    pub n: usize,
    /// Lines whose raw classifier argmax is the cjk group (a Chinese-major
    /// page goes to the engine that reads Chinese better: PP-OCRv6).
    pub cjk: usize,
    /// Lines the classifier is not sure about (double-read from the base
    /// recognizer).
    pub unsure: usize,
    /// The dominant raw script group (mode; alphabetical tie-break).
    pub dominant: Option<String>,
    /// Whether this build loads a rec variant for the dominant group
    /// (base always counts as loaded).
    pub dominant_has_variant: bool,
}

impl ScriptMetrics {
    pub fn cjk_frac(&self) -> f32 {
        self.frac(self.cjk)
    }
    pub fn unsure_frac(&self) -> f32 {
        self.frac(self.unsure)
    }
    fn frac(&self, k: usize) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            k as f32 / self.n as f32
        }
    }
}

/// Chain-stage tags, in pipeline order: the value carried in
/// `OcrPage.escalate_l2` and in the page's tier string.
pub const TAG_DET: &str = "det";
pub const TAG_CJK: &str = "cjk";
pub const TAG_NOVARIANT: &str = "novariant";
pub const TAG_UNSURE: &str = "unsure";
pub const TAG_CONF: &str = "conf";
pub const TAG_CHARS: &str = "chars";

/// [det] stage (plan §2.1): fewer line boxes than `det_min_lines` => L2 —
/// a near-image page, escalated before any recognition (the whole rec pass
/// is saved).
pub fn chain_det(boxes: usize, rc: &RouteConfig) -> Option<&'static str> {
    if boxes < rc.chain.det_min_lines {
        Some(TAG_DET)
    } else {
        None
    }
}

/// [script] stage (plan §2.1), evaluated after the pre-pass and BEFORE any
/// recognition. The first failing criterion wins, in plan order:
/// cjk fraction / dominant group without a rec variant / unsure fraction.
/// An empty pre-pass (no lines) never fires — the det stage has already
/// handled a near-image page.
pub fn chain_script(sm: &ScriptMetrics, rc: &RouteConfig) -> Option<&'static str> {
    if sm.n == 0 {
        return None;
    }
    if sm.cjk_frac() >= rc.chain.cjk_line_frac {
        return Some(TAG_CJK);
    }
    if sm.dominant.is_some() && !sm.dominant_has_variant {
        return Some(TAG_NOVARIANT);
    }
    // `None` threshold = criterion off (the default; see ChainConfig).
    if rc
        .chain
        .unsure_line_frac
        .is_some_and(|thr| sm.unsure_frac() >= thr)
    {
        return Some(TAG_UNSURE);
    }
    None
}

/// [rec] / [assemble] output-level gate — the single source of the two
/// output criteria (former G3 `hard_conf` / `min_chars`, new semantics:
/// OR, no char-count precondition on the conf item):
///
/// * mean line conf < `rec_hard_conf`        => `conf`
/// * output characters < `min_chars`         => `chars`
///
/// Callers:
/// * the chain's rec stage passes `chars = usize::MAX` (assemble has not
///   run yet, so only the conf criterion can fire);
/// * the chain's assemble stage passes the real markdown length (conf
///   already passed at the rec stage, so only chars can fire);
/// * the `use_indra = false` rollback path passes the PP-OCRv6 output's
///   line-text total and mean conf (the "G2" check).
pub fn output_gate(chars: usize, mean_conf: f32, rc: &RouteConfig) -> Option<&'static str> {
    if mean_conf < rc.chain.rec_hard_conf {
        Some(TAG_CONF)
    } else if chars < rc.chain.min_chars {
        Some(TAG_CHARS)
    } else {
        None
    }
}

/// Output-size signal for the output-level gate: the engine markdown's
/// length when the engine produces one (Indra), otherwise the line-text
/// total (PP-OCRv6).
pub fn output_chars(page: &OcrPage) -> usize {
    page.md
        .as_ref()
        .map_or_else(|| page.char_count(), |m| m.chars().count())
}

/// L2→L3 gate (plan §2.2, decision F): a weak or invalid PP-OCRv6 (L2)
/// result escalates the page to the L3 VLM.
///
/// * output characters < `l3.min_chars` => `chars` — an empty page (0)
///   hits here; this is the item that covers the mean_conf-of-empty-page
///   = 1.0 trap (plan §1.7), so it is checked first and never depends on
///   the conf item;
/// * mean line conf < `l3.hard`          => `conf` (non-empty pages).
pub fn l3_gate(chars: usize, mean_conf: f32, l3: &L3Config) -> Option<&'static str> {
    if chars < l3.min_chars {
        Some(TAG_CHARS)
    } else if mean_conf < l3.hard {
        Some(TAG_CONF)
    } else {
        None
    }
}

/// [cov] S0 stage (plan 0.3.2 T1'/C3, C3b): a usable-native-text page whose
/// L0 output covers less than `theta` of its complete text layer (the layer
/// holding at least `min_layer` non-whitespace chars) had its extraction
/// broken by the native flattening — reroute to L1. Returns the observed
/// cov when it fires (for logging), `None` otherwise.
///
/// The criterion is `<` (strict): a page covering exactly `theta` keeps the
/// native output. Calibrated on the firecrawl-pdfium per-page text layer
/// (PDFium FPDFText — the full per-page layer; C3b swapped the denominator
/// off pdf-inspector's under-covering positioned API): on OPB-48 the two
/// broken pages sit at cov 0.518 / 0.021 and every healthy page at ≥1.005;
/// on the 623-page fixture the minimum healthy cov is 0.839 and no page
/// falls in 0.6–0.839. θ=0.6 leaves 0.082 above the worst broken page and
/// 0.239 below the best healthy page.
pub fn coverage_should_reroute(
    l0_chars: usize,
    layer_chars: usize,
    theta: f32,
    min_layer: usize,
) -> Option<f32> {
    if layer_chars < min_layer {
        return None; // scan / near-empty: not a candidate
    }
    let cov = l0_chars as f32 / layer_chars as f32;
    if cov < theta {
        Some(cov)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rc() -> RouteConfig {
        RouteConfig::default()
    }

    // --- [det] ----------------------------------------------------------

    #[test]
    fn det_below_min_escalates() {
        assert_eq!(chain_det(0, &rc()), Some(TAG_DET));
        assert_eq!(chain_det(2, &rc()), Some(TAG_DET));
    }

    #[test]
    fn det_at_or_above_min_passes() {
        assert_eq!(chain_det(3, &rc()), None);
        assert_eq!(chain_det(40, &rc()), None);
    }

    // --- [script] -------------------------------------------------------

    fn sm(n: usize, cjk: usize, unsure: usize) -> ScriptMetrics {
        ScriptMetrics {
            n,
            cjk,
            unsure,
            dominant: Some(if cjk >= n - cjk {
                "cjk".to_string()
            } else {
                "base".to_string()
            }),
            dominant_has_variant: true,
        }
    }

    #[test]
    fn script_empty_never_fires() {
        assert_eq!(chain_script(&ScriptMetrics::default(), &rc()), None);
    }

    #[test]
    fn script_cjk_major_escalates() {
        // 30 of 40 lines cjk (0.75 >= 0.5) => cjk.
        assert_eq!(chain_script(&sm(40, 30, 0), &rc()), Some(TAG_CJK));
        // exactly half (0.5 >= 0.5) still fires.
        assert_eq!(chain_script(&sm(40, 20, 0), &rc()), Some(TAG_CJK));
        // English-major with a Chinese paragraph (10/40 = 0.25) stays L1.
        assert_eq!(chain_script(&sm(40, 10, 0), &rc()), None);
    }

    #[test]
    fn script_novariant_escalates() {
        let mut s = sm(40, 0, 0);
        s.dominant = Some("math".to_string());
        s.dominant_has_variant = false;
        assert_eq!(chain_script(&s, &rc()), Some(TAG_NOVARIANT));
    }

    #[test]
    fn script_unsure_major_escalates() {
        // Explicit threshold (the v0.3.0 behavior): 20 of 40 unsure
        // (0.5 >= 0.5) => unsure (cjk is low).
        let mut rc = rc();
        rc.chain.unsure_line_frac = Some(0.5);
        assert_eq!(chain_script(&sm(40, 0, 20), &rc), Some(TAG_UNSURE));
        // 19/40 stays L1.
        assert_eq!(chain_script(&sm(40, 0, 19), &rc), None);
    }

    #[test]
    fn script_unsure_off_by_default() {
        // The default (None) disables the unsure criterion: even a
        // majority-unsure page stays L1 (cjk/novariant still live).
        assert_eq!(chain_script(&sm(40, 0, 40), &rc()), None);
        // cjk still escalates with the default config.
        assert_eq!(chain_script(&sm(40, 30, 30), &rc()), Some(TAG_CJK));
    }

    #[test]
    fn script_cjk_wins_over_unsure() {
        // both fire: the plan order puts cjk first.
        let mut rc = rc();
        rc.chain.unsure_line_frac = Some(0.5);
        assert_eq!(chain_script(&sm(40, 30, 30), &rc), Some(TAG_CJK));
    }

    // --- [rec] / [assemble] output gate ---------------------------------

    #[test]
    fn gate_low_conf_escalates() {
        // mean conf 0.4 (< 0.5) escalates regardless of output size
        // (former G3 required chars < min_chars first; that precondition is
        // gone).
        assert_eq!(output_gate(5000, 0.4, &rc()), Some(TAG_CONF));
        // usize::MAX keeps only the conf criterion live (chain rec stage).
        assert_eq!(output_gate(usize::MAX, 0.4, &rc()), Some(TAG_CONF));
    }

    #[test]
    fn gate_low_conf_but_long_output_escalates() {
        // high-output low-conf page: the former G3 said Ok, the chain says
        // conf (plan §2.1: mean_conf < 0.5 with no char precondition).
        assert_eq!(output_gate(5000, 0.3, &rc()), Some(TAG_CONF));
    }

    #[test]
    fn gate_short_output_escalates() {
        // 20 chars (< 60), healthy conf => chars.
        assert_eq!(output_gate(20, 0.9, &rc()), Some(TAG_CHARS));
        assert_eq!(output_gate(0, 0.9, &rc()), Some(TAG_CHARS));
    }

    #[test]
    fn gate_conf_wins_over_chars() {
        // both fire: the gate checks conf first (the chain evaluates the
        // conf criterion at the earlier rec stage).
        assert_eq!(output_gate(20, 0.3, &rc()), Some(TAG_CONF));
    }

    #[test]
    fn gate_healthy_passes() {
        assert_eq!(output_gate(300, 0.9, &rc()), None);
        assert_eq!(output_gate(60, 0.5, &rc()), None);
        // boundaries: conf exactly 0.5 passes (criterion is `<`), chars
        // exactly 60 passes.
    }

    // --- L2→L3 gate (plan §2.2, decision F) ----------------------------

    fn l3() -> L3Config {
        L3Config::default()
    }

    #[test]
    fn l3_short_output_escalates() {
        // 20 chars (< 30), healthy conf => chars.
        assert_eq!(l3_gate(20, 0.9, &l3()), Some(TAG_CHARS));
        // an empty page (0 chars) hits here — the item that covers the
        // mean_conf-of-empty-page = 1.0 trap (plan §1.7).
        assert_eq!(l3_gate(0, 1.0, &l3()), Some(TAG_CHARS));
    }

    #[test]
    fn l3_low_conf_escalates() {
        // a non-empty page with mean conf below 0.5 => conf.
        assert_eq!(l3_gate(5000, 0.4, &l3()), Some(TAG_CONF));
        // boundary: conf exactly 0.5 passes (criterion is `<`).
        assert_eq!(l3_gate(5000, 0.5, &l3()), None);
    }

    #[test]
    fn l3_chars_wins_over_conf() {
        // both fire: the chars item is listed first in plan §2.2 and is
        // the empty-page cover.
        assert_eq!(l3_gate(20, 0.3, &l3()), Some(TAG_CHARS));
    }

    #[test]
    fn l3_healthy_passes() {
        assert_eq!(l3_gate(300, 0.9, &l3()), None);
        // boundary: exactly 30 chars passes.
        assert_eq!(l3_gate(30, 0.9, &l3()), None);
    }

    // --- [cov] S0 coverage gate (plan 0.3.2 T1'/C3) ---------------------

    #[test]
    fn cov_small_layer_never_fires() {
        // layer below min_layer is not a candidate, no matter how low cov.
        assert_eq!(coverage_should_reroute(0, 0, 0.5, 200), None);
        assert_eq!(coverage_should_reroute(10, 199, 0.5, 200), None);
    }

    #[test]
    fn cov_fires_at_min_layer() {
        // exactly min_layer with low coverage fires.
        assert_eq!(coverage_should_reroute(50, 200, 0.5, 200), Some(0.25));
    }

    #[test]
    fn cov_at_theta_passes() {
        // the criterion is `<`: covering exactly theta keeps the native
        // output (the healthy side of the calibration split).
        assert_eq!(coverage_should_reroute(1000, 2000, 0.5, 200), None);
        // just below fires.
        assert_eq!(coverage_should_reroute(999, 2000, 0.5, 200), Some(0.4995));
    }

    #[test]
    fn cov_above_theta_passes() {
        assert_eq!(coverage_should_reroute(1600, 2000, 0.5, 200), None);
    }

    #[test]
    fn cov_theta_is_configurable() {
        // theta = 0.7: a 0.5-coverage page now fires.
        assert_eq!(coverage_should_reroute(1000, 2000, 0.7, 200), Some(0.5));
        assert_eq!(coverage_should_reroute(1600, 2000, 0.7, 200), None);
    }

    #[test]
    fn cov_calibrated_values_fire() {
        // The two OPB-48 broken pages under the C3b firecrawl denominator
        // (plan §0.3): odot (1654/3190 ≈ 0.518) and the olmo table page
        // (94/4367 ≈ 0.021). Both fire at θ=0.6 (odot needs θ>0.518).
        let a = coverage_should_reroute(1654, 3190, 0.6, 200);
        assert!(a.is_some() && (a.unwrap() - 0.5185).abs() < 1e-3);
        let b = coverage_should_reroute(94, 4367, 0.6, 200);
        assert!(b.is_some() && (b.unwrap() - 0.0215).abs() < 1e-3);
        // A healthy page (cov ≥1.005 on OPB-48) stays native at θ=0.6.
        assert_eq!(coverage_should_reroute(3001, 2986, 0.6, 200), None);
    }

    // --- output_chars ---------------------------------------------------

    #[test]
    fn output_chars_prefers_markdown() {
        let p = OcrPage {
            lines: vec![],
            md: Some("abcd".to_string()),
            blocks: Vec::new(),
            escalate_l2: None,
        };
        assert_eq!(output_chars(&p), 4);
        let q = OcrPage {
            lines: vec![],
            md: None,
            blocks: Vec::new(),
            escalate_l2: None,
        };
        assert_eq!(output_chars(&q), 0);
    }
}
