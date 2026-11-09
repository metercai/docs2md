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
    if sm.unsure_frac() >= rc.chain.unsure_line_frac {
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
        // 20 of 40 unsure (0.5 >= 0.5) => unsure (cjk is low).
        assert_eq!(chain_script(&sm(40, 0, 20), &rc()), Some(TAG_UNSURE));
        // 19/40 stays L1.
        assert_eq!(chain_script(&sm(40, 0, 19), &rc()), None);
    }

    #[test]
    fn script_cjk_wins_over_unsure() {
        // both fire: the plan order puts cjk first.
        assert_eq!(chain_script(&sm(40, 30, 30), &rc()), Some(TAG_CJK));
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
