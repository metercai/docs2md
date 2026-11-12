//! Configuration for docs2md (serde, all fields defaulted, unknown fields
//! rejected so a typo in the config file is a loud error, not a silent
//! mis-route).
//!
//! The config is the single source that locates the external pieces:
//!
//! * the layout model (external ONNX, not embedded — the optional
//!   `--blocks` relabeling only; it takes part in no routing decision)
//! * the VLM service (an OpenAI-compatible llama-server, e.g. `docs2md-l2d`
//!   — the end of the cascade)
//!
//! Everything else (L0 native extraction, L1 Indra, L2 embedded
//! PP-OCRv6-small) is self-contained in the binary.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub l0: L0Config,
    pub l1: L1Config,
    pub l2: L2Config,
    pub l3: L3Config,
    pub route: RouteConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L0Config {
    /// Render DPI for needs-ocr pages (shared by the L2 send and the
    /// optional `--blocks` layout relabel).
    pub dpi: u32,
    /// B1 heading heuristic (L0 pages only): a line is labelled
    /// `heading` when its median span font size is at least this many
    /// times the page's median span font size AND at least
    /// `heading_bold_frac` of its spans are bold. Zero-model: it reads the
    /// PDF text layer's own font attributes. `f64::INFINITY` disables the
    /// rule (every block stays `text`, the v0.3.3 behaviour).
    pub heading_size_ratio: f64,
    /// Fraction of a line's bold spans required for the heading rule.
    pub heading_bold_frac: f64,
    /// C1b render-side cap: the longest rendered side (pixels) for
    /// needs-ocr pages. 0 disables the cap (the v0.3.3 behaviour). The
    /// effective dpi is the strictest of the dpi, the 100 MP pixel budget
    /// and this cap.
    pub max_render_side: u32,
}
impl Default for L0Config {
    fn default() -> Self {
        Self {
            dpi: 200,
            heading_size_ratio: 1.2,
            heading_bold_frac: 0.5,
            max_render_side: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L1Config {
    /// PP-OCRv6 tier. `small` is embedded in the binary (the "one binary"
    /// build). `medium`/`tiny` are not embedded; selecting one requires an
    /// external `model_dir` holding `PPOCR_v6_{tier}_rten/{det,rec}.onnx`.
    pub tier: String,
    /// External model directory, used when `tier != "small"` (the embedded
    /// build only carries small). Ignored for `small`.
    pub model_dir: Option<PathBuf>,
    /// External PP-DocLayout ONNX path (the ~130MB layout model is not
    /// embedded). Used ONLY by the optional `--blocks` L0 relabel (decision
    /// D) — it never participates in tier routing. When absent or
    /// unreadable, the relabel is skipped.
    pub layout_model: Option<PathBuf>,
    /// Layout detector input size (pixels on the long edge).
    pub layout_target_size: u32,
    /// When true, run the Indra engine (feature `indra`) as the primary L1
    /// engine, with PP-OCRv6 as the L2 fallback engine (reached through the
    /// serial metric chain). When false, run the PP-OCRv6-only v0.1.0 flow
    /// (the rollback path).
    pub use_indra: bool,
}
impl Default for L1Config {
    fn default() -> Self {
        Self {
            tier: "small".into(),
            model_dir: None,
            layout_model: None,
            layout_target_size: 800,
            use_indra: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L2Config {
    /// Base URL of the VLM service (an OpenAI-compatible llama-server,
    /// typically launched by `docs2md-l2d`) — the L3 end of the cascade.
    /// When empty, the VLM leg is off: a weak or invalid L2 (PP-OCRv6)
    /// result keeps the local best-effort text with a note (plan G —
    /// no empty page).
    pub base_url: String,
    /// Model name passed in the chat-completion body (llama-server often
    /// serves a single model; leave empty to let it default).
    pub model: Option<String>,
    /// Prompt prefix sent before the page image.
    pub prompt: String,
    pub max_tokens: u32,
    pub timeout_secs: u64,
    /// Probe `{base_url}/health` at startup; if it fails, L2 is treated as
    /// unavailable (L1 fallback) instead of failing every page.
    pub health_check: bool,
    /// Max seconds to keep polling `/health` before giving up. llama-server
    /// answers 503 `{"error":{"message":"Loading model"}}` while the GGUF is
    /// still loading, so a single probe right after service start is too
    /// early; we retry within this budget instead.
    #[serde(default = "default_health_wait")]
    pub health_wait_secs: u64,
}

fn default_health_wait() -> u64 {
    30
}
impl Default for L2Config {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: None,
            prompt: "OCR:".into(),
            max_tokens: 4096,
            timeout_secs: 120,
            health_check: true,
            health_wait_secs: default_health_wait(),
        }
    }
}

/// L2→L3 criteria (plan §2.2, decision F): a weak or invalid PP-OCRv6
/// (L2) result escalates the page to the L3 VLM. Operational parameters,
/// therefore configurable (Task 7 calibrates them on the L2 pages).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L3Config {
    /// PP-OCRv6 output characters below this => L3. An empty page (0)
    /// hits here, which also covers the mean_conf-of-empty-page = 1.0
    /// trap (plan §1.7).
    pub min_chars: usize,
    /// PP-OCRv6 mean line conf below this => L3 (non-empty pages).
    pub hard: f32,
}
impl Default for L3Config {
    fn default() -> Self {
        Self {
            min_chars: 30,
            hard: 0.5,
        }
    }
}

/// L1→L2 serial metric chain thresholds (plan §2.1, calibrated on the
/// 623-page fixture corpus; operational parameters, therefore configurable
/// and not hardcoded). The chain is evaluated as the indra pipeline
/// advances — after det, after the script pre-pass, after rec, after
/// assemble — and the first hit stops the pipeline and escalates the page
/// to L2 (PP-OCRv6).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChainConfig {
    /// [det] fewer line boxes than this => L2 (a near-image page; saves the
    /// whole recognition pass).
    pub det_min_lines: usize,
    /// [script] cjk line fraction >= this => L2 (a Chinese-major page goes
    /// to the engine that reads Chinese better: PP-OCRv6).
    pub cjk_line_frac: f32,
    /// C2: a lower cjk line-fraction threshold for direct L2 (PP-OCRv6)
    /// escalation. `None` (the default) is off — the existing
    /// `cjk_line_frac` is the only cjk criterion. When set, a page whose
    /// cjk line fraction reaches this lower bound is escalated to PP-OCRv6
    /// before recognition even though it is not cjk-major: indra's base
    /// recognizer reads the CJK lines on such mixed pages poorly, while
    /// PP-OCRv6 reads them well (ODB zh census: the 11 non-escalated
    /// Chinese pages sit at cjk_frac 0.22-0.44 with F1 0.78, vs the
    /// escalated ones at 0.96). English pages stay untouched (their
    /// cjk_frac tops out at 0.125, below the 0.15 default suggestion).
    pub cjk_direct_line_frac: Option<f32>,
    /// [script] unsure (double-read) line fraction >= this => L2 (the
    /// recognizer itself is uncertain on half the lines). `None` (the
    /// default) turns the criterion off: on the handwriting A/B (39 pages,
    /// 2026-11-10) the unsure-triggered L2 leg was quality-neutral to
    /// negative for text content and cost 2x the time; the other criteria
    /// (cjk/novariant/det/conf/chars) are unaffected and stay live. Set a
    /// number (e.g. 0.5) to re-enable the v0.3.0 behavior.
    pub unsure_line_frac: Option<f32>,
    /// [rec] mean line conf below this => L2 (unreliable reading; the former
    /// G3 `hard_conf`, now without the char-count precondition).
    pub rec_hard_conf: f32,
    /// [assemble] output characters below this => L2 (almost nothing read;
    /// the former G3 `min_chars`).
    pub min_chars: usize,
}
impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            det_min_lines: 3,
            cjk_line_frac: 0.5,
            cjk_direct_line_frac: None,
            unsure_line_frac: None,
            rec_hard_conf: 0.5,
            min_chars: 60,
        }
    }
}

/// RouteConfig's `structure_gate` default: OFF. The free `pages_with_columns`
/// signal mis-fires on form layouts (short scattered items — checkbox labels,
/// field names — trigger pdf-inspector's unguarded absolute-valley / XY-cut
/// paths, which lack the prose guard the relative-valley path has). Until the
/// upstream fix (U2) lands, the default keeps v0.3.0 behavior; `true` re-enables
/// the columns reroute.
fn default_false() -> bool {
    false
}

/// RouteConfig's `coverage_gate` default: ON (C4, after the C3b
/// acceptance run on OPB-48 + the 623-page fixture).
fn default_coverage_gate() -> bool {
    true
}

/// RouteConfig's `parallel_threads` default: 4 (page-level parallelism).
/// A 48-page document runs 34.6 s at 4 workers vs 72.9 s serial (2.1x);
/// single-page inputs are unaffected (one worker). 1 = serial (v0.3.0).
fn default_threads() -> usize {
    4
}

/// RouteConfig's `cov_theta` default: 0.6 (C3b, calibrated on the
/// firecrawl-pdfium per-page text layer: the two broken OPB-48 pages sit at
/// cov 0.518 / 0.021, every healthy page at ≥1.005 there and ≥0.839 on the
/// 623-page fixture — θ=0.6 leaves margin on both sides).
fn default_cov_theta() -> f32 {
    0.6
}

/// RouteConfig's `cov_min_layer` default: 200 non-whitespace text-layer
/// chars — below that a page is a scan / near-empty page, not a coverage
/// candidate (it is `needs_ocr` anyway).
fn default_cov_min_layer() -> usize {
    200
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouteConfig {
    pub chain: ChainConfig,
    /// Reroute multi-column native-text pages (pdf-inspector's free
    /// `pages_with_columns` signal) to L1, whose order model keeps column
    /// reading order the native layer flattens away. Default off (see
    /// `default_false`): the signal mis-fires on form pages. `true` =
    /// re-enable the columns reroute.
    #[serde(default = "default_false")]
    pub structure_gate: bool,
    /// Number of page-level worker threads for the L1/L2 pass. `4` (the
    /// default) splits the pages across worker threads, each with its own L1
    /// engine and PDFium document (render is stateless, so this is safe);
    /// output is reassembled in page order, so the markdown is unaffected.
    /// `1` is the original serial loop (v0.3.0 behavior). Both the serde
    /// and `Default` impls must agree — see the unit test below.
    #[serde(default = "default_threads")]
    pub parallel_threads: usize,
    /// Feed the page's PDF text layer into the L1 engine (plan 0.3.2 T3/T4):
    /// covered lines take their exact text-layer characters instead of an OCR
    /// read. Default off (C1 lands the interface with zero behavior change;
    /// C2 turns the fusion on after the byte-parity gate).
    #[serde(default = "default_false")]
    pub layer_fusion: bool,
    /// T1' coverage gate (plan 0.3.2 C3): reroute a usable-native-text page
    /// to L1 when its L0 output covers less than `cov_theta` of the text
    /// layer (`cov_min_layer` chars minimum) — the extraction is broken,
    /// multi-column / table pages the native layer flattens. Default on
    /// (C4, after the C3b acceptance run).
    #[serde(default = "default_coverage_gate")]
    pub coverage_gate: bool,
    /// cov threshold for `coverage_gate` (C3b, calibrated on the
    /// firecrawl-pdfium per-page text layer; the two broken OPB-48 pages sit
    /// at cov 0.518 / 0.021, the lowest healthy page at 0.839 on the
    /// 623-page fixture). NOTE: cov depends on pdf-inspector's markdown
    /// extraction (numerator) and PDFium's text (denominator) — re-verify
    /// the threshold when either crate is upgraded.
    #[serde(default = "default_cov_theta")]
    pub cov_theta: f32,
    /// Minimum text-layer non-whitespace chars for `coverage_gate` to
    /// consider a page (excludes scans / near-empty pages).
    #[serde(default = "default_cov_min_layer")]
    pub cov_min_layer: usize,
}
impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            chain: ChainConfig::default(),
            structure_gate: false,
            parallel_threads: 4,
            layer_fusion: false,
            coverage_gate: true,
            cov_theta: 0.6,
            cov_min_layer: 200,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Emit an HTML comment marker before each page that needed OCR.
    pub page_markers: bool,
    /// Write a provenance/quality summary to stderr at the end.
    pub summary: bool,
}
impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            page_markers: true,
            summary: true,
        }
    }
}

impl Config {
    /// Load from a TOML file; a missing file yields the all-default config.
    pub fn load(path: &std::path::Path) -> Result<Self, ConfigError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Read(path.to_path_buf(), e))?;
        let cfg: Config =
            toml::from_str(&raw).map_err(|e| ConfigError::Parse(path.to_path_buf(), e))?;
        Ok(cfg)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read config {0}: {1}")]
    Read(std::path::PathBuf, #[source] std::io::Error),
    #[error("cannot parse config {0}: {1}")]
    Parse(std::path::PathBuf, #[source] toml::de::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    // The hand-written `Default for RouteConfig` and the serde per-field
    // defaults must agree: `Config::load` on a missing file goes through
    // the former, parsing a toml that omits a field through the latter.
    // (C4 pinned coverage_gate + cov_theta; 0.3.3 P0 extends the pin to
    // every field — a missing per-field `#[serde(default = ...)]` or a
    // drift in either impl now fails the build, e.g. the parallel_threads
    // 1-vs-4 divergence.)
    #[test]
    fn route_defaults_agree_serde_and_default() {
        let d = RouteConfig::default();
        let s: RouteConfig = toml::from_str("").unwrap();
        assert_eq!(d, s, "RouteConfig: Default impl must equal serde defaults");
        assert!(d.coverage_gate, "coverage_gate must default to on (C4)");
        assert!(
            (d.cov_theta - 0.6).abs() < f32::EPSILON,
            "cov_theta 0.6 (C3b)"
        );
        assert_eq!(d.parallel_threads, 4, "parallel_threads 4 (0.3.3 P0)");
    }
}
