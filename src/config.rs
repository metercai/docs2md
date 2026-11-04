//! Configuration for docs2md (serde, all fields defaulted, unknown fields
//! rejected so a typo in the config file is a loud error, not a silent
//! mis-route).
//!
//! The config is the single source that locates the external pieces:
//!   * L1 layout model (external ONNX, not embedded)
//!   * L2 VLM service (a llama-server wrapper, e.g. `docs2md-l2d`)
//! Everything else (L0 native extraction, L1 embedded PP-OCRv6-small OCR)
//! is self-contained in the binary.

use std::path::PathBuf;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub l0: L0Config,
    pub l1: L1Config,
    pub l2: L2Config,
    pub route: RouteConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L0Config {
    /// Render DPI for needs-ocr pages (shared by L1 routing and L2 send).
    pub dpi: u32,
}
impl Default for L0Config {
    fn default() -> Self {
        Self { dpi: 150 }
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
    /// embedded). When absent or unreadable, G1 layout routing is skipped
    /// and every OCR page defaults to L1 (G2 output-level can still
    /// escalate to L2).
    pub layout_model: Option<PathBuf>,
    /// Layout detector input size (pixels on the long edge).
    pub layout_target_size: u32,
    /// When true, run the Indra engine (feature `indra`) instead of
    /// PP-OCRv6. Reserved slot — not ported to Rust in v1; enabling this
    /// returns a clear "not ported" error.
    pub use_indra: bool,
}
impl Default for L1Config {
    fn default() -> Self {
        Self {
            tier: "small".into(),
            model_dir: None,
            layout_model: None,
            layout_target_size: 800,
            use_indra: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct L2Config {
    /// Base URL of the L2 service (an OpenAI-compatible llama-server,
    /// typically launched by `docs2md-l2d`). When empty, L2 is disabled:
    /// complex pages fall back to L1 and the output is marked as
    /// "needs a higher-tier OCR model".
    pub base_url: String,
    /// Model name passed in the chat-completion body (llama-server often
    /// serves a single model; leave empty to let it default).
    pub model: Option<String>,
    /// Prompt prefix sent before the page image.
    pub prompt: String,
    pub max_tokens: u32,
    pub timeout_secs: u64,
    /// Probe `{base_url}/health` once at startup; if it fails, L2 is
    /// treated as unavailable (L1 fallback) instead of failing every page.
    pub health_check: bool,
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
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouteConfig {
    /// G2 output-level fallback thresholds (calibrated on the 623-page
    /// fixture corpus; see the indra/PP-OCRv6 fit plan). These are the
    /// "operational parameters" and are therefore configurable, not
    /// hardcoded.
    pub g2: G2Thresholds,
    /// G1 compact-complex-region area fraction (bbox covers < this fraction
    /// of the page and conf > compact_conf ⇒ L2).
    pub compact_frac: f32,
    pub compact_conf: f32,
    /// G1 dense-text-blob thresholds (a TEXT region covering >= dense_frac
    /// of the page with conf >= dense_conf ⇒ L2).
    pub dense_frac: f32,
    pub dense_conf: f32,
}
impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            g2: G2Thresholds::default(),
            compact_frac: 0.5,
            compact_conf: 0.4,
            dense_frac: 0.85,
            dense_conf: 0.3,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct G2Thresholds {
    /// Minimum markdown characters for a page to be considered "has output".
    pub min_chars: usize,
    /// mean line conf below this (with low char count) ⇒ hard fail.
    pub hard_conf: f32,
    /// mean line conf below this (with low char count) ⇒ medium doubt.
    pub mid_conf: f32,
}
impl Default for G2Thresholds {
    fn default() -> Self {
        Self {
            min_chars: 60,
            hard_conf: 0.5,
            mid_conf: 0.75,
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
        let raw = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Read(path.to_path_buf(), e))?;
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
