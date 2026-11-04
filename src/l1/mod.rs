//! L1: embedded pure-Rust OCR behind an `L1Engine` trait. Two engines:
//! PP-OCRv6-small (vendored rusto, `crates/ocr`) and the Indra pipeline
//! (feature `indra`, port of indra-ocr 1.1.2 under `src/l1/indra/`).
//!
//! The `small` tier's three files are `include_bytes!`-embedded (the "one
//! binary" build). rusto's engine API loads models **by path**, so the
//! embedded bytes are materialized to temp files once at init. `medium`/
//! `tiny` are not embedded; selecting them requires an external `model_dir`.

use std::path::PathBuf;
use rusto::{InitializeConfig, OcrRunOptions, RustO, TextResult};

use crate::config::L1Config;

/// One recognized text line (pixel frame, y-down) with its confidence.
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    /// Per-line recognition confidence (0..1). Feeds the G2 output-level
    /// gate (mean line conf).
    pub conf: f32,
    /// Bounding box: top, left, width, height (pixels).
    pub top: f32,
    pub left: f32,
    pub width: f32,
    pub height: f32,
}

/// A page's OCR result (reading order is the engine's; the orchestrator
/// re-sorts by line top for stable output).
#[derive(Debug, Clone)]
pub struct OcrPage {
    pub lines: Vec<OcrLine>,
}

impl OcrPage {
    /// Total visible characters (G2 `min_chars` signal).
    pub fn char_count(&self) -> usize {
        self.lines.iter().map(|l| l.text.chars().count()).sum()
    }
    /// Mean per-line confidence (G2 conf signal). 1.0 for an empty page.
    pub fn mean_conf(&self) -> f32 {
        if self.lines.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.lines.iter().map(|l| l.conf).sum();
        sum / self.lines.len() as f32
    }
}

/// The L1 engine abstraction. PP-OCRv6 is the v1 implementation; Indra is a
/// reserved slot (feature `indra`) that will implement this trait once its
/// ONNX pipeline is ported to Rust.
pub trait L1Engine: Send {
    /// Name for logs / provenance.
    fn name(&self) -> &str;
    /// Recognize one rendered page (PNG bytes, pixel y-down).
    fn ocr_page(&mut self, png: &[u8]) -> Result<OcrPage, String>;
}

/// PP-OCRv6 engine (embedded `small` by default).
pub struct PpOcrEngine {
    ocr: RustO,
    tier: String,
}

impl L1Engine for PpOcrEngine {
    fn name(&self) -> &str {
        &self.tier
    }

    fn ocr_page(&mut self, png: &[u8]) -> Result<OcrPage, String> {
        use rusto::{DetectTextResult, ImageSource};
        let det = self
            .ocr
            .detect_text(&ImageSource::Bytes(png.to_vec()), &OcrRunOptions::default())
            .map_err(|e| e.to_string())?;
        let DetectTextResult::Structured(results) = det else {
            return Err("PP-OCRv6 returned a non-structured result".into());
        };
        let lines = results
            .iter()
            .map(|r: &TextResult| OcrLine {
                text: r.text.clone(),
                conf: r.score,
                top: r.frame.top,
                left: r.frame.left,
                width: r.frame.width,
                height: r.frame.height,
            })
            .collect();
        Ok(OcrPage { lines })
    }
}

impl PpOcrEngine {
    /// Build the engine for the configured tier. `small` uses the embedded
    /// model; any other tier requires `model_dir` pointing at an external
    /// `PPOCR_v6_{tier}_rten/` directory (not embedded in this build).
    pub fn new(tier: &str, model_dir: Option<&PathBuf>) -> Result<Self, String> {
        let (det, rec, dict) = if tier == "small" {
            // Materialize the embedded model to temp files (rusto loads by
            // path). One set per process is enough; the engine keeps them
            // for its lifetime.
            let dir = std::env::temp_dir().join(format!(
                "docs2md-ppocrv6-small-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let det_p = dir.join("det.onnx");
            let rec_p = dir.join("rec.onnx");
            let dict_p = dir.join("dict.txt");
            std::fs::write(&det_p, EMBEDDED_DET_SMALL)
                .map_err(|e| format!("write embedded det.onnx: {e}"))?;
            std::fs::write(&rec_p, EMBEDDED_REC_SMALL)
                .map_err(|e| format!("write embedded rec.onnx: {e}"))?;
            std::fs::write(&dict_p, EMBEDDED_DICT_SMALL)
                .map_err(|e| format!("write embedded dict.txt: {e}"))?;
            (det_p, rec_p, dict_p)
        } else {
            let base = model_dir
                .ok_or_else(|| {
                    format!(
                        "PP-OCRv6 tier '{tier}' is not embedded in this build; \
                         set [l1] model_dir to a directory containing \
                         PPOCR_v6_{tier}_rten/{{det,rec}}.onnx"
                    )
                })?
                .join(format!("PPOCR_v6_{tier}_rten"));
            (
                base.join("det.onnx"),
                base.join("rec.onnx"),
                base.join("dict.txt"),
            )
        };

        let cfg = InitializeConfig::ppv6(det, rec, dict);
        let ocr = RustO::initialize(cfg).map_err(|e| {
            format!("PP-OCRv6 '{tier}' init failed: {e}")
        })?;
        Ok(Self { ocr, tier: tier.to_string() })
    }
}

// Embedded PP-OCRv6-small (rten) model. Total ≈ 31 MB, committed to the repo
// and baked into the binary so the release is a single self-contained file.
const EMBEDDED_DET_SMALL: &[u8] = include_bytes!("../../assets/models/small/det.onnx");
const EMBEDDED_REC_SMALL: &[u8] = include_bytes!("../../assets/models/small/rec.onnx");
const EMBEDDED_DICT_SMALL: &[u8] = include_bytes!("../../assets/models/small/dict.txt");

/// The Indra engine (feature `indra`): port of the indra-ocr 1.1.2 pipeline,
/// embedded under assets/models/indra/. P0-P2: loading and smoke inference;
/// the full page pipeline (det -> rec -> order -> layout -> markdown) lands
/// in P3-P6. Until then ocr_page returns a clear error so routing degrades
/// to the configured fallback instead of silently producing nothing.
#[cfg(feature = "indra")]
pub mod indra;

#[cfg(feature = "indra")]
pub struct IndraEngine {
    /// Loaded models + LM tables; consumed by the page pipeline (P1+).
    #[allow(dead_code)]
    engine: indra::IndraEngine,
}

#[cfg(feature = "indra")]
impl IndraEngine {
    pub fn new() -> Result<Self, String> {
        let engine = indra::IndraEngine::init()?;
        Ok(Self { engine })
    }
}

#[cfg(feature = "indra")]
impl L1Engine for IndraEngine {
    fn name(&self) -> &str {
        "indra"
    }

    fn ocr_page(&mut self, _png: &[u8]) -> Result<OcrPage, String> {
        Err("indra engine page pipeline is in progress (port P3-P6); \
             use PP-OCRv6 (l1.use_indra = false)".into())
    }
}

/// Build the configured L1 engine (Indra when enabled, else PP-OCRv6).
pub fn build_engine(cfg: &L1Config) -> Result<Box<dyn L1Engine>, String> {
    if cfg.use_indra {
        #[cfg(feature = "indra")]
        {
            return Ok(Box::new(IndraEngine::new()?));
        }
        #[cfg(not(feature = "indra"))]
        {
            return Err(
                "l1.use_indra = true but this build was compiled without the \
                 `indra` feature"
                    .into(),
            );
        }
    }
    Ok(Box::new(PpOcrEngine::new(&cfg.tier, cfg.model_dir.as_ref())?))
}
