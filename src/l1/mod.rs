//! L1: embedded pure-Rust OCR behind an `L1Engine` trait. Two engines:
//! PP-OCRv6-small (vendored rusto, `crates/ocr`) and the Indra pipeline
//! (feature `indra`, port of indra-ocr 1.1.2 under `src/l1/indra/`).
//!
//! The `small` tier's three files are `include_bytes!`-embedded (the "one
//! binary" build). rusto's engine API loads models **by path**, so the
//! embedded bytes are materialized to temp files once at init. `medium`/
//! `tiny` are not embedded; selecting them requires an external `model_dir`.

use rusto::{InitializeConfig, OcrRunOptions, RustO, TextResult};
use std::path::PathBuf;

use crate::config::L1Config;

/// One recognized text line (pixel frame, y-down) with its confidence.
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    /// Per-line recognition confidence (0..1). Feeds the chain's rec-stage
    /// conf criterion, the output-level gate, and the L2→L3 criteria
    /// (mean line conf).
    pub conf: f32,
    /// Bounding box: top, left, width, height (pixels).
    pub top: f32,
    pub left: f32,
    pub width: f32,
    pub height: f32,
}

/// One structured layout block: a labelled region with its text. `label` is
/// one of `crate::blocks::LABELS`; `bbox` is `[x0, y0, x1, y1]` normalized to
/// `[0, 1]` relative to the page (y-down, top-left origin) — the exact
/// coordinate space the layout judge consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub label: &'static str,
    pub bbox: [f64; 4],
    pub text: String,
}

/// A page's OCR result (reading order is the engine's; the orchestrator
/// re-sorts by line top for stable output).
#[derive(Debug, Clone)]
pub struct OcrPage {
    pub lines: Vec<OcrLine>,
    /// Engine markdown, when the engine produces it (Indra: the layout-aware
    /// markdown, table post-filter already applied; PP-OCRv6: `None` — the
    /// orchestrator builds markdown from the lines).
    pub md: Option<String>,
    /// Structured blocks in reading order (`[]` for engines that produce no
    /// layout structure — PP-OCRv6 today). A new output surface: it does not
    /// affect markdown bytes or routing.
    pub blocks: Vec<Block>,
    /// The serial metric chain hit during this page's pipeline run
    /// (plan §2.1): `None` = the chain passed, the L1 output stands;
    /// `Some(tag)` = the chain stopped at stage `tag`
    /// ("det" | "cjk" | "novariant" | "unsure" | "conf" | "chars") and the
    /// orchestrator escalates the page to L2 (PP-OCRv6). The `lines`/`md`
    /// carried with it are the (possibly partial) L1 output kept as the
    /// low-confidence fallback (the G no-empty-page guarantee).
    pub escalate_l2: Option<&'static str>,
}

impl OcrPage {
    /// Tag the page with the chain stage that stopped it (plan §2.1). The
    /// orchestrator reads this to escalate the page to L2 (PP-OCRv6).
    pub fn with_escalate(mut self, tag: Option<&'static str>) -> Self {
        self.escalate_l2 = tag;
        self
    }

    /// Total visible characters (the chain's chars signal).
    pub fn char_count(&self) -> usize {
        self.lines.iter().map(|l| l.text.chars().count()).sum()
    }
    /// Mean per-line confidence (the chain rec-stage / output-gate / L3
    /// conf signal). 1.0 for an empty page.
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
    /// Recognize one rendered page (PNG bytes, pixel y-down). `rc` carries
    /// the routing thresholds — the Indra engine evaluates the L1→L2 serial
    /// metric chain (plan §2.1) against it; PP-OCRv6 ignores it.
    fn ocr_page(&mut self, png: &[u8], rc: &crate::config::RouteConfig) -> Result<OcrPage, String>;
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

    fn ocr_page(
        &mut self,
        png: &[u8],
        _rc: &crate::config::RouteConfig,
    ) -> Result<OcrPage, String> {
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
        Ok(OcrPage {
            lines,
            md: None,
            blocks: Vec::new(), // PP-OCRv6 emits no layout structure
            escalate_l2: None,
        })
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
            let dir =
                std::env::temp_dir().join(format!("docs2md-ppocrv6-small-{}", std::process::id()));
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
        let ocr =
            RustO::initialize(cfg).map_err(|e| format!("PP-OCRv6 '{tier}' init failed: {e}"))?;
        Ok(Self {
            ocr,
            tier: tier.to_string(),
        })
    }
}

// Embedded PP-OCRv6-small (rten) model. Total ≈ 31 MB, committed to the repo
// and baked into the binary so the release is a single self-contained file.
const EMBEDDED_DET_SMALL: &[u8] = include_bytes!("../../assets/models/small/det.onnx");
const EMBEDDED_REC_SMALL: &[u8] = include_bytes!("../../assets/models/small/rec.onnx");
const EMBEDDED_DICT_SMALL: &[u8] = include_bytes!("../../assets/models/small/dict.txt");

/// The Indra engine (feature `indra`): port of the indra-ocr 1.1.2 pipeline,
/// embedded under assets/models/indra/. Page pipeline (P1-P6): preview ->
/// skew -> orientation -> detect -> recognize -> reading order -> markdown,
/// with the table post-filter applied in-engine (the engine's markdown is
/// the post-filtered one).
#[cfg(feature = "indra")]
pub mod indra;

#[cfg(feature = "indra")]
pub struct IndraEngine {
    /// Loaded models + LM tables; drives the page pipeline.
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

    fn ocr_page(&mut self, png: &[u8], rc: &crate::config::RouteConfig) -> Result<OcrPage, String> {
        let gray = indra::preproc::Gray::from_png(png)?;
        self.engine.ocr_page(&gray, rc)
    }
}

/// Build the configured L1 primary engine (Indra when enabled, else
/// PP-OCRv6). A `use_indra = true` build without the `indra` feature degrades
/// to PP-OCRv6 with a warning (the rollback build stays a working v0.1.0).
pub fn build_engine(cfg: &L1Config) -> Result<Box<dyn L1Engine>, String> {
    if cfg.use_indra {
        #[cfg(feature = "indra")]
        {
            return Ok(Box::new(IndraEngine::new()?));
        }
        #[cfg(not(feature = "indra"))]
        {
            eprintln!(
                "[l1] l1.use_indra = true but this build was compiled without the \
                 `indra` feature — using PP-OCRv6"
            );
            return Ok(Box::new(PpOcrEngine::new(
                &cfg.tier,
                cfg.model_dir.as_ref(),
            )?));
        }
    }
    Ok(Box::new(PpOcrEngine::new(
        &cfg.tier,
        cfg.model_dir.as_ref(),
    )?))
}
