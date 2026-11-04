//! docs2md — three-tier PDF → Markdown.
//!
//!   L0  native text extraction (pdf-inspector) — no rendering, no models.
//!   L1  embedded pure-Rust PP-OCRv6 (ONNX on rten) — simple scanned text.
//!   L2  external VLM service (a llama-server wrapper) — complex pages.
//!
//! Per-page routing:
//!   * page has a usable text layer            → L0 (direct)
//!   * needs OCR → render → G1 (layout model,
//!     external) decides L1 vs L2 → if L1, G2
//!     (output-level) may escalate to L2.
//!   * L2 unavailable (no config / health
//!     probe failed / call failed) → the L1
//!     output is kept and the page is marked
//!     "needs a higher-tier OCR model".

pub mod config;
pub mod l0;
pub mod l1;
pub mod l2;
pub mod markdown;
pub mod render;
pub mod route;

use std::path::PathBuf;
use config::Config;
use l1::{build_engine, L1Engine};
use l2::{L2Client, PageImage};
use route::{g1, g2, Tier, G2Verdict};
use rusto::{LayoutConfig, LayoutDetector};

/// What produced a page's final markdown, plus any quality note.
#[derive(Debug, Clone)]
pub struct PageRecord {
    /// 1-based page number.
    pub page: u32,
    /// Which tier produced the output, e.g. "L0-native", "L1-ppv6-small",
    /// "L2-vlm", "L1->L2(g2:hard)".
    pub tier: String,
    /// Final visible character count.
    pub chars: usize,
    /// A quality note (e.g. the "needs a higher-tier OCR model" marker), or
    /// None when the page is clean.
    pub note: Option<String>,
}

/// Result of converting a whole document.
#[derive(Debug, Clone)]
pub struct DocResult {
    pub markdown: String,
    pub pages: Vec<PageRecord>,
}

/// The orchestrator. Holds the (lazy) L1 engine, layout detector, and L2
/// client. Construct once per run; call `convert` on each document.
pub struct Engine {
    cfg: Config,
    l2: L2Client,
    l1: Option<Box<dyn L1Engine>>,
    l1_failed: bool,
    layout: Option<LayoutDetector>,
    /// Why L1 is unavailable (for notes), empty when available/unused.
    l1_unavailable: String,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config: {0}")]
    Config(#[from] config::ConfigError),
    #[error("pdf: {0}")]
    Pdf(String),
    #[error("l1: {0}")]
    L1(String),
}

impl Engine {
    pub fn new(cfg: Config) -> Result<Self, Error> {
        let l2 = L2Client::new(&cfg.l2);
        if !l2.available() && !cfg.l2.base_url.trim().is_empty() {
            eprintln!("[l2] service unavailable: {}", l2.unavailable_reason());
        } else if cfg.l2.base_url.trim().is_empty() {
            eprintln!("[l2] no base_url configured — complex pages fall back to L1 \
                      and are marked 'needs a higher-tier OCR model'");
        }
        Ok(Self {
            cfg,
            l2,
            l1: None,
            l1_failed: false,
            layout: None,
            l1_unavailable: String::new(),
        })
    }

    /// Ensure the L1 engine is built. Returns a human reason if it is
    /// unavailable (init failed), and `None` when ready.
    fn ensure_l1(&mut self) -> Option<&mut dyn L1Engine> {
        if self.l1.is_some() {
            return Some(self.l1.as_mut().unwrap().as_mut());
        }
        if self.l1_failed {
            return None;
        }
        match build_engine(&self.cfg.l1) {
            Ok(e) => {
                self.l1 = Some(e);
                Some(self.l1.as_mut().unwrap().as_mut())
            }
            Err(msg) => {
                eprintln!("[l1] engine unavailable: {msg}");
                self.l1_failed = true;
                self.l1_unavailable = msg;
                None
            }
        }
    }

    /// Ensure the layout detector is built (G1). `None` when no external
    /// layout model is configured or it failed to load — in that case G1 is
    /// skipped and OCR pages default to L1 (G2 can still escalate to L2).
    fn ensure_layout(&mut self) -> Option<&mut LayoutDetector> {
        if self.layout.is_none() {
            match &self.cfg.l1.layout_model {
                Some(p) if p.exists() => {
                    let mut lc = LayoutConfig::default(p.clone());
                    lc.target_size = self.cfg.l1.layout_target_size as i32;
                    match LayoutDetector::new(lc) {
                        Ok(ld) => {
                            eprintln!("[route] layout model loaded: {}", p.display());
                            self.layout = Some(ld);
                        }
                        Err(e) => eprintln!("[route] layout model failed to load: {e}; \
                                             G1 skipped (OCR pages default to L1)"),
                    }
                }
                Some(p) => eprintln!("[route] layout model not found: {}; G1 skipped", p.display()),
                None => eprintln!("[route] no [l1] layout_model configured; G1 skipped \
                                   (OCR pages default to L1)"),
            }
        }
        self.layout.as_mut()
    }

    /// Convert a PDF (bytes) to markdown via the three tiers. `pages` limits
    /// the pass to a set of 1-based page numbers (None = whole document).
    pub fn convert(&mut self, bytes: &[u8], pages: Option<&[u32]>) -> Result<DocResult, Error> {
        let t0 = std::time::Instant::now();
        let l0 = l0::extract(bytes).map_err(|e| Error::Pdf(e))?;
        let page_count = l0.pages.len();
        eprintln!(
            "[l0] {} pages in {}ms",
            page_count,
            t0.elapsed().as_millis()
        );

        let mut doc: Option<pdf_oxide::document::PdfDocument> = None;
        let mut out = String::new();
        let mut records: Vec<PageRecord> = Vec::new();

        for p in &l0.pages {
            let page1 = p.page + 1;
            if let Some(wp) = pages {
                if !wp.contains(&page1) {
                    continue;
                }
            }
            if !p.needs_ocr {
                out.push_str(&p.markdown);
                out.push('\n');
                records.push(PageRecord {
                    page: page1,
                    tier: "L0-native".into(),
                    chars: p.markdown.chars().count(),
                    note: None,
                });
                continue;
            }

            // needs_ocr → render once, then route.
            if doc.is_none() {
                doc = Some(
                    pdf_oxide::document::PdfDocument::from_bytes(bytes.to_vec())
                        .map_err(|e| Error::Pdf(format!("p{page1}: {e}")))?,
                );
            }
            let d = doc.as_ref().unwrap();
            let (png, pbox, rot) =
                render::render_pdf_page(d, p.page as usize, self.cfg.l0.dpi)
                    .map_err(|e| Error::Pdf(format!("p{page1} render: {e}")))?;
            let (pw, ph) = render::png_size(&png);
            let page_area = pw as f64 * ph as f64;
            let _ = (pbox, rot); // coordinate mapping available for span work

            // G1 (layout) decides L1 vs L2; without a layout model, default L1.
            // Clone the route config first (the closure must not hold an
            // immutable borrow of self while ensure_layout mutably borrows it).
            let route = self.cfg.route.clone();
            let tier = match self.ensure_layout().and_then(|ld| g1(Some(ld), &png, page_area, &route)) {
                Some(t) => t,
                None => {
                    eprintln!("[p{page1}] no layout route — defaulting to L1");
                    Tier::L1
                }
            };
            eprintln!(
                "[p{page1}] needs_ocr ({}), rendered {}x{}, G1 => {}",
                p.ocr_reason.as_deref().unwrap_or("?"),
                pw,
                ph,
                match tier { Tier::L1 => "L1", Tier::L2 => "L2" }
            );

            let (page_md, rec, note) = self.process_tier(page1, &png, tier);
            if self.cfg.output.page_markers {
                out.push_str(&format!("<!-- page:{page1} -->\n"));
            }
            if let Some(n) = &note {
                out.push_str(n);
            }
            out.push_str(&page_md);
            out.push('\n');
            records.push(rec);
        }

        Ok(DocResult {
            markdown: out,
            pages: records,
        })
    }

    /// Produce one page's markdown given its G1 verdict, applying G2 for L1
    /// and the L2-unavailable fallback note where required.
    fn process_tier(
        &mut self,
        page1: u32,
        png: &[u8],
        tier: Tier,
    ) -> (String, PageRecord, Option<String>) {
        match tier {
            Tier::L2 => self.l2_page(page1, png, "L2-vlm", None),
            Tier::L1 => {
                let l1_md = match self.ensure_l1() {
                    None => {
                        // L1 unavailable → try L2, else a clear note.
                        return self.l2_page(
                            page1,
                            png,
                            "L1->L2(l1-unavailable)",
                            Some(self.l1_unavailable.clone()),
                        );
                    }
                    Some(engine) => {
                        let page = match engine.ocr_page(png) {
                            Ok(p) => p,
                            Err(e) => {
                                eprintln!("[p{page1}] L1 failed: {e}");
                                return self.l2_page(
                                    page1,
                                    png,
                                    "L1->L2(l1-failed)",
                                    Some(e),
                                );
                            }
                        };
                        // G2: output-level gate on the L1 result.
                        let verdict = g2(page.char_count(), page.mean_conf(), &self.cfg.route);
                        if verdict != G2Verdict::Ok {
                            let tag = match verdict {
                                G2Verdict::Hard => "hard",
                                G2Verdict::Medium => "medium",
                                _ => unreachable!(),
                            };
                            eprintln!(
                                "[p{page1}] G2 {} (chars={}, conf={:.3}) — escalating to L2",
                                tag,
                                page.char_count(),
                                page.mean_conf()
                            );
                            // Escalate to L2 if available; otherwise keep the
                            // L1 output and mark the page.
                            if self.l2.available() {
                                let (md, rec, note) =
                                    self.l2_page(page1, png, &format!("L1->L2(g2:{tag})"), None);
                                if rec.tier.contains("unavailable") {
                                    // L2 was available but the call failed →
                                    // fall back to the L1 output with a note.
                                    let md = markdown::ocr_lines_to_markdown(&page.lines);
                                    (
                                        md,
                                        PageRecord {
                                            page: page1,
                                            tier: format!("L1(g2:{tag};L2-failed)"),
                                            chars: page.char_count(),
                                            note: Some(markdown::NEEDS_HIGHER_NOTE.to_string()),
                                        },
                                        Some(markdown::NEEDS_HIGHER_NOTE.to_string()),
                                    )
                                } else {
                                    (md, rec, note)
                                }
                            } else {
                                let md = markdown::ocr_lines_to_markdown(&page.lines);
                                let note = Some(markdown::NEEDS_HIGHER_NOTE.to_string());
                                (
                                    md,
                                    PageRecord {
                                        page: page1,
                                        tier: format!("L1(g2:{tag})"),
                                        chars: page.char_count(),
                                        note: note.clone(),
                                    },
                                    note,
                                )
                            }
                        } else {
                            let md = markdown::ocr_lines_to_markdown(&page.lines);
                            (
                                md,
                                PageRecord {
                                    page: page1,
                                    tier: format!("L1-ppv6-{}", self.cfg.l1.tier),
                                    chars: page.char_count(),
                                    note: None,
                                },
                                None,
                            )
                        }
                    }
                };
                l1_md
            }
        }
    }

    /// Send a page to L2; on failure (service down / call error) fall back to
    /// a clear "needs a higher-tier OCR model" marker with empty body.
    fn l2_page(
        &mut self,
        page1: u32,
        png: &[u8],
        ok_tier: &str,
        _why: Option<String>,
    ) -> (String, PageRecord, Option<String>) {
        if !self.l2.available() {
            let note = Some(markdown::NEEDS_HIGHER_NOTE.to_string());
            let reason = self.l2.unavailable_reason().to_string();
            eprintln!("[p{page1}] L2 unavailable ({reason}) — marking page");
            return (
                String::new(),
                PageRecord {
                    page: page1,
                    tier: ok_tier.to_string(),
                    chars: 0,
                    note: note.clone(),
                },
                note,
            );
        }
        let img = PageImage { png, page1 };
        let res = self.l2.ocr_page(&img);
        if res.ok {
            (
                res.markdown.clone(),
                PageRecord {
                    page: page1,
                    tier: ok_tier.to_string(),
                    chars: res.markdown.chars().count(),
                    note: None,
                },
                None,
            )
        } else {
            eprintln!("[p{page1}] L2 call failed: {} — marking page", res.error.unwrap_or_default());
            let note = Some(markdown::NEEDS_HIGHER_NOTE.to_string());
            (
                String::new(),
                PageRecord {
                    page: page1,
                    tier: format!("{ok_tier}(failed)"),
                    chars: 0,
                    note: note.clone(),
                },
                note,
            )
        }
    }
}

/// Resolve the config path: explicit arg > $DOCS2MD_CONFIG > ./docs2md.toml.
pub fn resolve_config_path(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(p) = explicit {
        return p.clone();
    }
    if let Ok(env) = std::env::var("DOCS2MD_CONFIG") {
        return PathBuf::from(env);
    }
    PathBuf::from("docs2md.toml")
}
