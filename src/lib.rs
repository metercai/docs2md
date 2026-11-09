//! docs2md — three-tier PDF → Markdown.
//!
//!   L0  native text extraction (pdf-inspector) — no rendering, no models.
//!   L1  Indra (feature `indra`, default): indra-ocr 1.1.2 port,
//!       layout-aware markdown output.
//!   L2  PP-OCRv6-small (embedded, decision E): the local fallback engine,
//!       reached ONLY through the L1→L2 serial metric chain (plan §2.1) —
//!       never by a pre-emptive layout detector (decision B).
//!   L3  external VLM service (a llama-server wrapper): the end of the
//!       cascade (plan §2.2, decision F). Reached when L2 (PP-OCRv6) is
//!       weak/invalid, or when L1 itself is unavailable/failed.
//!
//! Per-page routing:
//!   * page has a usable text layer            → L0 (direct); a page whose
//!     text layer is only textual yet carries a table (pdf-inspector's
//!     free `has_table` signal) is rerouted to L1, whose order model emits
//!     `<table>` structure (decision C).
//!   * needs OCR → render once → L1 (Indra). The serial metric chain is
//!     evaluated as the Indra pipeline advances (after det / the script
//!     pre-pass / rec / assemble); a hit stops the pipeline and escalates
//!     the page to L2 (PP-OCRv6) on the same rendered image. A weak or
//!     invalid L2 result (chars < l3.min_chars or mean conf < l3.hard,
//!     plan §2.2) escalates to L3 (the VLM). The cascade always lands on
//!     output: L3 result, or the best local text with a note, or an
//!     explicit low-confidence marker — no page comes out empty (G).

pub mod blocks;
pub mod config;
pub mod l0;
pub mod l1;
pub mod l2;
pub mod markdown;
pub mod render;
pub mod route;

use config::Config;
use firecrawl_pdfium::PdfDocument;
use l1::{build_engine, L1Engine, OcrPage, PpOcrEngine};
use l2::{L2Client, PageImage};
use route::{l3_gate, output_gate};
use rusto::layout::LayoutDetector;
use rusto::types::LayoutConfig;
use std::collections::HashSet;
use std::path::PathBuf;

/// What produced a page's final markdown, plus any quality note.
#[derive(Debug, Clone)]
pub struct PageRecord {
    /// 1-based page number.
    pub page: u32,
    /// Which tier produced the output, e.g. "L0-native", "L1-indra",
    /// "L2-ppv6(chain:cjk)", "L2-ppv6(chain:conf)(l3-unavailable)",
    /// "L2-ppv6(chain:det)(low-conf)", "L3-vlm(chain:cjk)",
    /// "L1->L2(l1-failed)", "L1-ppv6(small)" (rollback).
    pub tier: String,
    /// Final visible character count.
    pub chars: usize,
    /// A quality note (e.g. the "needs a higher-tier OCR model" marker), or
    /// None when the page is clean.
    pub note: Option<String>,
    /// Structured layout blocks (label + normalized bbox + text). A new
    /// output surface: empty for L2-vlm pages and for engines that emit no
    /// structure; never affects the markdown bytes or tier routing.
    pub blocks: Vec<crate::l1::Block>,
}

/// Result of converting a whole document.
#[derive(Debug, Clone)]
pub struct DocResult {
    pub markdown: String,
    pub pages: Vec<PageRecord>,
}

/// The orchestrator. Holds the (lazy) L1 engine, the L2 PP-OCRv6 engine,
/// the L3 VLM client, and the (lazy) layout detector for the optional
/// `--blocks` L0 relabel (decision D). Construct once per run; call
/// `convert` on each document.
pub struct Engine {
    cfg: Config,
    /// The VLM client (plan §2.2's L3 end of the cascade). Reached when
    /// L2 (PP-OCRv6) is weak/invalid, or when L1 itself is unavailable
    /// or failed.
    l2: L2Client,
    /// The L1 primary (Indra when enabled, else PP-OCRv6).
    l1: Option<Box<dyn L1Engine>>,
    l1_failed: bool,
    /// The L2 PP-OCRv6 engine (lazily built on the first chain hit).
    l2_pp: Option<Box<dyn L1Engine>>,
    l2_pp_failed: bool,
    /// Why L1 is unavailable (for notes), empty when available/unused.
    l1_unavailable: String,
    /// The layout detector (decision D: the optional `--blocks` L0 relabel
    /// only — never a routing signal). Lazily loaded on the first L0 page
    /// when `blocks_requested` is set; `None` when the flag is off, the
    /// model is missing, or it failed to load (the relabel is then
    /// silently skipped, the same way it behaved when no model existed).
    layout: Option<LayoutDetector>,
    layout_failed: bool,
    /// Set by the CLI when `--blocks` is passed explicitly. Off by
    /// default: the default path never loads or runs the layout model
    /// (plan §3 cost invariant).
    blocks_requested: bool,
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
            eprintln!(
                "[l2] no base_url configured — complex pages fall back to L1 \
                      and are marked 'needs a higher-tier OCR model'"
            );
        }
        Ok(Self {
            cfg,
            l2,
            l1: None,
            l1_failed: false,
            l2_pp: None,
            l2_pp_failed: false,
            l1_unavailable: String::new(),
            layout: None,
            layout_failed: false,
            blocks_requested: false,
        })
    }

    /// Mark the optional L0 layout relabel as requested (the CLI sets this
    /// when `--blocks` is passed). Default off: without it the layout model
    /// is never loaded or run, even when configured.
    pub fn set_blocks_requested(&mut self, on: bool) {
        self.blocks_requested = on;
    }

    /// Ensure the layout detector is loaded (decision D: the `--blocks`
    /// relabel only). Loaded at most once, on first request; `None` when
    /// no model is configured, it is missing, or it failed to load — in
    /// every case the caller silently skips the relabel (the blocks stay
    /// at their E1 labels), the same way it behaved before the model
    /// existed. Never reached on the default path (`blocks_requested` off).
    fn ensure_layout(&mut self) -> Option<&mut LayoutDetector> {
        if self.layout.is_some() {
            return self.layout.as_mut();
        }
        if self.layout_failed {
            return None;
        }
        match &self.cfg.l1.layout_model {
            Some(p) if p.exists() => {
                let mut lc = LayoutConfig::default(p.clone());
                lc.target_size = self.cfg.l1.layout_target_size as i32;
                match LayoutDetector::new(lc) {
                    Ok(ld) => {
                        eprintln!("[blocks] layout model loaded: {}", p.display());
                        self.layout = Some(ld);
                        self.layout.as_mut()
                    }
                    Err(e) => {
                        eprintln!("[blocks] layout model failed to load: {e}; relabel skipped");
                        self.layout_failed = true;
                        None
                    }
                }
            }
            Some(p) => {
                eprintln!(
                    "[blocks] layout model not found: {}; relabel skipped",
                    p.display()
                );
                self.layout_failed = true;
                None
            }
            None => {
                eprintln!("[blocks] no layout_model configured; relabel skipped");
                self.layout_failed = true;
                None
            }
        }
    }

    /// D (optional `--blocks` relabel): the normalized layout regions that
    /// [`l0_blocks`] should relabel an L0 page with, or `None` when the
    /// relabel does not apply (flag off — default; model missing; page
    /// unrenderable). Renders the page once and runs the layout detector;
    /// a render or detect failure silently degrades to `None` (the blocks
    /// keep their E1 labels — a relabel failure is never a page failure).
    fn l0_relabel_regions(
        &mut self,
        bytes: &[u8],
        page0: u32,
    ) -> Option<Vec<(f64, f64, f64, f64, &'static str)>> {
        if !self.blocks_requested {
            return None;
        }
        let (png, _, _) = render::render_pdf_bytes(bytes, page0 as usize, self.cfg.l0.dpi).ok()?;
        let (pw, ph) = render::png_size(&png);
        if pw == 0 || ph == 0 {
            return None;
        }
        let img = image::load_from_memory(&png).ok()?;
        let mat = rusto::image_impl::Mat::new(img);
        let out = self.ensure_layout()?.detect(&mat).ok()?;
        Some(blocks::layout_regions_to_relabel(
            &out.regions,
            pw as f64,
            ph as f64,
        ))
    }

    /// Ensure the L1 primary engine (Indra when enabled, else PP-OCRv6) is
    /// built. Returns a human reason if it is unavailable (init failed), and
    /// `None` when ready.
    fn ensure_l1(&mut self) -> Option<&mut dyn L1Engine> {
        if self.l1.is_some() {
            let engine = self.l1.as_mut()?;
            return Some(engine.as_mut());
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

    /// Ensure the L2 PP-OCRv6 engine is built (decision E). Returns `None`
    /// when it is unavailable (init failed) or when the primary already IS
    /// the PP-OCRv6 engine (the `use_indra = false` rollback path: nothing
    /// to escalate to).
    fn ensure_l2_pp(&mut self) -> Option<&mut dyn L1Engine> {
        if !self.cfg.l1.use_indra {
            return None;
        }
        if self.l2_pp_failed {
            return None;
        }
        if self.l2_pp.is_none() {
            match PpOcrEngine::new(&self.cfg.l1.tier, self.cfg.l1.model_dir.as_ref()) {
                Ok(e) => {
                    self.l2_pp = Some(Box::new(e));
                }
                Err(msg) => {
                    eprintln!("[l2] PP-OCRv6 unavailable: {msg}");
                    self.l2_pp_failed = true;
                    return None;
                }
            }
        }
        Some(self.l2_pp.as_mut()?.as_mut())
    }

    /// Convert a PDF (bytes) to markdown via the three tiers. `pages` limits
    /// the pass to a set of 1-based page numbers (None = whole document).
    pub fn convert(&mut self, bytes: &[u8], pages: Option<&[u32]>) -> Result<DocResult, Error> {
        let t0 = std::time::Instant::now();
        let l0 = l0::extract(bytes).map_err(Error::Pdf)?;
        let page_count = l0.pages.len();
        eprintln!(
            "[l0] {} pages in {}ms",
            page_count,
            t0.elapsed().as_millis()
        );

        let mut doc: Option<PdfDocument> = None;
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
                // S0 candidate: a usable native text layer. Post-route
                // reroute (zero-model): pdf-inspector's free `has_table`
                // signal (span geometry, no rendering, no models) marks a
                // page whose text layer is only textual while the page
                // carries a table — L1's order model emits `<table>`
                // structure, which the native layer flattens away. The
                // former C1 structural reroute (render + PP-DocLayout +
                // G1 a/b) is gone: L0 is model-free, and the layout model
                // no longer participates in routing at all (decision C).
                // Gated on use_indra: with use_indra = false (v0.1.0) the
                // page keeps the native L0 output.
                if self.cfg.l1.use_indra && p.has_table {
                    let (png, _pbox, _rot) =
                        render::render_pdf_bytes(bytes, p.page as usize, self.cfg.l0.dpi)
                            .map_err(|e| Error::Pdf(format!("p{page1} render: {e}")))?;
                    eprintln!("[p{page1}] S0 reroute: table page — L1");
                    let (page_md, rec, note) = self.l1_dual(page1, &png);
                    if self.cfg.output.page_markers {
                        out.push_str(&format!("<!-- page:{page1} -->\n"));
                    }
                    if let Some(n) = &note {
                        out.push_str(n);
                    }
                    out.push_str(&page_md);
                    out.push('\n');
                    records.push(rec);
                } else {
                    out.push_str(&p.markdown);
                    out.push('\n');
                    // D: optional layout relabel of the L0 blocks — only
                    // when --blocks was passed explicitly and a layout
                    // model is configured (default path: None, no render,
                    // no model load).
                    let relabel = self.l0_relabel_regions(bytes, p.page);
                    // Open the PDFium document lazily — only the first
                    // L0-native page pays for it; l0_blocks reuses it.
                    if doc.is_none() {
                        let pdfium = firecrawl_pdfium::Pdfium::load()
                            .map_err(|e| Error::Pdf(format!("p{page1}: {e}")))?;
                        doc = Some(
                            pdfium
                                .load_document(bytes.to_vec(), None)
                                .map_err(|e| Error::Pdf(format!("p{page1}: {e}")))?,
                        );
                    }
                    records.push(PageRecord {
                        page: page1,
                        tier: "L0-native".into(),
                        chars: p.markdown.chars().count(),
                        note: None,
                        blocks: self.l0_blocks(bytes, p.page, doc.as_ref(), relabel.as_deref()),
                    });
                }
                continue;
            }

            // needs_ocr → render once (PDFium), then route.
            let (png, _pbox, _rot) =
                render::render_pdf_bytes(bytes, p.page as usize, self.cfg.l0.dpi)
                    .map_err(|e| Error::Pdf(format!("p{page1} render: {e}")))?;
            let (pw, ph) = render::png_size(&png);

            // needs_ocr → render once, then L1. The L1→L2 serial metric
            // chain (plan §2.1) runs inside the Indra pipeline; a hit
            // escalates the page to L2 (PP-OCRv6) on this same rendered
            // image — no layout detector anywhere in routing (decision B).
            let (page_md, rec, note) = if self.cfg.l1.use_indra {
                self.l1_dual(page1, &png)
            } else {
                self.l1_pp_only(page1, &png)
            };
            eprintln!(
                "[p{page1}] needs_ocr ({}) rendered {}x{} => {}",
                p.ocr_reason.as_deref().unwrap_or("?"),
                pw,
                ph,
                rec.tier
            );
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

    /// L1 with `use_indra = false`: the v0.1.0 PP-OCRv6-only flow. The
    /// output-level gate (former G2/G3, now `route::output_gate`) escalates
    /// a weak page to the VLM; the tier strings keep the `g2:<tag>` shape
    /// with the chain's `conf`/`chars` tags (plan §5.1 rollback path).
    fn l1_pp_only(&mut self, page1: u32, png: &[u8]) -> (String, PageRecord, Option<String>) {
        let rc = self.cfg.route.clone();
        match self.ensure_l1() {
            None => {
                // L1 unavailable → try L2, else a clear note.
                self.l2_page(
                    page1,
                    png,
                    "L1->L2(l1-unavailable)",
                    Some(self.l1_unavailable.clone()),
                )
            }
            Some(engine) => {
                let page = match engine.ocr_page(png, &rc) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("[p{page1}] L1 failed: {e}");
                        return self.l2_page(page1, png, "L1->L2(l1-failed)", Some(e));
                    }
                };
                // G2: the output-level gate on the L1 result — the chain's
                // two output criteria (conf OR chars, single source in
                // route.rs) with no pipeline intermediates to read.
                let md = markdown::ocr_lines_to_markdown(&page.lines);
                let gate = output_gate(page.char_count(), page.mean_conf(), &rc);
                let Some(tag) = gate else {
                    return (
                        md,
                        PageRecord {
                            page: page1,
                            tier: format!("L1-ppv6-{}", self.cfg.l1.tier),
                            chars: page.char_count(),
                            note: None,
                            blocks: page.blocks.clone(),
                        },
                        None,
                    );
                };
                eprintln!(
                    "[p{page1}] G2 {tag} (chars={}, conf={:.3}) — escalating to L2",
                    page.char_count(),
                    page.mean_conf()
                );
                // Escalate to L2 (VLM) if available; otherwise keep the L1
                // output and mark the page.
                if self.l2.available() {
                    let (md, rec, note) =
                        self.l2_page(page1, png, &format!("L1->L2(g2:{tag})"), None);
                    if rec.tier.contains("failed") {
                        // L2 was available but the call failed → fall back
                        // to the L1 output with a note.
                        (
                            md,
                            PageRecord {
                                page: page1,
                                tier: format!("L1(g2:{tag};L2-failed)"),
                                chars: page.char_count(),
                                note: Some(markdown::NEEDS_HIGHER_NOTE.to_string()),
                                blocks: page.blocks.clone(),
                            },
                            Some(markdown::NEEDS_HIGHER_NOTE.to_string()),
                        )
                    } else {
                        (md, rec, note)
                    }
                } else {
                    let note = Some(markdown::NEEDS_HIGHER_NOTE.to_string());
                    (
                        md,
                        PageRecord {
                            page: page1,
                            tier: format!("L1(g2:{tag})"),
                            chars: page.char_count(),
                            note: note.clone(),
                            blocks: page.blocks.clone(),
                        },
                        note,
                    )
                }
            }
        }
    }

    /// L1 with `use_indra = true` (plan §2.1): the serial metric chain runs
    /// inside the Indra pipeline; a hit escalates the page to L2
    /// (PP-OCRv6) on the same rendered image — no second render, no layout
    /// model (decisions B + E). A chain-passed page stands as L1.
    fn l1_dual(&mut self, page1: u32, png: &[u8]) -> (String, PageRecord, Option<String>) {
        // L1 unavailable / O_I = Err: the existing L1-unavailable path —
        // escalate to the VLM with the v0.1.0 tier strings.
        let rc = self.cfg.route.clone();
        let primary = match self.ensure_l1() {
            None => {
                return self.l2_page(
                    page1,
                    png,
                    "L1->L2(l1-unavailable)",
                    Some(self.l1_unavailable.clone()),
                );
            }
            Some(engine) => match engine.ocr_page(png, &rc) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[p{page1}] Indra failed: {e}");
                    return self.l2_page(page1, png, "L1->L2(l1-failed)", Some(e));
                }
            },
        };
        let Some(tag) = primary.escalate_l2 else {
            // Chain passed: the L1 output stands.
            let chars = primary.char_count();
            let md = primary
                .md
                .unwrap_or_else(|| markdown::ocr_lines_to_markdown(&primary.lines));
            return (
                md,
                PageRecord {
                    page: page1,
                    tier: "L1-indra".to_string(),
                    chars,
                    note: None,
                    blocks: primary.blocks.clone(),
                },
                None,
            );
        };
        // Chain hit: read the page with L2 (PP-OCRv6) on the same rendered
        // image. At most one L2 read per page.
        eprintln!("[p{page1}] chain:{tag} — escalating to L2 (PP-OCRv6)");
        let pp = match self.ensure_l2_pp() {
            Some(engine) => match engine.ocr_page(png, &rc) {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("[p{page1}] L2 (PP-OCRv6) failed: {e} — keeping L1 output");
                    None
                }
            },
            None => {
                eprintln!("[p{page1}] L2 (PP-OCRv6) unavailable — keeping L1 output");
                None
            }
        };
        // L2→L3 (plan §2.2, decision F): the PP-OCRv6 result stands unless
        // it is weak/invalid — output chars < l3.min_chars (an empty page
        // hits here; the mean_conf-of-empty-page = 1.0 trap, plan §1.7) or
        // mean conf < l3.hard.
        let l3_tag = pp
            .as_ref()
            .and_then(|p| l3_gate(p.char_count(), p.mean_conf(), &self.cfg.l3));
        let Some(l3_tag) = l3_tag else {
            let pp = pp.unwrap();
            let md = markdown::ocr_lines_to_markdown(&pp.lines);
            return (
                md,
                PageRecord {
                    page: page1,
                    tier: format!("L2-ppv6(chain:{tag})"),
                    chars: pp.char_count(),
                    note: None,
                    blocks: Vec::new(),
                },
                None,
            );
        };
        eprintln!(
            "[p{page1}] l3-gate:{l3_tag} (pp chars={}, conf={:.3}) — escalating to L3 (VLM)",
            pp.as_ref().map_or(0, |p| p.char_count()),
            pp.as_ref().map_or(1.0, |p| p.mean_conf())
        );
        // §2.2 hit: L3 (the VLM service) on the same rendered image.
        let vlm = if self.l2.available() {
            let res = self.l2.ocr_page(&PageImage { png, page1 });
            match res {
                r if r.ok => Some(r.markdown),
                r => {
                    eprintln!(
                        "[p{page1}] L3 call failed: {} — keeping local best-effort",
                        r.error.as_deref().unwrap_or("unknown")
                    );
                    None
                }
            }
        } else {
            eprintln!("[p{page1}] L3 (VLM) unavailable — keeping local best-effort");
            None
        };
        let Some(vlm) = vlm else {
            return self.l3_fallback(page1, tag, &pp, &primary);
        };
        (
            vlm.clone(),
            PageRecord {
                page: page1,
                tier: format!("L3-vlm(chain:{tag})"),
                chars: vlm.chars().count(),
                note: None,
                blocks: Vec::new(),
            },
            None,
        )
    }

    /// G end of the cascade (plan §2.2 / §6): the L3 VLM was unavailable or
    /// failed after a §2.2 hit — keep the best local text with a note, or
    /// the explicit low-confidence marker when nothing readable exists. No
    /// page comes out empty.
    fn l3_fallback(
        &self,
        page1: u32,
        chain_tag: &str,
        pp: &Option<OcrPage>,
        primary: &OcrPage,
    ) -> (String, PageRecord, Option<String>) {
        // L2 (PP-OCRv6) has weak-but-readable text: it stands, noted.
        let pp_text = pp
            .as_ref()
            .map(|p| markdown::ocr_lines_to_markdown(&p.lines))
            .filter(|m| !m.trim().is_empty());
        if let Some(pp_text) = pp_text {
            let chars = pp_text.chars().count();
            let note = Some(markdown::L3_FALLBACK_NOTE.to_string());
            return (
                pp_text,
                PageRecord {
                    page: page1,
                    tier: format!("L2-ppv6(chain:{chain_tag})(l3-unavailable)"),
                    chars,
                    note: note.clone(),
                    blocks: Vec::new(),
                },
                note,
            );
        }
        // PP-OCRv6 read nothing: keep the (possibly partial) L1 output.
        let keep_md = primary
            .md
            .as_deref()
            .filter(|m| !m.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| markdown::ocr_lines_to_markdown(&primary.lines));
        if !keep_md.trim().is_empty() {
            let chars = keep_md.chars().count();
            let note = Some(markdown::KEEP_L1_NOTE.to_string());
            return (
                keep_md,
                PageRecord {
                    page: page1,
                    tier: format!("L2-ppv6(chain:{chain_tag})(keep-l1)"),
                    chars,
                    note: note.clone(),
                    blocks: primary.blocks.clone(),
                },
                note,
            );
        }
        // Nothing anywhere: the explicit low-confidence marker.
        let note = Some(markdown::LOW_CONF_MARKER.to_string());
        (
            String::new(),
            PageRecord {
                page: page1,
                tier: format!("L2-ppv6(chain:{chain_tag})(low-conf)"),
                chars: 0,
                note: note.clone(),
                blocks: Vec::new(),
            },
            note,
        )
    }

    /// Send a page to L3 (the VLM); on failure (service down / call error)
    /// fall back to a clear "needs a higher-tier OCR model" marker with
    /// empty body (the G no-empty-page invariant is enforced by the caller
    /// or the explicit marker).
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
                    blocks: Vec::new(),
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
                    blocks: Vec::new(),
                },
                None,
            )
        } else {
            eprintln!(
                "[p{page1}] L2 call failed: {} — marking page",
                res.error.unwrap_or_default()
            );
            let note = Some(markdown::NEEDS_HIGHER_NOTE.to_string());
            (
                String::new(),
                PageRecord {
                    page: page1,
                    tier: format!("{ok_tier}(failed)"),
                    chars: 0,
                    note: note.clone(),
                    blocks: Vec::new(),
                },
                note,
            )
        }
    }

    /// Structured blocks for an L0-native page, from the PDF's embedded text
    /// layer (pdf-inspector positioned text). E1: every block is label
    /// `text`. The items are clustered into reading-order line blocks and
    /// normalized to the rendered page's `[0,1]` y-down space. Coordinate
    /// contract: pdf-inspector reports items relative to the visible page
    /// box (CropBox ∩ MediaBox) with the box's lower-left origin, y-up,
    /// `/Rotate` not applied; the box size is read from PDFium's
    /// `FPDF_GetPageBoundingBox` (the same box the render path renders), so
    /// the items normalize against box origin (0,0). For `/Rotate 180` the
    /// items are mirrored into the display frame (point reflection about
    /// the box centre) before the reading-order sort, so within-row text
    /// keeps the displayed left-to-right order; for `/Rotate 0` they stay
    /// as reported. For `/Rotate 90/270` the mixed raw/rotated-content
    /// space is not trusted, so those pages emit no blocks. A failure at
    /// any step yields an empty vec — blocks never affect the markdown or
    /// routing.
    ///
    /// `relabel` (decision D): normalized layout regions already mapped to
    /// the rendered page's `[0,1]` y-down space (`heading`/`table` only).
    /// Each region relabels the L0 blocks whose centres it contains. Wired
    /// up by the `--blocks` optional relabel (Task 5); always `None` today.
    fn l0_blocks(
        &self,
        bytes: &[u8],
        page0: u32,
        doc: Option<&PdfDocument>,
        relabel: Option<&[(f64, f64, f64, f64, &'static str)]>,
    ) -> Vec<crate::l1::Block> {
        // Reuse the caller's open document when it has one; otherwise open a
        // local one and keep it alive for the duration of this call.
        let owned_doc: Option<PdfDocument> = match doc {
            Some(_) => None,
            None => {
                let pdfium = match firecrawl_pdfium::Pdfium::load() {
                    Ok(p) => p,
                    Err(_) => return Vec::new(),
                };
                match pdfium.load_document(bytes, None) {
                    Ok(d) => Some(d),
                    Err(_) => return Vec::new(),
                }
            }
        };
        let d: &PdfDocument = match doc {
            Some(d) => d,
            None => owned_doc
                .as_ref()
                .expect("owned_doc is set whenever doc is None"),
        };
        let ((_, _, bw, bh), rotation) = match render::page_geometry(d, page0 as usize) {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        if !(rotation == 0 || rotation == 180) {
            return Vec::new();
        }
        // 1-based page filter: pdf-inspector numbers pages from 1.
        let page_set = HashSet::from([page0 + 1]);
        let items = match pdf_inspector::extractor::extract_text_with_positions_mem_pages(
            bytes,
            Some(&page_set),
        ) {
            Ok(items) => items,
            Err(_) => return Vec::new(),
        };
        let spans: Vec<(f64, f64, f64, f64, String)> = items
            .iter()
            .filter(|it| matches!(it.item_type, pdf_inspector::types::ItemType::Text))
            .map(|it| {
                let (mut x0, mut y0, mut x1, mut y1) = (
                    it.x as f64,
                    it.y as f64,
                    (it.x + it.width) as f64,
                    (it.y + it.height) as f64,
                );
                if rotation == 180 {
                    // Display frame: point reflection about the box centre,
                    // before the reading-order sort (both axes mirror).
                    let (mx0, mx1) = (bw as f64 - x1, bw as f64 - x0);
                    let (my0, my1) = (bh as f64 - y1, bh as f64 - y0);
                    (x0, x1, y0, y1) = (mx0, mx1, my0, my1);
                }
                (x0, y0, x1, y1, it.text.clone())
            })
            .collect();
        let blocks = blocks::spans_to_text_blocks(&spans, 0.0, 0.0, bw as f64, bh as f64, 0);
        match relabel {
            Some(regions) => blocks::relabel_l0_blocks(blocks, regions),
            None => blocks,
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

#[cfg(test)]
mod tests {
    use super::*;
    use l1::{OcrLine, OcrPage};
    use markdown::{KEEP_L1_NOTE, L3_FALLBACK_NOTE, LOW_CONF_MARKER};

    fn page(lines: &[(&str, f32)]) -> OcrPage {
        OcrPage {
            lines: lines
                .iter()
                .map(|(t, c)| OcrLine {
                    text: (*t).to_string(),
                    conf: *c,
                    top: 0.0,
                    left: 0.0,
                    width: 10.0,
                    height: 10.0,
                })
                .collect(),
            md: None,
            blocks: Vec::new(),
            escalate_l2: None,
        }
    }

    fn engine() -> Engine {
        Engine::new(Config::default()).expect("default engine (no L2, no L1 models loaded)")
    }

    // G (no empty page): every branch of the cascade end emits markdown or
    // a non-empty note — the page is never silently empty.

    #[test]
    fn g_l3_fallback_weak_pp_text_stands_with_note() {
        // PP-OCRv6 read weak-but-readable text (hit the §2.2 gate on
        // confidence): it stands, with the L3-fallback note.
        let pp = page(&[("some weak text", 0.3)]);
        let primary = page(&[]);
        let (md, rec, note) = engine().l3_fallback(7, "cjk", &Some(pp.clone()), &primary);
        assert_eq!(md, "some weak text\n");
        assert_eq!(rec.tier, "L2-ppv6(chain:cjk)(l3-unavailable)");
        assert!(rec.chars > 0);
        let note = note.expect("note present");
        assert_eq!(note, L3_FALLBACK_NOTE);
        assert!(!md.trim().is_empty());
    }

    #[test]
    fn g_l3_fallback_pp_empty_keeps_l1_with_note() {
        // PP-OCRv6 read nothing: keep the (partial) L1 output, noted.
        let primary = page(&[("partial l1 text", 0.9)]);
        let (md, rec, note) = engine().l3_fallback(8, "det", &None, &primary);
        assert_eq!(md, "partial l1 text\n");
        assert_eq!(rec.tier, "L2-ppv6(chain:det)(keep-l1)");
        let note = note.expect("note present");
        assert_eq!(note, KEEP_L1_NOTE);
        assert!(!md.trim().is_empty());
    }

    #[test]
    fn g_l3_fallback_everything_empty_emits_marker() {
        // Nothing anywhere: the explicit low-confidence marker (non-empty).
        let primary = page(&[]);
        let (md, rec, note) = engine().l3_fallback(9, "conf", &None, &primary);
        assert!(
            md.trim().is_empty(),
            "marker lives in the note, not the body"
        );
        assert_eq!(rec.tier, "L2-ppv6(chain:conf)(low-conf)");
        let note = note.expect("note present");
        assert_eq!(note, LOW_CONF_MARKER);
        assert!(!note.trim().is_empty(), "the page itself is never empty");
    }
}
