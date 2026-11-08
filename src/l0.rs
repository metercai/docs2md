//! L0: native PDF → markdown extraction (pdf-inspector, no rendering, no
//! models, no OCR). Pages with an embedded text layer convert directly;
//! pages without one (scanned images) are flagged `needs_ocr` and fall to
//! the L1/L2 path.

use pdf_inspector::extract_pages_markdown_mem;

/// Per-page L0 outcome.
#[derive(Debug, Clone)]
pub struct L0Page {
    /// 0-based page index (as reported by pdf-inspector).
    pub page: u32,
    /// Markdown produced for this page (empty for image-only pages).
    pub markdown: String,
    /// True when the page has no usable text layer and needs rendering +
    /// OCR (L1 routing).
    pub needs_ocr: bool,
    /// Machine reason for `needs_ocr` (e.g. "no text"), for logging only.
    pub ocr_reason: Option<String>,
    /// pdf-inspector's free table signal (span geometry, no rendering, no
    /// models). Used to reroute table pages to L1 (whose order model emits
    /// `<table>` structure the native layer flattens away) without a layout
    /// model — the zero-model replacement for the former C1 structural
    /// reroute.
    pub has_table: bool,
}

/// Result of the L0 pass over a whole document.
#[derive(Debug, Clone)]
pub struct L0Result {
    pub pages: Vec<L0Page>,
}

/// Run L0 native extraction over the PDF bytes.
pub fn extract(bytes: &[u8]) -> Result<L0Result, String> {
    let doc = extract_pages_markdown_mem(bytes, None)
        .map_err(|e| format!("L0 extraction failed: {e}"))?;
    let tables = doc.pages_with_tables;
    let pages = doc
        .pages
        .into_iter()
        .map(|p| L0Page {
            page: p.page,
            markdown: p.markdown,
            needs_ocr: p.needs_ocr,
            ocr_reason: p.ocr_reason,
            // pdf-inspector reports these 1-indexed; `page` is 0-based.
            has_table: tables.contains(&(p.page + 1)),
        })
        .collect();
    Ok(L0Result { pages })
}
