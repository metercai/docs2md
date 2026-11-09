//! Page rendering (PDFium via firecrawl-pdfium) + coordinate mapping.
//!
//! Rendered only for `needs_ocr` pages (L0-native pages skip rendering
//! entirely). PDFium is a runtime-loaded shared library: the discovery
//! chain is `$PDFIUM_LIB_PATH` → next to the executable → `./target/pdfium`
//! → the system loader (firecrawl-pdfium library.rs). The offline package
//! ships `libpdfium.so` next to the binary, so the package runs with no
//! network access and no host-font dependency (PDFium carries its own
//! base-14 font outlines).
//!
//! `page_box` (the render content box, `FPDF_GetPageBoundingBox`) and
//! `rotation_degrees` stay here: they feed the pixel→PDF coordinate mapping
//! and the L0 block normalization (both read the same PDFium-visible box).

use firecrawl_pdfium::{Pdfium, RenderConfig};
use image::codecs::png::PngEncoder;
use image::ImageEncoder;

/// Page render raster budget in pixels. A page whose render at the
/// configured dpi would exceed this renders at a lower effective dpi
/// (posters, plans). Matches indra's `MAX_PAGE_PIXELS`
/// (document.py:35): scale = min(k, sqrt(MAX_PAGE_PIXELS / (w*h))).
const MAX_PAGE_PIXELS: u64 = 100_000_000;

/// One rendered page: PNG bytes, the content box (x0, y0, w, h) in PDF
/// points, and the page's `/Rotate` in degrees.
type RenderedPage = (Vec<u8>, (f32, f32, f32, f32), i32);

/// A page's geometry without rasterizing: the content box (x0, y0, w, h) in
/// PDF points and the page's `/Rotate` in degrees.
type PageGeometry = ((f32, f32, f32, f32), i32);

/// Render one page of the raw PDF bytes to an RGBA PNG (the pipeline
/// converts to grayscale itself — see `preproc::Gray::from_png`).
///
/// `dpi` is the requested resolution; the effective scale is lowered to
/// fit `MAX_PAGE_PIXELS` (see [`effective_dpi`]). The content box is
/// CropBox ∩ MediaBox — identical to what the PDFium viewport renders, so
/// the pixel→PDF coordinate mapping lines up.
pub fn render_pdf_bytes(bytes: &[u8], page: usize, dpi: u32) -> Result<RenderedPage, String> {
    let pdfium = Pdfium::load().map_err(|e| format!("load pdfium: {e}"))?;
    let doc = pdfium
        .load_document(bytes, None)
        .map_err(|e| format!("load document: {e}"))?;
    // Form-field appearances only draw once the form-fill env is active —
    // call it before opening pages, matching pypdfium2's may_draw_forms=True.
    doc.enable_form_rendering()
        .map_err(|e| format!("enable forms: {e}"))?;
    let page = doc
        .page(page)
        .map_err(|e| format!("open page {page}: {e}"))?;
    let page_no = page.index();
    let (w, h) = (page.width(), page.height());
    let scale = effective_dpi(dpi as f32, w, h) / 72.0;
    let cfg = RenderConfig::new()
        .scale(scale)
        // RGB, like the pypdfium2 reference: the pipeline's to_luma8() (Rec.709)
        // is the single luma conversion. A Gray8 render would apply PDFium's own
        // Rec.601 luma and drift a few greys from the reference path.
        .pixel_format(firecrawl_pdfium::PixelFormat::Rgba8)
        .annotations(false)
        .form_fields(true);
    let rendered = page
        .render(&cfg)
        .map_err(|e| format!("render page {page_no}: {e}"))?;
    let (pw, ph) = (rendered.width(), rendered.height());
    let px = rendered.into_pixels();
    let mut png: Vec<u8> = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&px, pw, ph, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("encode png: {e}"))?;
    let pbox = page_box(&page).map_err(|e| format!("page box: {e}"))?;
    let rotation = rotation_degrees(page.rotation());
    Ok((png, pbox, rotation))
}

/// The page's geometry without rasterizing: the content box (CropBox ∩
/// MediaBox, `FPDF_GetPageBoundingBox`) and the page's `/Rotate`, read from a
/// PDFium page handle. Feeds the L0 block normalization, which maps native
/// text-layer items into the rendered page's `[0,1]` y-down space without a
/// PNG render.
pub fn page_geometry(
    doc: &firecrawl_pdfium::PdfDocument,
    page: usize,
) -> Result<PageGeometry, String> {
    let page = doc
        .page(page)
        .map_err(|e| format!("open page {page}: {e}"))?;
    let pbox = page_box(&page).map_err(|e| format!("page box: {e}"))?;
    let rotation = rotation_degrees(page.rotation());
    Ok((pbox, rotation))
}

/// The content box from PDFium's page handle: `FPDF_GetPageBoundingBox`
/// (CropBox ∩ MediaBox), post-`/Rotate`.
fn page_box(page: &firecrawl_pdfium::PdfPage<'_>) -> Result<(f32, f32, f32, f32), String> {
    let r = page.bounding_box().map_err(|e| e.to_string())?;
    Ok((
        r.left as f32,
        r.bottom as f32,
        r.width() as f32,
        r.height() as f32,
    ))
}

/// `/Rotate` (clockwise 90° increments) as degrees.
fn rotation_degrees(rot: firecrawl_pdfium::Rotation) -> i32 {
    match rot {
        firecrawl_pdfium::Rotation::None => 0,
        firecrawl_pdfium::Rotation::Clockwise90 => 90,
        firecrawl_pdfium::Rotation::Rotate180 => 180,
        firecrawl_pdfium::Rotation::Clockwise270 => 270,
    }
}

/// The effective dpi after fitting the render into `MAX_PAGE_PIXELS`:
/// `min(dpi, 72 * sqrt(budget / (w*h)))` (same formula as indra's
/// `page_scale`).
fn effective_dpi(dpi: f32, w: f32, h: f32) -> f32 {
    let area = (w * h).max(1.0);
    let fit = 72.0 * (((MAX_PAGE_PIXELS as f32) / area).sqrt());
    dpi.min(fit)
}

/// Minimal PNG IHDR size reader (avoids pulling the image crate for a header).
/// PNG signature (8) + chunk length (4) + "IHDR" (4) ⇒ width@16, height@20.
pub fn png_size(png: &[u8]) -> (u32, u32) {
    if png.len() < 24 || &png[12..16] != b"IHDR" {
        return (0, 0);
    }
    let w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    (w, h)
}

#[cfg(test)]
mod tests {
    use super::{effective_dpi, png_size, render_pdf_bytes};
    use std::io::Read;

    /// Local fixture corpus (not shipped in the repo, CI skips cleanly).
    const FIXTURE: &str = "/home/ubuntu/pdf2md.fixtures/Q-Pharm-Pty-Ltd-Invoice-Report.pdf";

    /// PDFium is a runtime-loaded shared library and is strict about PDF
    /// structure (hand-built minimal PDFs fail validation); these tests
    /// need both the .so (PDFIUM_LIB_PATH or next to the binary) and a
    /// real fixture.
    fn setup() -> Option<std::fs::File> {
        if !firecrawl_pdfium::Pdfium::load().is_ok() {
            eprintln!("SKIP: libpdfium not loadable");
            return None;
        }
        match std::fs::File::open(FIXTURE) {
            Ok(f) => Some(f),
            Err(_) => {
                eprintln!("SKIP: fixture not present at {FIXTURE}");
                None
            }
        }
    }

    #[test]
    fn real_page_renders_at_the_requested_dpi() {
        let Some(mut f) = setup() else { return };
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).expect("read fixture");
        let (png, pbox, rot) = render_pdf_bytes(&bytes, 0, 200).expect("render");
        let (w, h) = png_size(&png);
        assert!(w > 0 && h > 0, "empty render");
        // Internal consistency: the content box at the requested scale must
        // match the rendered bitmap (the invariant coordinate mapping
        // relies on). Not an absolute-size check — fixtures vary.
        let scale = 200.0 / 72.0;
        assert!(
            ((pbox.2 * scale) as f64 - w as f64).abs() <= 2.0,
            "box width {pbox:?} vs {w}px"
        );
        assert!(
            ((pbox.3 * scale) as f64 - h as f64).abs() <= 2.0,
            "box height {pbox:?} vs {h}px"
        );
        // No budget engagement for a normal page: full 200 dpi, well under
        // the 100 MP ceiling.
        assert!(
            (w as u64) * (h as u64) < 100_000_000,
            "unexpectedly capped: {w}x{h}"
        );
        assert_eq!(rot % 360, 0, "rot: {rot}");
    }

    #[test]
    fn huge_page_renders_under_the_cap() {
        // Pure-math budget: a 4000x4000 pt page at 200 dpi would be ~11111
        // px (123 MP) without the cap; effective_dpi lowers it to 180 dpi,
        // landing at exactly 10000 px (100 MP) instead of failing. No PDF
        // needed: the capping is in effective_dpi.
        let d = effective_dpi(200.0, 4000.0, 4000.0);
        let side = (4000.0 * d / 72.0) as u64;
        assert!(d < 200.0, "budget did not engage: {d}");
        assert!(
            side * side <= 100_000_000 + 4000,
            "budget exceeded: {side}px"
        );
        assert!((9000..11111).contains(&side), "unexpected scale: {side}px");
    }

    #[test]
    fn effective_dpi_fits_the_budget() {
        // A4 at 200 dpi is well under the budget — unchanged.
        assert!((effective_dpi(200.0, 595.2, 841.9) - 200.0_f32).abs() < 1e-3);
        // 4000x4000 pt at 200 dpi = 123 MP > 100 MP — scaled down to ~182.6.
        let d = effective_dpi(200.0, 4000.0, 4000.0);
        let expect = 72.0_f32 * ((100_000_000.0_f32 / (4000.0_f32 * 4000.0_f32)).sqrt());
        assert!((d - expect).abs() < 1e-2, "d={d} expect={expect}");
    }
}
