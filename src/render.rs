//! Page rendering + coordinate mapping (pdf_oxide, pure-Rust tiny-skia).
//!
//! Rendered only for `needs_ocr` pages (L0-native pages skip rendering
//! entirely). The pixel→PDF user-space mapping mirrors pdf_oxide's
//! `page_render_box` / `page_base_transform` so OCR boxes (pixel, y-down)
//! can be mapped back to PDF points (y-up) for provenance and any future
//! span-level merging.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};

/// Render one page to PNG. Returns (png_bytes, render_box, rotation).
/// `render_box` is CropBox ∩ MediaBox — the visible page box, identical to
/// pdf_oxide's `page_render_box` (so the coordinate mapping matches).
/// One rendered page: PNG bytes, the content box (x0, y0, x1, y1) in PDF
/// points, and the page number.
type RenderedPage = (Vec<u8>, (f32, f32, f32, f32), i32);

pub fn render_pdf_page(doc: &PdfDocument, page: usize, dpi: u32) -> Result<RenderedPage, String> {
    let info = doc.get_page_info(page).map_err(|e| e.to_string())?;
    let pbox = render_box(&info.media_box, info.crop_box.as_ref());
    let mut opts = RenderOptions::default();
    opts.dpi = dpi;
    opts.format = ImageFormat::Png;
    opts.render_annotations = false;
    let img = render_page(doc, page, &opts).map_err(|e| e.to_string())?;
    Ok((img.data, pbox, info.rotation))
}

/// The visible page box: CropBox ∩ MediaBox.
fn render_box(
    media: &pdf_oxide::geometry::Rect,
    crop: Option<&pdf_oxide::geometry::Rect>,
) -> (f32, f32, f32, f32) {
    let (mx, my, mw, mh) = (media.x, media.y, media.width, media.height);
    let (cx0, cy0, cx1, cy1) = match crop {
        Some(c) => (
            c.x.max(mx),
            c.y.max(my),
            (c.x + c.width).min(mx + mw),
            (c.y + c.height).min(my + mh),
        ),
        None => (mx, my, mx + mw, my + mh),
    };
    if cx1 <= cx0 || cy1 <= cy0 {
        (mx, my, mw, mh)
    } else {
        (cx0, cy0, cx1 - cx0, cy1 - cy0)
    }
}

/// pdf_oxide's `page_base_transform` as (a,b,c,d,tx,ty):
/// device_x = a*px + b*py + tx; device_y = c*px + d*py + ty.
pub fn base_matrix(
    pbox: (f32, f32, f32, f32),
    rotation: i32,
    scale: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    let (x0, y0, w, h) = pbox;
    match rotation.rem_euclid(360) {
        90 => (0.0, scale, scale, 0.0, -x0, -y0),
        180 => (-scale, 0.0, 0.0, scale, x0 - w * scale, y0),
        270 => (0.0, -scale, -scale, 0.0, x0 + h * scale, y0 + w * scale),
        _ => (scale, 0.0, 0.0, -scale, x0, y0 + h * scale),
    }
}

/// Invert the base matrix: device (pixel) → PDF user space.
pub fn to_pdf(device_x: f32, device_y: f32, m: (f32, f32, f32, f32, f32, f32)) -> (f32, f32) {
    let (a, b, c, d, tx, ty) = m;
    let inv = 1.0 / (a * d - b * c);
    let u = device_x - tx;
    let v = device_y - ty;
    ((d * u - b * v) * inv, (-c * u + a * v) * inv)
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
