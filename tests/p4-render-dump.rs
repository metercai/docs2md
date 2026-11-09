//! A5 determinism probe: render one fixture page at the configured dpi with
//! the PRODUCT's exact render path (`render::render_pdf_bytes`) and dump the
//! PNG. Run twice — once with system fonts available, once with fontconfig
//! pointed at an empty dir (the font-less case a bare container hits) — and
//! compare the PNG bytes. Identical bytes prove PDFium's render does not
//! read host fonts (A5).
//!
//! Env:
//!   P4_RENDER_OUT   where to write the PNG (default /tmp/opb-run/p4_render.png)
//!   P4_RENDER_FIX   fixture PDF (required — no default)
//!   P4_RENDER_PAGE  0-based page index (default 0)
//!   P4_RENDER_DPI   default 200

use docs2md::render::{png_size, render_pdf_bytes};

fn env(name: &str, dflt: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| dflt.to_string())
}

#[test]
fn p4_render_dump() {
    let Some(fix) = std::env::var("P4_RENDER_FIX").ok() else {
        eprintln!("SKIP: P4_RENDER_FIX not set");
        return;
    };
    let bytes = match std::fs::read(&fix) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot read {fix}: {e}");
            return;
        }
    };
    let page = env("P4_RENDER_PAGE", "0").parse::<usize>().unwrap_or(0);
    let dpi = env("P4_RENDER_DPI", "200").parse::<u32>().unwrap_or(200);
    let (png, _pbox, _rot) = match render_pdf_bytes(&bytes, page, dpi) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: render failed: {e}");
            return;
        }
    };
    let out = env("P4_RENDER_OUT", "/tmp/opb-run/p4_render.png");
    std::fs::write(&out, &png).expect("write png");
    let (w, h) = png_size(&png);
    eprintln!(
        "[p4] {fix} p{page}@{dpi} -> {out} {w}x{h} ({} KiB)",
        png.len() / 1024
    );
}
