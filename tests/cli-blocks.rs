//! Regression: the `--blocks` CLI surface (QA blocker B) and `--version`
//! (QA blocker A). Runs the real binary on a synthetic single-page PDF with
//! a text layer, so it works on a CI runner with no fixtures or models.
//!
//! The JSON shape is the contract QA checks: per page `page`/`tier`/`blocks`,
//! per block `label`/`bbox`/`text` with bbox normalized to [0,1] y-down.

use std::io::Write;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_docs2md");

/// A minimal valid one-page PDF (612x792, Helvetica text layer) with a
/// correct xref table, built with computed byte offsets.
fn minimal_pdf() -> Vec<u8> {
    let content = b"BT /F1 12 Tf 72 720 Td (First line of the synthetic page) Tj 0 -20 Td (Second line for block clustering) Tj ET";
    let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
    let mut offsets: Vec<u32> = Vec::new();

    let mut obj = |out: &mut Vec<u8>, n: u32, body: &[u8]| {
        offsets.push(out.len() as u32);
        let head = format!("{n} 0 obj\n");
        out.extend_from_slice(head.as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };

    obj(
        &mut out,
        1,
        b"<< /Type /Catalog /Pages 2 0 R >>",
    );
    obj(
        &mut out,
        2,
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    );
    obj(
        &mut out,
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    );
    {
        let body = format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len(),
            std::str::from_utf8(content).unwrap()
        );
        obj(&mut out, 4, body.as_bytes());
    }
    obj(
        &mut out,
        5,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    );

    let xref_pos = out.len() as u32;
    let mut xref = format!("xref\n0 {}\n", offsets.len() + 1);
    xref.push_str("0000000000 65535 f \n");
    for o in &offsets {
        xref.push_str(&format!("{:010} 00000 n \n", o));
    }
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        offsets.len() + 1,
        xref_pos
    );
    out.extend_from_slice(xref.as_bytes());
    out.extend_from_slice(trailer.as_bytes());
    out
}

#[test]
fn version_matches_cargo_version() {
    let out = Command::new(BIN)
        .arg("--version")
        .output()
        .expect("run docs2md --version");
    assert!(out.status.success(), "exit: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.trim() == format!("docs2md {}", env!("CARGO_PKG_VERSION")),
        "version line was {stdout:?}, expected docs2md {}",
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn blocks_json_surface() {
    let dir = std::env::temp_dir().join(format!("docs2md-cli-blocks-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let pdf = dir.join("page.pdf");
    let blocks = dir.join("blocks.json");
    let md = dir.join("page.md");
    let mut f = std::fs::File::create(&pdf).expect("write pdf");
    f.write_all(&minimal_pdf()).expect("write pdf bytes");
    drop(f);

    let out = Command::new(BIN)
        .arg("--blocks")
        .arg(&blocks)
        .arg(&pdf)
        .arg(&md)
        .output()
        .expect("run docs2md --blocks");
    assert!(
        out.status.success(),
        "exit: {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(md.exists(), "markdown not written");

    let raw = std::fs::read_to_string(&blocks).expect("blocks json written");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
    let pages = doc.get("pages").and_then(|p| p.as_array()).expect("pages array");
    assert_eq!(pages.len(), 1, "one page in");

    let page = &pages[0];
    assert_eq!(page.get("page").and_then(|v| v.as_u64()), Some(1));
    assert!(page.get("tier").and_then(|v| v.as_str()).is_some());
    let blocks = page.get("blocks").and_then(|b| b.as_array()).expect("blocks array");
    // A text-layer page yields L0 line blocks: two text lines.
    assert!(!blocks.is_empty(), "L0 text-layer page must emit blocks");
    let total_text: String = blocks
        .iter()
        .map(|b| b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
        .collect();
    assert!(total_text.contains("First line"), "text preserved: {total_text:?}");

    for b in blocks {
        let label = b.get("label").and_then(|v| v.as_str()).expect("label");
        assert!(
            docs2md::blocks::LABELS.contains(&label),
            "label {label} not in the 11-label vocabulary"
        );
        let bbox = b.get("bbox").and_then(|v| v.as_array()).expect("bbox array");
        assert_eq!(bbox.len(), 4, "bbox [x0,y0,x1,y1]");
        for c in bbox {
            let v = c.as_f64().expect("numeric bbox");
            assert!((0.0..=1.0).contains(&v), "bbox {v} out of [0,1]");
        }
        assert!(b.get("text").is_some());
    }
    // Reading order: the first block is the top line.
    let first = blocks[0].get("text").and_then(|t| t.as_str()).unwrap_or("");
    assert!(first.contains("First line"), "top line first: {first:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_blocks_flag_means_no_json() {
    let dir = std::env::temp_dir().join(format!("docs2md-cli-nb-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let pdf = dir.join("page.pdf");
    let md = dir.join("page.md");
    std::fs::write(&pdf, minimal_pdf()).expect("write pdf");

    let out = Command::new(BIN)
        .arg(&pdf)
        .arg(&md)
        .output()
        .expect("run docs2md");
    assert!(out.status.success(), "exit: {:?}", out.status);
    assert!(md.exists());
    // Without --blocks there is no JSON side output.
    assert!(!dir.join("page.json").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
