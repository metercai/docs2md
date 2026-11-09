// P1 verification test (one-shot): score our pipeline on a given render set.
//
// Closes the 2x2 attribution matrix. Two entry points:
//
//   p1_render_pages : render the 32 needs_ocr pages with the product's own
//                     render path (render_pdf_bytes = PDFium @ dpi 200,
//                     font-environment-independent via core-14). Writes
//                     PNGs to P1B_RENDER_OUT.
//
//   p1_verify_renders: feed each rendered PNG through our exact L1 flow
//                     (mirrors Engine::l1_dual, lib.rs:496): Indra ocr_page
//                     -> serial metric chain -> PP-OCRv6 -> l3 gate, VLM off
//                     -> l3_fallback. 16 L0-native pages are render-
//                     independent (native text layer) and are copied
//                     verbatim from the t7 run so the 48-page comparison
//                     stays apples-to-apples. Writes /tmp/opb_run_*/{page}
//                     .json + md/ for .hermes/qa/opb48_score.py.
//
// Measured:
//   pypdfium2 renders -> layout 34/43 = 0.7907 (vs 30/43 = 0.6977 on the
//   product's old system-font renders) => the render change recovers
//   +0.0930 of the 0.1395 gap to native (67%); the same-render native-vs-ours
//   residual is +0.0465 (33%), 4 specific layout_kind tests.

use docs2md::config::{Config, L3Config, RouteConfig};
use docs2md::l1::{IndraEngine, L1Engine, PpOcrEngine};
use docs2md::markdown::ocr_lines_to_markdown;
use docs2md::route::l3_gate;
use std::path::Path;

const T7: &str = "/tmp/opb_run_t7_blocks_norel";
const DEFAULT_RENDERS: &str = "/tmp/opb_pypdfium2_renders";
const DEFAULT_OUT: &str = "/tmp/opb_run_p1_pypdfium2";

/// Render dir + output dir, overridable so the same test can score a
/// different render set (e.g. the Liberation-font renders).
fn renders_dir() -> String {
    std::env::var("P1B_RENDERS").unwrap_or_else(|_| DEFAULT_RENDERS.to_string())
}
fn out_dir() -> String {
    std::env::var("P1B_OUT").unwrap_or_else(|_| DEFAULT_OUT.to_string())
}

/// Mirror of Engine::l1_dual (lib.rs:496-609) minus the note strings (not
/// scored): returns (markdown, tier, blocks).
fn l1_dual(
    page1: u32,
    png: &[u8],
    indra: &mut IndraEngine,
    pp: &mut PpOcrEngine,
    rc: &RouteConfig,
    l3: &L3Config,
) -> (String, String, Vec<serde_json::Value>) {
    let primary = indra
        .ocr_page(png, rc)
        .unwrap_or_else(|e| panic!("p{page1}: {e}"));
    let Some(tag) = primary.escalate_l2 else {
        // Chain passed: the L1 output stands.
        let md = primary
            .md
            .unwrap_or_else(|| ocr_lines_to_markdown(&primary.lines));
        let blocks: Vec<serde_json::Value> = primary
            .blocks
            .iter()
            .map(|b| serde_json::json!({"label": b.label, "bbox": b.bbox, "text": b.text}))
            .collect();
        return (md, "L1-indra".to_string(), blocks);
    };
    // Chain hit: read with PP-OCRv6 on the same image.
    eprintln!("[p{page1}] chain:{tag}");
    let pp_page = match pp.ocr_page(png, rc) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[p{page1}] L2 failed: {e} - keeping L1");
            let md = primary
                .md
                .clone()
                .unwrap_or_else(|| ocr_lines_to_markdown(&primary.lines));
            let blocks: Vec<serde_json::Value> = primary
                .blocks
                .iter()
                .map(|b| serde_json::json!({"label": b.label, "bbox": b.bbox, "text": b.text}))
                .collect();
            return (md, format!("L2-ppv6(chain:{tag})(l2-failed)"), blocks);
        }
    };
    let Some(l3_tag) = l3_gate(pp_page.char_count(), pp_page.mean_conf(), l3) else {
        // PP-OCRv6 stands; it emits no structural blocks.
        let md = ocr_lines_to_markdown(&pp_page.lines);
        return (md, format!("L2-ppv6(chain:{tag})"), vec![]);
    };
    // l3-gate hit: VLM is off in this run (base_url empty) -> l3_fallback:
    // PP text stands when non-empty, else keep L1.
    eprintln!("[p{page1}] l3:{l3_tag} - VLM off, fallback");
    let pp_md = ocr_lines_to_markdown(&pp_page.lines);
    if !pp_md.trim().is_empty() {
        (
            pp_md,
            format!("L2-ppv6(chain:{l3_tag})(l3-unavailable)"),
            vec![],
        )
    } else {
        let md = primary
            .md
            .clone()
            .unwrap_or_else(|| ocr_lines_to_markdown(&primary.lines));
        let blocks: Vec<serde_json::Value> = primary
            .blocks
            .iter()
            .map(|b| serde_json::json!({"label": b.label, "bbox": b.bbox, "text": b.text}))
            .collect();
        (md, format!("L2-ppv6(chain:{l3_tag})(keep-l1)"), blocks)
    }
}

/// t7 run records for the 48 pages: (page_id, tier, md_file, pdf path).
fn t7_pages() -> Vec<(String, String, String, String)> {
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/opb66_pdf.json").unwrap()).unwrap();
    let path_by_id: std::collections::HashMap<_, _> = raw
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["page_id"].as_str().unwrap().to_string(),
                v["path"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    std::fs::read_dir(T7)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n.ends_with(".json") && n != "compare.json"
        })
        .map(|jf| {
            let rec: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&jf).unwrap()).unwrap();
            let pid = rec["page_id"].as_str().unwrap().to_string();
            let tier = rec["tier"].as_str().unwrap().to_string();
            let md_file = rec["md_file"].as_str().unwrap().to_string();
            let path = path_by_id[&pid].clone();
            (pid, tier, md_file, path)
        })
        .collect()
}

/// Render the 32 needs_ocr pages with the product's render path (PDFium @
/// dpi 200). Renders are font-environment-independent (PDFium core-14), so
/// no FONTCONFIG_FILE is needed. Writes {pid}.png to P1B_RENDER_OUT.
#[test]
fn p1_render_pages() {
    let out = std::env::var("P1B_RENDER_OUT").unwrap_or_else(|_| "/tmp/opb_p1_renders".to_string());
    std::fs::create_dir_all(&out).unwrap();
    let dpi = 200u32; // product default (render.rs / config.rs L0Config::default)
    for (pid, tier, _md, path) in t7_pages() {
        if tier == "L0-native" {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let (png, _pbox, _rot) = docs2md::render::render_pdf_bytes(&bytes, 0, dpi)
            .unwrap_or_else(|e| panic!("{pid} render: {e}"));
        std::fs::write(Path::new(&out).join(format!("{pid}.png")), &png).unwrap();
        eprintln!("[render] {pid}: {} bytes", png.len());
    }
    eprintln!("renders done -> {out}");
}

/// Score our pipeline on the renders in P1B_RENDERS (default: the
/// pypdfium2 set), writing a full 48-page run dir for the QA scorer.
#[test]
fn p1_verify_renders() {
    let cfg = Config::default(); // dpi=200, use_indra, tier=small, no VLM - same as t7
    let rc: RouteConfig = cfg.route.clone();
    let l3: L3Config = cfg.l3.clone();

    let mut indra = IndraEngine::new().expect("indra");
    let mut pp = PpOcrEngine::new(&cfg.l1.tier, cfg.l1.model_dir.as_ref()).expect("ppocrv6");

    let od = out_dir();
    let out = Path::new(&od);
    std::fs::create_dir_all(out.join("md")).unwrap();

    let mut n_l1fy = 0usize;
    let mut n_l0 = 0usize;
    let mut tiers: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let rdir = renders_dir();

    for (pid, tier, md_file, _path) in t7_pages() {
        let flat_md = md_file.strip_prefix("md/").unwrap_or(&md_file).to_string();
        if tier == "L0-native" {
            // Render-independent: copy verbatim from the t7 run.
            n_l0 += 1;
            let src = Path::new(T7).join(&md_file);
            let rec = std::fs::read_to_string(Path::new(T7).join(format!("{pid}.json"))).unwrap();
            std::fs::copy(&src, out.join("md").join(&flat_md)).unwrap();
            std::fs::write(out.join(format!("{pid}.json")), rec).unwrap();
            continue;
        }

        // needs_ocr: feed the render through our exact L1 flow.
        let png = std::fs::read(Path::new(&rdir).join(format!("{pid}.png")))
            .unwrap_or_else(|e| panic!("missing render for {pid} in {rdir}: {e}"));
        let t0 = std::time::Instant::now();
        let (md, tier_out, blocks) = l1_dual(1, &png, &mut indra, &mut pp, &rc, &l3);
        let latency = t0.elapsed().as_secs_f64();
        n_l1fy += 1;
        *tiers.entry(tier_out.clone()).or_insert(0) += 1;
        std::fs::write(out.join("md").join(&flat_md), &md).unwrap();
        std::fs::write(
            out.join(format!("{pid}.json")),
            serde_json::json!({
                "page_id": pid,
                "md_file": md_file,
                "latency": latency,
                "tier": tier_out,
                "blocks": blocks,
            })
            .to_string(),
        )
        .unwrap();
        eprintln!(
            "[{pid}] {tier} -> {tier_out} {latency:.1}s blocks={}",
            blocks.len()
        );
    }

    assert_eq!(n_l0, 16, "expected 16 L0-native pages");
    assert_eq!(n_l1fy, 32, "expected 32 needs_ocr pages");
    eprintln!(
        "done: {n_l0} L0 copied, {n_l1fy} re-run on renders from {rdir} -> {}",
        out_dir()
    );
    for (t, n) in &tiers {
        eprintln!("  tier {t}: {n}");
    }
}
