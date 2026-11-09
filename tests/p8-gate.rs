//! P8 gate: serial-chain integration smoke on the 32-page stratified
//! sample (the 623-page histogram + determinism + performance live in P9).
//!
//! A. 32/32 pages: the full `ocr_page` pipeline (preview -> skew -> orient
//!    -> detect -> [det] chain stage -> script pre-pass -> [script] chain
//!    stage -> recognize -> [rec] conf stage -> assemble -> [assemble]
//!    chars stage -> markdown + table post-filter) runs without error and
//!    yields one terminal outcome per page (no dead state): either the chain
//!    passed, or it stopped at a stage and tagged the page `escalate_l2`.
//! B. md cross-check: per-page 2-gram Dice vs the Python reference markdown
//!    (tests/p1/ref/*.md.json). The table post-filter is new logic (Python
//!    has none; unit-tested in the lib), and the reference md is pre-filter —
//!    a page whose whole-form pseudo-table the post-filter demotes is a
//!    structural divergence. So: non-demoted pages must hit Dice >= 0.95;
//!    demoted pages are reported, not failed. Chain-escalated pages carry
//!    the (possibly partial) L1 output — reported, not failed.
//! C. S1b reachability: the three dark-background CJK pages (29/30/31) must
//!    be escalated by the serial chain (the known fallback samples,
//!    plan §全状态覆盖验收).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use image::ImageReader;

use docs2md::l1::indra::preproc::Gray;
use docs2md::l1::indra::IndraEngine;
use docs2md::l1::OcrPage;

const REF_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/p1");
const GATE_DICE: f64 = 0.95;

/// 2-gram Dice: 2|A∩B| / (|A|+|B|) over character bigrams (bag of grams) —
/// the same criterion as the P6 gate.
fn dice2(a: &str, b: &str) -> f64 {
    fn grams(s: &str) -> HashMap<String, usize> {
        let mut m = HashMap::new();
        let cs: Vec<char> = s.chars().collect();
        for w in cs.windows(2) {
            let g: String = w.iter().collect();
            *m.entry(g).or_insert(0) += 1;
        }
        m
    }
    let (ga, gb) = (grams(a), grams(b));
    let inter: usize = ga
        .iter()
        .filter_map(|(g, &ca)| gb.get(g).map(|&cb| ca.min(cb)))
        .sum();
    let ta: usize = ga.values().sum();
    let tb: usize = gb.values().sum();
    if ta + tb == 0 {
        return 1.0;
    }
    2.0 * inter as f64 / (ta + tb) as f64
}

fn samples() -> Vec<PathBuf> {
    let dir = Path::new(REF_DIR).join("samples");
    std::fs::read_dir(&dir)
        .expect("tests/p1/samples")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect()
}

fn tag_of(sp: &Path) -> &str {
    sp.file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .strip_suffix(".png")
        .unwrap()
}

fn ref_md(tag: &str) -> Option<String> {
    let p = Path::new(REF_DIR)
        .join("ref")
        .join(format!("{tag}.md.json"));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).ok()?).ok()?;
    v["md"].as_str().map(|s| s.to_string())
}

#[test]
fn p8_gate_32_pages() {
    let sample_paths = samples();
    assert_eq!(sample_paths.len(), 32, "32 stratified sample pages");

    // A: one full pipeline run; every page must terminate (no dead state) —
    // chain-passed or tagged at a chain stage, never neither.
    let mut engine = IndraEngine::init().expect("indra init");
    let rc = docs2md::config::RouteConfig::default();
    let mut pages: Vec<(String, OcrPage)> = Vec::with_capacity(sample_paths.len());
    for sp in &sample_paths {
        let tag = tag_of(sp);
        let upright = ImageReader::open(sp)
            .expect("open sample")
            .decode()
            .expect("decode sample")
            .into_luma8();
        let gray = Gray {
            w: upright.width() as usize,
            h: upright.height() as usize,
            data: upright.as_raw().to_vec(),
        };
        let page = engine
            .ocr_page(&gray, &rc)
            .unwrap_or_else(|e| panic!("{tag}: ocr_page failed: {e}"));
        pages.push((tag.to_string(), page));
    }
    assert_eq!(
        pages.len(),
        32,
        "all 32 pages must produce a terminal outcome"
    );
    let escalated: Vec<&String> = pages
        .iter()
        .filter(|(_, p)| p.escalate_l2.is_some())
        .map(|(t, _)| t)
        .collect();
    eprintln!("P8 chain escalations: {escalated:?}");

    // B: md cross-check vs the Python reference (all 32 have a reference).
    // Non-demoted, non-escalated pages: dice must hit GATE_DICE. Demoted
    // pages (reference keeps the <table> html, Rust demotes it to plain
    // text): reported only. Chain-escalated pages carry the (possibly
    // partial) L1 output: reported only.
    let mut dice_sum = 0.0f64;
    let mut non_demoted_worst = (1.0f64, String::new());
    let mut below_gate: Vec<String> = Vec::new();
    let mut reported: Vec<String> = Vec::new();
    for (tag, page) in &pages {
        let want = ref_md(tag).expect("reference md.json for every sample");
        let got = page.md.as_deref().unwrap_or("");
        let d = dice2(got, &want);
        dice_sum += d;
        let demoted = !got.contains("<table") && want.contains("<table");
        if demoted || page.escalate_l2.is_some() {
            reported.push(format!(
                "{tag} dice {d:.4} (md_chars={} ref_chars={} escalate={:?})",
                got.chars().count(),
                want.chars().count(),
                page.escalate_l2
            ));
            continue;
        }
        if d < non_demoted_worst.0 {
            non_demoted_worst = (d, tag.clone());
        }
        if d < GATE_DICE {
            below_gate.push(format!(
                "{tag} dice {d:.4} (md_chars={} ref_chars={})",
                got.chars().count(),
                want.chars().count()
            ));
        }
    }
    assert!(
        below_gate.is_empty(),
        "pages below {GATE_DICE}: {below_gate:?}"
    );

    // C: the three dark-background CJK pages must be escalated by the
    // serial chain (S1b reachable).
    let dark = [
        "29_扫描文件_2025050_p0",
        "30_扫描文件_2025050_p1",
        "31_扫描文件_2025050_p2",
    ];
    for (tag, page) in &pages {
        if !dark.contains(&tag.as_str()) {
            continue;
        }
        let chars = page.char_count();
        let conf = page.mean_conf();
        assert!(
            page.escalate_l2.is_some(),
            "{tag}: expected the serial chain to escalate (S1b known sample), \
             got chain-passed (md_chars={chars}, conf={conf:.3}) — fallback unreachable"
        );
        eprintln!(
            "{tag}: chain:{:?} (md_chars={chars}, conf={conf:.3}) — S1b reachable",
            page.escalate_l2
        );
    }

    println!(
        "P8 gate: pages_ok=32/32 md_compared=32 mean_dice={:.4} \
         non_demoted_worst={} ({:.4}) reported={reported:?} escalated={escalated:?}",
        dice_sum / 32.0,
        non_demoted_worst.1,
        non_demoted_worst.0
    );
    println!("P8 gate PASS");
}
