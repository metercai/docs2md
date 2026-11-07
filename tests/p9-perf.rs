//! P9 performance probe: pure `ocr_page` wall time (the Indra pipeline only —
//! no PDF parse, no rendering, no G1 layout) on the two dense patent docs
//! (520 pages). This isolates the engine cost for the plan §P9 gate
//! ("single-core reading, median <= 1.0s/page; rten allowed <= 2x the Python
//! reference"). Compare per-page against the Python reference
//! (/tmp/indra-ab/route/indra_all.jsonl, threads=4 wall, median 376ms).
//!
//! Run with RTEN_NUM_THREADS=4 (the plan-aligned reading, mirroring Python's
//! per-page 4-thread wall) and RTEN_NUM_THREADS=1 for the strictest reading.

use std::path::Path;

use docs2md::l1::indra::preproc::Gray;
use docs2md::l1::indra::IndraEngine;
use image::ImageReader;

fn pages_dir() -> String {
    std::env::var("P9_PAGES").unwrap_or_else(|_| "/tmp/indra-ab/route/pages".into())
}
const DOCS: [&str; 2] = [
    "WO2024133739A1_amylin_NN_patent_oral",
    "WO2025114501A1",
];

#[test]
fn p9_ocr_page_perf() {
    // Local-only probe: the rendered page corpus is not shipped in the repo.
    // Skip cleanly on a runner without it (CI green); set P9_PAGES to run.
    let pages = pages_dir();
    let dir = Path::new(&pages);
    if !dir.is_dir() {
        eprintln!("[p9] SKIP: pages dir not present at {} (local-only probe; set P9_PAGES)", dir.display());
        return;
    }
    let mut engine = IndraEngine::init().expect("indra init");
    let mut ms: Vec<u128> = Vec::new();
    for doc in DOCS {
        let files: Vec<String> = std::fs::read_dir(dir)
            .expect("pages dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|f| f.starts_with(&format!("{doc}_")) && f.ends_with(".png"))
            .collect();
        assert!(!files.is_empty(), "no pages for {doc}");
        for f in &files {
            let p = dir.join(f);
            let img = ImageReader::open(&p)
                .expect("open")
                .decode()
                .expect("decode")
                .into_luma8();
            let gray = Gray {
                w: img.width() as usize,
                h: img.height() as usize,
                data: img.as_raw().to_vec(),
            };
            let t0 = std::time::Instant::now();
            let page = engine.ocr_page(&gray).expect("ocr_page");
            let dt = t0.elapsed();
            ms.push(dt.as_millis());
            let _ = page.char_count();
        }
    }
    assert_eq!(ms.len(), 520, "expected 520 pages, got {}", ms.len());
    ms.sort_unstable();
    let n = ms.len();
    let med = ms[n / 2] as f64 / 1000.0;
    let p90 = ms[n * 90 / 100] as f64 / 1000.0;
    let mx = ms[n - 1] as f64 / 1000.0;
    let total_s: u128 = ms.iter().sum::<u128>() / 1000;
    println!(
        "P9 perf: pages_dir={} n={} ocr_page median={med:.3}s p90={p90:.3}s max={mx:.3}s \
         total={}s ({:.3}s/page)",
        dir.display(),
        n,
        total_s,
        total_s as f64 / n as f64
    );
}
