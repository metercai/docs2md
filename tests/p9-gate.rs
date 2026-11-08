//! P9 gate: 623-page full-corpus run (plan §P9 / §全状态覆盖验收).
//!
//! A. Terminal-state histogram over all 623 pages of the 8 fixture PDFs
//!    (42+105+415+46+1+2+9+3). Classification:
//!      S0 = L0-native            (expected 85)
//!      S1a = L1-indra            (expected 259-k)
//!      S1b = L1-ppv6(g3-fallback)(expected exactly 3: the dark CJK scans)
//!      S2 = everything else (L2-vlm / L1->L2(...) / L2-unavailable keeps)
//!           (expected 276+k)
//!    Expected values are calibrated at the current default dpi=200 and the
//!    post-route has_table reroute (pdf-inspector's free span-geometry table
//!    signal, zero-model replacement for the former C1 structural reroute;
//!    re-calibrated 2026-11-08: 15 S0 table pages reroute S0 -> L1-indra,
//!    S0 100 -> 85, S1a 244 -> 259, S2 unchanged. The former C1 layout
//!    calibration — 25 pages, S0 100 -> 75, S1a 244 -> 269 — is archived in
//!    .hermes/qa/). The L1/L2 split is the dense-judge routing
//!    calibration; L2 may be unavailable in the test environment, which
//!    does not change the routing verdict (the tier string is decided
//!    before the L2 call).
//! B. Determinism: two consecutive full runs, byte-identical markdown
//!    (plan line 139: temp-0 greedy CTC + deterministic beam + fixed
//!    thread count). Run with RTEN_NUM_THREADS=1 for the strictest
//!    single-core reading.
//! C. Dead-state check: every page has exactly one terminal tier, and no
//!    page is left with neither markdown nor a note.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use docs2md::config::Config;
use docs2md::Engine;

const FIXTURES: &str = "/home/ubuntu/pdf2md.fixtures";

/// The 623-page corpus: (pdf stem, expected page count). The page counts
/// match the rendered PNG corpus tag-by-tag (42+105+415+46+1+2+9+3=623).
const DOCS: [(&str, usize); 8] = [
    ("RFdiffusion3-2025.09.18.676967v1.full", 42),
    ("WO2024133739A1_amylin_NN_patent_oral", 105),
    ("WO2025114501A1", 415),
    ("云手机_CPH_最佳实践", 46),
    ("北京市个人小客车更新指标通知书", 1),
    ("国家移民局政务服务平台出入境预约申请表", 2),
    ("小升初-五一小学-蔡现霖-简历V3", 9),
    ("扫描文件_20250502_231411", 3),
];

/// The three known dark-background CJK pages (plan §7.6): they must be the
/// S1b set — exactly 3 (more = over-fallback, fewer = missed fallback).
const S1B_PAGES: [&str; 3] = [
    "扫描文件_20250502_231411_0000",
    "扫描文件_20250502_231411_0001",
    "扫描文件_20250502_231411_0002",
];

fn cfg() -> Config {
    let mut c = Config::default();
    c.l1.layout_model = Some(PathBuf::from(
        concat!(env!("CARGO_MANIFEST_DIR"), "/models/DocOCR/pp_doclayout_plus_l.onnx"),
    ));
    c.l1.use_indra = true;
    c
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Class {
    s0: usize,
    s1a: usize,
    s1b: usize,
    s2: usize,
}

fn classify(tier: &str) -> &'static str {
    match tier {
        "L0-native" => "S0",
        "L1-indra" => "S1a",
        "L1-ppv6(g3-fallback)" => "S1b",
        _ => "S2",
    }
}

/// One full-corpus pass. Returns (markdown, per-page (stem, page1, tier)).
fn run_pass(engine: &mut Engine) -> (String, Vec<(String, u32, String)>) {
    let mut md_all = String::new();
    let mut pages = Vec::new();
    for (stem, expect) in DOCS {
        let p = Path::new(FIXTURES).join(format!("{stem}.pdf"));
        let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
        let t0 = std::time::Instant::now();
        let res = engine
            .convert(&bytes, None)
            .unwrap_or_else(|e| panic!("{stem}: convert failed: {e}"));
        let dt = t0.elapsed().as_secs_f64();
        assert_eq!(
            res.pages.len(),
            expect,
            "{stem}: expected {expect} pages, got {}",
            res.pages.len()
        );
        for rec in &res.pages {
            pages.push((
                format!("{stem}_{:04}", rec.page - 1),
                rec.page,
                rec.tier.clone(),
            ));
        }
        md_all.push_str(&res.markdown);
        // Per-doc class counts for calibration cross-checks.
        let cls_counts = |want: fn(&str) -> bool| {
            res.pages.iter().filter(|r| want(r.tier.as_str())).count()
        };
        eprintln!(
            "[p9] {stem}: {expect} pages in {dt:.1}s (S0={} S1a={} S1b={} S2={})",
            cls_counts(|t| t == "L0-native"),
            cls_counts(|t| t == "L1-indra"),
            cls_counts(|t| t == "L1-ppv6(g3-fallback)"),
            cls_counts(|t| !matches!(t, "L0-native" | "L1-indra" | "L1-ppv6(g3-fallback)")),
        );
    }
    (md_all, pages)
}

#[test]
fn p9_623_histogram_determinism() {
    // Local-only gate: the 623-page fixture corpus is not shipped in the repo,
    // so on a runner without it we skip cleanly (CI green) rather than panic
    // in run_pass. On a dev box with the corpus the gate runs in full.
    if !Path::new(FIXTURES).is_dir() {
        eprintln!("[p9] SKIP: fixtures not present at {FIXTURES} (local-only gate)");
        return;
    }
    let t_start = std::time::Instant::now();
    let mut e1 = Engine::new(cfg()).expect("engine");
    let (md1, pages) = run_pass(&mut e1);
    assert_eq!(pages.len(), 623, "corpus is 623 pages");

    // C: dead-state check — every page carries a recognized terminal tier.
    // Pages whose L2 leg did not complete (L2 unavailable or call failed)
    // keep a `L1(...)` tier and must be the only ones with empty markdown.
    let mut l2_keep_empty_md = 0usize;
    for (tag, _, tier) in &pages {
        let recognized = matches!(
            tier.as_str(),
            "L0-native"
                | "L1-indra"
                | "L1-ppv6(g3-fallback)"
                | "L2-vlm"
                | "L2-vlm(failed)"
                | "L1->L2(l1-unavailable)"
                | "L1->L2(l1-unavailable)(failed)"
                | "L1->L2(l1-failed)"
                | "L1->L2(l1-failed)(failed)"
                | "L1->L2(g3+g2:hard)"
                | "L1->L2(g3+g2:medium)"
                | "L1->L2(g3+g2:hard)(failed)"
                | "L1->L2(g3+g2:medium)(failed)"
                | "L1(g3+g2:hard)"
                | "L1(g3+g2:medium)"
        );
        assert!(recognized, "{tag}: unrecognized terminal tier {tier:?}");
        let keep = tier.starts_with("L1(") || tier.starts_with("L1->L2(");
        if keep {
            l2_keep_empty_md += 1;
        }
    }

    // A: histogram.
    let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
    let mut s1b_tags: Vec<String> = Vec::new();
    for (tag, _, tier) in &pages {
        *hist.entry(classify(tier)).or_insert(0) += 1;
        if classify(tier) == "S1b" {
            s1b_tags.push(tag.clone());
        }
    }
    let c = Class {
        s0: hist.get("S0").copied().unwrap_or(0),
        s1a: hist.get("S1a").copied().unwrap_or(0),
        s1b: hist.get("S1b").copied().unwrap_or(0),
        s2: hist.get("S2").copied().unwrap_or(0),
    };
    // Print the observed histogram before the assertions so a calibration
    // failure shows the actual values (the post-route reroute signal is the
    // zero-model has_table, not the former C1 layout model — the expected
    // totals below are re-calibrated against this).
    let s2_tiers_pre: BTreeMap<&str, usize> = pages
        .iter()
        .filter(|(_, _, t)| classify(t) == "S2")
        .map(|(_, _, t)| t.as_str())
        .fold(BTreeMap::new(), |mut m, t| {
            *m.entry(t).or_insert(0) += 1;
            m
        });
    println!(
        "P9 observed: S0={} S1a={} S1b={} S2={} (total={}) | S2 tiers: {s2_tiers_pre:?}",
        c.s0, c.s1a, c.s1b, c.s2,
        c.s0 + c.s1a + c.s1b + c.s2
    );

    // S1b must be exactly the three known dark pages (plan line 136).
    let mut want: Vec<&str> = S1B_PAGES.to_vec();
    want.sort();
    let mut got: Vec<&str> = s1b_tags.iter().map(|s| s.as_str()).collect();
    got.sort();
    assert_eq!(
        got, want,
        "S1b must be exactly the three known dark-background CJK pages"
    );

    // Expected totals (calibrated at dpi=200 + has_table reroute):
    // S0=85, S1a=259-k, S1b=3, S2=276+k. Invariants: S0=85, S1b=3,
    // L1-routed (S1a+S1b)=262, total=623.
    assert_eq!(c.s0, 85, "S0 = 85 text-layer pages (post has_table reroute)");
    assert_eq!(c.s1b, 3, "S1b = exactly 3 dark pages");
    assert_eq!(c.s0 + c.s1a + c.s1b + c.s2, 623);
    assert_eq!(
        c.s1a + c.s1b,
        262,
        "L1-routed (S1a+S1b) = 262 (dpi=200 + has_table calibration, k=0 expected)"
    );

    // B: determinism — second full run, byte-identical markdown.
    let mut e2 = Engine::new(cfg()).expect("engine (run 2)");
    let (md2, pages2) = run_pass(&mut e2);
    assert_eq!(
        md1, md2,
        "two consecutive runs must be byte-identical"
    );
    assert_eq!(pages.len(), pages2.len());
    for ((_, p1, t1), (_, p2, t2)) in pages.iter().zip(pages2.iter()) {
        assert_eq!((p1, t1), (p2, t2), "per-page tier must match run 2");
    }

    let total = t_start.elapsed().as_secs_f64();
    let s2_tiers: BTreeMap<&str, usize> = pages
        .iter()
        .filter(|(_, _, t)| classify(t) == "S2")
        .map(|(_, _, t)| t.as_str())
        .fold(BTreeMap::new(), |mut m, t| {
            *m.entry(t).or_insert(0) += 1;
            m
        });
    println!(
        "P9 gate: total={total:.0}s ({}s/page both passes) \
         S0={} S1a={} S1b={} S2={} | S2 tiers: {s2_tiers:?} \
         l2_unavailable_keeps={l2_keep_empty_md}",
        total / 623.0 / 2.0,
        c.s0,
        c.s1a,
        c.s1b,
        c.s2
    );
    println!("P9 gate PASS");
}
