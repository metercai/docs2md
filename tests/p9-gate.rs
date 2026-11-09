//! P9 gate: 623-page full-corpus run (plan §P9 / §全状态覆盖验收).
//!
//! A. Terminal-state histogram over all 623 pages of the 8 fixture PDFs
//!    (42+105+415+46+1+2+9+3). Classification (plan §2.1 serial chain):
//!      S0  = L0-native
//!      S1a = L1-indra (the serial chain passed)
//!      S1b = L2-ppv6(chain:*) (the chain hit a stage and PP-OCRv6 read the
//!            page — includes the three dark CJK scans plus the Chinese
//!            documents, which the serial chain's [script] stage routes to
//!            the engine that reads Chinese better)
//!      S2  = everything else (L1-ppv6-* with use_indra=false, L1(g2:*)
//!            keeps, L1->L2(l1-unavailable|l1-failed))
//!    Structural invariants are asserted; the S1a/S1b/S2 split is printed
//!    for calibration (the chain thresholds were re-tuned in this plan —
//!    the pre-chain S1a=259/S2=276 calibration is retired with G1/G3).
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
/// match the rendered PNG corpus tag-by-tag (42+105+415+46+1+2+9+3).
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

/// The three known dark-background CJK pages (plan §7.6): they must be
/// escalated by the serial chain (the known fallback samples).
const S1B_PAGES: [&str; 3] = [
    "扫描文件_20250502_231411_0000",
    "扫描文件_20250502_231411_0001",
    "扫描文件_20250502_231411_0002",
];

fn cfg() -> Config {
    let mut c = Config::default();
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
    if tier == "L0-native" {
        "S0"
    } else if tier == "L1-indra" {
        "S1a"
    } else if tier.starts_with("L2-ppv6(chain:") {
        "S1b"
    } else {
        "S2"
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
        let cls_counts =
            |want: fn(&str) -> bool| res.pages.iter().filter(|r| want(r.tier.as_str())).count();
        eprintln!(
            "[p9] {stem}: {expect} pages in {dt:.1}s (S0={} S1a={} S1b={} S2={})",
            cls_counts(|t| t == "L0-native"),
            cls_counts(|t| t == "L1-indra"),
            cls_counts(|t| t.starts_with("L2-ppv6(chain:")),
            cls_counts(
                |t| !matches!(t, "L0-native" | "L1-indra") && !t.starts_with("L2-ppv6(chain:")
            ),
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
    // The G no-empty-page invariant (every page has non-empty markdown or a
    // non-empty note) is enforced by construction in `l3_fallback` /
    // `l2_page` and unit-tested there; the low-conf / keep-l1 / l3-
    // unavailable tiers are the only ones whose markdown is empty by
    // design (their note is what makes the page non-empty).
    for (tag, _, tier) in &pages {
        let recognized = tier == "L0-native"
            || tier == "L1-indra"
            || tier.starts_with("L2-ppv6(chain:")
            || tier.starts_with("L1-ppv6-")
            || tier.starts_with("L1(g2:")
            || tier.starts_with("L1->L2(")
            || tier.starts_with("L3-vlm(chain:");
        assert!(recognized, "{tag}: unrecognized terminal tier {tier:?}");
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
    // failure shows the actual values (the S1a/S1b split is printed for
    // calibration; only the structural invariants below are asserted).
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
        c.s0,
        c.s1a,
        c.s1b,
        c.s2,
        c.s0 + c.s1a + c.s1b + c.s2
    );

    // Structural invariants (calibration-independent):
    // S0 is the L0-native text-layer count (the has_table reroute,
    // unchanged by the serial chain); S1b contains the three known dark
    // pages (the serial chain's script stage must catch them); nothing is
    // lost.
    assert_eq!(c.s0, 85, "S0 = 85 text-layer pages (has_table reroute)");
    let dark_hit = S1B_PAGES
        .iter()
        .filter(|t| s1b_tags.contains(&t.to_string()))
        .count();
    assert_eq!(
        dark_hit,
        S1B_PAGES.len(),
        "the three dark-background CJK pages must be chain-escalated (S1b); \
         S1b = {s1b_tags:?}"
    );
    assert_eq!(
        c.s0 + c.s1a + c.s1b + c.s2,
        623,
        "every page lands in exactly one class"
    );

    // B: determinism — second full run, byte-identical markdown.
    let mut e2 = Engine::new(cfg()).expect("engine (run 2)");
    let (md2, pages2) = run_pass(&mut e2);
    assert_eq!(md1, md2, "two consecutive runs must be byte-identical");
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
         S0={} S1a={} S1b={} S2={} | S2 tiers: {s2_tiers:?}",
        total / 623.0 / 2.0,
        c.s0,
        c.s1a,
        c.s1b,
        c.s2,
    );
    println!("P9 gate PASS");
}
