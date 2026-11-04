//! P4 gate: language model + beam search vs the Python reference (plan §P4).
//!
//! Three sub-gates:
//!   1. logp exact: 200 random (history, token) pairs per LM group must
//!      reproduce the Python NgramLM.logp values bit-for-bit (pure table
//!      lookups + f64 arithmetic, no tolerance).
//!   2. beam identical: every real beam-re-read line of the 32-page reference
//!      (Python product code, ORT tensors) is fed to the Rust beam_search and
//!      must return byte-identical (ids, steps).
//!   3. 32 pages with the LM on: the full Rust rec path (greedy + beam, same
//!      as the product) re-runs the P3 judged criteria on the LM reference
//!      (scripts/gen-p4-reference.py): text exact match >= 95%, mean |conf
//!      difference| < 0.02.

use std::path::PathBuf;

use image::ImageReader;

use docs2md::l1::indra::lm::{beam_search, NgramLm};
use docs2md::l1::indra::preproc::Gray;
use docs2md::l1::indra::rec;
use docs2md::l1::indra::IndraEngine;

const REF_DIR: &str = "tests/p1/ref";

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(REF_DIR)
}

/// Sub-gate 1: logp table lookups are exact (plan: 无容差).
#[test]
fn p4_gate_logp_exact() {
    let engine = IndraEngine::init().expect("indra init");
    let refv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir().join("p4-logp-ref.json")).unwrap())
            .unwrap();
    let mut checked = 0usize;
    let mut max_err = 0.0f64;
    for (group, rows) in ["base", "cjk", "world"][..]
        .iter()
        .map(|g| (*g, refv[g].clone()))
    {
        let lm: &NgramLm = engine.lms.get(group).unwrap();
        for row in rows.as_array().unwrap() {
            let hist: Vec<i64> = row["hist"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_i64().unwrap())
                .collect();
            let t = row["t"].as_i64().unwrap();
            let want = row["logp"].as_f64().unwrap();
            let got = lm.logp_lookup(&hist, t);
            checked += 1;
            if got != want {
                let d = (got - want).abs();
                max_err = max_err.max(d);
                eprintln!("{group}: hist={hist:?} t={t} got {got} want {want} (diff {d:e})");
            }
        }
    }
    println!(
        "P4 gate 1 (logp exact): {checked} values, bit-exact mismatches reported above, max |diff| {max_err:e}"
    );
    assert_eq!(
        max_err, 0.0,
        "logp must be bit-exact (pure table lookups); max |diff| {max_err:e}"
    );
}

/// Per-group alpha/beta exactly as the product wires them (meta lm_params
/// override, base defaults).
fn ab(engine: &IndraEngine, group: &str) -> (f64, f64) {
    match engine.meta.lm_params.get(group) {
        Some(p) => (p.alpha, p.beta),
        None => (rec::LM_ALPHA, rec::LM_BETA),
    }
}

/// Sub-gate 2: beam search on the Python reference tensors is byte-identical.
#[test]
fn p4_gate_beam_identical() {
    let engine = IndraEngine::init().expect("indra init");
    let mut files: Vec<PathBuf> = std::fs::read_dir(ref_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".beam.json")
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no beam reference files");
    let mut total = 0usize;
    let mut mismatch = 0usize;
    for f in &files {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        for e in v.as_array().unwrap() {
            total += 1;
            let group = e["group"].as_str().unwrap();
            let lm = engine.lms.get(group).unwrap();
            let (alpha, beta) = ab(&engine, group);
            let blank: Vec<f64> = e["blank"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let topv: Vec<f64> = e["topv"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let topi: Vec<i64> = e["topi"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect();
            let (ids, steps) = beam_search(
                &blank,
                &topv,
                &topi,
                lm,
                alpha,
                beta,
                rec::LM_BEAM,
                rec::LM_CAND_FLOOR,
            );
            let w_ids: Vec<i64> = e["beam"]["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect();
            let w_steps: Vec<usize> = e["beam"]["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            if ids != w_ids || steps != w_steps {
                mismatch += 1;
                eprintln!(
                    "{} line {}: MISMATCH\n  rust ids={:?}\n  ref  ids={:?}\n  rust steps={:?}\n  ref  steps={:?}",
                    f.file_name().unwrap().to_string_lossy(),
                    e["line"],
                    ids,
                    w_ids,
                    steps,
                    w_steps
                );
            }
        }
    }
    println!("P4 gate 2 (beam identical): {total} beam lines, {mismatch} mismatches");
    assert_eq!(mismatch, 0, "beam (ids, steps) must be byte-identical");
}

/// Sub-gate 3: 32 pages with the LM on re-run the P3 judged criteria.
#[test]
fn p4_gate_32_pages_with_lm() {
    let rd = ref_dir();
    let mut metas: Vec<PathBuf> = std::fs::read_dir(&rd)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".meta.json")
        })
        .collect();
    metas.sort();
    assert_eq!(metas.len(), 32, "32 reference pages");

    let engine = IndraEngine::init().expect("indra init");
    let mut text_match = 0usize;
    let mut text_total = 0usize;
    let mut conf_deltas = Vec::new();
    let mut group_diff = 0usize;
    let mut bad_pages = Vec::new();

    for meta_path in &metas {
        let tag = meta_path.file_name().unwrap().to_string_lossy().to_string();
        let tag = &tag[..tag.len() - ".meta.json".len()];

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
        let boxes: Vec<[f64; 4]> = meta["detect"]["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let a = b.as_array().unwrap();
                [
                    a[0].as_f64().unwrap(),
                    a[1].as_f64().unwrap(),
                    a[2].as_f64().unwrap(),
                    a[3].as_f64().unwrap(),
                ]
            })
            .collect();
        let rec_ref: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(rd.join(format!("{tag}.rec4.json"))).unwrap(),
        )
        .unwrap();
        let ref_lines: Vec<(String, f64, String)> = rec_ref["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                (
                    l["text"].as_str().unwrap().to_string(),
                    l["conf"].as_f64().unwrap(),
                    l["group"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(boxes.len(), ref_lines.len(), "{tag}: line count mismatch");

        let upright = ImageReader::open(rd.join(format!("{tag}.upright.png")))
            .unwrap()
            .decode()
            .unwrap()
            .into_luma8();
        let gray = Gray {
            w: upright.width() as usize,
            h: upright.height() as usize,
            data: upright.as_raw().to_vec(),
        };
        let lines = if boxes.is_empty() {
            Vec::new()
        } else {
            rec::recognize(&engine, &gray, &boxes)
        };

        let mut page_match = 0usize;
        for (i, l) in lines.iter().enumerate() {
            text_total += 1;
            let tm = l.text == ref_lines[i].0;
            if tm {
                text_match += 1;
                page_match += 1;
            }
            conf_deltas.push((l.conf as f64 - ref_lines[i].1).abs());
            if l.group != ref_lines[i].2 {
                group_diff += 1;
            }
            if !tm {
                eprintln!(
                    "{tag} line {i}: text mismatch\n  rust: {:?}\n  ref:  {:?} (conf rust {:.4} ref {:.4}, group {} vs {})",
                    l.text, ref_lines[i].0, l.conf, ref_lines[i].1, l.group, ref_lines[i].2
                );
            }
        }
        if page_match < ref_lines.len() {
            bad_pages.push(format!("{tag}: {}/{}", page_match, ref_lines.len()));
        }
    }

    let text_rate = text_match as f64 / text_total.max(1) as f64;
    let conf_mean: f64 = conf_deltas.iter().sum::<f64>() / conf_deltas.len().max(1) as f64;
    println!(
        "P4 gate 3 (32 pages, LM on): lines {text_total}, text exact match {text_match}/{text_total} = {text_rate:.4} (>= 0.95 required), \
         conf mean |diff| {conf_mean:.5} (< 0.02 required), group diagnostics {group_diff} lines"
    );
    if !bad_pages.is_empty() {
        println!("  differing pages: {}", bad_pages.join(", "));
    }

    assert!(text_rate >= 0.95, "text exact match {text_rate:.4} < 0.95");
    assert!(conf_mean < 0.02, "conf mean |diff| {conf_mean:.5} >= 0.02");
}
