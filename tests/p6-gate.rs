//! P6 gate: replay the product's markdown-assembly stage against the P6
//! references (tests/p1/ref/*.md.json, 32 pages) and check:
//!   A. gutter_cuts — per captured call, the per-line cut lists match.
//!   B. split_line  — per captured call, the segment (box, text) lists match
//!      (boxes bit-exact: both sides do f64 arithmetic on the float32 values).
//!   C. tidy_lines  — per captured call, the out boxes and texts match.
//!   D. page_markdown — per page, the 2-gram Dice overlap with the product's
//!      markdown is >= 0.95 (exact matches are counted in the summary).
//! The table post-filter itself is unit-tested in the lib (markdown tests).

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use docs2md::l1::indra::markdown::{gutter_cuts, page_markdown, split_line, tidy_lines, Cells, Tok};
use docs2md::l1::indra::vocab::{Meta, Vocab};

const REF_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/p1/ref");
const META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/models/indra/meta.json"
));

// ---------- reference schema ----------
// Every captured call is a JSON array (see gen-p6-reference.py).

#[derive(Deserialize)]
struct Page {
    boxes: Vec<[f64; 4]>,
    texts: Vec<String>,
    order: Vec<usize>,
    cls: Vec<Vec<f32>>,
    same: Vec<Vec<f32>>,
    cells: HashMap<String, Vec<Value>>,
    pictures: Option<Vec<Vec<f64>>>,
    #[allow(dead_code)]
    conf: Vec<f64>,
    md: Option<String>,
    // split:  [box4, gap, at|null, toks, [[box4, text], ...]]
    split_calls: Vec<Vec<Value>>,
    // gutter: [boxes, [toks per line], [cuts per line]]
    gutter_calls: Vec<Vec<Value>>,
    // tidy:   [boxes, [[text, conf, toks], ...], rtl, space, [out_boxes, out_rec]]
    tidy_calls: Vec<Vec<Value>>,
}

fn arr<'a>(v: &'a Value, what: &str) -> &'a [Value] {
    v.as_array().unwrap_or_else(|| panic!("{what}: not an array: {v:?}"))
}

fn f4(v: &Value, what: &str) -> [f64; 4] {
    let a = arr(v, what);
    assert_eq!(a.len(), 4, "{what}");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
        a[3].as_f64().unwrap(),
    ]
}

fn fmat(v: &Value, what: &str) -> Vec<[f64; 4]> {
    arr(v, what).iter().map(|r| f4(r, what)).collect()
}

fn toks(v: &Value) -> Vec<Tok> {
    arr(v, "tokens")
        .iter()
        .map(|t| {
            let a = arr(t, "token");
            assert_eq!(a.len(), 3, "token");
            let x = a[2].as_f64().expect("token x");
            match a[1].as_i64() {
                Some(i) => Tok::Id(i as i32, x),
                None => Tok::Ch(a[1].as_str().expect("char token").chars().next().unwrap(), x),
            }
        })
        .collect()
}

// ---------- comparisons ----------

/// 2-gram Dice: 2|A∩B| / (|A|+|B|) over character bigrams (bag of grams).
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

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-9)
}

#[test]
fn p6_gate() {
    let mut ref_dir = PathBuf::from(REF_DIR);
    let pages: Vec<PathBuf> = std::fs::read_dir(&mut ref_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".md.json"))
        .collect();
    assert_eq!(
        pages.len(),
        32,
        "expected 32 P6 references, found {}",
        pages.len()
    );

    let meta: Meta = Meta::parse(META).expect("meta.json");
    let vocab: &Vocab = &meta.vocab["base"];
    let space_id: i32 = 1; // base vocab space id (measured in the references)

    let mut gutter_mm = 0usize;
    let mut split_mm = 0usize;
    let mut tidy_mm = 0usize;
    let mut exact_md = 0usize;
    let mut worst = (1.0f64, String::new());
    let mut dice_sum = 0.0f64;
    let mut md_pages = 0usize;

    for path in &pages {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let page: Page = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        // A. gutter_cuts
        for (ci, call) in page.gutter_calls.iter().enumerate() {
            assert_eq!(call.len(), 3, "{name} gutter[{ci}] arity");
            let b64 = fmat(&call[0], "gutter boxes");
            let rtoks: Vec<Vec<Tok>> = arr(&call[1], "gutter toks").iter().map(toks).collect();
            let want: Vec<Vec<f64>> = arr(&call[2], "gutter cuts")
                .iter()
                .map(|l| arr(l, "gutter cut line").iter().map(|v| v.as_f64().unwrap()).collect())
                .collect();
            let got = gutter_cuts(&b64, &rtoks, space_id);
            if got.len() != want.len() {
                gutter_mm += 1;
                eprintln!(
                    "{name} gutter[{ci}]: {} lines vs {}",
                    got.len(),
                    want.len()
                );
                continue;
            }
            for (li, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                if g.len() != w.len() {
                    gutter_mm += 1;
                    eprintln!(
                        "{name} gutter[{ci}] line {li}: {} cuts vs {} (got {g:?} want {w:?})",
                        g.len(),
                        w.len()
                    );
                    continue;
                }
                for (pi, (gv, wv)) in g.iter().zip(w.iter()).enumerate() {
                    if !close(*gv, *wv, 1e-4) {
                        gutter_mm += 1;
                        eprintln!("{name} gutter[{ci}] line {li} cut {pi}: got {gv} want {wv}");
                    }
                }
            }
        }

        // B. split_line
        for (ci, call) in page.split_calls.iter().enumerate() {
            assert_eq!(call.len(), 5, "{name} split[{ci}] arity");
            let box4 = f4(&call[0], "split box");
            let rt = toks(&call[3]);
            let at: Option<Vec<f64>> = call[2]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect());
            let got = split_line(box4, &rt, call[1].as_f64().unwrap(), at.as_deref(), space_id, vocab, false);
            let wout = arr(&call[4], "split out");
            if got.len() != wout.len() {
                split_mm += 1;
                eprintln!("{name} split[{ci}]: {} segments vs {}", got.len(), wout.len());
                continue;
            }
            for (si, (g, w)) in got.iter().zip(wout.iter()).enumerate() {
                let wseg = arr(w, "segment");
                let wbox = f4(&wseg[0], "segment box");
                let wtext = wseg[1].as_str().expect("segment text");
                for (bi, wb) in wbox.iter().enumerate() {
                    if g.0[bi] != *wb {
                        split_mm += 1;
                        eprintln!("{name} split[{ci}] seg {si} box[{bi}]: got {} want {}", g.0[bi], wb);
                    }
                }
                if g.1 != wtext {
                    split_mm += 1;
                    eprintln!(
                        "{name} split[{ci}] seg {si} text:\n  got  {:?}\n  want {:?}",
                        g.1, wtext
                    );
                }
            }
        }

        // C. tidy_lines
        for (ci, call) in page.tidy_calls.iter().enumerate() {
            assert_eq!(call.len(), 5, "{name} tidy[{ci}] arity");
            let mut boxes = fmat(&call[0], "tidy boxes");
            let mut rec: Vec<(String, f64, Vec<Tok>)> = arr(&call[1], "tidy rec")
                .iter()
                .map(|r| {
                    let r = arr(r, "tidy rec line");
                    (
                        r[0].as_str().unwrap().to_string(),
                        r[1].as_f64().unwrap(),
                        toks(&r[2]),
                    )
                })
                .collect();
            let rtl = call[2].as_bool().unwrap();
            let sp = call[3].as_i64().unwrap() as i32;
            tidy_lines(&mut boxes, &mut rec, rtl, sp);
            let wout = arr(&call[4], "tidy out");
            let wboxes = fmat(&wout[0], "tidy out boxes");
            let wrec = arr(&wout[1], "tidy out rec");
            if boxes.len() != wboxes.len() {
                tidy_mm += 1;
                eprintln!("{name} tidy[{ci}]: {} lines vs {}", boxes.len(), wboxes.len());
                continue;
            }
            for (li, (g, w)) in boxes.iter().zip(wboxes.iter()).enumerate() {
                for (bi, wb) in w.iter().enumerate() {
                    if g[bi] != *wb {
                        tidy_mm += 1;
                        eprintln!("{name} tidy[{ci}] line {li} box[{bi}]: got {} want {}", g[bi], wb);
                    }
                }
            }
            for (li, (g, w)) in rec.iter().zip(wrec.iter()).enumerate() {
                let wtxt = arr(w, "tidy out text")[0].as_str().unwrap();
                if g.0 != wtxt {
                    tidy_mm += 1;
                    eprintln!(
                        "{name} tidy[{ci}] line {li} text:\n  got  {:?}\n  want {:?}",
                        g.0, wtxt
                    );
                }
            }
        }

        // D. page_markdown (no post-filter: table_keep=None keeps every table,
        // exactly the product's page_markdown).
        let Some(want_md) = &page.md else { continue };
        md_pages += 1;
        let cells: Cells = (0..page.boxes.len())
            .map(|i| {
                page.cells
                    .get(&i.to_string())
                    .map(|cs| {
                        cs.iter()
                            .map(|c| {
                                let c = arr(c, "cell");
                                (arr(&c[0], "cell box").iter().map(|v| v.as_f64().unwrap()).collect(), c[1].as_str().unwrap().to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
        let pics: Option<Vec<[f64; 4]>> =
            page.pictures.as_ref().map(|p| p.iter().map(|r| [r[0], r[1], r[2], r[3]]).collect());
        let cls: Vec<f32> = page.cls.iter().flatten().copied().collect();
        let same: Vec<f32> = page.same.iter().flatten().copied().collect();
        let got_md = page_markdown(
            &page.boxes,
            &page.texts,
            &page.order,
            &cls,
            &same,
            false,
            Some(&cells),
            None,
            pics.as_deref(),
            None,
        );
        let d = dice2(&got_md, want_md);
        dice_sum += d;
        if got_md == *want_md {
            exact_md += 1;
        }
        if d < worst.0 {
            worst = (d, name.clone());
        }
        if d < 0.95 {
            eprintln!("{name}: 2-gram dice {d:.4}");
            let cs: Vec<char> = got_md.chars().collect();
            let ws: Vec<char> = want_md.chars().collect();
            let at = cs
                .iter()
                .zip(ws.iter())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| cs.len().min(ws.len()));
            let lo = at.saturating_sub(40);
            let hi_g = (at + 60).min(cs.len());
            let hi_w = (at + 60).min(ws.len());
            eprintln!(
                "  first diff @{}:\n  got  {:?}\n  want {:?}",
                at,
                cs[lo..hi_g].iter().collect::<String>(),
                ws[lo..hi_w].iter().collect::<String>()
            );
        }
    }

    println!(
        "P6 gate: md_pages={} exact_md={} mean_dice={:.4} worst={} ({:.4}) gutter_mm={} split_mm={} tidy_mm={}",
        md_pages,
        exact_md,
        dice_sum / md_pages.max(1) as f64,
        worst.1,
        worst.0,
        gutter_mm,
        split_mm,
        tidy_mm
    );
    assert!(
        gutter_mm == 0 && split_mm == 0 && tidy_mm == 0,
        "line-splitting replays diverge (see stderr)"
    );
    assert!(
        worst.0 >= 0.95,
        "markdown 2-gram dice below 0.95: {} ({:.4})",
        worst.1,
        worst.0
    );
    println!("P6 gate PASS");
}
