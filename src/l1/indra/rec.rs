//! P3: greedy line recognition (port of pipeline.recognize / line_groups).
//!
//! A page's line boxes are cropped to ink maps (P1 primitives), routed to a
//! recognizer group by script.onnx (`line_groups`), each group's recognizer is
//! run on width-batched padded crops, and the per-timestep argmax output is
//! collapsed (CTC: drop blanks and merge runs) to the line text. The group
//! whose reading has the higher confidence wins a double-read line (the
//! classifier's unsure lines are read by base as well). A line under
//! `MIN_CONF` comes back empty.
//!
//! The recognizer output `ids` is the per-timestep argmax (blanks and repeats
//! included); the CTC collapse happens here, not in the ONNX graph. The
//! language-model beam search (plan P4) is NOT applied in this stage — the P3
//! gate compares the pure greedy reading (the reference generator disables
//! the LM to match).

use crate::l1::indra::lm::beam_search;
use crate::l1::indra::markdown::Tok;
use crate::l1::indra::preproc::{self, Gray, InkMap};
use crate::l1::indra::{IndraEngine, Session};

/// A recognized line (the greedy reading).
#[derive(Debug, Clone)]
pub struct Line {
    pub text: String,
    pub conf: f32,
    /// The group whose reading won ("base" when the classifier is unsure or
    /// the line is base-routed).
    pub group: String,
    /// Page x of each kept CTC step (token mapping for the order stage, P5).
    pub xs: Vec<f64>,
    /// Token ids paired with `xs` (the P6 split/tidy/gutter/markdown stages
    /// re-decode text from the ids, exactly as the product's
    /// `(token id, page x)` rec tokens). Empty for a low-confidence line.
    pub toks: Vec<Tok>,
}

/// Lines under this mean CTC peak probability come back with no text.
pub const MIN_CONF: f32 = 0.5;
/// A line goes to its group's recognizer only when the classifier is surer
/// than this; unsure lines are read by base as well.
pub const ROUTE_SURE: f32 = 0.95;
// Language-model beam search (pipeline constants): a greedy reading whose
// mean peak conf falls in [MIN_CONF, gate) is re-read by the CTC prefix beam;
// meta.json "lm_params" may override a group's alpha/beta/gate.
pub const LM_GATE: f64 = 0.9;
pub const LM_ALPHA: f64 = 0.5;
pub const LM_BETA: f64 = 1.5;
pub const LM_BEAM: usize = 8;
/// A class under this probability at a step is not tried by the beam.
pub const LM_CAND_FLOOR: f64 = 3e-3;
/// Width-batch limits (pipeline.width_batches defaults).
const BATCH_MAX: usize = 64;
const BATCH_RATIO: f64 = 1.3;
const BATCH_MIN_W: i32 = 64;
const BATCH_MAX_AREA: i32 = 64 * 640;

/// Per-line script-group routing (port of pipeline.line_groups): the group
/// whose recognizer reads the line, whether the classifier is sure of it
/// (max probability over `ROUTE_SURE`), and the raw classifier argmax (the
/// script class BEFORE the "no rec variant for it" fallback to base).
/// `names` follows `crops` order. A group without its recognizer is read by
/// base.
pub fn line_groups(
    script: &Session,
    crops: &[InkMap],
    script_groups: &[String],
    group_names: &[&str],
) -> (Vec<String>, Vec<bool>, Vec<String>) {
    let n = crops.len();
    let mut out = vec!["base".to_string(); n];
    let mut sure = vec![true; n];
    let mut raw = vec!["base".to_string(); n];
    if n == 0 {
        return (out, sure, raw);
    }
    let widths: Vec<i32> = crops.iter().map(|c| c.w as i32).collect();
    for part in preproc::width_batches(&widths, BATCH_MAX, BATCH_RATIO, BATCH_MIN_W, BATCH_MAX_AREA)
    {
        let idx: Vec<usize> = part;
        let batch: Vec<&InkMap> = idx.iter().map(|&i| &crops[i]).collect();
        let (arr, widths_in) = preproc::pad_batch(&batch, preproc::WIDTH_MULTIPLE);
        let h = batch[0].h;
        let wm = arr.len() / (idx.len() * h);
        // rten rejects script-model inputs narrower than 32 px (a MaxPool
        // layer sees W=1 and errors), where ORT runs them fine. Such narrow
        // lines are exactly the ones the classifier can't be sure of (a 22px
        // crop carries no script signal), and unsure lines are read by the
        // base recognizer from the tall crops, which accept any width — the
        // same path Python's own routing takes for them. So a failed run
        // degrades the whole batch to base/unsure instead of panicking.
        let o = match script.run_fi(
            "x",
            &[idx.len(), 1, h, wm],
            &arr,
            "widths",
            &[idx.len()],
            &widths_in,
        ) {
            Ok(o) => o,
            Err(_) => {
                for &j in &idx {
                    sure[j] = false;
                }
                continue;
            }
        };
        let p = o[0].as_float().expect("script probs");
        for (k, &j) in idx.iter().enumerate() {
            let row = &p[k * 4..k * 4 + 4];
            let (ai, pk) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            let g = script_groups.get(ai).map(|s| s.as_str()).unwrap_or("base");
            raw[j] = g.to_string();
            out[j] = if group_names.contains(&g) {
                g.to_string()
            } else {
                "base".to_string()
            };
            sure[j] = *pk > ROUTE_SURE;
        }
    }
    (out, sure, raw)
}

/// The script pre-pass (plan §2.1): the line crops (window-1 for the group
/// recognizers + the script classifier, window-`rec_window` for the base
/// recognizer) plus the per-line script routing, computed ONCE per page. It
/// is the pre-pass `recognize_crops` used to run internally; lifting it out
/// lets the L1→L2 chain evaluate the script-stage criteria (cjk / unsure /
/// no-rec-variant line fractions) BEFORE any recognition, so a page that
/// escalates at the script stage saves the whole recognition pass.
pub struct RouteScript {
    /// Window-1 crops (group recognizers + the script classifier).
    pub crops: Vec<InkMap>,
    /// Window-`rec_window` crops (the base recognizer); the window-1 crops
    /// when `rec_window == 1`.
    pub tall: Vec<InkMap>,
    /// The page is a vertical one turned a quarter turn (every line is CJK).
    pub vertical: bool,
    /// Per-line routed group (base fallback already applied; the vertical
    /// page is all "cjk").
    pub names: Vec<String>,
    /// Per-line classifier-sure flag (vertical page: all sure).
    pub sure: Vec<bool>,
    /// Per-line raw classifier argmax (before the base fallback): the
    /// "no rec variant" chain criterion reads this.
    pub raw: Vec<String>,
}

impl RouteScript {
    /// The chain's [script]-stage metrics (plan §2.1), per-line counts over
    /// the RAW classifier argmax (before the base fallback — the chain asks
    /// "what script does this page have", not "which recognizer runs").
    /// `loaded`: the script groups this build has a rec variant for
    /// (`engine.rec.keys()`); base always counts as loaded.
    pub fn metrics(&self, loaded: &[&str]) -> crate::route::ScriptMetrics {
        let n = self.raw.len();
        let cjk = self.raw.iter().filter(|g| g.as_str() == "cjk").count();
        let unsure = self.sure.iter().filter(|&s| !s).count();
        // dominant: mode of the raw argmax (BTreeMap key order gives a
        // deterministic alphabetical tie-break).
        let mut dominant: Option<String> = None;
        if n > 0 {
            let mut counts: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for g in &self.raw {
                *counts.entry(g.as_str()).or_insert(0) += 1;
            }
            dominant = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(g, _)| g.to_string());
        }
        let dominant_has_variant = dominant
            .as_deref()
            .map(|d| d == "base" || loaded.contains(&d))
            .unwrap_or(true);
        crate::route::ScriptMetrics {
            n,
            cjk,
            unsure,
            dominant,
            dominant_has_variant,
        }
    }
}

/// Run the script pre-pass for one page (crop + route). See [`RouteScript`].
pub fn route_script(
    engine: &IndraEngine,
    gray: &Gray,
    boxes: &[[f64; 4]],
    vertical: bool,
) -> RouteScript {
    let n = boxes.len();
    let height = engine.meta.rec_height as usize;
    let window = engine.meta.rec_window.max(1) as usize;
    let dark: Vec<Option<bool>> = boxes
        .iter()
        .map(|b| Some(preproc::is_dark_line(gray, *b)))
        .collect();
    let crop = |win: usize| -> Vec<InkMap> {
        boxes
            .iter()
            .enumerate()
            .map(|(i, b)| {
                preproc::crop_line(
                    gray,
                    *b,
                    height,
                    preproc::PAD_Y,
                    preproc::PAD_X,
                    4096,
                    win,
                    dark[i],
                )
            })
            .collect()
    };
    let crops = crop(1);
    let tall = if window > 1 {
        crop(window)
    } else {
        crops.clone()
    };
    let group_names: Vec<&str> = engine
        .meta
        .script_groups
        .iter()
        .map(|s| s.as_str())
        .filter(|g| *g != "base" && engine.rec.contains_key(*g))
        .collect();
    let (names, sure, raw) = if n == 0 {
        (vec![], vec![], vec![])
    } else if vertical && group_names.contains(&"cjk") {
        // A vertical page's lines are CJK columns (characters on their
        // side): the CJK recognizer reads all of them. The chain sees this
        // as a 100 % cjk page (plan §2.1 vertical boundary).
        (
            vec!["cjk".to_string(); n],
            vec![true; n],
            vec!["cjk".to_string(); n],
        )
    } else {
        line_groups(
            &engine.script,
            &crops,
            &engine.meta.script_groups,
            &group_names,
        )
    };
    RouteScript {
        crops,
        tall,
        vertical,
        names,
        sure,
        raw,
    }
}

/// Greedy recognition of a page's pre-routed lines (port of the
/// pipeline.recognize core): `rs` is the [`route_script`] pre-pass output,
/// `boxes` the line boxes in page pixels (for the token x mapping). Returns
/// one `Line` per crop.
pub fn recognize_groups(engine: &IndraEngine, rs: &RouteScript, boxes: &[[f64; 4]]) -> Vec<Line> {
    let crops = &rs.crops;
    let tall = &rs.tall;
    let n = crops.len();
    let mut out: Vec<Option<Line>> = vec![None; n];
    let mut best_conf = vec![-1.0f32; n];
    if n == 0 {
        return vec![];
    }
    // The group recognizers (everything but base, and only the ones this
    // build actually has loaded — a group without its recognizer must not be
    // scheduled, its lines are already base-routed).
    let group_names: Vec<&str> = rs
        .names
        .iter()
        .filter(|g| g.as_str() != "base" && engine.rec.contains_key(g.as_str()))
        .fold(Vec::new(), |mut v, g| {
            if !v.contains(&g.as_str()) {
                v.push(g.as_str());
            }
            v
        });

    // A vertical page's lines are CJK columns (characters on their side); the
    // CJK recognizer reads all of them (recognize: vertical branch). Otherwise
    // the script pre-pass routes each line.
    let groups: Vec<(&str, Vec<usize>, bool)> = if rs.vertical && group_names.contains(&"cjk") {
        vec![("cjk", (0..n).collect(), false)]
    } else {
        // Grouping exactly as recognize(): base reads its own group's
        // lines plus every line the pre-pass marked UNSURE (read from
        // the tall crops); each group recognizer reads only the lines
        // routed to it (window-1).
        let base_sel: Vec<usize> = (0..n)
            .filter(|&j| rs.names[j] == "base" || !rs.sure[j])
            .collect();
        let mut g = vec![("base", base_sel, true)];
        for gn in &group_names {
            let sel: Vec<usize> = (0..n).filter(|&j| rs.names[j] == *gn).collect();
            g.push((gn, sel, false));
        }
        g
    };

    for (gname, sel, use_tall) in groups {
        if sel.is_empty() {
            continue;
        }
        let sess = engine.rec.get(gname).unwrap();
        let vocab = &engine.meta.vocab[gname];
        // The group's language model, if its recognizer has an lm.npz (the
        // pipeline builds lms only for files that exist; rtl has none).
        let lm_params = engine.meta.lm_params.get(gname);
        let lm = engine.lms.get(gname);
        // Gate/alpha/beta stay f64 end to end (Python compares the f32-derived
        // conf as a Python float against the JSON f64 gate).
        let (alpha, beta, gate) = match lm_params {
            Some(p) => (p.alpha, p.beta, p.gate),
            None => (LM_ALPHA, LM_BETA, LM_GATE),
        };
        // Group the batch by the window-1 crop width (recognize:
        // width_batches over crops[j].shape[1]), then pad the SOURCE crop
        // (tall for the base recognizer, the window-1 crop for the groups).
        let widths: Vec<i32> = sel.iter().map(|&j| crops[j].w as i32).collect();
        for part in
            preproc::width_batches(&widths, BATCH_MAX, BATCH_RATIO, BATCH_MIN_W, BATCH_MAX_AREA)
        {
            let idx: Vec<usize> = part; // indices into sel
            let batch: Vec<&InkMap> = if use_tall {
                idx.iter().map(|&i| &tall[sel[i]]).collect()
            } else {
                idx.iter().map(|&i| &crops[sel[i]]).collect()
            };
            let (arr, widths_in) = preproc::pad_batch(&batch, preproc::WIDTH_MULTIPLE);
            let h = batch[0].h;
            let wm = arr.len() / (idx.len() * h);
            let o = sess
                .run_fi(
                    "x",
                    &[idx.len(), 1, h, wm],
                    &arr,
                    "widths",
                    &[idx.len()],
                    &widths_in,
                )
                .unwrap_or_else(|e| panic!("rec {gname}: {e}"));
            // outputs: ids, conf, lengths, blank, topv, topi (the beam uses 3-5).
            let ids = o[0].as_int().expect("rec ids");
            let conf = o[1].as_float().expect("rec conf");
            let lengths = o[2].as_int().expect("rec lengths");
            let t = ids.len() / idx.len();
            let stride = wm as f64 / (t as f64).max(1.0);
            let blank_t = o[3].as_float();
            let topv_t = o[4].as_float();
            let topi_t = o[5].as_int();
            let kt = topv_t
                .and_then(|v| (idx.len() * t != 0).then(|| v.len() / (idx.len() * t)))
                .unwrap_or(0);
            for (k, &si) in idx.iter().enumerate() {
                let j = sel[si];
                let n_len = lengths[k] as usize;
                let seq = &ids[k * t..k * t + n_len];
                let pm = &conf[k * t..k * t + n_len];
                // CTC collapse: keep = (seq != 0) & (first | seq != prev raw).
                let mut kept: Vec<usize> = Vec::new();
                for tt in 0..n_len {
                    let v = seq[tt];
                    let differs = tt == 0 || v != seq[tt - 1];
                    if v != 0 && differs {
                        kept.push(tt);
                    }
                }
                let confv = if kept.is_empty() {
                    0.0f32
                } else {
                    let mut s = 0.0f32;
                    for &tt in &kept {
                        s += pm[tt];
                    }
                    s / kept.len() as f32
                };
                let conf64 = f64::from(confv);
                let mut ids_kept: Vec<i32> = kept.iter().map(|&tt| seq[tt]).collect();
                let mut kept_steps: Vec<usize> = kept.clone();
                // A greedy reading under the group's gate is re-read by the
                // CTC prefix beam (recognize: ids/steps replaced, conf kept).
                if let Some(lm) = lm {
                    if MIN_CONF as f64 <= conf64 && conf64 < gate {
                        let base = k * t;
                        let blank: Vec<f64> = blank_t
                            .map(|v| v[base..base + n_len].iter().map(|&x| x as f64).collect())
                            .unwrap_or_default();
                        let topv: Vec<f64> = topv_t
                            .map(|v| {
                                v[base * kt..base * kt + n_len * kt]
                                    .iter()
                                    .map(|&x| x as f64)
                                    .collect()
                            })
                            .unwrap_or_default();
                        let topi: Vec<i64> = topi_t
                            .map(|v| {
                                v[base * kt..base * kt + n_len * kt]
                                    .iter()
                                    .map(|&x| x as i64)
                                    .collect()
                            })
                            .unwrap_or_default();
                        let (b_ids, b_steps) = beam_search(
                            &blank,
                            &topv,
                            &topi,
                            lm,
                            alpha,
                            beta,
                            LM_BEAM,
                            LM_CAND_FLOOR,
                        );
                        ids_kept = b_ids.iter().map(|&x| x as i32).collect();
                        kept_steps = b_steps;
                    }
                }
                if confv <= best_conf[j] {
                    continue;
                }
                best_conf[j] = confv;
                // Token x: the CTC peak's timestep mapped back through the crop.
                let [x0, y0, x1, y1] = boxes[j];
                let h = (y1 - y0).max(1.0);
                let ix0 = (x0 - 0.3 * h).floor();
                let cw = (x1 + 0.3 * h).ceil() - ix0;
                let f = cw / (crops[j].w as f64);
                if confv < MIN_CONF {
                    // Low-confidence read: the product stores no text and no
                    // token positions (recognize: ("", conf, [])).
                    out[j] = Some(Line {
                        text: String::new(),
                        conf: confv,
                        group: gname.to_string(),
                        xs: vec![],
                        toks: vec![],
                    });
                    continue;
                }
                let xs: Vec<f64> = kept_steps
                    .iter()
                    .map(|&tt| ix0 + (tt as f64 + 0.5) * stride * f)
                    .collect();
                let text = vocab.decode_seq(&ids_kept);
                // Token kind follows the product's `as_chars`: base reads token
                // IDs (re-decoded through the base vocab by the P6 stages); a
                // group recognizer's ids mean nothing in the base vocab, so its
                // tokens are the group-vocab CHARACTERS (decode_tokens joins
                // them, NFC + RTL flip).
                let as_chars = !use_tall;
                let toks: Vec<Tok> = ids_kept
                    .iter()
                    .zip(xs.iter())
                    .map(|(&id, &x)| {
                        if as_chars {
                            Tok::Ch(
                                vocab
                                    .chars
                                    .get((id - 1) as usize)
                                    .and_then(|s| s.chars().next())
                                    .unwrap_or('\0'),
                                x,
                            )
                        } else {
                            Tok::Id(id, x)
                        }
                    })
                    .collect();
                out[j] = Some(Line {
                    text,
                    conf: confv,
                    group: gname.to_string(),
                    xs,
                    toks,
                });
            }
        }
    }
    out.into_iter()
        .map(|l| {
            l.unwrap_or(Line {
                text: String::new(),
                conf: 0.0,
                group: "base".to_string(),
                xs: vec![],
                toks: vec![],
            })
        })
        .collect()
}

/// Product path (port of pipeline.recognize): the script pre-pass
/// ([`route_script`]: crop each line box from the page — window-1 for the
/// groups, window-`rec_window` for base, dark-on-light detected once per box
/// — plus the per-line script routing), then the greedy reading.
/// `vertical`: the page is a vertical one turned a quarter turn (every line
/// to the CJK recognizer).
pub fn recognize(
    engine: &IndraEngine,
    gray: &Gray,
    boxes: &[[f64; 4]],
    vertical: bool,
) -> Vec<Line> {
    if boxes.is_empty() {
        return vec![];
    }
    let rs = route_script(engine, gray, boxes, vertical);
    recognize_groups(engine, &rs, boxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> IndraEngine {
        IndraEngine::init().expect("indra init")
    }

    #[test]
    fn line_groups_routes_by_argmax() {
        let e = engine();
        // A 32x64 all-ink crop and an all-paper crop; the classifier must return
        // one of the four groups per line and a sane sure flag.
        let ink = InkMap {
            w: 64,
            h: 32,
            data: vec![1.0f32; 2048],
        };
        let paper = InkMap {
            w: 64,
            h: 32,
            data: vec![0.0f32; 2048],
        };
        let (names, sure, raw) = line_groups(
            &e.script,
            &[ink, paper],
            &e.meta.script_groups,
            &["cjk", "rtl", "world"],
        );
        assert_eq!(names.len(), 2);
        assert_eq!(sure.len(), 2);
        assert_eq!(raw.len(), 2);
        for g in &names {
            assert!(
                g == "base" || g == "cjk" || g == "rtl" || g == "world",
                "unexpected group {g}"
            );
        }
    }

    #[test]
    fn line_groups_degrades_narrow_crops_to_base_unsure() {
        // A crop narrower than 32 px (the script model's rten minimum) must
        // not panic: run_fi errors and the line degrades to base/unsure, which
        // the base recognizer then reads from the tall crops (any width).
        let e = engine();
        let narrow = InkMap {
            w: 24,
            h: 32,
            data: vec![1.0f32; 24 * 32],
        };
        let (names, sure, _raw) = line_groups(
            &e.script,
            &[narrow],
            &e.meta.script_groups,
            &["cjk", "rtl", "world"],
        );
        assert_eq!(names, ["base"]);
        assert_eq!(sure, [false]);
    }

    #[test]
    fn recognize_groups_gives_one_line_per_crop() {
        let e = engine();
        // Two 32x64 paper lines: the recognizer reads nothing of confidence, so
        // both lines come back empty (conf < MIN_CONF) but present.
        let paper = InkMap {
            w: 64,
            h: 32,
            data: vec![0.0f32; 2048],
        };
        let tall = InkMap {
            w: 64,
            h: 96,
            data: vec![0.0f32; 6144],
        };
        let rs = RouteScript {
            crops: vec![paper.clone(), paper.clone()],
            tall: vec![tall.clone(), tall.clone()],
            vertical: false,
            names: vec!["base".to_string(), "base".to_string()],
            sure: vec![true, true],
            raw: vec!["base".to_string(), "base".to_string()],
        };
        let boxes: Vec<[f64; 4]> = vec![[10.0, 10.0, 100.0, 20.0], [10.0, 40.0, 100.0, 50.0]];
        let lines = recognize_groups(&e, &rs, &boxes);
        assert_eq!(lines.len(), 2);
        for l in &lines {
            assert_eq!(l.text, "", "paper line must read empty");
            assert!(l.conf < MIN_CONF);
        }
    }

    #[test]
    fn recognize_product_path_runs() {
        let e = engine();
        // A 64x64 page with one dark horizontal bar (a crude "line"); the
        // product path must return one Line without panicking.
        let mut g = Gray {
            w: 64,
            h: 64,
            data: vec![255u8; 4096],
        };
        for x in 0..64 {
            g.data[20 * 64 + x] = 0;
            g.data[21 * 64 + x] = 0;
        }
        let boxes = [[4.0, 16.0, 60.0, 24.0]];
        let lines = recognize(&e, &g, &boxes, false);
        assert_eq!(lines.len(), 1);
    }
}
