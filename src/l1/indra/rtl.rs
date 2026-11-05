//! P7: right-to-left text (port of the product `indra_ocr/rtl.py`).
//!
//! A CTC recognizer reads every line left to right, so an Arabic/Hebrew line
//! comes back in visual order.  `flip` turns logical order into visual order
//! and (being its own inverse) back again:
//!
//! - RTL base line (more RTL letters than LTR): the order of the units is
//!   reversed; a unit is a run of LTR text (kept whole) or one other cluster.
//! - LTR base line: each run of RTL text is reversed in place.
//!
//! A cluster is a base character with its combining marks.  A small subset of
//! UAX #9: no embeddings or isolates, neutrals resolved by runs.

use super::order::char_tables;

/// True when the codepoint is right-to-left strong (bidirectional "R"/"AL").
pub fn is_rtl(c: char) -> bool {
    char_tables::in_runs(c as u32, char_tables::RTL_RUNS)
}

/// True when the codepoint is left-to-right strong ("L"/"EN"/"AN": digits read
/// left to right inside RTL text).
pub fn is_ltr(c: char) -> bool {
    char_tables::in_runs(c as u32, char_tables::LTRD_RUNS)
}

/// True when the codepoint is a left-to-right *letter* (bidirectional "L"
/// exactly; `rtl_base` counts letters only).
pub fn is_ltr_letter(c: char) -> bool {
    char_tables::in_runs(c as u32, char_tables::LONLY_RUNS)
}

/// True when the codepoint is a combining mark (category "Mn"/"Me").
pub fn is_combining(c: char) -> bool {
    char_tables::in_runs(c as u32, char_tables::COMB_RUNS)
}

/// True when the text contains at least one right-to-left character.
pub fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

/// Whether the line reads right to left: more RTL letters than LTR letters
/// (the counts do not depend on the order, so logical and visual agree).
pub fn rtl_base(text: &str) -> bool {
    let r = text.chars().filter(|&c| is_rtl(c)).count();
    let lt = text.chars().filter(|&c| is_ltr_letter(c)).count();
    r > lt
}

/// Presentation forms to base letters (NFKC on those two blocks only), no
/// tatweel, no combining mark without a base character.
pub fn normalize(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        let o = c as u32;
        if (0xFB1D..=0xFDFF).contains(&o) || (0xFE70..=0xFEFF).contains(&o) {
            if let Some(pos) = char_tables::NFKC_MAP
                .iter()
                .position(|(cp, _)| *cp == o)
            {
                out.push_str(char_tables::NFKC_MAP[pos].1);
            }
            continue;
        }
        if c != '\u{0640}' && (!out.is_empty() || !is_combining(c)) {
            out.push(c);
        }
    }
    out
}

/// A base character with its combining marks: each mark joins the previous
/// cluster, so marks stay after their base.
pub fn clusters(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in text.chars() {
        if let Some(last) = out.last_mut() {
            if is_combining(c) {
                last.push(c);
                continue;
            }
        }
        out.push(c.to_string());
    }
    out
}

fn kind(cl: &str) -> char {
    let c = cl.chars().next().unwrap_or('\0');
    if is_rtl(c) {
        'R'
    } else if is_ltr(c) {
        'L'
    } else {
        'N'
    }
}

/// [start, end) spans from a `strong` cluster to the last `strong` cluster
/// before a cluster of a kind in `stop`.
fn runs(kinds: &[char], strong: char, stop: &[char]) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    let mut last: Option<usize> = None;
    for (i, k) in kinds.iter().chain(std::iter::once(&'#')).enumerate() {
        if *k == strong {
            if start.is_none() {
                start = Some(i);
            }
            last = Some(i);
        } else if stop.contains(k) || *k == '#' {
            if let Some(s) = start {
                spans.push((s, last.unwrap() + 1));
            }
            start = None;
            last = None;
        }
    }
    spans
}

const MIRROR: &[(char, char)] = &[
    ('(', ')'),
    (')', '('),
    ('[', ']'),
    (']', '['),
    ('{', '}'),
    ('}', '{'),
    ('<', '>'),
    ('>', '<'),
    ('«', '»'),
    ('»', '«'),
    ('‹', '›'),
    ('›', '‹'),
];

/// A mirrored character (brackets) at right-to-left level changes to its
/// mirror image, as a renderer draws it; the rest of the cluster stays.
fn mirror_cluster(cl: &str) -> String {
    let mut chars = cl.chars();
    let first = chars.next().unwrap_or('\0');
    let rest: String = chars.collect();
    match MIRROR.iter().find(|(f, _)| *f == first) {
        Some((_, m)) => {
            let mut s = String::new();
            s.push(*m);
            s.push_str(&rest);
            s
        }
        None => cl.to_string(),
    }
}

/// Logical order to visual order, and visual order back to logical order.
/// `base_rtl`: the paragraph direction when known (a renderer lays out a line
/// by its paragraph's direction); `None` decides by `rtl_base`.
pub fn flip(text: &str, mirror: bool, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_string();
    }
    let cls = clusters(text);
    let kinds: Vec<char> = cls.iter().map(|c| kind(c)).collect();
    let rtl = base_rtl.unwrap_or_else(|| rtl_base(text));
    if rtl {
        // Units: LTR runs kept whole, every other cluster alone; reverse units.
        let spans = runs(&kinds, 'L', &['R']);
        let mut units: Vec<String> = Vec::new();
        let mut i = 0usize;
        for &(a, b) in spans.iter().chain(std::iter::once(&(cls.len(), cls.len()))) {
            for cl in &cls[i..a] {
                units.push(if mirror {
                    mirror_cluster(cl)
                } else {
                    cl.clone()
                });
            }
            if a < b {
                units.push(cls[a..b].join(""));
            }
            i = b;
        }
        units.into_iter().rev().collect()
    } else {
        // LTR base: reverse each RTL run in place.
        let mut out: Vec<String> = cls.clone();
        for &(a, b) in runs(&kinds, 'R', &['L']).iter() {
            let seg: Vec<String> = cls[a..b].iter().rev().cloned().collect();
            let seg = if mirror {
                seg.into_iter().map(|s| mirror_cluster(&s)).collect()
            } else {
                seg
            };
            out.splice(a..b, seg);
        }
        out.join("")
    }
}

/// The recognizer's label for a line stored in logical order.
pub fn visual_label(logical: &str, base_rtl: Option<bool>) -> String {
    flip(&normalize(logical), true, base_rtl)
}

/// The label for characters already in visual order (a PDF text layer read by
/// glyph position): normalize in logical order, then mirror.  `shapes`: the
/// text layer stores each bracket as the shape drawn, so it is already
/// mirrored.
pub fn label_from_visual(raw_visual: &str, shapes: bool) -> String {
    visual_label(&flip(raw_visual, shapes, None), None)
}

/// Whether a page's text layer stores RTL brackets as drawn shapes.  Most
/// producers store the logical character (the leftmost bracket of an RTL pair
/// reads ")"); some store the drawn shape (it reads "(").  A page uses one
/// convention, so its RTL lines without Latin letters vote by their first
/// bracket.
pub fn brackets_as_shapes(raw_visual_lines: &[String]) -> bool {
    let mut shapes = 0i32;
    let mut logical = 0i32;
    for t in raw_visual_lines {
        if !rtl_base(t) || t.chars().any(is_ltr_letter) {
            continue;
        }
        for c in t.chars() {
            if "([{".contains(c) {
                shapes += 1;
                break;
            }
            if ")]}".contains(c) {
                logical += 1;
                break;
            }
        }
    }
    shapes > logical
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l1::indra::order::{is_py_alpha, is_py_lower};

    #[test]
    fn flip_is_its_own_inverse_rtl_base() {
        // Arabic word with a Latin run and brackets.
        let logical = "مرحبا (test 123) العالم";
        let visual = flip(logical, true, None);
        assert_ne!(visual, logical);
        assert_eq!(flip(&visual, true, None), logical);
    }

    #[test]
    fn flip_ltr_base_reverses_rtl_runs() {
        // LTR base: the RTL run reverses in place, LTR order kept.
        let s = "hello مرحبا world";
        let v = flip(s, true, None);
        // "مرحبا" (5 letters) reversed -> "ابحرم", Latin words untouched.
        assert!(v.starts_with("hello "));
        assert!(v.ends_with(" world"));
        assert_eq!(v, "hello ابحرم world");
        assert_eq!(flip(&v, true, None), s);
    }

    #[test]
    fn mirror_brackets_rtl() {
        // In a flipped RTL line, a logical "(" at that level renders as ")".
        let logical = "(نص)";
        let visual = flip(logical, true, None);
        assert!(visual.contains(")"), "visual: {visual}");
        assert!(visual.contains("("), "visual: {visual}");
        assert_eq!(flip(&visual, true, None), logical);
    }

    #[test]
    fn no_rtl_passes_through() {
        assert_eq!(flip("plain 123", true, None), "plain 123");
    }

    #[test]
    fn normalize_presentation_forms_and_tatweel() {
        // An Arabic ligature presentation form maps to its base letters
        // (0xFEFC -> "لا"); tatweel is dropped; a leading combining mark is
        // dropped.
        let s = "\u{FEFC}\u{0640}\u{0627}ب";
        let n = normalize(s);
        assert!(!n.contains('\u{0640}'));
        assert!(!n.starts_with('\u{0627}'));
        assert!(n.contains("لا"));
        assert!(n.contains('ب'));
    }

    #[test]
    fn clusters_attach_combining_marks() {
        let s = "ب\u{064B}ح"; // ب + fatha + ح
        let c = clusters(s);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], "ب\u{064B}");
        assert_eq!(c[1], "ح");
    }

    #[test]
    fn rtl_base_counts_strong_letters() {
        assert!(rtl_base("مرحبا"));
        assert!(!rtl_base("hello"));
        // Digits are LTR-reading but not LTR letters: 10 digits vs 3 Arabic
        // letters is still RTL base.
        assert!(rtl_base("1234567890123مرحبا"));
    }

    #[test]
    fn visual_label_normalizes_then_flips() {
        // A logical RTL line with a presentation form: normalize maps the form,
        // then flip reverses the units.
        let logical = "أبج\u{0640}د";
        let label = visual_label(logical, None);
        assert!(!label.contains('\u{0640}'));
        assert_ne!(label, logical);
    }

    #[test]
    fn label_from_visual_roundtrip() {
        let logical = "كتاب قديم";
        let visual = visual_label(logical, None);
        // The recognizer emits the visual form; label_from_visual recovers the
        // normalized logical form (no shapes: brackets stored logically).
        assert_eq!(label_from_visual(&visual, false), visual_label(logical, None));
    }

    #[test]
    fn brackets_vote_by_first_bracket() {
        // RTL lines without Latin letters: first bracket "(" votes shapes,
        // ")" votes logical.  (Lines without RTL text do not vote at all.)
        let lines = vec![
            "مرحبا )".to_string(),
            "مرحبا )".to_string(),
            "مرحبا (".to_string(),
        ];
        assert!(!brackets_as_shapes(&lines));
        let lines2 = vec![
            "مرحبا (".to_string(),
            "مرحبا (".to_string(),
            "مرحبا )".to_string(),
        ];
        assert!(brackets_as_shapes(&lines2));
    }

    #[test]
    fn latin_run_kept_whole_inside_rtl() {
        // A Latin run inside an RTL line stays in its own order; only the unit
        // order reverses.
        let logical = "ا abc ب";
        let v = flip(logical, true, None);
        assert!(v.contains("abc"), "the Latin run keeps its order: {v}");
        assert_eq!(flip(&v, true, None), logical);
    }

    #[test]
    fn alpha_lower_tables_sanity() {
        assert!(is_py_alpha('a'));
        assert!(is_py_alpha('\u{4E2D}'));
        assert!(!is_py_alpha('1'));
        assert!(is_py_lower('a'));
        assert!(!is_py_lower('A'));
    }

    // --- P7 cross-language gate: replay the product's rtl.py output on
    // synthetic Arabic/Hebrew lines (tests/p1/ref/rtl-vectors.json, generated
    // by scripts/gen-rtl-vectors.py) and require byte-identical results.  The
    // fixture corpus has no RTL pages, so these vectors are the page-gate
    // equivalent for this module (plan P7).
    #[test]
    fn cross_language_product_vectors() {
        use serde::Deserialize;
        use std::path::Path;
        let raw = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/p1/ref/rtl-vectors.json"),
        )
        .expect("rtl-vectors.json (scripts/gen-rtl-vectors.py)");
        #[derive(Deserialize)]
        struct Case {
            line: String,
            base_rtl: Option<bool>,
            flip: String,
            flip_nomirror: String,
            visual_label: String,
            label_from_visual: String,
            label_from_visual_shapes: String,
            normalize: String,
            clusters: Vec<String>,
            has_rtl: bool,
            rtl_base: bool,
        }
        #[derive(Deserialize)]
        struct Root {
            cases: Vec<Case>,
            brackets: Vec<Br>,
        }
        #[derive(Deserialize)]
        struct Br {
            lines: Vec<String>,
            shapes: bool,
        }
        let root: Root = serde_json::from_str(&raw).unwrap();
        for (i, c) in root.cases.iter().enumerate() {
            let got_has = has_rtl(&c.line);
            let got_base = rtl_base(&c.line);
            let got_flip = flip(&c.line, true, c.base_rtl);
            let got_flip_nm = flip(&c.line, false, c.base_rtl);
            let got_vl = visual_label(&c.line, c.base_rtl);
            let got_lfv = label_from_visual(&c.line, false);
            let got_lfv_s = label_from_visual(&c.line, true);
            let got_norm = normalize(&c.line);
            let got_cls = clusters(&c.line);
            assert_eq!(got_has, c.has_rtl, "has_rtl[{i}]");
            assert_eq!(got_base, c.rtl_base, "rtl_base[{i}]");
            assert_eq!(
                got_flip, c.flip,
                "flip[{i}] line={:?}",
                c.line
            );
            assert_eq!(got_flip_nm, c.flip_nomirror, "flip_nomirror[{i}]");
            assert_eq!(got_vl, c.visual_label, "visual_label[{i}]");
            assert_eq!(got_lfv, c.label_from_visual, "label_from_visual[{i}]");
            assert_eq!(
                got_lfv_s, c.label_from_visual_shapes,
                "label_from_visual_shapes[{i}]"
            );
            assert_eq!(got_norm, c.normalize, "normalize[{i}]");
            assert_eq!(got_cls, c.clusters, "clusters[{i}]");
        }
        for (i, b) in root.brackets.iter().enumerate() {
            assert_eq!(
                brackets_as_shapes(&b.lines),
                b.shapes,
                "brackets_as_shapes[{i}]"
            );
        }
    }
}
