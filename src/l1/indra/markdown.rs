//! P6: page markdown assembly (port of the product `layout_post.py`) plus the
//! line splitting of `pipeline.py` (`gutter_cuts` / `split_line` /
//! `tidy_lines`) and the table post-filter (new logic, plan §1.5: a Table
//! block keeps its `<table>` only when its bbox covers less than half the
//! page and its lines read with mean confidence above 0.4, or when a
//! table-caption region exists).
//!
//! Numpy dtype discipline (as in order.rs): the model's boxes/logits are f32,
//! so every array computation runs in f32 and widens to f64 exactly where the
//! product calls `float()`.

use crate::l1::indra::order::{
    cluster, is_py_alpha, is_py_digit, is_py_lower, median_f32, py_round, N_BLOCK_CLASSES,
};
use crate::l1::indra::rtl::{flip, has_rtl};
use crate::l1::indra::vocab::{nfc, Vocab};

/// Block class indices (order of `BLOCK_CLASSES`).
pub const HEADER: usize = 2; // Page-Header
pub const FOOTER: usize = 3; // Page-Footer
pub const SECTION: usize = 1; // Section-Header
pub const LIST: usize = 4; // List-Group
pub const CAPTION: usize = 6; // Caption (a table-caption region keeps the table)
pub const TABLE: usize = 5; // Table

/// Compact-region thresholds: a compact, confident region is the condition
/// under which a table is trusted (kept as `<table>` in the post-filter).
pub const TABLE_COMPACT_FRAC: f64 = 0.5;
pub const TABLE_COMPACT_CONF: f64 = 0.4;

pub const SPLIT_GAP: f64 = 3.0; // cut a detected line at a wider gap (label rule outside tables)
pub const CELL_GAP: f64 = 1.8; // cut a table line into cells at a wider gap (label rule in tables)
const GUTTER_GAP: f64 = 1.6; // a gap that sits on a column gutter
const GUTTER_EDGES: usize = 3; // other line edges that must line up with it
const MARKER_GAP: f64 = 2.5; // a list marker and its item's text, in line heights

const LEADER_CHARS: &[char] = &['.', '·', '…', '‥', '_', '-', '–', '—', '='];
const DASHES: &[char] = &['-', '–', '—'];
const BULLETS: &[char] = &['•', '◦', '▪', '▫', '●', '○', '■', '□', '‣', '⁃', '∙', '*'];
const ITEM_GLYPHS: &[char] = &['•', '-', '–', '▪', '◦', '*', '·', '→'];

/// Han, kana, CJK punctuation and fullwidth forms: written without spaces.
fn cjk(c: char) -> bool {
    let o = c as u32;
    (0x3000..=0x30FF).contains(&o)
        || (0x3400..=0x9FFF).contains(&o)
        || (0xF900..=0xFAFF).contains(&o)
        || (0xFF00..=0xFFEF).contains(&o)
        || (0x20000..=0x2FA1F).contains(&o)
}

/// One formula: `$...$` with no `$` inside (the line is already stripped).
fn whole_math(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    c.len() >= 3
        && c[0] == '$'
        && c[c.len() - 1] == '$'
        && c[1..c.len() - 1].iter().all(|&x| x != '$')
}

/// A list marker at the start of a line: a bullet/dash glyph, "a." / "(a)",
/// "(12)." / "(12)" — followed by whitespace or end of line.
/// (Product `_ITEM` = `^\s*([•\-–▪◦*·→]|\(?[A-Za-z][.)]|\(\d{1,3}[.)])(\s|$)`;
/// `\d` is a Python Unicode decimal digit, i.e. `is_py_digit`.)
fn item_match(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    let n = c.len();
    let mut i = 0;
    while i < n && c[i].is_whitespace() {
        i += 1;
    }
    // Alt 1: one marker glyph.
    if i < n && ITEM_GLYPHS.contains(&c[i]) {
        let after = i + 1;
        if after == n || c[after].is_whitespace() {
            return true;
        }
    }
    // Alt 2: optional '(', one ASCII letter, '.' or ')'.
    {
        let mut j = i;
        if j < n && c[j] == '(' {
            j += 1;
        }
        if j + 2 <= n && c[j].is_ascii_alphabetic() && (c[j + 1] == '.' || c[j + 1] == ')') {
            let after = j + 2;
            if after == n || c[after].is_whitespace() {
                return true;
            }
        }
    }
    // Alt 3: optional '(', 1-3 digits, '.' or ')'.
    {
        let mut j = i;
        if j < n && c[j] == '(' {
            j += 1;
        }
        for dlen in 1..=3usize {
            let end = j + dlen;
            if end + 1 > n {
                break;
            }
            let digits = c[j..end].iter().all(|&x| is_py_digit(x));
            let close = c[end] == '.' || c[end] == ')';
            if digits && close {
                let after = end + 1;
                return after == n || c[after].is_whitespace();
            }
        }
    }
    false
}

/// A numbered list marker (the whole line): "1." / "12)" / "a." / "(3)" /
/// "(b)" / "(iv)".
/// (Product `ENUMERATOR` = `(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\)`,
/// fullmatch.)
fn enumerator_fullmatch(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    if c.len() >= 2 && (c[c.len() - 1] == '.' || c[c.len() - 1] == ')') {
        let pre = &c[..c.len() - 1];
        if !pre.is_empty() && pre.len() <= 3 {
            if pre.iter().all(|&x| is_py_digit(x)) {
                return true;
            }
            if pre.len() == 1 && pre[0].is_ascii_alphabetic() {
                return true;
            }
        }
    }
    if c.len() >= 3 && c[0] == '(' && c[c.len() - 1] == ')' {
        let mid = &c[1..c.len() - 1];
        if !mid.is_empty() {
            // 1-3 digits.
            if mid.len() <= 3 && mid.iter().all(|&x| is_py_digit(x)) {
                return true;
            }
            // 1 letter.
            if mid.len() == 1 && mid[0].is_ascii_alphabetic() {
                return true;
            }
            // 1-5 roman.
            if mid.len() <= 5 && mid.iter().all(|&x| "ivxlcIVXLC".chars().any(|m| m == x)) {
                return true;
            }
        }
    }
    false
}

/// Join the lines of one paragraph: 'tumori-' + 'genesis' -> 'tumorigenesis';
/// a line break between two CJK characters takes no space.
pub fn join_lines(texts: &[String]) -> String {
    let mut out = String::new();
    for t in texts {
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        let tc: Vec<char> = t.chars().collect();
        let oc: Vec<char> = out.chars().collect();
        if oc.len() > 1
            && oc[oc.len() - 1] == '-'
            && is_py_alpha(oc[oc.len() - 2])
            && is_py_alpha(tc[0])
            && is_py_lower(tc[0])
        {
            out.pop(); // drop the '-'
            out.push_str(t);
        } else if out.is_empty() || (cjk(oc[oc.len() - 1]) && cjk(tc[0])) {
            out.push_str(t);
        } else {
            out.push(' ');
            out.push_str(t);
        }
    }
    out
}

/// Runs of the reading order with a positive same-block logit between neighbours.
pub fn split_blocks(order: &[usize], same: &[f32]) -> Vec<Vec<usize>> {
    let n = order.len();
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for &i in order {
        if let Some(last) = blocks.last_mut() {
            let j = last[last.len() - 1];
            if same[j * n + i] > 0.0 {
                last.push(i);
                continue;
            }
        }
        blocks.push(vec![i]);
    }
    blocks
}

/// Group cell lines into rows by vertical centre, top to bottom, left to right.
fn rows(boxes: &[[f32; 4]]) -> Vec<Vec<usize>> {
    let n = boxes.len();
    let cy: Vec<f32> = (0..n)
        .map(|i| (boxes[i][1] + boxes[i][3]) / 2.0f32)
        .collect();
    let h = median_f32(
        &(0..n)
            .map(|i| boxes[i][3] - boxes[i][1])
            .collect::<Vec<_>>(),
    );
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| cy[i].total_cmp(&cy[j]).then_with(|| i.cmp(&j)));
    for &i in &order {
        if let Some(last) = rows.last() {
            let mean: f32 = last.iter().map(|&k| cy[k]).sum::<f32>() / last.len() as f32;
            if ((cy[i] - mean).abs() as f64) < 0.5 * h {
                rows.last_mut().unwrap().push(i);
                continue;
            }
        }
        rows.push(vec![i]);
    }
    for r in rows.iter_mut() {
        r.sort_by(|&a, &b| boxes[a][0].total_cmp(&boxes[b][0]).then_with(|| a.cmp(&b)));
    }
    rows
}

/// Join the line rows of wrapped cells into one table row.
fn merge_wrapped(rows: &[Vec<usize>], boxes: &[[f32; 4]]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows.to_vec();
    }
    let h = median_f32(
        &(0..boxes.len())
            .map(|i| boxes[i][3] - boxes[i][1])
            .collect::<Vec<_>>(),
    );
    let gaps: Vec<f64> = (0..rows.len() - 1)
        .map(|r| {
            let y_min_next = rows[r + 1]
                .iter()
                .map(|&k| boxes[k][1])
                .fold(f32::INFINITY, |a, b| a.min(b));
            let y_max_cur = rows[r]
                .iter()
                .map(|&k| boxes[k][3])
                .fold(0.0f32, |a, b| a.max(b));
            (y_min_next - y_max_cur) as f64
        })
        .collect();
    let med = crate::l1::indra::order::median_f64(&gaps);
    let mut out: Vec<Vec<usize>> = vec![rows[0].to_vec()];
    let mut prev = &rows[0];
    for (g, r) in gaps.iter().zip(rows.iter().skip(1)) {
        // A continuation fills only the wrapped columns: fewer lines than above.
        if *g < 0.5 * med && *g < 0.6 * h && r.len() < prev.len() {
            out.last_mut().unwrap().extend_from_slice(r);
        } else {
            out.push(r.to_vec());
        }
        prev = r;
    }
    out
}

/// Python `html.escape(text, quote=False)`: &, <, > (quotes stay).
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// An HTML table from cell lines (one text line per cell).
/// A first row that is one line across most of the table is a title: it goes
/// before the table as text. The first remaining row is the header (<th>).
pub fn table_html(boxes: &[[f32; 4]], texts: &[String]) -> String {
    let rows_ = rows(boxes);
    if rows_.is_empty() {
        return "<table>\n</table>".to_string();
    }
    let span_x: f32 = boxes
        .iter()
        .map(|b| b[2])
        .fold(f32::NEG_INFINITY, |a, b| a.max(b))
        - boxes
            .iter()
            .map(|b| b[0])
            .fold(f32::INFINITY, |a, b| a.min(b));
    if rows_.len() > 2
        && rows_[0].len() == 1
        && rows_.iter().map(|r| r.len()).max().unwrap() > 1
        && (boxes[rows_[0][0]][2] - boxes[rows_[0][0]][0]) as f64 > 0.6 * span_x as f64
    {
        let keep: Vec<usize> = (0..texts.len()).filter(|&k| k != rows_[0][0]).collect();
        let b2: Vec<[f32; 4]> = keep.iter().map(|&k| boxes[k]).collect();
        let t2: Vec<String> = keep.iter().map(|&k| texts[k].clone()).collect();
        let mut out = texts[rows_[0][0]].clone();
        out.push_str("\n\n");
        out.push_str(&table_html(&b2, &t2));
        return out;
    }
    let width = rows_.iter().map(|r| r.len()).max().unwrap();
    let full: Vec<&Vec<usize>> = rows_.iter().filter(|r| r.len() == width).collect();
    let rows_ = merge_wrapped(&rows_, boxes);
    // Column bands from the rows with the most cells: those rows rarely span.
    let mut bands: Vec<[f32; 2]> = Vec::with_capacity(width);
    for c in 0..width {
        let x0: Vec<f32> = full.iter().map(|r| boxes[r[c]][0]).collect();
        let x1: Vec<f32> = full.iter().map(|r| boxes[r[c]][2]).collect();
        bands.push([median_f32(&x0) as f32, median_f32(&x1) as f32]);
    }
    // Widen each band to the midpoints between neighbours.
    let mut edges: Vec<f32> = vec![f32::NEG_INFINITY];
    for c in 0..width.saturating_sub(1) {
        edges.push((bands[c][1] + bands[c + 1][0]) / 2.0f32);
    }
    edges.push(f32::INFINITY);
    // np.searchsorted(edges, x, side="right") - 1
    let search = |edges: &[f32], x: f32| -> usize {
        edges.iter().position(|&e| e > x).unwrap_or(edges.len()) - 1
    };
    let mut out = vec!["<table>".to_string()];
    for (ri, r) in rows_.iter().enumerate() {
        let tag = if ri == 0 && rows_.len() > 1 {
            "th"
        } else {
            "td"
        };
        let mut cells: std::collections::BTreeMap<usize, (usize, String)> = Default::default();
        let mut order: Vec<usize> = r.clone();
        order.sort_by(|&a, &b| {
            boxes[a][1]
                .total_cmp(&boxes[b][1])
                .then_with(|| boxes[a][0].total_cmp(&boxes[b][0]))
                .then_with(|| a.cmp(&b))
        });
        for k in order {
            let x0 = boxes[k][0];
            let x1 = boxes[k][2];
            let first = search(&edges, x0);
            let last = search(&edges, x1);
            // A cell spans every band it covers by more than half of the band.
            let mut span: Vec<usize> = (first..=last.min(width - 1))
                .filter(|&c| {
                    (x1.min(bands[c][1]) - x0.max(bands[c][0])) as f64
                        > 0.5 * (bands[c][1] - bands[c][0]) as f64
                })
                .collect();
            if span.is_empty() {
                span = vec![first.min(width - 1)];
            }
            let c0 = span[0];
            let text = match cells.get(&c0) {
                Some((_, prev)) => format!("{} {}", prev, texts[k]),
                None => texts[k].clone(),
            };
            let n_ = match cells.get(&c0) {
                Some((p, _)) => (*p).max(span.len()),
                None => span.len(),
            };
            cells.insert(c0, (n_, text));
        }
        let mut tds = Vec::new();
        let mut c = 0usize;
        while c < width {
            if let Some(&(n, ref text)) = cells.get(&c) {
                let attr = if n > 1 {
                    format!(" colspan=\"{n}\"")
                } else {
                    String::new()
                };
                tds.push(format!("<{tag}{attr}>{}</{tag}>", escape_html(text)));
                c += n;
            } else {
                tds.push(format!("<{tag}></{tag}>"));
                c += 1;
            }
        }
        out.push(format!("<tr>{}</tr>", tds.concat()));
    }
    out.push("</table>".to_string());
    out.join("\n")
}

/// The lines of one cell are stacked (they overlap horizontally).  When the
/// cell head joins side-by-side lines in more than 10% of its cells, it has
/// failed on this table and the geometric assembly is used instead.
pub fn heads_plausible(boxes: &[[f32; 4]], tab_cell: &[f32]) -> bool {
    let m = (tab_cell.len() as f64).sqrt() as usize;
    let cells = cluster(tab_cell, m);
    let mut bad = 0usize;
    for c in &cells {
        if c.len() < 2 {
            continue;
        }
        let x0: Vec<f32> = c.iter().map(|&k| boxes[k][0]).collect();
        let x1: Vec<f32> = c.iter().map(|&k| boxes[k][2]).collect();
        let min_overlap = (0..c.len())
            .map(|i| {
                (i + 1..c.len())
                    .map(|j| (x1[i].min(x1[j]) - x0[i].max(x0[j])) as f64)
                    .fold(f64::INFINITY, |a, b| a.min(b))
            })
            .fold(f64::INFINITY, |a, b| a.min(b));
        if min_overlap < 0.0 {
            bad += 1;
        }
    }
    bad as f64 <= 0.1 * cells.len() as f64
}

/// An HTML table from the OrderNet table heads: `tab` = 3 matrices [m][m]
/// (logits same row / same column / same cell) for the m lines of one table
/// block.
pub fn table_html_heads(boxes: &[[f32; 4]], texts: &[String], tab: &[Vec<f32>]) -> String {
    let m = tab[0].len();
    let cells = cluster(&tab[2], m);
    let cb: Vec<[f32; 4]> = cells
        .iter()
        .map(|c| {
            [
                c.iter()
                    .map(|&k| boxes[k][0])
                    .fold(f32::INFINITY, |a, b| a.min(b)),
                c.iter()
                    .map(|&k| boxes[k][1])
                    .fold(f32::INFINITY, |a, b| a.min(b)),
                c.iter()
                    .map(|&k| boxes[k][2])
                    .fold(f32::NEG_INFINITY, |a, b| a.max(b)),
                c.iter()
                    .map(|&k| boxes[k][3])
                    .fold(f32::NEG_INFINITY, |a, b| a.max(b)),
            ]
        })
        .collect();
    let ctext: Vec<String> = cells
        .iter()
        .map(|c| {
            let mut order: Vec<usize> = c.clone();
            order.sort_by(|&a, &b| {
                boxes[a][1]
                    .total_cmp(&boxes[b][1])
                    .then_with(|| boxes[a][0].total_cmp(&boxes[b][0]))
                    .then_with(|| a.cmp(&b))
            });
            order
                .iter()
                .map(|&k| texts[k].clone())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let cell_logits = |k: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; m * m];
        for a in 0..m {
            for b in a + 1..m {
                let mut s = 0.0f32;
                for &i in &cells[a] {
                    for &j in &cells[b] {
                        s += tab[k][i * m + j];
                    }
                }
                let v = s / (cells[a].len() * cells[b].len()) as f32;
                out[a * m + b] = v;
                out[b * m + a] = v;
            }
        }
        out
    };
    let min_box = |g: &[usize], axis: usize| -> f32 {
        g.iter()
            .map(|&c| cb[c][axis])
            .fold(f32::INFINITY, |a, b| a.min(b))
    };
    let rows_c = cluster(&cell_logits(0), m);
    let cols_c = cluster(&cell_logits(1), m);
    let mut rows_sorted = rows_c;
    rows_sorted.sort_by(|a, b| {
        min_box(a, 1)
            .total_cmp(&min_box(b, 1))
            .then_with(|| a.cmp(b))
    });
    let mut cols_sorted = cols_c;
    cols_sorted.sort_by(|a, b| {
        min_box(a, 0)
            .total_cmp(&min_box(b, 0))
            .then_with(|| a.cmp(b))
    });
    let mut row_of = vec![0usize; m];
    for (r, g) in rows_sorted.iter().enumerate() {
        for &c in g {
            row_of[c] = r;
        }
    }
    let mut col_of = vec![0usize; m];
    for (k, g) in cols_sorted.iter().enumerate() {
        for &c in g {
            col_of[c] = k;
        }
    }
    let centers: Vec<f32> = cols_sorted
        .iter()
        .map(|g| {
            let v: Vec<f32> = g.iter().map(|&c| (cb[c][0] + cb[c][2]) / 2.0f32).collect();
            median_f32(&v) as f32
        })
        .collect();
    let h = median_f32(
        &(0..boxes.len())
            .map(|i| boxes[i][3] - boxes[i][1])
            .collect::<Vec<_>>(),
    );
    let mut grid: std::collections::HashMap<(usize, usize), (usize, String)> = Default::default();
    for c in 0..m {
        let c0 = col_of[c];
        let mut span = 1;
        while c0 + span < cols_sorted.len()
            && (cb[c][0] as f64) - h <= centers[c0 + span] as f64
            && centers[c0 + span] as f64 <= (cb[c][2] as f64) + h
        {
            span += 1;
        }
        let key = (row_of[c], c0);
        match grid.get(&key) {
            Some(p) => {
                grid.insert(key, (p.0.max(span), format!("{} {}", p.1, ctext[c])));
            }
            None => {
                grid.insert(key, (span, ctext[c].clone()));
            }
        }
    }
    let mut out = vec!["<table>".to_string()];
    for r in 0..rows_sorted.len() {
        let tag = if r == 0 && rows_sorted.len() > 1 {
            "th"
        } else {
            "td"
        };
        let mut tds = Vec::new();
        let mut c = 0usize;
        while c < cols_sorted.len() {
            if let Some(&(span, ref text)) = grid.get(&(r, c)) {
                let attr = if span > 1 {
                    format!(" colspan=\"{span}\"")
                } else {
                    String::new()
                };
                tds.push(format!("<{tag}{attr}>{}</{tag}>", escape_html(text)));
                c += span;
            } else {
                tds.push(format!("<{tag}></{tag}>"));
                c += 1;
            }
        }
        out.push(format!("<tr>{}</tr>", tds.concat()));
    }
    out.push("</table>".to_string());
    out.join("\n")
}

/// One layout block in reading order: class, bbox (pixels, the union of its
/// lines, rounded to 0.1) and text (its lines, newline-joined).
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutBlock {
    pub class: usize,
    pub bbox: [f64; 4],
    pub text: String,
}

/// The argmax block class of a group of lines (cls summed over the lines).
pub fn block_class(blk: &[usize], cls: &[f32]) -> usize {
    let mut sum = [0.0f32; N_BLOCK_CLASSES];
    for &i in blk {
        for c in 0..N_BLOCK_CLASSES {
            sum[c] += cls[i * N_BLOCK_CLASSES + c];
        }
    }
    sum.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(c, _)| c)
        .unwrap()
}

/// The layout blocks behind `page_markdown`, in reading order.
pub fn page_blocks(
    boxes: &[[f64; 4]],
    texts: &[String],
    order: &[usize],
    cls: &[f32],
    same: &[f32],
) -> Vec<LayoutBlock> {
    let f32b: Vec<[f32; 4]> = boxes
        .iter()
        .map(|b| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32])
        .collect();
    let mut out = Vec::new();
    for blk in split_blocks(order, same) {
        let x0 = blk
            .iter()
            .map(|&i| f32b[i][0])
            .fold(f32::INFINITY, |a, b| a.min(b));
        let y0 = blk
            .iter()
            .map(|&i| f32b[i][1])
            .fold(f32::INFINITY, |a, b| a.min(b));
        let x1 = blk
            .iter()
            .map(|&i| f32b[i][2])
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let y1 = blk
            .iter()
            .map(|&i| f32b[i][3])
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        out.push(LayoutBlock {
            class: block_class(&blk, cls),
            bbox: [
                py_round(x0 as f64 * 10.0) as f64 / 10.0,
                py_round(y0 as f64 * 10.0) as f64 / 10.0,
                py_round(x1 as f64 * 10.0) as f64 / 10.0,
                py_round(y1 as f64 * 10.0) as f64 / 10.0,
            ],
            text: blk
                .iter()
                .map(|&i| texts[i].clone())
                .collect::<Vec<_>>()
                .join("\n"),
        });
    }
    out
}

/// The table post-filter (new logic, plan §1.5): one verdict per block, in
/// reading order.  Non-Table blocks are always kept; a Table block keeps its
/// `<table>` only when its bbox covers less than `TABLE_COMPACT_FRAC` of the
/// page and the block's lines read with mean confidence above
/// `TABLE_COMPACT_CONF`, or when a table-caption region exists.
pub fn table_postfilter(
    blocks: &[Vec<usize>],
    boxes: &[[f64; 4]],
    cls: &[f32],
    conf: &[f32],
    page_area: f64,
    has_table_caption: bool,
) -> Vec<bool> {
    blocks
        .iter()
        .map(|blk| {
            if block_class(blk, cls) != TABLE {
                return true;
            }
            let x0 = blk
                .iter()
                .map(|&i| boxes[i][0])
                .fold(f64::INFINITY, f64::min);
            let y0 = blk
                .iter()
                .map(|&i| boxes[i][1])
                .fold(f64::INFINITY, f64::min);
            let x1 = blk.iter().map(|&i| boxes[i][2]).fold(0.0f64, f64::max);
            let y1 = blk.iter().map(|&i| boxes[i][3]).fold(0.0f64, f64::max);
            let frac = if page_area > 0.0 {
                (x1 - x0) * (y1 - y0) / page_area
            } else {
                1.0
            };
            let mean_conf: f64 =
                blk.iter().map(|&i| conf[i] as f64).sum::<f64>() / blk.len() as f64;
            (frac < TABLE_COMPACT_FRAC && mean_conf > TABLE_COMPACT_CONF) || has_table_caption
        })
        .collect()
}

/// One line's table cells: (box, text) in cell order.
pub type Cell = (Vec<f64>, String);
pub type Cells = Vec<Vec<Cell>>;

/// The per-block [3][m][m] view of the page-level table-head logits
/// (product `tab[:, blk][:, :, blk]`), from the flat 3*N*N storage.
fn block_tab(tab: &[f32], n: usize, blk: &[usize]) -> Vec<Vec<f32>> {
    let m = blk.len();
    let mut out: Vec<Vec<f32>> = (0..3).map(|_| Vec::with_capacity(m * m)).collect();
    for k in 0..3 {
        for &i in blk {
            for &j in blk {
                out[k].push(tab[k * n * n + i * n + j]);
            }
        }
    }
    out
}

/// Markdown for one page (product `page_markdown`).  `cells` maps a line to
/// its table cells (None/empty = the whole line is one cell).  `tab` is the
/// page-level table-head logits, 3*N*N flat (same row / column / cell).
/// `pictures` [P][4]: each written as `![](picture-<k>)` before the first
/// block below it that it overlaps horizontally.  `table_keep`: the post-
/// filter's verdict per block (None: keep every table — product parity).
#[allow(clippy::too_many_arguments)]
pub fn page_markdown(
    boxes: &[[f64; 4]],
    texts: &[String],
    order: &[usize],
    cls: &[f32],
    same: &[f32],
    keep_furniture: bool,
    cells: Option<&Cells>,
    tab: Option<&[f32]>,
    pictures: Option<&[[f64; 4]]>,
    table_keep: Option<&[bool]>,
) -> String {
    let n = boxes.len();
    let f32b: Vec<[f32; 4]> = boxes
        .iter()
        .map(|b| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32])
        .collect();
    let mut parts: Vec<String> = Vec::new();
    let mut part_boxes: Vec<[f32; 4]> = Vec::new();
    for (bi, blk) in split_blocks(order, same).iter().enumerate() {
        let n_before = parts.len();
        let tab_blk = tab.map(|t| block_tab(t, n, blk));
        block_part(
            &mut parts,
            blk,
            &f32b,
            texts,
            cls,
            keep_furniture,
            cells,
            tab_blk.as_ref(),
            table_keep.is_none_or(|k| k[bi]),
        );
        if parts.len() > n_before {
            part_boxes.push([
                blk.iter()
                    .map(|&i| f32b[i][0])
                    .fold(f32::INFINITY, |a, b| a.min(b)),
                blk.iter()
                    .map(|&i| f32b[i][1])
                    .fold(f32::INFINITY, |a, b| a.min(b)),
                blk.iter()
                    .map(|&i| f32b[i][2])
                    .fold(f32::NEG_INFINITY, |a, b| a.max(b)),
                blk.iter()
                    .map(|&i| f32b[i][3])
                    .fold(f32::NEG_INFINITY, |a, b| a.max(b)),
            ]);
        }
    }
    if let Some(pics) = pictures {
        if !pics.is_empty() {
            let mut slots: std::collections::BTreeMap<usize, Vec<String>> = Default::default();
            for (k, p) in pics.iter().enumerate() {
                let cy = (p[1] + p[3]) / 2.0;
                let at = part_boxes
                    .iter()
                    .enumerate()
                    .find(|(_, b)| {
                        (b[1] + b[3]) as f64 / 2.0 > cy
                            && p[2].min(b[2] as f64) > p[0].max(b[0] as f64)
                    })
                    .map(|(j, _)| j)
                    .unwrap_or(parts.len());
                slots
                    .entry(at)
                    .or_default()
                    .push(format!("![](picture-{k})"));
            }
            let mut merged = Vec::new();
            for j in 0..=parts.len() {
                if let Some(v) = slots.get(&j) {
                    merged.extend_from_slice(v);
                }
                if j < parts.len() {
                    merged.push(parts[j].clone());
                }
            }
            parts = merged;
        }
    }
    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Append one block's markdown to `parts` (nothing for dropped furniture).
#[allow(clippy::too_many_arguments)]
fn block_part(
    parts: &mut Vec<String>,
    blk: &[usize],
    boxes: &[[f32; 4]],
    texts: &[String],
    cls: &[f32],
    keep_furniture: bool,
    cells: Option<&Cells>,
    tab_blk: Option<&Vec<Vec<f32>>>,
    keep_table: bool,
) {
    let c = block_class(blk, cls);
    let texts_b: Vec<String> = blk.iter().map(|&i| texts[i].clone()).collect();
    if (c == HEADER || c == FOOTER) && !keep_furniture {
        return;
    }
    if !texts_b.is_empty() && texts_b.iter().all(|t| whole_math(t.trim())) {
        parts.push(
            texts_b
                .iter()
                .map(|t| {
                    let t = t.trim();
                    format!("$${}$$", &t[1..t.len() - 1])
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
        return;
    }
    if c == SECTION {
        parts.push(format!("## {}", join_lines(&texts_b)));
    } else if c == TABLE && blk.len() > 1 {
        if !keep_table {
            // Post-filter: the table is demoted to plain joined text.
            parts.push(join_lines(&texts_b));
        } else if let Some(t) = tab_blk {
            if heads_plausible(&blk.iter().map(|&i| boxes[i]).collect::<Vec<_>>(), &t[2]) {
                let tb: Vec<[f32; 4]> = blk.iter().map(|&i| boxes[i]).collect();
                parts.push(table_html_heads(&tb, &texts_b, t));
            } else {
                parts.push(table_from_cells(blk, boxes, texts, cells));
            }
        } else {
            parts.push(table_from_cells(blk, boxes, texts, cells));
        }
    } else if c == LIST {
        let mut items: Vec<Vec<String>> = Vec::new();
        for t in &texts_b {
            if items.is_empty() || item_match(t) {
                items.push(vec![t.clone()]);
            } else {
                items.last_mut().unwrap().push(t.clone());
            }
        }
        parts.push(
            items
                .iter()
                .map(|it| join_lines(it))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    } else {
        parts.push(join_lines(&texts_b));
    }
}

/// A geometric table from the block's cells.
fn table_from_cells(
    blk: &[usize],
    boxes: &[[f32; 4]],
    texts: &[String],
    cells: Option<&Cells>,
) -> String {
    let (mut cb, mut ct): (Vec<[f32; 4]>, Vec<String>) = (Vec::new(), Vec::new());
    for &i in blk {
        match cells.and_then(|c| c.get(i)).filter(|v| !v.is_empty()) {
            Some(list) => {
                for (b, t) in list {
                    cb.push([b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32]);
                    ct.push(t.clone());
                }
            }
            None => {
                cb.push(boxes[i]);
                ct.push(texts[i].clone());
            }
        }
    }
    table_html(&cb, &ct)
}

// ---------------------------------------------------------------------------
// Line splitting (product pipeline.py): gutter_cuts / split_line / tidy_lines.
// ---------------------------------------------------------------------------

/// A token: a recognizer vocab id or a single character (group recognizers /
/// PDF text layers), with its page x.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Id(i32, f64),
    Ch(char, f64),
}

impl Tok {
    fn x(&self) -> f64 {
        match self {
            Tok::Id(_, x) => *x,
            Tok::Ch(_, x) => *x,
        }
    }
    fn is_char(&self) -> bool {
        matches!(self, Tok::Ch(_, _))
    }
    fn is_space(&self, space_id: i32) -> bool {
        match self {
            Tok::Id(i, _) => *i == space_id,
            Tok::Ch(c, _) => *c == ' ',
        }
    }
}

/// ' '.join(s.split()): collapse every whitespace run to one space, trim.
fn collapse_ws(s: &str) -> String {
    let mut out = String::new();
    let mut pending = false;
    for c in s.chars() {
        if c.is_whitespace() {
            pending = true;
        } else {
            if pending && !out.is_empty() {
                out.push(' ');
            }
            pending = false;
            out.push(c);
        }
    }
    out
}

/// Text of tokens (product `_decode`): char tokens join and collapse, then
/// NFC, then the RTL flip; id tokens go through the vocab.
pub fn decode_tokens(toks: &[Tok], vocab: &Vocab, page_rtl: bool) -> String {
    if toks.is_empty() {
        return String::new();
    }
    if toks[0].is_char() {
        let joined: String = toks
            .iter()
            .map(|t| match t {
                Tok::Ch(c, _) => *c,
                _ => '\0',
            })
            .collect();
        let collapsed = collapse_ws(&joined);
        let n = nfc(&collapsed);
        if has_rtl(&n) {
            flip(&n, true, if page_rtl { Some(true) } else { None })
        } else {
            n
        }
    } else {
        let ids: Vec<i32> = toks
            .iter()
            .map(|t| match t {
                Tok::Id(i, _) => *i,
                Tok::Ch(c, _) => *c as i32,
            })
            .collect();
        vocab.decode_seq(&ids)
    }
}

/// Per line, the x positions of gaps that sit on a column gutter: a gap over
/// GUTTER_GAP line heights where at least GUTTER_EDGES other lines end at its
/// left side or start at its right side.  `space_id`: the vocab space token
/// (its x's are excluded from the line's character positions).
pub fn gutter_cuts(boxes: &[[f64; 4]], rec: &[Vec<Tok>], space_id: i32) -> Vec<Vec<f64>> {
    let n = boxes.len();
    let mut cuts = Vec::with_capacity(n);
    for i in 0..n {
        let h = (boxes[i][3] - boxes[i][1]).max(1.0);
        let xs: Vec<f64> = rec[i]
            .iter()
            .filter(|t| !t.is_space(space_id))
            .map(|t| t.x())
            .collect();
        let mut mine = Vec::new();
        let mut a = 0usize;
        while a + 1 < xs.len() {
            let (ga, gc) = (xs[a], xs[a + 1]);
            a += 1;
            if gc - ga <= GUTTER_GAP * h {
                continue;
            }
            let (gl, gr) = (ga + 0.35 * h, gc - 0.35 * h);
            let mut near = 0usize;
            for (j, bj) in boxes.iter().enumerate() {
                if j == i {
                    continue;
                }
                if (bj[2] - gl).abs() < 0.5 * h || (bj[0] - gr).abs() < 0.5 * h {
                    near += 1;
                }
            }
            if near >= GUTTER_EDGES {
                mine.push((ga + gc) / 2.0);
            }
        }
        cuts.push(mine);
    }
    cuts
}

/// Cut a line where two neighbouring characters are more than `gap` line
/// heights apart (or at one of the given gutter positions).  Returns
/// [(box, text, tokens)]; one item when nothing is cut.
pub fn split_line(
    box_: [f64; 4],
    toks: &[Tok],
    gap: f64,
    at: Option<&[f64]>,
    space_id: i32,
    vocab: &Vocab,
    page_rtl: bool,
) -> Vec<([f64; 4], String, Vec<Tok>)> {
    // Product dtype discipline: the line box is float32, so the height is an
    // f32 subtraction, widened once for the f64 token arithmetic.
    let h = (((box_[3] as f32) - (box_[1] as f32)) as f64).max(1.0);
    let mut at: Vec<f64> = at.map(|a| a.to_vec()).unwrap_or_default();
    at.sort_by(|a, b| a.total_cmp(b));
    let mut segs: Vec<Vec<Tok>> = vec![Vec::new()];
    let mut prev: Option<f64> = None;
    for t in toks {
        let x = t.x();
        if !t.is_space(space_id) {
            let cut = prev
                .map(|p| x - p > gap * h || at.iter().any(|&c| p < c && c < x))
                .unwrap_or(false);
            if cut {
                segs.push(Vec::new());
            }
            prev = Some(x);
        }
        segs.last_mut().unwrap().push(t.clone());
    }
    if segs.len() == 1 {
        return vec![(box_, decode_tokens(toks, vocab, page_rtl), toks.to_vec())];
    }
    let mut out = Vec::new();
    for sg in &segs {
        let text = decode_tokens(sg, vocab, page_rtl).trim().to_string();
        let xs: Vec<f64> = sg
            .iter()
            .filter(|t| !t.is_space(space_id))
            .map(|t| t.x())
            .collect();
        if !text.is_empty() && !xs.is_empty() {
            let x0 = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let x1 = xs.iter().cloned().fold(0.0f64, f64::max);
            // The candidates are python floats (f64), the box row is float32.
            // numpy 2.x (NEP 50) compares in float32 and returns the winner
            // in its own dtype, so a candidate that loses only in float32
            // rounds away.
            let lo = x0 - 0.35 * h;
            let hi = x1 + 0.35 * h;
            let bx0 = if (lo as f32) > (box_[0] as f32) {
                lo
            } else {
                box_[0]
            };
            let bx2 = if (hi as f32) < (box_[2] as f32) {
                hi
            } else {
                box_[2]
            };
            out.push(([bx0, box_[1], bx2, box_[3]], text, sg.to_vec()));
        }
    }
    out
}

/// Two kinds of lines carry no text of their own: a line of leader characters
/// only is dropped; a lone list marker joins the line that starts within
/// MARKER_GAP line heights after it on the same row.  The item's box is the
/// union, and the marker's tokens are prepended (plus a space token between
/// when the item's tokens are the same kind).
pub fn tidy_lines(
    boxes: &mut Vec<[f64; 4]>,
    rec: &mut Vec<(String, f64, Vec<Tok>)>,
    rtl: bool,
    space_id: i32,
) {
    let n = rec.len();
    let mut keep = vec![true; n];
    // f32 array arithmetic (the product keeps the boxes' dtype).
    let h32: Vec<f32> = (0..n).map(|i| (boxes[i][3] - boxes[i][1]) as f32).collect();
    let w32: Vec<f32> = (0..n).map(|i| (boxes[i][2] - boxes[i][0]) as f32).collect();
    for i in 0..n {
        let core: Vec<char> = rec[i].0.chars().filter(|&c| c != ' ').collect();
        if !core.is_empty() && core.iter().all(|&c| LEADER_CHARS.contains(&c)) {
            let wide = w32[i] >= 3.0f32 * h32[i];
            let has_dash = core.iter().any(|&c| DASHES.contains(&c));
            if core.len() >= 3 || wide || !has_dash {
                keep[i] = false;
            }
        }
    }
    for i in 0..n {
        let t = rec[i].0.trim().to_string();
        // A marker line is a LONE bullet character (the product checks
        // `t in BULLETS`: the whole string equals one bullet), or a full
        // enumerator ("1.", "(a)", "(iv)").
        let is_bullet = t.chars().count() == 1 && BULLETS.contains(&t.chars().next().unwrap());
        if !keep[i] || (!is_bullet && !enumerator_fullmatch(&t)) {
            continue;
        }
        let mut best: Option<usize> = None;
        let mut best_gap: Option<f32> = None;
        for j in 0..n {
            if j == i || !keep[j] || rec[j].0.trim().is_empty() {
                continue;
            }
            // Vertical overlap and the horizontal gap; the box array is
            // float32, and numpy 2.x (NEP 50) keeps the weak python-float
            // factors (0.5, -0.3, MARKER_GAP) in float32, so every comparison
            // below runs in f32.
            let ov = (boxes[i][3] as f32).min(boxes[j][3] as f32)
                - (boxes[i][1] as f32).max(boxes[j][1] as f32);
            if ov < 0.5f32 * h32[i].min(h32[j]) {
                continue;
            }
            let gap = if rtl {
                (boxes[i][0] as f32) - (boxes[j][2] as f32)
            } else {
                (boxes[j][0] as f32) - (boxes[i][2] as f32)
            };
            if -0.3f32 * h32[i] <= gap
                && gap <= MARKER_GAP as f32 * h32[i].max(h32[j])
                && best_gap.is_none_or(|g| gap < g)
            {
                best = Some(j);
                best_gap = Some(gap);
            }
        }
        let Some(b) = best else { continue };
        let (tj, cj, kj) = (rec[b].0.clone(), rec[b].1, rec[b].2.clone());
        let chars = !kj.is_empty() && kj[0].is_char();
        // Same kind of tokens only (recognizer ids, or char tokens); the space
        // token matches the kind of the item line it is prepended to.
        if !rtl && !rec[i].2.is_empty() && !kj.is_empty() && rec[i].2[0].is_char() == chars {
            let x = ((boxes[i][2] as f32) + (boxes[b][0] as f32)) as f64 / 2.0;
            let sp_tok = if chars {
                Tok::Ch(' ', x)
            } else {
                Tok::Id(space_id, x)
            };
            let mut newk = rec[i].2.clone();
            newk.push(sp_tok);
            newk.extend(kj);
            rec[b] = (format!("{} {}", t, tj.trim()), cj, newk);
        } else {
            rec[b] = (format!("{} {}", t, tj.trim()), cj, kj);
        }
        boxes[b] = [
            boxes[i][0].min(boxes[b][0]),
            boxes[i][1].min(boxes[b][1]),
            boxes[i][2].max(boxes[b][2]),
            boxes[i][3].max(boxes[b][3]),
        ];
        keep[i] = false;
    }
    let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    *boxes = idx.iter().map(|&i| boxes[i]).collect();
    *rec = idx.iter().map(|&i| rec[i].clone()).collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        // A minimal vocab: blank=0, then "a"(1), "b"(2), " "(3).
        Vocab {
            chars: vec!["a".into(), "b".into(), " ".into()],
            has_commands: false,
            jamo: false,
        }
    }

    #[test]
    fn join_lines_hyphen_and_cjk() {
        assert_eq!(
            join_lines(&["tumori-".into(), "genesis".into()]),
            "tumorigenesis"
        );
        assert_eq!(join_lines(&["hello".into(), "world".into()]), "hello world");
        assert_eq!(join_lines(&["中文".into(), "文档".into()]), "中文文档");
        assert_eq!(join_lines(&["a-".into(), "B".into()]), "a- B"); // capital: no hyphen join
        assert_eq!(join_lines(&["".into(), "x".into(), "  ".into()]), "x");
    }

    #[test]
    fn item_match_markers() {
        assert!(item_match("• item"));
        assert!(item_match("- item"));
        assert!(item_match("1. item"));
        assert!(item_match("(3) item"));
        assert!(item_match("a. item"));
        assert!(item_match("  (b) item"));
        assert!(!item_match("no marker"));
        assert!(!item_match("12345.")); // more than 3 digits
    }

    #[test]
    fn enumerator_fullmatch_variants() {
        assert!(enumerator_fullmatch("1."));
        assert!(enumerator_fullmatch("12)"));
        assert!(enumerator_fullmatch("a."));
        assert!(enumerator_fullmatch("(3)"));
        assert!(enumerator_fullmatch("(b)"));
        assert!(enumerator_fullmatch("(iv)"));
        assert!(!enumerator_fullmatch("(roman)"));
        assert!(!enumerator_fullmatch("1234."));
        assert!(!enumerator_fullmatch("12345"));
    }

    #[test]
    fn split_line_single_when_no_wide_gap() {
        let v = vocab();
        let toks = vec![Tok::Id(1, 10.0), Tok::Id(2, 12.0)];
        let out = split_line([5.0, 0.0, 50.0, 10.0], &toks, SPLIT_GAP, None, 3, &v, false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, "ab");
    }

    #[test]
    fn split_line_cuts_at_wide_gap() {
        let v = vocab();
        let toks = vec![Tok::Id(1, 10.0), Tok::Id(2, 12.0), Tok::Id(1, 60.0)];
        let out = split_line(
            [5.0, 0.0, 100.0, 10.0],
            &toks,
            SPLIT_GAP,
            None,
            3,
            &v,
            false,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].1, "ab");
        assert_eq!(out[1].1, "a");
        assert!(out[1].0[0] > 50.0);
    }

    #[test]
    fn gutter_cuts_find_column_gutter() {
        // The first line has a wide internal gap whose left edge (x=17.5) sits
        // at the right edges (x=18) of the three lines below it: a column
        // gutter.  The cut lands at the gap centre (x=42).
        let boxes = [
            [10.0, 0.0, 100.0, 10.0],
            [10.0, 20.0, 18.0, 30.0],
            [10.0, 40.0, 18.0, 50.0],
            [10.0, 60.0, 18.0, 70.0],
        ];
        let rec = vec![
            vec![Tok::Id(1, 12.0), Tok::Id(2, 14.0), Tok::Id(1, 70.0)],
            vec![Tok::Id(1, 12.0)],
            vec![Tok::Id(1, 12.0)],
            vec![Tok::Id(1, 12.0)],
        ];
        let cuts = gutter_cuts(&boxes, &rec, 3);
        assert_eq!(cuts[0].len(), 1, "one gutter cut: {cuts:?}");
        assert!((cuts[0][0] - 42.0).abs() < 5.0);
        assert_eq!(cuts[1].len(), 0);
    }

    #[test]
    fn tidy_lines_drops_leaders_and_joins_markers() {
        let mut boxes = vec![
            [10.0, 0.0, 200.0, 10.0],  // "..." leader: dropped
            [40.0, 20.0, 60.0, 30.0],  // "•" marker: joins the item
            [70.0, 20.0, 200.0, 30.0], // item text
        ];
        let mut rec = vec![
            ("...".into(), 0.9f64, vec![]),
            ("•".into(), 0.9f64, vec![Tok::Id(1, 50.0)]),
            ("hello".into(), 0.9f64, vec![Tok::Id(1, 80.0)]),
        ];
        tidy_lines(&mut boxes, &mut rec, false, 3);
        // The leader is dropped and the marker merges into the item, leaving
        // one line whose box is the union (left edge pulled to the marker).
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].0, "• hello");
        assert_eq!(boxes[0][0], 40.0); // box union includes the marker
    }

    #[test]
    fn table_postfilter_keeps_compact_confident() {
        // Two table blocks: a compact, confident one and a full-page one.
        let boxes: Vec<[f64; 4]> = vec![[10.0, 10.0, 50.0, 30.0], [10.0, 100.0, 790.0, 800.0]];
        let cls: Vec<f32> = (0..2 * N_BLOCK_CLASSES)
            .map(|i| {
                if i % N_BLOCK_CLASSES == TABLE {
                    5.0
                } else {
                    -5.0
                }
            })
            .collect();
        let conf = [0.9f32, 0.9f32];
        let page_area = 800.0 * 800.0;
        let keep = table_postfilter(
            &[vec![0usize], vec![1usize]],
            &boxes,
            &cls,
            &conf,
            page_area,
            false,
        );
        assert_eq!(keep, vec![true, false]);
        // A table caption forces the big table through too.
        let keep2 = table_postfilter(
            &[vec![0usize], vec![1usize]],
            &boxes,
            &cls,
            &conf,
            page_area,
            true,
        );
        assert_eq!(keep2, vec![true, true]);
        // Low confidence demotes even a compact table.
        let keep3 = table_postfilter(&[vec![0usize]], &boxes, &cls, &[0.3f32], page_area, false);
        assert_eq!(keep3, vec![false]);
    }

    #[test]
    fn table_html_two_col_header() {
        let boxes: Vec<[f32; 4]> = vec![
            [10.0, 0.0, 50.0, 10.0],
            [60.0, 0.0, 110.0, 10.0],
            [10.0, 20.0, 50.0, 30.0],
            [60.0, 20.0, 110.0, 30.0],
        ];
        let texts: Vec<String> = vec!["name".into(), "value".into(), "a".into(), "1".into()];
        let html = table_html(&boxes, &texts);
        assert_eq!(
            html,
            "<table>\n<tr><th>name</th><th>value</th></tr>\n<tr><td>a</td><td>1</td></tr>\n</table>"
        );
    }

    #[test]
    fn page_markdown_section_and_text() {
        let boxes: Vec<[f64; 4]> = vec![[10.0, 0.0, 200.0, 10.0], [10.0, 20.0, 200.0, 30.0]];
        let texts: Vec<String> = vec!["Title here".into(), "Body text".into()];
        let mut cls = vec![0.0f32; 2 * N_BLOCK_CLASSES];
        cls[SECTION] = 9.0;
        cls[N_BLOCK_CLASSES] = 9.0;
        let same = [0.0f32, -1.0, -1.0, 0.0];
        let md = page_markdown(
            &boxes,
            &texts,
            &[0, 1],
            &cls,
            &same,
            false,
            None,
            None,
            None,
            None,
        );
        assert_eq!(md, "## Title here\n\nBody text");
    }

    #[test]
    fn page_markdown_demotes_table_when_filtered() {
        let boxes: Vec<[f64; 4]> = vec![
            [10.0, 0.0, 60.0, 10.0],
            [70.0, 0.0, 120.0, 10.0],
            [10.0, 20.0, 60.0, 30.0],
            [70.0, 20.0, 120.0, 30.0],
        ];
        let texts: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let mut cls = vec![0.0f32; 4 * N_BLOCK_CLASSES];
        for i in 0..4 {
            cls[i * N_BLOCK_CLASSES + TABLE] = 9.0;
        }
        let same = [
            0.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        ];
        // Kept: a two-column table.
        let md = page_markdown(
            &boxes,
            &texts,
            &[0, 1, 2, 3],
            &cls,
            &same,
            false,
            None,
            None,
            None,
            Some(&[true]),
        );
        assert!(md.contains("<table>"), "{md}");
        // Demoted: plain joined text, one space between cells.
        let md2 = page_markdown(
            &boxes,
            &texts,
            &[0, 1, 2, 3],
            &cls,
            &same,
            false,
            None,
            None,
            None,
            Some(&[false]),
        );
        assert!(!md2.contains("<table>"), "{md2}");
        assert_eq!(md2, "a b c d");
    }

    #[test]
    fn decode_tokens_char_path_collapses_and_ids() {
        let v = vocab();
        let toks = vec![Tok::Ch('a', 1.0), Tok::Ch(' ', 2.0), Tok::Ch('b', 3.0)];
        assert_eq!(decode_tokens(&toks, &v, false), "a b");
        let toks = vec![Tok::Id(1, 1.0), Tok::Id(2, 2.0)];
        assert_eq!(decode_tokens(&toks, &v, false), "ab");
    }

    #[test]
    fn split_gaps_are_ordered() {
        // Witness of the product's gap ordering (constants; the gate replays
        // the product's calls for the real checks).
        const _: () = {
            assert!(CELL_GAP > GUTTER_GAP);
            assert!(SPLIT_GAP > CELL_GAP);
        };
    }
}
